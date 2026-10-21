import CoreML
import FluidAudio
import Foundation
import os.log

private let logger = Logger(subsystem: AppPaths.logSubsystem, category: "CohereEngine")

/// Transcription engine backed by Cohere Transcribe 03-2026 (2B, Apache-2.0) via
/// FluidAudio CoreML. 14 languages; the most accurate open model on Italian
/// (FLEURS + Common Voice + MLS) at the cost of speed and a ~2 GB download.
///
/// The model returns text only, so the audio is cut into VAD speech segments of
/// at most `maxSegmentSeconds` first. Each segment keeps its own start/end, which
/// is what speaker diarization needs to attribute the words.
@MainActor
@Observable
final class CohereEngine: TranscribingEngine, StreamingTranscribingEngine {
    private(set) var modelState: EngineModelState = .unloaded
    private(set) var downloadProgress: Double = 0
    private(set) var transcriptionProgress: Double = 0

    /// ISO 639-1 code; Cohere has no auto-detect, so an unknown code falls back to Italian.
    var language: String = "it"

    /// The FluidAudio v2 decoder emits at most ~99 tokens per call; dense speech runs
    /// ~6.5 tokens/s, so 14 s stays under the cap. Shorter segments also give the
    /// diarizer finer speaker boundaries.
    nonisolated static let maxSegmentSeconds: TimeInterval = 14
    /// A segment whose output reaches this many tokens probably hit the cap and is re-split.
    nonisolated static let cappedTokenCount = 95

    private var models: CoherePipeline.LoadedModels?
    private var vad: VadManager?
    private let modelLoad = SingleFlight<Void>()

    private var cohereLanguage: CohereAsrConfig.Language {
        CohereAsrConfig.Language(rawValue: language) ?? .italian
    }

    func loadModel() async {
        await modelLoad.run { [self] in
            modelState = .downloading
            downloadProgress = 0
            do {
                let dir = try await Self.downloadModels { [weak self] fraction in
                    Task { @MainActor in self?.downloadProgress = fraction }
                }
                modelState = .loading
                downloadProgress = 1.0
                models = try await CoherePipeline.loadModels(encoderDir: dir, decoderDir: dir, vocabDir: dir)
                vad = try await VadManager(config: VadConfig(defaultThreshold: 0.5))
                modelState = .loaded
            } catch {
                logger.error("Cohere model load failed: \(error.localizedDescription, privacy: .public)")
                models = nil
                modelState = .unloaded
                downloadProgress = 0
            }
        }
    }

    /// Downloads the INT8 encoder + v2 decoder bundle (~2 GB) into FluidAudio's model cache once.
    private nonisolated static func downloadModels(progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true,
        )
        let base = appSupport.appendingPathComponent("FluidAudio/Models")
        let target = base.appendingPathComponent(Repo.cohereTranscribeCoreml.folderName)
        let complete = ModelNames.CohereTranscribe.requiredModels.allSatisfy {
            FileManager.default.fileExists(atPath: target.appendingPathComponent($0).path)
        }
        if !complete {
            try await ModelHub.download(.cohereTranscribeCoreml, to: base) { p in progress(p.fractionCompleted) }
        }
        return target
    }

    private func ensureModel() async throws -> (CoherePipeline.LoadedModels, VadManager) {
        if models == nil || vad == nil { await loadModel() }
        guard let models, let vad else { throw TranscriptionError.modelNotLoaded }
        return (models, vad)
    }

    func transcribeSegments(audioPath: URL) async throws -> [TimestampedSegment] {
        let (models, vad) = try await ensureModel()
        transcriptionProgress = 0
        defer { transcriptionProgress = 1.0 }
        let samples = try AudioConverter().resampleAudioFile(audioPath)
        let segments = try await Self.transcribe(
            samples: samples, models: models, vad: vad, language: cohereLanguage,
        ) { [weak self] fraction in
            Task { @MainActor in self?.transcriptionProgress = fraction }
        }
        logger.info("Cohere: \(segments.count, privacy: .public) segments from \(audioPath.lastPathComponent, privacy: .public)")
        return segments
    }

    func transcribeSamples(_ samples: [Float]) async throws -> String {
        let (models, vad) = try await ensureModel()
        let segments = try await Self.transcribe(
            samples: samples, models: models, vad: vad, language: cohereLanguage,
        ) { _ in }
        return segments.map(\.text).joined(separator: " ")
    }

    // MARK: - Off-main decoding

    /// VAD-segments `samples` (16 kHz mono) and decodes each speech segment.
    /// Nonisolated and with a local pipeline so the heavy work leaves the main actor.
    private nonisolated static func transcribe(
        samples: [Float],
        models: CoherePipeline.LoadedModels,
        vad: VadManager,
        language: CohereAsrConfig.Language,
        progress: @escaping @Sendable (Double) -> Void,
    ) async throws -> [TimestampedSegment] {
        let config = VadSegmentationConfig(minSilenceDuration: 0.5, maxSpeechDuration: maxSegmentSeconds)
        let ranges = try await vad.segmentSpeech(samples, config: config).map {
            (start: $0.startSample(sampleRate: 16000), end: min($0.endSample(sampleRate: 16000), samples.count))
        }
        let pipeline = CoherePipeline()
        var out: [TimestampedSegment] = []
        for (index, range) in ranges.enumerated() {
            try Task.checkCancellation()
            out += try await decode(samples, range.start, range.end, pipeline: pipeline, models: models, language: language)
            progress(Double(index + 1) / Double(max(ranges.count, 1)))
        }
        return out
    }

    /// Decodes one sample range; re-splits it in half when the output looks truncated by the decoder cap.
    private nonisolated static func decode(
        _ samples: [Float], _ start: Int, _ end: Int,
        pipeline: CoherePipeline, models: CoherePipeline.LoadedModels, language: CohereAsrConfig.Language,
    ) async throws -> [TimestampedSegment] {
        guard end - start >= 1600 else { return [] } // < 0.1 s: nothing to transcribe
        let result = try await pipeline.transcribe(audio: Array(samples[start ..< end]), models: models, language: language)
        if shouldSplit(tokenCount: result.tokenIds.count, sampleCount: end - start) {
            let mid = (start + end) / 2
            return try await decode(samples, start, mid, pipeline: pipeline, models: models, language: language)
                + decode(samples, mid, end, pipeline: pipeline, models: models, language: language)
        }
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        return [TimestampedSegment(start: Double(start) / 16000, end: Double(end) / 16000, text: text)]
    }

    /// Split only while the halves stay long enough (≥ 2 s each) to carry a sentence.
    nonisolated static func shouldSplit(tokenCount: Int, sampleCount: Int) -> Bool {
        tokenCount >= cappedTokenCount && sampleCount >= 4 * 16000
    }
}
