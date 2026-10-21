@testable import MeetingTranscriber
import XCTest

final class CohereEngineTests: XCTestCase {
    func test_shouldSplit_onlyWhenOutputLooksCappedAndHalvesStayUseful() {
        let fourSeconds = 4 * 16000
        XCTAssertTrue(CohereEngine.shouldSplit(tokenCount: CohereEngine.cappedTokenCount, sampleCount: fourSeconds))
        XCTAssertFalse(CohereEngine.shouldSplit(tokenCount: CohereEngine.cappedTokenCount - 1, sampleCount: fourSeconds))
        XCTAssertFalse(CohereEngine.shouldSplit(tokenCount: 200, sampleCount: fourSeconds - 1))
    }

    @MainActor
    func test_engineDeclaresTimestampsAndLiveSupport() {
        XCTAssertTrue(CohereEngine().providesTimestamps)
        XCTAssertTrue(TranscriptionEngineSetting.cohere.supportsLiveTranscription)
    }
}
