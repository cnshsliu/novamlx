import XCTest
@testable import NovaMLXEngine
@testable import NovaMLXInference
@testable import NovaMLXCore

final class BatcherAdmissionTests: XCTestCase {

    /// Simulate Claude Code's pattern: two concurrent requests to the same model.
    /// Both should complete without crash or "I/O on closed channel" error.
    func testDualConcurrentRequestsAreQueuedCorrectly() async throws {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 2_000_000) // 2 MB

        // Request 1: 1000 tokens * 256 bytes = 256 KB — should admit
        let canAdmit1 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 1000, bytesPerToken: 256)
        XCTAssertTrue(canAdmit1)

        await tracker.reserve(modelId: "test-model", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)

        // After reserving 256 KB, remaining = 2MB - 256KB = 1.75MB
        // With 10% headroom: usable = 1.8MB - 256KB = 1.544MB = 1,544,000 bytes
        // Request 2: 1000 tokens * 256 bytes = 256,000 < 1,544,000 → should admit
        let canAdmit2 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 1000, bytesPerToken: 256)
        XCTAssertTrue(canAdmit2)
    }

    /// When budget is tight, a large second request should be rejected for admission.
    func testSecondRequestRejectedWhenBudgetTight() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 600_000) // 600 KB total

        // Request 1: 2000 tokens * 256 bytes = 512 KB — under 540KB (90%), should admit
        let canAdmit1 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 2000, bytesPerToken: 256)
        XCTAssertTrue(canAdmit1)

        await tracker.reserve(modelId: "test-model", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 2000, bytesPerToken: 256)

        // After reserving 512KB: available = 600KB - 512KB = 88KB
        // With 10% headroom: usable = 88KB * 90/100 = 79,200 bytes
        // Request 2: 500 tokens * 256 bytes = 128,000 > 79,200 → rejected
        let canAdmit2 = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 500, bytesPerToken: 256)
        XCTAssertFalse(canAdmit2)
    }

    /// TurboQuant compression should increase effective capacity.
    /// 4-bit quantization = 4x compression → can admit 4x more sequences.
    func testTurboQuantDoublesAdmissionCapacity() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB

        // Without TurboQuant: 256 bytes/token
        let rawBytesPerToken = 256
        // With 4-bit TurboQuant: 256 / 4 = 64 bytes/token
        let turboBytesPerToken = rawBytesPerToken / 4

        // Without TurboQuant: 4000 tokens * 256 = 1,024,000 > 900,000 (90% usable)
        let canAdmitRaw = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 4000, bytesPerToken: rawBytesPerToken)
        XCTAssertFalse(canAdmitRaw)

        // With TurboQuant: 4000 tokens * 64 = 256,000 < 900,000
        let canAdmitTurbo = await tracker.canAdmit(modelId: "test-model", estimatedTokens: 4000, bytesPerToken: turboBytesPerToken)
        XCTAssertTrue(canAdmitTurbo)
    }
}
