import XCTest
@testable import NovaMLXEngine
import NovaMLXCore

final class MemoryBudgetTrackerTests: XCTestCase {

    // MARK: - Basic Reserve/Release

    func testCanAdmitWhenBudgetAvailable() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000_000) // 1 GB
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 1000,
            bytesPerToken: 256
        )
        XCTAssertTrue(canAdmit)
    }

    func testCannotAdmitWhenBudgetExceeded() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB
        // Reserve 900 KB (~90% of budget)
        await tracker.reserve(
            modelId: "test-model",
            sequenceId: UUID(),
            weightsBytes: 0,
            estimatedTokens: 3600,
            bytesPerToken: 256 // 3600 * 256 = 921,600 bytes
        )
        // Try to admit another 900 KB request — should fail (10% headroom)
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 3600,
            bytesPerToken: 256
        )
        XCTAssertFalse(canAdmit)
    }

    func testReleaseFreesBudget() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB
        let seqId = UUID()
        await tracker.reserve(
            modelId: "test-model",
            sequenceId: seqId,
            weightsBytes: 0,
            estimatedTokens: 3000,
            bytesPerToken: 256  // 768,000 bytes — under 90% threshold
        )
        await tracker.release(sequenceId: seqId)
        // After release, should be able to admit again
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 3000,
            bytesPerToken: 256
        )
        XCTAssertTrue(canAdmit)
    }

    // MARK: - Headroom

    func testHeadroomPreventsFullBudgetUsage() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1 MB
        // Try to use ~95% of budget (only 5% headroom, default is 10%)
        let canAdmit = await tracker.canAdmit(
            modelId: "test-model",
            estimatedTokens: 3700,
            bytesPerToken: 256 // 3700 * 256 = 947,200 bytes ≈ 94.7%
        )
        XCTAssertFalse(canAdmit) // Rejected because it exceeds 90%
    }

    // MARK: - Multiple Sequences

    func testMultipleSequencesTrackCorrectly() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000) // 10 MB
        let seq1 = UUID()
        let seq2 = UUID()

        await tracker.reserve(modelId: "model", sequenceId: seq1, weightsBytes: 0, estimatedTokens: 2500, bytesPerToken: 256)
        await tracker.reserve(modelId: "model", sequenceId: seq2, weightsBytes: 0, estimatedTokens: 2500, bytesPerToken: 256)

        // Total committed: 1,280,000. Available usable = 9,000,000 - 1,280,000 = 7,720,000
        let canAdmit = await tracker.canAdmit(modelId: "model", estimatedTokens: 2500, bytesPerToken: 256)
        XCTAssertTrue(canAdmit)

        await tracker.release(sequenceId: seq1)
        let canAdmit2 = await tracker.canAdmit(modelId: "model", estimatedTokens: 2500, bytesPerToken: 256)
        XCTAssertTrue(canAdmit2)
    }

    // MARK: - Available Budget Query

    func testAvailableKVBudget() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000)
        let available = await tracker.availableKVBudget
        XCTAssertEqual(available, 10_000_000)
    }

    func testAvailableKVBudgetAfterReserve() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000)
        await tracker.reserve(modelId: "model", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)
        let available = await tracker.availableKVBudget
        XCTAssertEqual(available, 10_000_000 - 256_000)
    }

    // MARK: - Metrics

    func testMetricsTracking() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 10_000_000)
        let seq1 = UUID()
        await tracker.reserve(modelId: "model", sequenceId: seq1, weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)

        let metrics = await tracker.metrics
        XCTAssertEqual(metrics.committedKVBytes, 256_000)
        XCTAssertEqual(metrics.activeSequenceCount, 1)
    }
}
