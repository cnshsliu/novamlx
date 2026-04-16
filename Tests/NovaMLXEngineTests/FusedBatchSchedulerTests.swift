import XCTest
@testable import NovaMLXEngine
@testable import NovaMLXCore

final class FusedBatchSchedulerTests: XCTestCase {

    // MARK: - Metrics and State

    func testInitialMetricsAreZero() {
        // We can't create a real MLXEngine in tests, but we can test the metrics type
        let metrics = FusedSchedulerMetrics(
            activeSequences: 0,
            queueDepth: 0,
            totalDecodeSteps: 0,
            totalTokensViaFused: 0,
            peakBatchWidth: 0
        )
        XCTAssertEqual(metrics.activeSequences, 0)
        XCTAssertEqual(metrics.totalDecodeSteps, 0)
        XCTAssertEqual(metrics.peakBatchWidth, 0)
    }

    func testMetricsAfterWork() {
        let metrics = FusedSchedulerMetrics(
            activeSequences: 3,
            queueDepth: 2,
            totalDecodeSteps: 100,
            totalTokensViaFused: 450,
            peakBatchWidth: 4
        )
        XCTAssertEqual(metrics.activeSequences, 3)
        XCTAssertEqual(metrics.queueDepth, 2)
        XCTAssertEqual(metrics.totalDecodeSteps, 100)
        XCTAssertEqual(metrics.totalTokensViaFused, 450)
        XCTAssertEqual(metrics.peakBatchWidth, 4)
    }

    // MARK: - Memory Budget Integration

    func testBudgetTrackerUsedByScheduler() async {
        // Verify the budget tracker correctly limits concurrent sequences
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000) // 1MB

        // Simulate 3 concurrent sequences (each ~256KB = 256,000 bytes)
        for i in 0..<3 {
            let canAdmit = await tracker.canAdmit(modelId: "test", estimatedTokens: 1000, bytesPerToken: 256)
            XCTAssertTrue(canAdmit, "Sequence \(i) should be admitted")
            await tracker.reserve(modelId: "test", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)
        }
        // Total committed: 768,000. Available: 1M - 768K = 232,000. Usable: 232K * 90% = 208,800

        // 4th should fail — 256,000 > 208,800
        let canAdmit4 = await tracker.canAdmit(modelId: "test", estimatedTokens: 1000, bytesPerToken: 256)
        XCTAssertFalse(canAdmit4, "4th sequence should be rejected")
    }

    func testBudgetReleaseAllowsMoreAdmissions() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000)
        let seq1 = UUID()
        let seq2 = UUID()

        await tracker.reserve(modelId: "test", sequenceId: seq1, weightsBytes: 0, estimatedTokens: 2000, bytesPerToken: 256)
        await tracker.reserve(modelId: "test", sequenceId: seq2, weightsBytes: 0, estimatedTokens: 2000, bytesPerToken: 256)

        // Both used 512KB each = 1024KB > 900KB (90%), so 3rd should fail
        let canAdmit3 = await tracker.canAdmit(modelId: "test", estimatedTokens: 500, bytesPerToken: 256)
        // 500 * 256 = 128,000. Available after 2 reserves: 1M - 1,024,000 = negative → 0
        // So no, can't admit
        XCTAssertFalse(canAdmit3)

        // Release seq1
        await tracker.release(sequenceId: seq1)

        // Now should be able to admit again
        let canAdmitAfterRelease = await tracker.canAdmit(modelId: "test", estimatedTokens: 500, bytesPerToken: 256)
        XCTAssertTrue(canAdmitAfterRelease)
    }

    // MARK: - FusedBatchKVCache Compaction

    func testFusedCacheCompaction() {
        let cache = FusedBatchKVCache(maxBatchSize: 4)

        // Create minimal KVCacheSimple for testing
        // Can't easily create real ones in unit tests without GPU,
        // but we can test the entry tracking
        // The compact() method removes finished entries
        // This is more of a smoke test since we need real MLX for KVCacheSimple
        XCTAssertEqual(cache.activeCount, 0)
        XCTAssertEqual(cache.batchSize, 0)
    }
}
