import XCTest
@testable import NovaMLXEngine

final class NGramSpeculatorTests: XCTestCase {

    // MARK: - NGramSpeculator Unit Tests

    func testSpeculateReturnsEmptyWhenCacheEmpty() {
        let spec = NGramSpeculator(config: .init(ngramSize: 3, maxDraft: 3, minHits: 1))
        let result = spec.speculate(context: [1, 2, 3])
        XCTAssertTrue(result.isEmpty, "Should return empty when cache has no entries")
    }

    func testSpeculateReturnsEmptyWhenContextTooShort() {
        let spec = NGramSpeculator(config: .init(ngramSize: 4, maxDraft: 3, minHits: 1))
        spec.record(tokens: [1, 2, 3, 4, 5])
        let result = spec.speculate(context: [1, 2])
        XCTAssertTrue(result.isEmpty, "Should return empty when context shorter than ngramSize")
    }

    func testRecordAndSpeculateBasicHit() {
        let spec = NGramSpeculator(config: .init(ngramSize: 2, maxDraft: 3, minHits: 1))

        // Record: [1,2] -> 3, [2,3] -> 4, [3,4] -> 5
        spec.record(tokens: [1, 2, 3, 4, 5])

        // Speculate with context ending in [3,4] — should predict 5
        let result = spec.speculate(context: [10, 3, 4])
        XCTAssertEqual(result, [5], "Should predict 5 after seeing [3,4] -> 5")
    }

    func testMinHitsFiltersLowConfidence() {
        let spec = NGramSpeculator(config: .init(ngramSize: 2, maxDraft: 3, minHits: 2))

        // Record once: [1,2] -> 3
        spec.record(tokens: [1, 2, 3])

        // minHits=2, only 1 hit — should not speculate
        let result1 = spec.speculate(context: [1, 2])
        XCTAssertTrue(result1.isEmpty, "Should not speculate with only 1 hit when minHits=2")

        // Record the same pattern again
        spec.record(tokens: [1, 2, 3])

        // Now 2 hits — should speculate
        let result2 = spec.speculate(context: [1, 2])
        XCTAssertEqual(result2, [3], "Should speculate after reaching minHits")
    }

    func testMaxDraftLimitsOutput() {
        let spec = NGramSpeculator(config: .init(ngramSize: 1, maxDraft: 2, minHits: 1))

        // Record many continuations for token 1
        for i in 10..<20 {
            spec.record(tokens: [1, i])
        }

        let result = spec.speculate(context: [1])
        XCTAssertLessThanOrEqual(result.count, 2, "Should not exceed maxDraft")
    }

    func testSpeculateWithInternalBuffer() {
        let spec = NGramSpeculator(config: .init(ngramSize: 2, maxDraft: 3, minHits: 1))

        // record(token:) builds internal buffer
        for t in [1, 2, 3, 4, 5] {
            spec.record(token: t)
        }

        // speculate() uses internal buffer — trailing 2 tokens are [4,5]
        let result = spec.speculate()
        // [4,5] was never recorded as having a continuation, so empty
        // Let's add continuation
        spec.record(token: 6)
        spec.record(token: 7)

        let result2 = spec.speculate()
        // Now trailing is [6,7], and [4,5]->6 was recorded, [5,6]->7 was recorded
        // But context is [6,7] which has no continuation yet
        XCTAssertTrue(result2.isEmpty)
    }

    func testEvictionWhenCacheFull() {
        let spec = NGramSpeculator(config: .init(ngramSize: 1, maxDraft: 1, minHits: 1, maxCacheSize: 4))

        // Fill cache with 5 entries (exceeds maxCacheSize=4)
        for i in 0..<10 {
            spec.record(tokens: [100, i])
        }

        let stats = spec.stats
        XCTAssertLessThanOrEqual(stats.cacheSize, 4, "Cache should be evicted to fit maxCacheSize")
    }

    func testStatsReporting() {
        let spec = NGramSpeculator(config: .init(ngramSize: 2, maxDraft: 3, minHits: 1))
        spec.record(tokens: [1, 2, 3, 4, 5])

        let stats = spec.stats
        XCTAssertGreaterThan(stats.cacheSize, 0, "Should have cache entries after recording")
        XCTAssertGreaterThan(stats.totalHits, 0, "Should have hits after recording")
        XCTAssertGreaterThan(stats.bufferSize, 0, "Should have buffered tokens")
    }

    // MARK: - SpeculativeDecoder Integration Tests

    func testSpeculativeDecoderTracksAcceptance() {
        let decoder = SpeculativeDecoder(ngramConfig: .init(ngramSize: 2, maxDraft: 3, minHits: 1))

        // Seed the speculator
        decoder.recordToken(1)
        decoder.recordToken(2)
        decoder.recordToken(3)
        decoder.recordToken(4)

        // recordAccepted tracks total stats
        decoder.recordAccepted(tokens: [5, 6, 7], accepted: 2)

        XCTAssertEqual(decoder.acceptanceRate, 2.0 / 3.0, accuracy: 0.01)
    }

    func testSpeculativeDecoderAcceptanceRateZeroWhenNoSpeculation() {
        let decoder = SpeculativeDecoder()
        XCTAssertEqual(decoder.acceptanceRate, 0, "Should be 0 when no tokens speculated")
    }

    func testSpeculativeDecoderSpeculateDelegates() {
        let decoder = SpeculativeDecoder(ngramConfig: .init(ngramSize: 2, maxDraft: 3, minHits: 1))
        decoder.recordToken(1)
        decoder.recordToken(2)
        decoder.recordToken(3)

        let result = decoder.speculate(context: [1, 2])
        // [1,2] -> 3 was recorded via recordToken, so should predict 3
        XCTAssertEqual(result, [3])
    }

    // MARK: - FusedSchedulerMetrics Spec Fields

    func testMetricsIncludeSpecFields() {
        let metrics = FusedSchedulerMetrics(
            activeSequences: 1,
            queueDepth: 0,
            totalDecodeSteps: 50,
            totalTokensViaFused: 200,
            peakBatchWidth: 2,
            totalPreemptions: 0,
            specAcceptanceRate: 0.75,
            specCacheSize: 42
        )
        XCTAssertEqual(metrics.specAcceptanceRate, 0.75, accuracy: 0.01)
        XCTAssertEqual(metrics.specCacheSize, 42)
    }

    func testMetricsDefaultSpecFieldsAreZero() {
        let metrics = FusedSchedulerMetrics()
        XCTAssertEqual(metrics.specAcceptanceRate, 0)
        XCTAssertEqual(metrics.specCacheSize, 0)
    }

    // MARK: - Thread Safety Smoke Test

    func testConcurrentRecordAndSpeculate() async {
        let spec = NGramSpeculator(config: .init(ngramSize: 2, maxDraft: 3, minHits: 1))

        // Seed initial data
        for i in 0..<50 {
            spec.record(tokens: [i, i + 1, i + 2])
        }

        // Concurrently record and speculate
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    for i in 100..<120 {
                        spec.record(tokens: [i, i + 1, i + 2])
                    }
                }
                group.addTask {
                    for _ in 0..<50 {
                        _ = spec.speculate(context: [1, 2])
                    }
                }
            }
        }

        // Should not crash — that's the main assertion
        let stats = spec.stats
        XCTAssertGreaterThan(stats.cacheSize, 0)
    }
}
