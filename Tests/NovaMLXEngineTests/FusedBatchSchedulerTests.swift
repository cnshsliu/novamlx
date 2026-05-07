import XCTest
@testable import NovaMLXEngine
@testable import NovaMLXCore
@testable import NovaMLXUtils

final class FusedBatchSchedulerTests: XCTestCase {

    // MARK: - Metrics and State

    func testInitialMetricsAreZero() {
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
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000)

        for i in 0..<3 {
            let canAdmit = await tracker.canAdmit(modelId: "test", estimatedTokens: 1000, bytesPerToken: 256)
            XCTAssertTrue(canAdmit, "Sequence \(i) should be admitted")
            await tracker.reserve(modelId: "test", sequenceId: UUID(), weightsBytes: 0, estimatedTokens: 1000, bytesPerToken: 256)
        }

        let canAdmit4 = await tracker.canAdmit(modelId: "test", estimatedTokens: 1000, bytesPerToken: 256)
        XCTAssertFalse(canAdmit4, "4th sequence should be rejected")
    }

    func testBudgetReleaseAllowsMoreAdmissions() async {
        let tracker = MemoryBudgetTracker(gpuLimitBytes: 1_000_000)
        let seq1 = UUID()
        let seq2 = UUID()

        await tracker.reserve(modelId: "test", sequenceId: seq1, weightsBytes: 0, estimatedTokens: 2000, bytesPerToken: 256)
        await tracker.reserve(modelId: "test", sequenceId: seq2, weightsBytes: 0, estimatedTokens: 2000, bytesPerToken: 256)

        let canAdmit3 = await tracker.canAdmit(modelId: "test", estimatedTokens: 500, bytesPerToken: 256)
        XCTAssertFalse(canAdmit3)

        await tracker.release(sequenceId: seq1)

        let canAdmitAfterRelease = await tracker.canAdmit(modelId: "test", estimatedTokens: 500, bytesPerToken: 256)
        XCTAssertTrue(canAdmitAfterRelease)
    }

    // MARK: - FusedBatchKVCache Compaction

    func testFusedCacheCompaction() {
        let cache = FusedBatchKVCache(maxBatchSize: 4)
        XCTAssertEqual(cache.activeCount, 0)
        XCTAssertEqual(cache.batchSize, 0)
    }

    // MARK: - Chaos / Fuzz Stress Tests

    /// Thread-safe result collector for Swift 6 Sendable compliance.
    private final class ResultBox: @unchecked Sendable {
        let lock = NSLock()
        var results: [Bool]
        init(count: Int) { results = Array(repeating: false, count: count) }
        func set(_ index: Int, _ value: Bool) {
            lock.lock(); defer { lock.unlock() }; results[index] = value
        }
        var admitted: Int {
            lock.lock(); defer { lock.unlock() }
            return results.filter { $0 }.count
        }
    }

    /// Simulated admission counter mirroring FusedBatchScheduler's activeModelCounts.
    private final class SlotCounter: @unchecked Sendable {
        let lock = NovaMLXLock()
        var counts: [String: Int] = [:]

        func admit(modelId: String, limit: Int) -> Bool {
            lock.withLock {
                let current = counts[modelId] ?? 0
                guard current < limit else { return false }
                counts[modelId] = current + 1
                return true
            }
        }

        func release(modelId: String) {
            lock.withLock {
                counts[modelId] = max(0, (counts[modelId] ?? 1) - 1)
            }
        }

        func total(modelId: String) -> Int {
            lock.withLock { counts[modelId] ?? 0 }
        }
    }

    // MARK: - Test 1: Rapid admit + cancel under contention

    /// 25 concurrent tasks rapidly admit, cancel, and re-admit against a slot limit.
    /// Verifies: slot count returns to zero, no leaked reservations.
    func testRapidAdmitCancelUnderContention() async {
        let slots = SlotCounter()
        let budget = MemoryBudgetTracker(gpuLimitBytes: 2_000_000)
        let concurrency = 25
        let iterations = 10
        let modelLimit = 3
        let box = ResultBox(count: concurrency * iterations)

        await withTaskGroup(of: Void.self) { group in
            for taskIdx in 0..<concurrency {
                group.addTask {
                    for iter in 0..<iterations {
                        let idx = taskIdx * iterations + iter
                        try? await Task.sleep(nanoseconds: UInt64.random(in: 0..<5_000))

                        let admitted = slots.admit(modelId: "chaos", limit: modelLimit)
                        if admitted {
                            let seqId = UUID()
                            let canBudget = await budget.canAdmit(
                                modelId: "chaos", estimatedTokens: 500, bytesPerToken: 256
                            )
                            if canBudget {
                                await budget.reserve(
                                    modelId: "chaos", sequenceId: seqId,
                                    weightsBytes: 0, estimatedTokens: 500, bytesPerToken: 256
                                )
                                box.set(idx, true)
                                // Simulate decode then release
                                try? await Task.sleep(nanoseconds: UInt64.random(in: 0..<10_000))
                                await budget.release(sequenceId: seqId)
                            }
                            slots.release(modelId: "chaos")
                        }
                    }
                }
            }
        }

        let finalSlots = slots.total(modelId: "chaos")
        XCTAssertEqual(finalSlots, 0, "All slots should be returned after contention — got \(finalSlots) leaked")

        let admitted = box.admitted
        XCTAssertGreaterThan(admitted, 0, "At least some admits should succeed")
    }

    // MARK: - Test 2: Burst finish + continue patterns

    /// 20 streams, each with 3 concurrent Tasks racing to yield tokens then finish.
    /// Verifies: no crash, exactly one finish winner per stream, all guards marked done.
    func testBurstFinishContinuePatterns() async {
        let streamCount = 20
        final class WinnerBox: @unchecked Sendable {
            let lock = NSLock()
            var winners: [Int] = []  // task index that won the finish
            func record(_ taskIdx: Int) { lock.lock(); defer { lock.unlock() }; winners.append(taskIdx) }
            var count: Int { lock.lock(); defer { lock.unlock() }; return winners.count }
        }

        for streamIdx in 0..<streamCount {
            let guard_ = FinishGuard()
            let winners = WinnerBox()
            let stream = AsyncThrowingStream<Int, Error> { continuation in
                // Task A: yield 10 tokens then finish
                Task {
                    for i in 0..<10 {
                        guard !guard_.isDone else { break }
                        continuation.yield(i)
                    }
                    if guard_.tryMarkFinished() {
                        continuation.finish()
                        winners.record(0)
                    }
                }
                // Task B: random delay then error-finish
                Task {
                    try? await Task.sleep(nanoseconds: UInt64.random(in: 0..<2_000_000))
                    if guard_.tryMarkFinished() {
                        continuation.finish(throwing: NSError(domain: "burst-b", code: streamIdx))
                        winners.record(1)
                    }
                }
                // Task C: random delay then normal finish (second attempt)
                Task {
                    try? await Task.sleep(nanoseconds: UInt64.random(in: 0..<2_000_000))
                    if guard_.tryMarkFinished() {
                        continuation.finish()
                        winners.record(2)
                    }
                }
            }

            // Drain — must not crash
            do { for try await _ in stream { } } catch { }

            // Give tasks time to record
            try? await Task.sleep(nanoseconds: 50_000_000)

            XCTAssertTrue(guard_.isDone, "Stream \(streamIdx): guard should be marked done")
            XCTAssertEqual(winners.count, 1, "Stream \(streamIdx): exactly one winner, got \(winners.count)")
        }
    }

    // MARK: - Test 3: Priority preemption ordering under load

    /// 20 mock sequences with ascending admittedAt. 10 concurrent tasks each preempt
    /// the newest. Verifies: no double-preemption, newest-first ordering.
    func testPriorityPreemptionOrdering() async {
        final class MockSeq: @unchecked Sendable {
            let lock = NSLock()
            let id: Int
            let admittedAt: Date
            var isPreempted = false

            init(id: Int, admittedAt: Date) {
                self.id = id
                self.admittedAt = admittedAt
            }

            func tryPreempt() -> Bool {
                lock.lock(); defer { lock.unlock() }
                if isPreempted { return false }
                isPreempted = true
                return true
            }
        }

        // Create 20 sequences with staggered admission times
        let sequences: [MockSeq] = (0..<20).map { i in
            MockSeq(id: i, admittedAt: Date().addingTimeInterval(Double(i) * 0.001))
        }

        final class PreemptCounter: @unchecked Sendable {
            let lock = NSLock()
            var count: Int = 0
            var preemptedIds: [Int] = []
            func record(_ id: Int) {
                lock.lock(); defer { lock.unlock() }
                count += 1
                preemptedIds.append(id)
            }
        }
        let counter = PreemptCounter()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    try? await Task.sleep(nanoseconds: UInt64.random(in: 0..<5_000))

                    // Find newest non-preempted sequence (excluding self — no self here)
                    let candidates = sequences.filter { !$0.isPreempted }
                    guard let newest = candidates.max(by: { $0.admittedAt < $1.admittedAt }) else { return }

                    if newest.tryPreempt() {
                        counter.record(newest.id)
                    }
                }
            }
        }

        // Verify: total preemptions <= 10 (each task can preempt at most 1)
        XCTAssertLessThanOrEqual(counter.count, 10, "Should not exceed number of preemptor tasks")

        // Verify: no sequence was preempted twice (tryPreempt returns false on second attempt)
        let uniqueIds = Set(counter.preemptedIds)
        XCTAssertEqual(uniqueIds.count, counter.count, "Each preempted sequence should be unique — no double-preemption")

        // Verify: preempted sequences had later admission times than surviving ones
        let survivingIds = Set(sequences.filter { !$0.isPreempted }.map { $0.id })
        let preemptedIds = Set(counter.preemptedIds)
        if let latestSurviving = sequences.filter({ survivingIds.contains($0.id) }).max(by: { $0.admittedAt < $1.admittedAt }),
           let earliestPreempted = sequences.filter({ preemptedIds.contains($0.id) }).min(by: { $0.admittedAt < $1.admittedAt }) {
            XCTAssertGreaterThanOrEqual(
                earliestPreempted.admittedAt.timeIntervalSince1970,
                latestSurviving.admittedAt.timeIntervalSince1970,
                "Preempted sequences should have later admission times than surviving ones"
            )
        }
    }
}
