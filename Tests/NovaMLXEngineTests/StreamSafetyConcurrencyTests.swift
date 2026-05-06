import XCTest
@testable import NovaMLXEngine
import NovaMLXCore

/// Tests the safeYield / safeFinish pattern used by ActiveStreamSequence.
/// Rather than constructing a full ActiveStreamSequence (which requires KVCache,
/// CompiledSampler, etc.), these tests replicate the exact guard+continuation
/// pattern to verify:
///   1. safeYield is a no-op after finish
///   2. safeFinish returns false on double-call (no double-finish)
///   3. Concurrent yield + finish don't crash the stream
///   4. Concurrent finish from multiple paths — exactly one wins
final class StreamSafetyConcurrencyTests: XCTestCase {

    // MARK: - safeFinish double-call prevention

    func testSafeFinishReturnsTrueOnFirstCallFalseOnSecond() {
        let guard_ = FinishGuard()
        XCTAssertTrue(guard_.tryMarkFinished(), "First call should return true")
        XCTAssertFalse(guard_.tryMarkFinished(), "Second call should return false")
    }

    // MARK: - safeYield after finish is no-op

    func testYieldAfterMarkIsSkipped() async {
        let guard_ = FinishGuard()
        guard_.tryMarkFinished()

        var yielded: [Int] = []
        let stream = AsyncThrowingStream<Int, Error> { continuation in
            // Should be skipped — guard already marked
            guard !guard_.isDone else {
                continuation.finish()
                return
            }
            continuation.yield(42)
            continuation.finish()
        }

        do {
            for try await val in stream {
                yielded.append(val)
            }
        } catch { }
        XCTAssertEqual(yielded, [], "No values should be yielded after guard is marked")
    }

    // MARK: - Concurrent yield + finish don't crash

    func testConcurrentYieldAndFinishNoCrash() async {
        let guard_ = FinishGuard()
        let stream = AsyncThrowingStream<Int, Error> { continuation in
            // Task A: yields tokens in a loop
            Task {
                for i in 0..<1000 {
                    guard !guard_.isDone else { break }
                    continuation.yield(i)
                }
            }
            // Task B: finishes after small delay
            Task {
                try? await Task.sleep(nanoseconds: 50_000) // 50µs
                if guard_.tryMarkFinished() {
                    continuation.finish()
                }
            }
        }

        // Drain — must not crash with "already finished" or EXC_BAD_ACCESS
        var count = 0
        do { for try await _ in stream { count += 1 } } catch { }
        // We got here without crash — that's the assertion
        XCTAssertGreaterThanOrEqual(count, 0)
    }

    // MARK: - Multiple concurrent finishers — exactly one wins

    func testMultipleConcurrentFinishersExactlyOneWins() async {
        let concurrency = 32
        let guard_ = FinishGuard()
        final class ResultBox: @unchecked Sendable {
            let lock = NSLock()
            var results: [Bool]
            init(count: Int) { results = Array(repeating: false, count: count) }
            func set(_ i: Int) { lock.lock(); defer { lock.unlock() }; results[i] = true }
            var winnerCount: Int {
                lock.lock(); defer { lock.unlock() }
                return results.filter { $0 }.count
            }
        }
        let box = ResultBox(count: concurrency)

        let stream = AsyncThrowingStream<Int, Error> { continuation in
            for i in 0..<concurrency {
                Task {
                    // Random tiny jitter to stagger starts
                    try? await Task.sleep(nanoseconds: UInt64.random(in: 0..<10_000))
                    if guard_.tryMarkFinished() {
                        continuation.finish()
                        box.set(i)
                    }
                }
            }
        }

        // Drain
        do { for try await _ in stream { } } catch { }

        // Give tasks time to record results
        try? await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(box.winnerCount, 1, "Exactly one concurrent finisher should win")
    }

    // MARK: - Simulates fusedDecodeStep vs abort race

    func testDecodeStepVsAbortRace() async {
        let guard_ = FinishGuard()
        final class Flags: @unchecked Sendable {
            let lock = NSLock()
            var _decode = false, _abort = false
            var decode: Bool { lock.lock(); defer { lock.unlock() }; return _decode }
            var abort: Bool { lock.lock(); defer { lock.unlock() }; return _abort }
            func setDecode() { lock.lock(); defer { lock.unlock() }; _decode = true }
            func setAbort() { lock.lock(); defer { lock.unlock() }; _abort = true }
        }
        let flags = Flags()

        let stream = AsyncThrowingStream<Int, Error> { continuation in
            // Decode step path: normal finish
            Task {
                for i in 0..<100 {
                    guard !guard_.isDone else { break }
                    continuation.yield(i)
                }
                if guard_.tryMarkFinished() {
                    continuation.finish()
                    flags.setDecode()
                }
            }
            // Abort path: error finish (slightly delayed)
            Task {
                try? await Task.sleep(nanoseconds: 10_000)
                if guard_.tryMarkFinished() {
                    continuation.finish(throwing: NSError(domain: "abort", code: 1))
                    flags.setAbort()
                }
            }
        }

        // Drain — must not crash regardless of which path wins
        do {
            for try await _ in stream { }
        } catch {
            // Abort path won — this is fine
        }

        try? await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertTrue(flags.decode || flags.abort, "At least one path should have finished")
        XCTAssertFalse(flags.decode && flags.abort, "Both paths should NOT have finished")
    }

    // MARK: - Struct copy shares guard (the actual bug scenario)

    /// ActiveStreamSequence is a struct — copies don't share `isFinished`
    /// but DO share `finishGuard` (reference type). This test verifies
    /// the guard prevents double-finish even across struct copies.
    func testStructCopySharesFinishGuard() async {
        let guard_ = FinishGuard()
        final class Flags: @unchecked Sendable {
            let lock = NSLock()
            var _c1 = false, _c2 = false
            var c1: Bool { lock.lock(); defer { lock.unlock() }; return _c1 }
            var c2: Bool { lock.lock(); defer { lock.unlock() }; return _c2 }
            func setC1() { lock.lock(); defer { lock.unlock() }; _c1 = true }
            func setC2() { lock.lock(); defer { lock.unlock() }; _c2 = true }
        }
        let flags = Flags()

        let stream = AsyncThrowingStream<Int, Error> { continuation in
            // "Copy 1" of the struct — has its own isFinished but shares guard
            Task {
                try? await Task.sleep(nanoseconds: 1_000)
                if guard_.tryMarkFinished() {
                    continuation.finish()
                    flags.setC1()
                }
            }
            // "Copy 2" — separate isFinished, same guard
            Task {
                try? await Task.sleep(nanoseconds: 1_000)
                if guard_.tryMarkFinished() {
                    continuation.finish()
                    flags.setC2()
                }
            }
        }

        do { for try await _ in stream { } } catch { }

        try? await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertTrue(flags.c1 || flags.c2)
        XCTAssertFalse(flags.c1 && flags.c2,
                       "Struct copies sharing a FinishGuard must not both finish the continuation")
    }
}
