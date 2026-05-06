import XCTest
@testable import NovaMLXEngine

/// Thread-safe result collector for Swift 6 sending compliance.
private final class ResultBox: @unchecked Sendable {
    let lock = NSLock()
    var results: [Bool]
    init(count: Int) { results = Array(repeating: false, count: count) }
    func set(_ index: Int, _ value: Bool) { lock.lock(); defer { lock.unlock() }; results[index] = value }
    var winnerCount: Int {
        lock.lock(); defer { lock.unlock() }
        return results.filter { $0 }.count
    }
}

final class FinishGuardConcurrencyTests: XCTestCase {

    // MARK: - Single-threaded correctness

    func testFirstTryMarkReturnsTrue() {
        let guard_ = FinishGuard()
        XCTAssertTrue(guard_.tryMarkFinished())
    }

    func testSecondTryMarkReturnsFalse() {
        let guard_ = FinishGuard()
        _ = guard_.tryMarkFinished()
        XCTAssertFalse(guard_.tryMarkFinished())
    }

    func testIsDoneFalseInitially() {
        let guard_ = FinishGuard()
        XCTAssertFalse(guard_.isDone)
    }

    func testIsDoneTrueAfterMark() {
        let guard_ = FinishGuard()
        _ = guard_.tryMarkFinished()
        XCTAssertTrue(guard_.isDone)
    }

    func testRepeatedIsDoneReadsConsistent() {
        let guard_ = FinishGuard()
        _ = guard_.tryMarkFinished()
        for _ in 0..<100 {
            XCTAssertTrue(guard_.isDone)
        }
    }

    // MARK: - Concurrent contention

    /// Fire N concurrent tasks that all call tryMarkFinished().
    /// Exactly ONE must return true — the rest false.
    func testConcurrentTryMarkOnlyOneWins() async {
        let concurrency = 64
        let guard_ = FinishGuard()
        let box = ResultBox(count: concurrency)

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<concurrency {
                group.addTask {
                    box.set(i, guard_.tryMarkFinished())
                }
            }
        }

        XCTAssertEqual(box.winnerCount, 1, "Exactly one concurrent tryMarkFinished should return true")
        XCTAssertTrue(guard_.isDone)
    }

    /// Hammer tryMarkFinished from many tasks, verify isDone is always true afterward
    func testHighContentionFinalState() async {
        let guard_ = FinishGuard()

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<200 {
                group.addTask {
                    _ = guard_.tryMarkFinished()
                }
            }
        }

        XCTAssertTrue(guard_.isDone)
        // All subsequent calls must return false
        for _ in 0..<50 {
            XCTAssertFalse(guard_.tryMarkFinished())
        }
    }

    /// Simulate the actual race: finish guard + continuation finish.
    /// Two tasks both try to "finish" an AsyncThrowingStream — only one
    /// should succeed without crashing.
    func testDoubleFinishPreventionWithRealContinuation() async {
        let guard_ = FinishGuard()
        final class AtomicCounter: @unchecked Sendable {
            let lock = NSLock()
            var value = 0
            func increment() { lock.lock(); defer { lock.unlock() }; value += 1 }
        }
        let finishCount = AtomicCounter()

        let stream = AsyncThrowingStream<Int, Error> { continuation in
            // Task A: tries to finish
            Task {
                if guard_.tryMarkFinished() {
                    continuation.finish()
                    finishCount.increment()
                }
            }
            // Task B: tries to finish concurrently
            Task {
                // Small delay to create overlap
                try? await Task.sleep(nanoseconds: 1_000)
                if guard_.tryMarkFinished() {
                    continuation.finish()
                    finishCount.increment()
                }
            }
        }

        // Drain the stream — must not crash
        var count = 0
        do { for try await _ in stream { count += 1 } } catch { }
        XCTAssertEqual(count, 0)

        // Give tasks time to update finishCount
        try? await Task.sleep(nanoseconds: 100_000_000)
        XCTAssertEqual(finishCount.value, 1, "Exactly one path should have called finish()")
    }
}
