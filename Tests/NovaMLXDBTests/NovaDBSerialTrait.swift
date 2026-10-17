import Foundation
import Testing

/// Serializes every suite that touches the `NovaDB.shared` singleton.
///
/// `NovaDB.setup(baseDir:)` rebinds the shared database pools, so two suites
/// running in parallel would rebind mid-test and read each other's freshly
/// created temp-dir databases (observed: TokenhubStore and LoadBalancerStore
/// suites racing, records "missing" from the DB a test just wrote to).
/// `.serialized` alone only serializes WITHIN a suite — this trait provides
/// the cross-suite critical section.
struct NovaDBSerialTrait: SuiteTrait, TestScoping {
    func provideScope(
        for test: Test,
        testCase: Test.Case?,
        performing function: @Sendable () async throws -> Void
    ) async throws {
        await NovaDBSerialGate.shared.wait()
        defer { NovaDBSerialGate.shared.signal() }
        try await function()
    }
}

extension Trait where Self == NovaDBSerialTrait {
    static var novaDBSerial: NovaDBSerialTrait { NovaDBSerialTrait() }
}

/// Async mutex (binary semaphore). NSLock is only held for the non-suspending
/// state transitions, never across `await`; signal() stays synchronous so it
/// can run in a `defer`.
private final class NovaDBSerialGate: @unchecked Sendable {
    static let shared = NovaDBSerialGate()

    private let lock = NSLock()
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        await withCheckedContinuation { cont in
            lock.lock()
            if busy {
                waiters.append(cont)
                lock.unlock()
            } else {
                busy = true
                lock.unlock()
                cont.resume()
            }
        }
    }

    func signal() {
        lock.lock()
        if waiters.isEmpty {
            busy = false
        } else {
            waiters.removeFirst().resume()
        }
        lock.unlock()
    }
}
