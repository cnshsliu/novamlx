import Foundation
import Testing

/// Serializes every test suite that touches the `NovaDB.shared` singleton.
///
/// `NovaDB.setup(baseDir:)` rebinds the shared database pools, so two suites
/// running in parallel would rebind mid-test and read each other's freshly
/// created temp-dir databases (observed: TokenhubStore / LoadBalancerStore /
/// ModelManager suites racing, records "missing" from the DB a test just
/// wrote to). `.serialized` alone only serializes WITHIN a suite — this
/// trait provides the cross-suite critical section.
///
/// The lock is an OS-level flock on a well-known temp file rather than a
/// static property, because test targets are separate modules — a `static
/// let shared` would exist once per module and provide no mutual exclusion
/// across test targets that share this file (keep copies in sync!).
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

/// Process-wide, cross-module mutex via flock(2). Acquisition polls with
/// LOCK_NB + Task.sleep so no cooperative-pool thread is ever blocked.
private final class NovaDBSerialGate: @unchecked Sendable {
    static let shared = NovaDBSerialGate()

    private let fd: Int32

    private init() {
        let path = NSTemporaryDirectory() + "novamlx-test-novadb.flock"
        fd = open(path, O_CREAT | O_RDWR, 0o644)
    }

    func wait() async {
        guard fd >= 0 else { return } // best effort; never block a test run
        while flock(fd, LOCK_EX | LOCK_NB) != 0 {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    func signal() {
        guard fd >= 0 else { return }
        flock(fd, LOCK_UN)
    }
}
