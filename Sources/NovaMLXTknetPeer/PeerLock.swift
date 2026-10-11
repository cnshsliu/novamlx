import Foundation
#if canImport(os)
import os
/// Darwin keeps `OSAllocatedUnfairLock`. Linux Swift 6.0 has no `os` module;
/// the wrapper below matches the `withLock` calls this target already makes.
public typealias PeerLock<State> = OSAllocatedUnfairLock<State>
#else
/// Lock with the same `init(initialState:)` / `withLock` shape as
/// `OSAllocatedUnfairLock`, for toolchains that don't ship the `os` module.
public final class PeerLock<State>: @unchecked Sendable {
    private let lock = NSLock()
    private var state: State

    public init(initialState: State) {
        self.state = initialState
    }

    public func withLock<R>(_ body: (inout State) throws -> R) rethrows -> R {
        lock.lock()
        defer { lock.unlock() }
        return try body(&state)
    }
}
#endif
