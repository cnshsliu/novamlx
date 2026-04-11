import Foundation
import NovaMLXCore

public final class NovaMLXLock: @unchecked Sendable {
    private var _lock: os_unfair_lock

    public init() {
        _lock = os_unfair_lock()
    }

    public func withLock<T>(_ body: () throws -> T) rethrows -> T {
        os_unfair_lock_lock(&_lock)
        defer { os_unfair_lock_unlock(&_lock) }
        return try body()
    }
}
