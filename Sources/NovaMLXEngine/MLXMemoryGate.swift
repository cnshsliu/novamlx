import Foundation
import MLX

/// Serializes `MLX.Memory` maintenance.
///
/// `clearCache` and `activeMemory` take MLX's memory lock. Calling one from
/// the generate thread while the memory enforcer or `checkMemoryFeasibility`
/// is in the other — especially while `MLXEngine`'s lock is also held —
/// deadlocks after a large OCR prefill. Callers must not invoke this gate
/// while already holding `MLXEngine`'s lock, and must not call it re-entrantly.
public enum MLXMemoryGate {
    private static let lock = NSLock()

    public static func clearCache() {
        lock.lock()
        defer { lock.unlock() }
        MLX.Memory.clearCache()
    }

    public static func activeBytes() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return UInt64(MLX.Memory.activeMemory)
    }
}
