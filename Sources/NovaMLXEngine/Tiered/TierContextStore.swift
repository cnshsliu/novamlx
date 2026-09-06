import Foundation
import MLX
import MLXNN
import MLXLMCommon
import NovaMLXUtils

// NovaMLX-TIE: TierContextStore
//
// Sync-safe registry for per-Linear/SwitchLinear TierHookContext. Lives OUTSIDE
// the TierHookCoordinator actor so that sync hook closures (called from
// Linear.callAsFunction before matmul) can look up contexts without `await`.
//
// All access goes through NSLock — safe from sync or async code.
//
// Phase 4: also tracks loaded instances with size + reset closure for LRU
// eviction. When memory budget exceeded, oldest entries are evicted (weight
// zeroed) so the next call triggers a fresh SSD load.

public final class TierContextStore: @unchecked Sendable {
    public static let shared = TierContextStore()

    private let lock = NSLock()
    private var switchLinearContexts: [ObjectIdentifier: TierHookContext] = [:]
    private var linearContexts: [ObjectIdentifier: TierHookContext] = [:]

    /// Per-loaded-instance tracking: reset closure (zeros the weight) + byte size + timing.
    /// Used for LRU eviction when memory budget is exceeded.
    private struct LoadedEntry {
        let reset: () -> Void
        let bytes: Int64
        let loadedAt: Date
        var lastAccess: Date
    }
    private var loadedEntries: [ObjectIdentifier: LoadedEntry] = [:]

    public enum HookKind: Sendable {
        case switchLinear
        case linear
    }

    private init() {}

    // MARK: - Setters (called by TierHookCoordinator actor)

    public func setSwitchLinear(_ id: ObjectIdentifier, _ ctx: TierHookContext) {
        lock.lock(); defer { lock.unlock() }
        switchLinearContexts[id] = ctx
    }

    public func setLinear(_ id: ObjectIdentifier, _ ctx: TierHookContext) {
        lock.lock(); defer { lock.unlock() }
        linearContexts[id] = ctx
    }

    /// Mark instance as loaded + register a reset closure that zeros its weight.
    /// Called by sync hook after SSD read. The closure captures the module weakly.
    public func markLoaded(_ id: ObjectIdentifier, bytes: Int64 = 0, reset: @escaping () -> Void = {}) {
        lock.lock(); defer { lock.unlock() }
        let now = Date()
        loadedEntries[id] = LoadedEntry(
            reset: reset, bytes: bytes, loadedAt: now, lastAccess: now
        )
    }

    /// Update lastAccess for LRU. Called by sync hook fast-path (cheap lock).
    public func touch(_ id: ObjectIdentifier) {
        lock.lock(); defer { lock.unlock() }
        loadedEntries[id]?.lastAccess = Date()
    }

    public func clearAll() {
        lock.lock(); defer { lock.unlock() }
        switchLinearContexts.removeAll()
        linearContexts.removeAll()
        loadedEntries.removeAll()
    }

    public func clearForPolicy(_ policyId: ObjectIdentifier, ids: [ObjectIdentifier]) {
        lock.lock(); defer { lock.unlock() }
        for id in ids {
            switchLinearContexts.removeValue(forKey: id)
            linearContexts.removeValue(forKey: id)
            loadedEntries.removeValue(forKey: id)
        }
    }

    // MARK: - Sync lookups (called from sync hooks)

    public func get(_ id: ObjectIdentifier, kind: HookKind) -> TierHookContext? {
        lock.lock(); defer { lock.unlock() }
        switch kind {
        case .switchLinear: return switchLinearContexts[id]
        case .linear: return linearContexts[id]
        }
    }

    public func isLoaded(_ id: ObjectIdentifier) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return loadedEntries[id] != nil
    }

    /// Total bytes of currently loaded weights. Used by MetricsStore gauges.
    public var loadedBytes: Int64 {
        lock.lock(); defer { lock.unlock() }
        return loadedBytesUnlocked
    }

    /// Number of currently loaded entries.
    public var loadedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return loadedEntries.count
    }

    /// Unlocked accessor for internal use (caller must hold `lock`).
    private var loadedBytesUnlocked: Int64 {
        loadedEntries.values.reduce(0) { $0 + $1.bytes }
    }

    // MARK: - LRU Eviction (Phase 4)

    /// Evict oldest loaded entries until total bytes <= byteBudget. Returns
    /// the number of entries evicted. Each evicted entry's reset closure runs
    /// (zeros the weight) and the entry is removed, so the next call to that
    /// Linear/SwitchLinear triggers a fresh SSD load.
    ///
    /// `minIdleSeconds` > 0 skips entries touched more recently than that —
    /// the generate loop must not evict the working set mid-token.
    /// `minIdleSeconds == 0` is a forced shrink (unbind / tests).
    ///
    /// Safe to call from sync or async context. Reset closures run OUTSIDE the lock.
    @discardableResult
    public func evictToFit(byteBudget: Int64, minIdleSeconds: TimeInterval = 0) -> Int {
        var evicted = 0
        var toReset: [() -> Void] = []
        let now = Date()
        lock.lock()
        while loadedBytesUnlocked > byteBudget && !loadedEntries.isEmpty {
            let candidates = loadedEntries.filter { entry in
                minIdleSeconds <= 0
                    || now.timeIntervalSince(entry.value.lastAccess) >= minIdleSeconds
            }
            guard let oldest = candidates.min(by: { $0.value.lastAccess < $1.value.lastAccess }) else {
                break
            }
            let id = oldest.key
            toReset.append(loadedEntries[id]!.reset)
            loadedEntries.removeValue(forKey: id)
            evicted += 1
        }
        lock.unlock()
        for reset in toReset { reset() }
        return evicted
    }
}
