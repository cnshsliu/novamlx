import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public struct PooledModel: @unchecked Sendable {
    let container: ModelContainer
    let loadedAt: Date
    var lastAccessed: Date
    var isPinned: Bool
    let estimatedSizeMB: UInt64

    init(container: ModelContainer, pinned: Bool = false) {
        self.container = container
        self.loadedAt = Date()
        self.lastAccessed = Date()
        self.isPinned = pinned
        // Use model weight size, not MLX.Memory.activeMemory which includes freed memory
        self.estimatedSizeMB = container.wiredReservationTicket.map { UInt64($0.size / 1_048_576) } ?? 0
    }
}

public final class EnginePool: @unchecked Sendable {
    public private(set) var pool: [String: PooledModel]
    private let maxMemoryMB: UInt64
    private let lock = NovaMLXLock()

    public init(maxMemoryMB: UInt64 = 0) {
        self.pool = [:]
        self.maxMemoryMB = maxMemoryMB
    }

    public func add(_ container: ModelContainer, pinned: Bool = false) {
        lock.withLock {
            pool[container.identifier.id] = PooledModel(container: container, pinned: pinned)
            NovaMLXLog.info("EnginePool: added \(container.identifier.id), pool size: \(pool.count)")
        }
    }

    public func remove(_ modelId: String) -> ModelContainer? {
        lock.withLock {
            guard let pooled = pool.removeValue(forKey: modelId) else { return nil }
            pooled.container.unload()
            NovaMLXLog.info("EnginePool: removed \(modelId), pool size: \(pool.count)")
            return pooled.container
        }
    }

    public func get(_ modelId: String) -> ModelContainer? {
        lock.withLock {
            guard var pooled = pool[modelId] else { return nil }
            pooled.lastAccessed = Date()
            pool[modelId] = pooled
            return pooled.container
        }
    }

    public func contains(_ modelId: String) -> Bool {
        lock.withLock { pool[modelId] != nil }
    }

    public func pin(_ modelId: String) {
        lock.withLock {
            pool[modelId]?.isPinned = true
        }
    }

    public func unpin(_ modelId: String) {
        lock.withLock {
            pool[modelId]?.isPinned = false
        }
    }

    @discardableResult
    public func evictLRU(excluding excludedModelId: String? = nil) -> UInt64? {
        lock.withLock {
            let candidates = pool.filter { !$0.value.isPinned && $0.key != excludedModelId }
                .sorted { $0.value.lastAccessed < $1.value.lastAccessed }

            guard let victim = candidates.first else {
                NovaMLXLog.warning("EnginePool: no evictable models")
                return nil
            }

            let freedMB = victim.value.estimatedSizeMB
            let modelId = victim.key
            pool[modelId]?.container.unload()
            pool.removeValue(forKey: modelId)
            NovaMLXLog.info("EnginePool: LRU evicted \(modelId) (\(freedMB)MB)")
            return freedMB
        }
    }

    // Removed evictIfNeeded(forNewModelSizeMB:) and totalMemoryMB().
    // Both used estimatedSizeMB (cumulative activeMemory at load time — double-counts shared
    // weights) and had zero callers. Replaced by MLXEngine.ensureMemoryHeadroom() which uses
    // ProcessMemoryEnforcer soft limit and real-time MLX.Memory.activeMemory.

    public var loadedModelIds: [String] {
        lock.withLock { Array(pool.keys) }
    }

    public var loadedModelCount: Int {
        lock.withLock { pool.count }
    }

    public func allModelInfo() -> [(id: String, pinned: Bool, lastAccessed: Date, sizeMB: UInt64)] {
        lock.withLock {
            pool.map { (id: $0.key, pinned: $0.value.isPinned, lastAccessed: $0.value.lastAccessed, sizeMB: $0.value.estimatedSizeMB) }
        }
    }
}
