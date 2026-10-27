import Foundation
import MLX
import NovaMLXUtils

/// One decoded prefix block kept in RAM.
struct HotPrefixBlock: @unchecked Sendable {
    let layers: [[MLXArray]]
    let layerClasses: [String]
    let metaStates: [[String]]
    let bytes: Int
    let createdAt: Date
    var lastAccess: Date
}

/// RAM tier in front of the long-term prefix files.
///
/// Hits are served from here without touching disk. A block that does not
/// fit, or that falls off the LRU list, is dropped from RAM only — the
/// long-term store keeps the file. Eviction is least-recently-used. A prefix
/// hit still requires the contiguous head to be resident, so a demoted head
/// is loaded again asynchronously.
final class HotPrefixTier: @unchecked Sendable {
    private var entries: [String: HotPrefixBlock] = [:]
    private var lru: [String] = []
    private var totalBytes: Int = 0
    private let maxBytes: Int
    private let lock = NovaMLXLock()

    init(maxBytes: Int) {
        self.maxBytes = max(maxBytes, 0)
    }

    var limit: Int { maxBytes }
    var count: Int { lock.withLock { entries.count } }
    var bytes: Int { lock.withLock { totalBytes } }

    func peek(_ hash: BlockHash) -> HotPrefixBlock? {
        lock.withLock { entries[Self.key(hash)] }
    }

    func get(_ hash: BlockHash, now: Date) -> HotPrefixBlock? {
        lock.withLock {
            let k = Self.key(hash)
            guard var entry = entries[k] else { return nil }
            entry.lastAccess = now
            entries[k] = entry
            lru.removeAll { $0 == k }
            lru.append(k)
            return entry
        }
    }

    /// Insert or refresh a block, then evict the LRU tail until the tier fits.
    /// A block larger than the budget is not retained. Returns how many
    /// other blocks were demoted.
    @discardableResult
    func insert(_ hash: BlockHash, block: HotPrefixBlock) -> Int {
        guard maxBytes > 0, block.bytes <= maxBytes else { return 0 }
        return lock.withLock {
            let k = Self.key(hash)
            if let existing = entries[k] {
                totalBytes -= existing.bytes
                lru.removeAll { $0 == k }
            }
            entries[k] = block
            lru.append(k)
            totalBytes += block.bytes
            var evicted = 0
            while totalBytes > maxBytes, let oldest = lru.first {
                lru.removeFirst()
                if let dropped = entries.removeValue(forKey: oldest) {
                    totalBytes -= dropped.bytes
                    if oldest != k { evicted += 1 }
                }
            }
            return evicted
        }
    }

    /// Remove blocks whose `createdAt` is older than `ttl`.
    func discardExpired(ttl: TimeInterval, now: Date) -> Int {
        lock.withLock {
            let stale = entries.compactMap { key, block -> String? in
                now.timeIntervalSince(block.createdAt) > ttl ? key : nil
            }
            for k in stale {
                if let dropped = entries.removeValue(forKey: k) {
                    totalBytes -= dropped.bytes
                    lru.removeAll { $0 == k }
                }
            }
            return stale.count
        }
    }

    func clear() {
        lock.withLock {
            entries.removeAll()
            lru.removeAll()
            totalBytes = 0
        }
    }

    static func estimatedBytes(_ layers: [[MLXArray]]) -> Int {
        var total = 0
        for layer in layers {
            for array in layer {
                let logical = array.nbytes
                if logical > 0 {
                    total += logical
                } else {
                    let elements = array.shape.reduce(1, *)
                    total += elements * max(array.dtype.size, 1)
                }
            }
        }
        return max(total, 1)
    }

    static func key(_ hash: BlockHash) -> String {
        hash.map { String(format: "%02x", $0) }.joined()
    }
}
