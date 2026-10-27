import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import NovaMLXPrefixCache

@Suite("PrefixCache Hot Tier")
struct PrefixCacheTierTests {

    private final class ScriptedStore: PrefixCacheDurableStore, @unchecked Sendable {
        struct Planted {
            var layers: [[MLXArray]]
            var classes: [String]
            var meta: [[String]]
            var createdAt: Date
        }

        private let lock = NSLock()
        private var planted: [String: Planted] = [:]
        private var loadCount = 0
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        var blocksLoads = false

        func plant(hash: BlockHash, layers: [[MLXArray]], createdAt: Date = Date()) {
            lock.lock()
            planted[Self.key(hash)] = Planted(
                layers: layers, classes: ["KVCache"], meta: [[]], createdAt: createdAt)
            lock.unlock()
        }

        func hasBlock(_ hash: BlockHash) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return planted[Self.key(hash)] != nil
        }

        func saveBlock(
            hash: BlockHash,
            cacheData: [[MLXArray]],
            layerClasses: [String],
            metaStates: [[String]],
            tokenCount: Int,
            modelName: String
        ) {
            lock.lock()
            planted[Self.key(hash)] = Planted(
                layers: cacheData,
                classes: layerClasses,
                meta: metaStates,
                createdAt: Date()
            )
            lock.unlock()
        }

        func loadBlock(_ hash: BlockHash) -> (arrays: [String: [MLXArray]], metadata: [String: String], numLayers: Int)? {
            if blocksLoads {
                entered.signal()
                _ = release.wait(timeout: .now() + 5)
            }
            lock.lock()
            loadCount += 1
            let entry = planted[Self.key(hash)]
            lock.unlock()
            guard let entry else { return nil }
            var arrays: [String: [MLXArray]] = [:]
            for (index, layer) in entry.layers.enumerated() {
                arrays["\(index)"] = layer
            }
            var metadata = [
                "num_layers": String(entry.layers.count),
                "created_at": String(entry.createdAt.timeIntervalSince1970),
                "token_count": "4",
            ]
            if let data = try? JSONEncoder().encode(entry.classes),
               let text = String(data: data, encoding: .utf8) {
                metadata["layer_cache_types"] = text
            }
            if let data = try? JSONEncoder().encode(entry.meta),
               let text = String(data: data, encoding: .utf8) {
                metadata["layer_meta_states"] = text
            }
            return (arrays, metadata, entry.layers.count)
        }

        func discardExpired(now: Date) {}

        func clear() {
            lock.lock()
            planted.removeAll()
            lock.unlock()
        }

        var blockCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return planted.count
        }

        var totalSize: UInt64 { UInt64(blockCount) }

        var loads: Int {
            lock.lock()
            defer { lock.unlock() }
            return loadCount
        }

        private static func key(_ hash: BlockHash) -> String {
            hash.map { String(format: "%02x", $0) }.joined()
        }
    }

    private final class ResultBox: @unchecked Sendable {
        var value: PrefixCacheManager.PrefixResult?
    }

    private func manager(
        store: ScriptedStore,
        hotBytes: Int,
        blockSize: Int = 4
    ) -> PrefixCacheManager {
        PrefixCacheManager(
            config: PrefixCacheConfig(
                blockSize: blockSize,
                maxBlocks: 32,
                initialBlocks: 8,
                ssdCacheDir: nil,
                hotCacheMaxBytes: hotBytes,
                ttl: 3_600
            ),
            modelName: "tier-model",
            durableStore: store
        )
    }

    private func plainCache(sequence: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray.zeros([1, 1, sequence, 4]),
            MLXArray.zeros([1, 1, sequence, 4]),
        ]
        return cache
    }

    @Test("a stored prefix is served from RAM and does not read disk")
    func hotHitDoesNotTouchDisk() {
        let store = ScriptedStore()
        let manager = manager(store: store, hotBytes: 8_000_000)
        let tokens = Array(0..<8)
        manager.storeCache(tokenIds: tokens, cache: [plainCache(sequence: 8)])

        let fetched = manager.fetchPrefix(tokenIds: tokens)
        #expect(fetched.cachedTokenCount == 8)
        #expect(fetched.cache != nil)
        #expect(fetched.scheduledAsyncLoads == 0)
        #expect(store.loads == 0)
        #expect(manager.getStats().hotCacheBlockCount == 2)
        #expect(store.blockCount == 2)
    }

    @Test("a cold long-term block is promoted in the background")
    func asyncLoadDoesNotBlockFetch() {
        let store = ScriptedStore()
        let tokens = Array(0..<4)
        let hash = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: tokens, modelName: "tier-model")
        store.plant(hash: hash, layers: [
            [MLXArray.zeros([1, 1, 4, 2]), MLXArray.zeros([1, 1, 4, 2])],
        ])
        let manager = manager(store: store, hotBytes: 8_000_000)
        store.blocksLoads = true
        defer { store.release.signal() }

        let box = ResultBox()
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            box.value = manager.fetchPrefix(tokenIds: tokens)
            done.signal()
        }
        #expect(done.wait(timeout: .now() + 0.5) == .success)
        #expect(box.value?.cache == nil)
        #expect(box.value?.cachedTokenCount == 0)
        #expect(box.value?.scheduledAsyncLoads == 1)
        #expect(store.entered.wait(timeout: .now() + 2) == .success)

        store.release.signal()
        var promoted: PrefixCacheManager.PrefixResult?
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            let again = manager.fetchPrefix(tokenIds: tokens)
            if again.cachedTokenCount > 0 {
                promoted = again
                break
            }
            Thread.sleep(forTimeInterval: 0.02)
        }
        #expect(promoted?.cachedTokenCount == 4)
        #expect(promoted?.cache != nil)
        #expect(manager.getStats().hotCacheBlockCount == 1)
    }

    @Test("the RAM tier drops the least recently used block when it is over budget")
    func demotesLeastRecentlyUsed() {
        let hot = HotPrefixTier(maxBytes: 100)
        let hashA: BlockHash = [0x01]
        let hashB: BlockHash = [0x02]
        let now = Date()
        let block = HotPrefixBlock(
            layers: [],
            layerClasses: ["KVCache"],
            metaStates: [[]],
            bytes: 60,
            createdAt: now,
            lastAccess: now
        )
        #expect(hot.insert(hashA, block: block) == 0)
        #expect(hot.insert(hashB, block: block) == 1)
        #expect(hot.count == 1)
        #expect(hot.bytes == 60)
        #expect(hot.peek(hashA) == nil)
        #expect(hot.peek(hashB) != nil)

        let oversized = HotPrefixBlock(
            layers: [],
            layerClasses: ["KVCache"],
            metaStates: [[]],
            bytes: 1_000,
            createdAt: now,
            lastAccess: now
        )
        #expect(hot.insert([0x03], block: oversized) == 0)
        #expect(hot.peek([0x03]) == nil)
        #expect(hot.peek(hashB) != nil)
    }

    @Test("a zero-byte hot budget does not retain blocks the long-term store kept")
    func zeroBudgetSkipsRam() {
        let store = ScriptedStore()
        let manager = manager(store: store, hotBytes: 0)
        manager.storeCache(tokenIds: Array(0..<8), cache: [plainCache(sequence: 8)])
        let stats = manager.getStats()
        #expect(stats.hotCacheBlockCount == 0)
        #expect(stats.ssdBlockCount == 2)
    }

    @Test("without a long-term store or RAM tier, a pool hit stays metadata-only")
    func metadataOnlyWhenUnconfigured() {
        let manager = PrefixCacheManager(
            config: PrefixCacheConfig(blockSize: 4, maxBlocks: 16, initialBlocks: 4),
            modelName: "meta-only"
        )
        let tokens = Array(0..<8)
        manager.storeCache(tokenIds: tokens, cache: [plainCache(sequence: 8)])
        let fetched = manager.fetchPrefix(tokenIds: tokens)
        #expect(fetched.cachedTokenCount == 8)
        #expect(fetched.cache == nil)
        #expect(fetched.scheduledAsyncLoads == 0)
    }
}
