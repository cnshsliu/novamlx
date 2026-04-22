import Foundation
import MLX
import MLXLMCommon
import NovaMLXUtils

public final class PrefixCacheManager: @unchecked Sendable {
    private let pool: PagedBlockPool
    private let ssdStore: SSDCacheStore?
    private let config: PrefixCacheConfig
    private let modelName: String

    private var hits: Int = 0
    private var misses: Int = 0
    private var tokensSaved: Int = 0
    private var evictions: Int = 0
    private let lock = NovaMLXLock()

    public init(config: PrefixCacheConfig, modelName: String) {
        self.config = config
        self.modelName = modelName
        self.pool = PagedBlockPool(
            blockSize: config.blockSize,
            maxBlocks: config.maxBlocks,
            initialBlocks: config.initialBlocks,
            modelName: modelName
        )
        if let ssdDir = config.ssdCacheDir {
            self.ssdStore = SSDCacheStore(cacheDir: ssdDir, maxSize: config.ssdMaxSizeBytes, ttl: config.ttl)
        } else {
            self.ssdStore = nil
        }
        NovaMLXLog.info("PrefixCacheManager initialized: model=\(modelName), blockSize=\(config.blockSize), ssd=\(config.ssdCacheDir != nil)")
    }

    public struct PrefixResult: @unchecked Sendable {
        public let cache: [KVCache]?
        public let cachedTokenCount: Int
        public let remainingTokenIds: [Int]
    }

    public func fetchPrefix(tokenIds: [Int]) -> PrefixResult {
        guard !tokenIds.isEmpty else {
            return PrefixResult(cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds)
        }

        let (matchedBlockIds, _) = pool.findSharedPrefix(tokenIds: tokenIds)

        if matchedBlockIds.isEmpty {
            lock.withLock { misses += 1 }
            NovaMLXLog.debug("Prefix cache miss for \(modelName): 0 blocks")
            return PrefixResult(cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds)
        }

        let cachedTokenCount = matchedBlockIds.count * config.blockSize
        let remaining = Array(tokenIds[cachedTokenCount...])

        guard let ssd = ssdStore else {
            lock.withLock {
                hits += 1
                tokensSaved += cachedTokenCount
            }
            NovaMLXLog.debug("Prefix cache hit (metadata only): \(matchedBlockIds.count) blocks, \(cachedTokenCount) tokens")
            return PrefixResult(cache: nil, cachedTokenCount: cachedTokenCount, remainingTokenIds: remaining)
        }

        var blockDataList: [[[MLXArray]]] = []
        var layerClasses: [String] = []
        var metaStatesList: [[String]] = []

        for blockId in matchedBlockIds {
            guard let block = pool.getBlock(blockId), let hash = block.blockHash else {
                continue
            }

            if let loaded = ssd.loadBlock(hash) {
                var layerData: [[MLXArray]] = []
                var sortedLayers: [(Int, [MLXArray])] = []
                for (key, arrays) in loaded.arrays {
                    if let layerIdx = Int(key) {
                        sortedLayers.append((layerIdx, arrays))
                    }
                }
                sortedLayers.sort { $0.0 < $1.0 }
                layerData = sortedLayers.map(\.1)
                blockDataList.append(layerData)

                if layerClasses.isEmpty {
                    if let typesStr = loaded.metadata["layer_cache_types"],
                       let data = typesStr.data(using: .utf8),
                       let types = try? JSONDecoder().decode([String].self, from: data) {
                        layerClasses = types
                    } else {
                        layerClasses = Array(repeating: "KVCache", count: loaded.numLayers)
                    }
                }
                if metaStatesList.isEmpty {
                    if let statesStr = loaded.metadata["layer_meta_states"],
                       let data = statesStr.data(using: .utf8),
                       let states = try? JSONDecoder().decode([[String]].self, from: data) {
                        metaStatesList = states
                    } else {
                        metaStatesList = Array(repeating: [], count: loaded.numLayers)
                    }
                }
            }
        }

        guard !blockDataList.isEmpty else {
            lock.withLock { misses += 1 }
            return PrefixResult(cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds)
        }

        let reconstructed = CacheBlockExtractor.reconstructKVCache(
            blockDataList: blockDataList,
            layerClasses: layerClasses,
            metaStatesList: metaStatesList
        )

        lock.withLock {
            hits += 1
            tokensSaved += cachedTokenCount
        }
        NovaMLXLog.info("Prefix cache hit: \(matchedBlockIds.count) blocks, \(cachedTokenCount) tokens cached, \(remaining.count) remaining")
        return PrefixResult(cache: reconstructed, cachedTokenCount: cachedTokenCount, remainingTokenIds: remaining)
    }

    public func storeCache(tokenIds: [Int], cache: [KVCache]) {
        guard !tokenIds.isEmpty, !cache.isEmpty else { return }

        let layerClasses = CacheBlockExtractor.getLayerClasses(from: cache)
        let metaStates = CacheBlockExtractor.getMetaStates(from: cache)

        let numFullBlocks = tokenIds.count / config.blockSize
        guard numFullBlocks > 0 else { return }

        let _ = lock.withLock { () -> Void in
            var parentHash: BlockHash? = nil

            for i in 0..<numFullBlocks {
                let start = i * config.blockSize
                let end = start + config.blockSize
                let blockTokens = Array(tokenIds[start..<end])

                let hash = BlockHasher.computeBlockHash(
                    parentHash: parentHash,
                    tokenIds: blockTokens,
                    modelName: modelName
                )

                if pool.getBlockByHash(hash) != nil {
                    parentHash = hash
                    continue
                }

                guard let block = pool.allocateBlock() else {
                    break
                }

                pool.registerBlock(block: block, tokenIds: blockTokens, parentHash: parentHash)

                if let ssd = ssdStore {
                    let sliced = CacheBlockExtractor.extractBlockSlices(
                        cache: cache,
                        startIdx: start,
                        endIdx: end
                    )
                    ssd.saveBlock(
                        hash: hash,
                        cacheData: sliced,
                        layerClasses: layerClasses,
                        metaStates: metaStates,
                        tokenCount: config.blockSize,
                        modelName: modelName
                    )
                }

                parentHash = hash
            }
        }
        NovaMLXLog.debug("Stored \(numFullBlocks) blocks for \(modelName)")
    }

    public func clear() {
        pool.clear()
        ssdStore?.clear()
        lock.withLock {
            hits = 0
            misses = 0
            tokensSaved = 0
            evictions = 0
        }
        NovaMLXLog.info("Prefix cache cleared for \(modelName)")
    }

    public func getStats() -> PrefixCacheStats {
        let (total, allocated, free, shared) = pool.getStats()
        return lock.withLock {
            PrefixCacheStats(
                hits: hits,
                misses: misses,
                tokensSaved: tokensSaved,
                evictions: evictions,
                totalBlocks: total,
                allocatedBlocks: allocated,
                freeBlocks: free,
                sharedBlocks: shared,
                ssdBlockCount: ssdStore?.blockCount ?? 0,
                ssdTotalSize: ssdStore?.totalSize ?? 0
            )
        }
    }
}
