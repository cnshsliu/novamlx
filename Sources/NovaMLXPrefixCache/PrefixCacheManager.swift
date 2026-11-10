import Foundation
import MLX
import MLXLMCommon
import NovaMLXUtils

public final class PrefixCacheManager: @unchecked Sendable {
    private let pool: PagedBlockPool
    private let durable: (any PrefixCacheDurableStore)?
    private let hot: HotPrefixTier

    /// Hybrid anchored entry: the sub-block tail (KV slice + recurrent state
    /// at the anchor length) parked under the matched chain's tip hash. Kept
    /// OUT of the block tables so a KV-only chain hit can never masquerade as
    /// a consistent hybrid restore.
    struct AnchorEntry {
        let tailTokens: [Int]
        let layers: [[MLXArray]]
        let layerClasses: [String]
        let anchorLen: Int
        var lastAccess: Date
    }
    private var anchors: [BlockHash: AnchorEntry] = [:]
    private let config: PrefixCacheConfig
    private let modelName: String
    private let clock: @Sendable () -> Date
    private let loadQueue = DispatchQueue(label: "com.novamlx.prefix-cache-reader", qos: .utility)

    private var hits: Int = 0
    private var misses: Int = 0
    private var tokensSaved: Int = 0
    private var evictions: Int = 0
    private var inflight: Set<String> = []
    private let lock = NovaMLXLock()

    /// How many SSD blocks one fetch may promote. The next request continues
    /// the chain, so a long prefix does not issue an unbounded read.
    static let maxAsyncLoadsPerFetch = 8

    public convenience init(
        config: PrefixCacheConfig,
        modelName: String,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        let store: SSDCacheStore?
        if let ssdDir = config.ssdCacheDir {
            store = SSDCacheStore(
                cacheDir: ssdDir,
                maxSize: config.ssdMaxSizeBytes,
                ttl: config.ttl,
                now: now
            )
        } else {
            store = nil
        }
        self.init(config: config, modelName: modelName, durableStore: store, now: now)
    }

    /// `durableStore` is the long-term tier. Pass nil for a RAM-only manager,
    /// or a test double whose `loadBlock` blocks, to prove fetch does not wait.
    public init(
        config: PrefixCacheConfig,
        modelName: String,
        durableStore: (any PrefixCacheDurableStore)?,
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.config = config
        self.modelName = modelName
        self.clock = now
        self.pool = PagedBlockPool(
            blockSize: config.blockSize,
            maxBlocks: config.maxBlocks,
            initialBlocks: config.initialBlocks,
            modelName: modelName
        )
        self.durable = durableStore
        self.hot = HotPrefixTier(maxBytes: config.hotCacheMaxBytes)
        NovaMLXLog.info(
            "PrefixCacheManager initialized: model=\(modelName), blockSize=\(config.blockSize), "
                + "ssd=\(durableStore != nil), hotBytes=\(config.hotCacheMaxBytes), ttl=\(Int(config.ttl))s"
        )
    }

    public struct PrefixResult: @unchecked Sendable {
        public let cache: [KVCache]?
        public let cachedTokenCount: Int
        public let remainingTokenIds: [Int]
        /// Blocks queued for a background read on this call. The returned
        /// cache never waits on those reads.
        public let scheduledAsyncLoads: Int

        public init(
            cache: [KVCache]?,
            cachedTokenCount: Int,
            remainingTokenIds: [Int],
            scheduledAsyncLoads: Int = 0
        ) {
            self.cache = cache
            self.cachedTokenCount = cachedTokenCount
            self.remainingTokenIds = remainingTokenIds
            self.scheduledAsyncLoads = scheduledAsyncLoads
        }
    }

    /// - Parameter requireAnchor: hybrid (GDN) callers pass `true` — a plain
    ///   KV-chain hit carries no recurrent state, and restoring one would
    ///   pair a KV prefix with a zero recurrent state (inconsistent cache).
    ///   Without an anchor continuation such a fetch is a miss.
    public func fetchPrefix(tokenIds: [Int], requireAnchor: Bool = false) -> PrefixResult {
        guard !tokenIds.isEmpty else {
            return PrefixResult(cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds)
        }

        let instant = clock()
        durable?.discardExpired(now: instant)
        let expiredHot = hot.discardExpired(ttl: config.ttl, now: instant)
        if expiredHot > 0 {
            lock.withLock { evictions += expiredHot }
        }

        // No long-term store and no RAM tier: keep the historical metadata
        // hit (token count only, no tensors) so callers that never configured
        // a directory behave as before.
        if durable == nil && config.hotCacheMaxBytes <= 0 {
            return fetchMetadataOnly(tokenIds: tokenIds)
        }

        let hashes = BlockHasher.computeChainHashes(
            tokenIds: tokenIds,
            blockSize: config.blockSize,
            modelName: modelName
        )
        guard !hashes.isEmpty else {
            lock.withLock { misses += 1 }
            return PrefixResult(cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds)
        }

        var blockDataList: [[[MLXArray]]] = []
        var layerClasses: [String] = []
        var metaStatesList: [[String]] = []
        var pending: [BlockHash] = []

        for (index, hash) in hashes.enumerated() {
            if let resident = hot.get(hash, now: instant) {
                blockDataList.append(resident.layers)
                if layerClasses.isEmpty {
                    layerClasses = resident.layerClasses
                    metaStatesList = resident.metaStates
                }
                continue
            }
            // The head of the missing suffix lives on disk. Promote it in
            // the background. Do not call loadBlock on this thread.
            if durable?.hasBlock(hash) == true {
                let rest = hashes[index...]
                for candidate in rest {
                    if hot.peek(candidate) != nil { continue }
                    if durable?.hasBlock(candidate) == true {
                        pending.append(candidate)
                    } else {
                        break
                    }
                    if pending.count >= Self.maxAsyncLoadsPerFetch { break }
                }
            }
            break
        }

        let scheduled = scheduleLoads(pending)

        guard !blockDataList.isEmpty else {
            lock.withLock { misses += 1 }
            return PrefixResult(
                cache: nil,
                cachedTokenCount: 0,
                remainingTokenIds: tokenIds,
                scheduledAsyncLoads: scheduled
            )
        }

        // Hybrid anchor continuation: probe anchor entries parked under
        // (matched tip hash, candidate tail lengths). The anchor's KV tail
        // slice and recurrent snapshot both end at the anchor length, so the
        // restored cache is internally consistent. Works for the multi-turn
        // case where the new prompt strictly extends the anchored prompt —
        // the matched tip is the same and the tail tokens coincide.
        let matchedBlocks = blockDataList.count
        let matchedTokenCount = matchedBlocks * config.blockSize
        var anchorHit: AnchorEntry?
        if !anchors.isEmpty, matchedBlocks > 0 || tokenIds.count < config.blockSize {
            let tip = matchedBlocks > 0 ? hashes[matchedBlocks - 1] : nil
            let maxTail = min(config.blockSize - 1, tokenIds.count - matchedTokenCount)
            for tailLen in 1...max(1, maxTail) where tailLen <= maxTail {
                let candidate = Array(
                    tokenIds[matchedTokenCount ..< (matchedTokenCount + tailLen)])
                let key = BlockHasher.computeBlockHash(
                    parentHash: tip, tokenIds: candidate, modelName: modelName)
                if var entry = anchors[key] {
                    entry.lastAccess = instant
                    anchors[key] = entry
                    anchorHit = entry
                    break
                }
            }
        }

        if let anchor = anchorHit {            var anchoredBlocks = blockDataList
            anchoredBlocks.append(anchor.layers)
            guard let reconstructedAnchored = CacheBlockExtractor.reconstructKVCache(
                blockDataList: anchoredBlocks,
                layerClasses: anchor.layerClasses,
                metaStatesList: metaStatesList)
            else {
                lock.withLock { misses += 1 }
                return PrefixResult(
                    cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds,
                    scheduledAsyncLoads: scheduled)
            }
            let cachedTokenCount = anchor.anchorLen
            let remaining = Array(tokenIds.dropFirst(cachedTokenCount))
            guard !remaining.isEmpty else {
                // Full-prompt hit leaves nothing to prefill and the decode
                // loop cannot start from an empty input — treat as a miss.
                lock.withLock { misses += 1 }
                return PrefixResult(
                    cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds,
                    scheduledAsyncLoads: scheduled)
            }
            lock.withLock {
                hits += 1
                tokensSaved += cachedTokenCount
            }
            NovaMLXLog.info(
                "Prefix cache anchored hit: \(cachedTokenCount) tokens, \(remaining.count) remaining"
            )
            return PrefixResult(
                cache: reconstructedAnchored,
                cachedTokenCount: cachedTokenCount,
                remainingTokenIds: remaining,
                scheduledAsyncLoads: scheduled
            )
        }

        if requireAnchor {
            NovaMLXLog.info(
                "[PrefixCache] hybrid fetch: matched=\(matchedBlocks) blocks, anchors=\(anchors.count), no anchor -> miss"
            )
            // Hybrid consistency: no anchor means no recurrent state — the
            // KV chain alone cannot restore this model.
            lock.withLock { misses += 1 }
            return PrefixResult(
                cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds,
                scheduledAsyncLoads: scheduled)
        }

        let reconstructed = CacheBlockExtractor.reconstructKVCache(
            blockDataList: blockDataList,
            layerClasses: layerClasses,
            metaStatesList: metaStatesList
        )
        let cachedTokenCount = matchedTokenCount
        let remaining = Array(tokenIds.dropFirst(cachedTokenCount))
        lock.withLock {
            hits += 1
            tokensSaved += cachedTokenCount
        }
        NovaMLXLog.info(
            "Prefix cache hot hit: \(blockDataList.count) blocks, \(cachedTokenCount) tokens, "
                + "asyncLoads=\(scheduled)"
        )
        return PrefixResult(
            cache: reconstructed,
            cachedTokenCount: cachedTokenCount,
            remainingTokenIds: remaining,
            scheduledAsyncLoads: scheduled
        )
    }

    /// Anchored hybrid store. Full blocks carry KV slices only; the partial
    /// tail block (or a single sub-block entry when the prompt is shorter
    /// than one block) carries the KV tail slice PLUS the recurrent state
    /// snapshotted at exactly `tokenIds.count` tokens — callers capture that
    /// snapshot right after prefill, before any decode step advances it, and
    /// pass it as `fixedStateAnchor` (per-layer arrays, empty for KV layers).
    /// A fetch that walks the full blocks continues into the tail and the
    /// restored cache is internally consistent.
    public func storeAnchoredCache(
        tokenIds: [Int], cache: [KVCache], fixedStateAnchor: [[MLXArray]]
    ) {
        guard !tokenIds.isEmpty, !cache.isEmpty else { return }
        guard CacheBlockExtractor.admitsLongTermKV(cache) else {
            NovaMLXLog.info("[PrefixCache] skip anchored store for \(modelName): wrapped ring")
            return
        }

        let layerClasses = CacheBlockExtractor.getLayerClasses(from: cache)
        let metaStates = CacheBlockExtractor.getMetaStates(from: cache)
        let blockSize = config.blockSize
        let numFullBlocks = tokenIds.count / blockSize
        let anchoredLen = tokenIds.count

        let instant = clock()
        lock.withLock {
            var parentHash: BlockHash? = nil

            // Full blocks: KV-only. The fixed-size state is attached to the
            // tail alone so byte accounting charges it once.
            for i in 0..<numFullBlocks {
                let start = i * blockSize
                let end = start + blockSize
                let blockTokens = Array(tokenIds[start..<end])
                let hash = BlockHasher.computeBlockHash(
                    parentHash: parentHash, tokenIds: blockTokens, modelName: modelName)
                if pool.getBlockByHash(hash) != nil {
                    parentHash = hash
                    continue
                }
                guard let block = pool.allocateBlock() else { break }
                pool.registerBlock(block: block, tokenIds: blockTokens, parentHash: parentHash)
                parentHash = hash

                var sliced = CacheBlockExtractor.extractBlockSlices(
                    cache: cache, startIdx: start, endIdx: end)
                // Strip fixed-size state from non-tail blocks.
                for li in 0..<min(sliced.count, layerClasses.count)
                where !CacheBlockExtractor.isKVClassName(layerClasses[li]) {
                    sliced[li] = []
                }
                let hotBlock = HotPrefixBlock(
                    layers: sliced, layerClasses: layerClasses, metaStates: metaStates,
                    bytes: HotPrefixTier.estimatedBytes(sliced),
                    createdAt: instant, lastAccess: instant)
                let demoted = hot.insert(hash, block: hotBlock)
                evictions += demoted
            }

            // Anchor entry: KV tail + recurrent state at `anchoredLen`, keyed
            // by the matched chain's tip hash so a later fetch that walks the
            // full blocks finds it WITHOUT the tail living in the block tables
            // (a KV-only chain hit must never masquerade as a hybrid restore).
            // Positional KV slices are valid against the live post-generation
            // cache because KV arrays only append; the fixed recurrent state
            // is replaced with the prefill-boundary anchor.
            let tailLen = anchoredLen % blockSize
            let tailTokens = Array(tokenIds.suffix(tailLen))
            let tailStart = anchoredLen - tailLen
            var tailSlices = CacheBlockExtractor.extractBlockSlices(
                cache: cache, startIdx: tailStart, endIdx: anchoredLen)
            for li in 0..<min(tailSlices.count, fixedStateAnchor.count)
            where !fixedStateAnchor[li].isEmpty {
                tailSlices[li] = fixedStateAnchor[li]
            }
            let anchorKey = BlockHasher.computeBlockHash(
                parentHash: parentHash, tokenIds: tailTokens, modelName: modelName)
            anchors[anchorKey] = AnchorEntry(
                tailTokens: tailTokens, layers: tailSlices,
                layerClasses: layerClasses, anchorLen: anchoredLen,
                lastAccess: instant)
            // Bounded LRU: each anchor holds a full recurrent snapshot, so a
            // handful is plenty for the multi-turn working set.
            while anchors.count > 8 {
                let oldest = anchors.min { $0.value.lastAccess < $1.value.lastAccess }?.key
                guard let oldest else { break }
                anchors.removeValue(forKey: oldest)
            }
        }
        NovaMLXLog.info("Anchored store: \(numFullBlocks) full blocks + anchor(len=\(anchoredLen)) for \(modelName)")
    }

    public func storeCache(tokenIds: [Int], cache: [KVCache]) {
        guard !tokenIds.isEmpty, !cache.isEmpty else { return }
        // Sliding rings and Flint's bounded window are not safe to flatten.
        // Flint's session prefix stays in memory (`select_prefix`).
        guard CacheBlockExtractor.admitsLongTermKV(cache) else {
            NovaMLXLog.info("[PrefixCache] skip long-term store for \(modelName): sliding-window KV")
            return
        }

        let layerClasses = CacheBlockExtractor.getLayerClasses(from: cache)
        let metaStates = CacheBlockExtractor.getMetaStates(from: cache)

        let numFullBlocks = tokenIds.count / config.blockSize
        guard numFullBlocks > 0 else { return }

        // Hybrid models: every block references the SAME fixed-size
        // conv/recurrent state (COW — no per-block copy), but naive byte
        // accounting would charge it once per block and evict the whole chain
        // immediately. Count fixed-size state bytes once, on the first block.
        let fixedStateBytes: Int = {
            let firstSlices = CacheBlockExtractor.extractBlockSlices(
                cache: cache, startIdx: 0, endIdx: config.blockSize)
            var total = 0
            for (i, cls) in layerClasses.enumerated()
            where !CacheBlockExtractor.isKVClassName(cls) {
                guard i < firstSlices.count else { continue }
                for array in firstSlices[i] {
                    let logical = array.nbytes
                    total += logical > 0
                        ? logical
                        : array.shape.reduce(1, *) * max(array.dtype.size, 1)
                }
            }
            return total
        }()

        let instant = clock()
        lock.withLock {
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

                let sliced = CacheBlockExtractor.extractBlockSlices(
                    cache: cache,
                    startIdx: start,
                    endIdx: end
                )
                let hotBlock = HotPrefixBlock(
                    layers: sliced,
                    layerClasses: layerClasses,
                    metaStates: metaStates,
                    bytes: max(
                        HotPrefixTier.estimatedBytes(sliced) - (i == 0 ? 0 : fixedStateBytes),
                        0),
                    createdAt: instant,
                    lastAccess: instant
                )
                let demoted = hot.insert(hash, block: hotBlock)
                evictions += demoted

                durable?.saveBlock(
                    hash: hash,
                    cacheData: sliced,
                    layerClasses: layerClasses,
                    metaStates: metaStates,
                    tokenCount: config.blockSize,
                    modelName: modelName
                )

                parentHash = hash
            }
        }
        NovaMLXLog.info("Stored \(numFullBlocks) blocks for \(modelName)")
    }

    public func clear() {
        anchors.removeAll()
        pool.clear()
        durable?.clear()
        hot.clear()
        lock.withLock {
            hits = 0
            misses = 0
            tokensSaved = 0
            evictions = 0
            inflight.removeAll()
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
                ssdBlockCount: durable?.blockCount ?? 0,
                ssdTotalSize: durable?.totalSize ?? 0,
                hotCacheBlockCount: hot.count,
                hotCacheTotalBytes: hot.bytes
            )
        }
    }

    private func fetchMetadataOnly(tokenIds: [Int]) -> PrefixResult {
        let (matchedBlockIds, _) = pool.findSharedPrefix(tokenIds: tokenIds)
        if matchedBlockIds.isEmpty {
            lock.withLock { misses += 1 }
            return PrefixResult(cache: nil, cachedTokenCount: 0, remainingTokenIds: tokenIds)
        }
        let cachedTokenCount = matchedBlockIds.count * config.blockSize
        let remaining = Array(tokenIds.dropFirst(cachedTokenCount))
        lock.withLock {
            hits += 1
            tokensSaved += cachedTokenCount
        }
        return PrefixResult(cache: nil, cachedTokenCount: cachedTokenCount, remainingTokenIds: remaining)
    }

    /// Enqueue durable reads. Returns how many new loads were queued.
    /// `loadBlock` runs on `loadQueue`, never on the caller.
    private func scheduleLoads(_ hashes: [BlockHash]) -> Int {
        guard durable != nil else { return 0 }
        var todo: [BlockHash] = []
        lock.withLock {
            for hash in hashes {
                let key = HotPrefixTier.key(hash)
                if inflight.insert(key).inserted {
                    todo.append(hash)
                }
            }
        }
        guard !todo.isEmpty else { return 0 }
        let instant = clock()
        loadQueue.async { [weak self] in
            guard let self else { return }
            for hash in todo {
                defer {
                    self.lock.withLock { self.inflight.remove(HotPrefixTier.key(hash)) }
                }
                if self.hot.peek(hash) != nil { continue }
                guard let loaded = self.durable?.loadBlock(hash) else { continue }
                guard let block = Self.hotBlock(from: loaded, now: instant) else { continue }
                if self.clock().timeIntervalSince(block.createdAt) > self.config.ttl {
                    continue
                }
                let demoted = self.hot.insert(hash, block: block)
                if demoted > 0 {
                    self.lock.withLock { self.evictions += demoted }
                }
            }
        }
        return todo.count
    }

    static func hotBlock(
        from loaded: (arrays: [String: [MLXArray]], metadata: [String: String], numLayers: Int),
        now: Date
    ) -> HotPrefixBlock? {
        guard loaded.numLayers > 0 else { return nil }
        var sorted: [(Int, [MLXArray])] = []
        for (key, arrays) in loaded.arrays {
            if let layerIdx = Int(key) {
                sorted.append((layerIdx, arrays))
            }
        }
        sorted.sort { $0.0 < $1.0 }
        let layers = sorted.map(\.1)
        guard !layers.isEmpty else { return nil }

        let classes: [String]
        if let typesStr = loaded.metadata["layer_cache_types"],
           let data = typesStr.data(using: .utf8),
           let types = try? JSONDecoder().decode([String].self, from: data) {
            classes = types
        } else {
            classes = Array(repeating: "KVCache", count: loaded.numLayers)
        }
        let meta: [[String]]
        if let statesStr = loaded.metadata["layer_meta_states"],
           let data = statesStr.data(using: .utf8),
           let states = try? JSONDecoder().decode([[String]].self, from: data) {
            meta = states
        } else {
            meta = Array(repeating: [], count: loaded.numLayers)
        }
        let createdAt: Date
        if let raw = loaded.metadata["created_at"], let seconds = Double(raw) {
            createdAt = Date(timeIntervalSince1970: seconds)
        } else {
            createdAt = now
        }
        return HotPrefixBlock(
            layers: layers,
            layerClasses: classes,
            metaStates: meta,
            bytes: HotPrefixTier.estimatedBytes(layers),
            createdAt: createdAt,
            lastAccess: now
        )
    }
}
