import Foundation
import MLX
import NovaMLXUtils

// NovaMLX-TIE: WeightTierManager
//
// Owns weight residency across three tiers:
//   Tier 0 (Wired)  — mlock'd shared layers (attn, embed, lm_head, router, shared
//                     experts). Always resident. Pinned before any Tier 2 mmap.
//   Tier 1 (Hot)    — bounded LRU cache of expert weights resident on GPU.
//                     Sized to WiredMemoryTicket reservation minus active KV cache.
//   Tier 2 (SSD)    — per-expert safetensors files (one file per expert per layer,
//                     produced by ExpertShardLayout). readExpert() loads on demand;
//                     madvise() issues kernel prefetch hints.
//
// Phase 2 scope (this file):
//   - bind(): record per-expert file URLs from manifest, no eager FDs
//   - readExpert(layer:expert:): synchronous read via mlx-swift loadArrays,
//     updates SSD metrics (bytes, ops, latency)
//   - madvise(layer:expert:): F_RDADVISE hint on the file FD
//   - Tier 1 cache: bounded LRU keyed by ExpertID, refcount-tracked
//   - acquire/release: refcount API the layer loop will use (Phase 2-final)
//
// Phase 3 will add:
//   - promote/demote pass driven by ExpertHeatMap
//   - mincore() residency introspection
//   - mlock on tier0.safetensors (currently deferred; requires raw mmap plumbing)

/// Residency tier for a weight tensor.
public enum WeightTier: Sendable {
    case wired      // Tier 0 — mlock'd, never evicted
    case hot        // Tier 1 — LRU resident on GPU
    case ssd        // Tier 2 — mmap'd on demand
    case unknown
}

/// Identifies an expert within a specific MoE layer.
public struct ExpertID: Hashable, Sendable {
    public let layer: Int
    public let expert: Int
    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }
}

/// Result of an acquire call. Tier 1 hit returns synchronously; Tier 2 miss
/// may block on page-in.
public enum AcquireResult {
    case tier1Hit           // already resident in hot cache
    case tier2Loaded        // demand-loaded from SSD
    case notPresent         // TIE not active for this model
}

/// Cached Tier 1 entry. Refcounted so we know when LRU can evict.
private final class Tier1Entry: @unchecked Sendable {
    let tensors: [String: MLXArray]
    var refcount: Int = 1
    var lastAccess: Date = Date()
    init(_ tensors: [String: MLXArray]) { self.tensors = tensors }
}

/// Per-model weight residency manager. One instance per loaded model.
public final class WeightTierManager: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var manifest: TierManifest?
    private(set) var modelDir: URL?
    public private(set) var isBound: Bool = false

    /// Tier 1 LRU cache keyed by ExpertID.
    private var tier1: [ExpertID: Tier1Entry] = [:]
    /// Bounded size in bytes. Default 16GB; Phase 3 will wire to WiredMemoryTicket.
    public var tier1BudgetBytes: Int64 = 16 * 1024 * 1024 * 1024
    private var tier1CurrentBytes: Int64 = 0

    /// Cumulative metrics for this manager (also mirrored to MetricsStore).
    private var ssdBytesReadTotal: Int64 = 0
    private var ssdReadOpsTotal: UInt64 = 0

    /// Decoded layer-file tensors. Cap keeps current + next + one spare so
    /// we do not pin the whole model twice (once on Linear, once here).
    private var layerTensorCache: [Int: [String: MLXArray]] = [:]
    private var layerCacheOrder: [Int] = []
    private let layerCacheLimit = 3

    /// Last prefetch we issued, to avoid 3 identical Tasks from SwitchGLU.
    private var lastPrefetchLayer: Int = -1
    private var lastPrefetchExperts: [Int] = []

    /// Layers we must not evict from the expert cache (current + next).
    private var hotLayers: Set<Int> = []

    /// Per-expert file URL cache: (layer, expert) -> URL
    private var expertFileURLs: [ExpertID: URL] = [:]
    /// Per-layer file URL cache: layer -> URL (dense strategy)
    private var layerFileURLs: [Int: URL] = [:]
    /// Set of files we moved into tie-shards/ subdir on bind (for restore on unbind)
    private var movedShardFiles: [URL] = []

    /// Phase 6: per-expert LRU cache. Activated experts stay in memory;
    /// cold experts evicted when budget exceeded. Keyed by (layer, expert).
    /// Value = the per-expert weight tensors loaded from per-expert safetensors.
    private var perExpertCache: [ExpertID: [String: MLXArray]] = [:]
    /// LRU ordering for perExpertCache (oldest first).
    private var perExpertLRU: [ExpertID] = []

    /// Convention: shards live in `modelDir/tie-shards/` so MLX's eager
    /// loadWeights skips them (Phase 4 SSD streaming). The manifest references
    /// files by basename (e.g. "layer.L00.safetensors"); we prefix internally.
    private static let shardsSubdir = "tie-shards"

    public init() {}

    public func bind(modelDir: URL, manifest: TierManifest) async throws {
        // Phase 4: move per-shard files into modelDir/tie-shards/ so the eager
        // loadWeights (patched in mlx-swift-lm/Load.swift) skips them. The
        // tier0.safetensors stays in modelDir/ and IS loaded eagerly.
        let shardsDir = modelDir.appendingPathComponent(Self.shardsSubdir)
        try FileManager.default.createDirectory(at: shardsDir, withIntermediateDirectories: true)

        var movedFiles: [URL] = []
        var newExpertURLs: [ExpertID: URL] = [:]
        var newLayerURLs: [Int: URL] = [:]

        // Move expert files
        for e in manifest.experts {
            let src = modelDir.appendingPathComponent(e.file)
            let dst = shardsDir.appendingPathComponent(e.file)
            if FileManager.default.fileExists(atPath: src.path) {
                if FileManager.default.fileExists(atPath: dst.path) {
                    // Already moved (idempotent bind) — leave in place
                } else {
                    try FileManager.default.moveItem(at: src, to: dst)
                }
            }
            // Always track so unbind can move back regardless of idempotent state.
            if FileManager.default.fileExists(atPath: dst.path) {
                movedFiles.append(dst)
            }
            newExpertURLs[ExpertID(layer: e.layer, expert: e.expert)] = dst
        }
        // Move layer files (dense)
        for l in manifest.layers ?? [] {
            let src = modelDir.appendingPathComponent(l.file)
            let dst = shardsDir.appendingPathComponent(l.file)
            if FileManager.default.fileExists(atPath: src.path) {
                if FileManager.default.fileExists(atPath: dst.path) {
                    // already moved
                } else {
                    try FileManager.default.moveItem(at: src, to: dst)
                }
            }
            if FileManager.default.fileExists(atPath: dst.path) {
                movedFiles.append(dst)
            }
            newLayerURLs[l.layer] = dst
        }

        // NSLock is not async-safe; wrap the mutation in a sync helper.
        Self.syncLock(self.lock) {
            self.modelDir = modelDir
            self.manifest = manifest
            self.isBound = true
            self.expertFileURLs = newExpertURLs
            self.layerFileURLs = newLayerURLs
            self.movedShardFiles = movedFiles
        }
        NovaMLXLog.info("[TIE] bound \(modelDir.lastPathComponent): layout=\(manifest.layout), tier0=\(manifest.tier0Bytes / 1_048_576)MB, experts=\(manifest.expertCount), layers=\(manifest.layers?.count ?? 0), tier1Budget=\(tier1BudgetBytes / 1_048_576)MB")
    }

    public func unbind() {
        lock.lock(); defer { lock.unlock() }
        // Restore shard files from tie-shards/ back to modelDir
        if let modelDir = modelDir {
            for dst in movedShardFiles {
                let basename = dst.lastPathComponent
                let src = modelDir.appendingPathComponent(basename)
                if FileManager.default.fileExists(atPath: dst.path),
                   !FileManager.default.fileExists(atPath: src.path) {
                    try? FileManager.default.moveItem(at: dst, to: src)
                }
            }
            // Remove now-empty tie-shards/ dir if it exists
            let shardsDir = modelDir.appendingPathComponent(Self.shardsSubdir)
            try? FileManager.default.removeItem(at: shardsDir)
        }
        // Release all Tier 1 entries. MLXArray deinit frees underlying buffers.
        tier1.removeAll()
        tier1CurrentBytes = 0
        expertFileURLs.removeAll()
        layerFileURLs.removeAll()
        movedShardFiles.removeAll()
        layerTensorCache.removeAll()
        layerCacheOrder.removeAll()
        perExpertCache.removeAll()
        perExpertLRU.removeAll()
        lastPrefetchLayer = -1
        lastPrefetchExperts = []
        hotLayers.removeAll()
        manifest = nil
        modelDir = nil
        isBound = false
        ssdBytesReadTotal = 0
        ssdReadOpsTotal = 0
    }

    // MARK: - Tier 2 (SSD) reads

    /// Synchronously load an expert's tensors from its file. Returns the dict
    /// of tensors keyed by original safetensors name.
    ///
    /// Called by `acquire` on Tier 1 miss, and by the prefetcher when warming.
    public func readExpert(layer: Int, expert: Int) throws -> [String: MLXArray] {
        lock.lock()
        guard isBound else {
            lock.unlock()
            throw TierError.notBound
        }
        guard let url = expertFileURLs[ExpertID(layer: layer, expert: expert)] else {
            lock.unlock()
            throw TierError.unknownExpert(layer: layer, expert: expert)
        }
        lock.unlock()
        return try readSafetensorsFile(url: url, label: "expert L\(layer)E\(expert)")
    }

    /// Synchronously load a full decoder layer's tensors (dense strategy).
    /// Used by Linear-level lazy loading when strategy=.layer.
    public func readLayer(layer: Int) throws -> [String: MLXArray] {
        guard isBound else { throw TierError.notBound }
        lock.lock()
        if let cached = layerTensorCache[layer] {
            touchLayerCacheLocked(layer)
            lock.unlock()
            return cached
        }
        lock.unlock()

        guard let url = layerFileURLs[layer] else {
            throw TierError.unknownLayer(layer: layer)
        }
        let tensors = try readSafetensorsFile(url: url, label: "layer L\(layer)")
        lock.lock()
        insertLayerCacheLocked(layer, tensors)
        lock.unlock()
        return tensors
    }

    /// SSD read count. Tests assert cache hits against this.
    public var ssdReadOps: UInt64 {
        lock.lock(); defer { lock.unlock() }
        return ssdReadOpsTotal
    }

    /// Highest layer index present in the manifest, or -1 if unbound.
    public var maxLayerIndex: Int {
        lock.lock(); defer { lock.unlock() }
        guard let m = manifest else { return -1 }
        let eMax = m.experts.map(\.layer).max() ?? -1
        let lMax = m.layers?.map(\.layer).max() ?? -1
        return max(eMax, lMax)
    }

    /// Mark layers that must stay resident (current forward + prefetch target).
    public func markHotLayers(_ layers: Set<Int>) {
        lock.lock(); defer { lock.unlock() }
        hotLayers = layers
    }

    /// Synchronously load all per-expert files for a layer and stack each
    /// tensor name along axis 0, producing the `[numExperts, out, in]` shape
    /// that SwitchLinear expects. Used by SwitchLinear sync hook for MoE
    /// strategy when SwitchLinear weights were skipped at load time.
    ///
    /// For a layer with 256 experts × 3 projs, this reads 256 files and stacks
    /// 3 tensors. Output dict has same keys as a single per-expert file but
    /// values are stacked along axis 0.
    public func readLayerStacked(layer: Int) throws -> [String: MLXArray] {
        guard isBound else { throw TierError.notBound }
        guard let manifest = manifest else { throw TierError.notBound }

        // Find all expert entries for this layer, grouped by tensor name.
        let layerEntries = manifest.experts.filter { $0.layer == layer }
        if layerEntries.isEmpty {
            throw TierError.unknownLayer(layer: layer)
        }

        // For each tensor name (e.g. "model.layers.0.mlp.switch_mlp.gate_proj.weight"),
        // collect the per-expert slice from each file in expert order.
        var slicesByName: [String: [(expert: Int, array: MLXArray)]] = [:]
        for entry in layerEntries {
            guard let url = expertFileURLs[ExpertID(layer: layer, expert: entry.expert)] else { continue }
            let tensors = try readSafetensorsFile(url: url, label: "expert L\(layer)E\(entry.expert)")
            for (name, arr) in tensors {
                slicesByName[name, default: []].append((entry.expert, arr))
            }
        }

        // Stack each tensor name's slices along axis 0, ordered by expert index.
        var result: [String: MLXArray] = [:]
        for (name, slices) in slicesByName {
            let ordered = slices.sorted { $0.expert < $1.expert }.map { $0.array }
            result[name] = MLX.stacked(ordered, axis: 0)
        }
        return result
    }

    /// Shared implementation for readExpert + readLayer.
    private func readSafetensorsFile(url: URL, label: String) throws -> [String: MLXArray] {
        let started = DispatchTime.now()
        let tensors = try MLX.loadArrays(url: url)
        let elapsedMs = Int(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds) / 1_000_000

        var bytes: Int64 = 0
        for arr in tensors.values {
            bytes &+= Int64(arr.size) * Int64(arr.dtype.size)
        }
        lock.lock()
        ssdBytesReadTotal &+= bytes
        ssdReadOpsTotal &+= 1
        lock.unlock()

        NovaMLXLog.debug("[TIE] readSafetensors \(label): \(bytes / 1024)KB in \(elapsedMs)ms")
        return tensors
    }

    private func touchLayerCacheLocked(_ layer: Int) {
        layerCacheOrder.removeAll { $0 == layer }
        layerCacheOrder.append(layer)
    }

    private func insertLayerCacheLocked(_ layer: Int, _ tensors: [String: MLXArray]) {
        layerTensorCache[layer] = tensors
        touchLayerCacheLocked(layer)
        while layerCacheOrder.count > layerCacheLimit {
            let oldest = layerCacheOrder.removeFirst()
            if oldest != layer {
                layerTensorCache.removeValue(forKey: oldest)
            }
        }
    }

    /// Number of experts currently in the per-expert LRU cache. Test/diagnostic.
    public func perExpertCacheCount() -> Int {
        lock.lock(); defer { lock.unlock() }
        return perExpertCache.count
    }

    /// Phase 6: load only the activated experts for a SwitchLinear call.
    /// Returns a dict {expertID: tensors} where tensors are the per-expert
    /// weight dict from the per-expert safetensors file. Hot experts come
    /// from perExpertCache (Tier 1); cold experts trigger SSD read.
    ///
    /// Caller (TierAwareSwitchLinear) stacks the weights into [k, out, in]
    /// and assigns to self.weight, remapping indices to local.
    public func loadActivatedExperts(layer: Int, expertIDs: [Int]) throws -> [Int: [String: MLXArray]] {
        var result: [Int: [String: MLXArray]] = [:]
        var misses: [Int] = []

        lock.lock()
        guard isBound else {
            lock.unlock()
            throw TierError.notBound
        }
        for e in expertIDs {
            let id = ExpertID(layer: layer, expert: e)
            if let cached = perExpertCache[id] {
                perExpertLRU.removeAll { $0 == id }
                perExpertLRU.append(id)
                result[e] = cached
            } else {
                misses.append(e)
            }
        }
        lock.unlock()

        if !misses.isEmpty {
            let loaded = loadExpertsParallel(layer: layer, experts: misses)
            lock.lock()
            for e in misses {
                guard let tensors = loaded[e] else { continue }
                let id = ExpertID(layer: layer, expert: e)
                perExpertCache[id] = tensors
                perExpertLRU.removeAll { $0 == id }
                perExpertLRU.append(id)
                result[e] = tensors
            }
            evictIfNeeded()
            let cacheCount = perExpertCache.count
            lock.unlock()
            NovaMLXLog.debug("[TIE-PE] layer \(layer): \(loaded.count)/\(expertIDs.count) experts from SSD, cache=\(cacheCount)")
        }
        return result
    }

    /// Parallel SSD reads for a miss set. `MLX.loadArrays` is the bottleneck
    /// on first-token / prefill; overlapping file decodes cuts that stall.
    private func loadExpertsParallel(layer: Int, experts: [Int]) -> [Int: [String: MLXArray]] {
        guard !experts.isEmpty else { return [:] }
        if experts.count == 1 {
            if let tensors = try? readExpert(layer: layer, expert: experts[0]) {
                return [experts[0]: tensors]
            }
            return [:]
        }
        let bag = ParallelLoadBag()
        DispatchQueue.concurrentPerform(iterations: experts.count) { i in
            let e = experts[i]
            guard let tensors = try? self.readExpert(layer: layer, expert: e) else { return }
            bag.set(e, tensors)
        }
        return bag.take()
    }

    /// Fire-and-forget: decode next-layer experts into the cache while the
    /// current layer GEMM runs. No-op if this exact set is already in flight.
    public func prefetchExperts(layer: Int, expertIDs: [Int]) {
        guard layer >= 0, !expertIDs.isEmpty else { return }
        lock.lock()
        guard isBound else {
            lock.unlock()
            return
        }
        if lastPrefetchLayer == layer && lastPrefetchExperts == expertIDs {
            lock.unlock()
            return
        }
        lastPrefetchLayer = layer
        lastPrefetchExperts = expertIDs
        lock.unlock()

        Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            _ = try? self.loadActivatedExperts(layer: layer, expertIDs: expertIDs)
        }
    }

    /// Fire-and-forget: decode the next dense layer file into the layer cache.
    public func prefetchLayer(_ layer: Int) {
        guard isBound, layer >= 0 else { return }
        lock.lock()
        if layerTensorCache[layer] != nil || lastPrefetchLayer == layer {
            lock.unlock()
            return
        }
        lastPrefetchLayer = layer
        lastPrefetchExperts = []
        lock.unlock()

        Task.detached(priority: .utility) { [weak self] in
            _ = try? self?.readLayer(layer: layer)
        }
    }

    /// Evict oldest per-expert entries when cache exceeds the Tier 1 budget.
    /// Prefers evicting layers that are not the current / next working set.
    /// Caller must hold `lock`.
    private func evictIfNeeded() {
        let maxBytes = max(tier1BudgetBytes, 512 * 1024 * 1024)
        var currentBytes: Int64 = 0
        for tensors in perExpertCache.values {
            for arr in tensors.values {
                currentBytes &+= Int64(arr.size * arr.dtype.size)
            }
        }
        func evict(fromColdOnly: Bool) {
            var i = 0
            while currentBytes > maxBytes && i < perExpertLRU.count {
                let oldest = perExpertLRU[i]
                if fromColdOnly && hotLayers.contains(oldest.layer) {
                    i += 1
                    continue
                }
                perExpertLRU.remove(at: i)
                if let tensors = perExpertCache.removeValue(forKey: oldest) {
                    for arr in tensors.values {
                        currentBytes &-= Int64(arr.size * arr.dtype.size)
                    }
                }
            }
        }
        evict(fromColdOnly: true)
        if currentBytes > maxBytes {
            evict(fromColdOnly: false)
        }
    }

    /// Kept for callers that still issue kernel hints. Prefer `prefetchExperts`.
    public func madvise(layer: Int, expert: Int) {
        prefetchExperts(layer: layer, expertIDs: [expert])
    }

    // MARK: - Tier 1 (LRU) acquire/release

    /// Acquire an expert's weights. Returns the loaded tensors and a result
    /// indicating which tier served the request. Increments refcount.
    public func acquire(layer: Int, expert: Int) -> (result: AcquireResult, tensors: [String: MLXArray]?) {
        guard isBound else { return (.notPresent, nil) }
        let id = ExpertID(layer: layer, expert: expert)

        lock.lock()
        if let entry = tier1[id] {
            entry.refcount += 1
            entry.lastAccess = Date()
            lock.unlock()
            return (.tier1Hit, entry.tensors)
        }
        lock.unlock()

        // Tier 1 miss — read from SSD
        do {
            let tensors = try readExpert(layer: layer, expert: expert)
            var bytes: Int64 = 0
            for arr in tensors.values {
                bytes &+= Int64(arr.size) * Int64(arr.dtype.size)
            }

            lock.lock()
            // Make room in Tier 1 if needed (LRU eviction)
            evictToFit(requiredBytes: bytes)
            let entry = Tier1Entry(tensors)
            tier1[id] = entry
            tier1CurrentBytes &+= bytes
            lock.unlock()

            return (.tier2Loaded, tensors)
        } catch {
            NovaMLXLog.warning("[TIE] readExpert failed: \(error.localizedDescription)")
            return (.notPresent, nil)
        }
    }

    /// Drop a reference. When refcount hits zero, the entry is eligible for
    /// LRU eviction (but not immediately freed).
    public func release(layer: Int, expert: Int) {
        guard isBound else { return }
        let id = ExpertID(layer: layer, expert: expert)
        lock.lock(); defer { lock.unlock() }
        if let entry = tier1[id] {
            entry.refcount = max(0, entry.refcount - 1)
        }
    }

    public func tier(for layer: Int, expert: Int) -> WeightTier {
        guard isBound else { return .unknown }
        let id = ExpertID(layer: layer, expert: expert)
        lock.lock(); defer { lock.unlock() }
        if tier1[id] != nil { return .hot }
        if expertFileURLs[id] != nil { return .ssd }
        return .unknown
    }

    public func tierSizes() -> (tier0: Int64, tier1: Int64, tier2: Int64) {
        guard let m = manifest else { return (0, 0, 0) }
        let t1: Int64
        lock.lock(); t1 = tier1CurrentBytes; lock.unlock()
        return (m.tier0Bytes, t1, m.totalExpertBytes)
    }

    /// Force eviction of everything from Tier 1. Called on model unload.
    public func flushTier1() {
        lock.lock(); defer { lock.unlock() }
        tier1.removeAll()
        tier1CurrentBytes = 0
    }

    // MARK: - Internals

    /// LRU eviction to fit a new entry. Must be called under `lock`.
    private func evictToFit(requiredBytes: Int64) {
        guard requiredBytes > 0 else { return }
        // Evict zero-refcount entries oldest-first until we fit
        while tier1CurrentBytes + requiredBytes > tier1BudgetBytes {
            // Find oldest with refcount == 0
            let candidates = tier1
                .filter { $0.value.refcount == 0 }
                .sorted { $0.value.lastAccess < $1.value.lastAccess }
            guard let oldest = candidates.first else {
                NovaMLXLog.warning("[TIE] Tier 1 full and no evictable entries (all in use). Skipping eviction; tier1=\(tier1CurrentBytes / 1_048_576)MB")
                return
            }
            var bytes: Int64 = 0
            for arr in oldest.value.tensors.values {
                bytes &+= Int64(arr.size) * Int64(arr.dtype.size)
            }
            tier1.removeValue(forKey: oldest.key)
            tier1CurrentBytes -= bytes
        }
    }

    private static func syncLock(_ lock: NSLock, _ body: () -> Void) {
        lock.lock()
        defer { lock.unlock() }
        body()
    }
}

/// Thread-safe bag for parallel SSD loads. Isolates the non-Sendable
/// `[Int: [String: MLXArray]]` mutation from the concurrentPerform closure.
private final class ParallelLoadBag: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Int: [String: MLXArray]] = [:]

    func set(_ expert: Int, _ tensors: [String: MLXArray]) {
        lock.lock()
        storage[expert] = tensors
        lock.unlock()
    }

    func take() -> [Int: [String: MLXArray]] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

public enum TierError: Error, Sendable {
    case notBound
    case unknownExpert(layer: Int, expert: Int)
    case unknownLayer(layer: Int)
    case readFailed(String)
}
