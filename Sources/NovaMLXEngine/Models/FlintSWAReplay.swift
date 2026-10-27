import Foundation
import MLX
import MLXFast
import MLXLMCommon
import NovaMLXPrefixCache

// Sliding-window replay for Flint, following rec `cache.py` `FixedSlidingKVCache`
// and `prefix.py` (checkpoint the ring, then replay the suffix).
//
// The live ring stores at most `window` tokens. A multi-token update still
// returns the prior tail plus the new keys so this step's attention is the
// same causal window; only the tail is kept. Once the ring has overwritten a
// token, `trim` is not an exact rewind — the scheduler restores a checkpoint
// and prefills from there. That is the replay tail.
//
// YOCO (You Only Cache Once) is a different compaction: a cross-decoder tail
// reuses KV projected by an earlier self-decoder layer and does not grow its
// own cache. Official Flint does not do this (`num_kv_shared_layers` absent).
// The layout stays identity unless that field is set.

// MARK: - Bounded sliding ring

/// Flint sliding-window KV. Memory is the window, including after the prompt
/// is longer than the window and during later decode steps.
///
/// Conforms to ``KVCache`` directly. ``BaseKVCache`` is not subclassable from
/// this module: its initializer is internal and its stored knobs are not open.
public final class FlintSlidingKVCache: KVCache, BoundedSlidingKVCache {
    public let windowSize: Int
    public var offset: Int = 0
    /// Tokens currently stored. Always `min(offset, windowSize)` after a write.
    public private(set) var storedLength: Int = 0

    private var capacity: Int
    private var ringIndex: Int = 0
    private var slot: Int = 0
    private var keySlots: [MLXArray?] = [nil, nil]
    private var valueSlots: [MLXArray?] = [nil, nil]

    public init(windowSize: Int) {
        precondition(windowSize > 0)
        self.windowSize = windowSize
        self.capacity = windowSize
    }

    public var maxSize: Int? { windowSize }

    public var evictedTokenCount: Int {
        max(0, offset - storedLength)
    }

    public func innerState() -> [MLXArray] {
        (keySlots + valueSlots).compactMap { $0 }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let count = keys.dim(2)
        if count == 1 {
            return updateOne(keys: keys, values: values)
        }
        return updateChunk(keys: keys, values: values)
    }

    /// Single-token decode writes one ring slot and returns every live row.
    private func updateOne(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        ensureBuffers(like: keys, values: values)
        let index: Int
        if storedLength < capacity {
            index = storedLength
            storedLength += 1
        } else {
            index = ringIndex
            ringIndex = (ringIndex + 1) % capacity
        }
        keySlots[slot]![.ellipsis, index ..< (index + 1), 0...] = keys
        valueSlots[slot]![.ellipsis, index ..< (index + 1), 0...] = values
        offset += 1
        let liveK = keySlots[slot]![.ellipsis, ..<storedLength, 0...]
        let liveV = valueSlots[slot]![.ellipsis, ..<storedLength, 0...]
        return (liveK, liveV)
    }

    /// Prefill chunk. Attention sees the previous tail plus the chunk.
    /// The ring keeps only the last `window` rows, left-aligned.
    private func updateChunk(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        ensureBuffers(like: keys, values: values)
        let prior = temporal(slot: slot)
        let attentionK: MLXArray
        let attentionV: MLXArray
        if let prior, storedLength > 0 {
            attentionK = concatenated([prior.0, keys], axis: 2)
            attentionV = concatenated([prior.1, values], axis: 2)
        } else {
            attentionK = keys
            attentionV = values
        }
        let retained = min(capacity, attentionK.dim(2))
        let start = attentionK.dim(2) - retained
        let next = 1 - slot
        let tailK = attentionK[.ellipsis, start..., 0...]
        let tailV = attentionV[.ellipsis, start..., 0...]
        keySlots[next]![.ellipsis, ..<retained, 0...] = tailK
        valueSlots[next]![.ellipsis, ..<retained, 0...] = tailV
        slot = next
        storedLength = retained
        ringIndex = 0
        offset += keys.dim(2)
        return (attentionK, attentionV)
    }

    private func ensureBuffers(like keys: MLXArray, values: MLXArray) {
        guard keySlots[0] == nil else { return }
        let shapeK = [keys.dim(0), keys.dim(1), capacity, keys.dim(3)]
        let shapeV = [values.dim(0), values.dim(1), capacity, values.dim(3)]
        let k0 = MLXArray.zeros(shapeK, dtype: keys.dtype)
        let v0 = MLXArray.zeros(shapeV, dtype: values.dtype)
        keySlots = [k0, MLXArray.zeros(like: k0)]
        valueSlots = [v0, MLXArray.zeros(like: v0)]
    }

    /// Oldest token at index 0. `nil` before the first write.
    private func temporal(slot: Int) -> (MLXArray, MLXArray)? {
        guard let keys = keySlots[slot], let values = valueSlots[slot], storedLength > 0 else {
            return nil
        }
        if storedLength == capacity, ringIndex != 0 {
            return (
                concatenated(
                    [
                        keys[.ellipsis, ringIndex..., 0...],
                        keys[.ellipsis, ..<ringIndex, 0...],
                    ], axis: 2),
                concatenated(
                    [
                        values[.ellipsis, ringIndex..., 0...],
                        values[.ellipsis, ..<ringIndex, 0...],
                    ], axis: 2)
            )
        }
        return (
            keys[.ellipsis, ..<storedLength, 0...],
            values[.ellipsis, ..<storedLength, 0...]
        )
    }

    public var state: [MLXArray] {
        get {
            guard let pair = temporal(slot: slot) else { return [] }
            return [pair.0, pair.1]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("FlintSlidingKVCache state must have exactly 2 arrays")
            }
            let keys = FlintArrayCopy.detach(newValue[0])
            let values = FlintArrayCopy.detach(newValue[1])
            let seq = keys.dim(2)
            capacity = windowSize
            ensureBuffers(like: keys, values: values)
            let retained = min(capacity, seq)
            let start = seq - retained
            keySlots[0]![.ellipsis, ..<retained, 0...] = keys[.ellipsis, start..., 0...]
            valueSlots[0]![.ellipsis, ..<retained, 0...] = values[.ellipsis, start..., 0...]
            keySlots[1] = MLXArray.zeros(like: keySlots[0]!)
            valueSlots[1] = MLXArray.zeros(like: valueSlots[0]!)
            slot = 0
            ringIndex = 0
            storedLength = retained
            if offset < retained { offset = retained }
        }
    }

    public var metaState: [String] {
        get {
            [
                String(windowSize), String(offset), String(storedLength),
                String(ringIndex), String(slot),
            ]
        }
        set {
            guard newValue.count >= 5,
                let offsetVal = Int(newValue[1]),
                let stored = Int(newValue[2]),
                let ring = Int(newValue[3]),
                let slotVal = Int(newValue[4])
            else { return }
            offset = offsetVal
            storedLength = stored
            ringIndex = ring
            slot = slotVal
        }
    }

    /// Exact only while every token is still in the ring (`offset == stored`).
    /// After the window overwrites a row, rewind goes through a checkpoint.
    public var isTrimmable: Bool {
        storedLength > 0 && offset == storedLength
    }

    public var maxTrimCount: Int {
        isTrimmable ? storedLength : 0
    }

    public func isTrimmable(after positions: Int) -> Bool {
        positions >= 0 && isTrimmable && offset + positions <= windowSize
    }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        guard n > 0, isTrimmable else { return 0 }
        let trimmed = min(n, storedLength)
        storedLength -= trimmed
        offset -= trimmed
        ringIndex = 0
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize requested: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let window = requested ?? windowSize
        if n == 1, !returnArray { return .none }
        if storedLength == 0, n <= window, !returnArray { return .causal }
        return .array(createCausalMask(n: n, offset: storedLength, windowSize: window))
    }

    public func copy() -> any KVCache {
        let clone = FlintSlidingKVCache(windowSize: windowSize)
        clone.offset = offset
        clone.storedLength = storedLength
        clone.ringIndex = ringIndex
        clone.slot = slot
        clone.capacity = capacity
        clone.keySlots = keySlots.map { $0.map(FlintArrayCopy.detach) }
        clone.valueSlots = valueSlots.map { $0.map(FlintArrayCopy.detach) }
        return clone
    }
}

enum FlintArrayCopy {
    private static let kernel = MLXFast.metalKernel(
        name: "flint_swa_detach",
        inputNames: ["values", "count"],
        outputNames: ["snapshot"],
        source: """
            uint i = thread_position_in_grid.x;
            if (i < count[0]) snapshot[i] = values[i];
            """,
        header: "",
        ensureRowContiguous: true
    )

    /// Independent storage. MLX views alias Metal buffers; rec copies for the same reason.
    static func detach(_ array: MLXArray) -> MLXArray {
        guard array.size > 0 else { return array }
        let count = MLXArray([UInt32(array.size)])
        let threads = min(256, max(array.size, 1))
        let grid = ((array.size + threads - 1) / threads) * threads
        let out = kernel(
            [array, count],
            grid: (grid, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [array.shape],
            outputDTypes: [array.dtype]
        )
        eval(out)
        return out[0]
    }
}

// MARK: - Windowed attention

/// Replay each query tile against only the sliding tail.
///
/// Matches a single `scaledDotProductAttention` over the same causal window:
/// the score tensor for a tile is `tile × window`, not `sequence × sequence`.
/// `nil` means the mask shape was not one this helper slices, and the caller
/// keeps the stock SDPA.
enum FlintSWAAttention {
    static func replay(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        sinks: MLXArray?,
        window: Int,
        tile: Int = 512
    ) -> MLXArray? {
        let queriesCount = queries.dim(2)
        let keysCount = keys.dim(2)
        guard queriesCount > 1, keysCount > window, window > 0, tile > 0 else { return nil }
        let prior = keysCount - queriesCount
        guard prior >= 0 else { return nil }
        var parts: [MLXArray] = []
        parts.reserveCapacity((queriesCount + tile - 1) / tile)
        var begin = 0
        while begin < queriesCount {
            let end = min(begin + tile, queriesCount)
            let position = prior + begin
            let start = max(0, position - (window - 1))
            let stop = prior + end
            guard let tileMask = sliceMask(
                mask, query: begin ..< end, key: start ..< stop,
                position: position, window: window)
            else { return nil }
            let out = MLXFast.scaledDotProductAttention(
                queries: queries[.ellipsis, begin ..< end, 0...],
                keys: keys[.ellipsis, start ..< stop, 0...],
                values: values[.ellipsis, start ..< stop, 0...],
                scale: scale,
                mask: tileMask,
                sinks: sinks
            )
            parts.append(out)
            begin = end
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }

    private static func sliceMask(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
        query: Range<Int>,
        key: Range<Int>,
        position: Int,
        window: Int
    ) -> MLXFast.ScaledDotProductAttentionMaskMode? {
        switch mask {
        case .array(let array):
            guard let sliced = slice(array, query: query, key: key) else { return nil }
            return .array(sliced)
        case .arrays(let arrays):
            var sliced: [MLXArray] = []
            sliced.reserveCapacity(arrays.count)
            for array in arrays {
                guard let piece = slice(array, query: query, key: key) else { return nil }
                sliced.append(piece)
            }
            return .arrays(sliced)
        case .causal, .none:
            return .array(
                createCausalMask(
                    n: query.count, offset: position - key.lowerBound, windowSize: window))
        default:
            return nil
        }
    }

    private static func slice(_ array: MLXArray, query: Range<Int>, key: Range<Int>) -> MLXArray? {
        guard array.ndim >= 2 else { return nil }
        let queryAxis = array.ndim - 2
        let keyAxis = array.ndim - 1
        guard array.dim(queryAxis) >= query.upperBound, array.dim(keyAxis) >= key.upperBound else {
            return nil
        }
        if array.ndim == 2 {
            return array[query, key]
        }
        return array[.ellipsis, query, key]
    }
}

// MARK: - Replay plan

/// What to do with a Flint prompt snapshot when the next prompt shares a prefix.
enum FlintSWAReplay {
    enum Decision: Equatable, Sendable {
        /// Every layer can drop `drop` newest tokens and stay exact.
        case trim(Int)
        /// Sliding ring comes back from the checkpoint; the suffix is prefilled.
        case replay(Int)
        case full
    }

    static func decide(
        snap: Int, kept: Int, prompt: Int, evicted: Bool, checkpoints: [Int]
    ) -> Decision {
        guard kept > 0, kept <= snap, kept < prompt else { return .full }
        let drop = snap - kept
        if !evicted || drop == 0 { return .trim(drop) }
        if let checkpoint = checkpoints.filter({ $0 <= kept && $0 < snap }).max() {
            return .replay(checkpoint)
        }
        return .full
    }

    static func isSliding(_ cache: KVCache) -> Bool {
        concrete(cache) is FlintSlidingKVCache
    }

    static func slidingHasEvicted(_ caches: [KVCache]) -> Bool {
        caches.contains { cache in
            guard let sliding = concrete(cache) as? FlintSlidingKVCache else { return false }
            return sliding.evictedTokenCount > 0
        }
    }

    static func concrete(_ cache: KVCache) -> KVCache {
        (cache as? FlintRecCacheProxy)?.inner ?? cache
    }

    /// Drop `drop` newest tokens on every layer that is not a sliding ring.
    /// The ring is restored from a checkpoint first; trimming it would delete
    /// the window the suffix still has to attend to.
    static func trimKeepingSliding(_ caches: [KVCache], drop: Int) -> Bool {
        guard drop >= 0 else { return false }
        if drop == 0 { return true }
        for cache in caches {
            if isSliding(cache) { continue }
            if cache.trim(drop) < drop { return false }
        }
        return true
    }
}

/// One sliding-ring checkpoint. Full-attention layers are not copied: they trim.
struct FlintSlidingCheckpoint {
    var position: Int
    var layers: [(index: Int, state: [MLXArray], meta: [String])]

    static func pushed(
        _ existing: [FlintSlidingCheckpoint], position: Int, caches: [KVCache]
    ) -> [FlintSlidingCheckpoint] {
        var next = Array(existing.suffix(1))
        next.append(capture(position: position, caches: caches))
        return next
    }

    static func capture(position: Int, caches: [KVCache]) -> FlintSlidingCheckpoint {
        var layers: [(index: Int, state: [MLXArray], meta: [String])] = []
        for (index, cache) in caches.enumerated() {
            guard let sliding = FlintSWAReplay.concrete(cache) as? FlintSlidingKVCache else {
                continue
            }
            let temporal = sliding.state
            guard temporal.count == 2 else { continue }
            let meta = [
                String(sliding.windowSize), String(sliding.offset),
                String(temporal[0].dim(2)), "0", "0",
            ]
            layers.append((
                index,
                temporal.map(FlintArrayCopy.detach),
                meta
            ))
        }
        return FlintSlidingCheckpoint(position: position, layers: layers)
    }

    func restore(into caches: [KVCache]) {
        for layer in layers {
            guard caches.indices.contains(layer.index) else { continue }
            var cache = caches[layer.index]
            cache.metaState = layer.meta
            cache.state = layer.state.map(FlintArrayCopy.detach)
            if let proxy = caches[layer.index] as? FlintRecCacheProxy {
                proxy.offset = proxy.inner.offset
                proxy.recAhead = 0
            }
        }
    }
}

// MARK: - YOCO tail

/// Cross-decoder tail. `source[i] == i` means the layer owns its KV.
/// A later layer whose source is earlier reuses that KV and stores none.
struct FlintYOCOLayout: Equatable, Sendable {
    var source: [Int]

    var sharesKV: Bool { source.enumerated().contains { $0.element != $0.offset } }

    func isAlias(_ index: Int) -> Bool {
        guard source.indices.contains(index) else { return false }
        return source[index] != index
    }

    /// `sharedTail` is `num_kv_shared_layers`: the last N layers are the
    /// cross-decoder. Each reuses the last earlier layer of the same kind.
    /// A tail layer with no earlier layer of its kind owns its own KV.
    static func make(layerTypes: [String], sharedTail: Int) -> FlintYOCOLayout {
        let count = layerTypes.count
        let tail = min(max(sharedTail, 0), count)
        let boundary = count - tail
        var lastOfType: [String: Int] = [:]
        var source = Array(0 ..< count)
        for index in 0 ..< boundary {
            lastOfType[layerTypes[index]] = index
            source[index] = index
        }
        guard tail > 0, boundary < count else {
            return FlintYOCOLayout(source: source)
        }
        for index in boundary ..< count {
            let kind = layerTypes[index]
            if let prior = lastOfType[kind], prior < index {
                source[index] = prior
            } else {
                source[index] = index
                lastOfType[kind] = index
            }
        }
        return FlintYOCOLayout(source: source)
    }
}

/// Placeholder for a YOCO cross-decoder layer. The source layer holds the KV.
public final class FlintYOCOAliasCache: KVCache {
    public let source: Int
    public var offset: Int = 0

    public init(source: Int) {
        self.source = source
    }

    public var maxSize: Int? { nil }

    public var isTrimmable: Bool { true }

    public var maxTrimCount: Int { Int.max }

    public func innerState() -> [MLXArray] { [] }

    public var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("YOCO alias layer \(source) has no KV of its own")
            }
        }
    }

    public var metaState: [String] {
        get { [String(source), String(offset)] }
        set {
            if newValue.count >= 2, let offsetVal = Int(newValue[1]) {
                offset = offsetVal
            }
        }
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("YOCO alias layer \(source) has no KV of its own")
    }

    @discardableResult
    public func trim(_ n: Int) -> Int { max(0, n) }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, !returnArray { return .none }
        return .causal
    }

    public func copy() -> any KVCache {
        let clone = FlintYOCOAliasCache(source: source)
        clone.offset = offset
        return clone
    }
}
