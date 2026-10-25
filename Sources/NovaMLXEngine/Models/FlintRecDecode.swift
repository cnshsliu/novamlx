import MLX
import MLXFast
import MLXLMCommon
import NovaMLXPrefixCache

// MARK: - Prefill KV → rec ring

/// Reads MLX caches the way rec's `QATSession.prefill` does: full-attention
/// layers are already temporal, sliding layers are unwound so index 0 is the
/// oldest token in the ring. The decode kernel then owns those bytes.
enum FlintRecCacheBridge {
    static func concrete(_ cache: KVCache) -> KVCache {
        (cache as? FlintRecCacheProxy)?.inner ?? cache
    }

    static func orderedPair(from cache: KVCache) -> (MLXArray, MLXArray)? {
        let cache = concrete(cache)
        let stored = cache.state
        guard stored.count >= 2 else { return nil }
        guard let rotating = cache as? RotatingKVCache else {
            return (stored[0], stored[1])
        }
        let meta = rotating.metaState
        guard meta.count >= 5, let keep = Int(meta[0]), let idx = Int(meta[4]) else {
            return (stored[0], stored[1])
        }
        let keys = stored[0]
        let values = stored[1]
        // Once the ring is live, `state` is the raw buffer and `idx` is the
        // next write (oldest). A linear prefill leaves idx at the buffer end.
        // Newer caches record that in meta[6]; older ones only have five fields
        // and `state` stays a temporal prefix until offset reaches the buffer.
        let wrapped: Bool
        if meta.count >= 7, let flag = Bool(meta[6]) {
            wrapped = flag && idx != keys.dim(2)
        } else {
            wrapped = rotating.offset >= keys.dim(2) && idx < rotating.offset && idx != keys.dim(2)
        }
        guard wrapped else { return (keys, values) }
        return (unwind(keys, keep: keep, idx: idx), unwind(values, keep: keep, idx: idx))
    }

    private static func unwind(_ array: MLXArray, keep: Int, idx: Int) -> MLXArray {
        var parts: [MLXArray] = []
        if keep > 0 {
            parts.append(array[.ellipsis, ..<keep, 0...])
        }
        parts.append(array[.ellipsis, idx..., 0...])
        if idx > keep {
            parts.append(array[.ellipsis, keep ..< idx, 0...])
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }
}

// MARK: - Decode session

/// One generation's rec KV. `mlxOffset` is the eager-prefill length; the
/// kernel position runs ahead of that by `recAhead` tokens.
final class FlintRecDecodeSession {
    let cacheID: ObjectIdentifier
    let state: FlintRecKernel.KVState
    let mlxOffset: Int
    var recAhead: Int = 0
    let proxies: [FlintRecCacheProxy]

    init(
        cacheID: ObjectIdentifier, state: FlintRecKernel.KVState, mlxOffset: Int,
        proxies: [FlintRecCacheProxy]
    ) {
        self.cacheID = cacheID
        self.state = state
        self.mlxOffset = mlxOffset
        self.proxies = proxies
    }
}

// MARK: - Cache proxy

/// Forwards eager prefill into a normal MLX cache, and counts tokens the rec
/// kernel has decoded past that snapshot so n-gram verify can `trim` them.
///
/// Trim rewinds the kernel position. Tokens the kernel already overwrote in a
/// full sliding ring cannot be restored — the same limitation
/// `RotatingKVCache.trim` has once `offset >= maxSize`.
final class FlintRecCacheProxy: KVCache, PrefixCacheLayerUnwrapping {
    let inner: any KVCache
    let layer: Int
    unowned let model: FlintModel

    public var prefixCacheLayer: any KVCache { inner }

    public var offset: Int = 0
    var recAhead: Int = 0
    /// This cache has already run one eager single-token forward.
    var sawSingleToken = false

    init(inner: any KVCache, layer: Int, model: FlintModel) {
        self.inner = inner
        self.layer = layer
        self.model = model
        self.offset = inner.offset
    }

    public var maxSize: Int? { inner.maxSize }

    public func innerState() -> [MLXArray] { inner.innerState() }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result = inner.update(keys: keys, values: values)
        recAhead = 0
        offset = inner.offset
        model.invalidateRecSession()
        return result
    }

    public var state: [MLXArray] {
        get { inner.state }
        set {
            inner.state = newValue
            recAhead = 0
            offset = inner.offset
        }
    }

    public var metaState: [String] {
        get { inner.metaState }
        set { inner.metaState = newValue }
    }

    public var isTrimmable: Bool { recAhead > 0 || inner.isTrimmable }

    @discardableResult
    public func trim(_ n: Int) -> Int {
        if recAhead > 0 {
            let fromRec = min(n, recAhead)
            recAhead -= fromRec
            offset = inner.offset + recAhead
            if layer == 0, fromRec > 0 {
                model.rewindRecSession(by: fromRec)
            }
            if fromRec == n { return fromRec }
            // Trim reached past the rec suffix into eager prefill. The kernel
            // snapshot no longer matches; each layer trims its own counter.
            let rest = inner.trim(n - fromRec)
            recAhead = 0
            offset = inner.offset
            if layer == 0 {
                model.abandonRecSession()
            }
            return fromRec + rest
        }
        let trimmed = inner.trim(n)
        offset = inner.offset
        if layer == 0 {
            model.abandonRecSession()
        }
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        inner.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        let copied = FlintRecCacheProxy(inner: inner.copy(), layer: layer, model: model)
        copied.offset = inner.offset
        copied.recAhead = 0
        copied.sawSingleToken = sawSingleToken
        return copied
    }

    func noteRecStep() {
        recAhead += 1
        offset = inner.offset + recAhead
    }

    func undoRecSteps(_ n: Int) {
        recAhead = max(0, recAhead - n)
        offset = inner.offset + recAhead
    }

    func resetRecAhead() {
        recAhead = 0
        offset = inner.offset
    }
}
