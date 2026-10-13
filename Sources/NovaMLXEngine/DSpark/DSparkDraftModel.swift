import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of mlx-dspark `model.py` — DSpark drafter, qwen3-family backbone (the
// vLLM-speculators heads, e.g. RedHatAI/Qwen3.8-27B-speculator.dspark). The head
// ships no embed_tokens and no lm_head: the target's raw table and head are bound
// at load time via `bind(embed:lmHead:)`.

/// One drafted block: `tokens[i]` was predicted at block slot `logitsStart + i`,
/// `confidences[i]` is the sigmoid of the confidence head's logit at that slot.
public struct DSparkDraft: Sendable, Equatable {
    public var tokens: [Int]
    public var confidences: [Float]

    public init(tokens: [Int], confidences: [Float]) {
        self.tokens = tokens
        self.confidences = confidences
    }
}

/// Per-layer cache of the target context's projected K/V (roped K, raw V) for
/// committed tokens only. Concat-based; preallocation is a later optimization.
public final class DSparkCtxCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    /// Committed context rows — also the rope offset of the next append.
    public internal(set) var offset: Int = 0

    public init() {}

    public func append(keys k: MLXArray, values v: MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, k], axis: 2)
            self.values = concatenated([values, v], axis: 2)
        } else {
            self.keys = k
            self.values = v
        }
        offset += k.dim(2)
    }

    /// `[ctx, block]` K/V views for the block forward. `window` keeps only the last
    /// `window` context rows (the drafting-only ctx window, not the trained mask).
    public func withBlock(keys kBlk: MLXArray, values vBlk: MLXArray, window: Int?)
        -> (keys: MLXArray, values: MLXArray)
    {
        guard let keys, let values else { return (kBlk, vBlk) }
        if let window, offset > window {
            let start = offset - window
            return (
                concatenated([keys[0..., 0..., start..., 0...], kBlk], axis: 2),
                concatenated([values[0..., 0..., start..., 0...], vBlk], axis: 2)
            )
        }
        return (
            concatenated([keys, kBlk], axis: 2),
            concatenated([values, vBlk], axis: 2)
        )
    }

    /// Keep only the first `length` context rows — prefix-cache rollback. The retained
    /// K was roped at its absolute position, so it stays valid after the trim.
    public func trim(to length: Int) {
        let n = Swift.max(0, Swift.min(offset, length))
        guard n != offset, let keys, let values else { return }
        self.keys = keys[0..., 0..., ..<n, 0...]
        self.values = values[0..., 0..., ..<n, 0...]
        offset = n
    }
}

/// Additive block attention mask (`0` = attend, `-inf` = hidden), causal within the
/// block and sliding-window over the context. Keys are `[context(0..<ctxLen), block]`
/// at contiguous absolute positions; query i sits at absolute `blockOffset + i`.
public enum DSparkBlockMask {
    public static func make(causal: Bool, window: Int?, blockOffset: Int, ctxLen: Int, qLen: Int)
        -> MLXArray?
    {
        let total = ctxLen + qLen
        if !causal && window.map({ total <= $0 }) ?? true { return nil }
        let qpos = (MLXArray.arange(0, qLen) + blockOffset).expandedDimensions(axis: 1)
        let kpos = MLXArray.arange(0, total).expandedDimensions(axis: 0)
        var allowed: MLXArray =
            causal ? (kpos .<= qpos) : MLXArray.ones([qLen, total], dtype: .bool)
        if let window {
            allowed = allowed .&& (kpos .> (qpos - window))
        }
        return which(allowed, Float(0), -Float.infinity)
    }
}

final class DSparkAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let causalBlock: Bool
    let slidingWindow: Int?
    /// Drafting-only context window (see `DSparkDraftModel.ctxWindow`).
    var ctxWindow: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: any RoPELayer

    init(_ config: DSparkConfig) {
        let h = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.nKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.causalBlock = config.causalBlock
        self.slidingWindow = config.slidingWindow
        _qProj.wrappedValue = Linear(h, nHeads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(h, nKVHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(h, nKVHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(nHeads * headDim, h, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        let ropeDims = Int((Float(config.headDim) * config.partialRotaryFactor).rounded())
        self.rope = initializeRope(
            dims: ropeDims, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: nil)
        super.init()
    }

    /// Project x -> (normed + roped K, raw V). K is roped at absolute position `offset`.
    private func kv(_ x: MLXArray, offset: Int) -> (MLXArray, MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)
        var k = kProj(x).reshaped(B, S, nKVHeads, headDim)
        k = kNorm(k).transposed(0, 2, 1, 3)
        k = applyRotaryPosition(rope, to: k, offset: .scalar(offset))
        let v = vProj(x).reshaped(B, S, nKVHeads, headDim).transposed(0, 2, 1, 3)
        return (k, v)
    }

    func updateCtx(_ fusedNew: MLXArray, offset: Int, cache: DSparkCtxCache) {
        let (k, v) = kv(fusedNew, offset: offset)
        cache.append(keys: k, values: v)
    }

    /// Q from the block rows, K/V from `[committed ctx, block]`. Q and block K rope at
    /// `blockOffset` — the number of cached ctx rows.
    func attend(_ x: MLXArray, blockOffset: Int, cache: DSparkCtxCache, mask: MLXArray?)
        -> MLXArray
    {
        let B = x.dim(0)
        let L = x.dim(1)
        var q = qProj(x).reshaped(B, L, nHeads, headDim)
        q = qNorm(q).transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, offset: .scalar(blockOffset))
        let (kBlk, vBlk) = kv(x, offset: blockOffset)
        let (k, v) = cache.withBlock(keys: kBlk, values: vBlk, window: ctxWindow)
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
            mask.map { .array($0) } ?? .none
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, nHeads * headDim))
    }
}

final class DSparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DSparkConfig) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Llama-style 2-norm decoder layer: input_layernorm -> attention ->
/// post_attention_layernorm -> SwiGLU. No bias, no k_eq_v, no layer_scalar.
final class DSparkDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: DSparkMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: DSparkConfig) {
        _selfAttn.wrappedValue = DSparkAttention(config)
        _mlp.wrappedValue = DSparkMLP(config)
        _inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, blockOffset: Int, cache: DSparkCtxCache, mask: MLXArray?)
        -> MLXArray
    {
        var h = selfAttn.attend(inputLayernorm(x), blockOffset: blockOffset, cache: cache, mask: mask)
        h = x + h
        return h + mlp(postAttentionLayernorm(h))
    }
}

/// Rank-N previous-token correction: `logits += w2(w1[prevToken])`. w1 is indexed by
/// the previous TARGET token; w2 emits draft-space logits (identical spaces on a
/// full-vocab head).
final class DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var w1: Embedding
    @ModuleInfo(key: "markov_w2") var w2: Linear

    init(vocabSize: Int, rank: Int, outVocabSize: Int) {
        _w1.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: rank)
        _w2.wrappedValue = Linear(rank, outVocabSize, bias: false)
        super.init()
    }

    func prevEmbeddings(_ tokenIds: MLXArray) -> MLXArray {
        w1(tokenIds)
    }

    func stepBias(_ tokenIds: MLXArray) -> MLXArray {
        w2(w1(tokenIds))
    }
}

final class DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(inputDim: Int) {
        _proj.wrappedValue = Linear(inputDim, 1)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        proj(features).squeezed(axis: -1)
    }
}

public final class DSparkDraftModel: Module {
    public let config: DSparkConfig
    public var blockSize: Int { config.blockSize }
    public var logitsStart: Int { config.logitsStart }
    /// Most tokens this head can propose from one block (anchor slots excluded).
    public var kdraft: Int { config.kdraft }

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DSparkDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: DSparkMarkovHead?
    @ModuleInfo(key: "confidence_head") var confidenceHead: DSparkConfidenceHead?

    /// Target's raw input embedding and lm_head, bound once at load time — this head
    /// ships neither weight.
    public var embedTokens: ((MLXArray) -> MLXArray)?
    public var lmHead: ((MLXArray) -> MLXArray)?
    /// Draft->target id table for reduced-vocab heads (leading underscore keeps it out
    /// of the parameter tree — it is an index table, not a weight).
    private var _draftToTarget: MLXArray?

    public init(_ config: DSparkConfig) {
        precondition(
            config.family == .qwen3,
            "DSparkDraftModel supports the qwen3 backbone (gemma4 is a later milestone)")
        self.config = config
        _fc.wrappedValue = Linear(
            config.targetLayerIds.count * config.hiddenSize, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            DSparkDecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.markovRank > 0 {
            _markovHead.wrappedValue = DSparkMarkovHead(
                vocabSize: config.vocabSize, rank: config.markovRank,
                outVocabSize: config.outVocabSize)
        }
        if config.enableConfidenceHead {
            let inDim =
                config.hiddenSize + (config.confidenceHeadWithMarkov ? config.markovRank : 0)
            _confidenceHead.wrappedValue = DSparkConfidenceHead(inputDim: inDim)
        }
        super.init()
    }

    public func bind(
        embed: @escaping (MLXArray) -> MLXArray, lmHead: @escaping (MLXArray) -> MLXArray
    ) {
        self.embedTokens = embed
        self.lmHead = lmHead
    }

    /// Strict 1:1 tensor-name load — the qwen3 speculators checkpoint is exactly
    /// fc/hidden_norm/norm, per-layer input_layernorm/post_attention_layernorm/
    /// self_attn.{q,k,v,o}_proj/self_attn.{q,k}_norm/mlp.{gate,up,down}_proj,
    /// markov_head.markov_w{1,2}, confidence_head.proj (62 tensors).
    ///
    /// DeepSpec-native packaging (8B/14B) also ships unused base `embed_tokens` /
    /// `lm_head` tensors; pass them in `allowedUnused` so verify stays strict on
    /// everything else. RedHat/vLLM speculators packaging has no extras.
    public func loadWeights(
        _ weights: [String: MLXArray],
        allowedUnused: Set<String> = ["embed_tokens", "lm_head"]
    ) throws {
        let filtered: [String: MLXArray]
        if allowedUnused.isEmpty {
            filtered = weights
        } else {
            filtered = weights.filter { key, _ in
                !allowedUnused.contains(where: { key == $0 || key.hasPrefix($0 + ".") })
            }
        }
        try update(parameters: ModuleParameters.unflattened(filtered), verify: [.all])
    }

    /// Install a reduced-vocab head's draft->target id mapping. Checkpoints store `d2t`
    /// as offsets (target = draft + d2t[draft]); materialize the absolute table once.
    public func setDraftVocabMap(_ d2tOffsets: MLXArray) {
        _draftToTarget =
            MLXArray.arange(0, d2tOffsets.count, dtype: .int32) + d2tOffsets.asType(.int32)
    }

    /// Map draft-vocabulary ids to target ids (identity on a full-vocab head).
    public func toTargetIds(_ draftIds: MLXArray) -> MLXArray {
        guard let table = _draftToTarget else { return draftIds }
        return take(table, draftIds, axis: 0)
    }

    /// Most recent context rows the block attends (nil = the whole context). A
    /// drafting-only knob for DeepSpec-style heads; heads with a trained causal /
    /// sliding-window block ignore it.
    public var ctxWindow: Int? {
        get { layers.first?.selfAttn.ctxWindow ?? nil }
        set {
            for layer in layers {
                let attn = layer.selfAttn
                attn.ctxWindow = (attn.causalBlock || attn.slidingWindow != nil) ? nil : newValue
            }
        }
    }

    public func makeCtxCache() -> [DSparkCtxCache] {
        layers.map { _ in DSparkCtxCache() }
    }

    public func fuseTarget(_ targetHiddenCat: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHiddenCat))
    }

    /// Append newly committed fused target rows to every layer's ctx cache, roped at
    /// absolute position `offset` (== the cache's current length).
    public func updateContext(
        _ targetHiddenCat: MLXArray, offset: Int, cache: [DSparkCtxCache]
    ) {
        let fused = fuseTarget(targetHiddenCat)
        for (layer, c) in zip(layers, cache) {
            layer.selfAttn.updateCtx(fused, offset: offset, cache: c)
        }
    }

    /// Dense-step context append: no draft forward, but the ctx cache must still grow
    /// or the next draft attends over a hole.
    public func appendContext(_ targetHiddenCat: MLXArray, cache: [DSparkCtxCache]) {
        updateContext(targetHiddenCat, offset: cache.first?.offset ?? 0, cache: cache)
        eval(cache.compactMap(\.keys) + cache.compactMap(\.values))
    }

    /// Block positions the backbone must compute to draft `cap` tokens. A causal block
    /// is invariant to truncation past the last read slot; a bidirectional block needs
    /// the full trained width.
    public func draftWidth(cap: Int) -> Int {
        guard config.causalBlock else { return config.blockSize }
        return Swift.min(config.blockSize, config.logitsStart + Swift.max(1, cap))
    }

    public func backbone(_ blockEmbed: MLXArray, blockOffset: Int, cache: [DSparkCtxCache])
        -> MLXArray
    {
        var h = blockEmbed
        var mask: MLXArray?
        if let attn = layers.first?.selfAttn, attn.causalBlock || attn.slidingWindow != nil {
            mask = DSparkBlockMask.make(
                causal: attn.causalBlock, window: attn.slidingWindow,
                blockOffset: blockOffset, ctxLen: cache.first?.offset ?? 0,
                qLen: h.dim(1)
            )?.asType(h.dtype)
        }
        for (layer, c) in zip(layers, cache) {
            h = layer(h, blockOffset: blockOffset, cache: c, mask: mask)
        }
        return norm(h)
    }

    /// The block positions whose logits are draft predictions: `[logitsStart, +cap)`.
    public func headSlice(_ blockHidden: MLXArray, cap: Int) -> MLXArray {
        let s = config.logitsStart
        return blockHidden[0..., s ..< (s + cap), 0...]
    }

    public func computeLogits(_ hidden: MLXArray) -> MLXArray {
        precondition(
            lmHead != nil,
            "DSparkDraftModel: bind(embed:lmHead:) before drafting — this head ships no lm_head")
        var logits = lmHead!(hidden)
        if let softcap = config.finalLogitSoftcapping {
            logits = tanh(logits / softcap) * softcap
        }
        return logits
    }

    /// Greedy draft walk over `[k, V]` base logits: position i gets the Markov bias of
    /// the token drafted at i-1 (the anchor token for i=0). Returns target ids `[k]`.
    public func sampleBlock(_ baseLogits: MLXArray, firstPrevToken: Int) -> MLXArray {
        guard let markovHead else { return toTargetIds(argMax(baseLogits, axis: -1)) }
        let k = baseLogits.dim(0)
        var tokens: [MLXArray] = []
        tokens.reserveCapacity(k)
        var prev = MLXArray([Int32(firstPrevToken)])
        for i in 0 ..< k {
            let step = baseLogits[i] + markovHead.stepBias(prev)[0]
            let next = toTargetIds(argMax(step, axis: -1, keepDims: true))
            tokens.append(next)
            prev = next
        }
        return concatenated(tokens)
    }

    /// Confidence logits per draft position. Features are `[hidden, markov_w1[prev]]`
    /// when the config joins the Markov embedding.
    public func confidenceLogits(_ blockHidden: MLXArray, prevTokenIds: MLXArray) -> MLXArray? {
        guard let confidenceHead else { return nil }
        let features: MLXArray
        if config.confidenceHeadWithMarkov, let markovHead {
            features = concatenated(
                [blockHidden, markovHead.prevEmbeddings(prevTokenIds)], axis: -1)
        } else {
            features = blockHidden
        }
        return confidenceHead(features)
    }

    /// One speculation round: append `pendingFused` (fused taps of the newly committed
    /// target tokens, `[1, S, nTap*H]`, or nil when the caller already flushed the
    /// context — the DSparkTokenIterator commits fused rows right after each verify)
    /// to the ctx caches, run the block (`[1, blockSize]` token ids: anchor + mask
    /// fills), and greedily draft up to `cap` tokens with per-position confidences.
    @discardableResult
    public func selectBlock(
        block: MLXArray, pendingFused: MLXArray? = nil, cache: [DSparkCtxCache],
        cap: Int, anchorId: Int
    ) -> DSparkDraft {
        let ctxOffset = cache.first?.offset ?? 0
        var blockOffset = ctxOffset
        if let pendingFused, pendingFused.dim(1) > 0 {
            updateContext(pendingFused, offset: ctxOffset, cache: cache)
            blockOffset = ctxOffset + pendingFused.dim(1)
        }

        let effCap = Swift.max(0, Swift.min(cap, kdraft))
        guard effCap > 0, let embedTokens else {
            return DSparkDraft(tokens: [], confidences: [])
        }
        let width = draftWidth(cap: effCap)
        let ids = block.dim(1) > width ? block[0..., 0 ..< width] : block
        let hidden = backbone(embedTokens(ids), blockOffset: blockOffset, cache: cache)
        let sliced = headSlice(hidden, cap: effCap)
        var logits = computeLogits(sliced)
        logits = logits[0]

        let tokens = sampleBlock(logits, firstPrevToken: anchorId)
        eval(tokens)
        let tokenList = tokens.asArray(Int.self)

        var confidences = [Float](repeating: 0, count: tokenList.count)
        if !tokenList.isEmpty {
            var prevIds = [anchorId]
            prevIds.append(contentsOf: tokenList.dropLast())
            if let confLogits = confidenceLogits(sliced[0], prevTokenIds: MLXArray(prevIds)) {
                let conf = sigmoid(confLogits)
                eval(conf)
                confidences = conf.asArray(Float.self)
            }
        }
        return DSparkDraft(tokens: tokenList, confidences: confidences)
    }
}
