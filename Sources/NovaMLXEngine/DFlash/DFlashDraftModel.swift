import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NovaMLXCore
import NovaMLXUtils

// Full DFlash2 block drafter. The public selector and grouped-conv types in
// `DFlashPath` stay for the speed-gate tests and the weightless install hook.
// This module tree matches the Hugging Face checkpoint (fc, per-layer conv,
// candidate_selector) so a Qwen3.8 companion is not a chat Qwen3Model.

public final class DFlashCtxCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    public internal(set) var offset: Int = 0

    public init() {}

    /// Append one context block. The setters stay on this type so other
    /// modules in the file can grow the cache without a public setter.
    func append(keys newKeys: MLXArray, values newValues: MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, newKeys], axis: 2)
            self.values = concatenated([values, newValues], axis: 2)
        } else {
            self.keys = newKeys
            self.values = newValues
        }
    }
}

/// Per-layer conv. `base_kernel` is `[2, taps, hidden]` (input side, output side).
final class DFlashLayerConv: Module {
    let taps: Int
    let groupSize: Int
    let numGroups: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, taps: Int, groupSize: Int) {
        let taps = max(taps, 1)
        let group = max(groupSize, 1)
        precondition(hiddenSize % group == 0, "conv group must divide hidden")
        self.taps = taps
        self.groupSize = group
        self.numGroups = hiddenSize / group
        let tap0 = MLXArray.ones([2, 1, hiddenSize])
        let rest = MLXArray.zeros([2, max(taps - 1, 0), hiddenSize])
        _baseKernel.wrappedValue = concatenated([tap0, rest], axis: 1)
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * taps * numGroups, bias: false)
        super.init()
    }

    func prepare(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let coeff = kernelProjection(x).reshaped(x.dim(0), x.dim(1), 2, taps, numGroups)
        let inDelta = coeff[0..., 0..., 0, 0..., 0...]
        let outDelta = coeff[0..., 0..., 1, 0..., 0...]
        return (convolve(x, delta: inDelta, side: 0), outDelta)
    }

    func finish(_ y: MLXArray, _ delta: MLXArray) -> MLXArray {
        convolve(y, delta: delta, side: 1)
    }

    private func convolve(_ x: MLXArray, delta: MLXArray, side: Int) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let hidden = x.dim(2)
        let grouped = x.reshaped(batch, length, numGroups, groupSize)
        let base = baseKernel[side, 0..., 0...].reshaped(1, 1, taps, numGroups, groupSize)
        let coeff = base + delta.expandedDimensions(axis: -1)
        var out = coeff[0..., 0..., 0, 0..., 0...] * grouped
        if taps > 1 {
            for tap in 1 ..< taps {
                let pad = MLXArray.zeros([batch, tap, numGroups, groupSize], dtype: x.dtype)
                let body = grouped[0..., ..<(length - tap), 0..., 0...]
                let shifted = concatenated([pad, body], axis: 1)
                out = out + coeff[0..., 0..., tap, 0..., 0...] * shifted
            }
        }
        return out.reshaped(batch, length, hidden)
    }
}

/// Checkpoint selector. Codebook keys match `candidate_selector.*`.
final class DFlashWeightSelector: Module {
    let topK: Int
    let rank: Int

    @ParameterInfo(key: "predecessor_codebook") var predecessorCodebook: MLXArray
    @ParameterInfo(key: "successor_codebook") var successorCodebook: MLXArray
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(hiddenSize: Int, vocabSize: Int, rank: Int, topK: Int) {
        self.topK = max(topK, 1)
        self.rank = max(rank, 1)
        _predecessorCodebook.wrappedValue = MLXArray.zeros([max(vocabSize, 1), self.rank])
        _successorCodebook.wrappedValue = MLXArray.zeros([max(vocabSize, 1), self.rank])
        _hiddenProjection.wrappedValue = Linear(max(hiddenSize, 1), self.rank, bias: false)
        super.init()
    }

    func lattice(
        candidateIds: MLXArray, unaryLogits: MLXArray, hidden: MLXArray, anchorId: Int
    ) -> MLXArray {
        let steps = candidateIds.dim(0)
        let k = candidateIds.dim(1)
        let projected = hiddenProjection(hidden)
        let anchor = MLXArray(Array(repeating: Int32(anchorId), count: k), [1, k])
        let predIds = concatenated([anchor, candidateIds[0 ..< (steps - 1), 0...]], axis: 0)
        let pred = take(predecessorCodebook, predIds.reshaped(-1), axis: 0).reshaped(steps, k, rank)
            * projected.expandedDimensions(axis: 1)
        let suc = take(successorCodebook, candidateIds.reshaped(-1), axis: 0).reshaped(steps, k, rank)
        let bilinear = matmul(pred, suc.transposed(0, 2, 1))
        let unary = unaryLogits.asType(.float32).expandedDimensions(axis: 1)
        return unary + bilinear.asType(.float32)
    }

    func walkGreedy(scores: MLXArray, candidateIds: MLXArray) -> MLXArray {
        walkGreedyDetail(scores: scores, candidateIds: candidateIds, marginScale: 2).ids
    }

    func walkGreedyDetail(
        scores: MLXArray, candidateIds: MLXArray, marginScale: Float
    ) -> (ids: MLXArray, confidences: MLXArray) {
        let steps = scores.dim(0)
        let row0 = scores[0, 0]
        var index = argMax(row0, axis: -1)
        var picks = take(candidateIds[0], index, axis: 0).reshaped([1])
        var confs: [MLXArray] = [dflashMarginConfidence(row: row0, scale: marginScale)]
        if steps > 1 {
            for step in 1 ..< steps {
                let row = take(scores[step], index, axis: 0)
                confs.append(dflashMarginConfidence(row: row, scale: marginScale))
                index = argMax(row, axis: -1)
                picks = concatenated([picks, take(candidateIds[step], index, axis: 0).reshaped([1])], axis: 0)
            }
        }
        return (picks, MLX.stacked(confs, axis: 0))
    }
}

final class DFlashMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class DFlashBlockAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlashConfig, layerIdx: Int) {
        let dim = max(config.hiddenSize, 1)
        let head = max(config.headDim, 1)
        self.nHeads = max(config.numAttentionHeads, 1)
        self.nKVHeads = max(config.numKeyValueHeads, 1)
        self.scale = pow(Float(head), -0.5)
        self.isSliding = layerIdx < config.layerTypes.count
            && config.layerTypes[layerIdx] == "sliding_attention"
        self.slidingWindow = isSliding ? config.slidingWindow : nil
        _qProj.wrappedValue = Linear(dim, nHeads * head, bias: false)
        _kProj.wrappedValue = Linear(dim, nKVHeads * head, bias: false)
        _vProj.wrappedValue = Linear(dim, nKVHeads * head, bias: false)
        _oProj.wrappedValue = Linear(nHeads * head, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: head, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: head, eps: config.rmsNormEps)
        super.init()
    }

    func appendCtx(_ xCtx: MLXArray, rope: RoPELayer, cache: DFlashCtxCache) -> (MLXArray, MLXArray) {
        var xCtx = xCtx
        var rows = xCtx.dim(1)
        let batch = xCtx.dim(0)
        if isSliding, let window = slidingWindow {
            let keep = window - 1
            if keep < rows {
                let skip = rows - keep
                xCtx = xCtx[0..., skip..., 0...]
                rows = xCtx.dim(1)
                cache.offset += skip
            }
        }
        var ctxKeys = kProj(xCtx)
        var ctxValues = vProj(xCtx)
        ctxKeys = kNorm(ctxKeys.reshaped(batch, rows, nKVHeads, -1)).transposed(0, 2, 1, 3)
        ctxValues = ctxValues.reshaped(batch, rows, nKVHeads, -1).transposed(0, 2, 1, 3)
        ctxKeys = applyRotaryPosition(rope, to: ctxKeys, offset: .scalar(cache.offset))
        cache.append(keys: ctxKeys, values: ctxValues)
        cache.offset += rows
        return (cache.keys!, cache.values!)
    }

    func callAsFunction(_ x: MLXArray, xCtx: MLXArray, rope: RoPELayer, cache: DFlashCtxCache) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let (keys0, values0) = appendCtx(xCtx, rope: rope, cache: cache)
        var queries = qProj(x)
        var propKeys = kProj(x)
        var propValues = vProj(x)
        queries = qNorm(queries.reshaped(batch, length, nHeads, -1)).transposed(0, 2, 1, 3)
        propKeys = kNorm(propKeys.reshaped(batch, length, nKVHeads, -1)).transposed(0, 2, 1, 3)
        propValues = propValues.reshaped(batch, length, nKVHeads, -1).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(cache.offset))
        propKeys = applyRotaryPosition(rope, to: propKeys, offset: .scalar(cache.offset))
        let keys = concatenated([keys0, propKeys], axis: 2)
        let values = concatenated([values0, propValues], axis: 2)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = isSliding ? .causal : .none
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

final class DFlashBlockLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlashBlockAttention
    @ModuleInfo(key: "mlp") var mlp: DFlashMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlashLayerConv?
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlashLayerConv?

    init(_ config: DFlashConfig, layerIdx: Int) {
        let hidden = max(config.hiddenSize, 1)
        _selfAttn.wrappedValue = DFlashBlockAttention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = DFlashMLP(
            dimensions: hidden, hiddenDimensions: max(config.intermediateSize, 1))
        _inputLayernorm.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)
        if config.convKernelSize > 0, hidden % max(config.convGroupSize, 1) == 0 {
            _attentionConv.wrappedValue = DFlashLayerConv(
                hiddenSize: hidden, taps: config.convKernelSize, groupSize: config.convGroupSize)
            _mlpConv.wrappedValue = DFlashLayerConv(
                hiddenSize: hidden, taps: config.convKernelSize, groupSize: config.convGroupSize)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, xCtx: MLXArray, rope: RoPELayer, cache: DFlashCtxCache) -> MLXArray {
        var attentionIn = inputLayernorm(x)
        var attentionOutCoeff: MLXArray?
        if let conv = attentionConv {
            let prepared = conv.prepare(attentionIn)
            attentionIn = prepared.0
            attentionOutCoeff = prepared.1
        }
        var attention = selfAttn(attentionIn, xCtx: xCtx, rope: rope, cache: cache)
        if let conv = attentionConv, let coeff = attentionOutCoeff {
            attention = conv.finish(attention, coeff)
        }
        let hidden = x + attention
        var mlpIn = postAttentionLayernorm(hidden)
        var mlpOutCoeff: MLXArray?
        if let conv = mlpConv {
            let prepared = conv.prepare(mlpIn)
            mlpIn = prepared.0
            mlpOutCoeff = prepared.1
        }
        var mlpOut = mlp(mlpIn)
        if let conv = mlpConv, let coeff = mlpOutCoeff {
            mlpOut = conv.finish(mlpOut, coeff)
        }
        return hidden + mlpOut
    }
}

/// Block-diffusion drafter. Loaded models are cast to this class at generate time.
/// `hasSpeculativeWeights` stays false until a real selector is installed, so the
/// iterator will not claim a boost for an empty module.
public final class DFlashDraftModel: Module, LanguageModel, KVCacheDimensionProvider {
    public let config: DFlashConfig
    public let kvHeads: [Int]
    /// Set only after the path selector and local conv have real parameters.
    public private(set) var pathSelector: DFlashCandidateSelector?
    public private(set) var localConv: DFlashGroupedConv?
    public private(set) var hasSpeculativeWeights = false
    /// True when the checkpoint module tree (layers, fc, selector) loaded.
    public private(set) var hasBlockWeights = false

    let rope: any RoPELayer
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var blockLayers: [DFlashBlockLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var blockSelector: DFlashWeightSelector?

    public var embedTokens: Embedding?
    public var lmHead: ((MLXArray) -> MLXArray)?
    public var embedScale: Float = 1

    public init(_ config: DFlashConfig) {
        self.config = config
        let layers = max(config.numHiddenLayers, 1)
        let hidden = max(config.hiddenSize, 1)
        self.kvHeads = Array(repeating: max(config.numKeyValueHeads, 1), count: layers)
        let taps = max(config.targetLayerIds.count, 1)
        _fc.wrappedValue = Linear(taps * hidden, hidden, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)
        _blockLayers.wrappedValue = (0 ..< layers).map { DFlashBlockLayer(config, layerIdx: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)
        self.rope = initializeRope(
            dims: max(config.headDim, 1),
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        if config.isDFlash2 {
            _blockSelector.wrappedValue = DFlashWeightSelector(
                hiddenSize: hidden,
                vocabSize: max(config.vocabSize, 1),
                rank: max(config.selectorRank, 1),
                topK: max(config.selectorTopK, 1)
            )
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.ndim >= 2 ? inputs.dim(0) : 1
        let sequence = inputs.ndim >= 2 ? inputs.dim(1) : inputs.dim(0)
        return MLXArray.zeros([batch, max(sequence, 1), max(config.vocabSize, 1)])
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func bind(
        embed: Embedding, lmHead: @escaping (MLXArray) -> MLXArray, embedScale: Float = 1
    ) {
        self.embedTokens = embed
        self.lmHead = lmHead
        self.embedScale = embedScale
    }

    /// Test and loader hook. A model without this install must not claim a DFlash2 boost.
    public func installSpeculativePath(selector: DFlashCandidateSelector, conv: DFlashGroupedConv) {
        pathSelector = selector
        localConv = conv
        hasSpeculativeWeights = true
    }

    /// Load the block module tree when the checkpoint matches it. Otherwise copy
    /// the selector and one conv kernel, the same fallback the speed gate already used.
    public func loadSpeculativeWeightsIfPresent(at dir: URL) {
        guard !hasSpeculativeWeights, config.isDFlash2 else { return }
        let weights = Self.readSafetensors(at: dir)
        guard !weights.isEmpty else { return }
        if (try? loadBlockWeights(weights)) != nil {
            hasBlockWeights = true
            hasSpeculativeWeights = true
            NovaMLXLog.info(
                "[DFlash] Loaded block drafter from \(dir.lastPathComponent) (selector, local conv, layers)"
            )
            return
        }
        loadLooseSelectorAndConv(weights, dir: dir)
    }

    public func makeCache() -> [DFlashCtxCache] {
        blockLayers.map { _ in DFlashCtxCache() }
    }

    public func appendContext(_ hidden: MLXArray, cache: [DFlashCtxCache]) {
        let ctx = hiddenNorm(fc(hidden))
        for (layer, slot) in zip(blockLayers, cache) {
            _ = layer.selfAttn.appendCtx(ctx, rope: rope, cache: slot)
        }
    }

    /// One greedy block from the loaded module tree. The caller owns `cache`.
    public func draftBlock(
        anchorId: Int, targetHidden: MLXArray, cache: [DFlashCtxCache], cap: Int,
        marginScale: Float = 2
    ) -> (tokens: [Int], confidences: [Float]) {
        let width = max(config.blockSize, 2)
        let masks = Array(repeating: Int32(config.maskTokenId), count: width - 1)
        let block = MLXArray([Int32(anchorId)] + masks, [1, width])
        let ctx: MLXArray
        if targetHidden.ndim >= 2, targetHidden.dim(1) > 1 {
            ctx = targetHidden[0..., (-1)..., 0...]
        } else {
            ctx = targetHidden
        }
        let proposal = selectBlock(
            block: block, targetHidden: ctx, cache: cache, cap: max(cap, 1),
            anchorId: anchorId, marginScale: marginScale)
        eval(proposal.tokens, proposal.confidences)
        let ids = Array(proposal.tokens.asArray(Int.self).prefix(max(cap, 1)))
        var conf = proposal.confidences.asArray(Float.self)
        if conf.count > ids.count { conf = Array(conf.prefix(ids.count)) }
        if conf.count < ids.count {
            conf.append(contentsOf: [Float](repeating: 0, count: ids.count - conf.count))
        }
        return (ids, conf)
    }

    /// Greedy draft ids from the loose selector. Empty when weights were not installed.
    public func proposeGreedy(
        anchorLogits: MLXArray, hidden: MLXArray, cap: Int, marginScale: Float = 2
    ) -> (ids: [Int], confidences: [Float]) {
        guard let selector = pathSelector, cap > 0 else { return ([], []) }
        let convHidden = convolved(hidden)
        let picks = selector.propose(
            anchorLogits: anchorLogits, hidden: convHidden, cap: cap, marginScale: marginScale)
        eval(picks.ids, picks.confidences)
        let ids = Array(picks.ids.asArray(Int.self).prefix(cap))
        var conf = picks.confidences.asArray(Float.self)
        if conf.count > ids.count { conf = Array(conf.prefix(ids.count)) }
        if conf.count < ids.count {
            conf.append(contentsOf: [Float](repeating: 0, count: ids.count - conf.count))
        }
        return (ids, conf)
    }

    func loadBlockWeights(_ weights: [String: MLXArray]) throws {
        var filtered: [String: MLXArray] = [:]
        filtered.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key == "embed_tokens" || key.hasPrefix("embed_tokens.")
                || key == "lm_head" || key.hasPrefix("lm_head.")
            {
                continue
            }
            let normalized = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
            filtered[normalized] = value
        }
        try update(parameters: ModuleParameters.unflattened(filtered), verify: [.all])
    }

    private func projectCtx(_ targetHidden: MLXArray) -> MLXArray {
        hiddenNorm(fc(targetHidden))
    }

    private func selectBlock(
        block: MLXArray, targetHidden: MLXArray, cache: [DFlashCtxCache],
        cap: Int, anchorId: Int, marginScale: Float
    ) -> (tokens: MLXArray, confidences: MLXArray) {
        let hiddenFull = forwardHidden(block, targetHidden: targetHidden, cache: cache)
        let sequence = hiddenFull.dim(1)
        let takeCount = min(cap, max(sequence, 1))
        let hidden = hiddenFull[0, 0 ..< takeCount, 0...]
        guard let selector = blockSelector, let lmHead else {
            let logits = lmHead?(hidden) ?? hidden
            return marginArgmax(logits, scale: marginScale)
        }
        var logits = lmHead(hidden)
        if logits.ndim == 3 { logits = logits[0] }
        let vocab = min(max(config.vocabSize, 1), logits.dim(-1))
        if logits.dim(-1) > vocab { logits = logits[0..., 0 ..< vocab] }
        let k = min(selector.topK, vocab)
        let order = argSort(logits, axis: -1)
        let start = max(logits.dim(-1) - k, 0)
        let candidateIds = order[0..., start...].asType(.int32)
        let unary = MLX.takeAlong(logits, candidateIds, axis: -1).asType(.float32)
        let scores = selector.lattice(
            candidateIds: candidateIds, unaryLogits: unary, hidden: hidden, anchorId: anchorId)
        return selector.walkGreedyDetail(
            scores: scores, candidateIds: candidateIds, marginScale: marginScale)
    }

    /// Argmax ids plus a top-2 margin confidence on each row of `[steps, vocab]`.
    private func marginArgmax(_ logits: MLXArray, scale: Float) -> (tokens: MLXArray, confidences: MLXArray) {
        let rows = logits.ndim >= 2 ? logits : logits.reshaped(1, -1)
        let steps = rows.dim(0)
        var confs: [MLXArray] = []
        confs.reserveCapacity(steps)
        for step in 0 ..< steps {
            confs.append(dflashMarginConfidence(row: rows[step], scale: scale))
        }
        let conf = confs.isEmpty ? MLXArray([Float(1)]) : MLX.stacked(confs, axis: 0)
        return (argMax(rows, axis: -1), conf)
    }

    private func forwardHidden(
        _ inputs: MLXArray, targetHidden: MLXArray, cache: [DFlashCtxCache]
    ) -> MLXArray {
        guard let embedTokens else {
            return MLXArray.zeros([1, 1, max(config.hiddenSize, 1)])
        }
        var hidden = embedTokens(inputs) * embedScale
        let ctx = projectCtx(targetHidden)
        for (layer, slot) in zip(blockLayers, cache) {
            hidden = layer(hidden, xCtx: ctx, rope: rope, cache: slot)
        }
        if hidden.dim(1) > 1 { hidden = hidden[0..., 1..., 0...] }
        return norm(hidden)
    }

    private func convolved(_ hidden: MLXArray) -> MLXArray {
        guard let conv = localConv, hidden.ndim >= 2 else { return hidden }
        let width = hidden.dim(-1)
        if width == conv.hiddenSize { return conv.prepare(hidden).0 }
        if width > conv.hiddenSize {
            return conv.prepare(hidden[0..., 0..., ..<conv.hiddenSize]).0
        }
        return hidden
    }

    private func loadLooseSelectorAndConv(_ weights: [String: MLXArray], dir: URL) {
        guard let predecessor = Self.firstTensor(weights, containing: "predecessor_codebook"),
              let successor = Self.firstTensor(weights, containing: "successor_codebook")
        else { return }
        let selector = DFlashCandidateSelector(
            hiddenSize: max(config.hiddenSize, 1),
            vocabSize: max(config.vocabSize, predecessor.dim(0)),
            rank: max(config.selectorRank, predecessor.dim(-1)),
            topK: max(config.selectorTopK, 1)
        )
        guard predecessor.shape == selector.predecessorCodebook.shape,
              successor.shape == selector.successorCodebook.shape
        else {
            NovaMLXLog.info(
                "[DFlash] Selector codebook shape does not match \(dir.lastPathComponent) — plain decode"
            )
            return
        }
        selector.predecessorCodebook = predecessor
        selector.successorCodebook = successor
        if let projection = Self.firstTensor(weights, containing: "hidden_projection"),
           projection.shape == selector.hiddenProjection.weight.shape
        {
            selector.hiddenProjection.weight = projection
        }
        let group = max(config.convGroupSize, 1)
        let hidden = max(config.hiddenSize, 1)
        guard hidden % group == 0 else {
            NovaMLXLog.info(
                "[DFlash] hidden \(hidden) is not divisible by conv group \(group) — plain decode"
            )
            return
        }
        let conv = DFlashGroupedConv(
            hiddenSize: hidden, taps: max(config.convKernelSize, 1), groupSize: group)
        var copiedKernel = false
        if let kernel = Self.firstTensor(weights, containing: "kernel_projection"),
           kernel.shape == conv.kernelProjection.weight.shape
        {
            conv.kernelProjection.weight = kernel
            copiedKernel = true
        }
        if let base = Self.firstTensor(weights, containing: "base_kernel"),
           base.shape == conv.baseKernel.shape
        {
            conv.baseKernel = base
            copiedKernel = true
        }
        guard copiedKernel else {
            NovaMLXLog.info(
                "[DFlash] Grouped conv kernel shape does not match \(dir.lastPathComponent) — plain decode"
            )
            return
        }
        installSpeculativePath(selector: selector, conv: conv)
        NovaMLXLog.info(
            "[DFlash] Loaded path selector and local conv from \(dir.lastPathComponent)"
        )
    }

    private static func readSafetensors(at dir: URL) -> [String: MLXArray] {
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else {
            return [:]
        }
        var weights: [String: MLXArray] = [:]
        for file in files where file.hasSuffix(".safetensors") {
            let url = dir.appendingPathComponent(file)
            if let loaded = try? MLX.loadArrays(url: url) {
                weights.merge(loaded) { _, new in new }
            }
        }
        return weights
    }

    private static func firstTensor(_ weights: [String: MLXArray], containing needle: String) -> MLXArray? {
        let needle = needle.lowercased()
        for (key, value) in weights where key.lowercased().contains(needle) {
            return value
        }
        return nil
    }
}
