import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import NovaMLXUtils

// Port of DeepSeek-V4 architecture based on mlx-lm PR #1201 (akashgoswami)
// Pure MLX operations, no custom Metal kernels.
// Compressor is ephemeral (recomputed per forward), Indexer is a load-only stub.

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    var modelType: String = "deepseek_v4"
    var vocabSize: Int = 129280
    var hiddenSize: Int = 4096
    var numHiddenLayers: Int = 43
    var numHashLayers: Int = 0
    var numNextnPredictLayers: Int = 1
    var numAttentionHeads: Int = 64
    var numKeyValueHeads: Int = 1
    var qLoraRank: Int = 1024
    var oLoraRank: Int = 1024
    var headDim: Int = 512
    var qkRopeHeadDim: Int = 64
    var oGroups: Int = 8
    var indexNHeads: Int = 64
    var indexHeadDim: Int = 128
    var indexTopk: Int = 512
    var nRoutedExperts: Int = 256
    var nSharedExperts: Int = 1
    var numExpertsPerTok: Int = 6
    var moeIntermediateSize: Int = 2048
    var scoringFunc: String = "sqrtsoftplus"
    var routedScalingFactor: Float = 1.5
    var swigluLimit: Float = 10.0
    var normTopkProb: Bool = true
    var slidingWindow: Int = 128
    var compressRatios: [Int] = []
    var kvSourceLayerIds: [Int] = []
    var indexSourceLayerIds: [Int] = []
    var compressRopeTheta: Float = 160000.0
    var hcMult: Int = 4
    var hcSinkhornIters: Int = 20
    var hcEps: Float = 1e-6
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 10000.0
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int = 1048576
    var attentionBias: Bool = false
    var tieWordEmbeddings: Bool = false
    var dsparkBlockSize: Int = 0
    var dsparkNRoutedExperts: Int = 0
    var dsparkNumExpertsPerTok: Int = 0
    /// When set, SwitchLinear only allocates this many expert slots (oMLX SSD offload).
    var expertResidentCapacity: Int? = nil
    var engramLayerIds: [Int] = []
    var engramMaxNgram: Int = 4
    var engramNHeads: Int = 8
    var engramHeadDim: Int = 256

    /// MTP / DSpark layers use fewer routed experts than the backbone.
    func mtpLayerConfig() -> DeepseekV4Configuration {
        var copy = self
        if dsparkNRoutedExperts > 0 { copy.nRoutedExperts = dsparkNRoutedExperts }
        if dsparkNumExpertsPerTok > 0 { copy.numExpertsPerTok = dsparkNumExpertsPerTok }
        copy.numHashLayers = 0
        return copy
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numHashLayers = "num_hash_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case qLoraRank = "q_lora_rank"
        case oLoraRank = "o_lora_rank"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case oGroups = "o_groups"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluLimit = "swiglu_limit"
        case normTopkProb = "norm_topk_prob"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case kvSourceLayerIds = "kv_source_layer_ids"
        case indexSourceLayerIds = "index_source_layer_ids"
        case compressRopeTheta = "compress_rope_theta"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case dsparkBlockSize = "dspark_block_size"
        case dsparkNRoutedExperts = "dspark_n_routed_experts"
        case dsparkNumExpertsPerTok = "dspark_num_experts_per_tok"
        case expertResidentCapacity = "expert_resident_capacity"
        case engramLayerIds = "engram_layer_ids"
        case engramMaxNgram = "engram_max_ngram_size"
        case engramNHeads = "engram_n_heads"
        case engramHeadDim = "engram_head_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v4"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129280
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 43
        numHashLayers = try c.decodeIfPresent(Int.self, forKey: .numHashLayers) ?? 0
        numNextnPredictLayers = try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 64
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1024
        oLoraRank = try c.decodeIfPresent(Int.self, forKey: .oLoraRank) ?? 1024
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 512
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        oGroups = try c.decodeIfPresent(Int.self, forKey: .oGroups) ?? 8
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 64
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 512
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 256
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sqrtsoftplus"
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.5
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10.0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        compressRatios = try c.decodeIfPresent([Int].self, forKey: .compressRatios) ?? []
        kvSourceLayerIds = try c.decodeIfPresent([Int].self, forKey: .kvSourceLayerIds) ?? []
        indexSourceLayerIds = try c.decodeIfPresent([Int].self, forKey: .indexSourceLayerIds) ?? []
        compressRopeTheta = try c.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160000.0
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1048576
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        dsparkBlockSize = try c.decodeIfPresent(Int.self, forKey: .dsparkBlockSize) ?? 0
        dsparkNRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .dsparkNRoutedExperts) ?? 0
        dsparkNumExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .dsparkNumExpertsPerTok) ?? 0
        expertResidentCapacity = try c.decodeIfPresent(Int.self, forKey: .expertResidentCapacity)
        engramLayerIds = try c.decodeIfPresent([Int].self, forKey: .engramLayerIds) ?? []
        engramMaxNgram = try c.decodeIfPresent(Int.self, forKey: .engramMaxNgram) ?? 4
        engramNHeads = try c.decodeIfPresent(Int.self, forKey: .engramNHeads) ?? 8
        engramHeadDim = try c.decodeIfPresent(Int.self, forKey: .engramHeadDim) ?? 256
    }
}

// MARK: - Helper Functions

private func softplus(_ x: MLXArray) -> MLXArray {
    MLX.log(1 + MLX.exp(x))
}

/// oMLX V4.1 YaRN params; applied only when `compress_ratio != 0`.
struct DeepseekV41Yarn {
    var originalSeqLen: Int
    var betaFast: Float
    var betaSlow: Float
    var factor: Float
}

/// oMLX `_rope`: rotate the last `dims` of `[B, L, …, D]`, with sequence on axis 1.
///
/// MLXFast.RoPE treats axis -2 as sequence, so `[B, L, H, D]` would rotate heads.
func deepseekV41Rope(
    _ x: MLXArray, positions: MLXArray, dims: Int, base: Float,
    yarn: DeepseekV41Yarn?, inverse: Bool
) -> MLXArray {
    let last = x.dim(-1)
    precondition(dims > 0 && dims % 2 == 0 && last >= dims)
    let half = dims / 2
    let idx = MLXArray(Array(stride(from: 0, to: dims, by: 2))).asType(.float32)
    var freq = 1 / MLX.pow(MLXArray(base), idx / Float(dims))
    if let yarn, yarn.originalSeqLen > 0 {
        func correction(_ rotations: Float) -> Float {
            Float(dims) * logf(Float(yarn.originalSeqLen) / (rotations * 2 * Float.pi))
                / (2 * logf(base))
        }
        let low = max(floor(correction(yarn.betaFast)), 0)
        let high = min(ceil(correction(yarn.betaSlow)), Float(dims - 1))
        let rampDen = max(high - low, Float(1e-3))
        let ar = MLXArray(Array(0..<half)).asType(.float32)
        let smooth = 1 - clip((ar - low) / rampDen, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
        freq = freq / yarn.factor * (1 - smooth) + freq * smooth
    }
    let pos = positions.asType(.float32)
    var angles = pos.reshaped([pos.size, 1]) * freq.reshaped([1, half])
    if inverse { angles = -angles }
    var angleShape = [1, pos.size]
    if x.ndim > 3 {
        for _ in 0..<(x.ndim - 3) { angleShape.append(1) }
    }
    angleShape.append(half)
    angles = angles.reshaped(angleShape)
    let tail = x[.ellipsis, (last - dims)...].asType(.float32)
    let pairs = tail.reshaped(tail.shape.dropLast() + [half, 2])
    let a = pairs[.ellipsis, 0]
    let b = pairs[.ellipsis, 1]
    let c = MLX.cos(angles)
    let s = MLX.sin(angles)
    let rotated = MLX.stacked([a * c - b * s, a * s + b * c], axis: -1)
        .reshaped(tail.shape)
        .asType(x.dtype)
    if last == dims { return rotated }
    return concatenated([x[.ellipsis, ..<(last - dims)], rotated], axis: -1)
}

func deepseekV41Rope(
    _ x: MLXArray, start: Int, dims: Int, base: Float,
    yarn: DeepseekV41Yarn?, inverse: Bool
) -> MLXArray {
    let L = x.dim(1)
    let pos = MLXArray((0..<L).map { Float(start + $0) })
    return deepseekV41Rope(x, positions: pos, dims: dims, base: base, yarn: yarn, inverse: inverse)
}

/// oMLX V4.1: `pre` is one-hot on stream 0 so only the first HC copy carries the embed.
func deepseekV41InitialPre(batch: Int, length: Int, hcMult: Int) -> MLXArray {
    var flags = [Float](repeating: 0, count: max(hcMult, 1))
    if hcMult > 0 { flags[0] = 1 }
    return MLX.broadcast(
        MLXArray(flags).reshaped([1, 1, hcMult]),
        to: [batch, length, hcMult])
}

func hcSplitSinkhorn(
    mixes: MLXArray, scale: MLXArray, base: MLXArray,
    hcMult: Int, nIters: Int, eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let hc = hcMult
    let pre = MLXNN.sigmoid(mixes[.ellipsis, ..<hc] * scale[0] + base[..<hc]) + eps
    let post = 2 * MLXNN.sigmoid(mixes[.ellipsis, hc..<(2 * hc)] * scale[1] + base[hc..<(2 * hc)])
    let combLogits = mixes[.ellipsis, (2 * hc)...].reshaped(mixes.shape.dropLast() + [hc, hc]) * scale[2]
        + base[(2 * hc)...].reshaped([hc, hc])
    var comb = MLX.softmax(combLogits, axis: -1) + eps
    comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
    if nIters > 1 {
        for _ in 0..<(nIters - 1) {
            comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        }
    }
    return (pre, post, comb)
}

/// Project 4-stream residual → (pre, post, comb). Does not collapse the streams.
func hcMixes(
    x: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray,
    hcMult: Int, nIters: Int, eps: Float, normEps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let xf = x.reshaped([B, L, H * D]).asType(.float32)
    let rsqrt = MLX.rsqrt(MLX.mean(xf * xf, axis: -1, keepDims: true) + normEps)
    let mixes = (xf.matmul(fn.T)) * rsqrt
    return hcSplitSinkhorn(
        mixes: mixes, scale: scale, base: base,
        hcMult: hcMult, nIters: nIters, eps: eps)
}

/// Weighted sum of HC copies with incoming `pre` (oMLX `hc_pre`).
func hcReduce(_ x: MLXArray, pre: MLXArray) -> MLXArray {
    MLX.sum(x.asType(.float32) * pre[.ellipsis, .newAxis], axis: 2).asType(x.dtype)
}

/// oMLX: `einsum("bsij,bsid->bsjd", comb, residual)` — comb is [in, out].
func hcPost(
    x: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
) -> MLXArray {
    let termNew = post[.ellipsis, .newAxis] * x[.ellipsis, .newAxis, 0...].asType(.float32)
    let combT = comb.asType(.float32).transposed(0, 1, 3, 2)
    let termRes = combT.matmul(residual.asType(.float32))
    return (termNew + termRes).asType(x.dtype)
}

/// oMLX RMSNorm: always accumulate in float32, keep the checkpoint eps (1e-20).
func rmsNormF32(_ norm: RMSNorm, _ x: MLXArray) -> MLXArray {
    let f = x.asType(.float32)
    let w = norm.weight.asType(.float32)
    return (f * MLX.rsqrt(MLX.mean(f * f, axis: -1, keepDims: true) + norm.eps) * w)
        .asType(x.dtype)
}

// MARK: - HyperConnection

public class DeepseekV4HyperConnection: Module {
    @ParameterInfo var base: MLXArray
    @ParameterInfo var fn: MLXArray
    @ParameterInfo var scale: MLXArray

    init(hcMult: Int, hiddenSize: Int, mixHC: Int? = nil) {
        let m = mixHC ?? (2 + hcMult) * hcMult
        self._fn.wrappedValue = zeros([m, hcMult * hiddenSize], dtype: .float32)
        self._base.wrappedValue = zeros([m], dtype: .float32)
        self._scale.wrappedValue = ones([3], dtype: .float32)
    }

    init(hcMult: Int, hiddenSize: Int, headMix: Bool) {
        self._fn.wrappedValue = zeros([hcMult, hcMult * hiddenSize], dtype: .float32)
        self._base.wrappedValue = zeros([hcMult], dtype: .float32)
        self._scale.wrappedValue = zeros([1], dtype: .float32)
    }
}

// MARK: - Shared CSA2 state (compressed KV produced by kv_source layers)

final class DeepseekV41AttnShared {
    var compressed: [Int: MLXArray] = [:]
    var indexK: MLXArray?
    var indexIdx: MLXArray?
    func reset() {
        compressed.removeAll()
        indexK = nil
        indexIdx = nil
    }
}

/// oMLX `sparse_attention`: gather-selected keys plus an attention sink logit.
func deepseekV41SparseAttention(
    q: MLXArray, selected: MLXArray, indices: MLXArray, sink: MLXArray, scale: Float
) -> MLXArray {
    let qf = q.asType(.float32)
    let sf = selected.asType(.float32)
    let scores = (qf.expandedDimensions(axis: 3) * sf.expandedDimensions(axis: 2)).sum(axis: -1)
        * scale
    let valid = indices.expandedDimensions(axis: 2) .>= 0
    let masked = MLX.which(valid, scores, MLXArray(-Float.infinity))
    let kCount = indices.dim(-1)
    let sinks = sink.asType(.float32).reshaped([1, 1, sink.dim(0), 1])
    let combined = concatenated([masked, MLX.broadcast(sinks, to: masked.shape.dropLast() + [1])], axis: -1)
    let weights = MLX.softmax(combined, axis: -1)[0..., 0..., 0..., ..<kCount]
    return (weights.expandedDimensions(axis: -1) * sf.expandedDimensions(axis: 2)).sum(axis: 3)
        .asType(q.dtype)
}

func deepseekV41GatherKV(_ kv: MLXArray, _ idx: MLXArray) -> MLXArray {
    // kv [B,T,D], idx [B,L,K] → [B,L,K,D]. oMLX serves one request (B=1).
    let B = idx.dim(0)
    let L = idx.dim(1)
    let K = idx.dim(2)
    let D = kv.dim(2)
    let flat = MLX.maximum(idx.reshaped([B, L * K]), 0)
    var rows: [MLXArray] = []
    rows.reserveCapacity(B)
    for b in 0..<B {
        rows.append(kv[b, flat[b], 0...])
    }
    return concatenated(rows, axis: 0).reshaped(B, L, K, D)
}

func deepseekV41WindowIndices(start: Int, length: Int, window: Int, oldLen: Int) -> MLXArray {
    // [1, L, W] int32, -1 where invalid. Matches oMLX Attention local window.
    var data = [Int32](repeating: -1, count: length * window)
    for t in 0..<length {
        let pos = start + t
        if start == 0 {
            let base = max(pos - window + 1, 0)
            let w = min(length, window)
            for j in 0..<w {
                let local = base + j
                let ok = local >= max(0, start - oldLen) && local <= pos
                data[t * window + j] = ok ? Int32(local - (start - oldLen)) : -1
            }
        } else {
            for j in 0..<window {
                let local = pos - window + 1 + j
                let ok = local >= max(0, start - oldLen) && local <= pos
                data[t * window + j] = ok ? Int32(local - (start - oldLen)) : -1
            }
        }
    }
    return MLXArray(data, [1, length, window])
}

// MARK: - Compressor (oMLX V4.1 pooling, no ape / overlap transform)

public class DeepseekV4Compressor: Module {
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear?
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let compressRatio: Int
    let headDim: Int
    var remKV: MLXArray?
    var remGate: MLXArray?

    init(config: DeepseekV4Configuration, compressRatio: Int, headDim: Int) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self._wkv.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        if compressRatio > 1 {
            self._wgate.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        }
    }

    func resetRemainder() {
        remKV = nil
        remGate = nil
    }

    /// oMLX Compressor.__call__: pool every `ratio` tokens with a softmax gate.
    func callAsFunction(_ x: MLXArray, start: Int) -> MLXArray? {
        let r = compressRatio
        if r <= 1 {
            return rmsNormF32(norm, wkv(x))
        }
        guard let wgate else { return rmsNormF32(norm, wkv(x)) }
        let xf = x.asType(.float32)
        var kv = wkv(xf)
        var gate = wgate(xf)
        let rem = start % r
        if rem > 0, let rk = remKV, let rg = remGate {
            kv = concatenated([rk[0..., ..<rem, 0...], kv], axis: 1)
            gate = concatenated([rg[0..., ..<rem, 0...], gate], axis: 1)
        }
        let cutoff = (kv.dim(1) / r) * r
        remKV = kv[0..., cutoff..., 0...]
        remGate = gate[0..., cutoff..., 0...]
        guard cutoff > 0 else { return nil }
        let groups = cutoff / r
        let d = kv.dim(2)
        let kvR = kv[0..., ..<cutoff, 0...].reshaped([kv.dim(0), groups, r, d])
        let gR = gate[0..., ..<cutoff, 0...].reshaped([gate.dim(0), groups, r, d])
        let pooled = (kvR * MLX.softmax(gR, axis: 2, precise: true)).sum(axis: 2)
        return rmsNormF32(norm, pooled.asType(x.dtype))
    }
}

// MARK: - Indexer (stub)

public class DeepseekV4Indexer: Module {
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "wk") var wk: Linear?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    let nHeads: Int
    let headDim: Int
    let ropeDims: Int
    let topk: Int
    let ropeBase: Float
    let yarn: DeepseekV41Yarn?
    let isKVSource: Bool

    init(layerId: Int, config: DeepseekV4Configuration, compressRatio: Int, isKVSource: Bool) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.ropeDims = config.qkRopeHeadDim
        self.topk = config.indexTopk
        self.isKVSource = isKVSource
        self.ropeBase = config.compressRopeTheta
        let scaling = config.ropeScaling
        self.yarn = DeepseekV41Yarn(
            originalSeqLen: scaling?["original_max_position_embeddings"]?.asInt() ?? 0,
            betaFast: scaling?["beta_fast"]?.asFloat() ?? 32,
            betaSlow: scaling?["beta_slow"]?.asFloat() ?? 1,
            factor: scaling?["factor"]?.asFloat() ?? 16)
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.indexNHeads * config.indexHeadDim, bias: false)
        self._weightsProj.wrappedValue = Linear(
            config.hiddenSize, config.indexNHeads, bias: false)
        if isKVSource {
            self._wk.wrappedValue = Linear(config.headDim, config.indexHeadDim, bias: false)
            self._kNorm.wrappedValue = RMSNorm(
                dimensions: config.indexHeadDim, eps: config.rmsNormEps)
        }
        _ = (layerId, compressRatio)
    }

    /// oMLX Indexer: score packed (here QAT-float) compressed keys, return top-k ids.
    func callAsFunction(
        x: MLXArray, qr: MLXArray, latent: MLXArray?, shared: DeepseekV41AttnShared,
        start: Int, ratio: Int
    ) -> MLXArray {
        let L = x.dim(1)
        let r = max(ratio, 1)
        if isKVSource, let latent, let wk, let kNorm {
            let nComp = latent.dim(1)
            let g0 = start / r
            let pos = MLXArray((0..<nComp).map { Float((g0 + $0) * r) })
            var key = deepseekV41Rope(
                kNorm(wk(latent)), positions: pos, dims: min(ropeDims, headDim),
                base: ropeBase, yarn: yarn, inverse: false)
            key = DeepseekV41Act.quantize(key, bits: 4, groupSize: 32, e4m3Scale: false)
            if let prev = shared.indexK, prev.dim(1) > 0 {
                shared.indexK = concatenated([prev, key], axis: 1)
            } else {
                shared.indexK = key
            }
        }
        var q = wqB(qr).reshaped(1, L, nHeads, headDim)
        q = deepseekV41Rope(
            q, start: start, dims: min(ropeDims, headDim), base: ropeBase, yarn: yarn,
            inverse: false)
        q = DeepseekV41Act.quantize(q, bits: 4, groupSize: 32, e4m3Scale: false)
        let wScale = pow(Float(headDim), -0.5) * pow(Float(nHeads), -0.5)
        let weights = weightsProj(x).asType(.float32) * wScale
        guard let keys = shared.indexK, keys.dim(1) > 0 else {
            return MLXArray.zeros([1, L, 0], dtype: .int32)
        }
        let T = keys.dim(1)
        let take = min(T, topk)
        if T <= topk {
            return causalIndexAll(start: start, length: L, keys: T, ratio: r)
        }
        return indexTopk(q: q, keys: keys, weights: weights, start: start, ratio: r, k: take)
    }

    private func causalIndexAll(start: Int, length: Int, keys: Int, ratio: Int) -> MLXArray {
        var data = [Int32](repeating: -1, count: length * keys)
        for t in 0..<length {
            let vis = min(keys, (start + t + 1) / max(ratio, 1))
            for j in 0..<vis { data[t * keys + j] = Int32(j) }
        }
        return MLXArray(data, [1, length, keys])
    }

    private func indexTopk(
        q: MLXArray, keys: MLXArray, weights: MLXArray, start: Int, ratio: Int, k: Int
    ) -> MLXArray {
        let L = q.dim(1)
        let T = keys.dim(1)
        let qf = q.asType(.float32)
        let kf = keys.asType(.float32)
        let dots = (qf.expandedDimensions(axis: 3) * kf.expandedDimensions(axis: 1).expandedDimensions(axis: 2))
            .sum(axis: -1)
        let relu = MLX.maximum(dots, MLXArray(Float(0)))
        var scores = (relu * weights.expandedDimensions(axis: 3)).sum(axis: 2)
        var mask = [Float](repeating: 0, count: L * T)
        let r = max(ratio, 1)
        for t in 0..<L {
            let vis = (start + t + 1) / r
            for j in 0..<T where j >= vis { mask[t * T + j] = -Float.infinity }
        }
        scores = scores + MLXArray(mask, [1, L, T])
        let order = MLX.argSort(-scores, axis: -1)[.ellipsis, ..<k].asType(.int32)
        return order
    }
}

// MARK: - Attention

public class DeepseekV4Attention: Module {
    let layerId: Int
    let nHeads: Int
    let headDim: Int
    let rd: Int
    let nGroups: Int
    let oLoraRank: Int
    let scale: Float
    let eps: Float
    let compressRatio: Int
    let windowSize: Int
    let isKVSource: Bool
    let kvSourceIds: [Int]
    let useFp8Act: Bool

    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear
    @ParameterInfo(key: "attn_sink") var attnSink: MLXArray
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV4Indexer?
    let ropeBase: Float
    let compressRopeBase: Float
    let yarn: DeepseekV41Yarn?

    init(layerId: Int, config: DeepseekV4Configuration) {
        self.layerId = layerId
        self.nHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.rd = config.qkRopeHeadDim
        self.nGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        let cr = config.compressRatios.count > layerId ? config.compressRatios[layerId] : 0
        self.compressRatio = cr
        self.windowSize = max(config.slidingWindow, 1)
        self.kvSourceIds = config.kvSourceLayerIds
        self.isKVSource = cr > 0 && (
            config.kvSourceLayerIds.isEmpty || config.kvSourceLayerIds.contains(layerId)
        )
        self.useFp8Act = config.expertResidentCapacity != nil

        self._wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)
        self._wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        // Grouped output: wo_a takes nGroups × headDim (= oGroups × headDim) not nHeads × headDim.
        // V4-Flash source weights confirm: wo_a.weight shape [8192, 512] uint32 = Linear(4096→8192)
        // where 4096 = oGroups(8) × headDim(512).
        let outputDim = config.oGroups * config.headDim
        self._woA.wrappedValue = Linear(outputDim, config.oGroups * config.oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)

        self._attnSink.wrappedValue = zeros([config.numAttentionHeads])

        // V4.1: only kv_source layers own compressor weights. Other layers
        // with compress_ratio > 0 read the shared compressed cache.
        // cr==1 source layers in this 2-bit dump lack wgate; skip them.
        if isKVSource {
            self._compressor.wrappedValue = DeepseekV4Compressor(
                config: config, compressRatio: max(cr, 1), headDim: config.headDim)
        }
        let isIndexSource = config.indexSourceLayerIds.contains(layerId)
        if isIndexSource {
            self._indexer.wrappedValue = DeepseekV4Indexer(
                layerId: layerId, config: config, compressRatio: max(cr, 1),
                isKVSource: isKVSource)
        }

        // oMLX applies YaRN only when compress_ratio != 0.
        self.compressRopeBase = config.compressRopeTheta
        if cr > 0 {
            self.ropeBase = config.compressRopeTheta
            let scaling = config.ropeScaling
            self.yarn = DeepseekV41Yarn(
                originalSeqLen: scaling?["original_max_position_embeddings"]?.asInt() ?? 0,
                betaFast: scaling?["beta_fast"]?.asFloat() ?? 32,
                betaSlow: scaling?["beta_slow"]?.asFloat() ?? 1,
                factor: scaling?["factor"]?.asFloat() ?? 16)
        } else {
            self.ropeBase = config.ropeTheta
            self.yarn = nil
        }
    }

    private func kvSourceLayer(for layer: Int) -> Int? {
        kvSourceIds.filter { $0 <= layer }.max()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?, xFull: MLXArray, shared: DeepseekV41AttnShared
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let start = cache?.offset ?? 0
        if start == 0 { compressor?.resetRemainder() }
        let xq = useFp8Act ? DeepseekV41Act.quantize(x) : x

        let qa = wqA(xq)
        let qr = rmsNormF32(qNorm, qa)
        var q = wqB(useFp8Act ? DeepseekV41Act.quantize(qr) : qr).reshaped(B, L, nHeads, headDim)
        var kv = rmsNormF32(kvNorm, wkv(xq))
        if useFp8Act, layerId == 0, start == 0 {
            MLX.eval(qa, qr, q, kv)
            NovaMLXLog.info(
                "[V41] L0 wq_a=\(qa.asType(.float32).abs().max().item(Float.self)) "
                    + "qr=\(qr.asType(.float32).abs().max().item(Float.self)) "
                    + "wq_b=\(q.asType(.float32).abs().max().item(Float.self)) "
                    + "kvn=\(kv.asType(.float32).abs().max().item(Float.self))"
            )
        }

        q = deepseekV41Rope(q, start: start, dims: rd, base: ropeBase, yarn: yarn, inverse: false)
        kv = deepseekV41Rope(kv, start: start, dims: rd, base: ropeBase, yarn: yarn, inverse: false)
        // CSA2 QAT: window KV is FP8/g32 (oMLX pack_activation default).
        if useFp8Act {
            kv = DeepseekV41Act.quantize(kv, bits: 8, groupSize: 32, e4m3Scale: false)
        }
        if useFp8Act, layerId == 0, start == 0 {
            MLX.eval(q, kv)
            NovaMLXLog.info(
                "[V41] L0 q_rope=\(q.asType(.float32).abs().max().item(Float.self)) "
                    + "kv_rope=\(kv.asType(.float32).abs().max().item(Float.self)) "
                    + "sink=\(attnSink.asType(.float32).abs().max().item(Float.self))"
            )
        }

        var windowKV = kv
        var oldLen = 0
        if let cache {
            let kvExpanded = kv.expandedDimensions(axis: 1)
            let (cachedK, _) = cache.update(keys: kvExpanded, values: kvExpanded)
            windowKV = cachedK.squeezed(axis: 1)
            if windowKV.dim(1) > windowSize {
                windowKV = windowKV[0..., (windowKV.dim(1) - windowSize)..., 0...]
            }
            oldLen = min(start, windowSize, max(windowKV.dim(1) - L, 0))
        }
        let wi = deepseekV41WindowIndices(
            start: start, length: L, window: min(windowSize, windowKV.dim(1)), oldLen: oldLen)
        var selected = deepseekV41GatherKV(windowKV, wi)
        var indices = wi

        if compressRatio > 0 {
            if isKVSource, let comp = compressor {
                if let pooled = comp(xFull, start: start) {
                    let ratio = max(compressRatio, 1)
                    let nComp = pooled.dim(1)
                    let g0 = start / ratio
                    let pos = MLXArray((0..<nComp).map { Float((g0 + $0) * ratio) })
                    var rotated = deepseekV41Rope(
                        pooled, positions: pos, dims: rd, base: compressRopeBase,
                        yarn: yarn, inverse: false)
                    // Compressed KV is FP4/g16 with e4m3 scales (oMLX pack_activation).
                    if useFp8Act {
                        rotated = DeepseekV41Act.quantize(
                            rotated, bits: 4, groupSize: 16, e4m3Scale: true)
                    }
                    if start == 0 {
                        shared.compressed[layerId] = rotated
                    } else if let prev = shared.compressed[layerId] {
                        shared.compressed[layerId] = concatenated([prev, rotated], axis: 1)
                    } else {
                        shared.compressed[layerId] = rotated
                    }
                    if let idxr = indexer {
                        shared.indexIdx = idxr(
                            x: xFull, qr: qr, latent: pooled, shared: shared,
                            start: start, ratio: ratio)
                    }
                }
            }
            let source = kvSourceLayer(for: layerId) ?? layerId
            if indexer != nil, shared.indexIdx == nil, let idxr = indexer {
                shared.indexIdx = idxr(
                    x: xFull, qr: qr, latent: nil, shared: shared,
                    start: start, ratio: max(compressRatio, 1))
            }
            if let pooled = shared.compressed[source], pooled.dim(1) > 0 {
                let cCount = pooled.dim(1)
                let ci: MLXArray
                if let top = shared.indexIdx, top.dim(2) > 0 {
                    ci = top
                } else {
                    let take = min(cCount, 512)
                    var ciData = [Int32](repeating: -1, count: L * take)
                    for t in 0..<L {
                        let vis = min(take, (start + t + 1) / max(compressRatio, 1))
                        for j in 0..<vis { ciData[t * take + j] = Int32(j) }
                    }
                    ci = MLXArray(ciData, [B, L, take])
                }
                let gatheredC = deepseekV41GatherKV(pooled, ci)
                selected = concatenated([selected, gatheredC], axis: 2)
                indices = concatenated([indices, ci], axis: 2)
            }
        }

        var output = deepseekV41SparseAttention(
            q: q, selected: selected, indices: indices, sink: attnSink, scale: scale)
        output = deepseekV41Rope(
            output, start: start, dims: rd, base: ropeBase, yarn: yarn, inverse: true)
        if useFp8Act, layerId == 0, start == 0 {
            MLX.eval(output)
            NovaMLXLog.info(
                "[V41] L0 attn_pre_wo=\(output.asType(.float32).abs().max().item(Float.self))"
            )
        }

        // oMLX: grouped = out.reshape(1, L, o_groups, -1);
        // weight = wo_a.weight.reshape(o_groups, o_lora_rank, -1);
        // einsum("bsgd,grd->bsgr") then wo_b. wo_a is dense; only wo_b FP8-quants input.
        let groupSize = nHeads / nGroups
        let perGroupInDim = groupSize * headDim
        let perGroupOutDim = oLoraRank
        let o = output.reshaped(B, L, nGroups, perGroupInDim)
        TierHooks.linearSyncHook?(woA)
        var groupOutputs: [MLXArray] = []
        for g in 0..<nGroups {
            let oG = o[0..., 0..., g, 0...]
            let rowStart = g * perGroupOutDim
            let rowEnd = rowStart + perGroupOutDim
            if let qLin = woA as? QuantizedLinear {
                let wSlice = qLin.weight[rowStart..<rowEnd, 0...]
                let sSlice = qLin.scales[rowStart..<rowEnd, 0...]
                let bSlice = qLin.biases?[rowStart..<rowEnd, 0...]
                groupOutputs.append(MLX.quantizedMM(
                    oG, wSlice,
                    scales: sSlice, biases: bSlice,
                    transpose: true, groupSize: qLin.groupSize, bits: qLin.bits, mode: qLin.mode))
            } else {
                let wSlice = woA.weight[rowStart..<rowEnd, 0...]
                groupOutputs.append(matmul(oG, wSlice.swappedAxes(-2, -1)))
            }
        }
        let flattened = MLX.stacked(groupOutputs, axis: 2).reshaped(B, L, nGroups * perGroupOutDim)
        let woBIn = useFp8Act ? DeepseekV41Act.quantize(flattened) : flattened
        return woB(woBIn)
    }
}

// MARK: - Gate

public class DeepseekV4Gate: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo var tid2eid: MLXArray?
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray?
    let topk: Int
    let scoringFunc: String
    let routeScale: Float
    let normTopkProb: Bool
    let isHash: Bool

    init(layerId: Int, config: DeepseekV4Configuration) {
        self.topk = config.numExpertsPerTok
        self.scoringFunc = config.scoringFunc
        self.routeScale = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.isHash = layerId < config.numHashLayers

        self._weight.wrappedValue = zeros([config.nRoutedExperts, config.hiddenSize])
        if isHash {
            self._tid2eid.wrappedValue = zeros([config.vocabSize, config.numExpertsPerTok], dtype: .int32)
        } else {
            self._eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts])
        }
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray? = nil) -> (MLXArray, MLXArray) {
        var scores = x.asType(.float32).matmul(weight.T)
        switch scoringFunc {
        case "softmax":
            scores = MLX.softmax(scores, axis: -1)
        case "sigmoid":
            scores = MLXNN.sigmoid(scores)
        default:  // sqrtsoftplus
            scores = MLX.sqrt(softplus(scores))
        }
        let originalScores = scores

        if !isHash {
            scores = scores + eScoreCorrectionBias!
        }

        let indices: MLXArray
        if isHash, let ids = inputIds {
            indices = tid2eid![ids.flattened()]
        } else {
            indices = stopGradient(argSort(-scores, axis: -1)[.ellipsis, ..<topk])
        }

        var weights = takeAlong(originalScores, indices, axis: -1)
        if scoringFunc != "softmax" && normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-9)
        }
        weights = (weights * routeScale).asType(x.dtype)
        return (weights, indices)
    }
}

// MARK: - Expert (shared experts, single MLP)

public class DeepseekV4Expert: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float = 0.0) {
        self.swigluLimit = swigluLimit
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = gateProj(x)
        var up = upProj(x)
        if swigluLimit > 0 {
            gate = MLX.minimum(gate, MLXArray(swigluLimit))
            up = clip(up, min: MLXArray(-swigluLimit), max: MLXArray(swigluLimit))
        }
        return downProj(MLXNN.silu(gate) * up)
    }
}

// MARK: - SwitchGLU (routed experts with swiglu_limit)

public class DeepseekV4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear
    let swigluLimit: Float

    init(config: DeepseekV4Configuration) {
        self.swigluLimit = config.swigluLimit
        let n = config.expertResidentCapacity ?? config.nRoutedExperts
        let d = config.hiddenSize
        let e = config.moeIntermediateSize
        NovaMLXLog.info(
            "[V41] SwitchGLU experts=\(n) hidden=\(d) inter=\(e) offload=\(config.expertResidentCapacity != nil)")
        if config.expertResidentCapacity != nil {
            // Packed MXFP4 resident slots (oMLX 5% capacity). Slot writes
            // keep a stable [capacity, ...] table; gather uses remapped ids.
            func mxfp4Switch(_ out: Int, _ inn: Int) -> QuantizedSwitchLinear {
                let packedIn = inn * 4 / 32
                let weight = MLXArray.zeros([n, out, packedIn], dtype: .uint32)
                let scales = MLXArray.zeros([n, out, inn / 32], dtype: .uint8)
                return QuantizedSwitchLinear(
                    inputDims: inn, outputDims: out, numExperts: n,
                    weight: weight, scales: scales, biases: nil,
                    groupSize: 32, bits: 4, mode: .mxfp4)
            }
            self._gateProj.wrappedValue = mxfp4Switch(e, d)
            self._upProj.wrappedValue = mxfp4Switch(e, d)
            self._downProj.wrappedValue = mxfp4Switch(d, e)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(inputDims: d, outputDims: e, numExperts: n, bias: false)
            self._upProj.wrappedValue = SwitchLinear(inputDims: d, outputDims: e, numExperts: n, bias: false)
            self._downProj.wrappedValue = SwitchLinear(inputDims: e, outputDims: d, numExperts: n, bias: false)
        }
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let xExp = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        var xIn = xExp
        if doSort {
            (xIn, idx, inverseOrder) = gatherSort(x: xExp, indices: indices)
        }
        var gate = gateProj(xIn, idx, sortedIndices: doSort)
        var up = upProj(xIn, idx, sortedIndices: doSort)
        if swigluLimit > 0 {
            gate = MLX.minimum(gate, MLXArray(swigluLimit))
            up = clip(up, min: MLXArray(-swigluLimit), max: MLXArray(swigluLimit))
        }
        var mid = MLXNN.silu(gate) * up
        if downProj is QuantizedSwitchLinear {
            mid = DeepseekV41Act.quantize(mid)
        }
        var out = downProj(mid, idx, sortedIndices: doSort)
        if doSort {
            out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
        }
        return (MLX.squeezed(out, axis: -2) * weights[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - MoE

public class DeepseekV4MoE: Module {
    @ModuleInfo(key: "gate") var gate: DeepseekV4Gate
    @ModuleInfo(key: "switch_mlp") var experts: DeepseekV4SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4Expert?
    let layerId: Int
    var expertBank: DeepseekV41ExpertBank?

    init(layerId: Int, config: DeepseekV4Configuration) {
        self.layerId = layerId
        self._gate.wrappedValue = DeepseekV4Gate(layerId: layerId, config: config)
        self._experts.wrappedValue = DeepseekV4SwitchGLU(config: config)
        if config.nSharedExperts > 0 {
            self._sharedExperts.wrappedValue = DeepseekV4Expert(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
                swigluLimit: config.swigluLimit)
        }
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray? = nil) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
        let xAct = expertBank != nil ? DeepseekV41Act.quantize(x) : x
        let xFlat = xAct.reshaped([-1, D])
        let (weights, indices) = gate(x.reshaped([-1, D]), inputIds: inputIds)
        if expertBank != nil, layerId == 0, x.dim(1) > 1 {
            MLX.eval(indices, weights)
            let last = indices.dim(0) - 1
            var idxList: [Int] = []
            for k in 0..<indices.dim(1) {
                idxList.append(Int(indices[last, k].item(Int32.self)))
            }
            NovaMLXLog.info("[V41] L0 gate idx last \(idxList)")
        }
        var routed: MLXArray
        if let bank = expertBank {
            // oMLX: chunk so each working set fits in resident expert slots.
            let k = max(indices.dim(-1), 1)
            let step = max(1, bank.capacity / k)
            let rows = xFlat.dim(0)
            var parts: [MLXArray] = []
            var start = 0
            while start < rows {
                let end = min(start + step, rows)
                let idx = indices[start..<end, 0...]
                let x = xFlat[start..<end, 0...]
                let w = weights[start..<end, 0...]
                let local = bank.ensure(layer: layerId, indices: idx, glu: experts)
                parts.append(experts(x, indices: local, weights: w))
                start = end
            }
            routed = concatenated(parts, axis: 0).reshaped(B, L, D)
        } else {
            routed = experts(xFlat, indices: indices, weights: weights).reshaped(B, L, D)
        }
        if let shared = sharedExperts {
            routed = (routed + shared(xAct)).asType(x.dtype)
        }
        return routed
    }
}

// MARK: - Block

public class DeepseekV4Block: Module {
    @ModuleInfo(key: "attn") var attn: DeepseekV4Attention
    @ModuleInfo(key: "ffn") var ffn: DeepseekV4MoE
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection
    var engram: DeepseekV41Engram?
    let config: DeepseekV4Configuration

    init(layerId: Int, config: DeepseekV4Configuration) {
        self.config = config
        self._attn.wrappedValue = DeepseekV4Attention(layerId: layerId, config: config)
        self._ffn.wrappedValue = DeepseekV4MoE(layerId: layerId, config: config)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV4HyperConnection(
            hcMult: config.hcMult, hiddenSize: config.hiddenSize)
        self._ffnHC.wrappedValue = DeepseekV4HyperConnection(
            hcMult: config.hcMult, hiddenSize: config.hiddenSize)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?, inputIds: MLXArray?, incomingPre: MLXArray,
        shared: DeepseekV41AttnShared, engramIndices: [Int]? = nil
    ) -> (MLXArray, MLXArray) {
        var x = x
        if let eng = engram, let idx = engramIndices, !idx.isEmpty {
            if let out = try? eng(x, indices: idx) {
                x = out
            }
        }
        let a = config
        // Attention mixes from current residual; reduce with the *incoming* pre
        // (previous sublayer). Matches oMLX Block.__call__.
        let (ap, ao, ac) = hcMixes(
            x: x, fn: attnHC.fn, scale: attnHC.scale, base: attnHC.base,
            hcMult: a.hcMult, nIters: a.hcSinkhornIters, eps: a.hcEps, normEps: a.rmsNormEps)
        let attnIn = rmsNormF32(attnNorm, hcReduce(x, pre: incomingPre))
        let attnOut = attn(attnIn, mask: mask, cache: cache, xFull: attnIn, shared: shared)
        var h = hcPost(x: attnOut, residual: x, post: ao, comb: ac)

        let (fp, fo, fc) = hcMixes(
            x: h, fn: ffnHC.fn, scale: ffnHC.scale, base: ffnHC.base,
            hcMult: a.hcMult, nIters: a.hcSinkhornIters, eps: a.hcEps, normEps: a.rmsNormEps)
        let ffnIn = rmsNormF32(ffnNorm, hcReduce(h, pre: ap))
        let ffnOut = ffn(ffnIn, inputIds: inputIds)
        if config.expertResidentCapacity != nil, ffn.layerId == 0 {
            MLX.eval(attnIn, attnOut, h, ffnIn, ffnOut)
            NovaMLXLog.info(
                "[V41] L0 attnIn=\(attnIn.asType(.float32).abs().max().item(Float.self)) "
                    + "attnOut=\(attnOut.asType(.float32).abs().max().item(Float.self)) "
                    + "afterAttnHC=\(h.asType(.float32).abs().max().item(Float.self)) "
                    + "ffnIn=\(ffnIn.asType(.float32).abs().max().item(Float.self)) "
                    + "ffnOut=\(ffnOut.asType(.float32).abs().max().item(Float.self))"
            )
        }
        h = hcPost(x: ffnOut, residual: h, post: fo, comb: fc)
        return (h, fp)
    }
}

// MARK: - Model (inner transformer body)

public class DeepseekV4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") public var embed: Embedding
    public var layers: [DeepseekV4Block]
    public var mtpLayers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    @ModuleInfo(key: "hc_head") public var hcHead: DeepseekV4HyperConnection
    let args: DeepseekV4Configuration
    /// V4.1 has no `hc_head_*`; collapse with the last FFN `pre` (oMLX `hc_pre`).
    var usesPipelinedMHC = true
    let attnShared = DeepseekV41AttnShared()
    var engramMeta: DeepseekV41EngramHash?
    var engramHistory: [[Int]] = []

    init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.usesPipelinedMHC = args.modelType == "deepseek_v41"
        self._embed.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0..<args.numHiddenLayers).map { DeepseekV4Block(layerId: $0, config: args) }
        let nMtp = args.numNextnPredictLayers >= 2 ? args.numNextnPredictLayers : 0
        let mtpCfg = args.mtpLayerConfig()
        // TIE maps `model.mtpLayers.N` → layer index 10_000+N. Keep the same
        // ids so SSD expert fetch and heat-map keys match the manifest.
        self.mtpLayers = (0..<nMtp).map {
            DeepseekV4Block(layerId: 10_000 + $0, config: mtpCfg)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._hcHead.wrappedValue = DeepseekV4HyperConnection(
            hcMult: args.hcMult, hiddenSize: args.hiddenSize, headMix: true)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let prefill = (cache?.first?.offset ?? 0) == 0
        var h = embed(inputs)  // [B, L, D]
        if args.expertResidentCapacity != nil, prefill {
            MLX.eval(h)
            NovaMLXLog.info("[V41] embed absMax=\(h.asType(.float32).abs().max().item(Float.self))")
        }
        h = MLX.repeated(MLX.expandedDimensions(h, axis: 2), count: args.hcMult, axis: 2)  // [B, L, hc, D]
        var pre = deepseekV41InitialPre(
            batch: h.dim(0), length: h.dim(1), hcMult: args.hcMult)
        if prefill {
            attnShared.reset()
            engramHistory = []
        }

        let mask = createAttentionMask(
            h: h[.ellipsis, 0, 0...], cache: cache?.first, windowSize: args.slidingWindow)

        var engramHashes: [[[Int]]]?
        if let meta = engramMeta {
            engramHashes = DeepseekV41EngramHashing.hashes(
                ids: inputs, meta: meta, history: &engramHistory)
        }

        for (i, layer) in layers.enumerated() {
            var engIdx: [Int]?
            if let hashes = engramHashes, let ix = args.engramLayerIds.firstIndex(of: i) {
                engIdx = hashes[ix].flatMap { $0 }
            }
            (h, pre) = layer(
                h, mask: mask, cache: cache?[i], inputIds: inputs, incomingPre: pre,
                shared: attnShared, engramIndices: engIdx)
            if i == 0, prefill {
                MLX.eval(h)
                NovaMLXLog.info("[V41] afterL0 absMax=\(h.asType(.float32).abs().max().item(Float.self))")
            }
        }

        let hOut: MLXArray
        if usesPipelinedMHC {
            hOut = hcReduce(h, pre: pre)
        } else {
            let (B, L, hc, D) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
            let hf = h.reshaped([B, L, hc * D]).asType(.float32)
            let rsqrt = MLX.rsqrt(MLX.mean(hf * hf, axis: -1, keepDims: true) + args.hcEps)
            let mixes = (hf.matmul(hcHead.fn.T)) * rsqrt
            let headPre = MLXNN.sigmoid(mixes * hcHead.scale[0] + hcHead.base) + args.hcEps
            hOut = MLX.sum(headPre[.ellipsis, .newAxis] * h.asType(.float32), axis: 2).asType(h.dtype)
        }

        return rmsNormF32(norm, hOut)
    }
}

// MARK: - Model (top-level, protocol conformance)

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel, MtpTarget, MtpDrafter {
    public var kvHeads: [Int] = []
    let args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var head: Linear
    public var nativeMtpAvailable = false

    public var mtpBlockSize: Int {
        guard nativeMtpAvailable, !model.mtpLayers.isEmpty else { return 0 }
        return args.dsparkBlockSize > 0 ? args.dsparkBlockSize : model.mtpLayers.count
    }

    init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = DeepseekV4ModelInner(args)
        self._head.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = model(inputs, cache: cache)
        let logits = head(h)
        if args.expertResidentCapacity != nil {
            MLX.eval(logits)
            MLX.Memory.clearCache()
        }
        return logits
    }

    public func mtpEmbed(_ tokens: MLXArray) -> MLXArray {
        model.embed(tokens)
    }

    public func mtpLmHead(_ hidden: MLXArray) -> MLXArray {
        head(hidden)
    }

    public func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (
        hidden: MLXArray, logits: MLXArray
    ) {
        let hidden = model(tokens, cache: cache)
        return (hidden, head(hidden))
    }

    public func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray, lmHead: @escaping (MLXArray) -> MLXArray
    ) {}

    public func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray {
        precondition(nativeMtpAvailable && !model.mtpLayers.isEmpty, "DeepSeek native DSpark/MTP weights missing")
        var h = hidden
        // lastTokenHidden can squeeze a size-1 seq dim; restore [B, S, D] then mHC.
        if h.ndim == 1 {
            h = h.reshaped(1, 1, h.dim(0))
        } else if h.ndim == 2 {
            h = h.dim(0) == 1 ? h.reshaped(1, 1, h.dim(1)) : h.reshaped(h.dim(0), 1, h.dim(1))
        }
        if h.ndim == 3 {
            h = MLX.repeated(MLX.expandedDimensions(h, axis: 2), count: args.hcMult, axis: 2)
        }
        precondition(
            h.ndim == 4,
            "DSpark hidden must be [batch, seq, hc, hidden], got \(h.shape)"
        )
        var pre = deepseekV41InitialPre(
            batch: h.dim(0), length: h.dim(1), hcMult: args.hcMult)
        let mask = createAttentionMask(
            h: h[.ellipsis, 0, 0...], cache: cache?.first, windowSize: args.slidingWindow)
        for (i, layer) in model.mtpLayers.enumerated() {
            (h, pre) = layer(
                h, mask: mask, cache: cache?[i], inputIds: nil, incomingPre: pre,
                shared: model.attnShared)
        }
        let hOut: MLXArray
        if model.usesPipelinedMHC {
            hOut = hcReduce(h, pre: pre)
        } else {
            let hc = h.dim(2)
            let d = h.dim(3)
            let hf = h.reshaped([h.dim(0), h.dim(1), hc * d]).asType(.float32)
            let rsqrt = MLX.rsqrt(MLX.mean(hf * hf, axis: -1, keepDims: true) + args.hcEps)
            let mixes = (hf.matmul(model.hcHead.fn.T)) * rsqrt
            let headPre = MLXNN.sigmoid(mixes * model.hcHead.scale[0] + model.hcHead.base)
                + args.hcEps
            hOut = MLX.sum(headPre[.ellipsis, .newAxis] * h.asType(.float32), axis: 2)
                .asType(h.dtype)
        }
        return rmsNormF32(model.norm, hOut)
    }

    public func mtpNewCache(parameters: GenerateParameters?) -> [KVCache] {
        model.mtpLayers.map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var flag = nativeMtpAvailable
        let remapped = DeepseekV4Sanitizer.remap(weights, config: args, nativeMtp: &flag)
        nativeMtpAvailable = flag
        model.usesPipelinedMHC = args.modelType == "deepseek_v41"
            || remapped["model.hc_head.fn"] == nil
        return remapped
    }

    public var layersList: [Module] { model.layers }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            !(key.contains("attn_hc.") || key.contains("ffn_hc.") ||
              key.contains("hc_head.") || key.contains("attn_sink"))
        }
    }

    public var loraLayers: [Module] { model.layers }
}

enum DeepseekV4Sanitizer {
    /// Hugging Face / mlx-community V4.1 keys → DeepseekV4 module tree.
    static func remap(
        _ weights: [String: MLXArray],
        config: DeepseekV4Configuration,
        nativeMtp: inout Bool
    ) -> [String: MLXArray] {
        // TIE streams SwitchLinear expert shards from SSD, so sanitize never
        // sees mtp*.experts / switch_mlp. Dense mtpLayers.* (attn, gate,
        // shared expert) in the in-memory dict is enough to enable DSpark.
        nativeMtp = weights.keys.contains {
            $0.hasPrefix("mtp.") || $0.contains("mtpLayers")
        }
        var w = [String: MLXArray]()
        w.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") { continue }
            if key.hasPrefix("vision.") || key.hasPrefix("aligner.") { continue }
            if key.hasPrefix("image_") { continue }
            // DSpark leftover: mtp.N.norm is not a DeepseekV4Block child (attn_norm/ffn_norm are).
            if key.contains("mtpLayers") && key.hasSuffix(".norm.weight")
                && !key.contains("attn_norm") && !key.contains("ffn_norm")
            {
                continue
            }
            // Indexer weights are live CSA2 parameters.
            if key.contains(".compressor.") {
                if let n = DeepseekV4Sanitizer.layerIndex(from: key) {
                    let cr = (n < config.compressRatios.count) ? config.compressRatios[n] : 0
                    let isSource = config.kvSourceLayerIds.isEmpty
                        || config.kvSourceLayerIds.contains(n)
                    if cr < 2 || !isSource { continue }
                }
            }
            if key.contains(".confidence_head") || key.contains(".markov_head")
                || key.contains(".main_proj") || key.contains(".main_norm")
            {
                continue
            }
            var k = key
            if k.hasPrefix("language_model.head.") {
                k = "lm_head." + k.dropFirst("language_model.head.".count)
            } else if k.hasPrefix("language_model.") {
                k = "model." + k.dropFirst("language_model.".count)
            } else if k == "norm.weight" || k.hasPrefix("norm.") {
                k = "model." + k
            } else if k.hasPrefix("mtp.") {
                let rest = k.dropFirst("mtp.".count)
                guard let dot = rest.firstIndex(of: ".") else { continue }
                let idx = rest[..<dot]
                let tail = rest[rest.index(after: dot)...]
                k = "model.mtpLayers.\(idx).\(tail)"
            }
            w[k] = value
        }

        remapHyperConnections(&w, layerCount: config.numHiddenLayers, prefix: "model.layers")
        let nMtp = config.numNextnPredictLayers >= 2 ? config.numNextnPredictLayers : 0
        if nMtp > 0 {
            remapHyperConnections(&w, layerCount: nMtp, prefix: "model.mtpLayers")
        }

        for i in 0..<config.numHiddenLayers {
            stackExperts(&w, prefix: "model.layers.\(i).ffn", expertCount: config.nRoutedExperts)
            remapSharedExpert(&w, prefix: "model.layers.\(i).ffn.shared_experts")
            remapGateBias(&w, prefix: "model.layers.\(i).ffn.gate")
        }
        let mtpExperts = config.dsparkNRoutedExperts > 0 ? config.dsparkNRoutedExperts : config.nRoutedExperts
        for i in 0..<nMtp {
            stackExperts(&w, prefix: "model.mtpLayers.\(i).ffn", expertCount: mtpExperts)
            remapSharedExpert(&w, prefix: "model.mtpLayers.\(i).ffn.shared_experts")
            remapGateBias(&w, prefix: "model.mtpLayers.\(i).ffn.gate")
        }

        if config.tieWordEmbeddings {
            w.removeValue(forKey: "lm_head.weight")
        }
        // mlx-community V4.1 Flash 2-bit writes `norm.weight` as all zeros.
        // Affine RMSNorm of zeros then zeros the hidden state and logits.
        if let nw = w["model.norm.weight"] {
            MLX.eval(nw)
            if nw.asType(.float32).abs().max().item(Float.self) == 0 {
                w["model.norm.weight"] = ones(nw.shape).asType(nw.dtype)
            }
        }
        return w
    }

    static func layerIndex(from key: String) -> Int? {
        guard let r = key.range(of: #"layers\.(\d+)"#, options: .regularExpression) else { return nil }
        let s = String(key[r])
        return Int(s.dropFirst("layers.".count))
    }

    private static func remapHyperConnections(_ w: inout [String: MLXArray], layerCount: Int, prefix: String) {
        for i in 0..<layerCount {
            for (src, dst) in [
                ("hc_attn_fn", "attn_hc.fn"),
                ("hc_attn_base", "attn_hc.base"),
                ("hc_attn_scale", "attn_hc.scale"),
                ("hc_ffn_fn", "ffn_hc.fn"),
                ("hc_ffn_base", "ffn_hc.base"),
                ("hc_ffn_scale", "ffn_hc.scale"),
            ] {
                let from = "\(prefix).\(i).\(src)"
                if let v = w.removeValue(forKey: from) {
                    w["\(prefix).\(i).\(dst)"] = v
                }
            }
        }
    }

    private static func remapGateBias(_ w: inout [String: MLXArray], prefix: String) {
        if let bias = w.removeValue(forKey: "\(prefix).bias") {
            w["\(prefix).e_score_correction_bias"] = bias
        }
        w.removeValue(forKey: "\(prefix).bias_vl")
    }

    private static func remapSharedExpert(_ w: inout [String: MLXArray], prefix: String) {
        for (src, dst) in [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")] {
            for suffix in ["weight", "scales", "biases"] {
                if let v = w.removeValue(forKey: "\(prefix).\(src).\(suffix)") {
                    w["\(prefix).\(dst).\(suffix)"] = v
                }
            }
        }
    }

    private static func stackExperts(_ w: inout [String: MLXArray], prefix: String, expertCount: Int) {
        guard expertCount > 0 else { return }
        guard w["\(prefix).experts.0.w1.weight"] != nil
            || w["\(prefix).experts.0.gate_proj.weight"] != nil else { return }
        let srcNames: [(String, String)]
        if w["\(prefix).experts.0.w1.weight"] != nil {
            srcNames = [("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")]
        } else {
            srcNames = [("gate_proj", "gate_proj"), ("up_proj", "up_proj"), ("down_proj", "down_proj")]
        }
        for (src, dst) in srcNames {
            for suffix in ["weight", "scales", "biases"] {
                var parts: [MLXArray] = []
                parts.reserveCapacity(expertCount)
                for e in 0..<expertCount {
                    let key = "\(prefix).experts.\(e).\(src).\(suffix)"
                    guard let t = w.removeValue(forKey: key) else { break }
                    parts.append(t)
                }
                if parts.count == expertCount {
                    w["\(prefix).switch_mlp.\(dst).\(suffix)"] = MLX.stacked(parts)
                }
            }
        }
    }
}
