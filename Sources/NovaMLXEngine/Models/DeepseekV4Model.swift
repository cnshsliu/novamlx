import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM

// Port of DeepSeek-V4 architecture based on mlx-lm PR #1201 (akashgoswami)
// Pure MLX operations, no custom Metal kernels.
// Compressor is ephemeral (recomputed per forward), Indexer is a load-only stub.

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    var modelType: String = "deepseek_v4"
    var vocabSize: Int = 129280
    var hiddenSize: Int = 4096
    var numHiddenLayers: Int = 43
    var numHashLayers: Int = 3
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
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v4"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129280
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 43
        numHashLayers = try c.decodeIfPresent(Int.self, forKey: .numHashLayers) ?? 3
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
    }
}

// MARK: - Helper Functions

private func softplus(_ x: MLXArray) -> MLXArray {
    MLX.log(1 + MLX.exp(x))
}

private func applyInverseRoPE(_ x: MLXArray, rope: any RoPELayer, offset: Int) -> MLXArray {
    let sh = x.shape
    let rd = sh.last!
    let pairShape = sh.dropLast() + [rd / 2, 2]
    let pairs = x.reshaped(pairShape)
    let flip = concatenated([MLXArray(1.0), MLXArray(-1.0)], axis: 0).asType(x.dtype)
    let yConj = rope((pairs * flip).reshaped(sh), offset: offset)
    return (yConj.reshaped(pairShape) * flip).reshaped(sh)
}

private func hcSplitSinkhorn(
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

private func hcPre(
    x: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray,
    hcMult: Int, nIters: Int, eps: Float, normEps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let (B, L, H, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let xf = x.reshaped([B, L, H * D]).asType(.float32)
    let rsqrt = MLX.rsqrt(MLX.mean(xf * xf, axis: -1, keepDims: true) + normEps)
    let mixes = (xf.matmul(fn.T)) * rsqrt
    let (pre, post, comb) = hcSplitSinkhorn(
        mixes: mixes, scale: scale, base: base,
        hcMult: hcMult, nIters: nIters, eps: eps)
    let combined = MLX.sum(pre[.ellipsis, .newAxis] * x.asType(.float32), axis: 2)
    return (combined.asType(x.dtype), post, comb)
}

private func hcPost(
    x: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
) -> MLXArray {
    let termNew = post[.ellipsis, .newAxis] * x[.ellipsis, .newAxis, 0...].asType(.float32)
    let termRes = comb.asType(.float32).matmul(residual.asType(.float32))
    return (termNew + termRes).asType(x.dtype)
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
        self._scale.wrappedValue = zeros([3], dtype: .float32)
    }

    init(hcMult: Int, hiddenSize: Int, headMix: Bool) {
        self._fn.wrappedValue = zeros([hcMult, hcMult * hiddenSize], dtype: .float32)
        self._base.wrappedValue = zeros([hcMult], dtype: .float32)
        self._scale.wrappedValue = zeros([1], dtype: .float32)
    }
}

// MARK: - Compressor

public class DeepseekV4Compressor: Module {
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "wgate") var wgate: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo var ape: MLXArray
    let compressRatio: Int
    let headDim: Int
    let overlap: Bool
    let outDim: Int

    init(config: DeepseekV4Configuration, compressRatio: Int, headDim: Int) {
        self.compressRatio = compressRatio
        self.headDim = headDim
        self.overlap = compressRatio == 4
        let coff = overlap ? 2 : 1
        self.outDim = coff * headDim
        self._wkv.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._wgate.wrappedValue = Linear(config.hiddenSize, outDim, bias: false)
        self._norm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._ape.wrappedValue = zeros([compressRatio, outDim], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, S, _) = (x.dim(0), x.dim(1), x.dim(2))
        let r = compressRatio
        let keep = (S / r) * r
        if keep == 0 { return MLXArray.zeros([B, 0, headDim], dtype: x.dtype) }
        let xc = x[.ellipsis, ..<keep].asType(.float32)
        var kv = wkv(xc).reshaped([B, keep / r, r, outDim])
        var score = wgate(xc).reshaped([B, keep / r, r, outDim]) + ape
        if overlap {
            let d = headDim
            var kvOv = MLXArray.zeros([B, keep / r, 2 * r, d], dtype: kv.dtype)
            kvOv[.ellipsis, r..., 0...] = kv[.ellipsis, 0..., 0..., d...]
            kvOv[.ellipsis, 1..., ..<r, 0...] = kv[.ellipsis, ..<(keep / r - 1), 0..., ..<d, 0...]
            kv = kvOv
            var scoreOv = full([B, keep / r, 2 * r, d], values: MLXArray(Float(-1e9)), dtype: score.dtype)
            scoreOv[.ellipsis, r..., 0...] = score[.ellipsis, 0..., 0..., d...]
            scoreOv[.ellipsis, 1..., ..<r, 0...] = score[.ellipsis, ..<(keep / r - 1), 0..., ..<d, 0...]
            score = scoreOv
        }
        let weights = MLX.softmax(score, axis: 2, precise: true)
        kv = (kv * weights).sum(axis: 2)
        return norm(kv.asType(x.dtype))
    }
}

// MARK: - Indexer (stub)

public class DeepseekV4Indexer: Module {
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV4Compressor

    init(config: DeepseekV4Configuration, compressRatio: Int) {
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.indexNHeads * config.indexHeadDim, bias: false)
        self._weightsProj.wrappedValue = Linear(
            config.hiddenSize, config.indexNHeads, bias: false)
        self._compressor.wrappedValue = DeepseekV4Compressor(
            config: config, compressRatio: compressRatio, headDim: config.indexHeadDim)
    }
}

// MARK: - Attention

public class DeepseekV4Attention: Module {
    let nHeads: Int
    let headDim: Int
    let rd: Int
    let nopeDim: Int
    let nGroups: Int
    let oLoraRank: Int
    let scale: Float
    let eps: Float
    let compressRatio: Int

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
    let rope: any RoPELayer

    init(layerId: Int, config: DeepseekV4Configuration) {
        self.nHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.rd = config.qkRopeHeadDim
        self.nopeDim = config.headDim - config.qkRopeHeadDim
        self.nGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        let cr = config.compressRatios.count > layerId ? config.compressRatios[layerId] : 0
        self.compressRatio = cr

        self._wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._wqB.wrappedValue = Linear(
            config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)
        self._wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        // Grouped output: single linear handles all groups
        let totalAttnDim = config.numAttentionHeads * config.headDim
        self._woA.wrappedValue = Linear(totalAttnDim, config.oGroups * config.oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)

        self._attnSink.wrappedValue = zeros([config.numAttentionHeads])

        if cr > 0 {
            self._compressor.wrappedValue = DeepseekV4Compressor(
                config: config, compressRatio: cr, headDim: config.headDim)
        }
        if cr == 4 {
            self._indexer.wrappedValue = DeepseekV4Indexer(config: config, compressRatio: cr)
        }

        // RoPE: compress layers use different base
        let yarnCfg: [String: StringOrNumber]?
        let ropeBase: Float
        if cr > 0, let scaling = config.ropeScaling {
            yarnCfg = scaling
            ropeBase = config.compressRopeTheta
        } else {
            yarnCfg = config.ropeScaling
            ropeBase = config.ropeTheta
        }

        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: ropeBase,
            traditional: true,
            scalingConfig: yarnCfg,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?, xFull: MLXArray
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let offset = cache?.offset ?? 0

        // Q projection with QK norm
        let qr = qNorm(wqA(x))
        var q = wqB(qr).reshaped(B, L, nHeads, headDim).transposed(0, 2, 1, 3)
        q = q * MLX.rsqrt(MLX.mean(q * q, axis: -1, keepDims: true) + eps)

        // KV projection (single head, K==V)
        var kv = kvNorm(wkv(x))

        // RoPE on positional dims
        let qNope = q[.ellipsis, ..<nopeDim]
        let qPe = rope(q[.ellipsis, nopeDim...], offset: offset)
        q = concatenated([qNope, qPe], axis: -1)

        let kvNope = kv[.ellipsis, ..<nopeDim]
        let kvPe = rope(kv[.ellipsis, nopeDim...].reshaped(B, 1, L, rd), offset: offset)
            .squeezed(axis: 1)
        kv = concatenated([kvNope, kvPe], axis: -1)

        // Compression
        var compressed: MLXArray? = nil
        if compressRatio > 0 && L >= compressRatio {
            let c = compressor!(xFull)
            if c.dim(1) > 0 { compressed = c }
        }

        // Cache update: K==V for single KV head
        var allKV: MLXArray
        if let cache = cache {
            let kvExpanded = kv[.ellipsis, .newAxis, 0..., 0...]  // [B, 1, L, headDim]
            let (cachedK, _) = cache.update(keys: kvExpanded, values: kvExpanded)
            allKV = cachedK.squeezed(axis: 1)  // [B, totalLen, headDim]
        } else {
            allKV = kv
        }

        // Build mask with compressed padding
        var attnMask = mask
        if let comp = compressed {
            allKV = concatenated([comp, allKV], axis: 1)
            let nComp = comp.dim(1)
            switch attnMask {
            case .array(let maskArr):
                let padShape = maskArr.shape.dropLast() + [nComp]
                let padMask = full(padShape, values: MLXArray(0.0), dtype: maskArr.dtype)
                attnMask = .array(concatenated([padMask, maskArr], axis: -1))
            case .causal:
                // Need to materialize for compressed padding
                // compressed positions are always visible
                break
            case .none:
                break
            @unknown default:
                break
            }
        }

        // Attention: K == V
        let k = allKV[.ellipsis, .newAxis, 0..., 0...]
        let v = k
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: attnMask)

        // Inverse RoPE on positional dims of output
        let oNope = output[.ellipsis, ..<nopeDim]
        let oPe = applyInverseRoPE(output[.ellipsis, nopeDim...], rope: rope, offset: offset)
        var o = concatenated([oNope, oPe], axis: -1)

        // Grouped output projection
        o = o.transposed(0, 2, 1, 3).reshaped(B, L, nHeads * headDim)
        return woB(woA(o))
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
            indices = stopGradient(
                argPartition(-scores, kth: topk, axis: -1)[.ellipsis, ..<topk])
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
        let n = config.nRoutedExperts
        let d = config.hiddenSize
        let e = config.moeIntermediateSize
        self._gateProj.wrappedValue = SwitchLinear(inputDims: d, outputDims: e, numExperts: n, bias: false)
        self._upProj.wrappedValue = SwitchLinear(inputDims: d, outputDims: e, numExperts: n, bias: false)
        self._downProj.wrappedValue = SwitchLinear(inputDims: e, outputDims: d, numExperts: n, bias: false)
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
        var out = downProj(MLXNN.silu(gate) * up, idx, sortedIndices: doSort)
        if doSort {
            out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
        }
        return (MLX.squeezed(out, axis: -2) * weights[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

// MARK: - MoE

public class DeepseekV4MoE: Module {
    var gate: DeepseekV4Gate
    @ModuleInfo(key: "switch_mlp") var experts: DeepseekV4SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4Expert?

    init(layerId: Int, config: DeepseekV4Configuration) {
        self.gate = DeepseekV4Gate(layerId: layerId, config: config)
        self._experts.wrappedValue = DeepseekV4SwitchGLU(config: config)
        if config.nSharedExperts > 0 {
            self._sharedExperts.wrappedValue = DeepseekV4Expert(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
                swigluLimit: 0.0)
        }
    }

    func callAsFunction(_ x: MLXArray, inputIds: MLXArray? = nil) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))
        let xFlat = x.reshaped([-1, D])
        let (weights, indices) = gate(xFlat, inputIds: inputIds)
        var routed = experts(xFlat, indices: indices, weights: weights).reshaped(B, L, D)
        if let shared = sharedExperts {
            routed = (routed + shared(x)).asType(x.dtype)
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
        cache: KVCache?, inputIds: MLXArray?
    ) -> MLXArray {
        let a = config
        // mHC attention sublayer
        var residual = x
        let (y, post, comb) = hcPre(
            x: x, fn: attnHC.fn, scale: attnHC.scale, base: attnHC.base,
            hcMult: a.hcMult, nIters: a.hcSinkhornIters, eps: a.hcEps, normEps: a.rmsNormEps)
        let attnOut = attn(attnNorm(y), mask: mask, cache: cache, xFull: y)
        var xOut = hcPost(x: attnOut, residual: residual, post: post, comb: comb)

        // mHC FFN sublayer
        residual = xOut
        let (y2, post2, comb2) = hcPre(
            x: xOut, fn: ffnHC.fn, scale: ffnHC.scale, base: ffnHC.base,
            hcMult: a.hcMult, nIters: a.hcSinkhornIters, eps: a.hcEps, normEps: a.rmsNormEps)
        let ffnOut = ffn(ffnNorm(y2), inputIds: inputIds)
        return hcPost(x: ffnOut, residual: residual, post: post2, comb: comb2)
    }
}

// MARK: - Model (inner transformer body)

public class DeepseekV4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") public var embed: Embedding
    public var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    @ModuleInfo(key: "hc_head") public var hcHead: DeepseekV4HyperConnection
    let args: DeepseekV4Configuration

    init(_ args: DeepseekV4Configuration) {
        self.args = args
        self._embed.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0..<args.numHiddenLayers).map { DeepseekV4Block(layerId: $0, config: args) }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._hcHead.wrappedValue = DeepseekV4HyperConnection(
            hcMult: args.hcMult, hiddenSize: args.hiddenSize, headMix: true)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embed(inputs)  // [B, L, D]
        h = MLX.repeated(MLX.expandedDimensions(h, axis: 2), count: args.hcMult, axis: 2)  // [B, L, hc, D]

        let mask = createAttentionMask(
            h: h[.ellipsis, 0, 0...], cache: cache?.first, windowSize: args.slidingWindow)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], inputIds: inputs)
        }

        // HyperHead reduction: sigmoid-weighted sum of hc copies
        let (B, L, hc, D) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        let hf = h.reshaped([B, L, hc * D]).asType(.float32)
        let rsqrt = MLX.rsqrt(MLX.mean(hf * hf, axis: -1, keepDims: true) + args.hcEps)
        let mixes = (hf.matmul(hcHead.fn.T)) * rsqrt
        let pre = MLXNN.sigmoid(mixes * hcHead.scale[0] + hcHead.base) + args.hcEps
        let hOut = MLX.sum(pre[.ellipsis, .newAxis] * h.asType(.float32), axis: 2).asType(h.dtype)

        return norm(hOut)
    }
}

// MARK: - Model (top-level, protocol conformance)

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public var kvHeads: [Int] = []
    let args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var head: Linear

    init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = DeepseekV4ModelInner(args)
        self._head.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let h = model(inputs, cache: cache)
        return head(h)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights
        w = w.filter { key, _ in
            !key.starts(with: "mtp.") && !key.contains("rotary_emb.inv_freq")
        }
        if args.tieWordEmbeddings {
            w.removeValue(forKey: "lm_head.weight")
        }
        return w
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
