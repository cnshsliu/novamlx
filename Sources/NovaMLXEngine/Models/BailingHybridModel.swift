import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM

// MARK: - Configuration

public struct BailingHybridConfiguration: Codable, Sendable {
    var modelType: String = ""
    var hiddenSize: Int = 4096
    var hiddenLayers: Int = 32
    var intermediateSize: Int = 9216
    var attentionHeads: Int = 32
    var kvHeads: Int = 32
    var rmsNormEps: Float = 1e-6
    var ropeTheta: Float = 6_000_000
    var vocabularySize: Int = 157184
    var tieWordEmbeddings: Bool = false
    var useBias: Bool = false
    var useQKVBias: Bool = false
    var useQKNorm: Bool = true
    var partialRotaryFactor: Float = 0.5
    var maxPositionEmbeddings: Int = 131072
    var ropeInterleave: Bool = true
    var ropeScaling: [String: StringOrNumber]?

    // MLA params
    var qLoraRank: Int = 1536
    var kvLoraRank: Int = 512
    var qkRopeHeadDim: Int = 64
    var qkNopeHeadDim: Int = 128
    var qkHeadDim: Int = 192
    var vHeadDim: Int = 128

    // Linear attention params
    var headDim: Int = 128
    var numKvHeadsForLinearAttn: Int = 32
    var groupNormSize: Int = 4
    var linearSilu: Bool = false

    // Hybrid routing
    var layerGroupSize: Int = 8
    var maxWindowLayers: Int = 20

    // MoE params
    var numExperts: Int = 256
    var numExpertsPerTok: Int = 8
    var numSharedExperts: Int = 1
    var moeIntermediateSize: Int = 1024
    var moeSharedExpertIntermediateSize: Int = 1024
    var firstKDenseReplace: Int = 1
    var nGroup: Int = 8
    var topkGroup: Int = 4
    var routedScalingFactor: Float = 2.5
    var normTopkProb: Bool = true
    var scoreFunction: String = "sigmoid"
    var numNextnPredictLayers: Int = 0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBias = "use_bias"
        case useQKVBias = "use_qkv_bias"
        case useQKNorm = "use_qk_norm"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeInterleave = "rope_interleave"
        case ropeScaling = "rope_scaling"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkHeadDim = "qk_head_dim"
        case vHeadDim = "v_head_dim"
        case headDim = "head_dim"
        case numKvHeadsForLinearAttn = "num_kv_heads_for_linear_attn"
        case groupNormSize = "group_norm_size"
        case linearSilu = "linear_silu"
        case layerGroupSize = "layer_group_size"
        case maxWindowLayers = "max_window_layers"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case scoreFunction = "score_function"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9216
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 32
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 6_000_000
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 157184
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useQKVBias = try c.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? false
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling) ?? nil
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1536
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        qkHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkHeadDim) ?? 192
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        numKvHeadsForLinearAttn = try c.decodeIfPresent(Int.self, forKey: .numKvHeadsForLinearAttn) ?? 32
        groupNormSize = try c.decodeIfPresent(Int.self, forKey: .groupNormSize) ?? 4
        linearSilu = try c.decodeIfPresent(Bool.self, forKey: .linearSilu) ?? false
        layerGroupSize = try c.decodeIfPresent(Int.self, forKey: .layerGroupSize) ?? 8
        maxWindowLayers = try c.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 20
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1024
        moeSharedExpertIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize) ?? 1024
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 8
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 4
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        scoreFunction = try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
        numNextnPredictLayers = try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
    }
}

// MARK: - MultiLinear (local copy — MLXLLM's is internal)

private class MultiLinear: Module, Quantizable {
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        let scale = sqrt(1.0 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [numHeads, outputDims, inputDims])
        super.init()
    }

    // transpose=True (default): x @ weight.T, maps inputDims → outputDims
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x.matmul(weight.swappedAxes(-1, -2))
    }

    // transpose=False: x @ weight, maps outputDims → inputDims
    func callNoTranspose(_ x: MLXArray) -> MLXArray {
        return x.matmul(weight)
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        let (qWeight, scales, biases) = MLX.quantized(weight, groupSize: groupSize, bits: bits, mode: mode)
        return QuantizedMultiLinear(weight: qWeight, scales: scales, biases: biases,
                                    inputDims: inputDims, outputDims: outputDims, numHeads: numHeads,
                                    groupSize: groupSize, bits: bits, mode: mode)
    }
}

private class QuantizedMultiLinear: Module, Quantized {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?

    init(weight: MLXArray, scales: MLXArray, biases: MLXArray?,
         inputDims: Int, outputDims: Int, numHeads: Int,
         groupSize: Int, bits: Int, mode: QuantizationMode) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        _weight.wrappedValue = weight
        _scales.wrappedValue = scales
        _biases.wrappedValue = biases
        super.init()
        freeze()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLX.quantizedMM(x, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
    }

    func callNoTranspose(_ x: MLXArray) -> MLXArray {
        return MLX.quantizedMM(x, weight, scales: scales, biases: biases, transpose: false, groupSize: groupSize, bits: bits)
    }
}

// MARK: - GroupRMSNorm

private class GroupRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupNormSize: Int

    init(dimensions: Int, groupNormSize: Int, eps: Float) {
        self.eps = eps
        self.groupNormSize = groupNormSize
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        var newShape = Array(shape.dropLast())
        newShape.append(groupNormSize)
        newShape.append(-1)
        let unflattened = x.reshaped(newShape)
        let identityWeight = MLXArray.ones([shape.last! / groupNormSize])
        let normed = MLXFast.rmsNorm(unflattened, weight: identityWeight, eps: eps)
        return weight * normed.reshaped(shape)
    }
}

// MARK: - MLA Attention (absorbed form)

// Port of mlx-lm bailing_hybrid.py MultiLatentAttention.
// Two paths: decode (L==1) projects q into latent, uses kvLatent as k=v;
// prefill (L>1) projects kvLatent into k/v space, uses raw qNope as queries.
// Position info via pe_scores = (qPe * scale) @ kPe^T as additive SDPA mask.
private class BailingHybridMLA: Module {
    let numHeads: Int
    let qkRopeHeadDim: Int
    let qkNopeHeadDim: Int
    let vHeadDim: Int
    let qkHeadDim: Int
    let kvLoraRank: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "dense") var oProj: Linear

    init(_ args: BailingHybridConfiguration) {
        self.numHeads = args.attentionHeads
        self.qkRopeHeadDim = args.qkRopeHeadDim
        self.qkNopeHeadDim = args.qkNopeHeadDim
        self.vHeadDim = args.vHeadDim
        self.qkHeadDim = args.qkHeadDim
        self.kvLoraRank = args.kvLoraRank
        self.scale = pow(Float(args.qkHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(args.hiddenSize, args.qLoraRank, bias: args.useQKVBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: args.qLoraRank, eps: args.rmsNormEps)
        _qBProj.wrappedValue = Linear(args.qLoraRank, args.attentionHeads * args.qkHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            args.hiddenSize, args.kvLoraRank + args.qkRopeHeadDim, bias: args.useQKVBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: args.kvLoraRank, eps: args.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(inputDims: args.qkNopeHeadDim, outputDims: args.kvLoraRank, numHeads: args.attentionHeads)
        _unembedOut.wrappedValue = MultiLinear(inputDims: args.kvLoraRank, outputDims: args.vHeadDim, numHeads: args.attentionHeads)
        _oProj.wrappedValue = Linear(args.attentionHeads * args.vHeadDim, args.hiddenSize, bias: args.useQKVBias)

        self.rope = initializeRope(
            dims: args.qkRopeHeadDim, base: args.ropeTheta,
            traditional: args.ropeInterleave,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // Q projection
        var q = qBProj(qALayerNorm(qAProj(x)))
        q = q.reshaped(B, L, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]  // [B, H, L, qkNopeHeadDim]
        var qPe = splitQ[1]    // [B, H, L, qkRopeHeadDim]

        // KV compression
        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)  // [B, 1, L, qkRopeHeadDim]
        let kvLatent = kvALayerNorm(compressedKv)  // [B, L, kvLoraRank]

        // RoPE — offset = cache.offset BEFORE update (number of previously cached tokens)
        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)

        // Expand kvLatent: [B, L, kvLoraRank] -> [B, 1, L, kvLoraRank]
        var kvExpanded = expandedDimensions(kvLatent, axis: 1)

        // Cache: store kvLatent as "keys", kPe as "values"
        if let cache {
            (kvExpanded, kPe) = cache.update(keys: kvExpanded, values: kPe)
        }
        // kvExpanded: [B, 1, S, kvLoraRank], kPe: [B, 1, S, qkRopeHeadDim]

        // Position encoding scores as additive attention mask
        var peScores = matmul(qPe * scale, kPe.transposed(0, 1, 3, 2))  // [B, H, L, S]

        // Apply causal mask for prefill (L > 1)
        if L > 1 {
            let S = kvExpanded.dim(2)
            let rowIdx = MLXArray(Int32(0)..<Int32(L)).reshaped([L, 1])
            let colIdx = MLXArray(Int32(0)..<Int32(S)).reshaped([1, S])
            let causalMask = colIdx .<= rowIdx  // [L, S] boolean
            peScores = MLX.where(causalMask.reshaped([1, 1, L, S]), peScores, MLXArray(Float(-1e9)))
        }

        let output: MLXArray
        if L == 1 {
            // Decode: project q into latent space, use kvLatent as both k and v
            let qProjected = callMultiLinear(embedQ, qNope)  // [B, H, 1, kvLoraRank]
            let k = kvExpanded  // [B, 1, S, kvLoraRank]
            let v = kvExpanded  // [B, 1, S, kvLoraRank]
            var attnOut = MLXFast.scaledDotProductAttention(
                queries: qProjected, keys: k, values: v,
                scale: scale, mask: peScores
            )
            attnOut = callMultiLinear(unembedOut, attnOut)  // [B, H, 1, vHeadDim]
            output = attnOut
        } else {
            // Prefill: project kvLatent into key/value space, use raw qNope as queries
            let k = callMultiLinearNoTranspose(embedQ, kvExpanded)  // [B, H, L, qkNopeHeadDim]
            let v = callMultiLinear(unembedOut, kvExpanded)  // [B, H, L, vHeadDim]
            output = MLXFast.scaledDotProductAttention(
                queries: qNope, keys: k, values: v,
                scale: scale, mask: peScores
            )
        }

        let result = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return oProj(result)
    }
}

private func callMultiLinear(_ module: Module, _ x: MLXArray) -> MLXArray {
    if let ml = module as? MultiLinear { return ml(x) }
    if let qml = module as? QuantizedMultiLinear { return qml(x) }
    fatalError("embed_q/unembed_out must be MultiLinear or QuantizedMultiLinear")
}

private func callMultiLinearNoTranspose(_ module: Module, _ x: MLXArray) -> MLXArray {
    if let ml = module as? MultiLinear { return ml.callNoTranspose(x) }
    if let qml = module as? QuantizedMultiLinear { return qml.callNoTranspose(x) }
    fatalError("embed_q/unembed_out must be MultiLinear or QuantizedMultiLinear")
}

// MARK: - GLA (Gated Linear Attention)

private class BailingHybridGLA: Module {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let scale: Float
    let rope: RoPELayer
    let slopeScalars: [Float]

    @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?
    @ModuleInfo(key: "dense") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "g_norm") var gNorm: GroupRMSNorm

    init(_ args: BailingHybridConfiguration, layerIdx: Int) {
        self.hiddenSize = args.hiddenSize
        self.numHeads = args.attentionHeads
        self.headDim = args.headDim
        self.ropeDim = Int(Float(args.headDim) * args.partialRotaryFactor)
        self.scale = pow(Float(args.headDim), -0.5)

        let numKvHeads = args.numKvHeadsForLinearAttn
        _queryKeyValue.wrappedValue = Linear(
            args.hiddenSize,
            (args.attentionHeads + 2 * numKvHeads) * args.headDim,
            bias: args.useQKVBias)

        if args.useQKNorm {
            _queryNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        }

        _oProj.wrappedValue = Linear(args.attentionHeads * args.headDim, args.hiddenSize, bias: args.useBias)
        _gProj.wrappedValue = Linear(args.hiddenSize, args.attentionHeads * args.headDim, bias: false)
        _gNorm.wrappedValue = GroupRMSNorm(
            dimensions: args.attentionHeads * args.headDim,
            groupNormSize: args.groupNormSize, eps: args.rmsNormEps)

        // Python uses traditional=False for GLA (half-stride rotation, NOT interleaved)
        self.rope = initializeRope(
            dims: ropeDim, base: args.ropeTheta, traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)

        // Per-head decay slopes scaled by layer index
        var slopes = Self.computeSlopes(args.attentionHeads)
        let denom = max(1, args.hiddenLayers - 1)
        let layerFactor = 1.0 - Float(layerIdx) / Float(denom) + 1e-5
        for i in 0..<slopes.count { slopes[i] = -slopes[i] * layerFactor }
        self.slopeScalars = slopes

        super.init()
    }

    private static func computeSlopes(_ n: Int) -> [Float] {
        func getSlopesPowerOf2(_ n: Int) -> [Float] {
            let start = pow(2.0, -(pow(2.0, -(log2(Float(n)) - 3))))
            return (0..<n).map { start * pow(start, Float($0)) }
        }

        let slopes: [Float]
        if log2(Float(n)).truncatingRemainder(dividingBy: 1) == 0 {
            slopes = getSlopesPowerOf2(n)
        } else {
            let closestPow2 = Int(pow(2.0, floor(log2(Float(n)))))
            slopes = getSlopesPowerOf2(closestPow2)
                + getSlopesPowerOf2(closestPow2 * 2).enumerated()
                    .filter { $0.offset % 2 == 1 }
                    .prefix(n - closestPow2)
                    .map { $0.element }
        }
        return slopes
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?, offset: Int) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let numKvHeads = numHeads

        // QKV projection
        var qkv = queryKeyValue(x)
        qkv = qkv.reshaped(B, L, numHeads + 2 * numKvHeads, headDim)
        let qkvSplit = split(qkv, indices: [numHeads, numHeads + numKvHeads], axis: -2)
        var q = qkvSplit[0]  // [B, L, H, D]
        var k = qkvSplit[1]  // [B, L, H, D]
        let v = qkvSplit[2]  // [B, L, H, D]

        // Optional QK norm
        if let qn = queryNorm { q = qn(q) }
        if let kn = keyNorm { k = kn(k) }

        // RoPE — uses explicit offset from MLA cache (shared position info)
        q = applyRope(rope, to: q, offset: offset)
        k = applyRope(rope, to: k, offset: offset)

        // Get or init recurrent state: [B, H, D, D]
        let stateShape = [B, numHeads, headDim, headDim]
        var state = cache?[0] ?? MLXArray.zeros(stateShape, dtype: x.dtype)

        // GLA recurrence: h = h * exp(g) + k^T @ v
        let slope = MLXArray(slopeScalars)
        let decay = exp(slope.reshaped([1, numHeads, 1, 1]))
        var output = MLXArray.zeros([B, L, numHeads, headDim], dtype: x.dtype)

        for t in 0..<L {
            let qt = q[0..., t, 0..., 0...]  // [B, H, D]
            let kt = k[0..., t, 0..., 0...]  // [B, H, D]
            let vt = v[0..., t, 0..., 0...]  // [B, H, D]

            // h = h * decay + k ⊗ v  (outer product)
            state = state * decay
                + kt.reshaped([B, numHeads, headDim, 1])
                * vt.reshaped([B, numHeads, 1, headDim])

            // o = q @ h — query retrieves from state (NOT state @ q)
            let ot = matmul(qt.reshaped([B, numHeads, 1, headDim]), state)
                .reshaped([B, numHeads, headDim])
            output[0..., t, 0..., 0...] = ot
        }

        // Save state
        if let cache { cache[0] = state }

        // Output: GroupRMSNorm + sigmoid gating
        output = gNorm(output.reshaped(B, L, -1))
        let gate = gProj(x)
        output = output * sigmoid(gate)
        return oProj(output)
    }

    // Apply RoPE in [B, S, H, D] format with explicit offset
    private func applyRope(_ rope: RoPELayer, to x: MLXArray, offset: Int) -> MLXArray {
        var xt = x.transposed(0, 2, 1, 3)  // [B, H, S, D]
        xt = rope(xt, offset: offset)
        return xt.transposed(0, 2, 1, 3)  // [B, S, H, D]
    }
}

// MARK: - MoE Components

private class BailingHybridGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ args: BailingHybridConfiguration) {
        self.topK = args.numExpertsPerTok
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb
        _gateProj.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _expertBias.wrappedValue = MLXArray.zeros([args.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let B = x.dim(0)
        let S = x.dim(1)
        let flat = x.reshaped(B * S, -1)

        let logits = gateProj(flat).asType(.float32)
        let origScores = sigmoid(logits)
        var routingScores = origScores + expertBias

        // Group-limited top-K
        if nGroup > 1 {
            let numExperts = routingScores.dim(-1)
            let expertsPerGroup = numExperts / nGroup
            routingScores = routingScores.reshaped(B * S, nGroup, expertsPerGroup)
            let groupScores = top(routingScores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            var groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
            groupIdx = broadcast(groupIdx, to: [B * S, k, expertsPerGroup])
            routingScores = putAlong(routingScores, stopGradient(groupIdx), values: MLXArray(0.0), axis: -2)
            routingScores = flattened(routingScores, start: -2, end: -1)
        }

        // Top-K selection
        let inds = argPartition(-routingScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var scores = takeAlong(origScores, inds, axis: -1)

        if topK > 1 && normTopkProb {
            let denom = scores.sum(axis: -1, keepDims: true) + 1e-20
            scores = scores / denom
        }
        scores = scores * routedScalingFactor

        return (inds.reshaped(B, S, -1), scores.reshaped(B, S, -1))
    }
}

private class BailingHybridMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private class BailingHybridSparseMoeBlock: Module {
    @ModuleInfo(key: "switch_mlp") var switchMlp: SwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingHybridGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: BailingHybridMLP?

    init(_ args: BailingHybridConfiguration) {
        _switchMlp.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts)
        _gate.wrappedValue = BailingHybridGate(args)
        if args.numSharedExperts > 0 {
            let sharedIntermediate = args.moeSharedExpertIntermediateSize * args.numSharedExperts
            _sharedExperts.wrappedValue = BailingHybridMLP(
                hiddenSize: args.hiddenSize, intermediateSize: sharedIntermediate)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMlp(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder Layer

private class BailingHybridDecoderLayer: Module {
    let isGlobal: Bool

    @ModuleInfo(key: "attention") var attention: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: BailingHybridConfiguration, layerIdx: Int) {
        let threshold = (args.hiddenLayers / args.layerGroupSize) * args.layerGroupSize
        self.isGlobal = (layerIdx + 1) % args.layerGroupSize == 0 || layerIdx >= threshold

        if isGlobal {
            _attention.wrappedValue = BailingHybridMLA(args)
        } else {
            _attention.wrappedValue = BailingHybridGLA(args, layerIdx: layerIdx)
        }

        if args.numExperts > 0 && layerIdx >= args.firstKDenseReplace {
            _mlp.wrappedValue = BailingHybridSparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = BailingHybridMLP(
                hiddenSize: args.hiddenSize, intermediateSize: args.intermediateSize)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?, offset: Int) -> MLXArray {
        let r: MLXArray
        if isGlobal {
            r = (attention as! BailingHybridMLA)(inputLayerNorm(x), cache: cache)
        } else {
            r = (attention as! BailingHybridGLA)(
                inputLayerNorm(x), cache: cache as? ArraysCache, offset: offset)
        }
        let h = x + r
        let mlpOut: MLXArray
        if let moe = mlp as? BailingHybridSparseMoeBlock {
            mlpOut = moe(postAttentionLayerNorm(h))
        } else {
            mlpOut = (mlp as! BailingHybridMLP)(postAttentionLayerNorm(h))
        }
        return h + mlpOut
    }
}

// MARK: - Inner Model

public class BailingHybridModelInner: Module {
    @ModuleInfo(key: "word_embeddings") var embedTokens: Embedding
    fileprivate let layers: [BailingHybridDecoderLayer]
    let norm: RMSNorm
    let faIdx: Int

    init(_ args: BailingHybridConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0..<args.hiddenLayers).map { BailingHybridDecoderLayer(args, layerIdx: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.faIdx = args.layerGroupSize - 1  // First MLA layer index
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]?) -> MLXArray {
        var hiddenStates = embedTokens(inputs)
        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        // Get offset from first MLA layer's cache — shared for GLA RoPE positioning
        let globalOffset: Int
        if let faCache = cacheArray[faIdx] {
            globalOffset = faCache.offset
        } else {
            globalOffset = 0
        }

        for (i, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, cache: cacheArray[i], offset: globalOffset)
        }

        return norm(hiddenStates)
    }
}

// MARK: - Top-level Model

public class BailingHybridModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: BailingHybridModelInner
    let configuration: BailingHybridConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { model.layers }

    public init(_ args: BailingHybridConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = BailingHybridModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.map { layer in
            if layer.isGlobal {
                return KVCacheSimple()
            }
            return ArraysCache(size: 1)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            // Drop MTP layers
            if key.contains("mtp.") { continue }
            // Drop rotary embedding inv_freq
            if key.contains("rotary_emb.inv_freq") { continue }
            sanitized[key] = value
        }

        // Tie word embeddings if configured
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        let nLayers = configuration.hiddenLayers

        for l in 0..<nLayers {
            let prefix = "model.layers.\(l)"

            // MLA kv_b_proj split: [num_heads * (qkNopeHeadDim + vHeadDim), kvLoraRank]
            // → embed_q.weight: [num_heads, kvLoraRank, qkNopeHeadDim]
            // → unembed_out.weight: [num_heads, vHeadDim, kvLoraRank]
            let kvBKey = "\(prefix).attention.kv_b_proj.weight"
            if let kvBWeight = sanitized[kvBKey] {
                sanitized.removeValue(forKey: kvBKey)
                let headDim = configuration.qkNopeHeadDim + configuration.vHeadDim
                let numHeads = configuration.attentionHeads
                let reshaped = kvBWeight.reshaped(numHeads, headDim, -1)
                let wk = reshaped[.ellipsis, ..<configuration.qkNopeHeadDim, 0...]
                    .swappedAxes(-1, -2)
                let wv = reshaped[.ellipsis, configuration.qkNopeHeadDim..., 0...]
                sanitized["\(prefix).attention.embed_q.weight"] = wk
                sanitized["\(prefix).attention.unembed_out.weight"] = wv
            }

            // Dequantize pre-quantized MultiLinear weights (embed_q, unembed_out)
            let attnPrefix = "\(prefix).attention"
            let isGlobal = (l + 1) % configuration.layerGroupSize == 0
                || l >= (nLayers / configuration.layerGroupSize) * configuration.layerGroupSize
            if isGlobal {
                for mlKey in ["embed_q", "unembed_out"] {
                    let wKey = "\(attnPrefix).\(mlKey).weight"
                    let sKey = "\(attnPrefix).\(mlKey).scales"
                    let bKey = "\(attnPrefix).\(mlKey).biases"
                    if let w = sanitized[wKey], w.dtype == .uint32,
                       let s = sanitized[sKey] {
                        let b = sanitized[bKey]
                        let bits = 4
                        let elemsPerInt = 32 / bits
                        let wDim = w.shape.last!
                        let sDim = s.shape.last!
                        let gs = max(1, (wDim / sDim) * elemsPerInt)
                        let deq = MLX.dequantized(w, scales: s, biases: b, groupSize: gs, bits: bits, mode: .affine)
                        sanitized[wKey] = deq
                        sanitized.removeValue(forKey: sKey)
                        sanitized.removeValue(forKey: bKey)
                    }
                }
            }

            // MoE expert stacking + gate remap
            if configuration.numExperts > 0 && l >= configuration.firstKDenseReplace {
                let mlpPrefix = "\(prefix).mlp"
                for m in ["gate_proj", "down_proj", "up_proj"] {
                    if sanitized["\(mlpPrefix).experts.0.\(m).weight"] != nil {
                        let toJoin = (0..<configuration.numExperts).compactMap { e in
                            sanitized.removeValue(forKey: "\(mlpPrefix).experts.\(e).\(m).weight")
                        }
                        if !toJoin.isEmpty {
                            sanitized["\(mlpPrefix).switch_mlp.\(m).weight"] = MLX.stacked(toJoin)
                        }
                    }
                }

                if let gateWeight = sanitized.removeValue(forKey: "\(mlpPrefix).gate.weight") {
                    sanitized["\(mlpPrefix).gate.gate_proj.weight"] = gateWeight
                }
                if let gateBias = sanitized.removeValue(forKey: "\(mlpPrefix).gate.bias") {
                    sanitized["\(mlpPrefix).gate.gate_proj.bias"] = gateBias
                }
            }
        }

        return sanitized
    }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("expert_bias")
        }
    }
}
