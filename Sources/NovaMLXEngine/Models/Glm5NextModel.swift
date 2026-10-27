import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXRandom
import NovaMLXUtils

// Text-only GLM-5.3-Flash (`model_type: glm5_next` / `glm5_next_text`).
//
// Numeric reference is mlx-vlm `glm5_next/language.py` (mlx-vlm >= 0.6.17):
// limited SwiGLU (limit 10), mHC mixes in float32, MoE router in float32,
// KDA linear attention, and NoPE MLA with a pooled sparse indexer.
// Vision weights are dropped. MTP layers (`layers.{num_hidden_layers}` and
// `mtp.*`) are dropped. Expert SSD streaming is not wired.
//
// Decode speed, measured by `GLM-CHAT` (prefill excluded):
// - Dense `scaledDotProductAttention` is the wrong kernel. Each query gathers
//   its own latent set, and absorbed decode stays in kv-lora space. Masking
//   the full context is the path mlx-vlm removed.
// - Checkpoint MTP is not a drafter here. A safe accept needs KDA state
//   rollback and a short-block verifier. Without those, a draft can be kept
//   when the target would have rejected it.
// - One-token decode does not `eval` inside every KDA step. The graph is
//   evaluated every `Glm5ChatDecode.decodeEvalStride` layers. Prefill still
//   evaluates inside the recurrent loop. Routed experts sort at 64 rows,
//   matching DeepSeek SwitchGLU.

// MARK: - Memory

public enum Glm5NextMemory {
    /// Long-context sparse attention is not validated. The checkpoint
    /// advertises 1,048,576 tokens; the text path stops at this cap.
    public static let validatedContextTokens = 32_768

    public static func isTextModel(_ json: [String: Any]) -> Bool {
        let root = (json["model_type"] as? String)?.lowercased() ?? ""
        if root == "glm5_next" || root == "glm5_next_text" { return true }
        let text = ((json["text_config"] as? [String: Any])?["model_type"] as? String)?.lowercased() ?? ""
        if text == "glm5_next_text" { return true }
        let architectures = json["architectures"] as? [String] ?? []
        return architectures.contains("Glm5NextForConditionalGeneration")
    }

    /// `metadata.total_size` from `model.safetensors.index.json`, when present.
    public static func indexedWeightBytes(at directory: URL) -> UInt64? {
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: index),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let meta = json["metadata"] as? [String: Any]
        else { return nil }
        if let number = meta["total_size"] as? NSNumber {
            let value = number.uint64Value
            return value > 0 ? value : nil
        }
        return nil
    }

    /// 4-bit catalog weights are ~190 GiB and need 256 GB. Hybrid 2-bit
    /// weights above 80 GiB need 128 GB. Smaller fixtures only need the
    /// file size plus 16 GiB.
    public static func requiredBytes(weightBytes: UInt64) -> UInt64 {
        let gib: UInt64 = 1024 * 1024 * 1024
        if weightBytes >= 160 * gib { return 256 * gib }
        if weightBytes >= 80 * gib { return 128 * gib }
        return weightBytes + 16 * gib
    }

    public static func refusalMessage(
        modelId: String,
        weightBytes: UInt64,
        physicalBytes: UInt64
    ) -> String? {
        guard weightBytes > 0, physicalBytes > 0 else { return nil }
        let needed = requiredBytes(weightBytes: weightBytes)
        guard physicalBytes < needed else { return nil }
        return """
        Insufficient unified memory to load \(modelId). This GLM-5.3-Flash \
        text checkpoint is \(gibibytes(weightBytes)) GB on disk and needs about \
        \(gibibytes(needed)) GB of unified memory. This machine has \
        \(gibibytes(physicalBytes)) GB. The mlx-community 4-bit quant (~204 GB) \
        needs 256 GB. The hybrid 2-bit quant (~111 GB) needs 128 GB. Vision \
        weights are not loaded. zai-org FP8 and BF16 are not catalog downloads.
        """
    }

    private static func gibibytes(_ bytes: UInt64) -> Int {
        Int((Double(bytes) / 1_073_741_824).rounded())
    }
}

// MARK: - Configuration

public struct Glm5NextConfiguration: Decodable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var nRoutedExperts: Int
    public var nSharedExperts: Int
    public var numExpertsPerTok: Int
    public var routedScalingFactor: Float
    public var normTopkProb: Bool
    public var nGroup: Int
    public var topkGroup: Int
    public var kvLoraRank: Int
    public var qLoraRank: Int
    public var qkNopeHeadDim: Int
    public var vHeadDim: Int
    public var rmsNormEps: Float
    public var swigluLimit: Float
    public var attentionBias: Bool
    public var tieWordEmbeddings: Bool
    public var layerTypes: [String]
    public var mlpLayerTypes: [String]
    public var indexerTypes: [String]
    public var indexTopk: Int
    public var indexHeadDim: Int
    public var indexNHeads: Int
    public var indexKPool: Int
    public var indexKPoolAlwaysSelectTail: Bool
    public var linearNumHeads: Int
    public var linearHeadDim: Int
    public var linearConvKernel: Int
    public var linearLowerBound: Float
    public var hcMult: Int
    public var hcEps: Float
    public var hcSinkhornIters: Int
    public var firstKDenseReplace: Int

    public var qScale: Float { pow(Float(qkNopeHeadDim), -0.5) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case swigluLimit = "swiglu_limit"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case mlpLayerTypes = "mlp_layer_types"
        case indexerTypes = "indexer_types"
        case indexTopk = "index_topk"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexKPool = "index_kpool"
        case indexKPoolAlwaysSelectTail = "index_kpool_always_select_tail"
        case linearAttnConfig = "linear_attn_config"
        case hcMult = "hc_mult"
        case hcEps = "hc_eps"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case firstKDenseReplace = "first_k_dense_replace"
    }

    private struct LinearAttnJSON: Decodable {
        var numHeads: Int?
        var headDim: Int?
        var kernel: Int?
        var lowerBound: Float?
        enum CodingKeys: String, CodingKey {
            case numHeads = "num_heads"
            case headDim = "head_dim"
            case kernel = "short_conv_kernel_size"
            case lowerBound = "gate_lower_bound"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm5_next_text"
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2048
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 288
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1536
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 256
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 256
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 32
        indexKPool = try c.decodeIfPresent(Int.self, forKey: .indexKPool) ?? 4
        indexKPoolAlwaysSelectTail = try c.decodeIfPresent(Bool.self, forKey: .indexKPoolAlwaysSelectTail) ?? true
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 3

        let linear = try c.decodeIfPresent(LinearAttnJSON.self, forKey: .linearAttnConfig)
        linearNumHeads = linear?.numHeads ?? 64
        linearHeadDim = linear?.headDim ?? 128
        linearConvKernel = linear?.kernel ?? 4
        linearLowerBound = linear?.lowerBound ?? -5

        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types.map { $0 == "full_attention" ? "deepseek_sparse_attention" : $0 }
        } else {
            layerTypes = (0..<numHiddenLayers).map { index in
                index % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention"
            }
        }
        if let mlp = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes) {
            mlpLayerTypes = mlp
        } else {
            let dense = min(firstKDenseReplace, numHiddenLayers)
            mlpLayerTypes = Array(repeating: "dense", count: dense)
                + Array(repeating: "sparse", count: numHiddenLayers - dense)
        }
        if let indexer = try c.decodeIfPresent([String].self, forKey: .indexerTypes) {
            indexerTypes = indexer
        } else {
            indexerTypes = Array(repeating: "full", count: numHiddenLayers)
        }

        if numAttentionHeads != numKeyValueHeads {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "GLM-5-Next requires num_attention_heads == num_key_value_heads"))
        }
        if layerTypes.count != numHiddenLayers || mlpLayerTypes.count != numHiddenLayers
            || indexerTypes.count != numHiddenLayers
        {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "GLM-5-Next layer_types, mlp_layer_types, and indexer_types must match num_hidden_layers"))
        }
        if indexKPool < 1 || indexTopk % indexKPool != 0 {
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "index_topk must be divisible by index_kpool"))
        }
    }

    /// Accepts a `glm5_next` wrapper (`text_config`) or a flat `glm5_next_text` document.
    public static func decode(from data: Data) throws -> Glm5NextConfiguration {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DecodingError.dataCorrupted(.init(
                codingPath: [], debugDescription: "GLM-5-Next config is not a JSON object"))
        }
        let text = (obj["text_config"] as? [String: Any]) ?? obj
        let payload = try JSONSerialization.data(withJSONObject: text)
        return try JSONDecoder().decode(Glm5NextConfiguration.self, from: payload)
    }
}

// MARK: - Small numeric helpers

private func glm5LimitedSwiGLU(_ gate: MLXArray, _ up: MLXArray, limit: Float) -> MLXArray {
    let limitedGate = MLX.minimum(gate, MLXArray(limit))
    let limitedUp = MLX.clip(up, min: MLXArray(-limit), max: MLXArray(limit))
    return MLXNN.silu(limitedGate) * limitedUp
}

private func glm5ZeroNaN(_ x: MLXArray) -> MLXArray {
    MLX.which(x .== x, x, MLXArray(Float(0)))
}

private func glm5Rms(_ x: MLXArray, eps: Float) -> MLXArray {
    let f = x.asType(.float32)
    let scale = MLX.rsqrt(MLX.mean(f * f, axis: -1, keepDims: true) + eps)
    return (f * scale).asType(x.dtype)
}

private func glm5BatchGather(_ values: MLXArray, _ indices: MLXArray) -> MLXArray {
    let batch = values.dim(0)
    let length = max(values.dim(1), 1)
    let rest = Array(values.shape.dropFirst(2))
    let indexShape = indices.shape
    var offsetShape = [batch]
    offsetShape.append(contentsOf: Array(repeating: 1, count: max(indices.ndim - 1, 0)))
    let offsets = MLXArray((0..<batch).map { Int32($0) }).reshaped(offsetShape) * Int32(length)
    let flatIndex = MLX.maximum(indices.asType(.int32) + offsets, MLXArray(Int32(0))).reshaped([-1])
    let flatValues = values.reshaped([batch * length] + rest)
    let gathered = MLX.take(flatValues, flatIndex, axis: 0)
    return gathered.reshaped(indexShape + rest)
}

/// Ops path of mlx-vlm `gated_delta_update` with `gate_lower_bound`.
/// State is float32. `q`, `k`, `v` are `[B, T, H, D]`. `beta` is `[B, T, H]`.
private func glm5GatedDelta(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    beta: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    lowerBound: Float,
    stateIn: MLXArray?
) -> (MLXArray, MLXArray) {
    let batch = q.dim(0)
    let length = q.dim(1)
    let heads = q.dim(2)
    let dim = q.dim(3)
    var state = stateIn ?? MLXArray.zeros([batch, heads, dim, dim], dtype: .float32)
    let aLogB = aLog.asType(.float32).reshaped([1, 1, heads, 1])
    let dtB = dtBias.asType(.float32).reshaped([1, 1, heads, dim])
    let decay = MLX.exp(MLXArray(lowerBound) * sigmoid(MLX.exp(aLogB) * (a.asType(.float32) + dtB)))
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(length)
    for t in 0..<length {
        let gt = decay[0..., t, 0..., 0...]
        state = state * gt.reshaped([batch, heads, 1, dim])
        let kt = k.asType(.float32)[0..., t, 0..., 0...]
        let vt = v.asType(.float32)[0..., t, 0..., 0...]
        let qt = q.asType(.float32)[0..., t, 0..., 0...]
        let kvMem = (state * kt.reshaped([batch, heads, 1, dim])).sum(axis: -1)
        let betaT = beta.asType(.float32)[0..., t, 0...].reshaped([batch, heads, 1])
        let delta = (vt - kvMem) * betaT
        state = state + kt.reshaped([batch, heads, 1, dim]) * delta.reshaped([batch, heads, dim, 1])
        let y = (state * qt.reshaped([batch, heads, 1, dim])).sum(axis: -1)
        // Prefill keeps the recurrent graph from spanning the whole prompt.
        // One-token decode leaves this lazy; `Glm5TextModel` evaluates every
        // `decodeEvalStride` layers instead of syncing inside each KDA step.
        if length > 1 {
            MLX.eval(state, y)
        }
        outputs.append(y)
    }
    let stacked = MLX.stacked(outputs, axis: 1).asType(q.dtype)
    return (stacked, state)
}

private func glm5HC(
    _ x: MLXArray,
    hc: DeepseekV4HyperConnection,
    norm: RMSNorm,
    config: Glm5NextConfiguration,
    _ branch: (MLXArray) -> MLXArray
) -> MLXArray {
    let (pre, post, comb) = hcMixes(
        x: x, fn: hc.fn, scale: hc.scale, base: hc.base,
        hcMult: config.hcMult, nIters: config.hcSinkhornIters,
        eps: config.hcEps, normEps: config.rmsNormEps)
    let collapsed = rmsNormF32(norm, hcReduce(x, pre: pre))
    let out = branch(collapsed)
    return hcPost(x: out, residual: x, post: post, comb: comb)
}

// MARK: - MultiLinear (absorbed MLA). Local: MLXLLM's copy is internal.

final class Glm5MultiLinear: Module, Quantizable {
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numHeads = numHeads
        let scale = sqrt(1 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [numHeads, outputDims, inputDims])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x.matmul(weight.swappedAxes(-1, -2))
    }

    func callNoTranspose(_ x: MLXArray) -> MLXArray {
        x.matmul(weight)
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        let (qWeight, scales, biases) = MLX.quantized(weight, groupSize: groupSize, bits: bits, mode: mode)
        return Glm5QuantizedMultiLinear(
            weight: qWeight, scales: scales, biases: biases,
            inputDims: inputDims, outputDims: outputDims, numHeads: numHeads,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

final class Glm5QuantizedMultiLinear: Module, Quantized {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let inputDims: Int
    let outputDims: Int
    let numHeads: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?

    init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray?,
        inputDims: Int, outputDims: Int, numHeads: Int,
        groupSize: Int, bits: Int, mode: QuantizationMode
    ) {
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
        MLX.quantizedMM(
            x, weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits, mode: mode)
    }

    func callNoTranspose(_ x: MLXArray) -> MLXArray {
        MLX.quantizedMM(
            x, weight, scales: scales, biases: biases,
            transpose: false, groupSize: groupSize, bits: bits, mode: mode)
    }
}

private func glm5MultiLinear(_ module: Module, _ x: MLXArray, transpose: Bool) -> MLXArray {
    if let linear = module as? Glm5MultiLinear {
        return transpose ? linear(x) : linear.callNoTranspose(x)
    }
    if let linear = module as? Glm5QuantizedMultiLinear {
        return transpose ? linear(x) : linear.callNoTranspose(x)
    }
    fatalError("embed_q/unembed_out must be Glm5MultiLinear")
}

// MARK: - MLP / MoE

final class Glm5MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float

    init(hiddenSize: Int, intermediateSize: Int, swigluLimit: Float) {
        self.swigluLimit = swigluLimit
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(glm5LimitedSwiGLU(gateProj(x), upProj(x), limit: swigluLimit))
    }
}

final class Glm5MoEGate: Module {
    @ParameterInfo var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    init(_ config: Glm5NextConfiguration) {
        topK = config.numExpertsPerTok
        nGroup = config.nGroup
        topkGroup = config.topkGroup
        routedScalingFactor = config.routedScalingFactor
        normTopkProb = config.normTopkProb
        _weight.wrappedValue = MLXArray.zeros(
            [config.nRoutedExperts, config.hiddenSize], dtype: .float32)
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros(
            [config.nRoutedExperts], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.asType(.float32).matmul(weight.asType(.float32).T)
        let scores = sigmoid(logits)
        var choice = scores + eScoreCorrectionBias
        if nGroup > 1 {
            let expertsPerGroup = choice.dim(-1) / nGroup
            let grouped = choice.reshaped([choice.dim(0), nGroup, expertsPerGroup])
            let groupScores = MLX.top(grouped, k: 2, axis: -1).sum(axis: -1)
            let dropCount = nGroup - topkGroup
            if dropCount > 0 {
                let drop = argPartition(groupScores, kth: dropCount - 1, axis: -1)[.ellipsis, ..<dropCount]
                var mask = MLXArray.ones(groupScores.shape, dtype: .bool)
                mask = putAlong(mask, stopGradient(drop), values: MLXArray(false), axis: -1)
                let repeated = MLX.repeated(
                    mask.expandedDimensions(axis: -1), count: expertsPerGroup, axis: -1)
                choice = MLX.which(
                    repeated.reshaped(choice.shape),
                    choice,
                    MLXArray(-Float.infinity))
            }
        }
        let indices = stopGradient(
            argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK].asType(.int32))
        var weights = takeAlong(scores, indices, axis: -1)
        if topK > 1 && normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        return (indices, weights * routedScalingFactor)
    }
}

final class Glm5SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear
    let swigluLimit: Float

    init(_ config: Glm5NextConfiguration) {
        swigluLimit = config.swigluLimit
        let experts = config.nRoutedExperts
        let hidden = config.hiddenSize
        let inner = config.moeIntermediateSize
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: hidden, outputDims: inner, numExperts: experts, bias: false)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: hidden, outputDims: inner, numExperts: experts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: inner, outputDims: hidden, numExperts: experts, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let xExp = MLX.expandedDimensions(x, axes: [-2, -3])
        // Same threshold as DeepseekV4SwitchGLU. One token × 8 experts stays
        // unsorted. A prefill row count at or above 64 sorts once and reuses
        // that order for gate, up, and down.
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        var xIn = xExp
        if doSort {
            (xIn, idx, inverseOrder) = gatherSort(x: xExp, indices: indices)
        }
        let gate = gateProj(xIn, idx, sortedIndices: doSort)
        let up = upProj(xIn, idx, sortedIndices: doSort)
        let mid = glm5LimitedSwiGLU(gate, up, limit: swigluLimit)
        var out = downProj(mid, idx, sortedIndices: doSort)
        if doSort {
            out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
        }
        let squeezed = MLX.squeezed(out, axis: -2)
        return (squeezed * weights.asType(squeezed.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

final class Glm5MoE: Module {
    @ModuleInfo(key: "gate") var gate: Glm5MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: Glm5SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Glm5MLP

    init(_ config: Glm5NextConfiguration) {
        _gate.wrappedValue = Glm5MoEGate(config)
        _switchMLP.wrappedValue = Glm5SwitchGLU(config)
        _sharedExperts.wrappedValue = Glm5MLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize * config.nSharedExperts,
            swigluLimit: config.swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, dim) = (x.dim(0), x.dim(1), x.dim(2))
        let flat = x.reshaped([-1, dim])
        let (indices, weights) = gate(flat)
        let routed = switchMLP(flat, indices: indices, weights: weights).reshaped(batch, length, dim)
        return routed + sharedExperts(x)
    }
}

// MARK: - Linear attention

final class Glm5LinearAttention: Module {
    let numHeads: Int
    let headDim: Int
    let convKernel: Int
    let lowerBound: Float
    let projectionDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_conv1d") var qConv: Conv1d
    @ModuleInfo(key: "k_conv1d") var kConv: Conv1d
    @ModuleInfo(key: "v_conv1d") var vConv: Conv1d
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: Glm5NextConfiguration) {
        numHeads = config.linearNumHeads
        headDim = config.linearHeadDim
        convKernel = config.linearConvKernel
        lowerBound = config.linearLowerBound
        projectionDim = numHeads * headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, projectionDim, bias: false)
        _qConv.wrappedValue = Conv1d(
            inputChannels: projectionDim, outputChannels: projectionDim,
            kernelSize: convKernel, groups: projectionDim, bias: false)
        _kConv.wrappedValue = Conv1d(
            inputChannels: projectionDim, outputChannels: projectionDim,
            kernelSize: convKernel, groups: projectionDim, bias: false)
        _vConv.wrappedValue = Conv1d(
            inputChannels: projectionDim, outputChannels: projectionDim,
            kernelSize: convKernel, groups: projectionDim, bias: false)
        _fAProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _bProj.wrappedValue = Linear(config.hiddenSize, numHeads, bias: false)
        _gAProj.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _aLog.wrappedValue = MLXArray.zeros([numHeads], dtype: .float32)
        _dtBias.wrappedValue = MLXArray.zeros([projectionDim], dtype: .float32)
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionDim, config.hiddenSize, bias: false)
        super.init()
    }

    private func causalConv(_ conv: Conv1d, _ x: MLXArray, cache: MLXArray?) -> (MLXArray, MLXArray) {
        let batch = x.dim(0)
        let channels = x.dim(-1)
        let history = cache ?? MLXArray.zeros([batch, max(convKernel - 1, 0), channels], dtype: x.dtype)
        let convInput = history.dim(1) > 0 ? concatenated([history, x], axis: 1) : x
        let end = convInput.dim(1)
        let start = max(0, end - max(convKernel - 1, 0))
        let next = convKernel > 1 ? convInput[0..., start..<end, 0...] : history
        return (silu(conv(convInput)), next)
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))
        // Slot 0 is the depthwise conv window (q, k, v concatenated on the
        // channel axis). Slot 1 is the float32 gated-delta state. They must
        // stay separate: the recurrent state is not a conv history.
        let packed = cache?[0]
        let qHist = packed.map { $0[0..., 0..., 0..<projectionDim] }
        let kHist = packed.map { $0[0..., 0..., projectionDim..<(2 * projectionDim)] }
        let vHist = packed.map { $0[0..., 0..., (2 * projectionDim)..<(3 * projectionDim)] }
        let (qConvOut, qState) = causalConv(qConv, qProj(x), cache: qHist)
        let (kConvOut, kState) = causalConv(kConv, kProj(x), cache: kHist)
        let (vConvOut, vState) = causalConv(vConv, vProj(x), cache: vHist)
        if let cache {
            cache[0] = concatenated([qState, kState, vState], axis: -1)
        }
        let shape = [batch, length, numHeads, headDim]
        let qRaw = qConvOut.reshaped(shape)
        let kRaw = kConvOut.reshaped(shape)
        let vRaw = vConvOut.reshaped(shape)
        let eps = (1e-6) / Float(headDim)
        let q = (scale * scale) * glm5Rms(qRaw, eps: eps)
        let k = scale * glm5Rms(kRaw, eps: eps)
        let fA = fAProj(x)
        let b = bProj(x)
        let gA = gAProj(x)
        let a = fBProj(fA).reshaped(shape)
        let (output, nextState) = glm5GatedDelta(
            q: q, k: k, v: vRaw, a: a, beta: sigmoid(b),
            aLog: aLog.reshaped([numHeads, 1]),
            dtBias: dtBias.reshaped([numHeads, headDim]),
            lowerBound: lowerBound,
            stateIn: cache?[1])
        if let cache { cache[1] = nextState }
        let gate = sigmoid(gBProj(gA).reshaped(shape))
        let normed = oNorm(output) * gate
        return oProj(normed.reshaped([batch, length, projectionDim]))
    }
}

// MARK: - Sparse attention

private func glm5SparseAttend(
    q: MLXArray, k: MLXArray, v: MLXArray, indices: MLXArray, scale: Float
) -> MLXArray {
    let length = q.dim(2)
    let kvLength = k.dim(2)
    let chunk = 16
    var outputs: [MLXArray] = []
    var start = 0
    while start < length {
        let end = min(start + chunk, length)
        let idx = indices[0..., start..<end, 0...]
        let valid = (idx .>= 0) .&& (idx .< kvLength)
        let safe = MLX.clip(idx, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(kvLength - 1, 0))))
        let selectedK = glm5HeadGather(k, safe)
        let selectedV = glm5HeadGather(v, safe)
        let query = q[0..., 0..., start..<end, 0...].asType(.float32)
        var scores = (query.expandedDimensions(axis: 3) * selectedK.asType(.float32)).sum(axis: -1) * scale
        scores = MLX.which(valid.expandedDimensions(axis: 1), scores, MLXArray(-Float.infinity))
        var weights = MLX.softmax(scores, axis: -1)
        weights = glm5ZeroNaN(weights)
        let out = (weights.expandedDimensions(axis: -1) * selectedV.asType(.float32)).sum(axis: 3)
        let cast = out.asType(q.dtype)
        if length > 1 {
            MLX.eval(cast)
        }
        outputs.append(cast)
        start = end
    }
    return concatenated(outputs, axis: 2)
}

/// `values` is `[B, H, T, D]`, `indices` is `[B, L, K]` → `[B, H, L, K, D]`.
private func glm5HeadGather(_ values: MLXArray, _ indices: MLXArray) -> MLXArray {
    let batch = values.dim(0)
    let heads = values.dim(1)
    let kvLength = values.dim(2)
    let dim = values.dim(3)
    let queryLength = indices.dim(1)
    let topk = indices.dim(2)
    let flat = values.reshaped([batch * heads * max(kvLength, 1), dim])
    let batchOff = MLXArray((0..<batch).map { Int32($0 * heads * kvLength) }).reshaped([batch, 1, 1, 1])
    let headOff = MLXArray((0..<heads).map { Int32($0 * kvLength) }).reshaped([1, heads, 1, 1])
    let flatIndex = (batchOff + headOff + indices.reshaped([batch, 1, queryLength, topk])).reshaped([-1])
    let gathered = MLX.take(flat, MLX.maximum(flatIndex, MLXArray(Int32(0))), axis: 0)
    return gathered.reshaped([batch, heads, queryLength, topk, dim])
}

final class Glm5Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let indexTopk: Int
    let indexKPool: Int
    let alwaysSelectTail: Bool
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ParameterInfo(key: "index_kpool_compress_ape") var compressApe: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var compressGate: MLXArray

    init(_ config: Glm5NextConfiguration) {
        nHeads = config.indexNHeads
        headDim = config.indexHeadDim
        indexTopk = config.indexTopk
        indexKPool = config.indexKPool
        alwaysSelectTail = config.indexKPoolAlwaysSelectTail
        softmaxScale = pow(Float(headDim), -0.5)
        _wqB.wrappedValue = Linear(config.qLoraRank, nHeads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, nHeads, bias: false)
        _compressApe.wrappedValue = MLXArray.zeros([indexKPool, headDim], dtype: .float32)
        _compressGate.wrappedValue = MLXArray.zeros([headDim, config.hiddenSize], dtype: .float32)
        super.init()
    }

    func project(x: MLXArray) -> (MLXArray, MLXArray) {
        let keys = kNorm(wk(x))
        let gate = x.asType(.float32).matmul(compressGate.T).asType(keys.dtype)
        return (keys, gate)
    }

    /// Full-sequence pool selection. `keys` / `gates` already include history.
    func select(x: MLXArray, qResid: MLXArray, keys: MLXArray, gates: MLXArray, queryLength: Int) -> MLXArray {
        let batch = keys.dim(0)
        let kvLength = keys.dim(1)
        let valid = MLXArray.ones([batch, kvLength], dtype: .bool)
        let (poolKeys, poolIndices, poolValid, keyValid) = pooledStates(keys: keys, gates: gates, valid: valid)
        let queryPositions = MLXArray((0..<queryLength).map { Int32(kvLength - queryLength + $0) })
        let poolCount = poolIndices.dim(1)
        let width = indexTopk + (alwaysSelectTail && indexKPool > 1 ? indexKPool - 1 : 0)
        if poolCount == 0 || kvLength == 0 {
            return MLXArray.zeros([batch, queryLength, width], dtype: .int32) - 1
        }
        let selectK = min(indexTopk / indexKPool, poolCount)
        let poolEnd = MLX.clip(
            poolIndices[.ellipsis, -1],
            min: MLXArray(Int32(0)),
            max: MLXArray(Int32(max(kvLength - 1, 0))))
        let ends = poolEnd.expandedDimensions(axis: 1)
        let positions = queryPositions.reshaped([1, queryLength, 1])
        let candidates = (poolValid.asType(.int32).expandedDimensions(axis: 1)
            * (ends .<= positions).asType(.int32)) .== MLXArray(Int32(1))
        let selected: MLXArray
        let selectedValid: MLXArray
        if selectK == poolCount {
            selected = MLX.broadcast(
                MLXArray((0..<poolCount).map { Int32($0) }).reshaped([1, 1, poolCount]),
                to: [batch, queryLength, poolCount])
            selectedValid = candidates
        } else {
            let q = wqB(qResid).reshaped([batch, x.dim(1), nHeads, headDim])
            let qChunk = q[0..., (x.dim(1) - queryLength)..<x.dim(1), 0..., 0...]
            let weights = weightsProj(x[0..., (x.dim(1) - queryLength)..<x.dim(1), 0...])
                .asType(.float32) * pow(Float(nHeads), -0.5)
            let scores = scorePools(q: qChunk, poolKeys: poolKeys, weights: weights)
            let masked = MLX.which(candidates, scores, MLXArray(-Float.infinity))
            let picked = stopGradient(
                argPartition(-masked, kth: selectK - 1, axis: -1)[.ellipsis, ..<selectK].asType(.int32))
            selected = picked
            selectedValid = takeAlong(candidates, picked, axis: -1)
        }
        let orderKey = MLX.which(selectedValid, selected, MLXArray(Int32(poolCount)))
        let order = argSort(orderKey, axis: -1)
        let ordered = takeAlong(selected, order, axis: -1)
        let orderedValid = takeAlong(selectedValid, order, axis: -1)
        let safe = MLX.clip(ordered, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(poolCount - 1, 0))))
        let expanded = MLX.broadcast(
            poolIndices.expandedDimensions(axis: 1),
            to: [batch, queryLength, poolCount, indexKPool])
        let gathered = takeAlong(expanded, safe.expandedDimensions(axis: -1), axis: 2)
        var topk = gathered.reshaped([batch, queryLength, selectK * indexKPool])
        let validRepeated = MLX.repeated(
            orderedValid.expandedDimensions(axis: -1), count: indexKPool, axis: -1
        ).reshaped(topk.shape)
        topk = MLX.which(validRepeated, topk, MLXArray(Int32(-1)))
        if topk.dim(-1) < indexTopk {
            let pad = indexTopk - topk.dim(-1)
            topk = MLX.padded(
                topk,
                widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, pad))],
                value: MLXArray(Int32(-1)))
        }
        topk = topk[.ellipsis, ..<indexTopk]
        if alwaysSelectTail && indexKPool > 1 {
            topk = concatenated([topk, tailIndices(
                keyValid: keyValid, queryPositions: queryPositions, kvLength: kvLength
            )], axis: -1)
        }
        return topk.asType(.int32)
    }

    private func scorePools(q: MLXArray, poolKeys: MLXArray, weights: MLXArray) -> MLXArray {
        let keysT = poolKeys.asType(.float32).expandedDimensions(axis: 1).swappedAxes(-1, -2)
        var scores = q.asType(.float32).matmul(keysT)
        scores = MLX.maximum(scores, MLXArray(Float(0)))
        let scaled = (weights.asType(.float32) * softmaxScale).expandedDimensions(axis: -1)
        return (scores * scaled).sum(axis: 2)
    }

    private func pooledStates(
        keys: MLXArray, gates: MLXArray, valid: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let batch = valid.dim(0)
        let length = valid.dim(1)
        let poolCount = (length + indexKPool - 1) / indexKPool
        let hasKey = valid.asType(.int32).sum(axis: -1).reshaped([batch]) .> MLXArray(Int32(0))
        let first = MLX.which(
            hasKey,
            argMax(valid.asType(.int32), axis: -1).reshaped([batch]),
            MLX.broadcast(MLXArray(Int32(length)).reshaped([1]), to: [batch]))
        let offsets = MLXArray((0..<(poolCount * indexKPool)).map { Int32($0) })
            .reshaped([1, poolCount, indexKPool])
        let indices = first.reshaped([batch, 1, 1]) + offsets
        let safe = MLX.clip(indices, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(length - 1, 0))))
        let groupedKeys = glm5BatchGather(keys, safe)
        let groupedGates = glm5BatchGather(gates, safe)
        var groupedValid = glm5BatchGather(valid.expandedDimensions(axis: -1).asType(keys.dtype), safe)
        groupedValid = MLX.squeezed(groupedValid, axis: -1)
        let inRange = indices .< length
        let groupedBool = (groupedValid .> MLXArray(Float(0))) .&& inRange
        let poolValid = groupedBool.asType(.int32).sum(axis: -1) .== MLXArray(Int32(indexKPool))
        let stored = MLX.which(groupedBool, indices, MLXArray(Int32(-1)))
        var logits = groupedGates.asType(.float32) + compressApe.reshaped([1, 1, indexKPool, headDim])
        logits = MLX.which(groupedBool.expandedDimensions(axis: -1), logits, MLXArray(-Float.infinity))
        var probs = MLX.softmax(logits, axis: 2)
        probs = glm5ZeroNaN(probs)
        let poolKeys = (probs.asType(groupedKeys.dtype) * groupedKeys).sum(axis: 2)
        return (poolKeys, stored, poolValid, valid)
    }

    private func tailIndices(keyValid: MLXArray, queryPositions: MLXArray, kvLength: Int) -> MLXArray {
        let batch = keyValid.dim(0)
        let queryLength = queryPositions.dim(0)
        let tailWidth = indexKPool - 1
        let prefix = MLX.cumsum(keyValid.asType(.int32), axis: -1)
        let safePos = MLX.clip(
            queryPositions, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(kvLength - 1, 0))))
        let positions = MLX.broadcast(safePos.reshaped([1, queryLength]), to: [batch, queryLength])
        let visible = takeAlong(prefix, positions, axis: -1)
        let kpool = MLXArray(Int32(indexKPool))
        let tailCount = visible - (visible / kpool) * kpool
        let hasKey = keyValid.asType(.int32).sum(axis: -1).reshaped([batch]) .> MLXArray(Int32(0))
        let first = MLX.which(
            hasKey,
            argMax(keyValid.asType(.int32), axis: -1).reshaped([batch]),
            MLX.broadcast(MLXArray(Int32(kvLength)).reshaped([1]), to: [batch]))
        let tailStart = first.reshaped([batch, 1]) + visible - tailCount
        let tail = tailStart.expandedDimensions(axis: -1)
            + MLXArray((0..<tailWidth).map { Int32($0) }).reshaped([1, 1, tailWidth])
        let ordinal = MLXArray((0..<tailWidth).map { Int32($0) }).reshaped([1, 1, tailWidth])
        var tailValid = (ordinal .< tailCount.expandedDimensions(axis: -1)) .&& (tail .< kvLength)
        let safeTail = MLX.clip(tail, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(kvLength - 1, 0))))
        let atKey = glm5BatchGather(keyValid.expandedDimensions(axis: -1).asType(.float32), safeTail)
        let keyBool = MLX.squeezed(atKey, axis: -1) .> MLXArray(Float(0))
        tailValid = tailValid .&& keyBool
        return MLX.which(tailValid, tail, MLXArray(Int32(-1)))
    }
}

final class Glm5SparseAttention: Module {
    let numHeads: Int
    let qLoraRank: Int
    let qHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let scale: Float
    let absorbedLimit: Int

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: Glm5Indexer?

    init(_ config: Glm5NextConfiguration, layerIndex: Int) {
        numHeads = config.numAttentionHeads
        qLoraRank = config.qLoraRank
        qHeadDim = config.qkNopeHeadDim
        kvLoraRank = config.kvLoraRank
        vHeadDim = config.vHeadDim
        scale = pow(Float(qHeadDim), -0.5)
        absorbedLimit = 8
        _qAProj.wrappedValue = Linear(
            config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        _kvAProj.wrappedValue = Linear(
            config.hiddenSize, config.kvLoraRank, bias: config.attentionBias)
        _qANorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        _kvANorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _embedQ.wrappedValue = Glm5MultiLinear(
            inputDims: qHeadDim, outputDims: kvLoraRank, numHeads: numHeads)
        _unembedOut.wrappedValue = Glm5MultiLinear(
            inputDims: kvLoraRank, outputDims: vHeadDim, numHeads: numHeads)
        _oProj.wrappedValue = Linear(
            numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)
        if config.indexerTypes[layerIndex] != "shared" {
            _indexer.wrappedValue = Glm5Indexer(config)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: CacheList?, previousTopk: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))
        let qResid = qANorm(qAProj(x))
        let q = qBProj(qResid).reshaped([batch, length, numHeads, qHeadDim]).transposed(0, 2, 1, 3)
        let newLatent = MLX.expandedDimensions(kvANorm(kvAProj(x)), axis: 1)
        let kvCache = cache?[0]
        let history = cache?[1] as? ArraysCache
        let latent: MLXArray
        if let kvCache {
            let dummy = MLXArray.zeros([batch, 1, length, 1], dtype: newLatent.dtype)
            latent = kvCache.update(keys: newLatent, values: dummy).0
        } else {
            latent = newLatent
        }

        let topk: MLXArray
        if let indexer {
            let (newKeys, newGates) = indexer.project(x: x)
            let keys: MLXArray
            let gates: MLXArray
            if let previousKeys = history?[0], let previousGates = history?[1] {
                keys = concatenated([previousKeys, newKeys], axis: 1)
                gates = concatenated([previousGates, newGates], axis: 1)
            } else {
                keys = newKeys
                gates = newGates
            }
            if let history {
                history[0] = keys
                history[1] = gates
            }
            topk = indexer.select(x: x, qResid: qResid, keys: keys, gates: gates, queryLength: length)
        } else if let previousTopk {
            topk = previousTopk
        } else {
            topk = MLXArray.zeros([batch, length, 0], dtype: .int32) - 1
        }

        let out: MLXArray
        if length <= absorbedLimit {
            out = absorbed(q: q, latent: latent, topk: topk)
        } else {
            let keys = glm5MultiLinear(embedQ, latent, transpose: false)
            let values = glm5MultiLinear(unembedOut, latent, transpose: true)
            let attended = glm5SparseAttend(q: q, k: keys, v: values, indices: topk, scale: scale)
            out = attended.transposed(0, 2, 1, 3).reshaped([batch, length, -1])
        }
        return (oProj(out), topk)
    }

    private func absorbed(q: MLXArray, latent: MLXArray, topk: MLXArray) -> MLXArray {
        let batch = q.dim(0)
        let length = q.dim(2)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(length)
        let kvLength = latent.dim(2)
        for index in 0..<length {
            let idx = topk[0..., index..<(index + 1), 0...]
            let valid = (idx .>= 0) .&& (idx .< kvLength)
            let safe = MLX.clip(idx, min: MLXArray(Int32(0)), max: MLXArray(Int32(max(kvLength - 1, 0))))
            let flat = safe.reshaped([batch, safe.dim(-1)])
            let selected = glm5BatchGather(latent.squeezed(axis: 1), flat).reshaped([batch, 1, flat.dim(1), latent.dim(-1)])
            let query = glm5MultiLinear(embedQ, q[0..., 0..., index..<(index + 1), 0...], transpose: true)
            let qf = query.asType(.float32).expandedDimensions(axis: 3)
            let sf = selected.asType(.float32).expandedDimensions(axis: 1)
            var scores = (qf * sf).sum(axis: -1) * scale
            scores = MLX.which(valid.expandedDimensions(axis: 1), scores, MLXArray(-Float.infinity))
            var weights = MLX.softmax(scores, axis: -1)
            weights = glm5ZeroNaN(weights)
            let mixed = (weights.expandedDimensions(axis: -1) * sf).sum(axis: 3).asType(query.dtype)
            let token = glm5MultiLinear(unembedOut, mixed, transpose: true)
            if length > 1 {
                MLX.eval(token)
            }
            outputs.append(token)
        }
        let attended = concatenated(outputs, axis: 2)
        return attended.transposed(0, 2, 1, 3).reshaped([batch, length, -1])
    }
}

// MARK: - Block

final class Glm5Block: Module {
    let isLinear: Bool
    let config: Glm5NextConfiguration
    @ModuleInfo(key: "self_attn") var selfAttn: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV4HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV4HyperConnection

    init(_ config: Glm5NextConfiguration, layerIndex: Int) {
        self.config = config
        isLinear = config.layerTypes[layerIndex] == "linear_attention"
        if isLinear {
            _selfAttn.wrappedValue = Glm5LinearAttention(config)
        } else {
            _selfAttn.wrappedValue = Glm5SparseAttention(config, layerIndex: layerIndex)
        }
        if config.mlpLayerTypes[layerIndex] == "sparse" {
            _mlp.wrappedValue = Glm5MoE(config)
        } else {
            _mlp.wrappedValue = Glm5MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
                swigluLimit: config.swigluLimit)
        }
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnHC.wrappedValue = DeepseekV4HyperConnection(
            hcMult: config.hcMult, hiddenSize: config.hiddenSize)
        _ffnHC.wrappedValue = DeepseekV4HyperConnection(
            hcMult: config.hcMult, hiddenSize: config.hiddenSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: KVCache?, previousTopk: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let afterAttn: MLXArray
        let topk: MLXArray?
        if let linear = selfAttn as? Glm5LinearAttention {
            afterAttn = glm5HC(x, hc: attnHC, norm: inputNorm, config: config) { hidden in
                linear(hidden, cache: cache as? ArraysCache)
            }
            topk = previousTopk
        } else if let sparse = selfAttn as? Glm5SparseAttention {
            var produced: MLXArray?
            afterAttn = glm5HC(x, hc: attnHC, norm: inputNorm, config: config) { hidden in
                let (out, next) = sparse(hidden, cache: cache as? CacheList, previousTopk: previousTopk)
                produced = next
                return out
            }
            topk = produced
        } else {
            afterAttn = x
            topk = previousTopk
        }
        let afterMLP = glm5HC(afterAttn, hc: ffnHC, norm: postNorm, config: config) { hidden in
            if let moe = mlp as? Glm5MoE { return moe(hidden) }
            if let dense = mlp as? Glm5MLP { return dense(hidden) }
            return hidden
        }
        return (afterMLP, topk)
    }
}

final class Glm5TextModel: Module {
    let config: Glm5NextConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [Glm5Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Glm5NextConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = (0..<config.numHiddenLayers).map { Glm5Block(config, layerIndex: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        hidden = MLX.repeated(
            MLX.expandedDimensions(hidden, axis: 2), count: config.hcMult, axis: 2)
        let decoding = inputs.dim(1) == 1
        var topk: MLXArray?
        for (index, layer) in layers.enumerated() {
            (hidden, topk) = layer(hidden, cache: cache?[index], previousTopk: topk)
            if decoding, (index + 1) % Glm5ChatDecode.decodeEvalStride == 0 {
                MLX.eval(hidden)
            }
        }
        let collapsed = hidden.mean(axis: 2).asType(hidden.dtype)
        return norm(collapsed)
    }
}

final class Glm5LanguageModel: Module {
    let config: Glm5NextConfiguration
    @ModuleInfo(key: "model") var model: Glm5TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Glm5NextConfiguration) {
        self.config = config
        _model.wrappedValue = Glm5TextModel(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}

// MARK: - Top-level model

public final class Glm5NextModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    /// `mtp.*` and `layers.{num_hidden_layers}` keys dropped by the last sanitize.
    /// Zero means the checkpoint had no next-token layer. Those weights are not
    /// a drafter: see `Glm5ChatDecode`.
    public private(set) var droppedMtpKeys: Int = 0
    let configuration: Glm5NextConfiguration
    @ModuleInfo(key: "language_model") var languageModel: Glm5LanguageModel

    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ configuration: Glm5NextConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabSize
        kvHeads = configuration.layerTypes.map { $0 == "deepseek_sparse_attention" ? 1 : 0 }
        _languageModel.wrappedValue = Glm5LanguageModel(configuration)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        configuration.layerTypes.map { kind in
            if kind == "linear_attention" {
                return ArraysCache(size: 2)
            }
            return CacheList(KVCacheSimple(), ArraysCache(size: 2))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let mapped = Glm5NextSanitizer.remap(weights, config: configuration)
        droppedMtpKeys = mapped.droppedMtpKeys
        return mapped.weights
    }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            let keep = key.hasSuffix("A_log")
                || key.hasSuffix("dt_bias")
                || key.hasSuffix("e_score_correction_bias")
                || key.contains(".attn_hc.")
                || key.contains(".ffn_hc.")
                || key.contains("index_kpool_compress")
            return !keep
        }
    }
}

// MARK: - Weight remap

enum Glm5NextSanitizer {
    static func remap(
        _ weights: [String: MLXArray], config: Glm5NextConfiguration
    ) -> (weights: [String: MLXArray], droppedMtpKeys: Int) {
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(weights.count)
        var droppedMtpKeys = 0
        for (rawKey, rawValue) in weights {
            if shouldDrop(rawKey, layerCount: config.numHiddenLayers) {
                if isMtpKey(rawKey, layerCount: config.numHiddenLayers) {
                    droppedMtpKeys += 1
                }
                continue
            }
            var key = rewritePrefix(rawKey)
            key = renameHyperConnection(key)
            var value = rawValue
            if key.contains("conv1d.weight"), value.ndim == 3,
               value.dim(2) == config.linearConvKernel, value.dim(1) != config.linearConvKernel
            {
                value = value.swappedAxes(1, 2)
            }
            mapped[key] = value
        }
        stackExperts(&mapped, config: config)
        splitKVB(&mapped, config: config)
        return (mapped, droppedMtpKeys)
    }

    private static func isMtpKey(_ key: String, layerCount: Int) -> Bool {
        if key.contains("mtp.") { return true }
        if let index = layerIndex(key), index >= layerCount { return true }
        return false
    }

    private static func shouldDrop(_ key: String, layerCount: Int) -> Bool {
        if key.contains("model.visual") || key.hasPrefix("vision") || key.contains("vision_tower") {
            return true
        }
        if key.contains("mtp.") { return true }
        if let index = layerIndex(key), index >= layerCount { return true }
        return false
    }

    private static func layerIndex(_ key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }

    private static func rewritePrefix(_ key: String) -> String {
        if key.hasPrefix("model.language_model.") {
            return "language_model.model." + key.dropFirst("model.language_model.".count)
        }
        if key.hasPrefix("model.layers.") {
            return "language_model.model.layers." + key.dropFirst("model.layers.".count)
        }
        if key.hasPrefix("lm_head.") {
            return "language_model." + key
        }
        return key
    }

    private static func renameHyperConnection(_ key: String) -> String {
        var key = key
        for site in ["attn", "ffn"] {
            for name in ["fn", "base", "scale"] {
                key = key.replacingOccurrences(of: ".hc_\(site)_\(name)", with: ".\(site)_hc.\(name)")
            }
        }
        return key
    }

    private static func stackExperts(
        _ weights: inout [String: MLXArray], config: Glm5NextConfiguration
    ) {
        for layer in 0..<config.numHiddenLayers where config.mlpLayerTypes[layer] == "sparse" {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases", "bias"] {
                    guard weights["\(prefix).experts.0.\(projection).\(suffix)"] != nil else { continue }
                    var parts: [MLXArray] = []
                    parts.reserveCapacity(config.nRoutedExperts)
                    var complete = true
                    for expert in 0..<config.nRoutedExperts {
                        let key = "\(prefix).experts.\(expert).\(projection).\(suffix)"
                        guard let value = weights[key] else {
                            complete = false
                            break
                        }
                        parts.append(value)
                    }
                    guard complete, parts.count == config.nRoutedExperts else { continue }
                    for expert in 0..<config.nRoutedExperts {
                        weights.removeValue(forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)")
                    }
                    weights["\(prefix).switch_mlp.\(projection).\(suffix)"] = MLX.stacked(parts)
                }
            }
        }
    }

    private static func splitKVB(
        _ weights: inout [String: MLXArray], config: Glm5NextConfiguration
    ) {
        for layer in 0..<config.numHiddenLayers
        where config.layerTypes[layer] == "deepseek_sparse_attention" {
            let prefix = "language_model.model.layers.\(layer).self_attn"
            let key = "\(prefix).kv_b_proj.weight"
            guard var value = weights.removeValue(forKey: key) else { continue }
            if let scales = weights.removeValue(forKey: "\(prefix).kv_b_proj.scales") {
                let biases = weights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
                let bits = value.dim(-1) * 32 / max(config.kvLoraRank, 1)
                let groupSize = max(1, config.kvLoraRank / max(scales.dim(-1), 1))
                let mode: QuantizationMode = (biases == nil && bits == 8) ? .mxfp8 : .affine
                value = MLX.dequantized(
                    value, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: mode)
            }
            let headSpan = config.qkNopeHeadDim + config.vHeadDim
            let reshaped = value.reshaped([config.numAttentionHeads, headSpan, -1])
            let wk = reshaped[.ellipsis, ..<config.qkNopeHeadDim, 0...].swappedAxes(-1, -2)
            let wv = reshaped[.ellipsis, config.qkNopeHeadDim..., 0...]
            weights["\(prefix).embed_q.weight"] = wk
            weights["\(prefix).unembed_out.weight"] = wv
        }
    }
}
