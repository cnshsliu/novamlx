import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Text-only MiniMax-M3.1-Flash (`model_type: minimax_m3_vl`).
//
// The mlx-community text checkpoints drop the vision tower and keep the
// language stack mlx-lm loads in `mlx_lm/models/minimax_m3_vl.py`
// (ml-explore/mlx-lm#2003). Weight names after sanitize are
// `model.model.layers.*` and `model.lm_head`, which is what
// `ExpertShardConverter.swiftTensorName` writes (`language_model.` → `model.`).
//
// MiniMax Sparse Attention is the functional block mask from that file:
// an indexer scores keys, max-pools them into blocks, and keeps the top
// blocks. When the top-k budget already covers every block (contexts up to
// `sparse_topk_blocks * sparse_block_size`, 2048 tokens on the published
// config) the mask is ordinary causal attention. This is not the CUDA MSA
// kernel. Prompt-cache restore does not keep indexer keys, so a restored
// cache falls back to that causal mask.
//
// Routed experts are a `SwitchLinear` SwiGLU-OAI. NovaMLX-TIE streams those
// experts from SSD the same way it streams DeepSeek-V4.1-Flash. Attention,
// the shared expert, and the router stay in tier 0.

// MARK: - License and catalog bytes

public enum MiniMaxM3 {
    public static let modelType = "minimax_m3_vl"
    /// Required by the MiniMax Community License. Not injected into completions.
    public static let attribution = "Built with MiniMax M3.1-Flash"
    /// `metadata.total_size` of `mlx-community/MiniMax-M3.1-Flash-Text-3bit`.
    public static let text3bitBytes: UInt64 = 186_488_590_848
    /// `metadata.total_size` of `mlx-community/MiniMax-M3.1-Flash-Text-4bit`.
    public static let text4bitBytes: UInt64 = 248_369_178_624
    public static let text3bitId = "mlx-community/MiniMax-M3.1-Flash-Text-3bit"
    public static let text4bitId = "mlx-community/MiniMax-M3.1-Flash-Text-4bit"
}

// MARK: - Memory

public enum MiniMaxM3Memory {
    /// Weights at or above this size are not loaded resident on machines
    /// under `fullResidentMemoryBytes` unless a TIE manifest is present.
    public static let residentWeightLimitBytes: UInt64 = 160 * 1024 * 1024 * 1024
    public static let fullResidentMemoryBytes: UInt64 = 256 * 1024 * 1024 * 1024

    public static func isTextCheckpoint(_ json: [String: Any]) -> Bool {
        if (json["model_type"] as? String)?.lowercased() == MiniMaxM3.modelType {
            return true
        }
        let architectures = json["architectures"] as? [String] ?? []
        return architectures.contains("MiniMaxM3SparseForConditionalGeneration")
            || architectures.contains("MiniMaxM3SparseForCausalLM")
    }

    public static func hasTieManifest(at directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tier-manifest.json").path)
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

    /// Nil when the load may proceed. A TIE manifest skips the gate: expert
    /// bytes stay on SSD. Distributed workers pass `skipMemoryGate`.
    public static func refusalMessage(
        modelId: String,
        weightBytes: UInt64,
        physicalBytes: UInt64,
        tieReady: Bool
    ) -> String? {
        if tieReady { return nil }
        guard weightBytes >= residentWeightLimitBytes else { return nil }
        guard physicalBytes > 0, physicalBytes < fullResidentMemoryBytes else { return nil }
        return """
        Insufficient unified memory to load \(modelId) resident. This \
        MiniMax-M3.1-Flash text checkpoint is \(gibibytes(weightBytes)) GB \
        on disk. Macs under 256 GB need a NovaMLX-TIE layout \
        (tier-manifest.json) so routed experts stream from SSD. Load converts \
        automatically when the checkpoint is larger than the GPU budget. A \
        256 GB Mac can load \(MiniMaxM3.text3bitId) resident (conversion \
        README peak about 187 GB) or \(MiniMaxM3.text4bitId) (README peak \
        about 249 GB).
        """
    }

    private static func gibibytes(_ bytes: UInt64) -> Int {
        Int((Double(bytes) / 1_073_741_824).rounded())
    }
}

// MARK: - Configuration

public struct MiniMaxM3TextConfiguration: Decodable, Sendable, Equatable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var denseIntermediateSize: Int
    public var sharedIntermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numHiddenLayers: Int
    public var numLocalExperts: Int
    public var numExpertsPerTok: Int
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var partialRotaryFactor: Float
    public var headDim: Int
    public var rotaryDim: Int
    public var vocabSize: Int
    public var tieWordEmbeddings: Bool
    public var routedScalingFactor: Float
    public var useQKNorm: Bool
    public var useGemmaNorm: Bool
    public var swigluAlpha: Float
    public var swigluLimit: Float
    public var useRoutingBias: Bool
    public var mlpLayerTypes: [String]?
    public var moeLayerFreq: [Int]?
    public var layerTypes: [String]?
    public var sparseAttentionFreq: [Int]?
    public var indexNHeads: Int?
    public var indexHeadDim: Int?
    public var indexBlockSize: Int?
    public var indexTopkBlocks: Int?
    public var indexLocalBlocks: Int

    public func isMoE(at layer: Int) -> Bool {
        if let mlpLayerTypes, layer < mlpLayerTypes.count {
            return mlpLayerTypes[layer] == "sparse"
        }
        if let moeLayerFreq, layer < moeLayerFreq.count {
            return moeLayerFreq[layer] != 0
        }
        return true
    }

    /// Functional MSA is on when the indexer shape is present and this layer
    /// is not marked full attention. A nil block size means dense attention.
    public func isSparseAttention(at layer: Int) -> Bool {
        guard let indexBlockSize, indexBlockSize > 0,
              (indexNHeads ?? 0) > 0,
              (indexHeadDim ?? 0) > 0,
              (indexTopkBlocks ?? 0) > 0
        else { return false }
        if let layerTypes, layer < layerTypes.count {
            return layerTypes[layer] != "full_attention"
        }
        if let sparseAttentionFreq, layer < sparseAttentionFreq.count {
            return sparseAttentionFreq[layer] != 0
        }
        return true
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case headDim = "head_dim"
        case rotaryDim = "rotary_dim"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case routedScalingFactor = "routed_scaling_factor"
        case useQKNorm = "use_qk_norm"
        case useGemmaNorm = "use_gemma_norm"
        case swigluAlpha = "swiglu_alpha"
        case swigluLimit = "swiglu_limit"
        case useRoutingBias = "use_routing_bias"
        case mlpLayerTypes = "mlp_layer_types"
        case moeLayerFreq = "moe_layer_freq"
        case layerTypes = "layer_types"
        case sparseAttentionFreq = "sparse_attention_freq"
        case sparseAttentionConfig = "sparse_attention_config"
        case ropeParameters = "rope_parameters"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexBlockSize = "index_block_size"
        case indexTopkBlocks = "index_topk_blocks"
        case indexLocalBlocks = "index_local_blocks"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        denseIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .denseIntermediateSize) ?? intermediateSize
        sharedIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize) ?? intermediateSize
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numLocalExperts = try container.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 1
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 1
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1048576
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        var theta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 5_000_000
        var partial = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        if let rope = try container.decodeIfPresent(MiniMaxM3RopeParameters.self, forKey: .ropeParameters) {
            if let value = rope.ropeTheta { theta = value }
            if let value = rope.partialRotaryFactor { partial = value }
        }
        ropeTheta = theta
        partialRotaryFactor = partial
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / max(numAttentionHeads, 1))
        if let rotary = try container.decodeIfPresent(Int.self, forKey: .rotaryDim) {
            rotaryDim = rotary
        } else {
            rotaryDim = Int(Float(headDim) * partialRotaryFactor)
        }
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        routedScalingFactor = try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2
        useQKNorm = try container.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true
        useGemmaNorm = try container.decodeIfPresent(Bool.self, forKey: .useGemmaNorm) ?? true
        swigluAlpha = try container.decodeIfPresent(Float.self, forKey: .swigluAlpha) ?? 1.702
        swigluLimit = try container.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 7
        useRoutingBias = try container.decodeIfPresent(Bool.self, forKey: .useRoutingBias) ?? true
        mlpLayerTypes = try container.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        moeLayerFreq = try container.decodeIfPresent([Int].self, forKey: .moeLayerFreq)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        indexNHeads = try container.decodeIfPresent(Int.self, forKey: .indexNHeads)
        indexHeadDim = try container.decodeIfPresent(Int.self, forKey: .indexHeadDim)
        indexBlockSize = try container.decodeIfPresent(Int.self, forKey: .indexBlockSize)
        indexTopkBlocks = try container.decodeIfPresent(Int.self, forKey: .indexTopkBlocks)
        indexLocalBlocks = try container.decodeIfPresent(Int.self, forKey: .indexLocalBlocks) ?? 1
        sparseAttentionFreq = try container.decodeIfPresent([Int].self, forKey: .sparseAttentionFreq)
        if let sparse = try container.decodeIfPresent(MiniMaxM3SparseAttentionConfig.self, forKey: .sparseAttentionConfig) {
            if let value = sparse.sparseNumIndexHeads { indexNHeads = value }
            if let value = sparse.sparseIndexDim { indexHeadDim = value }
            if let value = sparse.sparseBlockSize { indexBlockSize = value }
            if let value = sparse.sparseTopkBlocks { indexTopkBlocks = value }
            if let value = sparse.sparseLocalBlock { indexLocalBlocks = value }
            if let value = sparse.sparseAttentionFreq { sparseAttentionFreq = value }
        }
    }

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        denseIntermediateSize: Int,
        sharedIntermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        numHiddenLayers: Int,
        numLocalExperts: Int,
        numExpertsPerTok: Int,
        vocabSize: Int,
        headDim: Int,
        rotaryDim: Int? = nil,
        maxPositionEmbeddings: Int = 128,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000,
        partialRotaryFactor: Float = 0.5,
        tieWordEmbeddings: Bool = false,
        routedScalingFactor: Float = 2,
        useQKNorm: Bool = true,
        useGemmaNorm: Bool = true,
        swigluAlpha: Float = 1.702,
        swigluLimit: Float = 7,
        useRoutingBias: Bool = true,
        mlpLayerTypes: [String]? = nil,
        moeLayerFreq: [Int]? = nil,
        layerTypes: [String]? = nil,
        sparseAttentionFreq: [Int]? = nil,
        indexNHeads: Int? = nil,
        indexHeadDim: Int? = nil,
        indexBlockSize: Int? = nil,
        indexTopkBlocks: Int? = nil,
        indexLocalBlocks: Int = 1
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.denseIntermediateSize = denseIntermediateSize
        self.sharedIntermediateSize = sharedIntermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numHiddenLayers = numHiddenLayers
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.headDim = headDim
        self.rotaryDim = rotaryDim ?? Int(Float(headDim) * partialRotaryFactor)
        self.vocabSize = vocabSize
        self.tieWordEmbeddings = tieWordEmbeddings
        self.routedScalingFactor = routedScalingFactor
        self.useQKNorm = useQKNorm
        self.useGemmaNorm = useGemmaNorm
        self.swigluAlpha = swigluAlpha
        self.swigluLimit = swigluLimit
        self.useRoutingBias = useRoutingBias
        self.mlpLayerTypes = mlpLayerTypes
        self.moeLayerFreq = moeLayerFreq
        self.layerTypes = layerTypes
        self.sparseAttentionFreq = sparseAttentionFreq
        self.indexNHeads = indexNHeads
        self.indexHeadDim = indexHeadDim
        self.indexBlockSize = indexBlockSize
        self.indexTopkBlocks = indexTopkBlocks
        self.indexLocalBlocks = indexLocalBlocks
    }
}

struct MiniMaxM3RopeParameters: Decodable, Sendable {
    var ropeTheta: Float?
    var partialRotaryFactor: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

struct MiniMaxM3SparseAttentionConfig: Decodable, Sendable {
    var sparseNumIndexHeads: Int?
    var sparseIndexDim: Int?
    var sparseBlockSize: Int?
    var sparseTopkBlocks: Int?
    var sparseLocalBlock: Int?
    var sparseAttentionFreq: [Int]?

    enum CodingKeys: String, CodingKey {
        case sparseNumIndexHeads = "sparse_num_index_heads"
        case sparseIndexDim = "sparse_index_dim"
        case sparseBlockSize = "sparse_block_size"
        case sparseTopkBlocks = "sparse_topk_blocks"
        case sparseLocalBlock = "sparse_local_block"
        case sparseAttentionFreq = "sparse_attention_freq"
    }
}

public struct MiniMaxM3Configuration: Decodable, Sendable {
    public var modelType: String
    public var text: MiniMaxM3TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? MiniMaxM3.modelType
        if container.contains(.textConfig) {
            text = try container.decode(MiniMaxM3TextConfiguration.self, forKey: .textConfig)
        } else {
            text = try MiniMaxM3TextConfiguration(from: decoder)
        }
    }

    public init(modelType: String = MiniMaxM3.modelType, text: MiniMaxM3TextConfiguration) {
        self.modelType = modelType
        self.text = text
    }
}

// MARK: - Activations and norms

/// GPT-OSS / MiniMax SwiGLU-OAI. `up` is the linear branch, `gate` is clipped from above.
func minimaxSwigluOAI(up: MLXArray, gate: MLXArray, alpha: Float, limit: Float) -> MLXArray {
    let clippedGate = MLX.minimum(gate, MLXArray(limit))
    let clippedUp = clip(up, min: MLXArray(-limit), max: MLXArray(limit))
    return (clippedUp + 1) * (clippedGate * sigmoid(MLXArray(alpha) * clippedGate))
}

/// Gemma RMSNorm stores a zero-initialized residual: `rms_norm(x, 1 + weight)`.
/// Standard RMSNorm stores the multiplier directly.
final class MiniMaxM3Norm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let gemma: Bool

    init(dimensions: Int, eps: Float, gemma: Bool) {
        self.eps = eps
        self.gemma = gemma
        self._weight.wrappedValue = gemma
            ? MLXArray.zeros([dimensions], dtype: .float32)
            : MLXArray.ones([dimensions], dtype: .float32)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = gemma ? weight + MLXArray(Float(1)) : weight
        return MLXFast.rmsNorm(x, weight: scale, eps: eps)
    }
}

// MARK: - Cache

/// Main token cache plus indexer keys. Prompt-cache files do not restore the
/// indexer (`serializedClassName` stays nil). A plain `KVCacheSimple` left by
/// prefix restore is treated as dense causal attention.
public final class MiniMaxM3LayerCache: KVCache {
    private let tokens = KVCacheSimple()
    private var indexKeys: MLXArray?

    public var offset: Int { tokens.offset }
    public var maxSize: Int? { tokens.maxSize }
    public var isTrimmable: Bool { tokens.isTrimmable }

    public var state: [MLXArray] {
        get { tokens.state }
        set { tokens.state = newValue }
    }

    public var metaState: [String] {
        get { tokens.metaState }
        set { tokens.metaState = newValue }
    }

    public func innerState() -> [MLXArray] { state }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        tokens.update(keys: keys, values: values)
    }

    /// Append indexer keys on the sequence axis. Evaluated immediately so a
    /// later dense step (top-k covers every block) still keeps them.
    func updateIndexKeys(_ keys: MLXArray) -> MLXArray {
        let merged: MLXArray
        if let indexKeys {
            merged = concatenated([indexKeys, keys], axis: 2)
        } else {
            merged = keys
        }
        MLX.eval(merged)
        indexKeys = merged
        return merged
    }

    public func trim(_ n: Int) -> Int {
        let trimmed = tokens.trim(n)
        guard trimmed > 0, let keys = indexKeys else { return trimmed }
        let length = keys.dim(2)
        if trimmed >= length {
            indexKeys = nil
        } else {
            indexKeys = keys[0..., 0..., ..<(length - trimmed), 0...]
        }
        return trimmed
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        tokens.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public func copy() -> any KVCache {
        let clone = MiniMaxM3LayerCache()
        clone.state = state
        clone.metaState = metaState
        clone.indexKeys = indexKeys
        return clone
    }
}

// MARK: - MLP

final class MiniMaxM3DenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let alpha: Float
    let limit: Float

    init(hidden: Int, intermediate: Int, alpha: Float, limit: Float) {
        self.alpha = alpha
        self.limit = limit
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(minimaxSwigluOAI(up: upProj(x), gate: gateProj(x), alpha: alpha, limit: limit))
    }
}

final class MiniMaxM3SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear
    let alpha: Float
    let limit: Float

    init(hidden: Int, intermediate: Int, experts: Int, alpha: Float, limit: Float) {
        self.alpha = alpha
        self.limit = limit
        _gateProj.wrappedValue = SwitchLinear(
            inputDims: hidden, outputDims: intermediate, numExperts: experts, bias: false)
        _upProj.wrappedValue = SwitchLinear(
            inputDims: hidden, outputDims: intermediate, numExperts: experts, bias: false)
        _downProj.wrappedValue = SwitchLinear(
            inputDims: intermediate, outputDims: hidden, numExperts: experts, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        var xIn = expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (xIn, idx, inverseOrder) = gatherSort(x: xIn, indices: indices)
        }
        let up = upProj(xIn, idx, sortedIndices: doSort)
        let gate = gateProj(xIn, idx, sortedIndices: doSort)
        var hidden = minimaxSwigluOAI(up: up, gate: gate, alpha: alpha, limit: limit)
        hidden = downProj(hidden, idx, sortedIndices: doSort)
        if doSort {
            hidden = scatterUnsort(x: hidden, invOrder: inverseOrder, shape: indices.shape)
        }
        return (squeezed(hidden, axis: -2) * weights[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

final class MiniMaxM3SparseMoe: Module {
    let topK: Int
    let routedScalingFactor: Float
    let useRoutingBias: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "e_score_correction_bias") var scoreBias: MLXArray
    @ModuleInfo(key: "switch_mlp") var switchMLP: MiniMaxM3SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: MiniMaxM3DenseMLP

    init(_ args: MiniMaxM3TextConfiguration) {
        topK = args.numExpertsPerTok
        routedScalingFactor = args.routedScalingFactor
        useRoutingBias = args.useRoutingBias
        _gate.wrappedValue = Linear(args.hiddenSize, args.numLocalExperts, bias: false)
        _scoreBias.wrappedValue = MLXArray.zeros([args.numLocalExperts], dtype: .float32)
        _switchMLP.wrappedValue = MiniMaxM3SwitchGLU(
            hidden: args.hiddenSize,
            intermediate: args.intermediateSize,
            experts: args.numLocalExperts,
            alpha: args.swigluAlpha,
            limit: args.swigluLimit)
        _sharedExperts.wrappedValue = MiniMaxM3DenseMLP(
            hidden: args.hiddenSize,
            intermediate: args.sharedIntermediateSize,
            alpha: args.swigluAlpha,
            limit: args.swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let leading = x.shape.dropLast()
        let flat = x.reshaped(-1, x.dim(-1))
        let shared = sharedExperts(flat)
        let gates = gate(flat)
        let scores = sigmoid(gates.asType(.float32))
        var biased = scores
        if useRoutingBias {
            biased = scores + scoreBias
        }
        let inds = stopGradient(argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK])
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        weights = (weights * routedScalingFactor).asType(flat.dtype)
        let routed = switchMLP(flat, indices: inds, weights: weights)
        let hidden = leading.count
        var shape = Array(leading)
        shape.append(routed.dim(-1))
        if hidden == 0 {
            return routed + shared
        }
        return (routed + shared).reshaped(shape)
    }
}

// MARK: - Attention

final class MiniMaxM3Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float
    let rotaryDim: Int
    let useQKNorm: Bool
    let isSparse: Bool
    let indexHeads: Int
    let indexDim: Int
    let indexRotaryDim: Int
    let blockSize: Int
    let topkBlocks: Int
    let localBlocks: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: MiniMaxM3Norm?
    @ModuleInfo(key: "k_norm") var kNorm: MiniMaxM3Norm?
    @ModuleInfo(key: "index_q_proj") var indexQProj: Linear?
    @ModuleInfo(key: "index_k_proj") var indexKProj: Linear?
    @ModuleInfo(key: "index_q_norm") var indexQNorm: MiniMaxM3Norm?
    @ModuleInfo(key: "index_k_norm") var indexKNorm: MiniMaxM3Norm?

    init(_ args: MiniMaxM3TextConfiguration, layer: Int) {
        numHeads = args.numAttentionHeads
        numKVHeads = args.numKeyValueHeads
        headDim = args.headDim
        scale = pow(Float(headDim), -0.5)
        ropeTheta = args.ropeTheta
        rotaryDim = args.rotaryDim
        useQKNorm = args.useQKNorm
        isSparse = args.isSparseAttention(at: layer)
        indexHeads = args.indexNHeads ?? 0
        indexDim = args.indexHeadDim ?? 0
        indexRotaryDim = min(args.rotaryDim, indexDim)
        blockSize = args.indexBlockSize ?? 0
        topkBlocks = args.indexTopkBlocks ?? 0
        localBlocks = args.indexLocalBlocks
        let hidden = args.hiddenSize
        _qProj.wrappedValue = Linear(hidden, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, hidden, bias: false)
        if useQKNorm {
            _qNorm.wrappedValue = MiniMaxM3Norm(dimensions: headDim, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
            _kNorm.wrappedValue = MiniMaxM3Norm(dimensions: headDim, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
        }
        if isSparse {
            precondition(indexHeads > 0 && numHeads % indexHeads == 0, "index heads must divide query heads")
            _indexQProj.wrappedValue = Linear(hidden, indexHeads * indexDim, bias: false)
            _indexKProj.wrappedValue = Linear(hidden, indexDim, bias: false)
            _indexQNorm.wrappedValue = MiniMaxM3Norm(dimensions: indexDim, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
            _indexKNorm.wrappedValue = MiniMaxM3Norm(dimensions: indexDim, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let offset = cache?.offset ?? 0
        var queries = qProj(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(batch, length, numKVHeads, headDim).transposed(0, 2, 1, 3)
        if useQKNorm {
            if let qNorm { queries = qNorm(queries) }
            if let kNorm { keys = kNorm(keys) }
        }
        queries = MLXFast.RoPE(
            queries, dimensions: rotaryDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)
        keys = MLXFast.RoPE(
            keys, dimensions: rotaryDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)

        var attnMask = mask
        if isSparse, let indexCache = cache as? MiniMaxM3LayerCache {
            if let blocks = blockMask(x, mask: mask, offset: offset, cache: indexCache) {
                attnMask = .array(blocks)
            }
        }
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: attnMask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }

    /// Bool mask, true = keep. Nil means the incoming causal mode already
    /// attends every block, which is the short-context path.
    private func blockMask(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        offset: Int,
        cache: MiniMaxM3LayerCache
    ) -> MLXArray? {
        guard let indexKProj, let indexKNorm, let indexQProj, let indexQNorm else { return nil }
        let batch = x.dim(0)
        let length = x.dim(1)
        var indexK = indexKProj(x).reshaped(batch, length, 1, indexDim)
        indexK = indexKNorm(indexK).transposed(0, 2, 1, 3)
        indexK = MLXFast.RoPE(
            indexK, dimensions: indexRotaryDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)
        indexK = cache.updateIndexKeys(indexK)

        let sequence = indexK.dim(2)
        let fullBlocks = sequence / blockSize
        let tail = sequence % blockSize
        let blockCount = fullBlocks + (tail > 0 ? 1 : 0)
        if topkBlocks >= blockCount || blockCount == 0 || sequence == 0 {
            return nil
        }

        var indexQ = indexQProj(x).reshaped(batch, length, indexHeads, indexDim)
        indexQ = indexQNorm(indexQ).transposed(0, 2, 1, 3)
        indexQ = MLXFast.RoPE(
            indexQ, dimensions: indexRotaryDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)

        let queryPositions = MLXArray((0..<length).map { Int32(sequence - length + $0) })
        let columns = MLXArray((0..<sequence).map { Int32($0) })
        let causal = columns.reshaped([1, sequence]) .<= queryPositions.reshaped([length, 1])
        let perKey = perKeyMask(mask, causal: causal)
        var scores = matmul(indexQ.asType(.float32), indexK.asType(.float32).swappedAxes(-1, -2))
        scores = MLX.where(perKey, scores, MLXArray(-Float.infinity))

        let pooled: MLXArray
        if fullBlocks > 0 {
            let splitAt = fullBlocks * blockSize
            let head = scores[.ellipsis, ..<splitAt].reshaped([batch, indexHeads, length, fullBlocks, blockSize])
            var blocks = head.max(axis: -1, keepDims: false)
            if tail > 0 {
                let tailScore = scores[.ellipsis, splitAt...].max(axis: -1, keepDims: true)
                blocks = concatenated([blocks, tailScore], axis: -1)
            }
            pooled = blocks
        } else {
            pooled = scores.max(axis: -1, keepDims: true)
        }

        var blockScores = pooled
        if localBlocks > 0 {
            let queryBlock = MLXArray((0..<length).map { Int32((sequence - length + $0) / blockSize) })
            let ids = MLXArray((0..<blockCount).map { Int32($0) })
            let queryColumn = queryBlock.reshaped([length, 1])
            let notFuture = ids .<= queryColumn
            let recent = ids .> (queryColumn - MLXArray(Int32(localBlocks)))
            let isLocal = MLX.where(notFuture, recent, MLXArray(false))
            blockScores = MLX.where(isLocal, MLXArray(Float.infinity), blockScores)
        }

        let picked = stopGradient(
            argPartition(-blockScores, kth: topkBlocks - 1, axis: -1)[.ellipsis, ..<topkBlocks])
        let keep = putAlong(
            MLXArray.zeros(blockScores.shape, dtype: .bool),
            picked,
            values: MLXArray.ones(picked.shape, dtype: .bool),
            axis: -1)
        // mx.repeat repeats each element. Slice drops the padded tail of the last block.
        var expanded = repeated(keep, count: blockSize, axis: -1)
        expanded = expanded[.ellipsis, ..<sequence]
        let kept = MLX.where(expanded, perKey, MLXArray(false))
        return repeated(kept, count: numHeads / indexHeads, axis: 1)
    }

    private func perKeyMask(
        _ mask: MLXFast.ScaledDotProductAttentionMaskMode,
        causal: MLXArray
    ) -> MLXArray {
        if case .array(let array) = mask, array.dtype == .bool {
            return array
        }
        return causal
    }
}

// MARK: - Layers and model

final class MiniMaxM3DecoderLayer: Module {
    let isMoE: Bool

    @ModuleInfo(key: "input_layernorm") var inputNorm: MiniMaxM3Norm
    @ModuleInfo(key: "self_attn") var attention: MiniMaxM3Attention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: MiniMaxM3Norm
    @ModuleInfo(key: "mlp") var mlp: MiniMaxM3DenseMLP?
    @ModuleInfo(key: "block_sparse_moe") var moe: MiniMaxM3SparseMoe?

    init(_ args: MiniMaxM3TextConfiguration, layer: Int) {
        isMoE = args.isMoE(at: layer)
        _inputNorm.wrappedValue = MiniMaxM3Norm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
        _attention.wrappedValue = MiniMaxM3Attention(args, layer: layer)
        _postNorm.wrappedValue = MiniMaxM3Norm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
        if isMoE {
            _moe.wrappedValue = MiniMaxM3SparseMoe(args)
        } else {
            _mlp.wrappedValue = MiniMaxM3DenseMLP(
                hidden: args.hiddenSize,
                intermediate: args.denseIntermediateSize,
                alpha: args.swigluAlpha,
                limit: args.swigluLimit)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = x + attention(inputNorm(x), mask: mask, cache: cache)
        let normalized = postNorm(attended)
        if let moe {
            return attended + moe(normalized)
        }
        return attended + (mlp?(normalized) ?? normalized)
    }
}

final class MiniMaxM3TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [MiniMaxM3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: MiniMaxM3Norm

    init(_ args: MiniMaxM3TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        layers = (0..<args.numHiddenLayers).map { MiniMaxM3DecoderLayer(args, layer: $0) }
        _norm.wrappedValue = MiniMaxM3Norm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps, gemma: args.useGemmaNorm)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }
        return norm(hidden)
    }
}

final class MiniMaxM3LanguageModel: Module {
    let tieWordEmbeddings: Bool
    @ModuleInfo(key: "model") var text: MiniMaxM3TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ args: MiniMaxM3TextConfiguration) {
        tieWordEmbeddings = args.tieWordEmbeddings
        _text.wrappedValue = MiniMaxM3TextModel(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = text(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return text.embedTokens.asLinear(hidden)
    }
}

public final class MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: MiniMaxM3Configuration

    @ModuleInfo(key: "model") var languageModel: MiniMaxM3LanguageModel

    public var loraLayers: [Module] { languageModel.text.layers }

    public init(_ configuration: MiniMaxM3Configuration) {
        self.configuration = configuration
        vocabularySize = configuration.text.vocabSize
        kvHeads = Array(
            repeating: configuration.text.numKeyValueHeads,
            count: configuration.text.numHiddenLayers)
        _languageModel.wrappedValue = MiniMaxM3LanguageModel(configuration.text)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        (0..<configuration.text.numHiddenLayers).map { _ in MiniMaxM3LayerCache() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        MiniMaxM3Sanitizer.remap(weights, text: configuration.text)
    }

    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("block_sparse_moe.gate.")
        }
    }
}

extension MiniMaxM3Model {
    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "<mm:think>",
            endDelimiter: "</mm:think>",
            promptStrategy: .none
        )
    }
}

// MARK: - Sanitize

public enum MiniMaxM3Sanitizer {
    public static func remap(
        _ weights: [String: MLXArray],
        text: MiniMaxM3TextConfiguration
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        let dropPrefixes = [
            "vision_tower.", "multi_modal_projector.", "patch_merge_mlp.",
            "vision.", "dspark.", "mtp.",
        ]
        for (key, value) in weights {
            if dropPrefixes.contains(where: { key.hasPrefix($0) }) { continue }
            if key.contains(".dspark.") || key.contains("/dspark") { continue }
            var name = key
            if name.hasPrefix("language_model.") {
                name = "model." + name.dropFirst("language_model.".count)
            }
            if text.tieWordEmbeddings && name.hasPrefix("model.lm_head.") { continue }
            out[name] = value
        }

        let experts = text.numLocalExperts
        for layer in 0..<text.numHiddenLayers {
            let moe = "model.model.layers.\(layer).block_sparse_moe"
            let mlp = "model.model.layers.\(layer).mlp"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                guard out["\(moe).\(proj).weight"] != nil else { continue }
                for part in ["weight", "scales", "biases"] {
                    if let value = out.removeValue(forKey: "\(moe).\(proj).\(part)") {
                        out["\(mlp).\(proj).\(part)"] = value
                    }
                }
            }
            for proj in ["gate_up_proj", "gate_proj", "up_proj", "down_proj"] {
                guard let weight = out["\(moe).switch_mlp.\(proj).weight"],
                      weight.ndim >= 1, weight.dim(0) == experts + 1
                else { continue }
                for part in ["weight", "scales", "biases", "global_scale"] {
                    guard let value = out.removeValue(forKey: "\(moe).switch_mlp.\(proj).\(part)") else { continue }
                    out["\(moe).switch_mlp.\(proj).\(part)"] = value[..<experts]
                    out["\(moe).shared_experts.\(proj).\(part)"] = value[experts]
                }
            }
            for owner in ["\(moe).switch_mlp", "\(moe).shared_experts"] {
                // global_scale is one value per expert, not fused on the gate/up axis.
                for part in ["weight", "scales", "biases"] {
                    guard let fused = out.removeValue(forKey: "\(owner).gate_up_proj.\(part)") else { continue }
                    let half = fused.dim(-2) / 2
                    let parts = split(fused, indices: [half], axis: -2)
                    guard parts.count == 2 else { continue }
                    out["\(owner).gate_proj.\(part)"] = parts[0]
                    out["\(owner).up_proj.\(part)"] = parts[1]
                }
            }
            guard out["\(moe).experts.0.w1.weight"] != nil else { continue }
            let pairs = [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]
            for (source, dest) in pairs {
                for part in ["weight", "scales", "biases", "global_scale"] {
                    guard out["\(moe).experts.0.\(source).\(part)"] != nil else { continue }
                    var stacked: [MLXArray] = []
                    stacked.reserveCapacity(experts)
                    var complete = true
                    for expert in 0..<experts {
                        guard let value = out["\(moe).experts.\(expert).\(source).\(part)"] else {
                            complete = false
                            break
                        }
                        stacked.append(value)
                    }
                    guard complete else { continue }
                    for expert in 0..<experts {
                        out.removeValue(forKey: "\(moe).experts.\(expert).\(source).\(part)")
                    }
                    out["\(moe).switch_mlp.\(dest).\(part)"] = MLX.stacked(stacked)
                }
            }
        }
        return out
    }
}
