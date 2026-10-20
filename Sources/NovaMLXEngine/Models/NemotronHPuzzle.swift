import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Text-only Nemotron-H Puzzle (`model_type: nemotron_h_puzzle`).
//
// Bolt's language stack is the stock Nemotron-H hybrid (Mamba-2, attention,
// MoE) with two differences stock mlx-swift-lm `NemotronH` does not load:
//   * `block_configs` — expert count, expert width, and top-k differ per layer
//   * latent MoE — `fc1_latent_proj` → routed experts → `fc2_latent_proj`,
//     while the shared expert stays in the full hidden size
//
// Weight names follow the Rapid-MLX text checkpoints
// (`backbone.layers.*.mixer.switch_mlp`, latent projs, no vision tower).
// Behavior follows mlx-lm `nemotron_h.py` latent MoE and the divinetribe
// puzzle port: float32 RMSNorm, float32 router logits, float32 SSM operands.
// Audio, RADIO, and video are not part of this model.

// MARK: - Memory

/// Unified-memory gate for a full Bolt text checkpoint.
///
/// Rapid-MLX 4-bit is ~43 GB on disk and is published for 64 GB Macs.
/// 3-bit is ~29 GB. The check is weight size plus a fixed headroom so a
/// 32 GB machine fails before Metal OOMs, and a 64 GB machine can load 4-bit.
public enum NemotronHPuzzleMemory {
    /// Room for activations, the Mamba state, and the KV cache on top of weights.
    public static let headroomBytes: UInt64 = 16 * 1024 * 1024 * 1024

    public static func requiredBytes(weightBytes: UInt64) -> UInt64 {
        weightBytes + headroomBytes
    }

    /// Nil when `physicalBytes` can hold the checkpoint. Otherwise a message
    /// suitable for `NovaMLXError.configurationError`.
    public static func refusalMessage(
        modelId: String,
        weightBytes: UInt64,
        physicalBytes: UInt64
    ) -> String? {
        let needed = requiredBytes(weightBytes: weightBytes)
        guard weightBytes > 0, physicalBytes > 0, physicalBytes < needed else { return nil }
        return """
        Insufficient unified memory to load \(modelId). This Nemotron 3.5 Bolt \
        text checkpoint is \(gibibytes(weightBytes)) GB on disk and needs about \
        \(gibibytes(needed)) GB of unified memory (weights plus headroom). This \
        machine has \(gibibytes(physicalBytes)) GB. The Rapid-MLX 4-bit build \
        (~43 GB) needs a 64 GB Mac; the 3-bit build (~29 GB) needs about 48 GB.
        """
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

    public static func isTextPuzzle(_ json: [String: Any]) -> Bool {
        if (json["model_type"] as? String)?.lowercased() == "nemotron_h_puzzle" {
            return true
        }
        let architectures = json["architectures"] as? [String] ?? []
        return architectures.contains("NemotronHPuzzleForCausalLM")
    }

    private static func gibibytes(_ bytes: UInt64) -> Int {
        Int((Double(bytes) / 1_073_741_824).rounded())
    }
}

// MARK: - Configuration

public struct NemotronHPuzzleBlockConfig: Decodable, Sendable, Equatable {
    public var blockType: String
    public var moeIntermediateSize: Int?
    public var nRoutedExperts: Int?
    public var numExpertsPerTok: Int?

    enum CodingKeys: String, CodingKey {
        case blockType = "block_type"
        case moeIntermediateSize = "moe_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
    }

    public init(
        blockType: String,
        moeIntermediateSize: Int? = nil,
        nRoutedExperts: Int? = nil,
        numExpertsPerTok: Int? = nil
    ) {
        self.blockType = blockType
        self.moeIntermediateSize = moeIntermediateSize
        self.nRoutedExperts = nRoutedExperts
        self.numExpertsPerTok = numExpertsPerTok
    }
}

public struct NemotronHPuzzleConfiguration: Decodable, Sendable {
    public var modelType: String
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionBias: Bool
    public var mambaNumHeads: Int
    public var mambaHeadDim: Int
    public var mambaProjBias: Bool
    public var ssmStateSize: Int
    public var convKernel: Int
    public var nGroups: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var moeSharedExpertIntermediateSize: Int
    public var moeLatentSize: Int?
    public var nRoutedExperts: Int
    public var nSharedExperts: Int?
    public var numExpertsPerTok: Int
    public var hybridOverridePattern: String
    public var blockConfigs: [NemotronHPuzzleBlockConfig]
    public var layerNormEpsilon: Float
    public var mlpBias: Bool
    public var useConvBias: Bool
    public var tieWordEmbeddings: Bool
    public var headDim: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var timeStepLimitMin: Float
    public var timeStepLimitMax: Float

    public var layerCount: Int { hybridOverridePattern.count }

    public func blockType(at index: Int) -> Character {
        let chars = Array(hybridOverridePattern)
        return chars[index]
    }

    public func routedExpertCount(at index: Int) -> Int {
        blockConfigs[index].nRoutedExperts ?? nRoutedExperts
    }

    public func moeWidth(at index: Int) -> Int {
        blockConfigs[index].moeIntermediateSize ?? moeIntermediateSize
    }

    public func expertsPerToken(at index: Int) -> Int {
        blockConfigs[index].numExpertsPerTok ?? numExpertsPerTok
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case moeLatentSize = "moe_latent_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridOverridePattern = "hybrid_override_pattern"
        case layersBlockType = "layers_block_type"
        case blockConfigs = "block_configs"
        case layerNormEpsilon = "layer_norm_epsilon"
        case rmsNormEps = "rms_norm_eps"
        case normEps = "norm_eps"
        case mlpBias = "mlp_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimit = "time_step_limit"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_h_puzzle"
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mambaNumHeads = try container.decodeIfPresent(Int.self, forKey: .mambaNumHeads) ?? 128
        mambaHeadDim = try container.decodeIfPresent(Int.self, forKey: .mambaHeadDim) ?? 64
        mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        ssmStateSize = try container.decodeIfPresent(Int.self, forKey: .ssmStateSize) ?? 128
        convKernel = try container.decodeIfPresent(Int.self, forKey: .convKernel) ?? 4
        nGroups = try container.decodeIfPresent(Int.self, forKey: .nGroups) ?? 8
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2688
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor = try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        moeLatentSize = try container.decodeIfPresent(Int.self, forKey: .moeLatentSize)
        if moeLatentSize == 0 { moeLatentSize = nil }
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        moeSharedExpertIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize) ?? 0

        if let eps = try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) {
            layerNormEpsilon = eps
        } else if let eps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) {
            layerNormEpsilon = eps
        } else {
            layerNormEpsilon = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        }

        let decodedBlocks = try container.decodeIfPresent(
            [NemotronHPuzzleBlockConfig].self, forKey: .blockConfigs) ?? []
        let decodedLayerTypes = try container.decodeIfPresent([String].self, forKey: .layersBlockType)
        let decodedPattern = try Self.decodePattern(container)

        let pattern: String
        if !decodedBlocks.isEmpty {
            pattern = String(try decodedBlocks.map { try Self.patternCharacter($0.blockType) })
        } else if let decodedLayerTypes, !decodedLayerTypes.isEmpty {
            pattern = String(try decodedLayerTypes.map { try Self.patternCharacter($0) })
        } else if let decodedPattern {
            pattern = decodedPattern
        } else {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription:
                        "nemotron_h_puzzle needs block_configs, layers_block_type, or hybrid_override_pattern"
                ))
        }
        if let decodedPattern, decodedPattern != pattern {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription: "hybrid_override_pattern does not match block_configs"
                ))
        }
        if let decodedLayerTypes, !decodedBlocks.isEmpty, decodedLayerTypes.count != decodedBlocks.count {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription: "layers_block_type and block_configs have different lengths"
                ))
        }
        hybridOverridePattern = pattern
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? pattern.count
        if numHiddenLayers != pattern.count {
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: container.codingPath,
                    debugDescription:
                        "num_hidden_layers (\(numHiddenLayers)) != layer pattern length (\(pattern.count))"
                ))
        }

        let moeBlocks = decodedBlocks.filter { $0.blockType.lowercased() == "moe" }
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? moeBlocks.first?.moeIntermediateSize
            ?? 0
        nRoutedExperts =
            try container.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
            ?? moeBlocks.first?.nRoutedExperts
            ?? 0
        numExpertsPerTok =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
            ?? moeBlocks.first?.numExpertsPerTok
            ?? 0

        if decodedBlocks.isEmpty {
            blockConfigs = pattern.map { char in
                switch char {
                case "E":
                    NemotronHPuzzleBlockConfig(
                        blockType: "moe",
                        moeIntermediateSize: moeIntermediateSize,
                        nRoutedExperts: nRoutedExperts,
                        numExpertsPerTok: numExpertsPerTok
                    )
                case "*":
                    NemotronHPuzzleBlockConfig(blockType: "attention")
                case "-":
                    NemotronHPuzzleBlockConfig(blockType: "mlp")
                default:
                    NemotronHPuzzleBlockConfig(blockType: "mamba")
                }
            }
        } else {
            blockConfigs = decodedBlocks
        }

        if let limits = try container.decodeIfPresent([JSONFloatValue].self, forKey: .timeStepLimit),
           limits.count >= 2
        {
            timeStepLimitMin = limits[0].value
            timeStepLimitMax = limits[1].value
        } else {
            timeStepLimitMin = 0
            timeStepLimitMax = .infinity
        }
    }

    /// Memberwise initializer for tiny fixtures.
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeSharedExpertIntermediateSize: Int,
        moeLatentSize: Int?,
        nRoutedExperts: Int,
        nSharedExperts: Int?,
        numExpertsPerTok: Int,
        blockConfigs: [NemotronHPuzzleBlockConfig],
        layerNormEpsilon: Float = 1e-5,
        headDim: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1,
        timeStepLimitMin: Float = 0,
        timeStepLimitMax: Float = .infinity,
        tieWordEmbeddings: Bool = false,
        useConvBias: Bool = true
    ) {
        self.modelType = "nemotron_h_puzzle"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = false
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = false
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.moeLatentSize = moeLatentSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.blockConfigs = blockConfigs
        self.hybridOverridePattern = String(blockConfigs.map { char in
            switch char.blockType.lowercased() {
            case "attention": "*"
            case "moe": "E"
            case "mlp": "-"
            default: "M"
            }
        })
        self.numHiddenLayers = blockConfigs.count
        self.layerNormEpsilon = layerNormEpsilon
        self.mlpBias = false
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
    }

    private static func decodePattern(
        _ container: KeyedDecodingContainer<CodingKeys>
    ) throws -> String? {
        if let pattern = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            return pattern
        }
        if let parts = try? container.decode([String].self, forKey: .hybridOverridePattern) {
            return parts.joined()
        }
        return nil
    }

    static func patternCharacter(_ blockType: String) throws -> Character {
        switch blockType.lowercased() {
        case "mamba", "linear_attention", "m":
            return "M"
        case "attention", "full_attention", "*":
            return "*"
        case "moe", "e":
            return "E"
        case "mlp", "-":
            return "-"
        default:
            throw DecodingError.dataCorrupted(
                DecodingError.Context(
                    codingPath: [],
                    debugDescription: "Unknown Nemotron-H block type \(blockType)"
                ))
        }
    }
}

/// Accepts a JSON number or transformers' `{"__float__": "Infinity"}`.
private struct JSONFloatValue: Decodable {
    let value: Float

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let number = try? container.decode(Double.self) {
            if number.isInfinite {
                value = number > 0 ? .infinity : -.infinity
            } else {
                value = Float(number)
            }
            return
        }
        if let text = try? container.decode(String.self) {
            value = Self.parse(text)
            return
        }
        if let object = try? container.decode([String: String].self),
           let text = object["__float__"]
        {
            value = Self.parse(text)
            return
        }
        throw DecodingError.dataCorruptedError(
            in: container, debugDescription: "Expected a float or {\"__float__\": \"...\"}")
    }

    private static func parse(_ text: String) -> Float {
        switch text.lowercased() {
        case "infinity", "inf", "+infinity", "+inf":
            return .infinity
        case "-infinity", "-inf":
            return -.infinity
        default:
            return Float(text) ?? 0
        }
    }
}

// MARK: - Blocks

private enum NemotronHPuzzleBlockKind {
    case mamba
    case attention
    case mlp
    case moe

    init(_ character: Character) {
        switch character {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default: self = .mamba
        }
    }
}

private protocol NemotronHPuzzleMixer: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

private func puzzleRelu2(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0))
    return y * y
}

private final class NemotronHPuzzleRMSNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xf = x.asType(.float32)
        let scale = MLX.rsqrt(MLX.mean(xf * xf, axis: -1, keepDims: true) + eps)
        return (weight.asType(.float32) * xf * scale).asType(dtype)
    }
}

private final class NemotronHPuzzleMamba2Mixer: Module, NemotronHPuzzleMixer {
    let numHeads: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var dCoeff: MLXArray
    @ModuleInfo(key: "norm") var norm: NemotronHPuzzleRMSNormGated

    init(_ args: NemotronHPuzzleConfiguration) {
        numHeads = args.mambaNumHeads
        ssmStateSize = args.ssmStateSize
        convKernelSize = args.convKernel
        intermediateSize = args.mambaNumHeads * args.mambaHeadDim
        numGroups = args.nGroups
        headDim = args.mambaHeadDim
        timeStepLimit = (args.timeStepLimitMin, args.timeStepLimitMax)
        convDim = intermediateSize + 2 * numGroups * ssmStateSize

        _conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias
        )
        let projectionSize = intermediateSize + convDim + numHeads
        _inProj.wrappedValue = Linear(args.hiddenSize, projectionSize, bias: args.mambaProjBias)
        _dtBias.wrappedValue = MLXArray.ones([numHeads])
        let headsRange = MLXArray(0 ..< numHeads).asType(.float32) + 1
        _aLog.wrappedValue = MLX.log(headsRange)
        _dCoeff.wrappedValue = MLXArray.ones([numHeads])
        let groupSize = intermediateSize / numGroups
        _norm.wrappedValue = NemotronHPuzzleRMSNormGated(
            dimensions: intermediateSize,
            eps: args.layerNormEpsilon,
            groupSize: groupSize
        )
        _outProj.wrappedValue = Linear(intermediateSize, args.hiddenSize, bias: args.mambaProjBias)
        super.init()
    }

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        var convInput = input
        if let mask {
            convInput = MLX.where(
                expandedDimensions(mask, axis: -1), convInput, MLXArray.zeros(like: convInput))
        }
        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]
        if convState == nil {
            let history = max(0, convKernelSize - 1)
            convState = MLXArray.zeros([batch, history, convDim], dtype: dtype)
        }
        let padded = concatenated([convState!, convInput], axis: 1)
        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = contiguous(padded[0..., start ..< end, 0...])
        }
        return silu(conv1d(padded))
    }

    private func mambaForward(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]
        let convOutput = applyConv(convInput, mask: mask, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )
        let hidden = convSplits[0].reshaped([
            convSplits[0].dim(0), convSplits[0].dim(1), numHeads, headDim,
        ])
        let b = convSplits[1].reshaped([
            convSplits[1].dim(0), convSplits[1].dim(1), numGroups, ssmStateSize,
        ])
        let c = convSplits[2].reshaped([
            convSplits[2].dim(0), convSplits[2].dim(1), numGroups, ssmStateSize,
        ])
        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])
        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden.asType(.float32),
            ALog: aLog.asType(.float32),
            B: b.asType(.float32),
            C: c.asType(.float32),
            D: dCoeff.asType(.float32),
            dt: dtArray.asType(.float32),
            dtBias: dtBias.asType(.float32),
            state: cache?[1],
            timeStepLimit: timeStepLimit,
            mask: mask
        )
        if let cache {
            cache[1] = nextState
            cache.advance(hiddenStates.dim(1))
        }
        let flattened = y.asType(hiddenStates.dtype).flattened(start: 2)
        return outProj(norm(flattened, gate: gate))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        mambaForward(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

private final class NemotronHPuzzleRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int

    init(dimensions: Int, eps: Float, groupSize: Int) {
        self.eps = eps
        self.groupSize = groupSize
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        var states = x
        if let gate {
            states = states * silu(gate)
        }
        let shape = states.shape
        var grouped = Array(shape.dropLast())
        grouped.append(-1)
        grouped.append(groupSize)
        let identity = MLXArray.ones([groupSize])
        let normed = MLXFast.rmsNorm(states.reshaped(grouped), weight: identity, eps: eps)
        return weight * normed.reshaped(shape)
    }
}

private final class NemotronHPuzzleAttention: Module, NemotronHPuzzleMixer {
    let scale: Float
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    init(_ args: NemotronHPuzzleConfiguration) {
        numHeads = args.numAttentionHeads
        numKeyValueHeads = args.numKeyValueHeads
        headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        scale = pow(Float(headDim), -0.5)
        let dim = args.hiddenSize
        _wq.wrappedValue = Linear(dim, numHeads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(numHeads * headDim, dim, bias: args.attentionBias)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let queries = wq(x).reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        // Nemotron-H attention has no RoPE.
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return wo(output)
    }
}

private final class NemotronHPuzzleMLP: Module, UnaryLayer, NemotronHPuzzleMixer {
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool) {
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(puzzleRelu2(upProj(x)))
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

private func puzzleGroupExpertSelect(
    gates: MLXArray,
    correction: MLXArray,
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    let batch = gates.dim(0)
    let seqLen = gates.dim(1)
    let origScores = sigmoid(gates.asType(.float32))
    var scores = origScores + correction.asType(.float32)

    if nGroup > 1 {
        let expertsPerGroup = scores.dim(-1) / nGroup
        let groupScores = scores.reshaped(batch, seqLen, nGroup, -1)
        let topKGroupScores = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(axis: -1, keepDims: true)
        let k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [batch, seqLen, k, expertsPerGroup])
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)
        scores = flattened(scores, start: -2, end: -1)
    }

    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var finalScores = takeAlong(origScores, inds, axis: -1)
    if topK > 1 && normTopkProb {
        finalScores = finalScores / (finalScores.sum(axis: -1, keepDims: true) + 1e-20)
    }
    finalScores = finalScores * routedScalingFactor
    return (inds, finalScores)
}

private final class NemotronHPuzzleMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correction: MLXArray

    init(
        hiddenSize: Int,
        nRoutedExperts: Int,
        topK: Int,
        nGroup: Int,
        topkGroup: Int,
        routedScalingFactor: Float,
        normTopkProb: Bool
    ) {
        self.topK = topK
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.routedScalingFactor = routedScalingFactor
        self.normTopkProb = normTopkProb
        _weight.wrappedValue = MLXArray.zeros([nRoutedExperts, hiddenSize])
        _correction.wrappedValue = MLXArray.zeros([nRoutedExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = MLX.matmul(x.asType(.float32), weight.asType(.float32).transposed())
        return puzzleGroupExpertSelect(
            gates: logits,
            correction: correction,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }
}

private final class NemotronHPuzzleSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        _fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        _fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size > 64
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        var y = fc1(x, idx, sortedIndices: doSort)
        y = puzzleRelu2(y)
        y = fc2(y, idx, sortedIndices: doSort)
        if doSort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(y, axis: -2)
    }
}

private final class NemotronHPuzzleMoE: Module, UnaryLayer, NemotronHPuzzleMixer {
    @ModuleInfo(key: "gate") var gate: NemotronHPuzzleMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronHPuzzleSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHPuzzleMLP?
    @ModuleInfo(key: "fc1_latent_proj") var fc1Latent: Linear?
    @ModuleInfo(key: "fc2_latent_proj") var fc2Latent: Linear?

    init(_ args: NemotronHPuzzleConfiguration, layer: Int) {
        let experts = args.routedExpertCount(at: layer)
        let width = args.moeWidth(at: layer)
        let topK = args.expertsPerToken(at: layer)
        let expertInput = args.moeLatentSize ?? args.hiddenSize
        _gate.wrappedValue = NemotronHPuzzleMoEGate(
            hiddenSize: args.hiddenSize,
            nRoutedExperts: experts,
            topK: topK,
            nGroup: args.nGroup,
            topkGroup: args.topkGroup,
            routedScalingFactor: args.routedScalingFactor,
            normTopkProb: args.normTopkProb
        )
        _switchMLP.wrappedValue = NemotronHPuzzleSwitchMLP(
            inputDims: expertInput, hiddenDims: width, numExperts: experts)
        if let latent = args.moeLatentSize, latent > 0 {
            _fc1Latent.wrappedValue = Linear(args.hiddenSize, latent, bias: args.mlpBias)
            _fc2Latent.wrappedValue = Linear(latent, args.hiddenSize, bias: args.mlpBias)
        }
        if let shared = args.nSharedExperts, shared > 0 {
            let sharedWidth = args.moeSharedExpertIntermediateSize > 0
                ? args.moeSharedExpertIntermediateSize
                : args.intermediateSize
            _sharedExperts.wrappedValue = NemotronHPuzzleMLP(
                hiddenSize: args.hiddenSize, intermediateSize: sharedWidth, bias: args.mlpBias)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var routed = x
        if let fc1Latent {
            routed = fc1Latent(x)
        }
        var y = switchMLP(routed, indices)
        y = weightedExpertSum(y, scores).asType(y.dtype)
        if let fc2Latent {
            y = fc2Latent(y)
        }
        // Shared expert sees the full hidden state, not the latent projection.
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

private final class NemotronHPuzzleBlock: Module {
    let kind: NemotronHPuzzleBlockKind
    @ModuleInfo(key: "norm") var norm: NemotronHPuzzleRMSNorm
    @ModuleInfo(key: "mixer") var mixer: Module

    init(_ args: NemotronHPuzzleConfiguration, index: Int) {
        let character = args.blockType(at: index)
        kind = NemotronHPuzzleBlockKind(character)
        _norm.wrappedValue = NemotronHPuzzleRMSNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        switch kind {
        case .mamba:
            _mixer.wrappedValue = NemotronHPuzzleMamba2Mixer(args)
        case .attention:
            _mixer.wrappedValue = NemotronHPuzzleAttention(args)
        case .mlp:
            _mixer.wrappedValue = NemotronHPuzzleMLP(
                hiddenSize: args.hiddenSize,
                intermediateSize: args.intermediateSize,
                bias: args.mlpBias
            )
        case .moe:
            _mixer.wrappedValue = NemotronHPuzzleMoE(args, layer: index)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let hidden = norm(x)
        let mixerFunc = mixer as! NemotronHPuzzleMixer
        return x + mixerFunc(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)
    }
}

private final class NemotronHPuzzleBackbone: Module {
    let firstAttentionCacheIndex: Int?
    let firstMambaCacheIndex: Int?

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHPuzzleBlock]
    @ModuleInfo(key: "norm_f") var normF: NemotronHPuzzleRMSNorm

    init(_ args: NemotronHPuzzleConfiguration) {
        precondition(args.vocabSize > 0)
        _embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        _layers.wrappedValue = (0 ..< args.layerCount).map { NemotronHPuzzleBlock(args, index: $0) }
        _normF.wrappedValue = NemotronHPuzzleRMSNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        var faIdx: Int?
        var mambaCount = 0
        for char in args.hybridOverridePattern {
            if char == "*" {
                faIdx = mambaCount
                break
            } else if char == "M" {
                mambaCount += 1
            }
        }
        firstAttentionCacheIndex = faIdx

        var ssmIdx: Int?
        var attentionCount = 0
        for char in args.hybridOverridePattern {
            if char == "M" {
                ssmIdx = attentionCount
                break
            } else if char == "*" {
                attentionCount += 1
            }
        }
        firstMambaCacheIndex = ssmIdx
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embeddings(inputs)
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let cacheIdx = firstAttentionCacheIndex,
                  let cache, cacheIdx < cache.count
            else { return .none }
            return createAttentionMask(h: hidden, cache: cache[cacheIdx])
        }()
        let ssmMask: MLXArray? = {
            guard let cacheIdx = firstMambaCacheIndex,
                  let cache, cacheIdx < cache.count,
                  let mamba = cache[cacheIdx] as? MambaCache
            else { return nil }
            return mamba.makeMask(N: hidden.dim(1))
        }()

        var cacheCounter = 0
        for layer in layers {
            let layerCache: KVCache?
            if layer.kind == .mamba || layer.kind == .attention {
                layerCache = cache?[cacheCounter]
                cacheCounter += 1
            } else {
                layerCache = nil
            }
            hidden = layer(
                hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: layerCache)
        }
        return normF(hidden)
    }
}

// MARK: - Model

public final class NemotronHPuzzleModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: NemotronHPuzzleConfiguration

    @ModuleInfo(key: "backbone") private var backbone: NemotronHPuzzleBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { backbone.layers }

    public init(_ args: NemotronHPuzzleConfiguration) {
        configuration = args
        vocabularySize = args.vocabSize
        kvHeads = args.hybridOverridePattern.compactMap { char in
            switch char {
            case "*": args.numKeyValueHeads
            case "M": 0
            default: nil
            }
        }
        _backbone.wrappedValue = NemotronHPuzzleBackbone(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = backbone(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return backbone.embeddings.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try configuration.hybridOverridePattern.compactMap { char -> KVCache? in
            switch char {
            case "M":
                return MambaCache()
            case "*":
                return try makeAttentionKVCache(parameters: parameters)
            default:
                return nil
            }
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (rawKey, value) in weights {
            if rawKey.hasPrefix("mtp.") || rawKey.contains(".mtp.") { continue }
            var key = rawKey
            if key.hasPrefix("language_model.") {
                key = String(key.dropFirst("language_model.".count))
            }
            if key.hasPrefix("vision_") || key.hasPrefix("mlp1.") || key.hasPrefix("sound_") {
                continue
            }
            if configuration.tieWordEmbeddings && key.split(separator: ".").contains(where: { $0 == "lm_head" }) {
                continue
            }
            var finalValue = value
            if key.contains("conv1d.weight"), value.ndim == 3, value.dim(-1) != 1 {
                finalValue = value.swappedAxes(1, 2)
            }
            sanitized[key] = finalValue
        }

        for layer in 0 ..< configuration.layerCount where configuration.blockType(at: layer) == "E" {
            let experts = configuration.routedExpertCount(at: layer)
            let prefix = "backbone.layers.\(layer).mixer"
            for (source, destination) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
                for suffix in ["weight", "scales", "biases"] {
                    var stackedWeights: [MLXArray] = []
                    stackedWeights.reserveCapacity(experts)
                    var complete = true
                    for expert in 0 ..< experts {
                        let expertKey = "\(prefix).experts.\(expert).\(source).\(suffix)"
                        guard sanitized[expertKey] != nil else {
                            complete = false
                            break
                        }
                        stackedWeights.append(sanitized[expertKey]!)
                    }
                    guard complete, stackedWeights.count == experts, !stackedWeights.isEmpty else { continue }
                    for expert in 0 ..< experts {
                        sanitized.removeValue(forKey: "\(prefix).experts.\(expert).\(source).\(suffix)")
                    }
                    sanitized["\(prefix).switch_mlp.\(destination).\(suffix)"] = MLX.stacked(stackedWeights)
                }
            }
        }
        return sanitized
    }
}

extension NemotronHPuzzleModel {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }

    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "<think>",
            endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
            implicitEndDelimiters: ["<tool_call>"]
        )
    }
}
