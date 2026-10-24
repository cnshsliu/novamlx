import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// Flint (RecursiveSI/flint) — ported from mlx-lm mlx_lm/models/flint.py.
//
// Architecture: 49-layer decoder, GQA (32 q / 4 kv heads, head_dim 128) with
// per-head QK RMSNorm and a learned per-head attention sink (`softmax_offset`,
// GPT-OSS style). `layer_types` mixes sliding-window-513 layers (RoPE,
// RotatingKVCache) with full-attention layers (no positional encoding,
// KVCacheSimple). MoE from layer 2: sigmoid routing to 6 of 256 experts with
// renormalized scores plus an ungated shared expert; layers 0-1 are dense.

public struct FlintConfiguration: Codable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var mlpOnlyLayers: [Int]
    var layerTypes: [String]
    var slidingWindow: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float = 10000.0
    var maxPositionEmbeddings: Int?
    var tieWordEmbeddings: Bool = false
    /// Transformers nests the theta in `rope_parameters` instead of the
    /// top-level `rope_theta`; the nested value wins when present.
    var ropeParameters: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeParameters = try c.decodeIfPresent([String: Double].self, forKey: .ropeParameters)
        if let nested = ropeParameters, let theta = nested["rope_theta"] {
            ropeTheta = Float(theta)
        } else {
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        }
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - Attention

class FlintAttention: Module {
    let args: FlintConfiguration
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray

    /// RoPE only on sliding-window layers; full-attention layers have none.
    let rope: RoPELayer?

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.args = args
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)
        let dim = args.hiddenSize

        _qProj.wrappedValue = Linear(dim, args.attentionHeads * args.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(args.attentionHeads * args.headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _softmaxOffset.wrappedValue = zeros([args.attentionHeads])

        if !args.layerTypes.isEmpty, args.layerTypes[layerIdx] == "sliding_attention" {
            self.rope = initializeRope(
                dims: args.headDim,
                base: args.ropeTheta,
                traditional: false,
                scalingConfig: nil,
                maxPositionEmbeddings: args.maxPositionEmbeddings
            )
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        // Sinks are only supported by the full-precision SDPA path; a quantized
        // cache (TurboQuant) silently drops them, same trade-off as GPT-OSS.
        if let qcache = cache as? QuantizedKVCacheProtocol {
            let (qKeys, qValues) = qcache.updateQuantized(keys: keys, values: values)
            let out = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: qKeys,
                quantizedValues: qValues,
                scale: scale,
                mask: mask,
                groupSize: qcache.groupSize,
                bits: qcache.bits,
                mode: qcache.mode
            )
            return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale,
            mask: mask,
            sinks: softmaxOffset
        )

        return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

// MARK: - MLP / MoE

class FlintMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class FlintSparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP

    init(_ args: FlintConfiguration) {
        self.topK = args.numExpertsPerTok
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )
        _sharedExpert.wrappedValue = FlintMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
    }

    /// Sigmoid routing: top-k by score, renormalized; plus an ungated shared
    /// expert (flint.py's fallback math — the Swift port has no
    /// `expert_routing` Metal kernel).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x.asType(.float32))
        let scores0 = sigmoid(logits)
        let k = topK
        let inds = argPartition(-scores0, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = takeAlong(scores0, inds, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)

        let y = switchMLP(x, inds)
        let combined = weightedExpertSum(y, scores.asType(y.dtype))
        return combined + sharedExpert(x)
    }
}

// MARK: - Decoder

class FlintDecoderLayer: Module {
    let useSliding: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: FlintAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.useSliding =
            !args.layerTypes.isEmpty && args.layerTypes[layerIdx] == "sliding_attention"
        _selfAttn.wrappedValue = FlintAttention(args, layerIdx: layerIdx)
        if args.mlpOnlyLayers.contains(layerIdx) {
            self._mlp.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            self._mlp.wrappedValue = FlintSparseMoeBlock(args)
        }
        _inputNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputNorm(x), mask: mask, cache: cache)
        let normed = postNorm(h)
        if let moe = mlp as? FlintSparseMoeBlock {
            return h + moe(normed)
        }
        return h + (mlp as! FlintMLP)(normed)
    }
}

// MARK: - Model

public class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layerTypes: [String]
    let layers: [FlintDecoderLayer]
    let windowSize: Int
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int

    init(_ config: FlintConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layerTypes = config.layerTypes
        self.layers = (0 ..< config.hiddenLayers).map { FlintDecoderLayer(config, layerIdx: $0) }
        self.windowSize = config.slidingWindow
        self.slidingAttentionIndex = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.fullAttentionIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var x = inputEmbeddings ?? embedTokens(inputs)
        let caches: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)

        let seqLen = x.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        for (i, layer) in layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if layerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = makeAttentionMask(
                        n: seqLen, cache: caches[fullAttentionIndex], windowSize: nil)
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(
                        n: seqLen, cache: caches[slidingAttentionIndex], windowSize: windowSize)
                }
                maskMode = slidingMask!
            }
            x = layer(x, mask: maskMode, cache: caches[i])
        }

        return norm(x)
    }
}

public class FlintModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: FlintModelInner
    let configuration: FlintConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { model.layers }

    /// Flint emits tool calls in the GLM4 dialect:
    /// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
    /// Declared as a class member (not an extension) so it wins over
    /// `LanguageModel`'s nil default when the factory reads the convention.
    public var toolCallFormat: ToolCallFormat? { .glm4 }

    public init(_ args: FlintConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = FlintModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layerType != "full_attention")
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights
        if configuration.tieWordEmbeddings {
            w["lm_head.weight"] = nil
        }

        // RecursiveSI/flint-int4 stores packed experts as uint8; view them as
        // uint32 so the quantize(model:) pass pairs them with their scales.
        for (k, v) in w where v.dtype == .uint8 {
            let name = k.hasSuffix(".weight") ? String(k.dropLast(7)) : k
            if w["\(name).scales"] != nil {
                w[k] = v.view(dtype: .uint32)
            }
        }

        // Layer 0 is dense in Flint (mlp_only_layers), so probe per layer
        // instead of using a layer-0 guard like Qwen3Next does.

        for l in 0 ..< configuration.hiddenLayers {
            guard !configuration.mlpOnlyLayers.contains(l) else { continue }
            let prefix = "model.layers.\(l).mlp"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                // Stack only when the full expert set is present; a partial
                // set means the checkpoint is not per-expert sharded here.
                guard w["\(prefix).experts.0.\(n).weight"] != nil,
                    w["\(prefix).experts.\(configuration.numExperts - 1).\(n).weight"] != nil
                else { continue }
                let toJoin = (0 ..< configuration.numExperts).map { e in
                    w.removeValue(forKey: "\(prefix).experts.\(e).\(n).weight")!
                }
                w["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
            }
        }

        return w
    }
}
