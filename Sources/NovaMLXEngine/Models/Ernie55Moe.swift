import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import NovaMLXCore

// ERNIE-5.5-Flash-Base (`model_type: ernie5_5_moe`).
//
// The checkpoint is a pretrained base model, not an instruct model. Callers
// use POST /v1/completions. Stock mlx-swift-lm has no `ernie5_5_moe` class,
// and the attention stack is not a stock MLA or GQA block:
//
// - 37 Heavily Compressed Attention layers (compress ratio 128, head dim 512)
// - 6 MLA layers with DSA (compress ratio -2, head dim 256)
// - low-rank gated attention and virtual-head post-mix on every layer
// - multi-router latent MoE: 512 routed experts, top-10, latent 2048
// - manifold-constrained hyper-connections (expansion 4)
// - MultiMax output head, and ECHO reusing the final layer as a draft
//
// This file registers the type and refuses to allocate the 280B graph.
// `Scripts/ernie55_fp8_to_mlx.py --recipe ernie55-128gb` writes the MLX
// weights. HCA / MLA kernels and a Mac end-to-end run are the follow-up.
// A fixture smaller than the size gate builds a stub so the registry can be
// tested without the 288GB FP8 checkpoint.

public enum Ernie55MoeSupport {
    public static func isTextModel(_ json: [String: Any]) -> Bool {
        if (json["model_type"] as? String)?.lowercased() == "ernie5_5_moe" {
            return true
        }
        let architectures = json["architectures"] as? [String] ?? []
        return architectures.contains("Ernie5_5_MoeForCausalLM")
    }

    public static func isOfficialFP8(_ json: [String: Any]) -> Bool {
        let quant = json["quantization_config"] as? [String: Any]
        return (quant?["quant_method"] as? String)?.lowercased() == "fp8"
    }

    /// Official Flash-Base is hidden 4096, 43 layers, 512 experts. Fixtures
    /// used by tests stay under every one of these caps.
    public static func needsFullKernels(_ json: [String: Any]) -> Bool {
        let hidden = int(json["hidden_size"])
        let experts = int(json["n_routed_experts"])
        let layers = int(json["num_hidden_layers"])
        return hidden > 256 || experts > 8 || layers > 4
    }

    public static func loadRefusal(modelId: String, json: [String: Any]) -> String? {
        guard isTextModel(json) else { return nil }
        if isOfficialFP8(json) {
            return """
            \(modelId) is the official ERNIE-5.5-Flash-Base FP8 checkpoint \
            (~288GB, model_type ernie5_5_moe). NovaMLX does not load it. \
            Convert with Scripts/ernie55_fp8_to_mlx.py --src <fp8> --dest <mlx> \
            --recipe ernie55-128gb. Heavily Compressed Attention and MLA \
            kernels are not implemented yet, so the converted weights are not \
            runnable in this build. When they are, serve the base model with \
            POST /v1/completions.
            """
        }
        if needsFullKernels(json) {
            return """
            \(modelId) is ernie5_5_moe. The type is registered, and the \
            ernie55-128gb quant can sit on disk, but Heavily Compressed \
            Attention and MLA kernels are not implemented. This build will \
            not allocate the 280B graph. The follow-up is those kernels plus \
            a Mac end-to-end run through POST /v1/completions.
            """
        }
        return nil
    }

    public static func completionsRefusal(modelId: String, config: [String: Any]) -> String? {
        guard isTextModel(config) else { return nil }
        return completionsMessage(modelId)
    }

    /// Chat, Anthropic messages, and Responses are the wrong API. A missing
    /// config still matches the official repo id.
    public static func completionsRefusal(modelId: String) -> String? {
        let dir = NovaMLXPaths.directory(forModelId: modelId)
        let url = dir.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return completionsRefusal(modelId: modelId, config: json)
        }
        if modelId.lowercased().contains("ernie-5.5-flash-base") {
            return completionsMessage(modelId)
        }
        return nil
    }

    private static func completionsMessage(_ modelId: String) -> String {
        """
        \(modelId) is ERNIE-5.5-Flash-Base (ernie5_5_moe), the pretrained base \
        checkpoint. Use POST /v1/completions with a prompt. Chat completions, \
        /v1/messages, and /v1/responses are not its endpoint.
        """
    }

    private static func int(_ value: Any?) -> Int {
        if let number = value as? Int { return number }
        if let number = value as? NSNumber { return number.intValue }
        return 0
    }
}

public struct Ernie55MoeSupportError: Error, CustomStringConvertible, Equatable {
    public let message: String

    public var description: String { message }

    public init(_ message: String) {
        self.message = message
    }
}

public struct Ernie55MoeConfiguration: Decodable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var vocabSize: Int
    public var nRoutedExperts: Int
    public var numExpertsPerTok: Int
    public var nSharedExperts: Int
    public var moeIntermediateSize: Int
    public var moeLatentSize: Int
    public var intermediateSize: Int
    public var firstKDenseReplace: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var compressRatios: [Int]
    public var maxPositionEmbeddings: Int
    public var rmsNormEps: Float
    public var tieWordEmbeddings: Bool

    public var heavilyCompressedLayerCount: Int { compressRatios.filter { $0 > 0 }.count }
    public var uncompressedAttentionLayerCount: Int { compressRatios.filter { $0 < 0 }.count }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nSharedExperts = "n_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeLatentSize = "moe_latent_size"
        case intermediateSize = "intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case compressRatios = "compress_ratios"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ernie5_5_moe"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 1
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 1
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 1
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 1
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? hiddenSize
        moeLatentSize = try c.decodeIfPresent(Int.self, forKey: .moeLatentSize) ?? hiddenSize
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? hiddenSize
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 1
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 16
        compressRatios = try c.decodeIfPresent([Int].self, forKey: .compressRatios) ?? []
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

final class Ernie55MoeStub: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Ernie55MoeConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Fixture graph only. Official checkpoints never construct this type.
        _ = cache
        return norm(embedTokens(inputs))
    }
}

public final class Ernie55MoeModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    let configuration: Ernie55MoeConfiguration
    @ModuleInfo(key: "model") var model: Ernie55MoeStub
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { [] }

    public init(_ configuration: Ernie55MoeConfiguration) {
        self.configuration = configuration
        vocabularySize = configuration.vocabSize
        kvHeads = Array(repeating: configuration.numKeyValueHeads, count: configuration.numHiddenLayers)
        _model.wrappedValue = Ernie55MoeStub(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(configuration.hiddenSize, configuration.vocabSize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        _ = parameters
        return (0 ..< configuration.numHiddenLayers).map { _ in KVCacheSimple() }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
