import Foundation
import Testing
import MLX
import MLXLLM
import MLXLMCommon
@testable import NovaMLXEngine

@Suite("ERNIE-5.5-Flash-Base registration", .serialized)
struct Ernie55MoeTests {
    @Test("official config is a completions base and refuses the FP8 graph")
    func officialConfigRefusesFP8() throws {
        let json = try Self.jsonObject(Self.officialJSON)
        #expect(Ernie55MoeSupport.isTextModel(json))
        #expect(Ernie55MoeSupport.isOfficialFP8(json))
        #expect(Ernie55MoeSupport.needsFullKernels(json))
        let config = try JSONDecoder().decode(Ernie55MoeConfiguration.self, from: Data(Self.officialJSON.utf8))
        #expect(config.modelType == "ernie5_5_moe")
        #expect(config.hiddenSize == 4096)
        #expect(config.numHiddenLayers == 43)
        #expect(config.nRoutedExperts == 512)
        #expect(config.numExpertsPerTok == 10)
        #expect(config.moeLatentSize == 2048)
        #expect(config.vocabSize == 201216)
        #expect(config.compressRatios.count == 44)
        #expect(config.heavilyCompressedLayerCount == 37)
        #expect(config.uncompressedAttentionLayerCount == 7)
        let refusal = try #require(
            Ernie55MoeSupport.loadRefusal(modelId: "baidu/ERNIE-5.5-Flash-Base", json: json))
        #expect(refusal.contains("ernie55-128gb"))
        #expect(refusal.contains("POST /v1/completions"))
        #expect(refusal.contains("Heavily Compressed Attention"))
        let chat = try #require(
            Ernie55MoeSupport.completionsRefusal(
                modelId: "baidu/ERNIE-5.5-Flash-Base", config: json))
        #expect(chat.contains("/v1/messages"))
    }

    @Test("converted MLX config still waits on HCA and MLA kernels")
    func convertedConfigWaitsOnKernels() throws {
        let json = try Self.jsonObject("""
        {"model_type":"ernie5_5_moe","hidden_size":4096,"num_hidden_layers":43,
         "n_routed_experts":512,"quantization":{"bits":2,"group_size":32,"mode":"affine"}}
        """)
        #expect(Ernie55MoeSupport.isOfficialFP8(json) == false)
        let refusal = try #require(
            Ernie55MoeSupport.loadRefusal(modelId: "ernie-mlx", json: json))
        #expect(refusal.contains("Heavily Compressed Attention"))
        #expect(refusal.contains("ernie55-128gb"))
        #expect(refusal.contains("POST /v1/completions"))
    }

    @Test("a tiny fixture is registered and does not displace qwen2")
    func registryBuildsTinyFixture() async throws {
        await CustomModelRegistration.ensureRegistered()
        let registry = LLMTypeRegistry.shared
        #expect(await registry.contains("ernie5_5_moe"))
        #expect(await registry.contains("qwen2"))
        #expect(await registry.contains("flint"))

        let model = try await registry.createModel(
            configuration: Data(Self.tinyJSON.utf8), modelType: "ernie5_5_moe")
        let ernie = try #require(model as? Ernie55MoeModel)
        #expect(ernie.vocabularySize == 32)
        #expect(ernie.kvHeads == [1])

        do {
            _ = try await registry.createModel(
                configuration: Data(Self.officialJSON.utf8), modelType: "ernie5_5_moe")
            Issue.record("official ernie5_5_moe should refuse before allocation")
        } catch let error as Ernie55MoeSupportError {
            #expect(error.message.contains("POST /v1/completions"))
        }
    }

    private static func jsonObject(_ text: String) throws -> [String: Any] {
        let data = Data(text.utf8)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static let tinyJSON = """
    {"model_type":"ernie5_5_moe","hidden_size":16,"num_hidden_layers":1,"vocab_size":32,"n_routed_experts":2}
    """

    /// Counts match baidu/ERNIE-5.5-Flash-Base config.json. No weights.
    private static let officialJSON = """
    {
      "model_type": "ernie5_5_moe",
      "architectures": ["Ernie5_5_MoeForCausalLM"],
      "hidden_size": 4096,
      "num_hidden_layers": 43,
      "vocab_size": 201216,
      "n_routed_experts": 512,
      "num_experts_per_tok": 10,
      "n_shared_experts": 1,
      "moe_intermediate_size": 2048,
      "moe_latent_size": 2048,
      "intermediate_size": 14336,
      "first_k_dense_replace": 1,
      "num_attention_heads": 64,
      "num_key_value_heads": 1,
      "head_dim": 512,
      "max_position_embeddings": 1048576,
      "compress_ratios": [128,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,128,128,128,128,128,128,-2,-2],
      "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}
    }
    """
}
