import Testing
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
@testable import NovaMLXEngine
@testable import NovaMLXModelManager
import NovaMLXCore

@Suite("Qwen4-Exp / Qwen3.8-Flash-Next loader")
struct Qwen4ExpTests {

    private var tinyTextJSON: Data {
        """
        {
          "model_type": "qwen4_exp_text",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "linear_num_value_heads": 4,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 8,
          "linear_value_head_dim": 8,
          "linear_conv_kernel_dim": 4,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "shared_expert_intermediate_size": 16,
          "moe_intermediate_size": 16,
          "vocab_size": 32,
          "max_position_embeddings": 64,
          "hc_count": 2,
          "hc_lowrank": 8,
          "full_attention_interval": 2,
          "ple_layer_ids": [1],
          "ple_embed_dim": 32,
          "ngram_size": 2,
          "heads_per_ngram": 1,
          "ngram_vocab_size_base": 11,
          "make_ngram_vocab_size_divisible_by": 4,
          "split_ngram_parts": 2,
          "indexer_n_heads": 2,
          "indexer_kv_heads": 1,
          "indexer_head_dim": 8,
          "indexer_budget": 8,
          "indexer_compress_ratio": 4,
          "eos_token_id": 0,
          "rms_norm_eps": 1e-6
        }
        """.data(using: .utf8)!
    }

    private var tinyNestedJSON: Data {
        """
        {
          "model_type": "qwen4_exp",
          "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "shared_expert_intermediate_size": 16,
            "moe_intermediate_size": 16,
            "vocab_size": 32,
            "max_position_embeddings": 64,
            "hc_count": 2,
            "hc_lowrank": 8,
            "full_attention_interval": 2,
            "ple_layer_ids": [],
            "ple_embed_dim": 32,
            "ngram_size": 2,
            "heads_per_ngram": 1,
            "ngram_vocab_size_base": 11,
            "make_ngram_vocab_size_divisible_by": 4,
            "split_ngram_parts": 2,
            "indexer_n_heads": 2,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 8,
            "indexer_budget": 8,
            "indexer_compress_ratio": 4,
            "eos_token_id": 0
          }
        }
        """.data(using: .utf8)!
    }

    @Test("LLMTypeRegistry has qwen4_exp and qwen4_exp_text")
    func registryContainsTypes() async throws {
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: tinyNestedJSON, modelType: "qwen4_exp")
        } catch let error as ModelFactoryError {
            if case .unsupportedModelType(let t) = error {
                Issue.record("qwen4_exp not registered — \(t)")
            }
        }
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: tinyTextJSON, modelType: "qwen4_exp_text")
        } catch let error as ModelFactoryError {
            if case .unsupportedModelType(let t) = error {
                Issue.record("qwen4_exp_text not registered — \(t)")
            }
        }
    }

    @Test("Tiny qwen4_exp_text constructs mixed GDN/QSA caches")
    func tinyConstructsCaches() throws {
        let config = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: tinyTextJSON)
        let model = Qwen4ExpTextModel(config)
        #expect(model.kvHeads.count == 2)
        let cache = try model.newCache(parameters: nil)
        #expect(cache.count == 2)
        #expect(cache[0] is ArraysCache)
        #expect(cache[1] is QSAKVCache)
    }

    @Test("QSA-only tiny model runs a forward pass")
    func tinyQSAForward() throws {
        var json = try JSONSerialization.jsonObject(with: tinyTextJSON) as! [String: Any]
        json["num_hidden_layers"] = 1
        json["full_attention_interval"] = 1
        json["ple_layer_ids"] = [Int]()
        json["layer_types"] = ["qwen_sparse_attention"]
        let data = try JSONSerialization.data(withJSONObject: json)
        let config = try JSONDecoder().decode(Qwen4ExpTextConfiguration.self, from: data)
        let model = Qwen4ExpTextModel(config)
        let cache = try model.newCache(parameters: nil)
        #expect(cache.count == 1)
        #expect(cache[0] is QSAKVCache)
        let tokens = MLXArray(Array(Int32(1) ..< Int32(5))).reshaped([1, 4])
        let logits = model(tokens, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 4, config.vocabularySize])
    }

    @Test("sanitize remaps language_model prefix, conv1d, and shard keys")
    func sanitizeRemapsKeys() {
        let unsanitized = MLXArray.ones([8, 1, 4])
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.linear_attn.conv1d.weight": unsanitized,
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight":
                MLXArray.ones([4, 8]),
            "vision_tower.patch_embed.proj.weight": MLXArray.ones([1]),
            "mtp.layers.0.self_attn.q_proj.weight": MLXArray.ones([1]),
        ]
        let out = Qwen4ExpModel.remapWeights(weights, layerCount: 2)
        #expect(out["vision_tower.patch_embed.proj.weight"] == nil)
        #expect(out.keys.contains { $0.contains("mtp.") } == false)
        let convKey = "language_model.model.layers.0.linear_attn.conv1d.weight"
        #expect(out[convKey] != nil)
        #expect(out[convKey]?.dim(-1) == 1)
        #expect(out[convKey]?.shape == [8, 4, 1])
        #expect(
            out["language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"]
                != nil)
    }

    @Test("ModelDiscovery maps qwen4_exp to VLM / qwen")
    func discoveryRoutesFlashNext() throws {
        let modelId = "novamlx-test-qwen4-\(UUID().uuidString.prefix(8))"
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let configJSON: [String: Any] = [
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "vision_config": ["model_type": "qwen4_exp"],
            "text_config": ["model_type": "qwen4_exp_text"],
        ]
        try JSONSerialization.data(withJSONObject: configJSON)
            .write(to: modelDir.appendingPathComponent("config.json"))
        try "{}".data(using: .utf8)!
            .write(to: modelDir.appendingPathComponent("tokenizer_config.json"))

        let found = ModelDiscovery().discover(in: NovaMLXPaths.modelsDir)
            .first { $0.modelId == modelId }
        #expect(found != nil)
        #expect(found?.family == .qwen)
        #expect(found?.modelType == .vlm)
        #expect(found?.configModelType == "qwen4_exp")
    }
}
