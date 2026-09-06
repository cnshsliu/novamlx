import Testing
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
@testable import NovaMLXEngine
@testable import NovaMLXModelManager
import NovaMLXCore

@Suite("HyV4 / Hy4-preview loader")
struct HyV4Tests {

    private var tinyJSON: Data {
        """
        {
          "model_type": "hy_v4",
          "hidden_size": 32,
          "intermediate_size": 64,
          "moe_intermediate_size": 16,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "q_lora_rank": 16,
          "kv_lora_rank": 8,
          "qk_nope_head_dim": 8,
          "qk_rope_head_dim": 4,
          "v_head_dim": 8,
          "index_head_dim": 8,
          "index_n_heads": 2,
          "index_topk": 8,
          "n_routed_experts": 4,
          "n_shared_experts": 1,
          "num_experts_per_tok": 2,
          "hc_mult": 2,
          "vocab_size": 32,
          "max_position_embeddings": 64,
          "num_nextn_predict_layers": 1,
          "indexer_types": ["full", "shared"],
          "mlp_layer_types": ["dense", "sparse"],
          "rms_norm_eps": 1e-5,
          "gating_type": "elementwise",
          "rope_parameters": {"rope_theta": 10000, "rope_type": "default"}
        }
        """.data(using: .utf8)!
    }

    @Test("LLMTypeRegistry has hy_v4")
    func registryContainsType() async throws {
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: tinyJSON, modelType: "hy_v4")
        } catch let error as ModelFactoryError {
            if case .unsupportedModelType(let t) = error {
                Issue.record("hy_v4 not registered — \(t)")
            }
        }
    }

    @Test("Tiny hy_v4 constructs CacheList per layer")
    func tinyConstructsCaches() throws {
        let config = try JSONDecoder().decode(HyV4Configuration.self, from: tinyJSON)
        let model = HyV4Model(config)
        #expect(model.kvHeads.count == 2)
        let cache = try model.newCache(parameters: nil)
        #expect(cache.count == 2)
        #expect(cache[0] is CacheList)
        #expect(cache[1] is CacheList)
        let draft = model.mtpNewCache(parameters: nil)
        #expect(draft.count == 1)
        #expect(draft[0] is CacheList)
    }

    @Test("Tiny hy_v4 forward pass")
    func tinyForward() throws {
        let config = try JSONDecoder().decode(HyV4Configuration.self, from: tinyJSON)
        let model = HyV4Model(config)
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray(Array(Int32(1) ..< Int32(5))).reshaped([1, 4])
        let logits = model(tokens, cache: cache)
        eval(logits)
        #expect(logits.shape == [1, 4, config.vocabSize])
    }

    @Test("sanitize keeps mtp_layers and splits kv_b_proj + fused experts")
    func sanitizeKeepsMtpAndSplits() {
        let config = try! JSONDecoder().decode(HyV4Configuration.self, from: tinyJSON)
        let model = HyV4Model(config)
        let heads = config.numAttentionHeads
        let headDim = config.qkNopeHeadDim + config.vHeadDim
        let kv = config.kvLoraRank
        let moe = config.moeIntermediateSize
        let hidden = config.hiddenSize
        let experts = config.nRoutedExperts

        var weights: [String: MLXArray] = [
            "model.layers.0.self_attn.kv_b_proj.weight": MLXArray.ones([heads * headDim, kv]),
            "model.layers.1.mlp.experts.gate_up_proj": MLXArray.ones([experts, 2 * moe, hidden]),
            "model.layers.1.mlp.experts.down_proj": MLXArray.ones([experts, hidden, moe]),
            "model.mtp_layers.0.eh_proj.weight": MLXArray.ones([hidden, 2 * hidden]),
            "model.mtp_layers.0.self_attn.kv_b_proj.weight": MLXArray.ones([heads * headDim, kv]),
            "model.layers.99.mlp.gate.weight": MLXArray.ones([1]),
        ]
        let out = model.sanitize(weights: weights)
        #expect(model.nativeMtpAvailable)
        #expect(out["model.layers.0.self_attn.embed_q.weight"] != nil)
        #expect(out["model.layers.0.self_attn.unembed_out.weight"] != nil)
        #expect(out["model.layers.0.self_attn.kv_b_proj.weight"] == nil)
        #expect(out["model.layers.1.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(out["model.layers.1.mlp.switch_mlp.up_proj.weight"] != nil)
        #expect(out["model.layers.1.mlp.switch_mlp.down_proj.weight"] != nil)
        #expect(out["model.mtp_layers.0.eh_proj.weight"] != nil)
        #expect(out["model.mtp_layers.0.self_attn.embed_q.weight"] != nil)
        #expect(out["model.layers.99.mlp.gate.weight"] == nil)
        #expect(model.mtpBlockSize == 3)
    }

    @Test("ModelDiscovery maps hy_v4 to hunyuan LLM")
    func discoveryRoutesHy4() throws {
        let modelId = "novamlx-test-hyv4-\(UUID().uuidString.prefix(8))"
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let configJSON: [String: Any] = [
            "model_type": "hy_v4",
            "architectures": ["HYV4ForCausalLM"],
            "hidden_size": 32,
            "vocab_size": 32,
        ]
        try JSONSerialization.data(withJSONObject: configJSON)
            .write(to: modelDir.appendingPathComponent("config.json"))
        try "{}".data(using: .utf8)!
            .write(to: modelDir.appendingPathComponent("tokenizer_config.json"))

        let found = ModelDiscovery().discover(in: NovaMLXPaths.modelsDir)
            .first { $0.modelId == modelId }
        #expect(found != nil)
        #expect(found?.family == .hunyuan)
        #expect(found?.modelType == .llm)
        #expect(found?.configModelType == "hy_v4")
    }
}
