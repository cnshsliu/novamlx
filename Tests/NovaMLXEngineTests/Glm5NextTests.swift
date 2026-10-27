import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import NovaMLXEngine

@Suite("GLM-5.3-Flash text model", .serialized)
struct Glm5NextTests {
    @Test("Nested text_config decodes glm5_next_text")
    func decodesNestedConfig() throws {
        let config = try Glm5NextConfiguration.decode(from: wrapperJSON())
        #expect(config.modelType == "glm5_next_text")
        #expect(config.vocabSize == 32)
        #expect(config.numHiddenLayers == 2)
        #expect(config.layerTypes == ["linear_attention", "deepseek_sparse_attention"])
        #expect(config.mlpLayerTypes == ["dense", "sparse"])
        #expect(config.nRoutedExperts == 2)
        #expect(config.numExpertsPerTok == 1)
        #expect(config.swigluLimit == 10)
        #expect(config.linearNumHeads == 2)
        #expect(config.linearHeadDim == 8)
        #expect(config.linearLowerBound == -5)
        #expect(config.hcMult == 2)
        #expect(config.indexTopk == 2)
        #expect(config.indexKPool == 2)
    }

    @Test("Mismatched head counts are rejected")
    func rejectsMismatchedHeads() {
        var object = tinyObject()
        object["num_key_value_heads"] = 1
        let data = try! JSONSerialization.data(withJSONObject: object)
        #expect(throws: DecodingError.self) {
            _ = try Glm5NextConfiguration.decode(from: data)
        }
    }

    @Test("Memory gate: 4-bit needs 256 GB, hybrid 2-bit needs 128 GB")
    func memoryGate() {
        let fourBit: UInt64 = 203_992_076_296
        let twoBit: UInt64 = 111_317_859_192
        let gibibyte: UInt64 = 1024 * 1024 * 1024
        #expect(Glm5NextMemory.requiredBytes(weightBytes: fourBit) == 256 * gibibyte)
        #expect(Glm5NextMemory.requiredBytes(weightBytes: twoBit) == 128 * gibibyte)
        #expect(Glm5NextMemory.requiredBytes(weightBytes: 1024) == 1024 + 16 * gibibyte)
        #expect(Glm5NextMemory.isTextModel([
            "model_type": "glm5_next",
            "vision_config": ["hidden_size": 8],
            "architectures": ["Glm5NextForConditionalGeneration"],
        ]))
        #expect(
            Glm5NextMemory.refusalMessage(
                modelId: "mlx-community/GLM-5.3-Flash-4bit",
                weightBytes: fourBit,
                physicalBytes: 256 * gibibyte
            ) == nil
        )
        let refused64 = Glm5NextMemory.refusalMessage(
            modelId: "mlx-community/GLM-5.3-Flash-4bit",
            weightBytes: fourBit,
            physicalBytes: 64 * gibibyte
        )
        #expect(refused64?.contains("256 GB") == true)
        #expect(
            Glm5NextMemory.refusalMessage(
                modelId: "TensorFold/GLM-5.3-Flash-MLX-2bit-MTP",
                weightBytes: twoBit,
                physicalBytes: 128 * gibibyte
            ) == nil
        )
        #expect(
            Glm5NextMemory.refusalMessage(
                modelId: "TensorFold/GLM-5.3-Flash-MLX-2bit-MTP",
                weightBytes: twoBit,
                physicalBytes: 64 * gibibyte
            ) != nil
        )
    }

    @Test("Registry builds a tiny glm5_next model")
    func registrationBuildsTinyModel() async throws {
        await CustomModelRegistration.ensureRegistered()
        let registry = LLMTypeRegistry.shared
        #expect(await registry.contains("glm5_next"))
        #expect(await registry.contains("glm5_next_text"))
        let model = try await registry.createModel(
            configuration: wrapperJSON(), modelType: "glm5_next")
        let glm = try #require(model as? Glm5NextModel)
        #expect(glm.vocabularySize == 32)
        #expect(glm.kvHeads == [0, 1])
    }

    @Test("Sanitize renames mHC, stacks experts, splits kv_b, and drops MTP and vision")
    func sanitizeRemapsCheckpointKeys() throws {
        let config = try Glm5NextConfiguration.decode(from: wrapperJSON())
        let model = Glm5NextModel(config)
        let expert = MLXArray.ones([16, 32])
        var weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": MLXArray.ones([32, 32]),
            "model.language_model.norm.weight": MLXArray.ones([32]),
            "lm_head.weight": MLXArray.ones([32, 32]),
            "model.language_model.layers.0.hc_attn_fn": MLXArray.ones([8, 64]),
            "model.language_model.layers.0.hc_ffn_base": MLXArray.ones([8]),
            "model.language_model.layers.0.self_attn.q_conv1d.weight": MLXArray.ones([16, 1, 4]),
            "model.language_model.layers.1.self_attn.kv_b_proj.weight": MLXArray.ones([32, 8]),
            "model.visual.merger.weight": MLXArray.ones([2, 2]),
            "model.language_model.layers.2.mlp.gate.weight": MLXArray.ones([2, 2]),
            "mtp.layers.0.weight": MLXArray.ones([2, 2]),
        ]
        for expertIndex in 0..<2 {
            weights["model.language_model.layers.1.mlp.experts.\(expertIndex).gate_proj.weight"] = expert
        }
        weights["model.language_model.layers.1.mlp.experts.0.up_proj.weight"] = expert

        let sanitized = model.sanitize(weights: weights)
        #expect(model.droppedMtpKeys == 2)
        #expect(sanitized["model.visual.merger.weight"] == nil)
        #expect(sanitized["mtp.layers.0.weight"] == nil)
        #expect(sanitized["language_model.model.layers.2.mlp.gate.weight"] == nil)
        #expect(sanitized["language_model.model.embed_tokens.weight"]?.shape == [32, 32])
        #expect(sanitized["language_model.lm_head.weight"]?.shape == [32, 32])
        #expect(sanitized["language_model.model.layers.0.attn_hc.fn"]?.shape == [8, 64])
        #expect(sanitized["language_model.model.layers.0.ffn_hc.base"]?.shape == [8])
        #expect(sanitized["language_model.model.layers.0.self_attn.q_conv1d.weight"]?.shape == [16, 4, 1])
        #expect(sanitized["language_model.model.layers.1.self_attn.kv_b_proj.weight"] == nil)
        #expect(sanitized["language_model.model.layers.1.self_attn.embed_q.weight"]?.shape == [2, 8, 8])
        #expect(sanitized["language_model.model.layers.1.self_attn.unembed_out.weight"]?.shape == [2, 8, 8])
        #expect(sanitized["language_model.model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)
        #expect(sanitized["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [2, 16, 32])
        #expect(sanitized["language_model.model.layers.1.mlp.experts.0.up_proj.weight"] != nil)
    }

    @Test("Tiny prefill and one decode step stay finite")
    func tinyForwardStaysFinite() throws {
        let config = try Glm5NextConfiguration.decode(from: wrapperJSON())
        let model = Glm5NextModel(config)
        let cache = try model.newCache(parameters: nil)
        let prompt = MLXArray((0..<9).map { Int32($0) }).reshaped([1, 9])
        let prefill = model(prompt, cache: cache)
        MLX.eval(prefill)
        #expect(prefill.shape == [1, 9, 32])
        #expect(prefill.asType(.float32).sum().item(Float.self).isFinite)
        let step = model(MLXArray(Int32(3)).reshaped([1, 1]), cache: cache)
        MLX.eval(step)
        #expect(step.shape == [1, 1, 32])
        #expect(step.asType(.float32).sum().item(Float.self).isFinite)
    }

    @Test("GLM-CHAT line excludes prefill and cites the community ceiling")
    func chatLineCitesCommunityCeiling() {
        let line = Glm5ChatDecode.line(
            decodeTokens: 32, seconds: 1, promptTokens: 12, mtp: .ignored)
        #expect(line?.contains("GLM-CHAT: 32.0 tok/s") == true)
        #expect(line?.contains("prefill excluded") == true)
        #expect(line?.contains("promptTokens=12") == true)
        #expect(line?.contains("mtp=ignored") == true)
        #expect(line?.contains("38.5") == true)
        #expect(line?.contains("https://z.ai/blog/glm-5.3-flash") == true)
        #expect(line?.contains("https://omlx.ai/benchmarks/performance/0g84suc9") == true)
        #expect(line?.contains("not that run") == true)
        #expect(Glm5ChatDecode.line(
            decodeTokens: 0, seconds: 1, promptTokens: 4, mtp: .absent) == nil)
        #expect(Glm5ChatDecode.line(
            decodeTokens: 4, seconds: 0, promptTokens: 4, mtp: .absent) == nil)
        #expect(Glm5ChatDecode.decodeEvalStride == 8)
    }
}

private func tinyObject() -> [String: Any] {
    [
        "model_type": "glm5_next_text",
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 16,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "kv_lora_rank": 8,
        "q_lora_rank": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 8,
        "swiglu_limit": 10,
        "tie_word_embeddings": false,
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "indexer_types": ["full", "full"],
        "index_topk": 2,
        "index_head_dim": 8,
        "index_n_heads": 2,
        "index_kpool": 2,
        "linear_attn_config": [
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5,
        ],
        "hc_mult": 2,
        "hc_sinkhorn_iters": 1,
        "first_k_dense_replace": 1,
    ]
}

private func wrapperJSON() -> Data {
    let wrapper: [String: Any] = [
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "vision_config": ["hidden_size": 8],
        "text_config": tinyObject(),
    ]
    return try! JSONSerialization.data(withJSONObject: wrapper)
}
