import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing
import NovaMLXCore
@testable import NovaMLXEngine

@Suite("MiniMax-M3.1-Flash text", .serialized)
struct MiniMaxM3Tests {
    @Test("Nested text_config decodes the published M3.1-Flash scalars")
    func decodesPublishedTextConfig() throws {
        let data = """
        {
          "model_type": "minimax_m3_vl",
          "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
          "text_config": {
            "hidden_size": 6144,
            "intermediate_size": 3072,
            "dense_intermediate_size": 12288,
            "shared_intermediate_size": 3072,
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 200064,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "partial_rotary_factor": 1.0,
            "hidden_act": "swigluoai",
            "use_qk_norm": true,
            "use_gemma_norm": true,
            "tie_word_embeddings": false,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "use_routing_bias": true,
            "routed_scaling_factor": 2.0,
            "swiglu_alpha": 1.702,
            "swiglu_limit": 7.0,
            "moe_layer_freq": [0, 1],
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.5},
            "sparse_attention_config": {
              "sparse_index_dim": 128,
              "sparse_num_index_heads": 4,
              "sparse_topk_blocks": 16,
              "sparse_block_size": 128,
              "sparse_score_type": "max",
              "sparse_init_block": 0,
              "sparse_local_block": 1,
              "sparse_attention_freq": [1, 1]
            }
          }
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: data)
        #expect(config.modelType == "minimax_m3_vl")
        #expect(config.text.hiddenSize == 6144)
        #expect(config.text.intermediateSize == 3072)
        #expect(config.text.denseIntermediateSize == 12288)
        #expect(config.text.numLocalExperts == 128)
        #expect(config.text.numExpertsPerTok == 4)
        #expect(config.text.vocabSize == 200064)
        #expect(config.text.maxPositionEmbeddings == 1_048_576)
        #expect(config.text.ropeTheta == Float(10_000_000))
        #expect(config.text.partialRotaryFactor == Float(0.5))
        #expect(config.text.rotaryDim == 64)
        #expect(config.text.swigluAlpha == Float(1.702))
        #expect(config.text.swigluLimit == Float(7))
        #expect(config.text.routedScalingFactor == Float(2))
        #expect(config.text.useGemmaNorm)
        #expect(config.text.useQKNorm)
        #expect(config.text.indexNHeads == 4)
        #expect(config.text.indexHeadDim == 128)
        #expect(config.text.indexBlockSize == 128)
        #expect(config.text.indexTopkBlocks == 16)
        #expect(config.text.indexLocalBlocks == 1)
        #expect(!config.text.isMoE(at: 0))
        #expect(config.text.isMoE(at: 1))
        #expect(config.text.isSparseAttention(at: 0))
        #expect(config.text.isSparseAttention(at: 1))
        // 16 blocks * 128 tokens. Shorter prompts stay on the causal mask.
        #expect((config.text.indexTopkBlocks ?? 0) * (config.text.indexBlockSize ?? 0) == 2048)
    }

    @Test("A flat re-save without text_config still decodes")
    func decodesFlatConfig() throws {
        let data = """
        {
          "model_type": "minimax_m3_vl",
          "hidden_size": 32,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "num_hidden_layers": 1,
          "head_dim": 8,
          "vocab_size": 16,
          "num_local_experts": 2,
          "num_experts_per_tok": 1,
          "index_n_heads": 2,
          "index_head_dim": 8,
          "index_block_size": 4,
          "index_topk_blocks": 1
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(MiniMaxM3Configuration.self, from: data)
        #expect(config.text.hiddenSize == 32)
        #expect(config.text.rotaryDim == 4)
        #expect(config.text.isMoE(at: 0))
        #expect(config.text.isSparseAttention(at: 0))
    }

    @Test("SwiGLU-OAI clips the gate from above and adds one to the linear branch")
    func swigluOAIMatchesReference() {
        let up = MLXArray([Float(1), Float(-8), Float(0)])
        let gate = MLXArray([Float(1), Float(8), Float(-1)])
        let y = minimaxSwigluOAI(up: up, gate: gate, alpha: 1.702, limit: 7)
        MLX.eval(y)
        let expected = [referenceSwiglu(1, 1), referenceSwiglu(8, -8), referenceSwiglu(-1, 0)]
        for index in 0..<3 {
            let got = y[index].item(Float.self)
            #expect(abs(got - expected[index]) < Float(1e-4))
        }
    }

    @Test("Registry builds a tiny model and a short sequence stays finite")
    func registryAndTinyForward() async throws {
        await CustomModelRegistration.ensureRegistered()
        let registry = LLMTypeRegistry.shared
        #expect(await registry.contains("minimax_m3_vl"))
        let data = tinyConfigJSON(blockSize: 4, topkBlocks: 1)
        let created = try await registry.createModel(configuration: data, modelType: "minimax_m3_vl")
        let model = try #require(created as? MiniMaxM3Model)
        #expect(model.vocabularySize == 32)
        #expect(model.kvHeads == [2, 2])
        #expect(model.reasoningConfig?.startDelimiter == "<mm:think>")
        #expect(model.reasoningConfig?.endDelimiter == "</mm:think>")
        #expect(model.reasoningConfig?.promptStrategy == .none)
        #expect(MiniMaxM3.attribution == "Built with MiniMax M3.1-Flash")

        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32](repeating: 0, count: 6).enumerated().map { Int32($0.offset + 1) })
            .reshaped([1, 6])
        let logits = model(tokens, cache: cache)
        MLX.eval(logits)
        #expect(logits.shape == [1, 6, 32])
        #expect(logits.asType(.float32).abs().max().item(Float.self).isFinite)
        #expect(cache[0].offset == 6)
        #expect(cache[0].trim(2) == 2)
        #expect(cache[0].offset == 4)

        let step = MLXArray(Int32(3)).reshaped([1, 1])
        let next = model(step, cache: cache)
        MLX.eval(next)
        #expect(next.shape == [1, 1, 32])
        #expect(next.asType(.float32).abs().max().item(Float.self).isFinite)
        #expect(cache[0].offset == 5)
    }

    @Test("Top-k covering every block uses the causal mask and still runs")
    func denseFallbackForward() throws {
        let model = MiniMaxM3Model(MiniMaxM3Configuration(text: tinyText(blockSize: 8, topkBlocks: 4)))
        let cache = try model.newCache(parameters: nil)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, cache: cache)
        MLX.eval(logits)
        #expect(logits.shape == [1, 3, 32])
        #expect(cache[0].offset == 3)
    }

    @Test("Sanitize renames language_model, stacks experts, and drops the vision tower")
    func sanitizeRemapsCheckpointKeys() {
        let text = tinyText(blockSize: 4, topkBlocks: 1)
        var weights: [String: MLXArray] = [
            "language_model.lm_head.weight": MLXArray.ones([4, 2]),
            "language_model.model.embed_tokens.weight": MLXArray.ones([4, 2]),
            "vision_tower.proj.weight": MLXArray.ones([2, 2]),
            "dspark.draft.weight": MLXArray.ones([2, 2]),
            "language_model.model.layers.0.block_sparse_moe.gate_proj.weight": MLXArray.ones([3, 2]),
            "language_model.model.layers.0.block_sparse_moe.up_proj.weight": MLXArray.ones([3, 2]),
            "language_model.model.layers.0.block_sparse_moe.down_proj.weight": MLXArray.ones([2, 3]),
        ]
        for expert in 0..<2 {
            let scale = Float(expert + 1)
            weights["language_model.model.layers.1.block_sparse_moe.experts.\(expert).w1.weight"] =
                MLXArray.ones([2, 2]) * scale
            weights["language_model.model.layers.1.block_sparse_moe.experts.\(expert).w2.weight"] =
                MLXArray.ones([2, 2]) * scale
            weights["language_model.model.layers.1.block_sparse_moe.experts.\(expert).w3.weight"] =
                MLXArray.ones([2, 2]) * scale
        }
        let sanitized = MiniMaxM3Sanitizer.remap(weights, text: text)
        #expect(sanitized["model.lm_head.weight"] != nil)
        #expect(sanitized["model.model.embed_tokens.weight"] != nil)
        #expect(sanitized["vision_tower.proj.weight"] == nil)
        #expect(sanitized["dspark.draft.weight"] == nil)
        #expect(sanitized["model.model.layers.0.mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.model.layers.0.block_sparse_moe.gate_proj.weight"] == nil)
        let stacked = sanitized["model.model.layers.1.block_sparse_moe.switch_mlp.gate_proj.weight"]
        #expect(stacked?.shape == [2, 2, 2])
        #expect(sanitized["model.model.layers.1.block_sparse_moe.experts.0.w1.weight"] == nil)
        let down = sanitized["model.model.layers.1.block_sparse_moe.switch_mlp.down_proj.weight"]
        #expect(down?.shape == [2, 2, 2])
    }

    @Test("Memory gate refuses a large resident checkpoint under 256 GB without TIE")
    func memoryGate() throws {
        let large = MiniMaxM3.text3bitBytes
        let mac128: UInt64 = 128 * 1024 * 1024 * 1024
        let mac256: UInt64 = 256 * 1024 * 1024 * 1024
        let refused = MiniMaxM3Memory.refusalMessage(
            modelId: MiniMaxM3.text3bitId,
            weightBytes: large,
            physicalBytes: mac128,
            tieReady: false
        )
        #expect(refused?.contains("NovaMLX-TIE") == true)
        #expect(
            MiniMaxM3Memory.refusalMessage(
                modelId: MiniMaxM3.text4bitId,
                weightBytes: MiniMaxM3.text4bitBytes,
                physicalBytes: mac256,
                tieReady: false
            ) == nil
        )
        #expect(
            MiniMaxM3Memory.refusalMessage(
                modelId: MiniMaxM3.text3bitId,
                weightBytes: large,
                physicalBytes: mac128,
                tieReady: true
            ) == nil
        )
        #expect(
            MiniMaxM3Memory.refusalMessage(
                modelId: "tiny",
                weightBytes: 1024,
                physicalBytes: mac128,
                tieReady: false
            ) == nil
        )

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("minimax-index-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let index = """
        {"metadata":{"total_size":\(MiniMaxM3.text3bitBytes)}}
        """.data(using: .utf8)!
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        #expect(MiniMaxM3Memory.indexedWeightBytes(at: directory) == MiniMaxM3.text3bitBytes)
        #expect(MiniMaxM3Memory.isTextCheckpoint(["model_type": "minimax_m3_vl"]))
        #expect(MiniMaxM3Memory.isTextCheckpoint([
            "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
        ]))
        #expect(!MiniMaxM3Memory.isTextCheckpoint(["model_type": "qwen2"]))
        let manifest = directory.appendingPathComponent("tier-manifest.json")
        try Data("{}".utf8).write(to: manifest)
        #expect(MiniMaxM3Memory.hasTieManifest(at: directory))
    }

    private func referenceSwiglu(_ gate: Float, _ up: Float) -> Float {
        let limit: Float = 7
        let alpha: Float = 1.702
        let clippedGate = min(gate, limit)
        let clippedUp = min(max(up, -limit), limit)
        let sig = 1 / (1 + exp(-alpha * clippedGate))
        return (clippedUp + 1) * (clippedGate * sig)
    }

    private func tinyText(blockSize: Int, topkBlocks: Int) -> MiniMaxM3TextConfiguration {
        MiniMaxM3TextConfiguration(
            hiddenSize: 32,
            intermediateSize: 16,
            denseIntermediateSize: 16,
            sharedIntermediateSize: 8,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            numHiddenLayers: 2,
            numLocalExperts: 2,
            numExpertsPerTok: 1,
            vocabSize: 32,
            headDim: 8,
            rotaryDim: 4,
            moeLayerFreq: [0, 1],
            sparseAttentionFreq: [1, 1],
            indexNHeads: 2,
            indexHeadDim: 8,
            indexBlockSize: blockSize,
            indexTopkBlocks: topkBlocks,
            indexLocalBlocks: 1
        )
    }

    private func tinyConfigJSON(blockSize: Int, topkBlocks: Int) -> Data {
        """
        {
          "model_type": "minimax_m3_vl",
          "text_config": {
            "hidden_size": 32,
            "intermediate_size": 16,
            "dense_intermediate_size": 16,
            "shared_intermediate_size": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "head_dim": 8,
            "rotary_dim": 4,
            "vocab_size": 32,
            "num_local_experts": 2,
            "num_experts_per_tok": 1,
            "moe_layer_freq": [0, 1],
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_block_size": \(blockSize),
            "index_topk_blocks": \(topkBlocks),
            "index_local_blocks": 1,
            "sparse_attention_freq": [1, 1]
          }
        }
        """.data(using: .utf8)!
    }
}
