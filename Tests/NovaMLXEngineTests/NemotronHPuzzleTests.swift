import Foundation
import Testing
import MLX
import MLXLMCommon
import NovaMLXCore
@testable import NovaMLXEngine

@Suite("Nemotron-H Puzzle text model", .serialized)
struct NemotronHPuzzleTests {
    @Test("Bolt-like config decodes heterogeneous experts and an infinite time limit")
    func decodesBoltSnippet() throws {
        let data = """
        {
          "model_type": "nemotron_h_puzzle",
          "vocab_size": 131072,
          "hidden_size": 4096,
          "rms_norm_eps": 1e-5,
          "moe_latent_size": 1024,
          "n_shared_experts": 1,
          "routed_scaling_factor": 5.0,
          "time_step_limit": [0.0, {"__float__": "Infinity"}],
          "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "moe_intermediate_size": 2048, "n_routed_experts": 240, "num_experts_per_tok": 9},
            {"block_type": "attention"},
            {"block_type": "moe", "moe_intermediate_size": 1536, "n_routed_experts": 304, "num_experts_per_tok": 8}
          ]
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(NemotronHPuzzleConfiguration.self, from: data)
        #expect(config.numHiddenLayers == 4)
        #expect(config.hybridOverridePattern == "ME*E")
        #expect(config.routedExpertCount(at: 1) == 240)
        #expect(config.routedExpertCount(at: 3) == 304)
        #expect(config.moeWidth(at: 1) == 2048)
        #expect(config.moeWidth(at: 3) == 1536)
        #expect(config.expertsPerToken(at: 3) == 8)
        #expect(config.moeLatentSize == 1024)
        #expect(config.layerNormEpsilon == 1e-5)
        #expect(config.timeStepLimitMin == 0)
        #expect(config.timeStepLimitMax.isInfinite)
        #expect(config.routedScalingFactor == 5)
    }

    @Test("Mismatched hybrid_override_pattern is rejected")
    func rejectsMismatchedPattern() {
        let data = """
        {
          "vocab_size": 32,
          "hidden_size": 16,
          "hybrid_override_pattern": "M*",
          "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "moe", "n_routed_experts": 2}
          ]
        }
        """.data(using: .utf8)!
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(NemotronHPuzzleConfiguration.self, from: data)
        }
    }

    @Test("Registry builds a tiny puzzle model and keeps qwen2 registered")
    func registrationBuildsTinyModel() async throws {
        await CustomModelRegistration.ensureRegistered()
        let registry = LLMTypeRegistry.shared
        #expect(await registry.contains("nemotron_h_puzzle"))
        #expect(await registry.contains("qwen2"))
        #expect(await registry.contains("nemotron_h"))

        let data = tinyConfigJSON()
        let model = try await registry.createModel(configuration: data, modelType: "nemotron_h_puzzle")
        let puzzle = try #require(model as? NemotronHPuzzleModel)
        #expect(puzzle.toolCallFormat == .xmlFunction)
        #expect(puzzle.reasoningConfig?.promptStrategy == .templateFlag(key: "enable_thinking", defaultOn: true))
        #expect(puzzle.reasoningConfig?.implicitEndDelimiters == ["<tool_call>"])
        #expect(puzzle.kvHeads == [0, 2])
    }

    @Test("Sanitize stacks each MoE layer's own expert count and leaves pre-stacked weights")
    func sanitizeStacksPerLayer() throws {
        let model = NemotronHPuzzleModel(tinyConfiguration())
        let expert = MLXArray.ones([4, 8])
        var weights: [String: MLXArray] = [
            "language_model.backbone.norm_f.weight": MLXArray.ones([16]),
            "mtp.0.layers.0.weight": MLXArray.ones([2, 2]),
            "vision_tower.weight": MLXArray.ones([2, 2]),
            "backbone.layers.0.mixer.conv1d.weight": MLXArray.ones([2, 3, 4]),
            "backbone.layers.1.mixer.switch_mlp.fc2.weight": MLXArray.ones([2, 8, 4]),
        ]
        for expertIndex in 0 ..< 2 {
            weights["backbone.layers.1.mixer.experts.\(expertIndex).up_proj.weight"] = expert
            weights["backbone.layers.1.mixer.experts.\(expertIndex).up_proj.scales"] = MLXArray.ones([4])
        }
        for expertIndex in 0 ..< 3 {
            weights["backbone.layers.2.mixer.experts.\(expertIndex).up_proj.weight"] = expert
        }
        // A partial expert list must stay unstacked.
        weights["backbone.layers.2.mixer.experts.0.down_proj.weight"] = expert

        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized["mtp.0.layers.0.weight"] == nil)
        #expect(sanitized["vision_tower.weight"] == nil)
        #expect(sanitized["backbone.norm_f.weight"] != nil)
        #expect(sanitized["backbone.layers.1.mixer.experts.0.up_proj.weight"] == nil)
        #expect(sanitized["backbone.layers.1.mixer.switch_mlp.fc1.weight"]?.shape == [2, 4, 8])
        #expect(sanitized["backbone.layers.1.mixer.switch_mlp.fc1.scales"]?.shape == [2, 4])
        #expect(sanitized["backbone.layers.1.mixer.switch_mlp.fc2.weight"]?.shape == [2, 8, 4])
        #expect(sanitized["backbone.layers.2.mixer.switch_mlp.fc1.weight"]?.shape == [3, 4, 8])
        #expect(sanitized["backbone.layers.2.mixer.experts.0.down_proj.weight"] != nil)
        #expect(sanitized["backbone.layers.2.mixer.switch_mlp.fc2.weight"] == nil)
        #expect(sanitized["backbone.layers.0.mixer.conv1d.weight"]?.shape == [2, 4, 3])
    }

    @Test("Memory gate accepts 4-bit on 64 GB and 3-bit on 48 GB")
    func memoryGate() throws {
        let fourBit: UInt64 = 43_256_967_168
        let threeBit: UInt64 = 31_653_163_008
        let gibibyte: UInt64 = 1024 * 1024 * 1024
        #expect(
            NemotronHPuzzleMemory.refusalMessage(
                modelId: "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit",
                weightBytes: fourBit,
                physicalBytes: 64 * gibibyte
            ) == nil
        )
        let refused32 = NemotronHPuzzleMemory.refusalMessage(
            modelId: "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit",
            weightBytes: fourBit,
            physicalBytes: 32 * gibibyte
        )
        #expect(refused32?.contains("64 GB") == true)
        #expect(
            NemotronHPuzzleMemory.refusalMessage(
                modelId: "4bit",
                weightBytes: fourBit,
                physicalBytes: 48 * gibibyte
            ) != nil
        )
        #expect(
            NemotronHPuzzleMemory.refusalMessage(
                modelId: "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit",
                weightBytes: threeBit,
                physicalBytes: 48 * gibibyte
            ) == nil
        )
        #expect(
            NemotronHPuzzleMemory.refusalMessage(
                modelId: "3bit",
                weightBytes: threeBit,
                physicalBytes: 32 * gibibyte
            ) != nil
        )

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-bolt-index-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let index = """
        {"metadata":{"total_size":43256967168}}
        """.data(using: .utf8)!
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        #expect(NemotronHPuzzleMemory.indexedWeightBytes(at: directory) == fourBit)

        let puzzle: [String: Any] = [
            "model_type": "nemotron_h_puzzle",
            "architectures": ["NemotronHPuzzleForCausalLM"],
        ]
        #expect(NemotronHPuzzleMemory.isTextPuzzle(puzzle))
        #expect(NemotronHPuzzleMemory.isTextPuzzle(["model_type": "qwen2"]) == false)
    }

    private func tinyConfiguration() -> NemotronHPuzzleConfiguration {
        NemotronHPuzzleConfiguration(
            vocabSize: 32,
            hiddenSize: 16,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            mambaNumHeads: 2,
            mambaHeadDim: 4,
            ssmStateSize: 4,
            convKernel: 2,
            nGroups: 2,
            intermediateSize: 16,
            moeIntermediateSize: 8,
            moeSharedExpertIntermediateSize: 8,
            moeLatentSize: 8,
            nRoutedExperts: 2,
            nSharedExperts: 1,
            numExpertsPerTok: 1,
            blockConfigs: [
                NemotronHPuzzleBlockConfig(blockType: "mamba"),
                NemotronHPuzzleBlockConfig(
                    blockType: "moe", moeIntermediateSize: 8, nRoutedExperts: 2, numExpertsPerTok: 1),
                NemotronHPuzzleBlockConfig(
                    blockType: "moe", moeIntermediateSize: 12, nRoutedExperts: 3, numExpertsPerTok: 2),
                NemotronHPuzzleBlockConfig(blockType: "attention"),
            ],
            headDim: 8
        )
    }

    private func tinyConfigJSON() -> Data {
        """
        {
          "model_type": "nemotron_h_puzzle",
          "vocab_size": 32,
          "hidden_size": 16,
          "num_attention_heads": 2,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "mamba_num_heads": 2,
          "mamba_head_dim": 4,
          "ssm_state_size": 4,
          "n_groups": 2,
          "conv_kernel": 2,
          "intermediate_size": 16,
          "moe_intermediate_size": 8,
          "moe_shared_expert_intermediate_size": 8,
          "moe_latent_size": 8,
          "n_routed_experts": 2,
          "n_shared_experts": 1,
          "num_experts_per_tok": 1,
          "block_configs": [
            {"block_type": "mamba"},
            {"block_type": "attention"}
          ]
        }
        """.data(using: .utf8)!
    }
}
