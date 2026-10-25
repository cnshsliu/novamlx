import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import NovaMLXEngine

@Suite("Flint model")
struct FlintModelTests {

    /// RecursiveSI/flint config.json with the standard num_hidden_layers
    /// spelled out and the 49-entry layer_types list.
    static func configJSON(
        numHiddenLayers: Int = 49, ropeOverride: String? = nil, extra: String = ""
    ) -> String {
        let layerTypes = ([
            "full_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
            "full_attention",
        ] as [String]).map { "\"\($0)\"" }.joined(separator: ", ")
        let rope = ropeOverride ?? "\"rope_theta\": 10000.0,"
        return """
        {
          "architectures": ["FlintForCausalLM"],
          "model_type": "flint",
          "num_hidden_layers": \(numHiddenLayers),
          "head_dim": 128,
          "hidden_size": 2048,
          "intermediate_size": 6144,
          "num_attention_heads": 32,
          "num_key_value_heads": 4,
          "num_experts": 256,
          "num_experts_per_tok": 6,
          "moe_intermediate_size": 512,
          "shared_expert_intermediate_size": 1024,
          "mlp_only_layers": [0, 1],
          "sliding_window": 513,
          "rms_norm_eps": 1e-06,
          "vocab_size": 150144,
          \(rope)
          "tie_word_embeddings": false,
          "layer_types": [\(layerTypes)]\(extra.isEmpty ? "" : ",\n" + extra)
        }
        """
    }

    /// Official RecursiveSI/flint-int4 `mlx_export`: numeric markers plus a
    /// layer-index array and a string scale rule.
    static let officialExport = """
        "mlx_export": {
          "bits": 4,
          "embedding_bits": 8,
          "routed_expert_bits": 4,
          "scale_rule": "mse",
          "full_attention_layers": [0, 1, 48]
        }
        """

    @Test("configuration decodes the real RecursiveSI/flint config")
    func testConfigDecode() throws {
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self, from: Data(Self.configJSON().utf8))
        #expect(config.modelType == "flint")
        #expect(config.hiddenLayers == 49)
        #expect(config.ropeTheta == 10000.0)
        #expect(config.mlpOnlyLayers == [0, 1])
        #expect(config.layerTypes.count == 49)
        #expect(config.layerTypes.filter { $0 == "full_attention" }.count == 14)
        #expect(config.layerTypes.filter { $0 == "sliding_attention" }.count == 35)

        let model = FlintModel(config)
        #expect(model.vocabularySize == 150144)
        #expect(model.kvHeads.count == 49)
        #expect(model.kvHeads.allSatisfy { $0 == 4 })
        #expect(!model.isOfficialInt4)
    }

    @Test("mlx_export numeric markers survive the mixed official layout")
    func testOfficialInt4Markers() throws {
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self,
            from: Data(Self.configJSON(extra: Self.officialExport).utf8))
        #expect(config.mlxExport?["bits"] == 4)
        #expect(config.mlxExport?["embedding_bits"] == 8)
        #expect(config.mlxExport?["routed_expert_bits"] == 4)
        #expect(config.mlxExport?["scale_rule"] == nil)
        let model = FlintModel(config)
        #expect(model.isOfficialInt4)

        let other = try JSONDecoder.json5().decode(
            FlintConfiguration.self,
            from: Data(Self.configJSON(extra: #""mlx_export": {"bits": 8, "embedding_bits": 8, "routed_expert_bits": 4}"#).utf8))
        #expect(!FlintModel(other).isOfficialInt4)
    }

    @Test("rope_parameters.rope_theta overrides the top-level rope_theta")
    func testRopeParametersOverride() throws {
        let json = Self.configJSON(
            ropeOverride: "\"rope_theta\": 10000.0, \"rope_parameters\": {\"rope_theta\": 123456.0},")
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self, from: Data(json.utf8))
        #expect(config.ropeTheta == 123456.0)
    }

    @Test("newCache mixes rotating caches (SWA) and full attention caches")
    func testHybridCache() throws {
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self, from: Data(Self.configJSON().utf8))
        let model = FlintModel(config)
        let caches = try model.newCache(parameters: nil)
        #expect(caches.count == 49)
        #expect(caches.filter { $0 is RotatingKVCache }.count == 35)
        // Full-attention layers get a non-rotating cache.
        #expect(caches.filter { !($0 is RotatingKVCache) }.count == 14)
    }

    @Test("sanitize stacks per-expert weights into switch_mlp and views uint8 as uint32")
    func testSanitize() throws {
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self, from: Data(Self.configJSON().utf8))
        let model = FlintModel(config)

        var weights: [String: MLXArray] = [:]
        // Layer 0 is dense (mlp_only), layer 2 is MoE.
        for n in ["gate_proj", "up_proj", "down_proj"] {
            weights["model.layers.0.mlp.\(n).weight"] = MLXRandom.normal([6144, 2048])
        }
        for e in 0 ..< 256 {
            for n in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.2.mlp.experts.\(e).\(n).weight"] = MLXRandom.normal(
                    [512, 2048])
            }
        }
        weights["model.layers.2.mlp.gate.weight"] = MLXRandom.normal([256, 2048])

        let sanitized = model.sanitize(weights: weights)
        #expect(sanitized["model.layers.2.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.2.mlp.switch_mlp.gate_proj.weight"]!.dim(0) == 256)
        #expect(sanitized["model.layers.2.mlp.experts.0.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.mlp.gate_proj.weight"] != nil)  // dense untouched
        #expect(sanitized["model.layers.2.mlp.gate.weight"] != nil)  // router untouched

        // uint8 packed + scales → uint32 view (flint-int4 format).
        var packed: [String: MLXArray] = [
            "model.layers.2.mlp.experts.0.gate_proj.weight": MLXRandom.uniform(low: 0, high: 255, [1024]).asType(.uint8)
        ]
        packed["model.layers.2.mlp.experts.0.gate_proj.scales"] = MLXRandom.normal([1024])
        let viewed = model.sanitize(weights: packed)
        #expect(viewed["model.layers.2.mlp.experts.0.gate_proj.weight"]?.dtype == .uint32)
    }
}
