import Foundation
import MLX
import MLXLMCommon
import NovaMLXPrefixCache
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

    @Test("official INT4 newCache wraps eager caches for rec trim")
    func testOfficialCacheProxy() throws {
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self,
            from: Data(Self.configJSON(extra: Self.officialExport).utf8))
        let model = FlintModel(config)
        let caches = try model.newCache(parameters: nil)
        let proxies = caches.compactMap { $0 as? FlintRecCacheProxy }
        #expect(proxies.count == 49)
        #expect(proxies.map(\.inner).filter { $0 is RotatingKVCache }.count == 35)
        #expect(proxies.map(\.inner).filter { !($0 is RotatingKVCache) }.count == 14)
        // Prefix cache must still see the sliding rings through the proxy,
        // or the rotating-cache skip turns off and a later restore flattens them.
        let classes = CacheBlockExtractor.getLayerClasses(from: caches)
        #expect(classes.filter { $0 == "RotatingKVCache" }.count == 35)
        #expect(classes.filter { $0 == "KVCache" }.count == 14)
        #expect(CacheBlockExtractor.usesRotatingKV(caches))
        #expect(!CacheBlockExtractor.usesRotatingKV(Array(caches.prefix(2))))
    }

    @Test("rec trim drops the decode suffix and leaves eager prefill in place")
    func testRecTrimKeepsPrefill() throws {
        let config = try JSONDecoder.json5().decode(
            FlintConfiguration.self,
            from: Data(Self.configJSON(extra: Self.officialExport).utf8))
        let model = FlintModel(config)
        let caches = try model.newCache(parameters: nil)
        let proxy = try #require(caches[0] as? FlintRecCacheProxy)
        let prefill = proxy.inner.offset
        proxy.noteRecStep()
        proxy.noteRecStep()
        #expect(proxy.offset == prefill + 2)
        #expect(proxy.trim(1) == 1)
        #expect(proxy.recAhead == 1)
        #expect(proxy.inner.offset == prefill)
        #expect(proxy.offset == prefill + 1)
    }

    @Test("rec launch profile follows the chip name, with an explicit override")
    func testRecLaunchProfile() {
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M4 Pro") == .pro)
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M5 Pro") == .pro)
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M4 Max") == .max)
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M5 Max") == .max)
        #expect(FlintRecLaunch.resolve(deviceName: "") == .pro)
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M4 Max", override: "36") == .pro)
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M4 Pro", override: "76") == .max)
        #expect(FlintRecLaunch.resolve(deviceName: "Apple M4 Pro", override: "nope") == .pro)
        #expect(FlintRecLaunch.pro.label == "36/32/32")
        #expect(FlintRecLaunch.max.label == "76/64/64")
        // 36 uses 4 KV teams of 9. 76 uses 4 teams of 19 and a 64-wide head.
        #expect(FlintRecKernelSources.bodyPro.contains("kv_team=group/9"))
        #expect(FlintRecKernelSources.bodyMax.contains("work/19"))
        #expect(FlintRecKernelSources.headerMax.contains("group/19"))
        #expect(FlintRecKernelSources.bodyMax.contains("if(group<64) q4_local<2048>(weight_matrix(tail,final+30)"))
        #expect(FlintRecKernelSources.bodyPro.contains("if(group<32) q4_local<2048>(weight_matrix(tail,final+30)"))
        #expect(FlintRecKernelSources.bodyPro.contains("#define routing routing_rw"))
        #expect(FlintRecKernelSources.bodyMax.contains("#define routing routing_rw"))
    }

    @Test("rec handoff waits until one single-token forward has landed")
    func testRecHandoffReady() {
        // Wide prefill, and the first one-token call (prompt tail or first
        // decode — they look the same), stay eager.
        #expect(!FlintModel.recHandoffReady(seq: 200, cacheOffset: 0, continuing: false, sawSingleToken: false))
        #expect(!FlintModel.recHandoffReady(seq: 1, cacheOffset: 0, continuing: false, sawSingleToken: false))
        #expect(!FlintModel.recHandoffReady(seq: 1, cacheOffset: 100, continuing: false, sawSingleToken: false))
        #expect(!FlintModel.recHandoffReady(seq: 1, cacheOffset: 512, continuing: false, sawSingleToken: false))
        // The next one-token call is decode. The eager token is already in KV.
        #expect(FlintModel.recHandoffReady(seq: 1, cacheOffset: 101, continuing: false, sawSingleToken: true))
        #expect(FlintModel.recHandoffReady(seq: 1, cacheOffset: 513, continuing: false, sawSingleToken: true))
        // An in-flight session also verifies multi-token drafts.
        #expect(FlintModel.recHandoffReady(seq: 5, cacheOffset: 100, continuing: true, sawSingleToken: true))
        #expect(!FlintModel.recHandoffReady(seq: 5, cacheOffset: 100, continuing: false, sawSingleToken: true))
    }

    @Test("sliding KV unwind puts the oldest token first")
    func testSlidingUnwind() {
        func feed(_ cache: RotatingKVCache, count: Int) {
            for t in 0 ..< count {
                var raw = [Float](repeating: 0, count: 4 * 128)
                raw[0] = Float(t)
                let token = MLXArray(raw).reshaped(1, 4, 1, 128).asType(.bfloat16)
                _ = cache.update(keys: token, values: token)
            }
        }
        func seq(_ cache: KVCache) -> [Float] {
            let pair = FlintRecCacheBridge.orderedPair(from: cache)!
            let keys = pair.0.asType(.float32)
            return (0 ..< keys.dim(2)).map { keys[0, 0, $0, 0].item(Float.self) }
        }

        let warm = RotatingKVCache(maxSize: 4, step: 4)
        feed(warm, count: 3)
        #expect(seq(warm) == [0, 1, 2])

        let wrapped = RotatingKVCache(maxSize: 4, step: 4)
        feed(wrapped, count: 6)
        #expect(seq(wrapped) == [2, 3, 4, 5])
    }
}
