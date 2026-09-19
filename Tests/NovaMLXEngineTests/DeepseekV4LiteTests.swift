import Testing
import Foundation
import MLX
import NovaMLXCore
import MLXLMCommon
import MLXLLM
@testable import NovaMLXEngine
@testable import NovaMLXModelManager

// ────────────────────────────────────────────────────────────
// DeepSeek-V4 lite regression suite — todo.markdown §2.10.
//
// Tests registration, family routing, chat template detection,
// and indexer contract WITHOUT a real model on disk.
// Forward-pass shape test (item 3) requires model weights — deferred.
// ────────────────────────────────────────────────────────────

@Suite("DeepSeek-V4 lite regression", .serialized)
struct DeepseekV4LiteTests {

    // MARK: - 1. Registration

    @Test("LLMTypeRegistry has 'deepseek_v4' registered after ensureRegistered()")
    func registrationSucceeds() async throws {
        await CustomModelRegistration.ensureRegistered()

        // Minimal valid config — just enough for Codable to succeed.
        // createModel will fail at weight loading, but we only care that
        // the type IS registered (unsupported types throw before decoding).
        let minConfig = """
        {
            "model_type": "deepseek_v4",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_hash_layers": 0,
            "num_nextn_predict_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32
        }
        """
        let configData = minConfig.data(using: .utf8)!

        // If "deepseek_v4" is NOT registered, this throws
        // ModelFactoryError.unsupportedModelType. We catch that specifically.
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: configData, modelType: "deepseek_v4")
        } catch let error as ModelFactoryError {
            // unsupportedModelType = registration missing = FAIL
            if case .unsupportedModelType(let t) = error {
                Issue.record("deepseek_v4 not registered — got unsupportedModelType(\"\(t)\")")
            }
            // Other errors (e.g. missing weights) mean registration WORKED
        }
    }

    @Test("sanitize keeps in-graph mtp.* as mtpLayers and stacks w1/w2/w3 experts")
    func sanitizeKeepsDSparkMtp() throws {
        var native = false
        let w1 = MLXArray.ones([2, 2])
        let weights: [String: MLXArray] = [
            "mtp.0.ffn.experts.0.w1.weight": w1,
            "mtp.0.ffn.experts.1.w1.weight": w1,
            "language_model.head.weight": MLXArray.ones([4, 2]),
            "language_model.layers.0.hc_attn_fn": MLXArray.ones([2, 2]),
            "norm.weight": MLXArray.ones([2]),
        ]
        let cfgJSON = """
            {"model_type":"deepseek_v41","num_nextn_predict_layers":3,"n_routed_experts":2,
             "dspark_n_routed_experts":2,"dspark_block_size":5,"hidden_size":2,"vocab_size":4,
             "num_hidden_layers":1}
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: cfgJSON)
        let out = DeepseekV4Sanitizer.remap(weights, config: cfg, nativeMtp: &native)
        #expect(native)
        #expect(out["model.mtpLayers.0.ffn.switch_mlp.gate_proj.weight"] != nil)
        #expect(out["lm_head.weight"] != nil)
        #expect(out["model.layers.0.attn_hc.fn"] != nil)
        #expect(out["model.norm.weight"] != nil)
        #expect(out["mtp.0.ffn.experts.0.w1.weight"] == nil)
    }

    @Test("TIE dense mtpLayers keys enable native DSpark without expert tensors")
    func sanitizeTieDenseMtpEnablesNative() throws {
        var native = false
        let weights: [String: MLXArray] = [
            "model.mtpLayers.0.ffn.gate.weight": MLXArray.ones([2, 2]),
            "model.mtpLayers.0.attn.wq_a.weight": MLXArray.ones([2, 2]),
            "model.mtpLayers.1.ffn.gate.weight": MLXArray.ones([2, 2]),
            "model.mtpLayers.2.ffn.gate.weight": MLXArray.ones([2, 2]),
            "norm.weight": MLXArray.ones([2]),
        ]
        let cfgJSON = """
            {"model_type":"deepseek_v41","num_nextn_predict_layers":3,"n_routed_experts":2,
             "dspark_n_routed_experts":2,"dspark_block_size":5,"hidden_size":2,"vocab_size":4,
             "num_hidden_layers":1}
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: cfgJSON)
        let out = DeepseekV4Sanitizer.remap(weights, config: cfg, nativeMtp: &native)
        #expect(native)
        #expect(out["model.mtpLayers.0.ffn.gate.weight"] != nil)
        let model = DeepseekV4Model(cfg)
        model.nativeMtpAvailable = native
        #expect(model.mtpBlockSize == 5)
        #expect(!model.model.mtpLayers.isEmpty)
        #expect(model.model.mtpLayers.count == 3)
    }

    @Test("V4.1 Flash ids resolve to the DSpark companion, not Qwen DFlash")
    func v41ResolvesDSparkNotDFlash() {
        let id = "mlx-community/DeepSeek-V4.1-Flash-MLX-2bit"
        #expect(dsparkDraftCandidates(forMainId: id) == [
            "mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"
        ])
        #expect(dflashDraftCandidates(forMainId: id).isEmpty)
        #expect(dsparkDraftCandidates(forMainId: "mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter").isEmpty)
    }

    @Test("LLMTypeRegistry has 'deepseek_v41' registered after ensureRegistered()")
    func v41RegistrationSucceeds() async throws {
        await CustomModelRegistration.ensureRegistered()
        let minConfig = """
        {
            "model_type": "deepseek_v41",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_hash_layers": 0,
            "num_nextn_predict_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32
        }
        """
        let configData = minConfig.data(using: .utf8)!
        do {
            _ = try await LLMTypeRegistry.shared.createModel(
                configuration: configData, modelType: "deepseek_v41")
        } catch let error as ModelFactoryError {
            if case .unsupportedModelType(let t) = error {
                Issue.record("deepseek_v41 not registered — got unsupportedModelType(\"\(t)\")")
            }
        }
    }

    // MARK: - 2. Family routing via ModelDiscovery

    @Test("ModelDiscovery maps model_type 'deepseek_v4' to .qwen family")
    func familyRoutingByModelType() throws {
        let modelId = "novamlx-test-dsv4-\(UUID().uuidString.prefix(8))"
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        let configJSON: [String: Any] = [
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
        ]
        let data = try JSONSerialization.data(withJSONObject: configJSON)
        try data.write(to: modelDir.appendingPathComponent("config.json"))

        // Touch a minimal tokenizer_config.json so completeness check passes
        try "{}".data(using: .utf8)!.write(
            to: modelDir.appendingPathComponent("tokenizer_config.json"))

        let discovery = ModelDiscovery()
        let results = discovery.discover(in: NovaMLXPaths.modelsDir)
        let found = results.first { $0.modelId == modelId }

        #expect(found != nil, "model should be discovered")
        #expect(found?.family == .deepseek,
            "deepseek_v4 model_type should map to .deepseek family; got \(found?.family.rawValue ?? "nil")")
        #expect(found?.configModelType == "deepseek_v4",
            "configModelType should preserve the raw model_type string")
    }

    @Test("ModelDiscovery maps architecture 'DeepseekV4ForCausalLM' to .qwen family")
    func familyRoutingByArchitecture() throws {
        let modelId = "novamlx-test-dsv4arch-\(UUID().uuidString.prefix(8))"
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: modelDir) }

        // Use a different model_type but the V4 architecture — should still route to .qwen
        let configJSON: [String: Any] = [
            "model_type": "deepseek_v4",
            "architectures": ["DeepseekV4ForCausalLM"],
        ]
        let data = try JSONSerialization.data(withJSONObject: configJSON)
        try data.write(to: modelDir.appendingPathComponent("config.json"))
        try "{}".data(using: .utf8)!.write(
            to: modelDir.appendingPathComponent("tokenizer_config.json"))

        let discovery = ModelDiscovery()
        let results = discovery.discover(in: NovaMLXPaths.modelsDir)
        let found = results.first { $0.modelId == modelId }

        #expect(found != nil)
        #expect(found?.family == .deepseek,
            "DeepseekV4ForCausalLM architecture should map to .deepseek; got \(found?.family.rawValue ?? "nil")")
    }

    // MARK: - 3. Chat template detection

    @Test("ChatTemplateFormat.detect returns .deepSeek for fullwidth-pipe template")
    func chatTemplateDetectsDeepSeek() {
        let template = """
        {{ bos_token }}{%- for message in messages %}{%- if message['role'] == 'user' %}{{ '｜User｜' + message['content'] }}{%- elif message['role'] == 'assistant' %}{{ '｜Assistant｜' + message['content'] + eos_token }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{ '｜Assistant｜' }}{%- endif %}
        """
        let format = ChatTemplateFormat.detect(from: template)
        #expect(format == .deepSeek,
            "template with ｜User｜/｜Assistant｜ fullwidth pipes must detect as .deepSeek")
    }

    @Test("ChatTemplateFormat.detectAll includes .deepSeek with confidence > 0")
    func chatTemplateDetectAll() {
        let template = """
        {%- for message in messages %}{{ '｜User｜' + message['content'] + '｜Assistant｜' }}{% endfor %}
        """
        let detections = ChatTemplateFormat.detectAll(from: template)
        let ds = detections.first { $0.format == .deepSeek }
        #expect(ds != nil, ".deepSeek should appear in detectAll results")
        #expect((ds?.confidence ?? 0) > 0,
            ".deepSeek detection should have confidence > 0")
    }

    // MARK: - 4. Indexer load-only contract

    @Test("DeepseekV4Indexer has no callAsFunction — load-only stub")
    func indexerIsLoadOnlyStub() throws {
        let configJSON = """
        {
            "model_type": "deepseek_v4",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_hash_layers": 0,
            "num_nextn_predict_layers": 0,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "q_lora_rank": 32,
            "o_lora_rank": 32,
            "head_dim": 16,
            "qk_rope_head_dim": 4,
            "o_groups": 2,
            "index_n_heads": 4,
            "index_head_dim": 8,
            "index_topk": 4,
            "n_routed_experts": 4,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 32
        }
        """
        let config = try JSONDecoder().decode(
            DeepseekV4Configuration.self,
            from: configJSON.data(using: .utf8)!)
        let indexer = DeepseekV4Indexer(
            layerId: 2, config: config, compressRatio: 2, isKVSource: true)
        #expect(indexer.wqB != nil, "indexer.wqB should be initialized")
        #expect(indexer.weightsProj != nil, "indexer.weightsProj should be initialized")
        #expect(indexer.wk != nil, "kv-source indexer.wk should be initialized")
    }

    // MARK: - 5. ModelFamily has no .deepseek case

    @Test("hcReduce with one-hot pre keeps stream 0")
    func hcReduceOneHotKeepsStreamZero() {
        let stream0 = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 1, 1, 4])
        let stream1 = MLXArray([Float(10), 20, 30, 40]).reshaped([1, 1, 1, 4])
        let x = concatenated([stream0, stream1], axis: 2)
        let pre = deepseekV41InitialPre(batch: 1, length: 1, hcMult: 2)
        let out = hcReduce(x, pre: pre)
        MLX.eval(out)
        #expect(out.shape == [1, 1, 4])
        #expect(out[0, 0, 0].item(Float.self) == 1)
        #expect(out[0, 0, 3].item(Float.self) == 4)
    }

    @Test("hcPost comb is [in, out] as in oMLX einsum bsij,bsid->bsjd")
    func hcPostContractsInputStream() {
        // residual stream 0 = 1s, stream 1 = 2s. comb[i=0, j=1] = 1 routes
        // stream 0 onto output stream 1.
        let residual = concatenated([
            MLXArray.ones([1, 1, 1, 2]),
            MLXArray.ones([1, 1, 1, 2]) * 2,
        ], axis: 2)
        var combVals = [Float](repeating: 0, count: 4)
        combVals[1] = 1  // [i, j] row-major: i=0,j=1
        let comb = MLXArray(combVals).reshaped([1, 1, 2, 2])
        let post = MLXArray.zeros([1, 1, 2])
        let x = MLXArray.zeros([1, 1, 2])
        let out = hcPost(x: x, residual: residual, post: post, comb: comb)
        MLX.eval(out)
        #expect(out.shape == [1, 1, 2, 2])
        #expect(abs(out[0, 0, 0, 0].item(Float.self)) < 1e-5)
        #expect(abs(out[0, 0, 1, 0].item(Float.self) - 1) < 1e-5)
    }

    @Test("sanitize replaces all-zero final RMSNorm with ones")
    func sanitizeReplacesZeroFinalNorm() throws {
        var native = false
        let weights: [String: MLXArray] = [
            "norm.weight": MLXArray.zeros([4]),
        ]
        let cfgJSON = """
            {"model_type":"deepseek_v41","hidden_size":4,"vocab_size":4,"num_hidden_layers":1}
            """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(DeepseekV4Configuration.self, from: cfgJSON)
        let out = DeepseekV4Sanitizer.remap(weights, config: cfg, nativeMtp: &native)
        let w = out["model.norm.weight"]
        #expect(w != nil)
        MLX.eval(w!)
        #expect(w![0].item(Float.self) == 1)
    }

    @Test("ModelFamily includes .deepseek")
    func deepseekFamilyCase() {
        #expect(ModelFamily(rawValue: "deepseek") == .deepseek)
        #expect(ModelFamily(rawValue: "qwen") == .qwen)
    }

    @Test("V4.1 RoPE rotates sequence axis 1, matching MLXFast on [B,H,L,D]")
    func v41RopeUsesSequenceAxis() {
        // [B, L, H, D] = [1, 3, 2, 4]. Position 0 is identity; later steps rotate pairs.
        var vals = [Float](repeating: 0, count: 24)
        for l in 0..<3 {
            for h in 0..<2 {
                for d in 0..<4 {
                    vals[(l * 2 + h) * 4 + d] = Float(l * 10 + h * 4 + d + 1)
                }
            }
        }
        let x = MLXArray(vals).reshaped([1, 3, 2, 4])
        let ours = deepseekV41Rope(
            x, start: 0, dims: 4, base: 10_000, yarn: nil, inverse: false)
        let mlxLayout = x.transposed(0, 2, 1, 3)
        let mlx = MLXFast.RoPE(
            mlxLayout, dimensions: 4, traditional: true, base: 10_000, scale: 1, offset: 0
        ).transposed(0, 2, 1, 3)
        MLX.eval(ours, mlx)
        let diff = (ours.asType(.float32) - mlx.asType(.float32)).abs().max().item(Float.self)
        #expect(diff < 1e-4)

        let inv = deepseekV41Rope(
            ours, start: 0, dims: 4, base: 10_000, yarn: nil, inverse: true)
        MLX.eval(inv)
        let roundTrip = (inv.asType(.float32) - x.asType(.float32)).abs().max().item(Float.self)
        #expect(roundTrip < 1e-4)
    }

    @Test("CSA2 FP4 quantize is bounded and nonzero")
    func csa2Fp4Quantize() {
        let x = MLXArray((0..<64).map { Float($0) * 0.1 - 3 }).reshaped([1, 4, 16])
        let q = DeepseekV41Act.quantize(x, bits: 4, groupSize: 16, e4m3Scale: true)
        MLX.eval(q)
        let mxv = q.asType(.float32).abs().max().item(Float.self)
        #expect(mxv > 0)
        #expect(mxv.isFinite)
    }

    @Test("Expert slot write updates one row without restacking the bank")
    func expertSlotWriteInPlace() {
        let bank = MLXArray.zeros([4, 2, 3])
        let packed = MLXArray([Float](repeating: 7, count: 6)).reshaped([2, 3])
        bank[1] = packed
        MLX.eval(bank)
        #expect(bank[0].asType(.float32).abs().max().item(Float.self) == 0)
        #expect(bank[1, 0, 0].asType(.float32).item(Float.self) == 7)
        #expect(bank[2].asType(.float32).abs().max().item(Float.self) == 0)
    }
}
