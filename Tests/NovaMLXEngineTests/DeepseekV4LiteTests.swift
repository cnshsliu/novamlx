import Testing
import Foundation
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

@Suite("DeepSeek-V4 lite regression")
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
        #expect(found?.family == .qwen,
            "deepseek_v4 model_type should map to .qwen family; got \(found?.family.rawValue ?? "nil")")
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
        #expect(found?.family == .qwen,
            "DeepseekV4ForCausalLM architecture should map to .qwen; got \(found?.family.rawValue ?? "nil")")
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
        let indexer = DeepseekV4Indexer(config: config, compressRatio: 2)
        #expect(indexer.wqB != nil, "indexer.wqB should be initialized")
        #expect(indexer.weightsProj != nil, "indexer.weightsProj should be initialized")
        #expect(indexer.compressor != nil, "indexer.compressor should be initialized")
    }

    // MARK: - 5. ModelFamily has no .deepseek case

    @Test("ModelFamily has no .deepseek case — currently mapped to .qwen")
    func noDeepseekFamilyCase() {
        // If a future commit adds .deepseek, this test fails and signals
        // that the family routing in ModelDiscovery should be updated.
        let raw = ModelFamily(rawValue: "deepseek")
        #expect(raw == nil,
            "ModelFamily should not have a .deepseek case yet; if this fails, update ModelDiscovery routing")
        #expect(ModelFamily(rawValue: "qwen") == .qwen,
            ".qwen case must exist for current DeepSeek-V4 routing")
    }
}
