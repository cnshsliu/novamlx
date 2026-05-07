import Testing
import Foundation
@testable import NovaMLXAPI
import NovaMLXCore

@Suite("Models Endpoint / Capabilities")
struct ModelsEndpointTests {

    // MARK: - OpenAIModel Codable round-trip

    @Test("OpenAIModel encodes nova.capabilities correctly")
    func testCapabilitiesEncode() throws {
        let caps = ModelCapabilities(reasoning: true, thinking: true, tools: true, vision: false)
        let model = OpenAIModel(id: "test-model", nova: OpenAIModelNova(capabilities: caps))

        let data = try JSONEncoder().encode(model)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        #expect(json["id"] as? String == "test-model")
        #expect(json["object"] as? String == "model")
        #expect(json["owned_by"] as? String == "novamlx")

        let nova = json["nova"] as! [String: Any]
        let capabilities = nova["capabilities"] as! [String: Any]
        #expect(capabilities["reasoning"] as? Bool == true)
        #expect(capabilities["thinking"] as? Bool == true)
        #expect(capabilities["tools"] as? Bool == true)
        #expect(capabilities["vision"] as? Bool == false)
    }

    @Test("OpenAIModel without nova omits the field")
    func testModelWithoutNova() throws {
        let model = OpenAIModel(id: "plain-model")

        let data = try JSONEncoder().encode(model)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        #expect(json["id"] as? String == "plain-model")
        #expect(json["nova"] == nil)
    }

    @Test("OpenAIModel round-trips through Codable")
    func testRoundTrip() throws {
        let caps = ModelCapabilities(reasoning: false, thinking: false, tools: false, vision: true)
        let original = OpenAIModel(id: "vlm-model", nova: OpenAIModelNova(capabilities: caps))

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(OpenAIModel.self, from: data)

        #expect(decoded.id == original.id)
        #expect(decoded.nova?.capabilities == caps)
    }

    // MARK: - Capability detection

    @Test("VLM model gets vision=true")
    func testVisionCapabilityForVLM() {
        let detector = ModelCapabilitiesDetector()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-vlm-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let caps = detector.capabilities(for: "test-vlm", modelType: .vlm, localURL: tmpDir)
        #expect(caps.vision == true)
        #expect(caps.reasoning == false)
    }

    @Test("LLM model without template gets all-false capabilities")
    func testNonReasoningModel() {
        let detector = ModelCapabilitiesDetector()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-llm-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let caps = detector.capabilities(for: "test-llm", modelType: .llm, localURL: tmpDir)
        #expect(caps.vision == false)
        #expect(caps.reasoning == false)
        #expect(caps.thinking == false)
        #expect(caps.tools == false)
    }

    @Test("Template with tool_call markers sets tools=true")
    func testToolsCapability() {
        let detector = ModelCapabilitiesDetector()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-tools-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Write a tokenizer_config.json with tool markers
        let config: [String: Any] = [
            "chat_template": "{% if tool_calls %}{{ tool_call }}{% endif %}{{ message }}"
        ]
        let configData = try! JSONSerialization.data(withJSONObject: config)
        try! configData.write(to: tmpDir.appendingPathComponent("tokenizer_config.json"))

        let caps = detector.capabilities(for: "tool-model", modelType: .llm, localURL: tmpDir)
        #expect(caps.tools == true)
        #expect(caps.vision == false)
    }

    @Test("Template with implicit think injection sets thinking=true and reasoning=true")
    func testImplicitThinkingDetection() {
        let detector = ModelCapabilitiesDetector()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-think-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Template that injects opening think tag (Qwen3-style Jinja)
        let template = "{% if add_generation_prompt %}special_system_token+'<think\\n'{% endif %}"
        let config: [String: Any] = ["chat_template": template]
        let configData = try! JSONSerialization.data(withJSONObject: config)
        try! configData.write(to: tmpDir.appendingPathComponent("tokenizer_config.json"))

        let caps = detector.capabilities(for: "thinking-model", modelType: .llm, localURL: tmpDir)
        #expect(caps.thinking == true)
        #expect(caps.reasoning == true)
    }

    @Test("Template with explicit think tags (no injection) sets reasoning=true, thinking=false")
    func testExplicitThinkingDetection() {
        let detector = ModelCapabilitiesDetector()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-explicit-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Template mentions <think and </think but doesn't inject
        let template = "{% if enable_thinking %}<think{% endif %}{{ message }}{% if enable_thinking %}</think{% endif %}"
        let config: [String: Any] = ["chat_template": template]
        let configData = try! JSONSerialization.data(withJSONObject: config)
        try! configData.write(to: tmpDir.appendingPathComponent("tokenizer_config.json"))

        let caps = detector.capabilities(for: "explicit-model", modelType: .llm, localURL: tmpDir)
        #expect(caps.thinking == false)
        #expect(caps.reasoning == true)
    }

    // MARK: - Cache behavior

    @Test("Cached capabilities are returned without recomputation")
    func testCachedCapabilitiesNoRecompute() {
        let detector = ModelCapabilitiesDetector()
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-cache-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let caps1 = detector.capabilities(for: "cached-model", modelType: .llm, localURL: tmpDir)

        // Delete the config file — if cache works, second call still returns same result
        try? FileManager.default.removeItem(at: tmpDir.appendingPathComponent("tokenizer_config.json"))

        let caps2 = detector.capabilities(for: "cached-model", modelType: .vlm, localURL: tmpDir)

        // Should return cached result (llm, all false) not recompute with vlm
        #expect(caps1 == caps2)
        #expect(caps2.vision == false) // would be true if recomputed with .vlm
    }

    // MARK: - OpenAIModelsResponse

    @Test("OpenAIModelsResponse encodes list of models")
    func testModelsResponseEncoding() throws {
        let caps = ModelCapabilities(reasoning: true, thinking: false, tools: true, vision: false)
        let models = [
            OpenAIModel(id: "model-a", nova: OpenAIModelNova(capabilities: caps)),
            OpenAIModel(id: "model-b"),
        ]
        let response = OpenAIModelsResponse(data: models)

        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        #expect(json["object"] as? String == "list")
        let dataArray = json["data"] as! [[String: Any]]
        #expect(dataArray.count == 2)
        #expect(dataArray[0]["nova"] != nil)
        #expect(dataArray[1]["nova"] == nil)
    }
}
