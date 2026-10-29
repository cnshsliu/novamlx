import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("GLM-5.3-Flash discovery")
struct Glm5DiscoveryTests {
    @Test("glm5_next with vision_config stays a text LLM in the glm family")
    func glm5NextIsTextLLM() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-glm5-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(
            "mlx-community/GLM-5.3-Flash-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = """
        {
          "model_type": "glm5_next",
          "architectures": ["Glm5NextForConditionalGeneration"],
          "vision_config": {"hidden_size": 1024},
          "text_config": {"model_type": "glm5_next_text"}
        }
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("config.json"))

        let found = ModelDiscovery().discover(in: root)
        let glm = try #require(found.first { $0.modelId == "mlx-community/GLM-5.3-Flash-4bit" })
        #expect(glm.modelType == .llm)
        #expect(glm.family == .glm)
        #expect(glm.configModelType == "glm5_next")
    }

    @Test("glm4_moe does not inherit the glm5_next family")
    func glm4StaysOffTheGlm5Family() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-glm4-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(
            "mlx-community/GLM-4.7-Flash-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = """
        {"model_type":"glm4_moe","architectures":["Glm4MoeForCausalLM"]}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("config.json"))

        let found = ModelDiscovery().discover(in: root)
        let glm4 = try #require(found.first { $0.modelId.contains("GLM-4.7") })
        #expect(glm4.family != .glm)
        #expect(glm4.modelType == .llm)
    }
}
