import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("ERNIE-5.5 discovery")
struct Ernie55DiscoveryTests {
    @Test("ernie5_5_moe is a text LLM and does not steal a neighboring qwen folder")
    func discoversTextBase() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-ernie55-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("baidu/ERNIE-5.5-Flash-Base", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = """
        {"model_type":"ernie5_5_moe","architectures":["Ernie5_5_MoeForCausalLM"]}
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("config.json"))

        let qwen = root.appendingPathComponent("mlx-community/Qwen2.5-0.5B-Instruct", isDirectory: true)
        try FileManager.default.createDirectory(at: qwen, withIntermediateDirectories: true)
        let qwenConfig = """
        {"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}
        """.data(using: .utf8)!
        try qwenConfig.write(to: qwen.appendingPathComponent("config.json"))

        let found = ModelDiscovery().discover(in: root)
        let ernie = try #require(found.first { $0.modelId == "baidu/ERNIE-5.5-Flash-Base" })
        #expect(ernie.modelType == .llm)
        #expect(ernie.family == .other)
        #expect(ernie.configModelType == "ernie5_5_moe")

        let neighbor = try #require(found.first { $0.modelId.contains("Qwen2.5") })
        #expect(neighbor.family == .qwen)
        #expect(neighbor.modelType == .llm)
    }
}
