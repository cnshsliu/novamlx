import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("Nemotron Bolt discovery")
struct NemotronDiscoveryTests {
    @Test("nemotron_h_puzzle config is a text LLM in the nemotron family")
    func puzzleConfigIsTextLLM() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-nemotron-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(
            "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = """
        {
          "model_type": "nemotron_h_puzzle",
          "architectures": ["NemotronHPuzzleForCausalLM"],
          "vision_config": {"hidden_size": 1280}
        }
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("config.json"))

        let qwen = root.appendingPathComponent("mlx-community/Qwen2.5-0.5B-Instruct", isDirectory: true)
        try FileManager.default.createDirectory(at: qwen, withIntermediateDirectories: true)
        let qwenConfig = """
        {"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}
        """.data(using: .utf8)!
        try qwenConfig.write(to: qwen.appendingPathComponent("config.json"))

        let found = ModelDiscovery().discover(in: root)
        let bolt = try #require(found.first {
            $0.modelId == "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit"
        })
        #expect(bolt.modelType == .llm)
        #expect(bolt.family == .nemotron)
        #expect(bolt.configModelType == "nemotron_h_puzzle")

        let qwenModel = try #require(found.first { $0.modelId.contains("Qwen2.5") })
        #expect(qwenModel.modelType == .llm)
        #expect(qwenModel.family == .qwen)
    }
}
