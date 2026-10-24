import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("MiniMax-M3.1-Flash discovery")
struct MiniMaxDiscoveryTests {
    @Test("minimax_m3_vl stays a text LLM in the minimax family")
    func textCheckpointIsLLM() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-minimax-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent(
            "mlx-community/MiniMax-M3.1-Flash-Text-3bit", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let config = """
        {
          "model_type": "minimax_m3_vl",
          "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
          "vision_config": {"hidden_size": 1024}
        }
        """.data(using: .utf8)!
        try config.write(to: model.appendingPathComponent("config.json"))

        let found = ModelDiscovery().discover(in: root)
        let row = try #require(found.first { $0.modelId.contains("MiniMax-M3.1-Flash") })
        #expect(row.modelType == .llm)
        #expect(row.family == .minimax)
        #expect(row.configModelType == "minimax_m3_vl")
    }
}
