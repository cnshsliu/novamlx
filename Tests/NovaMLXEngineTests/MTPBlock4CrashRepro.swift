import Foundation
import NovaMLXCore
import NovaMLXEngine
import Testing

// TEMPORARY in-process repro for the block-4 MTP crash. Loads the real 27B
// oQ8e backbone + MTP-4bit draft from the models dir and runs one generate
// with draft_model. Delete after the crash is fixed.
@Suite("MTP block-4 crash repro")
struct MTPBlock4CrashReproTests {
    @Test func block4CountTo() async throws {
        let modelId = "Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated-oQ8e"
        let dir = NovaMLXPaths.directory(forModelId: modelId)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            Issue.record("model not on disk")
            return
        }
        let engine = MLXEngine()
        let config = ModelConfig(identifier: ModelIdentifier(id: modelId, family: .qwen))
        _ = try await engine.loadModel(from: dir, config: config, progress: nil)
        let request = InferenceRequest(
            model: modelId,
            messages: [ChatMessage(role: .user, content: "从1数到10，每行一个数字")],
            temperature: 0,
            maxTokens: 60,
            draftModel: "mlx-community/Qwen3.8-27B-MTP-4bit",
            numDraftTokens: 3
        )
        var text = ""
        for try await token in engine.stream(request) {
            text += token.text
        }
        print("MTP-BLOCK4-OUTPUT: \(text.prefix(200))")
        // The numbers must appear in order — corrupted GDN rewind at depth 3
        // breaks the sequence long before token 10.
        var cursor = 1
        for word in text.split(whereSeparator: { $0.isNewline || $0 == " " || $0 == "," }) {
            if let n = Int(word), n == cursor {
                cursor += 1
                if cursor > 10 { break }
            }
        }
        #expect(cursor > 10, "expected 1...10 in order, stopped at \(cursor); text=\(text.prefix(300))")
    }
}
