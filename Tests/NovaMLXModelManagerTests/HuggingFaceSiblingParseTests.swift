import Testing
import NovaMLXModelManager

@Suite("Hugging Face file list")
struct HuggingFaceSiblingParseTests {
    @Test("string cardData.language does not reject the file list")
    func looseCard() throws {
        let json = """
        {"id":"aac6fef/laya-multilingual-mlx","cardData":{"language":"multilingual","license":"apache-2.0"},"siblings":[{"rfilename":"model.safetensors","size":700},{"rfilename":"tokenizer/tokenizer.json","size":12}]}
        """.data(using: .utf8)!
        let files = try HuggingFaceService.parseSiblingFiles(data: json, statusCode: 200)
        #expect(files.map { $0.rfilename } == ["model.safetensors", "tokenizer/tokenizer.json"])
    }

    @Test("HTML rate-limit page is not reported as a corrupt file")
    func rateLimit() {
        let html = "<!DOCTYPE html><html>rate limit</html>".data(using: .utf8)!
        do {
            _ = try HuggingFaceService.parseSiblingFiles(data: html, statusCode: 429)
            Issue.record("expected rate-limit error")
        } catch {
            #expect(error.localizedDescription.contains("429"))
            #expect(!error.localizedDescription.contains("correct format"))
        }
    }
}
