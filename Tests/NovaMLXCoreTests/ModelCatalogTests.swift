import Testing
import Foundation
@testable import NovaMLXCore

@Suite("ModelCatalog")
struct ModelCatalogTests {
    private let sampleJSON = """
    {
      "schemaVersion": 1,
      "updatedAt": "2026-08-16T00:00:00Z",
      "models": [
        {
          "id": "mlx-community/Qwen3.6-27B-OptiQ-4bit",
          "url": "https://huggingface.co/mlx-community/Qwen3.6-27B-OptiQ-4bit",
          "name": "Qwen3.6-27B",
          "category": "llm",
          "family": "qwen",
          "format": "mlx",
          "description": "Latest Qwen 3.6",
          "status": "verified",
          "tags": ["MLX", "4-bit"]
        },
        {
          "id": "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
          "url": "https://huggingface.co/lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
          "name": "Qwen3-VL-4B",
          "category": "vlm",
          "family": "qwen",
          "format": "mlx",
          "status": "preview"
        }
      ]
    }
    """.data(using: .utf8)!

    @Test("Decodes envelope and entries")
    func decodesEnvelope() throws {
        let file = try CatalogFile.decode(sampleJSON)
        #expect(file.schemaVersion == 1)
        #expect(file.models.count == 2)
        #expect(file.models[0].id == "mlx-community/Qwen3.6-27B-OptiQ-4bit")
        #expect(file.models[0].category == .llm)
        #expect(file.models[0].format == .mlx)
        #expect(file.models[0].status == .verified)
        #expect(file.models[1].status == .preview)
    }

    @Test("Ignores unknown fields and future schemaVersion")
    func forwardCompatible() throws {
        let json = """
        {
          "schemaVersion": 99,
          "updatedAt": "2026-08-16T00:00:00Z",
          "extraTop": true,
          "models": [
            {
              "id": "org/model",
              "url": "https://huggingface.co/org/model",
              "name": "Model",
              "category": "audio",
              "family": "whisper",
              "format": "gguf",
              "newField": 1
            }
          ]
        }
        """.data(using: .utf8)!
        let file = try CatalogFile.decode(json)
        #expect(file.schemaVersion == 99)
        #expect(file.models[0].category == .audio)
        #expect(file.models[0].format == .gguf)
        #expect(file.models[0].status == .verified)
    }

    @Test("Missing required fields fail decode")
    func missingRequiredFails() {
        let json = """
        { "schemaVersion": 1, "models": [{ "id": "x" }] }
        """.data(using: .utf8)!
        #expect(throws: Error.self) { try CatalogFile.decode(json) }
    }

    @Test("Allow listed id; refuse unknown when Advanced off")
    func allowRefuse() throws {
        let file = try CatalogFile.decode(sampleJSON)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "mlx-community/Qwen3.6-27B-OptiQ-4bit",
            catalog: file.models,
            allowUnlisted: false) == true)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "some-org/Random-7B",
            catalog: file.models,
            allowUnlisted: false) == false)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "some-org/Random-7B",
            catalog: file.models,
            allowUnlisted: true) == true)
    }

    @Test("Similar name is not a match")
    func similarNameNotAllowed() throws {
        let file = try CatalogFile.decode(sampleJSON)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "other-org/Qwen3.6-27B-OptiQ-4bit",
            catalog: file.models,
            allowUnlisted: false) == false)
    }

    @Test("Refuse message names the id and the Settings toggle")
    func refuseMessage() {
        let msg = ModelCatalogPolicy.refuseMessage(id: "foo/bar")
        #expect(msg.contains("foo/bar"))
        #expect(msg.contains("Allow unverified downloads"))
    }

    @Test("Search filters by query and category")
    func search() throws {
        let file = try CatalogFile.decode(sampleJSON)
        let qwen = ModelCatalogPolicy.search(file.models, query: "qwen3.6", category: nil)
        #expect(qwen.map(\.id) == ["mlx-community/Qwen3.6-27B-OptiQ-4bit"])
        let vlms = ModelCatalogPolicy.search(file.models, query: "", category: .vlm)
        #expect(vlms.count == 1)
        #expect(vlms[0].category == .vlm)
    }

    @Test("Lookup by id")
    func lookup() throws {
        let file = try CatalogFile.decode(sampleJSON)
        #expect(ModelCatalogPolicy.entry(id: "mlx-community/Qwen3.6-27B-OptiQ-4bit", in: file.models)?.format == .mlx)
        #expect(ModelCatalogPolicy.entry(id: "missing", in: file.models) == nil)
    }
}
