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

    @Test("Refuse message names the id and the Download Models toggle")
    func refuseMessage() {
        let msg = ModelCatalogPolicy.refuseMessage(id: "foo/bar")
        #expect(msg.contains("foo/bar"))
        #expect(msg.contains("Allow unverified downloads"))
        #expect(msg.contains("Download Models"))
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

    @Test("Search matches description")
    func searchByDescription() throws {
        let file = try CatalogFile.decode(sampleJSON)
        let hits = ModelCatalogPolicy.search(file.models, query: "Latest Qwen", category: nil)
        #expect(hits.map(\.id) == ["mlx-community/Qwen3.6-27B-OptiQ-4bit"])
    }

    @Test("Search matches tags")
    func searchByTag() throws {
        let file = try CatalogFile.decode(sampleJSON)
        let hits = ModelCatalogPolicy.search(file.models, query: "4-bit", category: nil)
        #expect(hits.map(\.id) == ["mlx-community/Qwen3.6-27B-OptiQ-4bit"])
    }

    @Test("Search matches family rawValue")
    func searchByFamily() throws {
        let file = try CatalogFile.decode(sampleJSON)
        let hits = ModelCatalogPolicy.search(file.models, query: "qwen", category: nil)
        #expect(hits.count == 2)
        #expect(hits.map(\.id) == [
            "mlx-community/Qwen3.6-27B-OptiQ-4bit",
            "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit",
        ])
    }

    @Test("Whitespace-only query matches all like empty query")
    func searchWhitespaceQuery() throws {
        let file = try CatalogFile.decode(sampleJSON)
        let all = ModelCatalogPolicy.search(file.models, query: "   ", category: nil)
        #expect(all.count == 2)
        let vlms = ModelCatalogPolicy.search(file.models, query: "   ", category: .vlm)
        #expect(vlms.count == 1)
        #expect(vlms[0].category == .vlm)
    }

    @Test("Multi-word query matches tokens across fields")
    func searchMatchesAllTokens() {
        let catalog = [
            CatalogEntry(
                id: "mlx-community/Qwen3.8-*",
                url: "https://huggingface.co/models?search=Qwen3.8",
                name: "Qwen3.8 family",
                category: .llm,
                family: .qwen,
                format: .mlx,
                description: "Dense Qwen3.8-27B checkpoints"
            ),
            CatalogEntry(
                id: "pipenetwork/Qwen3.8-Flash-Next-*",
                url: "https://huggingface.co/models?search=Qwen3.8-Flash-Next",
                name: "Qwen3.8-Flash-Next",
                category: .vlm,
                family: .qwen,
                format: .mlx,
                description: "125B MoE qwen4_exp preview",
                tags: ["MLX", "Flash", "MoE"]
            ),
        ]
        let flash = ModelCatalogPolicy.search(catalog, query: "qwen3.8 flash", category: nil)
        #expect(flash.map(\.id) == ["pipenetwork/Qwen3.8-Flash-Next-*"])
        let dense = ModelCatalogPolicy.search(catalog, query: "qwen3.8", category: nil)
        #expect(dense.map(\.id) == [
            "mlx-community/Qwen3.8-*",
            "pipenetwork/Qwen3.8-Flash-Next-*",
        ])
        let miss = ModelCatalogPolicy.search(catalog, query: "qwen3.8 flash missing", category: nil)
        #expect(miss.isEmpty)
        #expect(ModelCatalogPolicy.shouldExpandFamilyGlobs(
            query: "qwen3.8 flash", catalog: catalog) == true)
        #expect(ModelCatalogPolicy.hubSearchQuery(
            forPattern: "pipenetwork/Qwen3.8-Flash-Next-*") == "Qwen3.8-Flash-Next")
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "pipenetwork/Qwen3.8-Flash-Next-MLX-4bit",
            catalog: catalog,
            allowUnlisted: false) == true)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX",
            catalog: catalog,
            allowUnlisted: false) == false)
    }

    @Test("Search query combined with category filter")
    func searchQueryAndCategory() throws {
        let file = try CatalogFile.decode(sampleJSON)
        let hits = ModelCatalogPolicy.search(file.models, query: "qwen", category: .llm)
        #expect(hits.map(\.id) == ["mlx-community/Qwen3.6-27B-OptiQ-4bit"])
        #expect(hits[0].category == .llm)
    }

    @Test("looksLikeMLXRepo keeps mlx-tagged and mlx-community ids")
    func looksLikeMLX() {
        #expect(ModelCatalogPolicy.looksLikeMLXRepo(
            id: "mlx-community/Qwen3.6-27B-OptiQ-4bit", tags: ["4-bit"]) == true)
        #expect(ModelCatalogPolicy.looksLikeMLXRepo(
            id: "Qwen/Qwen3-VL-4B", tags: ["mlx", "safetensors"]) == true)
        #expect(ModelCatalogPolicy.looksLikeMLXRepo(
            id: "Qwen/Qwen3.8-27B", tags: ["transformers", "safetensors"]) == false)
        #expect(ModelCatalogPolicy.looksLikeMLXRepo(
            id: "unsloth/Qwen3.8-27B-GGUF", tags: ["gguf", "llama.cpp"]) == false)
    }

    @Test("Family glob expand is only for queries about that family")
    func familyExpandIsScoped() {
        let catalog = [
            CatalogEntry(
                id: "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
                url: "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
                name: "Ornith-1.5-35B-A3B",
                category: .llm,
                family: .qwen,
                format: .mlx,
                tags: ["MLX", "Ornith"]
            ),
            CatalogEntry(
                id: "mlx-community/Qwen3.8-*",
                url: "https://huggingface.co/models?search=Qwen3.8",
                name: "Qwen3.8 family",
                category: .llm,
                family: .qwen,
                format: .mlx
            ),
        ]
        #expect(ModelCatalogPolicy.shouldExpandFamilyGlobs(query: "ornith", catalog: catalog) == false)
        #expect(ModelCatalogPolicy.shouldExpandFamilyGlobs(query: "qwen3.8", catalog: catalog) == true)
        #expect(ModelCatalogPolicy.shouldExpandFamilyGlobs(
            query: "mlx-community/Qwen3.8-27B-4bit", catalog: catalog) == true)
        let ornith = ModelCatalogPolicy.search(catalog, query: "ornith", category: nil)
        #expect(ornith.map(\.id) == ["ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit"])
    }

    @Test("Browse lists newest addedAt first; undated keep file order")
    func searchNewestAddedFirst() {
        let older = CatalogEntry(
            id: "org/old",
            url: "https://huggingface.co/org/old",
            name: "Old",
            category: .llm,
            family: .qwen,
            format: .mlx,
            addedAt: "2026-01-01T00:00:00Z"
        )
        let newer = CatalogEntry(
            id: "org/new",
            url: "https://huggingface.co/org/new",
            name: "New",
            category: .llm,
            family: .qwen,
            format: .mlx,
            addedAt: "2026-08-22T12:00:00Z"
        )
        let undated = CatalogEntry(
            id: "org/undated",
            url: "https://huggingface.co/org/undated",
            name: "Undated",
            category: .llm,
            family: .qwen,
            format: .mlx
        )
        let catalog = [older, undated, newer]
        let hits = ModelCatalogPolicy.search(catalog, query: "", category: nil)
        #expect(hits.map(\.id) == ["org/new", "org/old", "org/undated"])
    }

    @Test("Lookup by id")
    func lookup() throws {
        let file = try CatalogFile.decode(sampleJSON)
        #expect(ModelCatalogPolicy.entry(id: "mlx-community/Qwen3.6-27B-OptiQ-4bit", in: file.models)?.format == .mlx)
        #expect(ModelCatalogPolicy.entry(id: "missing", in: file.models) == nil)
    }

    @Test("Trailing glob is a family pattern; org-wide * is not")
    func idPatternShape() {
        #expect(ModelCatalogPolicy.isIdPattern("mlx-community/Qwen3.8-*") == true)
        #expect(ModelCatalogPolicy.isIdPattern("mlx-community/Qwen3.8-27B-8bit") == false)
        #expect(ModelCatalogPolicy.isIdPattern("mlx-community/*") == false)
        #expect(ModelCatalogPolicy.isIdPattern("*") == false)
        #expect(ModelCatalogPolicy.isIdPattern("Qwen3.8-*") == false)
        #expect(ModelCatalogPolicy.isIdPattern("mlx-community/Qwen3.8-*-8bit") == false)
        #expect(ModelCatalogPolicy.hubSearchQuery(forPattern: "mlx-community/Qwen3.8-*") == "Qwen3.8")
    }

    @Test("Family glob allows every matching Hub id")
    func prefixAllowsFamily() {
        let family = CatalogEntry(
            id: "mlx-community/Qwen3.8-*",
            url: "https://huggingface.co/models?search=Qwen3.8&author=mlx-community",
            name: "Qwen3.8 family",
            category: .llm,
            family: .qwen,
            format: .mlx
        )
        let catalog = [family]
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "mlx-community/Qwen3.8-27B-8bit", catalog: catalog, allowUnlisted: false) == true)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "mlx-community/Qwen3.8-27B-MTP-8bit", catalog: catalog, allowUnlisted: false) == true)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "mlx-community/Qwen3.8-27B-4bit", catalog: catalog, allowUnlisted: false) == true)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "mlx-community/Qwen3.5-9B-OptiQ-4bit", catalog: catalog, allowUnlisted: false) == false)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "other-org/Qwen3.8-27B-8bit", catalog: catalog, allowUnlisted: false) == false)
        #expect(ModelCatalogPolicy.isDownloadAllowed(
            id: "mlx-community/Qwen3.8-*", catalog: catalog, allowUnlisted: false) == false)
        #expect(ModelCatalogPolicy.entry(
            id: "mlx-community/Qwen3.8-27B-MTP-4bit", in: catalog)?.id == "mlx-community/Qwen3.8-*")
    }

    @Test("Longer family glob wins over a shorter one")
    func longestPrefixWins() {
        let catalog = [
            CatalogEntry(
                id: "mlx-community/Qwen3.8-*",
                url: "https://huggingface.co/mlx-community",
                name: "Qwen3.8",
                category: .llm,
                family: .qwen,
                format: .mlx
            ),
            CatalogEntry(
                id: "mlx-community/Qwen3.8-27B-*",
                url: "https://huggingface.co/mlx-community",
                name: "Qwen3.8-27B",
                category: .llm,
                family: .qwen,
                format: .mlx
            ),
        ]
        #expect(ModelCatalogPolicy.entry(
            id: "mlx-community/Qwen3.8-27B-8bit", in: catalog)?.id == "mlx-community/Qwen3.8-27B-*")
        #expect(ModelCatalogPolicy.entry(
            id: "mlx-community/Qwen3.8-35B-A3B-8bit", in: catalog)?.id == "mlx-community/Qwen3.8-*")
    }
}
