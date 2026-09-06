import Testing
import Foundation
@testable import NovaMLXModelManager
@testable import NovaMLXCore

private struct FixedTransport: CatalogTransport, Sendable {
    let result: Result<Data, Error>
    func data(from url: URL) async throws -> Data {
        try result.get()
    }
}

@Suite("ModelCatalogStore")
struct ModelCatalogStoreTests {
    private func tmpDir() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private let valid = """
    {"schemaVersion":1,"updatedAt":"2026-08-16T00:00:00Z","models":[
      {"id":"org/a","url":"https://huggingface.co/org/a","name":"A","category":"llm","family":"qwen","format":"mlx"}
    ]}
    """.data(using: .utf8)!

    @Test("Remote 200 replaces memory and cache")
    func remoteWins() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: nil,
            transport: FixedTransport(result: .success(valid))
        )
        await store.refresh()
        #expect(store.models.count == 1)
        #expect(FileManager.default.fileExists(atPath: cache.path))
        let cached = try CatalogFile.decode(Data(contentsOf: cache))
        #expect(cached.models[0].id == "org/a")
    }

    @Test("Remote failure uses disk cache")
    func cacheFallback() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        try valid.write(to: cache)
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: nil,
            transport: FixedTransport(result: .failure(URLError(.timedOut)))
        )
        await store.refresh()
        #expect(store.models.map(\.id) == ["org/a"])
    }

    @Test("No cache uses bundled snapshot")
    func bundleFallback() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        let bundle = dir.appendingPathComponent("bundle.json")
        try valid.write(to: bundle)
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: bundle,
            transport: FixedTransport(result: .failure(URLError(.notConnectedToInternet)))
        )
        await store.refresh()
        #expect(store.models.count == 1)
    }

    @Test("All sources empty leaves catalog empty")
    func allEmpty() async throws {
        let dir = try tmpDir()
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: dir.appendingPathComponent("missing.json"),
            bundleURL: nil,
            transport: FixedTransport(result: .failure(URLError(.timedOut)))
        )
        await store.refresh()
        #expect(store.models.isEmpty)
    }

    @Test("Repo catalog snapshot decodes preview models")
    func repoSnapshotDecodes() throws {
        // Tests/NovaMLXModelManagerTests/… → repo root is three levels up
        let testFile = URL(fileURLWithPath: #filePath)
        let repoRoot = testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let snapshot = repoRoot
            .appendingPathComponent("Sources/NovaMLXUtils/Resources/catalog/models.json")
        let data = try Data(contentsOf: snapshot)
        let file = try CatalogFile.decode(data)
        #expect(file.models.count >= 24)
        #expect(file.models.contains { $0.id.hasPrefix("ornith-ai/Ornith-1.5-35B-A3B-MLX") })
        #expect(file.models.contains { $0.id == "mlx-community/Qwen3.8-27B-8bit" })
        #expect(file.models.contains { $0.id == "mlx-community/Qwen3.8-27B-4bit" })
        #expect(file.models.contains { $0.id == "orcarouter/Qwen3.8-27B-Uncensored-MLX" })
        #expect(file.models.contains { $0.id == "incoai/Qwen3.8-27B-DFlash2" })
        #expect(file.models.contains { $0.id == "mlx-community/Qwen3.8-27B-MTP-*" })
        #expect(!file.models.contains { $0.id == "mlx-community/Qwen3.8-*" })
        #expect(!file.models.contains { $0.id.contains("Qwen3.8-27B-OptiQ") })
        #expect(file.models.contains { $0.id == "pipenetwork/Qwen3.8-Flash-Next-*" })
        #expect(file.models.contains { $0.id == "orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX" })
        #expect(
            ModelCatalogPolicy.isDownloadAllowed(
                id: "mlx-community/Qwen3.8-27B-OptiQ-4bit",
                catalog: file.models,
                allowUnlisted: false) == false)
        #expect(
            ModelCatalogPolicy.isDownloadAllowed(
                id: "mlx-community/Qwen3.8-27B-4bit",
                catalog: file.models,
                allowUnlisted: false) == true)
        #expect(
            ModelCatalogPolicy.isDownloadAllowed(
                id: "mlx-community/Qwen3.8-27B-8bit",
                catalog: file.models,
                allowUnlisted: false) == true)
        #expect(
            ModelCatalogPolicy.isDownloadAllowed(
                id: "orcarouter/Qwen3.8-27B-Uncensored-MLX",
                catalog: file.models,
                allowUnlisted: false) == true)
        #expect(
            ModelCatalogPolicy.isDownloadAllowed(
                id: "incoai/Qwen3.8-27B-DFlash2",
                catalog: file.models,
                allowUnlisted: false) == true)
        #expect(
            ModelCatalogPolicy.isDownloadAllowed(
                id: "mlx-community/Qwen3.8-27B-MTP-4bit",
                catalog: file.models,
                allowUnlisted: false) == true)
        #expect(file.models.allSatisfy { $0.status == .preview || $0.status == .verified })
        #expect(file.models.first { $0.id == "pipenetwork/Qwen3.8-Flash-Next-*" }?.status == .preview)
        // When the Utils resource bundle is present, multi-path lookup must succeed.
        // Under plain `swift test` the bundle may be missing — that is OK if decode above passes.
        if let bundled = ModelCatalogStore.bundledSnapshotURL() {
            let bundledFile = try CatalogFile.decode(Data(contentsOf: bundled))
            #expect(bundledFile.models.map(\.id) == file.models.map(\.id))
        }
        let githubCatalog = repoRoot.appendingPathComponent("catalog/models.json")
        let githubFile = try CatalogFile.decode(Data(contentsOf: githubCatalog))
        #expect(githubFile.models.map(\.id) == file.models.map(\.id))
    }

    @Test("Remote catalog URL is GitHub raw, not novamlx.ai")
    func remoteURLIsGitHub() {
        let url = ModelCatalogStore.defaultRemoteURL
        #expect(url.host == "raw.githubusercontent.com")
        #expect(url.path.hasSuffix("/cnshsliu/novamlx/main/catalog/models.json"))
        #expect(!(url.host ?? "").contains("novamlx.ai"))
    }

    @Test("Newer cache wins over stale remote and is not overwritten")
    func newerCacheBeatsStaleRemote() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        let newer = """
        {"schemaVersion":1,"updatedAt":"2026-08-22T12:00:00Z","models":[
          {"id":"org/new","url":"https://huggingface.co/org/new","name":"New","category":"llm","family":"qwen","format":"mlx"}
        ]}
        """.data(using: .utf8)!
        try newer.write(to: cache)
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: nil,
            transport: FixedTransport(result: .success(valid))
        )
        await store.refresh()
        let ids = store.models.map(\.id)
        #expect(ids.contains("org/new"))
        #expect(ids.contains("org/a"))
        let cached = try CatalogFile.decode(Data(contentsOf: cache))
        #expect(Set(cached.models.map(\.id)) == Set(ids))
    }

    @Test("Newer remote replaces stale cache")
    func newerRemoteBeatsStaleCache() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        try valid.write(to: cache)
        let remote = """
        {"schemaVersion":1,"updatedAt":"2026-08-22T12:00:00Z","models":[
          {"id":"org/b","url":"https://huggingface.co/org/b","name":"B","category":"llm","family":"qwen","format":"mlx"}
        ]}
        """.data(using: .utf8)!
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: nil,
            transport: FixedTransport(result: .success(remote))
        )
        await store.refresh()
        let ids = Set(store.models.map(\.id))
        #expect(ids.contains("org/b"))
        #expect(ids.contains("org/a"))
        let cached = try CatalogFile.decode(Data(contentsOf: cache))
        #expect(Set(cached.models.map(\.id)) == ids)
    }

    @Test("Each refresh re-reads sources so a new cache entry appears")
    func refreshPicksUpNewCacheEntry() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        try valid.write(to: cache)
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: nil,
            transport: FixedTransport(result: .failure(URLError(.timedOut)))
        )
        await store.refresh()
        #expect(store.models.map(\.id) == ["org/a"])

        let updated = """
        {"schemaVersion":1,"updatedAt":"2026-08-22T15:00:00Z","models":[
          {"id":"org/a","url":"https://huggingface.co/org/a","name":"A","category":"llm","family":"qwen","format":"mlx"},
          {"id":"mlx-community/Qwen3.8-*","url":"https://huggingface.co/models?search=Qwen3.8","name":"Qwen3.8","category":"llm","family":"qwen","format":"mlx"}
        ]}
        """.data(using: .utf8)!
        try updated.write(to: cache)
        await store.refresh()
        #expect(store.models.contains { $0.id == "mlx-community/Qwen3.8-*" })
        #expect(store.models.count == 2)
    }

    @Test("Bundle-only ids survive a remote catalog that omits them")
    func mergeKeepsLocalOnlyEntries() async throws {
        let dir = try tmpDir()
        let cache = dir.appendingPathComponent("models.json")
        let bundle = dir.appendingPathComponent("bundle.json")
        let withOrnith = """
        {"schemaVersion":1,"updatedAt":"2026-08-22T12:00:00Z","models":[
          {"id":"org/a","url":"https://huggingface.co/org/a","name":"A","category":"llm","family":"qwen","format":"mlx"},
          {"id":"ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit","url":"https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit","name":"Ornith","category":"llm","family":"qwen","format":"mlx","tags":["Ornith"]}
        ]}
        """.data(using: .utf8)!
        try withOrnith.write(to: bundle)
        let store = ModelCatalogStore(
            remoteURL: URL(string: "https://example.invalid/catalog/models.json")!,
            cacheURL: cache,
            bundleURL: bundle,
            transport: FixedTransport(result: .success(valid))
        )
        await store.refresh()
        #expect(store.models.contains { $0.id == "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit" })
        let hits = ModelCatalogPolicy.search(store.models, query: "ornith", category: nil)
        #expect(hits.map(\.id) == ["ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit"])
    }
}
