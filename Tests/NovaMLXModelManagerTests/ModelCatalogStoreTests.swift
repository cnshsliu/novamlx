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
}
