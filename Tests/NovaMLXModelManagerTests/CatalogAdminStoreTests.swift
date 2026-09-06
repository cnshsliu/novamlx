import Foundation
import Testing
import NovaMLXCore
@testable import NovaMLXModelManager

@Suite("CatalogAdminStore")
struct CatalogAdminStoreTests {
    @Test("Discovers a checkout that contains catalog/models.json")
    func discoversRepo() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-admin-\(UUID().uuidString)", isDirectory: true)
        let catalogDir = tmp.appendingPathComponent("catalog", isDirectory: true)
        try FileManager.default.createDirectory(at: catalogDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let json = """
        {"schemaVersion":1,"models":[]}
        """
        try json.write(
            to: catalogDir.appendingPathComponent("models.json"),
            atomically: true, encoding: .utf8
        )
        let store = CatalogAdminStore.discover(
            environment: ["NOVAMLX_REPO": tmp.path],
            home: tmp,
            bundleURL: tmp,
            cwd: tmp
        )
        #expect(store?.repoRoot.path == tmp.path)
        let loaded = try store?.load()
        #expect(loaded?.models.isEmpty == true)
    }

    @Test("Save writes catalog, bundle, and cache")
    func saveWritesCopies() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-admin-save-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: tmp.appendingPathComponent("catalog", isDirectory: true),
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Data(#"{"schemaVersion":1,"models":[]}"#.utf8)
            .write(to: tmp.appendingPathComponent("catalog/models.json"))
        let store = CatalogAdminStore(repoRoot: tmp)
        let cache = tmp.appendingPathComponent("cache/models.json")
        let entry = CatalogEntry(
            id: "org/foo",
            url: "https://huggingface.co/org/foo",
            name: "Foo",
            category: .llm,
            family: .qwen,
            format: .mlx,
            status: .verified
        )
        let saved = try store.save(
            CatalogFile(schemaVersion: 1, models: [entry]),
            cacheURL: cache
        )
        #expect(saved.models.count == 1)
        #expect(FileManager.default.fileExists(atPath: store.catalogURL.path))
        #expect(FileManager.default.fileExists(atPath: store.bundleURL.path))
        #expect(FileManager.default.fileExists(atPath: cache.path))
        let roundTrip = try CatalogFile.decode(Data(contentsOf: cache))
        #expect(roundTrip.models[0].id == "org/foo")
    }
}
