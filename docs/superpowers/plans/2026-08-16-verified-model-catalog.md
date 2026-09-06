# Verified Model Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a manually tested catalog on `novamlx.ai` the only default browse/search/download surface in NovaMLX, with a Settings toggle that unlocks arbitrary Hub URLs.

**Architecture:** A versioned JSON catalog is authored at `novamlx-website/static/catalog/models.json` and served at `https://novamlx.ai/catalog/models.json`. The app decodes it into `CatalogFile` / `CatalogEntry`, caches it under `~/.nova/cache/catalog/`, and ships a release snapshot in `NovaMLXUtils/Resources`. A pure `ModelCatalogPolicy` decides allow/refuse. Admin search and both download endpoints read the **live** `ServerConfig.allowUnlistedDownloads` from `NovaMLXConfiguration.shared` on every request (the `APIServer` constructor snapshot is stale after a Settings toggle). Weights still download from each entry’s original `url`.

**Tech Stack:** Swift 6.0, Swift Testing, Hummingbird 2, SwiftUI, SvelteKit (website repo at `~/dev/novamlx-website`).

**Spec:** `docs/superpowers/specs/2026-08-16-verified-model-catalog-design.md` — read it first.

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `Sources/NovaMLXCore/ModelCatalog.swift` | `CatalogFile`, `CatalogEntry`, `CatalogFormat`, `CatalogStatus`, `ModelCatalogPolicy` (decode / search / allow / refuse message) |
| `Tests/NovaMLXCoreTests/ModelCatalogTests.swift` | Decode, search, allow/refuse unit tests |
| `Sources/NovaMLXModelManager/ModelCatalogStore.swift` | Fetch remote → disk cache → bundled snapshot; no Hub I/O |
| `Tests/NovaMLXModelManagerTests/ModelCatalogStoreTests.swift` | Load-order and search-wiring tests against fixture files |
| `Sources/NovaMLXUtils/Resources/catalog/models.json` | Release-bundled snapshot (copy of the website file) |
| `~/dev/novamlx-website/static/catalog/models.json` | Source of truth, public at `/catalog/models.json` |

### Modified files

| Path | Change |
|---|---|
| `Sources/NovaMLXCore/Types.swift` | Add `allowUnlistedDownloads` to `ServerConfig` (default `false`) |
| `Sources/NovaMLXCore/Configuration.swift` | Persist / load the flag via `ConfigRecord` |
| `Sources/NovaMLXCore/NovaMLXPaths.swift` | Add `catalogCacheFile` |
| `Sources/NovaMLXDB/Models/ConfigRecords.swift` | `allowUnlistedDownloads` column mapping |
| `Sources/NovaMLXDB/Migrations/ConfigDBSchema.swift` | `v6AllowUnlistedDownloads` |
| `Sources/NovaMLXDB/NovaDB.swift` | Register `v6_allow_unlisted_downloads` |
| `Tests/NovaMLXEngineTests/PrefixCacheKillSwitchTests.swift` | Add sibling tests for the new default (or new file below) |
| `Tests/NovaMLXCoreTests/ServerConfigAllowUnlistedTests.swift` | Decode default / explicit / legacy JSON |
| `Tests/NovaMLXDBTests/ConfigStoreTests.swift` | Persist + migrate the new column |
| `Sources/NovaMLXModelManager/ModelManager.swift` | Replace GitHub `SuggestedModel` fetch with `ModelCatalogStore` |
| `Sources/NovaMLXModelManager/HuggingFaceService.swift` | Optional `revision` on `startDownload` / `resolveURL` |
| `Sources/NovaMLXAPI/APIServer.swift` | Gate `/admin/api/hf/search`, `/admin/api/hf/download`, `/admin/models/download` |
| `Sources/NovaMLXApp/main.swift` | `fetchCatalog()` instead of `fetchSuggestedModels()` |
| `Sources/NovaMLXMenuBar/DownloadsPageView.swift` | Catalog search by default; Hub search only if Advanced on |
| `Sources/NovaMLXMenuBar/SettingsPageView.swift` | Toggle under Models Path |
| `Sources/NovaMLXMenuBar/MenuBarAppState.swift` | Published flag + persist helper |
| `Sources/NovaMLXCore/LocalizationStrings.swift` | New keys in every locale table |
| `Sources/NovaMLXCLI/main.swift` | `nova download` uses `/admin/api/hf/download`; print 403 body |
| `suggested-models.json` | Delete after seed is copied to the website file |
| `~/dev/novamlx-website/src/routes/models/+page.svelte` | Render catalog; chips All / LLM / VLM / Embed / Audio / Image |
| `README.md` | Search/download copy: catalog, not “any HF model” |

---

### Task 1: Catalog types and policy (pure, no I/O)

**Files:**
- Create: `Sources/NovaMLXCore/ModelCatalog.swift`
- Test: `Tests/NovaMLXCoreTests/ModelCatalogTests.swift`

- [ ] **Step 1: Write the failing tests**

```swift
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/lucas/dev/novamlx
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalog
```

Expected: compile error `cannot find 'CatalogFile' in scope`.

- [ ] **Step 3: Implement types and policy**

Create `Sources/NovaMLXCore/ModelCatalog.swift`:

```swift
import Foundation

public enum CatalogFormat: String, Codable, Sendable {
    case mlx
    case gguf
}

public enum CatalogStatus: String, Codable, Sendable {
    case verified
    case preview
}

public struct CatalogEntry: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let url: String
    public let name: String
    public let category: ModelType
    public let family: ModelFamily
    public let format: CatalogFormat
    public let description: String?
    public let revision: String?
    public let quant: String?
    public let size: String?
    public let sizeBytes: UInt64?
    public let minRamGB: Int?
    public let tags: [String]
    public let capabilities: [String]
    public let testedOn: String?
    public let status: CatalogStatus

    public init(
        id: String,
        url: String,
        name: String,
        category: ModelType,
        family: ModelFamily,
        format: CatalogFormat,
        description: String? = nil,
        revision: String? = nil,
        quant: String? = nil,
        size: String? = nil,
        sizeBytes: UInt64? = nil,
        minRamGB: Int? = nil,
        tags: [String] = [],
        capabilities: [String] = [],
        testedOn: String? = nil,
        status: CatalogStatus = .verified
    ) {
        self.id = id
        self.url = url
        self.name = name
        self.category = category
        self.family = family
        self.format = format
        self.description = description
        self.revision = revision
        self.quant = quant
        self.size = size
        self.sizeBytes = sizeBytes
        self.minRamGB = minRamGB
        self.tags = tags
        self.capabilities = capabilities
        self.testedOn = testedOn
        self.status = status
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        url = try c.decode(String.self, forKey: .url)
        name = try c.decode(String.self, forKey: .name)
        category = try c.decode(ModelType.self, forKey: .category)
        family = try c.decode(ModelFamily.self, forKey: .family)
        format = try c.decode(CatalogFormat.self, forKey: .format)
        description = try c.decodeIfPresent(String.self, forKey: .description)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decodeIfPresent(String.self, forKey: .quant)
        size = try c.decodeIfPresent(String.self, forKey: .size)
        sizeBytes = try c.decodeIfPresent(UInt64.self, forKey: .sizeBytes)
        minRamGB = try c.decodeIfPresent(Int.self, forKey: .minRamGB)
        tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
        testedOn = try c.decodeIfPresent(String.self, forKey: .testedOn)
        status = try c.decodeIfPresent(CatalogStatus.self, forKey: .status) ?? .verified
    }
}

public struct CatalogFile: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let updatedAt: String?
    public let models: [CatalogEntry]

    public init(schemaVersion: Int, updatedAt: String? = nil, models: [CatalogEntry]) {
        self.schemaVersion = schemaVersion
        self.updatedAt = updatedAt
        self.models = models
    }

    public static func decode(_ data: Data) throws -> CatalogFile {
        try JSONDecoder().decode(CatalogFile.self, from: data)
    }
}

public enum ModelCatalogPolicy {
    public static func isDownloadAllowed(
        id: String,
        catalog: [CatalogEntry],
        allowUnlisted: Bool
    ) -> Bool {
        if allowUnlisted { return true }
        return catalog.contains { $0.id == id }
    }

    public static func entry(id: String, in catalog: [CatalogEntry]) -> CatalogEntry? {
        catalog.first { $0.id == id }
    }

    public static func refuseMessage(id: String) -> String {
        "\(id) is not in the NovaMLX verified catalog. Turn on Settings → Allow unverified downloads if you want to try it anyway."
    }

    public static func search(
        _ catalog: [CatalogEntry],
        query: String,
        category: ModelType?
    ) -> [CatalogEntry] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return catalog.filter { entry in
            if let category, entry.category != category { return false }
            if q.isEmpty { return true }
            if entry.id.lowercased().contains(q) { return true }
            if entry.name.lowercased().contains(q) { return true }
            if (entry.description ?? "").lowercased().contains(q) { return true }
            if entry.family.rawValue.lowercased().contains(q) { return true }
            if entry.tags.contains(where: { $0.lowercased().contains(q) }) { return true }
            return false
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalog
```

Expected: all `ModelCatalog` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXCore/ModelCatalog.swift Tests/NovaMLXCoreTests/ModelCatalogTests.swift
git commit -m "feat(engine): catalog types and allow/search policy"
```

---

### Task 2: ServerConfig.allowUnlistedDownloads + SQLite persist

**Files:**
- Modify: `Sources/NovaMLXCore/Types.swift` (`ServerConfig`)
- Modify: `Sources/NovaMLXCore/Configuration.swift`
- Modify: `Sources/NovaMLXDB/Models/ConfigRecords.swift`
- Modify: `Sources/NovaMLXDB/Migrations/ConfigDBSchema.swift`
- Modify: `Sources/NovaMLXDB/NovaDB.swift`
- Create: `Tests/NovaMLXCoreTests/ServerConfigAllowUnlistedTests.swift`
- Modify: `Tests/NovaMLXDBTests/ConfigStoreTests.swift`

- [ ] **Step 1: Write failing ServerConfig tests**

```swift
import Testing
import Foundation
import NovaMLXCore

@Suite("ServerConfig allowUnlistedDownloads")
struct ServerConfigAllowUnlistedTests {
    @Test("Defaults to false")
    func defaultOff() {
        #expect(ServerConfig().allowUnlistedDownloads == false)
    }

    @Test("Legacy JSON without the key decodes as false")
    func legacyJSON() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591 }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.allowUnlistedDownloads == false)
    }

    @Test("Decodes true")
    func decodesTrue() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591, "allowUnlistedDownloads": true }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.allowUnlistedDownloads == true)
    }
}
```

Add to `ConfigStoreTests.swift`:

```swift
@Test("v6 migration defaults allowUnlistedDownloads to false")
func v6MigrationDefault() async throws {
    let tmp = try makeTmpDir()
    let nova = NovaDB.shared
    try nova.setup(baseDir: tmp)
    let record = try nova.configStore.get()
    #expect(record.allowUnlistedDownloads == false)
}

@Test("ConfigStore persists allowUnlistedDownloads")
func persistAllowUnlisted() async throws {
    let tmp = try makeTmpDir()
    try NovaDB.shared.setup(baseDir: tmp)
    try NovaDB.shared.configStore.update { rec in
        rec.allowUnlistedDownloads = true
    }
    #expect(try NovaDB.shared.configStore.get().allowUnlistedDownloads == true)
}
```

- [ ] **Step 2: Run to verify fail**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ServerConfigAllowUnlisted
```

Expected: `'ServerConfig' has no member 'allowUnlistedDownloads'`.

- [ ] **Step 3: Add the field end-to-end**

In `Types.swift` `ServerConfig`:

- Add `public let allowUnlistedDownloads: Bool`
- Add `case allowUnlistedDownloads` to `CodingKeys`
- Add parameter `allowUnlistedDownloads: Bool = false` to `init` and assign it
- In `init(from:)`: `allowUnlistedDownloads = try container.decodeIfPresent(Bool.self, forKey: .allowUnlistedDownloads) ?? false`

In `ConfigRecords.swift`:

- Add `public var allowUnlistedDownloads: Bool = false`
- Add it to the public `init` (default `false`) and `CodingKeys` as `allow_unlisted_downloads`

In `ConfigDBSchema.swift` add:

```swift
static func v6AllowUnlistedDownloads(in db: Database) throws {
    try db.alter(table: "config") { t in
        t.add(column: "allow_unlisted_downloads", .boolean).notNull().defaults(to: false)
    }
}
```

In `NovaDB.swift` `runMigrations()`, after `v5_api_key_usage_events`:

```swift
configMigrator.registerMigration("v6_allow_unlisted_downloads") { db in
    try ConfigDBSchema.v6AllowUnlistedDownloads(in: db)
}
```

In `Configuration.swift` `syncToStore()` `ConfigRecord(...)` add `allowUnlistedDownloads: server.allowUnlistedDownloads`.

In `loadFromStore()` pass `allowUnlistedDownloads: record.allowUnlistedDownloads` into `ServerConfig(...)`.

Every other `ServerConfig(` call site keeps compiling because of the default `false`.

- [ ] **Step 4: Run tests**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ServerConfigAllowUnlisted
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ConfigStore
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXCore/Types.swift Sources/NovaMLXCore/Configuration.swift \
  Sources/NovaMLXDB/Models/ConfigRecords.swift Sources/NovaMLXDB/Migrations/ConfigDBSchema.swift \
  Sources/NovaMLXDB/NovaDB.swift Tests/NovaMLXCoreTests/ServerConfigAllowUnlistedTests.swift \
  Tests/NovaMLXDBTests/ConfigStoreTests.swift
git commit -m "feat(core): persist allowUnlistedDownloads (default off)"
```

---

### Task 3: Catalog store (fetch → disk cache → bundle)

**Files:**
- Create: `Sources/NovaMLXModelManager/ModelCatalogStore.swift`
- Modify: `Sources/NovaMLXCore/NovaMLXPaths.swift`
- Create: `Sources/NovaMLXUtils/Resources/catalog/models.json` (minimal fixture for tests / first bundle)
- Test: `Tests/NovaMLXModelManagerTests/ModelCatalogStoreTests.swift`

- [ ] **Step 1: Write failing store tests**

`ModelCatalogStore` must be injectable: remote URL, cache URL, bundle URL, and a `CatalogTransport` so tests never hit the network.

```swift
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
```

- [ ] **Step 2: Run to verify fail**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalogStore
```

Expected: `cannot find 'ModelCatalogStore' in scope`.

- [ ] **Step 3: Implement store + path**

Add to `NovaMLXPaths.swift`:

```swift
public static var catalogCacheDir: URL {
    let dir = baseDir.appendingPathComponent("cache/catalog", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}
public static var catalogCacheFile: URL {
    catalogCacheDir.appendingPathComponent("models.json")
}
```

Create `Sources/NovaMLXModelManager/ModelCatalogStore.swift`:

```swift
import Foundation
import NovaMLXCore
import NovaMLXUtils

public protocol CatalogTransport: Sendable {
    func data(from url: URL) async throws -> Data
}

public struct URLSessionCatalogTransport: CatalogTransport {
    public init() {}
    public func data(from url: URL) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}

public final class ModelCatalogStore: @unchecked Sendable {
    public static let defaultRemoteURL = URL(string: "https://novamlx.ai/catalog/models.json")!

    private let remoteURL: URL
    private let cacheURL: URL
    private let bundleURL: URL?
    private let transport: CatalogTransport
    private let lock = NSLock()
    private var _models: [CatalogEntry] = []

    public var models: [CatalogEntry] {
        lock.withLock { _models }
    }

    public init(
        remoteURL: URL = ModelCatalogStore.defaultRemoteURL,
        cacheURL: URL = NovaMLXPaths.catalogCacheFile,
        bundleURL: URL? = ModelCatalogStore.bundledSnapshotURL(),
        transport: CatalogTransport = URLSessionCatalogTransport()
    ) {
        self.remoteURL = remoteURL
        self.cacheURL = cacheURL
        self.bundleURL = bundleURL
        self.transport = transport
    }

    public static func bundledSnapshotURL() -> URL? {
        ResourceBundleLocator.url(
            forResource: "models",
            withExtension: "json",
            subdirectory: "catalog",
            inBundle: "NovaMLX_NovaMLXUtils"
        )
    }

    public func refresh() async {
        if let data = try? await transport.data(from: remoteURL),
           let file = try? CatalogFile.decode(data) {
            lock.withLock { _models = file.models }
            try? FileManager.default.createDirectory(
                at: cacheURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? data.write(to: cacheURL, options: .atomic)
            NovaMLXLog.info("[Catalog] loaded \(file.models.count) models from remote")
            return
        }
        if let data = try? Data(contentsOf: cacheURL),
           let file = try? CatalogFile.decode(data) {
            lock.withLock { _models = file.models }
            NovaMLXLog.info("[Catalog] loaded \(file.models.count) models from cache")
            return
        }
        if let bundleURL,
           let data = try? Data(contentsOf: bundleURL),
           let file = try? CatalogFile.decode(data) {
            lock.withLock { _models = file.models }
            NovaMLXLog.info("[Catalog] loaded \(file.models.count) models from bundle")
            return
        }
        NovaMLXLog.error("[Catalog] no catalog available")
    }
}
```

Write a **minimal valid** `Sources/NovaMLXUtils/Resources/catalog/models.json`:

```json
{
  "schemaVersion": 1,
  "updatedAt": "2026-08-16T00:00:00Z",
  "models": []
}
```

(Task 8 replaces this with the seeded list.)

- [ ] **Step 4: Run tests**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalogStore
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXModelManager/ModelCatalogStore.swift \
  Sources/NovaMLXCore/NovaMLXPaths.swift \
  Sources/NovaMLXUtils/Resources/catalog/models.json \
  Tests/NovaMLXModelManagerTests/ModelCatalogStoreTests.swift
git commit -m "feat(engine): catalog store with remote/cache/bundle fallback"
```

---

### Task 4: Replace SuggestedModel with the catalog store

**Files:**
- Modify: `Sources/NovaMLXModelManager/ModelManager.swift`
- Modify: `Sources/NovaMLXApp/main.swift`
- Modify: `Tests/NovaMLXModelManagerTests/ModelManagerTests.swift`

- [ ] **Step 1: Replace the GitHub suggested-models block in ModelManager**

Delete `suggestedModelsURL`, `SuggestedModel`, `_suggestedModelsCache`, `fetchSuggestedModels()`, `suggestedModels(forCategory:)`.

Add:

```swift
public let catalogStore: ModelCatalogStore

// in init(modelsDirectory:):
self.catalogStore = ModelCatalogStore()

public func fetchCatalog() async {
    await catalogStore.refresh()
}

public func catalogModels(forCategory category: ModelType?) -> [CatalogEntry] {
    ModelCatalogPolicy.search(catalogStore.models, query: "", category: category)
}

public func registerPopularModels() {
    for model in catalogStore.models {
        if lock.withLock({ _registry[model.id] }) == nil {
            register(
                id: model.id,
                family: model.family,
                modelType: model.category,
                remoteURL: model.url,
                sizeBytes: model.sizeBytes ?? 0
            )
        }
    }
}
```

Keep a temporary typealias only if a compile error remains:

```swift
public typealias SuggestedModel = CatalogEntry
```

Prefer updating call sites instead of the alias.

In `main.swift` replace:

```swift
await modelManager.fetchSuggestedModels()
```

with:

```swift
await modelManager.fetchCatalog()
```

Update `registerPopularModelsEmptyWithoutFetch` — still valid (empty store → no-op).

- [ ] **Step 2: Build**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh
```

Expected: compile succeeds. `DownloadsPageView` will fail until Task 6 if it still references `SuggestedModel` / `fetchSuggestedModels` — if so, add the typealias and a `fetchSuggestedModels()` wrapper that calls `fetchCatalog()`, then remove them in Task 6.

- [ ] **Step 3: Run ModelManager tests**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelManager
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXModelManager/ModelManager.swift Sources/NovaMLXApp/main.swift \
  Tests/NovaMLXModelManagerTests/ModelManagerTests.swift
git commit -m "refactor(engine): load catalog instead of GitHub suggested-models"
```

---

### Task 5: Gate admin search and both download endpoints

**Files:**
- Modify: `Sources/NovaMLXAPI/APIServer.swift` (hf search ~2906, hf download ~2952, models download ~2170)
- Modify: `Sources/NovaMLXModelManager/HuggingFaceService.swift` (`startDownload` + `resolveURL`)
- Modify: `Sources/NovaMLXCLI/main.swift`

There is no live-server unit test harness for these routes. Cover the **decision** in `ModelCatalogTests` (already done). This task wires that decision. Do not cache `allowUnlistedDownloads` on `APIServer`.

- [ ] **Step 1: Add a private helper on APIServer**

```swift
private func catalogAllowUnlisted() async -> Bool {
    await NovaMLXConfiguration.shared.serverConfig.allowUnlistedDownloads
}

private func catalogEntries() -> [CatalogEntry] {
    modelManager.catalogStore.models
}

private func refuseUnlistedIfNeeded(id: String) async -> Response? {
    let allowed = ModelCatalogPolicy.isDownloadAllowed(
        id: id,
        catalog: catalogEntries(),
        allowUnlisted: await catalogAllowUnlisted()
    )
    guard allowed else {
        return try? Self.jsonResponse(
            ["error": ModelCatalogPolicy.refuseMessage(id: id)],
            httpStatus: .forbidden
        )
    }
    return nil
}
```

- [ ] **Step 2: Gate `GET /admin/api/hf/search`**

Before calling `searchService.searchModels`:

```swift
if await !self.catalogAllowUnlisted() {
    let category: ModelType? = {
        switch params["category"] {
        case "llm": return .llm
        case "vlm": return .vlm
        case "embedding": return .embedding
        case "audio": return .audio
        case "image": return .image
        default: return nil
        }
    }()
    let hits = ModelCatalogPolicy.search(self.catalogEntries(), query: q, category: category)
    let models = hits.map { entry in
        HFModelInfo(
            id: entry.id,
            author: nil,
            downloads: nil,
            likes: nil,
            trendingScore: nil,
            tags: entry.tags,
            pipelineTag: entry.category.rawValue,
            createdAt: nil,
            lastModified: nil,
            privateRepo: false,
            gated: false
        )
    }
    return try Self.jsonResponse(HFSearchResult(models: models, total: models.count))
}
```

Keep the existing Hub branch when Advanced is on.

- [ ] **Step 3: Gate `POST /admin/api/hf/download`**

After validating `repoId`:

```swift
if let refused = await self.refuseUnlistedIfNeeded(id: repoId) {
    return refused
}
```

Then start the download. If the id is in the catalog, pass `revision` from the entry:

```swift
let entry = ModelCatalogPolicy.entry(id: repoId, in: self.catalogEntries())
let task = try await hf.startDownload(
    repoId: repoId,
    hfToken: hfToken,
    mirrorEndpoint: endpoint,
    revision: entry?.revision
)
```

- [ ] **Step 4: Gate `POST /admin/models/download`**

After decoding `req.modelId`:

```swift
if let refused = await self.refuseUnlistedIfNeeded(id: req.modelId) {
    return refused
}
```

If Advanced is on and `models.getRecord(req.modelId) == nil`, register it first (`family: .other`, `remoteURL: "https://huggingface.co/\(req.modelId)"`) so the existing Hub download path can run.

- [ ] **Step 5: Optional revision on HuggingFaceService**

Add `revision: String? = nil` to `startDownload`. Thread it into `runDownload` / `resolveURL`. When non-nil, `HFMirrorAdapter.resolveURL` uses that commit instead of `defaultRevision` (`main`). When nil, keep today’s behavior.

- [ ] **Step 6: Point `nova download` at the gated HF endpoint**

In `Sources/NovaMLXCLI/main.swift` `handleDownload`:

```swift
let resp = try await CLIClient.post(
    "/admin/api/hf/download",
    body: "{\"repo_id\":\"\(modelId)\"}",
    admin: true
)
if resp.statusCode == 200 {
    print("Download started.")
} else {
    print("Failed: \(resp.statusCode) — \(resp.body)")
}
```

`handleSearch` stays on `/admin/api/hf/search` — the server now returns catalog hits when Advanced is off.

- [ ] **Step 7: Build**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalog
```

Expected: build 0 warnings on touched files; catalog tests still PASS.

- [ ] **Step 8: Commit**

```bash
git add Sources/NovaMLXAPI/APIServer.swift \
  Sources/NovaMLXModelManager/HuggingFaceService.swift \
  Sources/NovaMLXCLI/main.swift
git commit -m "feat(api): gate search and download on verified catalog"
```

---

### Task 6: Downloads tab + Settings toggle

**Files:**
- Modify: `Sources/NovaMLXMenuBar/DownloadsPageView.swift`
- Modify: `Sources/NovaMLXMenuBar/SettingsPageView.swift`
- Modify: `Sources/NovaMLXMenuBar/MenuBarAppState.swift`
- Modify: `Sources/NovaMLXCore/LocalizationStrings.swift`

- [ ] **Step 1: Add localization keys to every locale table**

| Key | en | zh-CN | zh-TW / zh-HK | ja | ko | fr | de | ru |
|---|---|---|---|---|---|---|---|---|
| `settings.allowUnlisted` | Allow unverified downloads | 允许下载未验证模型 | 允許下載未驗證模型 | 未検証モデルのダウンロードを許可 | 미검증 모델 다운로드 허용 | Autoriser les téléchargements non vérifiés | Ungeprüfte Downloads erlauben | Разрешить непроверенные загрузки |
| `settings.allowUnlistedCaption` | Search and download any Hugging Face URL. Unverified models may fail to load. | 可搜索并下载任意 Hugging Face 地址。未验证模型可能无法加载。 | 可搜尋並下載任意 Hugging Face 網址。未驗證模型可能無法載入。 | 任意の Hugging Face URL を検索・ダウンロードできます。未検証モデルは読み込めない場合があります。 | 임의의 Hugging Face URL을 검색·다운로드합니다. 미검증 모델은 로드에 실패할 수 있습니다. | Rechercher et télécharger n’importe quelle URL Hugging Face. Les modèles non vérifiés peuvent échouer. | Beliebige Hugging-Face-URL suchen und laden. Ungeprüfte Modelle können fehlschlagen. | Поиск и загрузка любого URL Hugging Face. Непроверенные модели могут не загрузиться. |
| `models.verifiedTitle` | Verified models | 已验证模型 | 已驗證模型 | 検証済みモデル | 검증된 모델 | Modèles vérifiés | Geprüfte Modelle | Проверенные модели |
| `models.unverifiedBanner` | Unverified downloads are on. Models may not load. | 已开启未验证下载。模型可能无法加载。 | 已開啟未驗證下載。模型可能無法載入。 | 未検証ダウンロードがオンです。読み込めない場合があります。 | 미검증 다운로드가 켜져 있습니다. 로드에 실패할 수 있습니다. | Téléchargements non vérifiés activés. Les modèles peuvent échouer. | Ungeprüfte Downloads sind an. Modelle können fehlschlagen. | Непроверенные загрузки включены. Модель может не загрузиться. |
| `models.previewBadge` | Preview | 预览 | 預覽 | プレビュー | 미리보기 | Aperçu | Vorschau | Предпросмотр |

Insert each key next to the existing `settings.modelsPath` / `models.search` entries so the tables stay grouped.

- [ ] **Step 2: Publish the flag on MenuBarAppState**

```swift
@Published public var allowUnlistedDownloads: Bool = false

public func setAllowUnlistedDownloads(_ enabled: Bool) async {
    allowUnlistedDownloads = enabled
    let current = await NovaMLXConfiguration.shared.serverConfig
    let updated = ServerConfig(
        host: current.host,
        port: current.port,
        adminPort: current.adminPort,
        maxConcurrentRequests: current.maxConcurrentRequests,
        requestTimeout: current.requestTimeout,
        contextScalingTarget: current.contextScalingTarget,
        tlsCertPath: current.tlsCertPath,
        tlsKeyPath: current.tlsKeyPath,
        tlsKeyPassword: current.tlsKeyPassword,
        maxRequestSizeMB: current.maxRequestSizeMB,
        maxProcessMemory: current.maxProcessMemory,
        prefixCacheEnabled: current.prefixCacheEnabled,
        autoLoad: current.autoLoad,
        cluster: current.cluster,
        allowUnlistedDownloads: enabled
    )
    await NovaMLXConfiguration.shared.setServerConfig(updated)
    await NovaMLXConfiguration.shared.syncToStore()
}
```

On app launch (where `appState.serverPort` is assigned) also set `appState.allowUnlistedDownloads = serverConfig.allowUnlistedDownloads`.

When Settings saves a full `ServerConfig`, **preserve** `allowUnlistedDownloads` from `appState` / current config so the editor cannot silently reset the toggle.

- [ ] **Step 3: Settings toggle under Models Path**

In `SettingsPageView` `configPathRow` (or a new row immediately below it), outside the Edit Config panel:

```swift
Toggle(isOn: Binding(
    get: { appState.allowUnlistedDownloads },
    set: { newValue in
        Task { await appState.setAllowUnlistedDownloads(newValue) }
    }
)) {
    VStack(alignment: .leading, spacing: 2) {
        Text(l10n.tr("settings.allowUnlisted"))
            .font(.system(size: 13))
        Text(l10n.tr("settings.allowUnlistedCaption"))
            .font(.system(size: 11))
            .foregroundColor(.secondary)
    }
}
.toggleStyle(.switch)
.padding(.horizontal, 12)
.padding(.vertical, 8)
```

- [ ] **Step 4: Downloads tab behavior**

Replace `suggestedModelsSection` to use `modelManager.catalogModels(forCategory: typeFilter.matchType)` and title `l10n.tr("models.verifiedTitle")`. Remove the subtitle “use the search above to find and download any model from Hugging Face.”

Show a Preview badge when `model.status == .preview`.

`.task { await modelManager.fetchCatalog() }`

`performSearch()`:

- If `!appState.allowUnlistedDownloads`: locally filter `modelManager.catalogStore.models` with `ModelCatalogPolicy.search`; set `searchResults` from those ids/tags; **do not** call `/admin/api/hf/search`.
- If Advanced on: keep today’s Hub search (mirror, regex, `mlx-community/`).

UI:

- Advanced off: hide `regex` and `mlx-community/` toggles; keep Mirror.
- Advanced on: show a banner `l10n.tr("models.unverifiedBanner")`.
- Delete `showCompatWarning` / `compatWarningRepoId` / the “doesn’t look like MLX” alert. `triggerDownload` always calls `appState.startDownload(repoId:)`.
- Existing 403 handling in `startDownload` already surfaces `error` from the JSON body.

Origin link on a catalog card: open `model.url`, not a hardcoded `https://huggingface.co/\(repo)`.

- [ ] **Step 5: Build**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh
```

Expected: 0 errors, 0 new warnings.

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXMenuBar/DownloadsPageView.swift \
  Sources/NovaMLXMenuBar/SettingsPageView.swift \
  Sources/NovaMLXMenuBar/MenuBarAppState.swift \
  Sources/NovaMLXCore/LocalizationStrings.swift \
  Sources/NovaMLXApp/main.swift
git commit -m "feat(ui): catalog-only downloads; Advanced toggle in Settings"
```

---

### Task 7: Website catalog file + `/models` page

**Files (website repo `~/dev/novamlx-website`):**
- Create: `static/catalog/models.json`
- Modify: `src/routes/models/+page.svelte`

- [ ] **Step 1: Seed `static/catalog/models.json` from NovaMLX `suggested-models.json`**

Transform each object:

| old | new |
|---|---|
| `repo` | `id` |
| — | `url` = `https://huggingface.co/` + `repo` |
| `name`, `description`, `size`, `tags`, `category`, `sizeBytes`, `family` | unchanged |
| — | `format`: `"mlx"` |
| — | `status`: `"preview"` |

Do **not** mark the seed `verified`. Wrap in:

```json
{
  "schemaVersion": 1,
  "updatedAt": "<ISO-8601 now>",
  "models": [ ... ]
}
```

Validate with `python3 -m json.tool static/catalog/models.json >/dev/null`.

- [ ] **Step 2: Rewrite `/models` to fetch that file**

Replace the hardcoded `models` array. On mount:

```ts
interface CatalogEntry {
	id: string;
	url: string;
	name: string;
	category: 'llm' | 'vlm' | 'embedding' | 'audio' | 'image';
	family: string;
	format: 'mlx' | 'gguf';
	description?: string;
	size?: string;
	status?: 'verified' | 'preview';
}

let entries = $state<CatalogEntry[]>([]);
let loadError = $state(false);
let filter = $state<'all' | 'llm' | 'vlm' | 'embedding' | 'audio' | 'image'>('all');
let search = $state('');

$effect(() => {
	fetch('/catalog/models.json')
		.then((r) => {
			if (!r.ok) throw new Error(String(r.status));
			return r.json();
		})
		.then((data) => {
			entries = data.models ?? [];
		})
		.catch(() => {
			loadError = true;
		});
});
```

Chips (replace All / Trending / Vision / Tool Calling):

```
All / LLM / VLM / Embed / Audio / Image
```

Keys: `all`, `llm`, `vlm`, `embedding`, `audio`, `image`. Filter by `entry.category`. Search by `name`, `id`, `family`.

Table columns: name (+ Preview badge if `status === 'preview'`), category, size, format, link to `url`.

Empty / error: “Catalog unavailable.” Do not fall back to the old hardcoded table.

Remove “50+ model families” and “And 40+ more model families”.

- [ ] **Step 3: Locally verify**

```bash
cd ~/dev/novamlx-website
# serve however the repo already does, e.g.
npm run dev
curl -sf http://localhost:5173/catalog/models.json | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['schemaVersion']==1; print(len(d['models']))"
```

Open `/models`, click each chip, confirm the table filters. Confirm a failed fetch (rename the file temporarily) shows “Catalog unavailable.”

- [ ] **Step 4: Commit in the website repo**

```bash
cd ~/dev/novamlx-website
git add static/catalog/models.json src/routes/models/+page.svelte
git commit -m "feat: verified model catalog JSON and /models category chips"
```

Deploy with the repo’s existing `deploy.sh` when ready so `https://novamlx.ai/catalog/models.json` is live before or with the app release.

---

### Task 8: Bundle snapshot, delete GitHub list, docs

**Files:**
- Replace: `Sources/NovaMLXUtils/Resources/catalog/models.json` with a copy of the website file
- Delete: `suggested-models.json`
- Modify: `README.md` (search/download wording around the `nova search` / “Any SafeTensors model from HuggingFace” lines)

- [ ] **Step 1: Copy the seeded catalog into the app bundle**

```bash
cp ~/dev/novamlx-website/static/catalog/models.json \
   /Users/lucas/dev/novamlx/Sources/NovaMLXUtils/Resources/catalog/models.json
```

- [ ] **Step 2: Delete `suggested-models.json`**

Grep first:

```bash
rg -n "suggested-models" /Users/lucas/dev/novamlx --glob '!docs/**'
```

Remove any leftover references, then `git rm suggested-models.json`.

- [ ] **Step 3: README**

Change Hub-open-ended search copy to: browse/search the verified catalog; enable **Settings → Allow unverified downloads** for arbitrary URLs.

- [ ] **Step 4: Full test + build**

```bash
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalog
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelCatalogStore
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ServerConfigAllowUnlisted
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ConfigStore
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh test --filter ModelManager
```

Expected: all PASS, 0 warnings on touched files.

- [ ] **Step 5: Manual smoke (after website deploy or against a local catalog URL override if you add one for debug)**

1. Advanced off: Downloads tab shows seeded cards; search “qwen” filters locally; download of `not-a-real/model` shows the refuse message (GUI + `nova download not-a-real/model` → 403).
2. Toggle Advanced on: banner appears; Hub search returns; unlisted download is attempted.
3. Toggle off: Hub search stops; local models still load.
4. Quit and relaunch: toggle state persisted.

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXUtils/Resources/catalog/models.json README.md
git rm suggested-models.json
git commit -m "chore: ship catalog snapshot and drop GitHub suggested-models"
```

---

## Spec coverage

| Spec requirement | Task |
|---|---|
| Catalog-only browse/search/download by default | 5, 6 |
| Advanced toggle, off by default, immediate, persisted | 2, 6 |
| GUI + CLI + admin same gate | 5 |
| Local disk / load / unload never gated | 5 (only search/download paths) |
| File at novamlx.ai/catalog/models.json | 7 |
| Disk cache + bundled snapshot | 3, 8 |
| Envelope + entry fields + forward compatible | 1 |
| Search is local over catalog when Advanced off | 5, 6 |
| Download uses catalog url + pinned revision | 5 |
| Mirror is transport only | unchanged + 6 keeps Mirror picker |
| Seed from suggested-models as `preview` | 7, 8 |
| Website chips All / LLM / VLM / Embed / Audio / Image | 7 |
| No hardcoded Swift model array | 3, 4 |
| Drop GitHub raw fetch | 4, 8 |
| No CMS / auto-import | not built |

## Type consistency

- `CatalogFile` / `CatalogEntry` / `CatalogFormat` / `CatalogStatus` / `ModelCatalogPolicy` defined in Task 1, used in Tasks 3–6.
- `ServerConfig.allowUnlistedDownloads` defined in Task 2, read live in Task 5, written in Task 6.
- `ModelCatalogStore.models` / `refresh()` defined in Task 3, owned by `ModelManager` in Task 4.
- Refuse copy is exactly `ModelCatalogPolicy.refuseMessage(id:)` everywhere.

## Out of plan (spec follow-ups)

- In-session catalog refresh without relaunch
- Catalog validation CLI / stub-entry helper
- Pinning revisions on the seed list
- GGUF entries beyond what was actually run
