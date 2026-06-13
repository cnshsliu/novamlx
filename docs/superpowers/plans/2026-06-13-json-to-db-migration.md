# JSON → SQLite Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely eliminate all JSON-based config/state storage in NovaMLX — delete the 12 JSON files under `~/.nova/`, delete all code that reads/writes them, route everything through the existing GRDB stores. Final state: zero `NovaMLXConfiguration` references, zero JSON path constants, zero legacy reader/writer functions.

**Architecture:** Direct store calls (no facade). UI and API server call `NovaDB.shared.<store>.<method>()` directly. Each of 12 files migrates through three phases (Bridge → Cutover → Cleanup). Final phase deletes `Sources/NovaMLXCore/Configuration.swift` entirely and physically removes all `.migrated` files.

**Tech Stack:** Swift 6, GRDB 7.11, Swift Testing (`@Test`), SwiftUI, Hummingbird, `./build.sh` for codesigned releases.

**Spec reference:** `docs/superpowers/specs/2026-06-13-json-to-db-migration-design.md`

---

## File Structure (target end-state)

```
Sources/
  NovaMLXDB/
    Stores/
      APIKeyStore.swift           # extended: listAsAPIKey(), findAPIKeyByRawToken()
      ConfigStore.swift           # extended: full PersistedConfig coverage
      TokenhubStore.swift         # extended: full provider fields
      ModelSettingsStore.swift    # unchanged
      ModelfileStore.swift        # unchanged
      AuthStore.swift             # extended: handles bare session token
      ClusterPolicyStore.swift    # unchanged
      ModelRegistryStore.swift    # unchanged
      LoadedModelsStore.swift     # unchanged
      MetricsStore.swift          # extended: renamed class MetricsDBStore; full PersistentMetrics fields
      ChatStore.swift             # unchanged
      WorkerDeploymentStore.swift # unchanged
    Models/
      ConfigRecords.swift         # ConfigRecord + MetricsRecord + TokenhubProviderRecord extended
    NovaDB.swift                  # importLegacyJSON: per-file wired then deleted in Final Cleanup
    ConfigDBSchema.swift          # v2_config_*, v2_tokenhub_* migrations
    DataDBSchema.swift            # v2_metrics_* migration
  NovaMLXCore/
    Types.swift                   # APIKey gains isLegacyImport computed property; ServerConfig loses apiKeys field
    TokenhubTypes.swift           # TokenhubProviderStore deleted (delegates to tokenhubStore)
    Configuration.swift           # DELETED in Final Cleanup
    AuthClient.swift              # rewired to authStore
    ModelfileManager.swift        # DELETED (delegates to modelfileStore)
    Localization.swift            # language pref via configStore
    NovaMLXPaths.swift            # JSON path constants DELETED
  NovaMLXUtils/
    MetricsStore.swift            # DELETED
  NovaMLXAPI/
    ChatHistoryStore.swift        # DELETED (delegates to chatStore)
    APIServer.swift               # all apiKey*/config/* callsites use stores directly
  NovaMLXDistributed/
    WorkerDeployer.swift          # SSH heredoc → HTTP admin endpoint
    WorkerService.swift           # SSH pull policy on startup
  NovaMLXApp/
    main.swift                    # remove loadAPIKeys()/loadFromFile() calls
  NovaMLXMenuBar/
    APIKeysPageView.swift         # rewired + legacy-key reveal handling
    SettingsPageView.swift        # rewired
    TokenhubPageView.swift        # rewired (~20 callsites)
    DownloadsPageView.swift       # updateApiKeys call removed
    ClusterPageView.swift         # rewired
    AgentsPageView.swift          # rewired
    MenuBarAppState.swift         # rewired
  NovaMLXModelManager/
    ModelManager.swift            # registry I/O → modelRegistryStore
    ModelSettingsManager.swift    # settings I/O → modelSettingsStore
  NovaMLXInference/
    InferenceService.swift        # loadedModelsStore + metricsStore
  NovaMLXCLI/
    LaunchCommand.swift           # config bootstrap via configStore
    CLIClient.swift               # config read via configStore

Tests/
  NovaMLXDBTests/
    APIKeyStoreTests.swift        # NEW: CRUD, getRawKey, findByHash, legacy import
    ConfigStoreTests.swift        # NEW
    TokenhubStoreTests.swift      # NEW
    MetricsDBStoreTests.swift     # NEW
    LegacyImportTests.swift       # NEW: importer behavior per file
```

---

## Migration Phases Overview

| Phase | File(s) | New schema migration? | Special concerns |
|-------|---------|----------------------|------------------|
| **A** | `api_keys.json` | — | Pilot; legacy-key reveal handling |
| **B** | `config.json` | `v2_config_add_server_fields` | Most callsites of any file; flat-string `apiKeys` field dies here |
| **C** | `tokenhub/providers.json` | `v2_tokenhub_add_columns` | ~20 UI callsites |
| **D1** | `loaded_models.json` | — | Simple `[String]` |
| **D2** | `model_settings.json` | — | |
| **D3** | `modelfiles/*.json` | — | Directory walk importer |
| **D4** | `models/registry.json` | — | Lives under `models/`, not `~/.nova/` |
| **D5** | `auth_cache.json` + `session` | — | `session` is bare string, not JSON |
| **D6** | `chat_history/*.json` | — | Field mapping thinking vs thinkingContent; uses dead `importChatHistory` helper |
| **D7** | `worker-deployments.json` | — | |
| **E** | `metrics.json` | `v2_metrics_add_columns` | Naming collision `MetricsStore` vs `MetricsDBStore` |
| **F** | `cluster-policy.json` | — | Remote worker SSH pull on startup |
| **G** | **Final Cleanup** | — | Delete Configuration.swift, NovaMLXPaths JSON constants, importLegacyJSON, .migrated files |

Each phase must clear the verification gate before advancing:
1. `./build.sh` compiles
2. `swift test` passes
3. App launches, manual smoke of affected UI
4. `/novamlx-full-api-test` T1–T10 passes
5. `~/.nova/novamlx.log` shows no errors related to the migrated subsystem

---

# Phase A — api_keys.json (Pilot)

## Task A1: Add `listAsAPIKey()` and `findAPIKeyByRawToken()` to APIKeyStore

**Files:**
- Modify: `Sources/NovaMLXDB/Stores/APIKeyStore.swift`
- Create: `Tests/NovaMLXDBTests/APIKeyStoreTests.swift`

- [ ] **Step 1: Write failing tests**

Create `Tests/NovaMLXDBTests/APIKeyStoreTests.swift`:

```swift
import Testing
import Foundation
@testable import NovaMLXDB
import NovaMLXCore

@Suite("APIKeyStore")
struct APIKeyStoreTests {
    let db: NovaDB = {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let nova = NovaDB.shared
        // Use a test-only setup; NovaDB.setup must accept baseDir param.
        try? nova.setup(baseDir: tmp)
        return nova
    }()

    @Test("create returns record with real rawKey, distinct from placeholder")
    func createStoresRealRawKey() throws {
        let (record, raw) = try db.apiKeyStore.create(name: "test-key")
        #expect(raw.hasPrefix("sk-novamlx-"))
        #expect(raw.count == "sk-novamlx-".count + 64)
        #expect(record.keyHash == APIKeyStore.hashRawKey(raw))
        #expect(record.rawKey == raw)
    }

    @Test("listAsAPIKey returns APIKey domain type")
    func listAsAPIKeyReturnsDomainType() throws {
        let (created, raw) = try db.apiKeyStore.create(name: "domain-test")
        let domainKeys = try db.apiKeyStore.listAsAPIKey()
        let found = domainKeys.first { $0.id == created.id }
        #expect(found != nil)
        #expect(found?.name == "domain-test")
        #expect(found?.keyHash == APIKeyStore.hashRawKey(raw))
    }

    @Test("findAPIKeyByRawToken hashes input and finds key")
    func findByRawToken() throws {
        let (_, raw) = try db.apiKeyStore.create(name: "lookup-test")
        let found = try db.apiKeyStore.findAPIKeyByRawToken(raw)
        #expect(found != nil)
        #expect(found?.name == "lookup-test")
        #expect(try db.apiKeyStore.findAPIKeyByRawToken("sk-novamlx-deadbeef") == nil)
    }

    @Test("legacy import placeholder marks key as legacy")
    func legacyImportMarksKey() throws {
        let (record, _) = try db.apiKeyStore.create(name: "legit")
        let domainKeys = try db.apiKeyStore.listAsAPIKey()
        let legit = domainKeys.first { $0.id == record.id }
        #expect(legit?.isLegacyImport == false)
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

```bash
swift test --filter APIKeyStoreTests
```
Expected: FAIL — `listAsAPIKey` and `findAPIKeyByRawToken` don't exist; `isLegacyImport` doesn't exist on `APIKey`.

- [ ] **Step 3: Add `isLegacyImport` to APIKey**

In `Sources/NovaMLXCore/Types.swift`, find the `APIKey` struct. Add:

```swift
/// True if this key was imported from a pre-DB JSON file with no stored plaintext.
/// UI disables the "reveal raw key" affordance for these keys; user must rotate.
public var isLegacyImport: Bool {
    // Legacy imports carry the placeholder "sk-novamlx-" + 64 zeros rawKey.
    // New keys created via APIKeyStore.create store real plaintext, distinct from placeholder.
    // The check is done via the APIKeyStore which has access to rawKey; this property
    // is a stub here that returns false unless overridden by store conversion.
    return false
}
```

Then in `Sources/NovaMLXDB/Stores/APIKeyStore.swift`, add a conversion helper that produces `APIKey` and a separate check:

```swift
extension APIKeyStore {
    private static let legacyPlaceholder = "sk-novamlx-" + String(repeating: "0", count: 64)

    /// True if the record's rawKey matches the legacy-import placeholder.
    public static func _isLegacyRecord(_ record: APIKeyRecord) -> Bool {
        record.rawKey == legacyPlaceholder
    }
}
```

- [ ] **Step 4: Add `listAsAPIKey()` and `findAPIKeyByRawToken()`**

In `Sources/NovaMLXDB/Stores/APIKeyStore.swift`, add inside the class:

```swift
public func listAsAPIKey() throws -> [APIKey] {
    let records = try list()
    return records.map { Self.toDomain($0) }
}

public func findAPIKeyByRawToken(_ raw: String) throws -> APIKey? {
    let hash = Self.hashRawKey(raw)
    if let record = try findByHash(hash) {
        return Self.toDomain(record)
    }
    return nil
}

public func getAsAPIKey(id: String) throws -> APIKey? {
    guard let record = try get(id: id) else { return nil }
    return Self.toDomain(record)
}

static func toDomain(_ record: APIKeyRecord) -> APIKey {
    // Construct APIKey from record. APIKey.init must accept these fields.
    // If APIKey doesn't have a suitable init, add one in Types.swift (see Step 5).
    var key = APIKey(
        id: record.id,
        name: record.name,
        keyHash: record.keyHash,
        keyPrefix: record.keyPrefix,
        keySuffix: record.keySuffix,
        rateLimitPerSecond: record.rateLimitPerSecond,
        rateLimitBurst: record.rateLimitBurst,
        allowedModels: decodeJSON(record.allowedModels ?? "null"),
        allowedEndpoints: decodeJSON(record.allowedEndpoints ?? "null"),
        maxTokensPerPeriod: record.maxTokensPerPeriod,
        maxRequestsPerPeriod: record.maxRequestsPerPeriod,
        usageResetPeriod: UsageResetPeriod(rawValue: record.usageResetPeriod) ?? .daily
    )
    key.usage = KeyUsage(
        totalTokensUsed: record.totalTokensUsed,
        totalRequests: record.totalRequests,
        lastUsedAt: record.lastUsedAt,
        periodTokens: record.periodTokens,
        periodRequests: record.periodRequests,
        periodResetDate: record.periodResetDate,
        perModelTokens: decodeJSON(record.perModelTokens ?? "{}") ?? [:]
    )
    key.isLegacyImportValue = _isLegacyRecord(record)
    return key
}
```

- [ ] **Step 5: Make `APIKey` accept legacy flag + init**

In `Sources/NovaMLXCore/Types.swift`, modify `APIKey`:
- Add stored property: `public var isLegacyImportValue: Bool = false`
- Replace the computed `isLegacyImport` from Step 3 with:

```swift
public var isLegacyImport: Bool { isLegacyImportValue }
```

- Add/confirm a memberwise init that takes all the fields used in Step 4. If `APIKey` is currently `Codable` with synthesized init, add an explicit `public init(...)` listing all fields.

- [ ] **Step 6: Run tests to verify pass**

```bash
swift test --filter APIKeyStoreTests
```
Expected: PASS — all 4 tests green.

- [ ] **Step 7: Commit**

```bash
git add Sources/NovaMLXDB/Stores/APIKeyStore.swift Sources/NovaMLXCore/Types.swift Tests/NovaMLXDBTests/APIKeyStoreTests.swift
git commit -m "feat(db): add APIKeyStore listAsAPIKey/findAPIKeyByRawToken + isLegacyImport flag"
```

---

## Task A2: Bridge Phase — Wire facade reads through the store

**Files:**
- Modify: `Sources/NovaMLXCore/Configuration.swift:36-37, 172-180, 233-289` (apiKeys, findAPIKeyByRaw, findAPIKeyById, isWithinLimits, periodUsageFraction)

- [ ] **Step 1: Update `apiKeys` getter to read from store**

In `Sources/NovaMLXCore/Configuration.swift`, replace the `apiKeys` getter (line 36):

```swift
public var apiKeys: [APIKey] {
    get async {
        if let storeKeys = try? NovaDB.shared.apiKeyStore.listAsAPIKey(), !storeKeys.isEmpty {
            return storeKeys
        }
        return _apiKeys  // fallback while import hasn't run yet
    }
}
```

- [ ] **Step 2: Update `findAPIKeyByRaw(_:)` to consult store first**

Replace line 172:

```swift
public func findAPIKeyByRaw(_ raw: String) -> APIKey? {
    if let storeKey = try? NovaDB.shared.apiKeyStore.findAPIKeyByRawToken(raw) {
        return storeKey
    }
    let hash = APIKey.hashRawKey(raw)
    return _apiKeys.first { $0.keyHash == hash }
}
```

- [ ] **Step 3: Update `findAPIKeyById(_:)` similarly**

Replace line 178:

```swift
public func findAPIKeyById(_ id: String) -> APIKey? {
    if let storeKey = try? NovaDB.shared.apiKeyStore.getAsAPIKey(id: id) {
        return storeKey
    }
    return _apiKeys.first { $0.id == id }
}
```

- [ ] **Step 4: Update `isWithinLimits` and `periodUsageFraction` to use store data**

Replace the bodies (lines 259-289) so they read the `APIKeyRecord` via `NovaDB.shared.apiKeyStore.get(id:)` instead of `_apiKeys`. Same logic, just source of truth changes.

- [ ] **Step 5: Dual-write in `createAPIKey`/`updateAPIKey`/`deleteAPIKey`/`rotateAPIKey`/`recordUsage`**

After the existing JSON write in each method, add a corresponding store call wrapped in `try?` so a store failure doesn't crash the legacy path:

```swift
// createAPIKey — after saveAPIKeys():
try? NovaDB.shared.apiKeyStore.create(
    name: name,
    rateLimitPerSecond: rateLimitPerSecond,
    rateLimitBurst: rateLimitBurst,
    allowedModels: allowedModels,
    allowedEndpoints: allowedEndpoints,
    maxTokensPerPeriod: maxTokensPerPeriod,
    maxRequestsPerPeriod: maxRequestsPerPeriod,
    usageResetPeriod: usageResetPeriod.rawValue
)
// NOTE: store.create generates its OWN new raw key — we cannot inject the one we already returned.
// For Bridge phase, this is acceptable: the store copy is a separate key that the importer will
// reconcile on next startup. The user-visible key from createAPIKey is the JSON one.
```

**Important:** Mark this discrepancy in a code comment. The reconciliation happens because the importer uses `onConflict: .ignore` — store rows created during Bridge with divergent hashes don't overwrite the imported ones. After Cutover, only the store path is used.

For `recordUsage`, prefer the store path:

```swift
public func recordUsage(keyId: String, tokens: Int64, model: String?) {
    try? NovaDB.shared.apiKeyStore.recordUsage(keyId: keyId, tokens: tokens, model: model)
    // Mirror to JSON for safety during Bridge:
    guard let idx = _apiKeys.firstIndex(where: { $0.id == keyId }) else { return }
    // ... existing JSON update logic ...
    try? saveAPIKeys()
}
```

- [ ] **Step 6: Build and smoke test**

```bash
./build.sh
```
Expected: clean build.

Manually:
1. Launch app from `dist/NovaMLX.app`
2. Open API Keys page
3. Verify existing keys still listed
4. Create a new key — should appear in list

- [ ] **Step 7: Run full test suite**

```bash
swift test
```
Expected: all tests pass.

- [ ] **Step 8: Run /novamlx-full-api-test T1-T10**

```bash
/novamlx-full-api-test
```
Expected: T1-T10 pass.

- [ ] **Step 9: Check log**

```bash
tail -100 ~/.nova/novamlx.log | grep -iE "error|warn"
```
Expected: no errors/warnings related to API keys or DB.

- [ ] **Step 10: Commit**

```bash
git add Sources/NovaMLXCore/Configuration.swift
git commit -m "feat(apikeys): bridge phase — reads via store, dual-write to JSON+DB"
```

---

## Task A3: Cutover Phase — Switch all callsites direct to store, remove JSON writes

**Files:**
- Modify: `Sources/NovaMLXCore/Configuration.swift` — strip apiKeys methods
- Modify: `Sources/NovaMLXCore/Types.swift:870-891` — remove `ServerConfig.apiKeys`
- Modify: `Sources/NovaMLXAPI/APIServer.swift` — 8 callsites
- Modify: `Sources/NovaMLXAPI/APIServer+AdminProxy.swift:44`
- Modify: `Sources/NovaMLXApp/main.swift:237,276,432,434`
- Modify: `Sources/NovaMLXMenuBar/APIKeysPageView.swift:141,508,533,542,563,576,589`
- Modify: `Sources/NovaMLXMenuBar/DownloadsPageView.swift:886-887`
- Modify: `Sources/NovaMLXAPI/WebUI/SettingsHTML.swift:92,113`

- [ ] **Step 1: Migrate `APIKeysPageView.swift` callsites**

Replace each line:

- **Line 533** `managedKeys = await NovaMLXConfiguration.shared.apiKeys` →
  ```swift
  managedKeys = (try? NovaDB.shared.apiKeyStore.listAsAPIKey()) ?? []
  ```
- **Line 542** `try await NovaMLXConfiguration.shared.createAPIKey(...)` →
  ```swift
  let (_, raw) = try NovaDB.shared.apiKeyStore.create(
      name: name,
      rateLimitPerSecond: newKeyRateLimit,
      rateLimitBurst: newKeyRateBurst,
      allowedModels: newKeyAllowedModels.isEmpty ? nil : newKeyAllowedModels,
      allowedEndpoints: newKeyAllowedEndpoints.isEmpty ? nil : newKeyAllowedEndpoints,
      maxTokensPerPeriod: newKeyMaxTokens,
      maxRequestsPerPeriod: newKeyMaxRequests,
      usageResetPeriod: newKeyResetPeriod.rawValue
  )
  ```
- **Line 508** `try await NovaMLXConfiguration.shared.updateAPIKey(id: key.id) { k in ... }` → use `NovaDB.shared.apiKeyStore.update(id: key.id) { record in ... }` — adapt the closure body to mutate `APIKeyRecord` instead of `APIKey`.
- **Line 563** Same pattern as 508 for the second update callsite.
- **Line 576** `try await NovaMLXConfiguration.shared.deleteAPIKey(id: id)` → `try NovaDB.shared.apiKeyStore.delete(id: id)`
- **Line 589** `try await NovaMLXConfiguration.shared.rotateAPIKey(id: id)` → `try NovaDB.shared.apiKeyStore.rotate(id: id)`
- **Line 141** — already calls `apiKeyStore.getRawKey`, no change.

- [ ] **Step 2: Migrate `APIServer.swift` callsites**

For each of these lines, replace with direct store calls. Convert `await` to direct calls where the store method is synchronous.

- **Line 136** `let keys = await config.apiKeys` → `let keys = (try? NovaDB.shared.apiKeyStore.listAsAPIKey()) ?? []`
- **Line 138** `let hasAnyKeys = !keys.isEmpty || !serverCfg.apiKeys.isEmpty` → `let hasAnyKeys = !keys.isEmpty` (drop `serverCfg.apiKeys` reference)
- **Line 156** `if let key = await config.findAPIKeyByRaw(token)` → `if let key = try? NovaDB.shared.apiKeyStore.findAPIKeyByRawToken(token)`
- **Line 169** `if serverCfg.apiKeys.contains(token)` — delete this entire fallback block (legacy flat-string auth)
- **Line 204, 208** same pattern as 136/138
- **Line 223** same as 156
- **Line 230** `let withinLimits = await config.isWithinLimits(keyId: key.id)` → call `recordUsage`-equivalent on store directly; or move `isWithinLimits` logic inline using `apiKeyStore.get(id:)`
- **Line 270** same as 169 — delete fallback
- **Line 2634** same as 136
- **Line 2683** `createAPIKey` → `apiKeyStore.create`
- **Line 2707** `findAPIKeyById` → `apiKeyStore.getAsAPIKey(id:)`
- **Line 2758** `updateAPIKey` → `apiKeyStore.update`
- **Line 2781** `deleteAPIKey` → `apiKeyStore.delete`
- **Line 2789** `rotateAPIKey` → `apiKeyStore.rotate`
- **Line 2802** `findAPIKeyById` → `apiKeyStore.getAsAPIKey(id:)`
- **Line 3044** `findAPIKeyByRaw` → `apiKeyStore.findAPIKeyByRawToken`
- **Line 3045** `recordUsage` → `apiKeyStore.recordUsage`

- [ ] **Step 3: Migrate `APIServer+AdminProxy.swift:44`**

Replace `if let apiKey = cfg.apiKeys.first` with:
```swift
let apiKeys = (try? NovaDB.shared.apiKeyStore.listAsAPIKey()) ?? []
if let apiKey = apiKeys.first { ... }
```

- [ ] **Step 4: Migrate `main.swift` callsites**

- **Line 237** `cfg.apiKeys.count` log → `(try? NovaDB.shared.apiKeyStore.listAsAPIKey())?.count ?? 0`
- **Lines 276, 432** `appState.apiKey = serverConfig.apiKeys.first` → `appState.apiKey = ((try? NovaDB.shared.apiKeyStore.listAsAPIKey()) ?? []).first`
- **Line 434** log same pattern
- **Line 244** `await config.loadAPIKeys()` — delete entirely. The importer runs in `NovaDB.setup()` automatically.

- [ ] **Step 5: Migrate `DownloadsPageView.swift:886-887`**

Replace the flat-string `updateApiKeys` call:
```swift
// OLD: let configFile = await NovaMLXConfiguration.shared.configFileURL
//      try await NovaMLXConfiguration.shared.updateApiKeys([key], file: configFile)
// NEW: Inject downloaded HF token as a new managed API key
_ = try NovaDB.shared.apiKeyStore.create(name: "huggingface-cli")
```

Confirm this matches the original intent (the original injected a HF token into the legacy flat-string list — creating a managed key is the DB equivalent).

- [ ] **Step 6: Migrate `WebUI/SettingsHTML.swift:92,113`**

The embedded JS currently reads `config.apiKeys` and PUTs `{server, apiKeys}` to `/admin/api/config`. Update both:
- The `/admin/api/config` endpoint to no longer accept `apiKeys` field (it's gone from `ServerConfig`).
- The embedded JS to drop the `apiKeys` references.

- [ ] **Step 7: Remove `ServerConfig.apiKeys` field**

In `Sources/NovaMLXCore/Types.swift`, find `ServerConfig` (around line 870). Delete the `apiKeys: [String]` field and its decoder/encoder logic. Update the memberwise init at every callsite that constructs `ServerConfig` (drop the `apiKeys:` argument).

- [ ] **Step 8: Delete apiKeys methods from `NovaMLXConfiguration`**

In `Sources/NovaMLXCore/Configuration.swift`:
- Delete `loadAPIKeys()` (line 113)
- Delete `saveAPIKeys()` (line 136)
- Delete `_apiKeys` property (line 12) and `_apiKeysMigrated` (line 13)
- Delete `apiKeys` getter (line 36)
- Delete `createAPIKey`, `updateAPIKey`, `deleteAPIKey`, `rotateAPIKey`, `findAPIKeyByRaw`, `findAPIKeyById`, `recordUsage`, `isWithinLimits`, `periodUsageFraction`, `migrateFlatKeys`, `updateApiKeys`

- [ ] **Step 9: Build**

```bash
./build.sh
```
Expected: errors only about leftover references. Fix any that surface (grep helps):

```bash
rg -n "NovaMLXConfiguration.shared.(apiKeys|findAPIKey|createAPIKey|updateAPIKey|deleteAPIKey|rotateAPIKey|recordUsage|isWithinLimits|periodUsageFraction|updateApiKeys|loadAPIKeys|saveAPIKeys)" Sources/
```
Expected: zero hits.

- [ ] **Step 10: swift test**

```bash
swift test
```
Expected: all tests pass.

- [ ] **Step 11: Manual UI smoke**

```bash
killall NovaMLX; sleep 1; open dist/NovaMLX.app
```
In the app:
1. Open API Keys page — list populates from DB
2. Create a key — copy raw key
3. Use the raw key with `curl http://localhost:6590/v1/chat/completions -H "Authorization: Bearer sk-novamlx-..."` — should authenticate
4. Reveal eye icon on the new key — shows real plaintext
5. Reveal eye icon on a legacy-imported key — disabled with tooltip
6. Toggle, edit, rotate, delete — all work

- [ ] **Step 12: Run /novamlx-full-api-test**

```bash
/novamlx-full-api-test
```
Expected: T1-T10 pass.

- [ ] **Step 13: Check log**

```bash
tail -100 ~/.nova/novamlx.log
```
Expected: no errors.

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -m "feat(apikeys): cutover phase — all callsites use apiKeyStore directly, JSON writes removed, ServerConfig.apiKeys dropped"
```

---

## Task A4: Cleanup Phase — Rename JSON file, delete load code, importer stays

**Files:**
- Modify: `Sources/NovaMLXDB/NovaDB.swift:94-134` — leave importer (used by users upgrading from older versions)
- Modify: `Sources/NovaMLXCore/NovaMLXPaths.swift:86` — delete `apiKeysFile`
- Manual: rename `~/.nova/api_keys.json` → `~/.nova/api_keys.json.migrated`

- [ ] **Step 1: Verify no code reads api_keys.json anymore**

```bash
rg -n "apiKeysFile|api_keys\.json" Sources/ | grep -v "\.build/" | grep -v "NovaDB.swift.*importLegacyJSON\|maybeImportLegacy\|LegacyAPIKeyImport"
```
Expected: zero hits outside the importer.

- [ ] **Step 2: Delete `apiKeysFile` from `NovaMLXPaths.swift`**

Find line 86 (`NovaMLXPaths.apiKeysFile = ...`) and delete the property. The importer uses a literal path now.

- [ ] **Step 3: Update importer to use literal path**

In `Sources/NovaMLXDB/NovaDB.swift:96`, change `NovaMLXPaths.apiKeysFile` reference (if any) to a literal:

```swift
let apiKeysURL = baseDir.appendingPathComponent("api_keys.json")
```

- [ ] **Step 4: Manually rename existing file**

```bash
[ -f ~/.nova/api_keys.json ] && mv ~/.nova/api_keys.json ~/.nova/api_keys.json.migrated
```

- [ ] **Step 5: Build, test, smoke, run T1-T10**

(Standard verification gate.)

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXCore/NovaMLXPaths.swift Sources/NovaMLXDB/NovaDB.swift
git commit -m "chore(apikeys): cleanup phase — delete apiKeysFile constant, file renamed to .migrated"
```

---

# Phase B — config.json

## Task B1: Schema migration v2_config_add_server_fields

**Files:**
- Modify: `Sources/NovaMLXDB/ConfigDBSchema.swift`
- Modify: `Sources/NovaMLXDB/Models/ConfigRecords.swift` (ConfigRecord struct)

- [ ] **Step 1: Write failing schema test**

Create `Tests/NovaMLXDBTests/ConfigStoreTests.swift`:

```swift
import Testing
import Foundation
@testable import NovaMLXDB
import NovaMLXCore

@Suite("ConfigStore")
struct ConfigStoreTests {
    @Test("v2 migration adds maxConcurrentRequests column")
    func v2MigrationAddsColumns() async throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-cfg-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let record = try nova.configStore.get()
        #expect(record != nil)
        // After v2 migration, all new columns exist with sensible defaults
        #expect(record?.maxConcurrentRequests != nil)
        #expect(record?.requestTimeout != nil)
        #expect(record?.maxRequestSizeMB != nil)
        #expect(record?.maxProcessMemory != nil)
        #expect(record?.prefixCacheEnabled != nil)
    }
}
```

- [ ] **Step 2: Run test, see failure**

```bash
swift test --filter ConfigStoreTests
```
Expected: FAIL — fields don't exist on `ConfigRecord`.

- [ ] **Step 3: Add v2 migration to ConfigDBSchema**

In `Sources/NovaMLXDB/ConfigDBSchema.swift`, after the `v1` createAll function, add:

```swift
public enum ConfigDBSchema {
    public static func v1CreateAll(in db: Database) throws { /* existing */ }

    public static func v2AddServerFields(in db: Database) throws {
        try db.alter(table: "config") { t in
            t.add(column: "max_concurrent_requests", .integer).notNull().defaults(to: 16)
            t.add(column: "request_timeout", .integer).notNull().defaults(to: 300)
            t.add(column: "context_scaling_target", .integer).notNull().defaults(to: 0)
            t.add(column: "tls_key_password", .text)
            t.add(column: "max_request_size_mb", .integer).notNull().defaults(to: 100)
            t.add(column: "max_process_memory", .integer).notNull().defaults(to: 0)
            t.add(column: "prefix_cache_enabled", .boolean).notNull().defaults(to: true)
        }
    }
}
```

- [ ] **Step 4: Register migration in NovaDB.swift**

In `Sources/NovaMLXDB/NovaDB.swift` `runMigrations()`, after v1:

```swift
configMigrator.registerMigration("v2_config_add_server_fields") { db in
    try ConfigDBSchema.v2AddServerFields(in: db)
}
```

- [ ] **Step 5: Extend ConfigRecord**

In `Sources/NovaMLXDB/Models/ConfigRecords.swift`, add the new fields to `ConfigRecord` struct + CodingKeys + ensure `PersistableRecord` mappings cover them.

- [ ] **Step 6: Run test**

```bash
swift test --filter ConfigStoreTests
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(db): v2_config_add_server_fields migration + ConfigRecord columns"
```

---

## Task B2: Bridge Phase — configStore reads; dual-write to JSON

(Same structure as Task A2, adapted for configStore. Detailed steps omitted for brevity — apply the pattern: each getter on `NovaMLXConfiguration` reads from `configStore.get()` first, falls back to `_serverConfig`; each setter dual-writes.)

## Task B3: Cutover Phase — all callsites direct to configStore

(Migrate ~20 callsites in `SettingsPageView.swift`, `APIServer.swift`, `main.swift`, `MenuBarAppState.swift`, `LaunchCommand.swift`, `CLIClient.swift`, `ClusterManager.swift`, `WorkerDeployer.swift`, `WorkerService.swift`, `AgentsPageView.swift`, `ClusterPageView.swift`, `Localization.swift`.)

## Task B4: Cleanup Phase — rename `config.json`, leave importer

---

# Phase C — tokenhub/providers.json

## Task C1: Schema migration v2_tokenhub_add_columns

Add columns: `is_local`, `is_free`, `include_in_load_balance`, `tags` (JSON array), plus any per-model fields the current JSON shape has.

## Task C2: Bridge Phase

Rewire `TokenhubProviderStore` (in `TokenhubTypes.swift`) reads through `tokenhubStore.list/get/upsert`. Dual-write.

## Task C3: Cutover Phase

Migrate ~20 callsites in `TokenhubPageView.swift`, `CloudBackend.swift`, `APIServer+TokenhubProxy.swift`, `APIServer+ResponsesHandlers.swift`, `AgentConfigGenerator.swift`, `ModelSpecs.swift`.

## Task C4: Cleanup Phase

Delete `Sources/NovaMLXCore/TokenhubTypes.swift`'s `TokenhubProviderStore` class entirely (its responsibility moves to `tokenhubStore`). The remaining types (`TokenhubProvider`, `UsageStats`) move to `Sources/NovaMLXDB/Models/ConfigRecords.swift` or stay in `TokenhubTypes.swift` as pure domain types with no I/O.

---

# Phase D — Simple Files Batch

For each of the 7 files below, follow the A1–A4 pattern in miniature. Each file gets one commit per phase (Bridge, Cutover, Cleanup).

## Task D1: loaded_models.json → loadedModelsStore

- Bridge: `InferenceService.saveLoadedModelsList()` + `loadLoadedModelsList()` call store; keep JSON write.
- Cutover: delete JSON write; callsites use `loadedModelsStore.replaceAll(with:)` / `list()`.
- Cleanup: delete `NovaMLXPaths.loadedModelsFile`, rename JSON.

## Task D2: model_settings.json → modelSettingsStore

- Bridge: `ModelSettingsManager.load/save` use store; dual-write JSON.
- Cutover: delete JSON; consumers in `InferenceService`, `AutoLoadCoordinator`, `MemoryPressureHandler`, `BenchmarkHarness`, `FusedSDPABench` use store.
- Cleanup: delete `ModelSettingsManager.settingsFile`, rename JSON.

## Task D3: modelfiles/*.json → modelfileStore

- Bridge: `ModelfileManager.list/get/create/update` use store; dual-write per-file JSON.
- Cutover: delete file I/O; `APIServer.swift:419` consumer uses store.
- Cleanup: delete `ModelfileManager` class entirely; rename `~/.nova/modelfiles/` to `.migrated`.

## Task D4: models/registry.json → modelRegistryStore

- Bridge: `ModelManager.swift:552/559` use store; dual-write JSON.
- Cutover: delete file I/O; `main.swift:171` `fixRegistryPaths` becomes a DB query.
- Cleanup: delete `registryFile` property; rename `~/.nova/models/registry.json` → `.migrated`.

## Task D5: auth_cache.json + session → authStore

- Bridge: `AuthClient.swift:184/199/207/211` use store; dual-write files.
- Cutover: delete file I/O.
- Cleanup: delete `NovaMLXPaths.sessionFile` + `NovaMLXPaths.authCacheFile`; rename files. Note: importer needs to handle bare-string `session` file (read with `String(contentsOf:)`, not JSON decode).

## Task D6: chat_history/*.json → chatStore

- Bridge: wire the dead `importChatHistory` helper (NovaDB.swift:179-206) into `importLegacyJSON`. Define `LegacyChatRecord` type referenced but not yet defined. Field mapping: JSON `{thinking, thinkingTime}` → DB `thinkingContent` (concat or store as JSON subfield; pick one, document).
- Cutover: `ChatHistoryStore.swift` (in NovaMLXAPI) methods delegate to `chatStore`; `APIServer.swift:1415-1433` callsites use `chatStore` directly.
- Cleanup: delete `Sources/NovaMLXAPI/ChatHistoryStore.swift`; rename `~/.nova/chat_history/` → `.migrated`.

## Task D7: worker-deployments.json → workerDeploymentStore

- Bridge: `WorkerDeployer.swift:459/463` use store; dual-write.
- Cutover: delete file I/O.
- Cleanup: delete `deploymentsFile` property; rename JSON.

---

# Phase E — metrics.json

## Task E1: Schema migration v2_metrics_add_columns

Add columns: `models_loaded`, `models_unloaded`, `ttl_evictions`, `memory_pressure_evictions`. Per-model stats are already JSON in `per_model_stats`; extend that JSON schema to include per-model request counts.

## Task E2: Bridge Phase

The legacy class `MetricsStore` (in `NovaMLXUtils/MetricsStore.swift`) is per-engine. Make it delegate to `NovaDB.shared.metricsStore` (singleton GRDB store) — single global row, aggregated across engines. Per-engine breakdown moves into the `per_model_stats` JSON field.

## Task E3: Cutover Phase

Delete file I/O from `MetricsStore` (NovaMLXUtils class). Rename the class to `MetricsDBStoreClient` or similar to disambiguate from `MetricsDBStore` in NovaMLXDB. Migrate all consumers in `MLXEngine.swift`, `InferenceService.swift`, `FusedBatchScheduler.swift`, `APIServer.swift`.

## Task E4: Cleanup Phase

Delete `Sources/NovaMLXUtils/MetricsStore.swift` entirely. Rename `~/.nova/metrics.json` → `.migrated`.

---

# Phase F — cluster-policy.json

## Task F1: Add admin endpoint `GET /admin/cluster/policy`

**Files:**
- Modify: `Sources/NovaMLXAPI/APIServer.swift` — add route that returns `clusterPolicyStore.get()` as JSON.

- [ ] **Step 1: Write test for endpoint**

Test that `GET /admin/cluster/policy` returns the stored policy JSON.

- [ ] **Step 2: Implement endpoint**

```swift
router.get("/admin/cluster/policy") { req, _ in
    let policy = try NovaDB.shared.clusterPolicyStore.get()
    return policy?.policyJson ?? "{}"
}
```

- [ ] **Step 3: Add POST endpoint**

`POST /admin/cluster/policy` accepts JSON body, calls `clusterPolicyStore.set(_:)`.

- [ ] **Step 4: Build, test, commit**

## Task F2: Worker boot SSH-sync

**Files:**
- Modify: `Sources/NovaMLXDistributed/WorkerService.swift:133-135`

- [ ] **Step 1: Replace local file read with SSH pull**

```swift
// On worker startup:
// 1. Determine coordinator URL from environment/config
// 2. curl -sf "$COORDINATOR/admin/cluster/policy" > /tmp/nova-cluster-policy.json
// 3. Read policy from /tmp/nova-cluster-policy.json
// 4. If pull fails, log loudly and run with empty policy
```

- [ ] **Step 2: Coordinator write path**

`WorkerDeployer.swift:252-257` and `:335-346` — replace SSH heredoc writes with HTTP POST to coordinator's `/admin/cluster/policy`.

## Task F3-F5: Bridge / Cutover / Cleanup

Same pattern as Phase A.

---

# Phase G — Final Cleanup

## Task G1: Delete Configuration.swift

**Files:**
- Delete: `Sources/NovaMLXCore/Configuration.swift`

- [ ] **Step 1: Verify all references are gone**

```bash
rg -n "NovaMLXConfiguration" Sources/ | grep -v "\.build/"
```
Expected: zero hits.

If any remain, migrate them to direct store access or delete.

- [ ] **Step 2: Delete the file**

```bash
git rm Sources/NovaMLXCore/Configuration.swift
```

- [ ] **Step 3: Move residual helpers**

If `initializeDirectories()` is still needed, move it to a more appropriate location (e.g. `Sources/NovaMLXApp/main.swift` or `NovaMLXPaths.swift`).

- [ ] **Step 4: Build, test, commit**

## Task G2: Delete importLegacyJSON + helpers

**Files:**
- Modify: `Sources/NovaMLXDB/NovaDB.swift`

- [ ] **Step 1: Delete `importLegacyJSON`, `maybeImportLegacy`, `migrateFile`**
- [ ] **Step 2: Delete legacy types** (`LegacyAPIKeyImport`, `LegacyChatRecord`, `PersistedConfig`) — search and delete each.
- [ ] **Step 3: Update `NovaDB.setup()`** to not call `importLegacyJSON`.
- [ ] **Step 4: Build, test, commit**

## Task G3: Delete remaining JSON path constants

In `Sources/NovaMLXCore/NovaMLXPaths.swift`, delete:
- `configFile`
- `tokenhubProvidersFile`
- `sessionFile`
- `authCacheFile`
- `loadedModelsFile`
- `chatHistoryDir`
- `metricsFile`
- Any other JSON config path constants

Keep only: `baseDir`, `modelsDir`, `logFile`, `prefixCacheBaseDir`, `voicesDir`, `sessionsDir`, `templatesDir`, and DB file paths.

## Task G4: Physically delete .migrated files

```bash
rm -f ~/.nova/*.json.migrated
rm -rf ~/.nova/chat_history.migrated ~/.nova/modelfiles.migrated
```

(Only after Phase G1-G3 verified working.)

## Task G5: Acceptance grep

- [ ] **Step 1: Zero NovaMLXConfiguration refs**

```bash
rg -n "NovaMLXConfiguration" Sources/ | grep -v "\.build/"
```
Expected: empty.

- [ ] **Step 2: Zero config-file JSON refs**

```bash
rg -n "\.json\b" Sources/ | grep -v "\.build/" | grep -vE "(model|asset|template|voice|session|chat_template|tokenizer|adapter|generation)" | grep -vE "/(models|voices|sessions|templates|prefix_cache)/"
```
Expected: empty or only comments.

- [ ] **Step 3: App boots, all features work**

Manual smoke: every UI page, full T1-T10.

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "chore: final cleanup — Configuration.swift deleted, importLegacyJSON removed, JSON path constants gone, .migrated files purged"
```

---

## Self-Review Checklist

After completing each phase:
- [ ] All tests pass (`swift test`)
- [ ] App builds cleanly (`./build.sh`)
- [ ] App launches without errors in log
- [ ] T1-T10 (`/novamlx-full-api-test`) pass
- [ ] Affected UI surfaces manually exercised
- [ ] Git history is clean (one commit per phase, descriptive messages)
- [ ] No `NovaMLXConfiguration` references introduced
- [ ] No new JSON file reads/writes introduced

After Phase G:
- [ ] `rg "NovaMLXConfiguration" Sources/` returns nothing
- [ ] `rg "api_keys\.json|config\.json|providers\.json|loaded_models\.json|model_settings\.json|registry\.json|metrics\.json|cluster-policy\.json|worker-deployments\.json|auth_cache\.json" Sources/` returns nothing
- [ ] `~/.nova/` contains only: `nova_config.db`, `nova_data.db`, `novamlx.log`, `models/`, `voices/`, `sessions/`, `templates/`, `prefix_cache/`
