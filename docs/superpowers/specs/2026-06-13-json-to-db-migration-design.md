# JSON → SQLite Migration — Design Spec

**Date:** 2026-06-13
**Status:** Approved (user-confirmed approach + cleanup)
**Owner:** lucasliu

## Context

NovaMLX currently runs **two parallel storage systems** for configuration/state:

1. **JSON files** in `~/.nova/` — the actual source of truth today (`api_keys.json`, `config.json`, `tokenhub/providers.json`, etc.)
2. **SQLite (GRDB)** — `nova_config.db` + `nova_data.db` with 12 stores fully implemented (schema + CRUD + legacy import scaffolding) but **almost entirely dormant**

This duality has already caused real bugs: the API key "reveal" eye icon calls `apiKeyStore.getRawKey()` which queries a DB table that has no rows (UI creates keys via `NovaMLXConfiguration.createAPIKey()` which writes to `api_keys.json`, bypassing the store entirely). The eye silently fails.

**Goal:** Completely eliminate all JSON-based config/state storage and all code that processes those files. The final codebase must contain **zero** JSON config file code — no readers, no writers, no path constants, no facade methods whose names reference the legacy storage. Future developers reading the code should see one storage system, not two.

**Non-goals:**
- Model-asset JSON files inside `~/.nova/models/<id>/` (HF format) are out of scope — those are model files, not app config.
- `~/.nova/voices/<uuid>/profile.json` (user voice-cloning profiles) is out of scope — no GRDB store exists, treat as user content.
- `~/.nova/sessions/<id>.json` (KV-cache metadata) is out of scope — runtime performance cache, not config.
- `~/.nova/templates/registry.json` (user overrides of bundled chat-template fixes) — out of scope, no store.
- `~/.nova/prefix_cache/` — binary cache, out of scope.
- Log file `~/.nova/novamlx.log` — not config.

## Architecture Decision

**Pattern:** Direct store calls — no facade, no bridge layer in `NovaMLXConfiguration`.

- Final state: UI and API server code calls `NovaDB.shared.<store>.<method>()` directly.
- The existing `NovaMLXConfiguration` actor **gets deleted entirely** at the Final Cleanup phase. It currently holds only JSON-backed state (`_modelsDirectory`, `_serverConfig`, `_defaultModel`, `_huggingfaceEndpoint`, `_apiKeys`). After migration all of this lives in `configStore` / `apiKeyStore`; callers query the stores directly, no actor wrapper needed.
- During the per-file Bridge and Cutover phases, `NovaMLXConfiguration` survives temporarily as the facade while each subsystem is migrated in turn. It is removed only at Final Cleanup.
- Every method whose name reflects JSON-era semantics (`loadAPIKeys`, `saveAPIKeys`, `loadFromFile`, `saveToFile`, `findAPIKeyByRaw`, `recordUsage` on the actor, etc.) is removed when its host file reaches Phase 2 or 3.

**Domain types:** Keep current public types (`APIKey`, `TokenhubProvider`, `ModelSettings`, `Modelfile`, etc.) as the API surface that stores return. Stores internally use `*Record` types and convert at the boundary. This avoids forcing every callsite to learn a new type, while keeping persistence details inside `NovaMLXDB`.

## Strategy Per File

Each file follows the same three-phase pattern. Each phase is a discrete commit boundary and verification gate.

### Phase 1 — Bridge
- Store becomes writable. `importLegacyJSON` importer for this file is wired (if not already) and runs on next startup, copying JSON data into the SQLite table.
- JSON file is **still written** (dual-write) so any code still on the legacy path doesn't lose data.
- All read callsites for this file are switched to query the store instead of decoding JSON.
- After this phase: DB is source of truth for reads; JSON file receives writes but is read by no one (kept as safety net + rollback buffer).

### Phase 2 — Cutover
- Remove dual-write. All writers (UI save handlers, API endpoints, background savers) switch to store writes only.
- The JSON file is **no longer touched** by any code path.
- After this phase: DB is the sole source of truth for reads AND writes.

### Phase 3 — Cleanup (per file)
- Rename `~/.nova/<file>.json` to `~/.nova/<file>.json.migrated` (preserved as local safety net).
- Delete the legacy reader/writer functions from Swift sources.
- Delete the importer block for this file from `importLegacyJSON` (or leave the import hook in place until the **Final Cleanup** phase below, since users may upgrade from even older versions).

## Verification Gate (every phase)

1. `./build.sh` — must compile cleanly.
2. `swift test` — all existing unit tests pass.
3. Launch app, manually exercise affected UI surface (e.g. for api_keys: open API Keys page, create a key, reveal via eye, toggle, rotate, delete).
4. `/novamlx-full-api-test` T1–T10 — must all pass.
5. Read `~/.nova/novamlx.log` — no errors, no warnings related to the migrated subsystem.

Phase advance requires all 5 to pass.

## Migration Order (priority-driven)

Ordered by user-facing impact + lowest risk first. Each item is one full Bridge → Cutover → Cleanup cycle.

| # | File | Store | DB | Schema work needed? | Notes |
|---|------|-------|----|---------------------|-------|
| 1 | `~/.nova/api_keys.json` | `apiKeyStore` | configDB | None (already complete) | Fixes the eye-icon bug. **Pilot migration** — proves the pattern. |
| 2 | `~/.nova/config.json` | `configStore` | configDB | **v2_config_add_server_fields** — add `max_concurrent_requests`, `request_timeout`, `context_scaling_target`, `tls_key_password`, `max_request_size_mb`, `max_process_memory`, `prefix_cache_enabled`. Drop `apiKeys` column (flat-string array, dies with file). | Most readers of any file. Many callers across UI, API server, CLI, distributed. |
| 3 | `~/.nova/tokenhub/providers.json` | `tokenhubStore` | configDB | **v2_tokenhub_add_columns** — add `is_local`, `is_free`, `include_in_load_balance`, `tags`, per-model fields if any. | Heavy use across UI/cloud backend. |
| 4 | `~/.nova/loaded_models.json` | `loadedModelsStore` | dataDB | None | Simple `[String]`. |
| 5 | `~/.nova/model_settings.json` | `modelSettingsStore` | configDB | None | |
| 6 | `~/.nova/modelfiles/<name>.json` | `modelfileStore` | configDB | None | Per-file directory walk in importer. |
| 7 | `~/.nova/models/registry.json` | `modelRegistryStore` | dataDB | None | Note: file lives under `models/`, not `~/.nova/` directly. |
| 8 | `~/.nova/auth_cache.json` + `~/.nova/session` | `authStore` | configDB | None | `session` is plaintext (not JSON) — importer needs to handle bare string. |
| 9 | `~/.nova/chat_history/<id>.json` | `chatStore` | dataDB | Field mapping: JSON `{thinking, thinkingTime}` → DB `thinkingContent`; message order via `sort_order`. | `NovaDB.swift:179-206` has dead `importChatHistory` helper — wire it up. |
| 10 | `~/.nova/worker-deployments.json` | `workerDeploymentStore` | dataDB | None | |
| 11 | `~/.nova/metrics.json` | `metricsStore` (rename instance to disambiguate from `NovaMLXUtils.MetricsStore`) | dataDB | **v2_metrics_add_columns** — add `models_loaded`, `models_unloaded`, `ttl_evictions`, `memory_pressure_evictions`, split per-model stats properly. | Naming collision: `MetricsDBStore` (GRDB) vs `MetricsStore` (legacy JSON class in NovaMLXUtils). Legacy class gets deleted in Phase 3. |
| 12 | `~/.nova/cluster-policy.json` | `clusterPolicyStore` | configDB | None | **Special:** remote workers SSH-read this file. Migration adds an SSH-sync on worker startup: worker SSH-pulls the policy row from coordinator's `clusterPolicyStore`, writes it to local `/tmp/nova-cluster-policy.json`, and reads from there. Coordinator writes to DB only. |

## Per-File Migration Detail — File #1 (api_keys.json)

This is the pilot. The pattern established here repeats for the other 11.

### Phase 1 — Bridge

**Wiring changes:**
- `NovaDB.importLegacyJSON` already imports `api_keys.json` (NovaDB.swift:94-134) with a placeholder rawKey (`"sk-novamlx-" + 64 zeros`). The placeholder remains — see legacy-key handling below.
- `APIKeyStore.create()` already stores the real rawKey in the `raw_key` column.
- Add a `listAsAPIKey()` method to `APIKeyStore` that returns `[APIKey]` (converting from `APIKeyRecord`) so UI/API code can keep using the `APIKey` domain type.
- Add `findAPIKeyByRawToken(_ raw: String) -> APIKey?` (hashes input, calls existing `findByHash`, converts to `APIKey`).

**Readers switched (internal to facade):**
- `NovaMLXConfiguration.apiKeys` getter internally calls `NovaDB.shared.apiKeyStore.listAsAPIKey()`.
- `NovaMLXConfiguration.findAPIKeyByRaw(_:)` internally calls `NovaDB.shared.apiKeyStore.findAPIKeyByRawToken(_:)`.
- `NovaMLXConfiguration.findAPIKeyById(_:)` internally calls `NovaDB.shared.apiKeyStore.get(id:)` and converts to `APIKey`.
- `NovaMLXConfiguration.isWithinLimits(keyId:)` / `periodUsageFraction(keyId:)` query `apiKeyStore.get(id:)` and compute from record fields.
- External callsites still call `NovaMLXConfiguration.*` (unchanged) — this is the Bridge phase, the facade survives.

**Dual-write:**
- `createAPIKey`, `updateAPIKey`, `deleteAPIKey`, `rotateAPIKey`, `recordUsage` continue to write JSON (legacy). Additionally call the equivalent store method to write to DB.

**Legacy-key reveal handling:**
- `APIKey` gets a computed `isLegacyImport: Bool` (true when rawKey equals the placeholder-zero pattern, or when rawKey is nil).
- `APIKeysPageView.swift:141` eye button: `if key.isLegacyImport { disable button; tooltip = "Pre-DB key — rotate to enable reveal" } else { reveal as before }`.

### Phase 2 — Cutover

- Delete `saveAPIKeys()` from `Configuration.swift` — no more JSON writes.
- Rewrite `createAPIKey`, `updateAPIKey`, `deleteAPIKey`, `rotateAPIKey`, `recordUsage` in `NovaMLXConfiguration` to **only** call the store. Remove all JSON encoder/decoder code.
- Switch all 15+ external callsites to call `NovaDB.shared.apiKeyStore.*` directly. Then delete the now-unused methods from `NovaMLXConfiguration`.
- Remove the flat-string `ServerConfig.apiKeys: [String]` fallback array and all 8+ callsites that reference it (APIServer.swift:138,169,208,270; APIServer+AdminProxy.swift:44; main.swift:237,276,432,434; SettingsPageView.swift:745; WebUI/SettingsHTML.swift:92,113; Types.swift:870,891).
- Remove `_apiKeysMigrated` dead flag.
- Remove `migrateFlatKeys()`.

### Phase 3 — Cleanup
- Rename `~/.nova/api_keys.json` → `~/.nova/api_keys.json.migrated`.
- Delete `loadAPIKeys()` from `Configuration.swift`.
- Delete `apiKeysFile` constant from `NovaMLXPaths.swift`.
- Delete the `_apiKeys` array and all its references in `Configuration.swift`.
- Leave the import block in `NovaDB.importLegacyJSON` for now (so users upgrading from JSON-era still get migrated) — it gets removed in Final Cleanup.

## Final Cleanup (after all 12 files done)

After Phase 3 of file #12 completes:

1. **Physically delete** all `~/.nova/*.json.migrated` files.
2. **Delete `Sources/NovaMLXCore/Configuration.swift` entirely.** After all 12 migrations, no state remains that isn't in a store. Move any residual non-persistent runtime helpers (e.g. `initializeDirectories`) to a more appropriate location if still needed, then delete the file.
3. **Delete `NovaMLXPaths.apiKeysFile`**, `NovaMLXPaths.configFile`, `NovaMLXPaths.tokenhubProvidersFile`, `NovaMLXPaths.sessionFile`, `NovaMLXPaths.authCacheFile`, `NovaMLXPaths.loadedModelsFile`, `NovaMLXPaths.chatHistoryDir`, `NovaMLXPaths.metricsFile`, and any other path constants that point to JSON config files.
4. **Delete `importLegacyJSON`** and all its helpers from `NovaDB.swift`.
5. **Delete `migrateFile`** helper.
6. **Delete the legacy JSON-shape types** (`LegacyAPIKeyImport`, `LegacyChatRecord`, `PersistedConfig`, etc.) once importers are gone.
7. **Delete `Sources/NovaMLXCore/TokenhubTypes.swift`'s `TokenhubProviderStore`** legacy file-I/O methods (replaced by `tokenhubStore`).
8. **Delete `Sources/NovaMLXUtils/MetricsStore.swift`** (legacy JSON class).
9. **Delete `Sources/NovaMLXAPI/ChatHistoryStore.swift`** (replaced by `chatStore`).
10. **Delete `Sources/NovaMLXCore/ModelfileManager.swift`** (replaced by `modelfileStore`).
11. **Grep the entire `Sources/` for `.json"` path references and `File(contentsOf:)` / `data.write(to:)` patterns** to catch any remaining leaks. Review each one — keep only model-asset / user-content reads.

**Acceptance criteria:**
- `rg -n "NovaMLXConfiguration" Sources/ | grep -v "\.build/"` returns zero hits.
- `rg -n "\.json\b" Sources/ | grep -v "\.build/" | grep -vE "(model|asset|template|voice|session|chat_template|tokenizer|adapter|generation|config\.json|registry\.json)" | grep -vE "/(models|voices|sessions|templates|prefix_cache)/"` returns nothing.
- App boots clean, all stores have data, no errors in `~/.nova/novamlx.log`.

## Special-Case Handling

### Remote worker cluster-policy sync (file #12)

- Coordinator (where NovaMLX UI runs): stores policy in `clusterPolicyStore`. No more local `~/.nova/cluster-policy.json`.
- Worker boot sequence in `Sources/NovaMLXDistributed/WorkerService.swift`:
  1. On startup, SSH to coordinator, fetch policy JSON via a new admin endpoint `GET /admin/cluster/policy` (returns the row as JSON).
  2. Cache to `/tmp/nova-cluster-policy.json` (temp location, not user config).
  3. Read policy from cache for the worker's lifetime.
- `WorkerDeployer` SSH heredoc writes (`WorkerDeployer.swift:252-257, 335-346`) get replaced by HTTP POST to the admin endpoint, which writes to the coordinator's DB.
- Workers do **not** get a local SQLite copy of the policy — they only have the temp cache.

### MetricsStore naming collision (file #11)

- The GRDB class is `MetricsDBStore` in `Sources/NovaMLXDB/Stores/MetricsStore.swift`.
- The JSON class is `MetricsStore` in `Sources/NovaMLXUtils/MetricsStore.swift`.
- During Phase 2: rewrite `MetricsStore` (NovaMLXUtils) to delegate to `MetricsDBStore`. During Phase 3: delete the JSON I/O from NovaMLXUtils class. Either keep it as a thin wrapper or migrate all consumers directly to `NovaDB.shared.metricsStore` and delete the wrapper.
- Naming convention going forward: only `MetricsDBStore` exists; the legacy `MetricsStore` name is gone.

### api_keys legacy import rawKey

- Old keys imported from `api_keys.json` carry placeholder `rawKey = "sk-novamlx-" + 64 zeros` (because JSON didn't store plaintext).
- UI marks them as legacy and disables reveal. User can rotate to enable reveal.
- This is permanent — there is no way to recover plaintext that was never stored.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Phase 1 dual-write introduces inconsistency | All reads already go through store in Phase 1; mismatch would surface as UI showing stale data — caught at verification gate |
| Phase 2 cutover misses a callsite | `rg NovaMLXConfiguration.shared.apiKey` after Phase 2 must return zero hits before advancing |
| Schema v2 migration breaks existing installs | Run on app startup before any store access; migration is forward-only; backup `~/.nova/*.db` to `*.db.bak` before running |
| Worker nodes can't reach coordinator on startup | Worker falls back to "no policy, run unscheduled" mode with loud log warning; admin endpoint retry loop |
| Chat history field mapping (thinking vs thinkingContent) loses data | Importer preserves both fields; Phase 3 only deletes JSON after at least one successful launch with DB reads confirmed in logs |

## Test Plan

- **Unit tests:** Add tests for each store's CRUD operations + legacy importer. Update existing tests that touch `NovaMLXConfiguration` JSON methods.
- **Integration:** `/novamlx-full-api-test` T1–T10 after every phase. Add T11/T12 for cluster-policy worker boot sequence and metrics aggregation.
- **Manual smoke:** For each migrated file, the verification gate above (5 steps) must pass.

## Critical Files (will be modified across phases)

- `Sources/NovaMLXCore/Configuration.swift` — biggest deletion target
- `Sources/NovaMLXCore/NovaMLXPaths.swift` — path constants deleted
- `Sources/NovaMLXCore/Types.swift` — `APIKey`, `ServerConfig` types updated
- `Sources/NovaMLXCore/TokenhubTypes.swift` — `TokenhubProviderStore` rewired
- `Sources/NovaMLXCore/AuthClient.swift` — file I/O → `authStore`
- `Sources/NovaMLXCore/ModelfileManager.swift` — file I/O → `modelfileStore`
- `Sources/NovaMLXCore/Localization.swift` — language pref moves to `configStore`
- `Sources/NovaMLXDB/NovaDB.swift` — `importLegacyJSON` blocks added per-file, then entirely deleted at Final Cleanup
- `Sources/NovaMLXDB/Stores/*.swift` — receive new methods (e.g. `listAsAPIKey`, `findAPIKeyByRawToken`), schema v2 migrations
- `Sources/NovaMLXDB/Models/ConfigRecords.swift` — `ConfigRecord` extended with new fields
- `Sources/NovaMLXDB/Models/DataRecords.swift` — `MetricsRecord` extended
- `Sources/NovaMLXDB/ConfigDBSchema.swift` — v2 migration SQL
- `Sources/NovaMLXDB/DataDBSchema.swift` — v2 migration SQL
- `Sources/NovaMLXApp/main.swift` — startup calls (`loadAPIKeys`, `loadFromFile`) deleted/replaced
- `Sources/NovaMLXAPI/APIServer.swift` — 8 callsites for api_keys alone, plus auth, config, tokenhub, metrics endpoints
- `Sources/NovaMLXAPI/APIServer+AdminProxy.swift` — config reads
- `Sources/NovaMLXAPI/ChatHistoryStore.swift` — deleted, replaced by `chatStore`
- `Sources/NovaMLXAPI/WebUI/SettingsHTML.swift` — embedded JS updated
- `Sources/NovaMLXMenuBar/APIKeysPageView.swift` — UI rewired
- `Sources/NovaMLXMenuBar/SettingsPageView.swift` — config UI rewired
- `Sources/NovaMLXMenuBar/TokenhubPageView.swift` — ~20 tokenhub references rewired
- `Sources/NovaMLXMenuBar/DownloadsPageView.swift` — `updateApiKeys([key], file:)` flat-API call rewritten
- `Sources/NovaMLXMenuBar/ClusterPageView.swift` — cluster policy UI rewired
- `Sources/NovaMLXMenuBar/AgentsPageView.swift` — config/modelfile reads rewired
- `Sources/NovaMLXMenuBar/MenuBarAppState.swift` — config save/load rewired
- `Sources/NovaMLXModelManager/ModelManager.swift` — registry I/O → `modelRegistryStore`
- `Sources/NovaMLXModelManager/ModelSettingsManager.swift` — settings I/O → `modelSettingsStore`
- `Sources/NovaMLXInference/InferenceService.swift` — `loadedModelsStore`, `metricsStore`
- `Sources/NovaMLXUtils/MetricsStore.swift` — deleted or thinned
- `Sources/NovaMLXDistributed/WorkerDeployer.swift` — SSH heredoc → HTTP admin endpoint
- `Sources/NovaMLXDistributed/WorkerService.swift` — SSH pull on startup, cache locally
- `Sources/NovaMLXDistributed/ClusterManager.swift` — config reads → `configStore`
- `Sources/NovaMLXCLI/LaunchCommand.swift` — config bootstrap rewritten
- `Sources/NovaMLXCLI/CLIClient.swift` — config read rewritten

## Out of Scope (will not change)

- Model asset JSON inside `~/.nova/models/<id>/` (HF format files)
- `~/.nova/voices/<uuid>/profile.json` (user voice profiles)
- `~/.nova/sessions/<id>.json` (KV cache metadata)
- `~/.nova/templates/registry.json` (user template overrides)
- `~/.nova/prefix_cache/` (binary cache)
- `~/.nova/novamlx.log` (log file)
- Agent config outputs to `~/.codex/`, `~/.opencode/`, etc. (exported to external tools)
