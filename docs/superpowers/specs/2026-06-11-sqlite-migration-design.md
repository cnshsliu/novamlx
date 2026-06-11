# SQLite Migration Design

**Date:** 2026-06-11
**Status:** Draft

## Problem

NovaMLX stores all persistent state as 20+ individual JSON/text files in `~/.nova/`. This creates:
- **No atomicity** — crash mid-write corrupts data (loaded_models.json wipe bug)
- **No queryability** — searching chat history scans N files, metrics aggregation requires full deserialization
- **No concurrency safety** — multiple writers race on JSON files
- **No referential integrity** — API key usage tracking, model-to-settings mapping, all ad-hoc
- **API key plaintext not stored** — users can't view full key after creation

## Decision: Store API key plaintext

The raw key will be stored in the `api_keys` table alongside the hash. Justification:
- Local/personal/enterprise app, not cloud SaaS — threat model is local filesystem access
- Hash is used for API request authentication; plaintext is only for UI display
- Convenience beats security theater for this use case
- Future: optional column-level encryption if needed

## Architecture

### Two SQLite databases

**`~/.nova/nova_config.db`** — Config, credentials, settings (small, rarely changes):

| Table | Source file(s) | Primary key | Notes |
|---|---|---|---|
| `config` | config.json | id (singleton) | Server host/port/adminPort, TLS, default model, HF endpoint, auth URL, tknet key, cluster config, autoLoad |
| `api_keys` | api_keys.json | id | **Includes `raw_key` column for plaintext.** Plus hash, prefix, suffix, name, limits, usage JSON |
| `model_settings` | model_settings.json | model_id | Per-model overrides: alias, default, pinned, sampling params, TTL, context window, draft model |
| `modelfiles` | modelfiles/*.json | name | Base model, system prompt, parameters, tools |
| `tokenhub_providers` | tokenhub/providers.json | name | Endpoint, API key, remote model, load-balance, metrics, managed flags |
| `cluster_policy` | cluster-policy.json | id (singleton) | Thunderbolt subnet, cluster topology JSON |
| `auth_session` | session + auth_cache.json | id (singleton) | Session token + cached auth state |

**`~/.nova/nova_data.db`** — Runtime state, analytics, user data (frequent writes):

| Table | Source file(s) | Primary key | Notes |
|---|---|---|---|
| `model_registry` | models/registry.json | model_id | Family, type, source, URL, size, download date, version |
| `loaded_models` | loaded_models.json | model_id | Currently loaded model IDs (single-column table) |
| `metrics` | metrics.json | id (singleton) | Cumulative counters: total requests, tokens, inference time, per-model breakdown, cache stats |
| `worker_deployments` | worker-deployments.json | hostname | Deployment phase, username, version, timestamps |
| `chats` | chat_history/*.json | id | Chat metadata: title, model, system prompt, created/updated |
| `chat_messages` | (extracted from chat_history) | id, chat_id (FK) | Individual messages: role, content, thinking, timestamps. **FTS5 virtual table for search** |

### What stays as files

| Item | Path | Reason |
|---|---|---|
| Model weights | ~/.nova/models/ | Multi-GB binary, MLX loads directly |
| Prefix cache | ~/.nova/prefix_cache/ | Safetensors binary, block-level access |
| Voice profiles | ~/.nova/voices/ | Audio WAV + small JSON, infrequent access |
| Templates | ~/.nova/templates/ | Jinja text files, user-edited externally |
| Logs | ~/.nova/novamlx.log | Append-only text, rotated |
| SSH deploy keys | ~/.nova/deploy_key{,.pub} | SSH format, used by ssh-keygen/ssh |
| Path config | ~/.config/novamlx/path | Read before DB available |

### New module: `NovaMLXDB`

New Swift package target providing all database access:

```
Sources/NovaMLXDB/
├── NovaDB.swift              // Singleton: opens pools, runs migrations
├── Migrations/
│   ├── ConfigDBMigration.swift    // v1 schema for nova_config.db
│   └── DataDBMigration.swift      // v1 schema for nova_data.db
├── Stores/
│   ├── ConfigStore.swift          // config table CRUD
│   ├── APIKeyStore.swift          // api_keys table CRUD + plaintext access
│   ├── ModelSettingsStore.swift   // model_settings table
│   ├── ModelfileStore.swift       // modelfiles table
│   ├── TokenhubStore.swift        // tokenhub_providers table
│   ├── AuthStore.swift            // auth_session table
│   ├── ModelRegistryStore.swift   // model_registry table
│   ├── LoadedModelsStore.swift    // loaded_models table
│   ├── MetricsStore.swift         // metrics table (replaces MetricsStore.swift)
│   ├── WorkerDeploymentStore.swift // worker_deployments table
│   └── ChatStore.swift            // chats + chat_messages + FTS5
└── Models/
    └── (GRDB record types for each table)
```

### GRDB dependency

Add to Package.swift:
```swift
.package(url: "https://github.com/groue/GRDB.swift", from: "7.0.0")
```

GRDB 7.x supports Swift 6 concurrency, async/await, and migrations.

### Dependency graph changes

```
NovaMLXDB depends on: GRDB.swift, swift-log
NovaMLXCore depends on: NovaMLXDB (was: direct file I/O)
NovaMLXUtils depends on: NovaMLXDB (MetricsStore moves here)
NovaMLXModelManager depends on: NovaMLXDB
NovaMLXInference depends on: NovaMLXDB
NovaMLXAPI depends on: NovaMLXDB
NovaMLXDistributed depends on: NovaMLXDB
NovaMLXEngine depends on: NovaMLXDB
NovaMLXMenuBar depends on: NovaMLXDB
```

## Migration Strategy

### Auto-migration on startup

1. `NovaDB.setup()` called early in `main.swift`
2. Opens/creates both `.db` files
3. Runs GRDB migrations (idempotent — GRDB tracks migration versions)
4. For each table, checks if the old JSON file exists:
   - If JSON exists AND table is empty → import data, rename file to `.json.migrated`
   - If JSON exists AND table has data → skip (already migrated), rename file to `.json.migrated`
   - If JSON doesn't exist → normal, fresh install
5. Log each migration step

### Migration order (dependencies matter)

1. `config` (singleton) — must exist first, other stores reference port/host
2. `auth_session` — depends on config for auth URL
3. `api_keys` — standalone
4. `model_settings` — references model IDs
5. `model_registry` — references model IDs
6. `modelfiles` — references model IDs
7. `tokenhub_providers` — references model IDs
8. `loaded_models` — references model IDs from registry
9. `metrics` — standalone
10. `chats` + `chat_messages` — bulk import from directory
11. `worker_deployments` — standalone
12. `cluster_policy` — standalone

## Schema Details

### api_keys table

```sql
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    raw_key TEXT NOT NULL,          -- plaintext for UI display
    key_prefix TEXT NOT NULL,
    key_suffix TEXT NOT NULL DEFAULT '',
    created_at DATETIME NOT NULL,
    expires_at DATETIME,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    rate_limit_per_second REAL,
    rate_limit_burst INTEGER,
    allowed_models TEXT,             -- JSON array
    allowed_endpoints TEXT,          -- JSON array
    max_tokens_per_period INTEGER,
    max_requests_per_period INTEGER,
    usage_reset_period TEXT NOT NULL DEFAULT 'daily',
    total_tokens_used INTEGER NOT NULL DEFAULT 0,
    total_requests INTEGER NOT NULL DEFAULT 0,
    last_used_at DATETIME,
    period_tokens INTEGER NOT NULL DEFAULT 0,
    period_requests INTEGER NOT NULL DEFAULT 0,
    period_reset_date TEXT,
    per_model_tokens TEXT DEFAULT '{}'  -- JSON object {model: tokens}
);
```

### chats table

```sql
CREATE TABLE chats (
    id TEXT PRIMARY KEY,
    title TEXT,
    model TEXT NOT NULL,
    system_prompt TEXT,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

CREATE TABLE chat_messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT,
    thinking_content TEXT,
    created_at DATETIME NOT NULL,
    sort_order INTEGER NOT NULL
);

CREATE VIRTUAL TABLE chat_messages_fts USING fts5(
    content,
    content=chat_messages,
    content_rowid=rowid
);
```

### metrics table

```sql
CREATE TABLE metrics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_inference_time_ms INTEGER NOT NULL DEFAULT 0,
    cache_hits INTEGER NOT NULL DEFAULT 0,
    cache_misses INTEGER NOT NULL DEFAULT 0,
    evictions INTEGER NOT NULL DEFAULT 0,
    per_model_stats TEXT DEFAULT '{}',    -- JSON {model: {requests, tokens, time}}
    per_model_cache TEXT DEFAULT '{}',     -- JSON {model: {hits, misses}}
    updated_at DATETIME
);
```

### config table

```sql
CREATE TABLE config (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    host TEXT NOT NULL DEFAULT '0.0.0.0',
    port INTEGER NOT NULL DEFAULT 6590,
    admin_port INTEGER NOT NULL DEFAULT 6591,
    tls_enabled INTEGER NOT NULL DEFAULT 0,
    tls_cert_path TEXT,
    tls_key_path TEXT,
    default_model TEXT,
    models_dir TEXT,
    hf_endpoint TEXT DEFAULT 'https://huggingface.co',
    auth_url TEXT,
    tknet_api_key TEXT,
    cluster_config TEXT DEFAULT '{}',    -- JSON
    auto_load TEXT DEFAULT '{}',         -- JSON
    log_level TEXT DEFAULT 'info'
);
```

### model_settings table

```sql
CREATE TABLE model_settings (
    model_id TEXT PRIMARY KEY,
    alias TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    is_pinned INTEGER NOT NULL DEFAULT 0,
    sampling_params TEXT DEFAULT '{}',   -- JSON {temperature, topP, topK, etc.}
    ttl_seconds INTEGER,
    context_window INTEGER,
    draft_model TEXT,
    updated_at DATETIME
);
```

### model_registry table

```sql
CREATE TABLE model_registry (
    model_id TEXT PRIMARY KEY,
    family TEXT,
    model_type TEXT,                -- llm, vlm, embed, audio, image
    source TEXT,                    -- huggingface, local, tokenhub
    local_path TEXT,
    remote_url TEXT,
    size_bytes INTEGER,
    downloaded_at DATETIME,
    version TEXT,
    architecture TEXT
);
```

### tokenhub_providers table

```sql
CREATE TABLE tokenhub_providers (
    name TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    api_key TEXT,
    remote_model TEXT,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    is_managed INTEGER NOT NULL DEFAULT 0,
    load_balance_weight REAL DEFAULT 1.0,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL,
    last_used_at DATETIME,
    extra_config TEXT DEFAULT '{}'
);
```

## API Surface: NovaDB

```swift
public final class NovaDB {
    public static let shared = NovaDB()

    public var configDB: DatabasePool    // nova_config.db
    public var dataDB: DatabasePool      // nova_data.db

    public func setup(baseDir: URL) throws
    // Opens pools, runs migrations, imports legacy JSON
}
```

Each store follows the same pattern:
```swift
public final class APIKeyStore {
    private let db: DatabasePool

    public func list() throws -> [APIKey]
    public func get(id: String) throws -> APIKey?
    public func create(_ key: APIKey) throws
    public func update(id: String, _ updates: (inout APIKey) -> Void) throws
    public func delete(id: String) throws
    public func findByHash(_ hash: String) throws -> APIKey?
    public func findByRawKey(_ raw: String) throws -> APIKey?
    public func recordUsage(keyId: String, tokens: Int64, model: String?) throws
}
```

## Rollout Plan

Phase 1: **Add NovaMLXDB module + migrate API keys only** (highest value: plaintext storage)
Phase 2: **Migrate config, model settings, model registry, tokenhub providers**
Phase 3: **Migrate chat history, metrics, loaded models, remaining tables**
Phase 4: **Remove old JSON file I/O code, clean up**

Each phase is independently deployable. Old code keeps working via JSON fallback until its phase completes.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| GRDB binary size increase | ~2MB added. Acceptable for desktop app |
| Migration corrupts data | Keep `.migrated` backup files. Log each step. Manual recovery path |
| Performance regression on hot paths | GRDB's DatabasePool uses WAL mode = concurrent reads. Benchmark metrics recording path |
| sqlite locking under high write load | WAL mode + PRAGMA synchronous=NORMAL. Metrics already debounced (every 10th event) |
| Multi-instance SQLite locking | Document: only one NovaMLX process per ~/.nova/ directory (already the case) |
