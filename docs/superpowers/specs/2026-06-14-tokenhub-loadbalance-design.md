# TokenHub + Multi-Load-Balancer — Design Spec

**Date:** 2026-06-14
**Status:** Approved (5/5 design sections user-confirmed)
**Owner:** lucasliu

## Context

NovaMLX's TokenHub currently conflates two concerns under a single provider list:

1. **Remote provider management** — HuggingFace-compatible remote endpoints (tknet.ai, OpenAI, Anthropic, etc.)
2. **Implicit single Load Balancer** — every provider has an `includeInLoadBalance: Bool` flag; the union of flagged providers forms one global LB with a fixed priority-tiered strategy (`local+free > local > free > paid`)

This single-LB design is hitting its limits:

- **No task-based routing.** Users cannot say "use this pool for coding, that pool for cheap chat" — every request hits the same global pool.
- **Local models pollute TokenHub.** `provisionLocalProviders()` wraps local models as virtual `is_managed=true` providers, mixing them with real remote providers in the same UI list and confusing the mental model.
- **Provider cards do too much.** Each provider card shows an LB toggle that conflates "is this a remote provider" with "should this be in some LB".

This spec defines a **clean separation**: TokenHub manages remote providers only, and a new first-class **Load Balancer** entity lets users define arbitrarily many named pools whose members can be local models, remote providers, or any mix.

## Goals

1. Rename left-menu "Models" → **"Local Inference"** (clarity: this is where you load/unload local MLX models).
2. **TokenHub = remote providers only.** Local inference models no longer appear in the TokenHub provider list.
3. TokenHub provider card exposes only **Enable / Free / RESPS** checkboxes. The LB toggle is removed.
4. New top-level **"Load Balancers"** page as a sibling of TokenHub in the sidebar. Users can create arbitrarily many LBs; each LB has its own members (local models and/or remote providers) and its own selection strategy.

## Non-Goals

- **Per-API-key LB binding** (considered, deferred — model-name routing is sufficient for v1).
- **Auto-routing by request inspection** (considered, rejected — too opaque).
- **Health-check pingers** (no proactive background polling; member health is updated only by request outcomes).
- **LB audit logs** (a v2 concern).
- **Cross-vocabulary LB** (members must share the same tokenizer / chat template — operator responsibility, not enforced).
- **Weighted-RBAC / multi-tenant** (single-user app).

## Architecture Decision: Separate Entity

**Pattern:** Load Balancer is a first-class entity with its own SQLite tables, NOT a special TokenhubProvider kind.

- New tables: `load_balancers`, `lb_members`, `lb_member_stats`.
- Existing `tokenhub_providers` table: DROP `includeInLoadBalance` and `is_managed` columns; DELETE rows where `is_managed = 1` during migration.
- Locals are NOT wrapped as providers. They remain in `ModelManager`; LB references them by `model_id`.

**Why not composite-provider** (a `kind = "lb"` row inside `tokenhub_providers` with a `member_ids` JSON column): would require wrapping locals as virtual providers to be referenceable, which directly reverses goal #2.

## Request Routing

The `model` field of every API request (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`, etc.) is parsed by **prefix**:

| Prefix | Example | Routing |
|---|---|---|
| `lb:<slug>` | `lb:coding-pool` | Look up LoadBalancer by slug; apply selection strategy among healthy members; auto-retry on member failure up to `maxRetries`. |
| `tknet:<model>` | `tknet:deepseek-v4-pro` | Direct lookup of remote provider where `remoteModel == "deepseek-v4-pro" && isEnabled`. No LB layer, no failover. |
| `<bare>` | `gemma4-12b` | Local inference only. Requires model loaded in `MLXEngine`; otherwise 404. |

**Routing lifecycle for `lb:<slug>`:**

```
1. Parse slug from model field
2. Fetch LoadBalancer row by slug → 404 if missing
3. Fetch enabled members for LB
4. Partition members: local vs remote
5. For each local member: check MLXEngine.isLoaded(model_id)
     - drop unloaded locals from candidate pool
6. Apply strategy → ordered candidate list (see Strategies below)
7. Iterate candidates up to maxRetries (default 3):
     - send request to candidate
     - on success: update member stats (request_count, success_count, latency), return response
     - on failure (timeout, 5xx, network): update member stats (failure_count, count_5xx, last_error), try next
8. If all candidates exhausted: return 502 with "all members failed" detail
```

**Streaming caveat:** Failover applies only **before the first byte** is sent to the client. Once streaming has begun, mid-stream errors propagate directly to the client — we cannot retry after the client has already received partial output. Stats still record the failure.

## Selection Strategies

Each LB declares a `strategy` enum value applied at step 6 above:

| Strategy | Behavior | Member data used |
|---|---|---|
| `tiered` (default) | Priority tiers: `local+free > local > free > paid`. Within a tier, members rotate round-robin. Preserves today's implicit behavior. | `MemberKind`, provider's `isFree` |
| `round_robin` | Equal rotation across all healthy members, regardless of locality or cost. | None |
| `weighted` | Members selected with probability proportional to `weight`. Locals and remotes treated symmetrically — operator expresses preference via weights. | `weight` |
| `lowest_latency` | Pick the member with the lowest `avg_latency_ms` over the last N successes (N = 20). Ties broken by `success_rate` desc, then by `last_used_at` asc. | `lb_member_stats` |
| `random` | Uniform random across healthy members. Useful for canary / experiment. | None |

`weight` is nullable on `lb_members`. Only `weighted` strategy reads it; others ignore it. If `weighted` is selected and any member has `weight IS NULL`, it's treated as `weight = 1`. A weight of `0` is invalid and rejected at write time (admin API returns 400). To exclude a member from routing without deleting it, use the per-member `is_enabled = 0` toggle.

**`lowest_latency` cold-start:** when a member has zero recorded successes (`success_count = 0`), it's treated as having `avg_latency_ms = 0` — i.e., preferred — so newly added members get an initial probe request before settling into their true latency tier. Once any success is recorded, the real average takes over.

## Local Member Semantics

When an LB contains a local member that is **not currently loaded**:

- The member is **skipped** for that request (no auto-load, no failure recorded — unloaded is treated as "not in the pool right now", not as "broken").
- The request proceeds to the next healthy member.
- If all locals are unloaded AND no remote members exist: return 503 "no healthy members in load balancer".
- If all locals are unloaded AND remote members exist: route to remote members normally.

**Operator contract:** the user is expected to load a model via Local Inference before adding it to an LB. The UI surfaces a warning when an LB's local member is not currently loaded ("⚠ deepseek-v4-flash is not loaded — requests will skip it until loaded").

## Data Model

### SQLite tables (in `nova_data.db`)

```sql
CREATE TABLE load_balancers (
    id            TEXT PRIMARY KEY,        -- UUID string
    name          TEXT NOT NULL,
    slug          TEXT NOT NULL UNIQUE,    -- ^[a-z0-9-]+$
    strategy      TEXT NOT NULL DEFAULT 'tiered',  -- tiered|round_robin|weighted|lowest_latency|random
    max_retries   INTEGER NOT NULL DEFAULT 3,
    is_enabled    INTEGER NOT NULL DEFAULT 1,
    request_count INTEGER NOT NULL DEFAULT 0,  -- per-LB counter
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE lb_members (
    id         TEXT PRIMARY KEY,           -- UUID string
    lb_id      TEXT NOT NULL,
    kind       TEXT NOT NULL,              -- local|remote
    ref        TEXT NOT NULL,              -- model_id if local | provider_id if remote
    weight     INTEGER,                    -- nullable; only used by weighted strategy
    is_enabled INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (lb_id) REFERENCES load_balancers(id) ON DELETE CASCADE
);
CREATE INDEX idx_lb_members_lb_id ON lb_members(lb_id);

CREATE TABLE lb_member_stats (
    member_id         TEXT PRIMARY KEY,
    request_count     INTEGER NOT NULL DEFAULT 0,
    success_count     INTEGER NOT NULL DEFAULT 0,
    failure_count     INTEGER NOT NULL DEFAULT 0,
    count_5xx         INTEGER NOT NULL DEFAULT 0,
    total_latency_ms  INTEGER NOT NULL DEFAULT 0,
    last_used_at      TEXT,
    last_error        TEXT,
    updated_at        TEXT NOT NULL,
    FOREIGN KEY (member_id) REFERENCES lb_members(id) ON DELETE CASCADE
);
```

### Derived metrics

- `avg_latency_ms = total_latency_ms / max(success_count, 1)`
- `success_rate   = success_count / max(request_count, 1)`

### Swift types (in `NovaMLXCore`)

```swift
public struct LoadBalancer: Codable, Sendable, Identifiable {
    public let id: UUID
    public var name: String
    public var slug: String
    public var strategy: LBStrategy
    public var maxRetries: Int
    public var isEnabled: Bool
    public var requestCount: Int
    public let createdAt: Date
    public var updatedAt: Date
}

public struct LBMember: Codable, Sendable, Identifiable {
    public let id: UUID
    public var lbId: UUID
    public var kind: MemberKind      // .local | .remote
    public var ref: String           // model_id or provider_id
    public var weight: Int?          // nil = 1 for weighted strategy
    public var isEnabled: Bool
}

public struct LBMemberStats: Codable, Sendable {
    public let memberId: UUID
    public var requestCount: Int
    public var successCount: Int
    public var failureCount: Int
    public var count5xx: Int
    public var totalLatencyMs: Int64
    public var lastUsedAt: Date?
    public var lastError: String?
    public var updatedAt: Date
}

public enum LBStrategy: String, Codable, Sendable, CaseIterable {
    case tiered          // default
    case roundRobin
    case weighted
    case lowestLatency
    case random
}

public enum MemberKind: String, Codable, Sendable {
    case local
    case remote
}
```

### Storage layer (in `NovaMLXDB`)

Three new stores on `NovaDB.shared`, mirroring existing `tokenhubStore` pattern:

- `loadBalancerStore: LoadBalancerStore` — CRUD on `load_balancers`
- `lbMemberStore: LBMemberStore` — CRUD on `lb_members` (supports `listByLB(lbId:)`)
- `lbMemberStatsStore: LBMemberStatsStore` — atomic increments: `recordRequest(memberId:succeeded:latencyMs:httpStatus:errorMessage:)`

### Validation rules

- `slug` must match `^[a-z0-9-]+$` and be unique across `load_balancers`.
- `ref` for `kind = .local` must exist in `ModelManager.downloadedModels()` at add time. If the model is later uninstalled: keep the member row, mark `is_enabled = 0`, surface in UI as "model removed".
- `ref` for `kind = .remote` must exist in `tokenhub_providers` at add time. If provider is deleted: keep the member row, mark `is_enabled = 0`, surface as "provider removed".

## UI Structure

### Sidebar (revised)

```
● Status
💻 Local Inference          ← renamed from "Models"
⭐ TokenHub                 ← unchanged entry; clicking shows providers page
⚖ Load Balancers           ← NEW sibling, placed directly below TokenHub
🎮 Playground
🌐 Cluster
🔑 API Keys
⚙  Settings
```

"TokenHub" and "Load Balancers" are **siblings**, not parent/child. Both are top-level menu items.

### TokenHub page (provider list, scoped)

- Show only `tokenhub_providers` rows (no `is_managed` filter needed after migration DELETEs them).
- Provider card exposes three checkboxes: **Enable**, **Free**, **RESPS**. The `includeInLoadBalance` checkbox is removed from the form and the data model.
- "+ Add Provider" flow unchanged.

### Load Balancers page (new)

Two views: list and edit.

**List view:**
- Header: "Load Balancers" + "+ New LB" button.
- One card per LB: name, slug badge (e.g., `lb:coding-pool`), member count, strategy badge, total request count, enabled dot.
- Disabled LBs render at 50% opacity.
- Click card → opens edit view.

**Edit view:**
- Top: form fields — Name, Slug (with live validation), Strategy dropdown, Max Retries.
- Middle: Members section with "+ Add member" button and one row per member:
  - Badge: `LOCAL` (green) or `REMOTE` (yellow)
  - Name: model_id or `tknet:remoteModel`
  - Status: "✓ loaded" for locals currently in MLXEngine / "120ms avg" for remotes (from `lb_member_stats`)
  - Weight input (visible only when `strategy = weighted`)
  - Enable dot (per-member)
  - Remove button (×)
- "+ Add member" opens a sheet with two tabs: **Local** (lists `ModelManager.downloadedModels()` not already members) and **Remote** (lists enabled `tokenhub_providers` not already members). Multi-select.
- Bottom: "Test" button — POSTs a sample request to `lb:<slug>` and shows a trace (which member was picked, latency, success/error).

## Admin REST API

All under `/admin/load-balancers`:

| Method | Path | Purpose |
|---|---|---|
| GET | `/admin/load-balancers` | List all LBs with aggregate stats |
| POST | `/admin/load-balancers` | Create `{name, slug, strategy, maxRetries}` |
| GET | `/admin/load-balancers/:id` | LB detail + members + per-member stats |
| PATCH | `/admin/load-balancers/:id` | Update name/slug/strategy/maxRetries/isEnabled |
| DELETE | `/admin/load-balancers/:id` | Delete (CASCADE members + stats) |
| POST | `/admin/load-balancers/:id/members` | Add member `{kind, ref, weight?}` |
| PATCH | `/admin/load-balancers/:id/members/:memberId` | Update weight/isEnabled |
| DELETE | `/admin/load-balancers/:id/members/:memberId` | Remove member (CASCADE stats) |
| POST | `/admin/load-balancers/:id/test` | Invoke `lb:<slug>` with sample payload, return trace |

Auth via existing admin API key middleware (same as other `/admin/*` routes).

## Migration Plan (clean break)

GRDB does not support `DROP COLUMN` directly on SQLite < 3.35. Even where supported, the project's existing migration pattern uses table rewrite. Steps:

1. **Add new tables** (additive migration, idempotent):
   - `CREATE TABLE IF NOT EXISTS load_balancers …`
   - `CREATE TABLE IF NOT EXISTS lb_members …`
   - `CREATE TABLE IF NOT EXISTS lb_member_stats …`
   - Index: `CREATE INDEX IF NOT EXISTS idx_lb_members_lb_id ON lb_members(lb_id);`

2. **Drop legacy local-virtual-provider rows:**
   ```sql
   DELETE FROM tokenhub_providers WHERE is_managed = 1;
   ```

3. **Drop legacy columns from `tokenhub_providers`:**
   - Use the SQLite table-rewrite pattern (the same one already used by `2026-06-13-sqlite-migration`):
     1. `CREATE TABLE tokenhub_providers_new ( … )` without `includeInLoadBalance` and `is_managed`.
     2. `INSERT INTO tokenhub_providers_new SELECT (mapped cols) FROM tokenhub_providers;`
     3. `DROP TABLE tokenhub_providers;`
     4. `ALTER TABLE tokenhub_providers_new RENAME TO tokenhub_providers;`
     5. Recreate indexes.

4. **Stop calling `provisionLocalProviders()`** in `TokenhubManager` init / reload. Local models no longer get inserted as provider rows.

5. **Behavior change:** API requests that previously relied on the implicit global LB (model was not `tknet:` and not a loaded local) now return 404. **Users must create LBs explicitly.** This is the "clean break" — no auto-Default-LB.

## Error Handling Matrix

| Scenario | HTTP Status | Response Detail |
|---|---|---|
| `lb:<unknown-slug>` | 404 | `"Unknown load balancer: <slug>"` |
| `lb:foo` exists, all members disabled | 503 | `"Load balancer 'foo' has no enabled members"` |
| `lb:foo` exists, all locals unloaded + no remote members | 503 | `"No healthy members in load balancer 'foo'"` |
| `lb:foo` request, member errors mid-request (before first byte) | retries internally | Up to `maxRetries`; client sees only final outcome |
| `lb:foo` request, all members fail | 502 | `"All members of load balancer 'foo' failed; last error: <detail>"` |
| `lb:foo` streaming, error after first byte | propagate upstream error | No retry; client sees the error |
| `tknet:<unknown-model>` | 404 | `"Unknown remote model: <model>"` |
| `tknet:<model>` matched but provider disabled | 503 | `"Provider '<name>' is disabled"` |
| Bare model not loaded | 404 | `"Model '<name>' is not loaded"` |
| Admin API: slug validation fails | 400 | `"Slug must match ^[a-z0-9-]+$ and be unique"` |
| Admin API: ref doesn't exist at add time | 400 | `"Referenced <local_model|provider> does not exist: <ref>"` |

## Code Surface (touch points)

### New files

- `Sources/NovaMLXCore/LoadBalancerTypes.swift` — `LoadBalancer`, `LBMember`, `LBMemberStats`, `LBStrategy`, `MemberKind`
- `Sources/NovaMLXDB/LoadBalancerStore.swift` — `LoadBalancerStore`, `LBMemberStore`, `LBMemberStatsStore`
- `Sources/NovaMLXDB/Migrations/2026-06-14-load-balancers.swift` — table creation + provider cleanup
- `Sources/NovaMLXCore/LBRouter.swift` — pure function `(LB, [Member], MLXEngine, StatsStore) -> Member?` per strategy
- `Sources/NovaMLXMenuBar/LoadBalancersPageView.swift` — list + edit SwiftUI views
- `Sources/NovaMLXMenuBar/LBMemberPickerSheet.swift` — add-member sheet
- `Sources/NovaMLXAPI/APIServer+LoadBalancerAdmin.swift` — admin REST endpoints
- `Sources/NovaMLXAPI/LBProxy.swift` — request-proxy layer that orchestrates member selection + retry

### Modified files

- `Sources/NovaMLXMenuBar/NovaAppView.swift` — rename "Models" → "Local Inference"; add "Load Balancers" menu item as sibling of "TokenHub"
- `Sources/NovaMLXMenuBar/ModelsPageView.swift` — update page title to "Local Inference" (optional; could keep internal name as "Models")
- `Sources/NovaMLXCore/TokenhubTypes.swift` — remove `includeInLoadBalance` and `isManaged` fields from `TokenhubProvider`
- `Sources/NovaMLXCore/TokenhubStore+Domain.swift` — delete `provisionLocalProviders()`; remove priority-tiered `resolve()` (LB router replaces it)
- `Sources/NovaMLXMenuBar/TokenhubPageView.swift` — remove `formIncludeInLB` form binding (line ~995); remove `lbProviders` filter (line ~406); remove the "Load Balance" checkbox from provider card
- `Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift` — change model-prefix dispatch: `lb:` → LBProxy, `tknet:` → existing provider proxy, bare → existing local inference

## Testing

Unit tests in `Tests/NovaMLXCoreTests/`:

- `LBRouterTests.swift` — each strategy (tiered / round_robin / weighted / lowest_latency / random) under: all-locals-loaded, all-locals-unloaded, mixed, no-members, all-disabled, single-member.
- `LoadBalancerStoreTests.swift` — CRUD on all three tables; cascade delete; slug uniqueness; migration idempotency.

Integration tests in `Tests/NovaMLXAPIIntegrationTests/` (new dir if needed):

- POST `/v1/chat/completions` with `model=lb:foo`:
  - happy path (member returns 200)
  - first member times out, second succeeds (retry visible in stats)
  - all members fail → 502
  - streaming: first member errors mid-stream → client sees error, no retry, stats reflect failure
  - unknown slug → 404
  - all locals unloaded + no remotes → 503

## Rollout

Single PR. No feature flag — clean break is the design.

After merge:
- Users who upgrade will find their TokenHub provider list unchanged (locals already there get DELETED — but they were virtual; the actual local models in `~/.nova/models/` are untouched).
- Users who relied on the implicit LB will need to create explicit LBs to restore multi-provider routing. This is documented in the release notes.
