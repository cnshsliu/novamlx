# Tknet Peer Gateway — Segment 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move NovaMLX’s load-balancer pools and Tokenhub providers into Peer config; make Peer the only multi-engine entry (`127.0.0.1:6600`). Strip NovaMLX of `lb:` routing, Tokenhub / `tknet:` proxying, and the three sidebar pages (Load Balancers, Tokenhub, Tknet Peer). NovaMLX keeps local inference, distributed sharding, and its own OpenAI API on `:6590`.

**Architecture:** One-shot **migrator** (pure Peer types + optional SQLite reader in the CLI) converts NovaDB LB + Tokenhub rows into `MemberConfig` / `PoolConfig` + `SecretStore` entries. NovaMLX API then returns **404** for `lb:*` and `tknet:*` (and bare `tknet`) with a message that those entries live on Peer. Sidebar pages and admin CRUD for LB/Tokenhub are removed or return Gone. Embedded `TknetPeerState` auto-tunnel in the NovaMLX app is stopped (tunnel ownership moves to `tknet-peer serve` until Segment 3’s menu-bar app).

**Tech Stack:** Swift 6 (StrictConcurrency), Foundation, NovaMLXDB/GRDB only on NovaMLX side (never in `NovaMLXTknetPeer`), Peer `ConfigStore` + `FileSecretStore`, ArgumentParser CLI, Swift Testing. Build via `./build.sh`. Peer product stays MLX-free: `swift build -c release --product tknet-peer`.

**Spec:** `docs/superpowers/specs/2026-10-11-tknet-peer-gateway-design.md`  
**Phase overview:** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway.md` (Segment 2 section — leave the four-segment outline intact)  
**Segment 1 plan (style + contracts):** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway-phase1.md`  
**Sources of truth for data shape:** `Sources/NovaMLXCore/LoadBalancerTypes.swift`, `Sources/NovaMLXCore/TokenhubTypes.swift`, `Sources/NovaMLXDB/Stores/LoadBalancerStore.swift`, `Sources/NovaMLXDB/Stores/TokenhubStore.swift`, `Sources/NovaMLXAPI/APIServer.swift` (`lb:` / Tokenhub branches), `Sources/NovaMLXMenuBar/NovaAppView.swift`

## Global Constraints

- `NovaMLXTknetPeer` / `tknet-peer` MUST NOT import: `MLX*`, `NovaMLXEngine`, `NovaMLXInference`, `NovaMLXImage`, `NovaMLXAudio`, `NovaMLXDB`, `NovaMLXCore`, `SwiftUI`/`AppKit`. Migration **core** is pure Codable DTOs inside Peer. Opening `nova_config.db` may use raw SQLite / a tiny reader in the **executable** only — do not link `NovaMLXDB` into the Peer library.
- Do **not** re-implement selection / retry in NovaMLX. After cutover, NovaMLX does not route pools.
- Secrets: Tokenhub `apiKey` values move into Peer `SecretStore` (`apiKeyRef` = `src/<memberId>`). Never log keys. Prefer not leaving usable plaintext copies in hot routing paths (DB rows may remain unread).
- Preserve NovaMLX: local model load/unload, `/v1/chat/completions` for **loaded local model ids**, distributed cluster/sharding admin, audio/image/video local features, API keys page, Requests log for local traffic.
- Segment 3 (menu-bar Peer app) and Segment 4 (repo split) are **out of scope**. Do not create a new Peer `.app` here.
- UI copy: a claimed Tknet Peer is a **「token 工厂」** / "token factory", never "我的机器" / "my machine".
- All new tests use Swift Testing. Prefer `./build.sh test --filter …`. Clean debug build: zero `warning:` from changed targets.
- Do not commit unrelated untracked local files (`mlx-swift-lm`, `vendors`, `speech.wav`, etc.).

### Key behavioural contracts (Segment 2)

| Case | Behaviour |
|---|---|
| Client → Peer `:6600`, pool name = old LB **slug** (no `lb:` prefix) | Completions match direct NovaMLX (or remote member) via Segment 1 gateway |
| Client → NovaMLX `:6590` with `model: "lb:<slug>"` | HTTP **404**, body explains use Peer `:6600` with model = `<slug>` |
| Client → NovaMLX with `model: "tknet:<provider>"` or bare `tknet` | HTTP **404**, same Peer pointer |
| Client → NovaMLX with a **loaded local model id** | Unchanged local inference |
| Sidebar | No Load Balancers, Tokenhub, or Tknet Peer pages |
| `local` LB members after migrate | HTTP members at `http://127.0.0.1:6590/v1`, `upstreamModel` = old model id, `type` = `.localNovaMLX` (triggers `X-No-Autoload`) |
| Remote Tokenhub members | HTTP members at normalized OpenAI-compatible base (`…/v1`), key in SecretStore, `upstreamModel` = `remoteModel`, `isFree` preserved |
| Pool id / name | Pool `name` (and preferably `id`) = LB **slug** so published names stay stable without the `lb:` prefix |
| Shared sources | Same local model id or same Tokenhub provider across LBs → **one** `MemberConfig`, referenced by multiple pools |

### Migration rules (Nova → Peer)

Input (from `nova_config.db` or fixture JSON):

1. Every **enabled or disabled** `TokenhubProvider` → one `MemberConfig`:
   - `id` = `th-<stable>` where stable is provider `id` (or slugified `name` if id empty)
   - `name` = provider `name`
   - `type` = `.openaiCompatible` (Anthropic-native Tokenhub path is **not** migrated as a Peer upstream — design: Peer only speaks OpenAI-compatible HTTP; skip or map only the OpenAI endpoint)
   - `endpoint` = normalize to a `/v1` base (strip trailing `/chat/completions` if pasted full URL; use same spirit as `EndpointNormalizer`)
   - `apiKeyRef` = `src/<id>`; save key if non-empty
   - `upstreamModel` = `remoteModel`
   - `isEnabled` / `isFree` / `weight` = from provider (`weight` nil)
2. Every **LoadBalancer** → one `PoolConfig`:
   - `id` = `slug` (if collision with an existing Peer pool id, suffix `-migrated` and log)
   - `name` = `slug` (clients drop `lb:` and use this name on `:6600`)
   - `strategy` / `maxRetries` / `isEnabled` copied (raw strategy strings already match `PoolStrategy`)
   - `memberIds` built from LB members in store order:
     - `kind == local` → member id `local-<modelId>` (create member if missing): endpoint `http://127.0.0.1:6590/v1`, `type` `.localNovaMLX`, `upstreamModel` = model id, empty api key ok
     - `kind == remote` → reuse `th-<provider>` member from step 1 (resolve `ref` as provider **name**, matching today’s `LBProxy` / picker). If provider missing, skip member and warn
3. Merge into existing `peer.json`:
   - Do **not** delete existing Peer members/pools/capabilities from Segment 1 / tknet sell list unless `--replace-pools` is set
   - Default: upsert by id; if pool id exists, leave Peer copy unless `--force`
   - Write secrets only for new/updated refs
4. Mark migration done: write `peer.json` field `migratedFromNovaAt` (ISO8601) or a sibling `migration-stamp.json` so re-runs are idempotent no-ops unless `--force`
5. NovaMLX DB rows may remain on disk but **must not** be consulted by API routing after cutover

---

### Task 1: Pure migrator types + fixture tests (Peer library, no NovaDB)

**Files:**
- Create: `Sources/NovaMLXTknetPeer/NovaConfigMigrator.swift`
- Create: `Tests/NovaMLXTknetPeerTests/NovaConfigMigratorTests.swift`
- Optional fixture: `Tests/NovaMLXTknetPeerTests/Fixtures/nova-lb-tokenhub-sample.json`

**Interfaces:**
- Snapshot DTOs (Codable, Peer-owned names — do not import NovaMLXCore types):
  - `NovaProviderSnapshot` (id, name, endpoint, apiKey, remoteModel, isEnabled, isFree)
  - `NovaLBSnapshot` (id UUID string, name, slug, strategy, maxRetries, isEnabled)
  - `NovaLBMemberSnapshot` (lbSlug or lbId, kind: `local`|`remote`, ref, weight, isEnabled)
- `NovaMigrationInput` + `NovaMigrationResult` (`members`, `pools`, `secrets: [String: String]`, `warnings: [String]`)
- `NovaConfigMigrator.migrate(_:existing:options:) -> NovaMigrationResult` merging into an existing `PeerConfig`

- [ ] **Step 1: Write failing tests**

```swift
@Test("local LB member becomes loopback NovaMLX HTTP member")
func localMemberMaps() {
    // one LB slug "coding", member kind local ref "Qwen3-8B"
    // expect member id "local-Qwen3-8B", endpoint http://127.0.0.1:6590/v1,
    // type localNovaMLX, pool name/id "coding", memberIds contain that id
}

@Test("remote LB member reuses Tokenhub provider member and secret")
func remoteReusesProvider() {
    // provider name "OpenAI", key "sk-test", remoteModel "gpt-4o"
    // LB member kind remote ref "OpenAI"
    // expect one th-* member, secrets[apiKeyRef] == "sk-test", pool lists it
}

@Test("same local model in two LBs shares one member id")
func sharedLocalMember() { /* … */ }

@Test("merge without --force leaves existing Peer pool untouched")
func mergeRespectsExisting() { /* … */ }
```

- [ ] **Step 2: Implement `NovaConfigMigrator`**

Follow migration rules above. Normalize endpoints:
- If path already ends with `/v1`, keep as base
- If path ends with `/v1/chat/completions`, strip to `/v1`
- If bare host, append `/v1`
- Local always `http://127.0.0.1:6590/v1` (override via options `localNovaBaseURL` for tests)

- [ ] **Step 3: `./build.sh test --filter NovaConfigMigratorTests`**

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXTknetPeer/NovaConfigMigrator.swift \
  Tests/NovaMLXTknetPeerTests/NovaConfigMigratorTests.swift
git commit -m "feat(peer): Nova LB/Tokenhub → Peer pool migrator (pure fixtures)"
```

---

### Task 2: CLI `tknet-peer migrate-from-novamlx`

**Files:**
- Modify: `Sources/tknet-peer/TknetPeerCLI.swift`
- Create: `Sources/tknet-peer/NovaDBSnapshotReader.swift` (executable-only; raw SQLite3 or `sqlite3` CLI — **not** NovaMLXDB)
- Modify: `Sources/NovaMLXTknetPeer/PeerConfig.swift` only if adding `migratedFromNovaAt: Date?`
- Test: unit-test the SQL→snapshot mapping with a temp SQLite file built in the test **or** keep reader thin and cover migrator in Task 1; add `CLIMigrateTests` for path expand + dry-run JSON if feasible without linking DB into the library

**Command:**

```text
tknet-peer migrate-from-novamlx \
  [--config ~/.config/tknet-peer/peer.json] \
  [--nova-db ~/Library/.../nova_config.db | auto-detect] \
  [--local-base http://127.0.0.1:6590/v1] \
  [--dry-run] [--force]
```

Auto-detect: read `~/.config/novamlx/path` first line as Nova base dir, then `nova_config.db` (same as `NovaDB`). If missing, print path hint and exit non-zero.

Behaviour:
1. Load Peer config (or default empty)
2. Read providers + load_balancers + lb_members tables (column names from `ConfigDBSchema` / store records — verify against live schema in `Sources/NovaMLXDB/Migrations/ConfigDBSchema.swift` and `LBMigration.swift` before writing SQL)
3. Run `NovaConfigMigrator`
4. Unless `--dry-run`: save secrets via `FileSecretStore`, save Peer config mode 0600
5. Print summary: `N members, M pools, K secrets, W warnings`

- [ ] **Step 1: Implement reader + command**

- [ ] **Step 2: Manual dry-run against Lucas’s machine DB if present** (document output; do not commit secrets)

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer): CLI migrate-from-novamlx reads nova_config.db into peer.json"
```

---

### Task 3: NovaMLX API — `lb:` and Tokenhub routes return 404

**Files:**
- Modify: `Sources/NovaMLXAPI/APIServer.swift` (chat completions, messages, responses entry points that branch on `isLBModel` / `isTokenhubModel`)
- Modify: `Sources/NovaMLXAPI/APIServer+LBProxy.swift` — keep `isLBModel` / `lbSlug` helpers; change callers to 404 helper
- Modify: `Sources/NovaMLXAPI/APIServer+TokenhubProxy.swift` — replace passthrough body with 404 (or delete handlers and inline)
- Modify: `Sources/NovaMLXAPI/APIServer+ResponsesHandlers.swift` — same for Responses Tokenhub path
- Modify / delete usage: `Sources/NovaMLXAPI/LBProxy.swift` (may remain unused; prefer delete if no references)
- Modify: `Sources/NovaMLXAPI/APIServer+TokenhubAnthropicBridge.swift` — unreachable after cutover; remove route wiring
- Tests: add `Tests/NovaMLXAPITests/GatewayCutoverTests.swift` (or extend existing API tests) asserting 404 JSON shape

**404 body (stable message for clients):**

```json
{
  "error": {
    "message": "Load-balanced and Tokenhub models moved to Tknet Peer. Point your client at http://127.0.0.1:6600/v1 and use the pool name without the 'lb:' or 'tknet:' prefix (run: tknet-peer migrate-from-novamlx).",
    "type": "invalid_request_error"
  }
}
```

Apply for: `lb:*`, `tknet`, `tknet:*`. Do **not** 404 ordinary local model ids.

- [ ] **Step 1: Failing API tests** — POST `/v1/chat/completions` with `model: "lb:coding"` → 404; `tknet:foo` → 404; a normal model id still hits local path (may 404 for unknown model — different message)

- [ ] **Step 2: Replace LB/Tokenhub dispatch branches with the shared 404 helper** in all three surfaces (OpenAI chat, Anthropic messages, Responses)

- [ ] **Step 3: Remove or stub `makeLBProxy` call sites** so LBProxy is dead code; delete if easy

- [ ] **Step 4: `./build.sh test --filter 'GatewayCutoverTests|APITests'`** (and any Tokenhub-specific tests updated to expect 404)

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(api): lb: and tknet: models return 404; use Peer :6600"
```

---

### Task 4: Remove LB / Tokenhub admin HTTP APIs

**Files:**
- Modify: `Sources/NovaMLXAPI/APIServer.swift` — `RouteGroup("admin/load-balancers")`, `RouteGroup("admin/tokenhub")`
- Modify or delete: `Sources/NovaMLXAPI/APIServer+LoadBalancerAdmin.swift`

**Behaviour:** Remove route groups **or** make every method return **410 Gone** with the same Peer migration message. Prefer **remove** routes (cleaner) and update any UI/tests that called them (Task 5 removes UI).

- [ ] **Step 1: Remove route registration + handler file if unused**

- [ ] **Step 2: Fix compile breakages**

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(api): remove load-balancer and tokenhub admin routes"
```

---

### Task 5: Remove three sidebar pages from NovaMLX

**Files:**
- Modify: `Sources/NovaMLXMenuBar/NovaAppView.swift` — drop `AppPage.tokenhub`, `.loadBalancers`, `.tknetPeer` from enum, icons, sidebar list, `detailView` switch, l10n titles
- Modify: `Sources/NovaMLXMenuBar/PageMemory.swift` — remove `tokenhub` / `loadBalancers` / `tknet` memories (or leave structs unused — prefer delete to avoid dead code)
- Delete or stop compiling: `TokenhubPageView.swift`, `LoadBalancersPageView.swift`, `TknetPeerPageView.swift`, `LBMemberPickerSheet.swift`
- Modify: `Sources/NovaMLXApp/main.swift` — stop constructing / injecting UI for those pages; **stop** `tknetPeer.restoreTunnelAfterLaunch()` so NovaMLX no longer owns the tunnel (Segment 1 CLI / Segment 3 app owns it)
- Modify: any `requestedPage` / deep-link that targeted those pages
- Localization: leave unused strings or delete keys in `LocalizationStrings.swift` only if safe (no drive-by mass edits)

**Keep for Segment 3 reuse (optional):** `TknetPeerState.swift` may remain in the target but unused, **or** move later. Prefer keep file, remove sidebar + auto-connect, to avoid a large behaviour rewrite in this segment.

- [ ] **Step 1: Remove enum cases and switch arms; fix exhaustiveness**

- [ ] **Step 2: Remove page files from the Xcode/SwiftPM target (delete files)**

- [ ] **Step 3: Disable Peer auto-connect in `main.swift`; drop `TknetPeerPageView` from window**

- [ ] **Step 4: `./build.sh -c debug` (app target compiles)**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(ui): remove Load Balancers, Tokenhub, and Tknet Peer sidebar pages"
```

---

### Task 6: Retarget in-app callers that used `lb:` / Tokenhub

**Files (known call sites — re-scan with rg before editing):**
- `Sources/NovaMLXMenuBar/VideoSlicePageView.swift` — creates/uses `lb:for-video-slice`
- `Sources/NovaMLXCore/LocalizationStrings.swift` — videoSlice LB strings
- `Sources/NovaMLXMenuBar/SettingsPageView.swift` — `provisionTknetProviders` / tknet API key → Tokenhub
- `Sources/NovaMLXApp/main.swift` — launch-time `provisionTknetProviders`
- `Sources/NovaMLXCore/AgentConfigGenerator.swift` — TokenHub load-balanced model blurb / `base_url` 6590
- `Sources/NovaMLXMenuBar/ModelsPageView.swift` — playground routing for `tknet:*` / `lb:*`

**Recommended behaviour:**
1. **Video Slice:** After migration, call `http://127.0.0.1:6600/v1/chat/completions` with `model: "for-video-slice"` (pool name = slug). If Peer unreachable or pool missing, show error directing user to run Peer + migrate (do **not** recreate NovaMLX LB rows).
2. **Settings tknet marketplace key:** Remove “provision into Tokenhub” flow from NovaMLX Settings **or** replace copy with “configure remote members in Peer”. Do not keep writing Tokenhub rows that nothing routes.
3. **AgentConfigGenerator:** Point multi-engine users at `http://127.0.0.1:6600/v1` and pool names; local-only snippets may stay on 6590.
4. **Playground:** Opening `lb:` / `tknet:` demos should either disappear with the pages or open a note string — no NovaMLX proxy.

- [ ] **Step 1: rg audit for `lb:`, `tknet:`, `TokenhubManager`, `LoadBalancer(`, `LoadBalancersPage`, `TknetPeerPage`**

- [ ] **Step 2: Patch Video Slice + Settings + generator + any leftovers**

- [ ] **Step 3: Build + relevant tests**

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(app): retarget video-slice and settings off NovaMLX LB/Tokenhub"
```

---

### Task 7: Optional cleanup — stop compiling unused LB/Tokenhub core paths

**Files (only if Task 3–6 left zero references):**
- Consider leaving `LoadBalancerTypes.swift`, stores, and `TokenhubManager` in tree for DB compatibility / Segment 3 data export — **do not** delete NovaDB tables in Segment 2
- Remove dead imports; fix `NovaMLXCoreTests/LBRouterTests` — either keep testing pure router (still valid reference for Peer) or skip if Core LB types remain
- `TokenhubStoreTests` / `LoadBalancerStoreTests` can stay as persistence tests (data still readable for migrate)

- [ ] **Step 1: `./build.sh -c debug` zero warnings from touched targets**

- [ ] **Step 2: Commit only if there are real deletions**

```bash
git commit -m "chore: drop dead LB/Tokenhub API wiring left after Peer cutover"
```

---

### Task 8: Segment 2 acceptance suite

**Files:**
- Create: `Tests/NovaMLXTknetPeerTests/NovaMigrationAcceptanceTests.swift` (fixture → Peer config → LocalGateway → MockSourceServer as Nova stand-in on loopback)
- Create or extend: `Tests/NovaMLXAPITests/GatewayCutoverTests.swift` (404 contracts)
- Manual script notes in commit body (not necessarily committed script): migrate → serve → curl Peer vs curl NovaMLX

**Acceptance (must pass):**

1. **Migrator:** Fixture with one local + one remote member → Peer pool; gateway failover/selection still works (Segment 1 suite green).
2. **Cutover:** NovaMLX API returns 404 for `lb:x` and `tknet:y`.
3. **Live Mac (manual / optional CI):** With NovaMLX running on 6590 and Peer `serve --no-tunnel` after migrate, `curl :6600` with pool name ≈ `curl :6590` with local model id (same completion path for a trivial prompt). Document if sandbox cannot start the full app.
4. **UI:** App builds; sidebar has no three pages (smoke by code assertion: `AppPage.allCases` does not contain removed raw values — add a tiny MenuBar test if practical, else checklist).
5. **Regression:** `./build.sh test --filter NovaMLXTknetPeerTests` green; Peer product MLX-free.

- [ ] **Step 1: Automated tests**

- [ ] **Step 2: Full Peer suite + cutover tests**

```bash
./build.sh test --filter 'NovaMLXTknetPeerTests|GatewayCutoverTests|NovaConfigMigratorTests'
swift build -c release --product tknet-peer
```

- [ ] **Step 3: Commit**

```bash
git commit -m "test: Segment 2 acceptance — migrate, Peer gateway, NovaMLX lb: 404"
```

---

### Task 9: Verification gate (no new features)

- [ ] **Step 1:** `./build.sh -c debug` — app + Peer compile; no new `warning:` from Segment 2 files
- [ ] **Step 2:** Peer tests + cutover tests green
- [ ] **Step 3:** `swift build -c release --product tknet-peer` — still no MLX/NovaMLXCore/NovaMLXDB in Peer graph
- [ ] **Step 4:** Manual on Lucas’s Mac (when available):
  1. Quit duplicate Peer processes
  2. `tknet-peer migrate-from-novamlx`
  3. Start NovaMLX (6590) with a loaded model
  4. `tknet-peer serve --no-tunnel`
  5. `curl` Peer pool vs direct NovaMLX model
  6. `curl` NovaMLX with `lb:<slug>` → 404
- [ ] **Step 5: Stop** — do not start Segment 3 menu-bar app in this plan

---

## Acceptance criteria mapping

| Design / overview criterion | Covered by |
|---|---|
| Write existing pools + Tokenhub providers into Peer config | Tasks 1–2 |
| `local` members → HTTP `127.0.0.1:6590` | Task 1 |
| Remove `lb:` routing + Tokenhub proxy | Tasks 3–4 |
| Remove Load Balancers, Tokenhub, Tknet Peer pages | Task 5 |
| NovaMLX keeps local inference / sharding / local API | Global Constraints; Task 3 only blocks lb/tknet prefixes |
| Client → Peer :6600 → running NovaMLX, matches direct | Tasks 1–2, 8–9 |
| Client → NovaMLX `lb:` → 404 | Tasks 3, 8 |
| Tunnel / Peer UI ownership leaves NovaMLX | Task 5 (auto-connect off); Segment 3 for menu-bar app |

## Out of scope (do not implement in Segment 2)

- Mac menu-bar Peer-only application (Segment 3)
- Extracting Peer to a separate git repo / Windows-Linux packaging (Segment 4)
- Active health probes, Anthropic-native Peer upstream, embeddings/image/audio on Peer gateway
- Re-registering peer identity; changing tknet website scheduling/billing
- Dropping NovaDB `load_balancers` / `tokenhub_providers` tables (keep for one-shot migrate / history)
- Implementing new pool CRUD UI inside NovaMLX

## Ambiguities / risks for Lucas

1. **Video Slice `lb:for-video-slice`.** After cutover NovaMLX cannot serve that model name. Plan retargets Video Slice to Peer `:6600` / pool `for-video-slice`. Confirm that running Peer alongside NovaMLX during slice jobs is acceptable (recommended).

2. **Settings “tknet.ai API key → provision Tokenhub providers”.** That path filled remote members for LB. After Tokenhub routing dies, provisioning into NovaDB is pointless. Plan removes or disables it in Segment 2; new remotes are `tknet-peer member add` until Segment 3 UI. Confirm OK to remove from Settings (the Peer **registration** token from `tknet-peer setup` is separate).

3. **NovaMLX-embedded tunnel.** Removing the Tknet Peer **page** without disabling `restoreTunnelAfterLaunch` would leave a hidden second Peer process fighting the CLI for the same identity. Plan disables auto-connect in Segment 2. Confirm you will use `tknet-peer serve` (or Segment 3 later) for the tunnel.

4. **Pool name vs `lb:` prefix.** Clients must switch base URL to `:6600` and drop the `lb:` / `tknet:` prefix. Migrator keeps **slug** as pool name. No automatic rewrite proxy on :6590 (explicit 404 per design).

5. **Anthropic-only Tokenhub providers.** Peer forwards OpenAI-compatible HTTP only. Migrator should warn and skip providers that only have `anthropicEndpoint` and no usable OpenAI endpoint — or map OpenAI endpoint when both exist. Confirm skip-with-warning.

6. **SQLite reader in `tknet-peer`.** To keep Peer MLX/DB-free, the executable uses raw SQLite. Schema drift is a risk — pin column names to current migrations and add a clear error if tables missing.

7. **Orphan DB data.** Leaving LB/Tokenhub rows in `nova_config.db` is intentional for migrate `--force` re-runs; they must not be read by API. OK?

8. **`TknetPeerState.swift` retention.** File may linger unused until Segment 3 moves it into the Peer app target. Prefer unused file over a risky cross-target move in Segment 2.
