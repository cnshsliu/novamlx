# Tknet Peer Gateway — Segment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `NovaMLXTknetPeer` a member table, named pools, and the five NovaMLX selection strategies; expose an OpenAI-compatible local gateway on `127.0.0.1:6600` (even with no tunnel); extend `tknet-peer` so operators can CRUD members/pools and `serve` without dialing tknet. Stay inside this repo — do not split packages, do not touch NovaMLX LB/Tokenhub UI (Segment 2), menu-bar Peer app (Segment 3), or cross-platform packaging (Segment 4).

**Architecture:** Evolve the existing single-`SourceConfig` / single-target `Relay` into **members + pools + a shared forwarder**. Selection algorithms are a **port** of `Sources/NovaMLXCore/LBRouter.swift` + retry semantics of `Sources/NovaMLXAPI/LBProxy.swift` — copied into the Peer module so `NovaMLXTknetPeer` never imports `NovaMLXCore` / MLX / GRDB. Local clients and the tunnel both call the same `PoolForwarder`. `LocalGateway` (Hummingbird) listens on the loopback OpenAI port; the tunnel path keeps using `Relay.handle` which now resolves `Capability.sourceId` as a **pool id**.

**Tech Stack:** Swift 6 (StrictConcurrency), Foundation, Hummingbird 2.x + HummingbirdRouter (local gateway), AsyncHTTPClient 1.33 (forwarding), ArgumentParser (CLI), Swift Testing. Build via `./build.sh`; Peer product MLX-free via `swift build --product tknet-peer`.

**Spec:** `docs/superpowers/specs/2026-10-11-tknet-peer-gateway-design.md`  
**Phase overview:** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway.md` (leave that four-segment outline intact)  
**Algorithm source of truth:** `Sources/NovaMLXCore/LBRouter.swift`, `Sources/NovaMLXCore/LoadBalancerTypes.swift`, `Sources/NovaMLXAPI/LBProxy.swift`, `Sources/NovaMLXAPI/APIServer+LBProxy.swift`

## Global Constraints

- `NovaMLXTknetPeer` / `tknet-peer` MUST NOT import or depend on: `MLX*`, `NovaMLXEngine`, `NovaMLXInference`, `NovaMLXImage`, `NovaMLXAudio`, `NovaMLXDB`, `NovaMLXCore`, `SwiftUI`/`AppKit`. Allowed: Foundation, Hummingbird, HummingbirdRouter, HummingbirdWebSocket, AsyncHTTPClient, Logging, ArgumentParser (CLI only).
- Build on macOS via `./build.sh` for the app/shared graph. Peer-only product check may use `swift build -c release --product tknet-peer` (same exception as Phase 1). Tests: `./build.sh test --filter <suite>`. Clean debug build: zero `warning:` lines.
- Do **not** import NovaMLX's `LBRouter` / `LBProxy` — port the algorithms. Selection modes and retry rules must match the existing NovaMLX LB behaviour (see Task 2 / Task 4 notes).
- Secrets stay in `SecretStore` (`apiKeyRef` only in config). Never log keys, never put them in frames, gateway responses, or CLI output.
- Segment 1 stays in this git repo. No repo split, no copying inference code, no re-registration of peer identity, no active health probes, no Anthropic native path, no embeddings/image/audio routes, no per-API-key pools.
- All new tests use Swift Testing (`import Testing`, `@Test`, `#expect`), matching `Tests/NovaMLXTknetPeerTests/`.
- Prefer incremental evolution of `SourceConfig` / `Relay` / `PeerService` / CLI over a greenfield rewrite. Existing tunnel tests must keep passing.
- UI copy (if any user-facing string appears): a claimed Tknet Peer is a **「token 工厂」** / "token factory", never "我的机器" / "my machine". Segment 1 is mostly CLI/library; still apply if help text mentions it.
- Do not commit unrelated untracked local files (`speech.wav`, `next_claude`, `tmp/talking-head-options.md`, etc.).

### Key behavioural contracts (from design + current LB)

| Case | Behaviour |
|---|---|
| Unknown pool name | HTTP **404** |
| Pool disabled / no enabled members after filter | HTTP **503** |
| Connect fail / timeout / **5xx** | Record failure, try next candidate up to `maxRetries` |
| Member **4xx** | Pass status + body to caller; **do not** failover |
| All candidates exhausted | HTTP **502** with last error |
| Default strategy | `tiered` |
| Default `maxRetries` | `3` (attempts = `min(maxRetries, candidates.count)`, at least 1) |
| `X-No-Autoload: 1` | Only when the chosen member is loopback (`127.0.0.1` / `localhost` / `::1`) **or** `type == .localNovaMLX` |
| Tiered ranking | Loopback / `localNovaMLX` = tier 2; non-loopback `isFree` = tier 1; else tier 0. Rotate within tier via `requestCount`. **No** "is model loaded" filter — fail-fast + failover instead. |

### Config migration (on load)

Existing `peer.json` has `sources: [SourceConfig]` and `capabilities[].sourceId` pointing at a source. On load:

1. Decode members from `sources` (new fields default: `weight = nil`, `isEnabled = true`, `isFree = false`).
2. If `pools` is absent/empty: for each capability, create one `PoolConfig` with `id == capability.sourceId`, `name == capability.model`, `memberIds == [capability.sourceId]`, `strategy == .tiered`, `maxRetries == 3`. Deduplicate by pool id (two capabilities sharing a source → one pool; see ambiguity note at end — prefer one pool per capability.model when ids would collide).
3. Persist the migrated shape on next save so Segment 2 migration has a stable schema.

Wire `Capability.sourceId` keeps its JSON name (tknet hello frames unchanged) but **means pool id** after Segment 1.

---

### Task 1: Member + Pool domain types and PeerConfig migration

**Files:**
- Modify: `Sources/NovaMLXTknetPeer/PeerConfig.swift`
- Create: `Sources/NovaMLXTknetPeer/PoolTypes.swift` (or keep types in `PeerConfig.swift` if small — prefer a dedicated file matching LB's split)
- Test: `Tests/NovaMLXTknetPeerTests/PeerConfigTests.swift` (extend), create `Tests/NovaMLXTknetPeerTests/PoolConfigMigrationTests.swift`

**Interfaces:**
- Produces: `MemberConfig` (evolved `SourceConfig`), `PoolConfig`, `PoolStrategy` (raw values match `LBStrategy`: `tiered`, `round_robin`, `weighted`, `lowest_latency`, `random`), `PeerConfig.members` / `PeerConfig.pools`, load-time migration from legacy `sources`-only files, `ConfigStore` encode/decode of the new fields.

- [ ] **Step 1: Write failing migration + codec tests**

```swift
import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("Pool config migration")
struct PoolConfigMigrationTests {
    @Test("legacy sources-only file becomes one pool per capability")
    func migratesLegacySources() throws {
        let json = """
        {
          "serverURL": "wss://tknet.ai",
          "sources": [{
            "id": "src-d1", "name": "qwen", "type": "openaiCompatible",
            "endpoint": "http://127.0.0.1:6590/v1",
            "apiKeyRef": "src/src-d1", "upstreamModel": "Qwen3-8B"
          }],
          "capabilities": [{
            "demandId": "d1", "model": "qwen-3-8b", "sourceId": "src-d1",
            "sourceType": "openaiCompatible", "priceIn": 0, "priceOut": 0
          }]
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(PeerConfig.self, from: json)
        #expect(config.members.count == 1)
        #expect(config.members[0].id == "src-d1")
        #expect(config.members[0].isEnabled == true)
        #expect(config.members[0].isFree == false)
        #expect(config.pools.count == 1)
        #expect(config.pools[0].id == "src-d1")
        #expect(config.pools[0].name == "qwen-3-8b")
        #expect(config.pools[0].memberIds == ["src-d1"])
        #expect(config.pools[0].strategy == .tiered)
        #expect(config.pools[0].maxRetries == 3)
    }

    @Test("explicit pools round-trip without inventing extras")
    func poolsRoundTrip() throws {
        var config = PeerConfig.defaultConfig()
        config.members = [
            MemberConfig(id: "m1", name: "a", type: .openaiCompatible,
                         endpoint: URL(string: "http://10.0.0.2:8080/v1")!,
                         apiKeyRef: "src/m1", upstreamModel: "x",
                         weight: 2, isEnabled: true, isFree: true)
        ]
        config.pools = [
            PoolConfig(id: "coding", name: "coding", strategy: .roundRobin,
                       memberIds: ["m1"], maxRetries: 2)
        ]
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(PeerConfig.self, from: data)
        #expect(decoded.pools == config.pools)
        #expect(decoded.members == config.members)
    }
}
```

- [ ] **Step 2: Add `PoolTypes.swift`**

Port enum names/raw values from `LoadBalancerTypes.swift`:

```swift
public enum PoolStrategy: String, Codable, Sendable, CaseIterable, Equatable {
    case tiered
    case roundRobin = "round_robin"
    case weighted
    case lowestLatency = "lowest_latency"
    case random
}

public struct MemberConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var type: SourceType
    public var endpoint: URL
    public var apiKeyRef: String
    public var upstreamModel: String
    public var weight: Int?
    public var isEnabled: Bool
    public var isFree: Bool
    // init with defaults weight=nil, isEnabled=true, isFree=false
    // custom decode: missing weight/isEnabled/isFree → defaults
}

public struct PoolConfig: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var strategy: PoolStrategy
    public var memberIds: [String]
    public var maxRetries: Int
    public var isEnabled: Bool
    public var requestCount: Int  // bumped per accepted request (not per retry)
    // defaults: strategy=.tiered, maxRetries=3, isEnabled=true, requestCount=0
}

/// Loopback detection for tiered ranking + X-No-Autoload.
public enum MemberAddress {
    public static func isLoopback(_ url: URL) -> Bool { /* 127.0.0.1, localhost, ::1 */ }
    public static func isLocalTier(_ member: MemberConfig) -> Bool {
        member.type == .localNovaMLX || isLoopback(member.endpoint)
    }
}
```

Keep `public typealias SourceConfig = MemberConfig` **or** rename call sites in Peer + tests in the same commit. Prefer keeping the `SourceConfig` name as a typealias so existing test helpers compile during the transition, then delete the alias once call sites are updated (same Task).

- [ ] **Step 3: Extend `PeerConfig`**

- Replace `sources: [SourceConfig]` storage with `members: [MemberConfig]`.
- Add `pools: [PoolConfig]`.
- CodingKeys: decode `sources` **or** `members` (prefer `members` when both present); always encode `members` + `pools`.
- After decode, if `pools.isEmpty` and `members` non-empty, run migration from capabilities (see above).
- Helpers: `member(id:)`, `pool(id:)`, `pool(named:)` (match by `name` for local gateway model lookup), `upsertMember`, `removeMember` (also strip id from all pools), `upsertPool`, `removePool`.
- `ServerMemory` currently embeds `sources` — migrate that field the same way (`members` + `pools` per saved server), with decode defaults so old memory rows still load.

- [ ] **Step 4: Update existing PeerConfig / Relay / CLI tests that construct `SourceConfig`**

Rename call sites through the typealias (or drop the alias once renamed). Run:

`./build.sh test --filter 'PeerConfigTests|PoolConfigMigrationTests|CLIConfigTests'`

Expected: new migration tests pass; existing PeerConfig tests pass.

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXTknetPeer/PoolTypes.swift Sources/NovaMLXTknetPeer/PeerConfig.swift \
  Tests/NovaMLXTknetPeerTests/PoolConfigMigrationTests.swift Tests/NovaMLXTknetPeerTests/PeerConfigTests.swift
git commit -m "feat(peer): member + pool config types with legacy source migration"
```

---

### Task 2: PoolRouter — port LBRouter selection (no NovaMLXCore import)

**Files:**
- Create: `Sources/NovaMLXTknetPeer/PoolRouter.swift`
- Create: `Tests/NovaMLXTknetPeerTests/PoolRouterTests.swift`
- Reference (read-only): `Sources/NovaMLXCore/LBRouter.swift`, `Tests/NovaMLXCoreTests/LBRouterTests.swift`

**Interfaces:**
- Produces: `PoolRouterInput`, `PoolRouter.plan(_:) -> [MemberConfig]` (ordered candidates, preferred first). Empty = no healthy members.

**Adaptation from NovaMLX LB (do not invent a new model):**

| NovaMLX `LBRouter` | Peer `PoolRouter` |
|---|---|
| `member.kind == .local` + `isLocalLoaded` | `MemberAddress.isLocalTier` — **always** eligible if `isEnabled` (no loaded check) |
| `isProviderFree(ref)` | `member.isFree` |
| `lb.requestCount` rotation | `pool.requestCount` |
| `lb.strategy` | `pool.strategy` |
| Stats keyed by `UUID` | Stats keyed by `member.id: String` |
| Unloaded local skipped | Loopback still planned; fail-fast at forward time |

- [ ] **Step 1: Write failing router tests** (mirror `LBRouterTests` cases that still apply)

Cover at least: empty → empty; disabled skipped; tiered local > free > paid; round_robin rotates by `requestCount`; weighted expands nil→1 / 0→1; lowest_latency cold-start = 0 preferred; random deterministic by id hash; snake_case raw values.

- [ ] **Step 2: Implement `PoolRouter`**

Copy structure from `LBRouter.swift` (`applyTiered` / `applyRoundRobin` / `applyWeighted` / `applyLowestLatency` / `applyRandom` + `Array.rotated(by:)`). Keep comments citing the NovaMLX original so Segment 2 reviewers see the intentional port.

- [ ] **Step 3: Run tests**

`./build.sh test --filter PoolRouterTests`  
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXTknetPeer/PoolRouter.swift Tests/NovaMLXTknetPeerTests/PoolRouterTests.swift
git commit -m "feat(peer): PoolRouter ports NovaMLX LB selection strategies"
```

---

### Task 3: In-memory member stats

**Files:**
- Create: `Sources/NovaMLXTknetPeer/MemberStats.swift`
- Create: `Tests/NovaMLXTknetPeerTests/MemberStatsTests.swift`

**Interfaces:**
- Produces: `MemberStats` (fields aligned with `LBMemberStats`: request/success/failure/count5xx, totalLatencyMs, lastUsedAt, lastError, avgLatencyMs, successRate) and `MemberStatsStore` (actor or lock-backed map `String -> MemberStats`) with `record(memberId:succeeded:latencyMs:httpStatus:errorMessage:)` and `snapshot() -> [String: MemberStats]`.
- Stats are **process-local** for Segment 1 (not persisted). Persisting can wait; `lowest_latency` only needs live data.

- [ ] **Step 1: Failing tests** — record success updates avg; 5xx bumps `count5xx`; 4xx counts as request but `succeeded` per LBProxy rule (`status < 500` → treat as non-failover success for stats, matching `LBProxy.recordOutcome`).

- [ ] **Step 2: Implement store**

- [ ] **Step 3: `./build.sh test --filter MemberStatsTests`**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(peer): in-memory member stats for lowest_latency selection"
```

---

### Task 4: Package.swift — Hummingbird for the Peer library

**Files:**
- Modify: `Package.swift` (`NovaMLXTknetPeer` target dependencies)

**Why now:** `LocalGateway` and possibly shared response helpers need Hummingbird types. Tests already depend on Hummingbird; the library currently only links `HummingbirdWebSocket`.

- [ ] **Step 1: Add products to the Peer target**

```swift
.target(
    name: "NovaMLXTknetPeer",
    dependencies: [
        .product(name: "Hummingbird", package: "hummingbird"),
        .product(name: "HummingbirdRouter", package: "hummingbird"),
        .product(name: "HummingbirdWebSocket", package: "hummingbird-websocket"),
        .product(name: "HummingbirdWSClient", package: "hummingbird-websocket"),
        .product(name: "AsyncHTTPClient", package: "async-http-client"),
        .product(name: "Logging", package: "swift-log"),
    ],
    swiftSettings: concurrencySettings
),
```

- [ ] **Step 2: Build Peer product and confirm MLX-free graph**

```bash
./build.sh -c debug --product tknet-peer 2>&1 | tee /tmp/peer-build.txt
# Fallback / explicit graph check (allowed for product-only):
swift build -c release --product tknet-peer
swift package show-dependencies --product tknet-peer 2>/dev/null || \
  swift package describe --type json | python3 -c "
import json,sys
# assert 'mlx' not in dependency names of NovaMLXTknetPeer / tknet-peer
"
```

Expected: build succeeds; dependency list has no `mlx-swift` / `MLX` / `NovaMLXEngine`.

- [ ] **Step 3: Commit**

```bash
git add Package.swift Package.resolved
git commit -m "chore(peer): link Hummingbird into NovaMLXTknetPeer for local gateway"
```

---

### Task 5: PoolForwarder — HTTP forward + retry (shared by gateway and tunnel)

**Files:**
- Create: `Sources/NovaMLXTknetPeer/PoolForwarder.swift`
- Modify: `Sources/NovaMLXTknetPeer/Relay.swift` (delegate multi-member path to PoolForwarder; keep frame emission API)
- Modify: `Tests/NovaMLXTknetPeerTests/RelayTests.swift` (X-No-Autoload now conditional)
- Create: `Tests/NovaMLXTknetPeerTests/PoolForwarderTests.swift`
- Reuse: `Tests/NovaMLXTknetPeerTests/MockSourceServer.swift`

**Interfaces:**
- Produces: `PoolForwarder` owning `HTTPClient`, `SecretStore`, config holder, stats store.
- `func forward(poolName:body:timeoutSeconds:) async -> PoolForwardOutcome` for the local gateway (raw HTTP semantics).
- `func forwardTunnel(request: RequestFrame) -> AsyncStream<Frame>` **or** Relay calls into a lower-level `attempt(member:body:)` used by both.
- Outcomes mirror `LBProxyOutcome`: `.success(status:headers:bodyStream:)`, `.unknownPool`, `.poolDisabled`, `.noHealthyMembers`, `.allMembersFailed(lastError:)`.

**Retry rules (copy `LBProxy.handle`):**

```text
candidates = PoolRouter.plan(...)
bump pool.requestCount once
maxAttempts = max(1, min(pool.maxRetries, candidates.count))
for member in candidates.prefix(maxAttempts):
    try HTTP POST member.endpoint + "/chat/completions"
    rewrite JSON "model" → member.upstreamModel
    Authorization: Bearer <SecretStore.load(apiKeyRef)> if non-empty
    if MemberAddress.isLocalTier(member): header X-No-Autoload: 1
    if status < 500: return body to caller (4xx pass-through)
    else: record failure, continue
    on connect error / deadlineExceeded: record failure, continue
return allMembersFailed
```

Do **not** stitch partial SSE from attempt N into attempt N+1 (design: 挑下一台时不把上一家的半截输出拼进去).

- [ ] **Step 1: Failing acceptance-shaped test (Segment 1 gate)**

```swift
@Test("two upstreams in one pool; first 500 falls through to second")
func failoverOn5xx() async throws {
    let a = MockSourceServer(); try await a.start()
    let b = MockSourceServer(); try await b.start()
    defer { Task { await a.stop(); await b.stop() } }
    a.failNextWithStatus = 500
    // build config: pool "coding" with members a,b; strategy tiered or round_robin
    // POST via PoolForwarder.forward(poolName: "coding", body: ...)
    // expect 200 SSE from b; a saw one request; b saw one request; b.lastBodyModel == upstream of b
}
```

Also test: 4xx from first member is returned, second never hit; unknown pool → unknownPool; disabled pool → poolDisabled; timeout retries; loopback gets `X-No-Autoload`, non-loopback does not.

- [ ] **Step 2: Implement `PoolForwarder`**

Share one `HTTPClient(eventLoopGroupProvider: .singleton)` lifecycle with `shutdown()` — same trap as today's `Relay`.

- [ ] **Step 3: Rewire `Relay.run`**

Resolve `request.model` → capability → **pool** (`capability.sourceId` as pool id) → `PoolForwarder`. Map forward outcomes to existing `responseChunk` / `responseEnd` frames (failed/timeout/cancelled unchanged for tunnel). Preserve Anthropic rejection. Update `RelayTests.noAutoloadAndNestedError` to use a loopback endpoint (already is `127.0.0.1`) so header still expected; add a non-loopback case expecting nil header.

- [ ] **Step 4: Run**

`./build.sh test --filter 'PoolForwarderTests|RelayTests|RelayUsageSplitTests'`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(peer): PoolForwarder with LB-style retry; Relay uses pools"
```

---

### Task 6: LocalGateway — OpenAI `/v1/chat/completions` + `/v1/models` on loopback

**Files:**
- Create: `Sources/NovaMLXTknetPeer/LocalGateway.swift`
- Create: `Tests/NovaMLXTknetPeerTests/LocalGatewayTests.swift`
- Modify: `Tests/NovaMLXTknetPeerTests/MockSourceServer.swift` only if `/v1/models` helpers are needed (prefer not)

**Interfaces:**
- Produces: `LocalGateway` — `start(host:port:forwarder:configProvider:)` / `stop()`, bound port readable for tests (bind `0` in tests, `6600` in production default).
- Routes:
  - `POST /v1/chat/completions` — model field = **pool name**; stream and non-stream both supported (Segment 1: streaming SSE passthrough is enough if non-stream is forwarded as-is).
  - `GET /v1/models` — list enabled pools as `{ id: pool.name, object: "model", owned_by: "tknet-peer" }`.
- Map outcomes to HTTP like `APIServer+LBProxy.lbOutcomeToResponse`:
  - unknown → 404
  - disabled / no healthy → 503
  - all failed → 502
  - success → upstream status + body (including 4xx pass-through)

Default listen: `127.0.0.1:6600` (configurable via PeerConfig field `gatewayHost` / `gatewayPort` with decode defaults — add in this task).

- [ ] **Step 1: Failing gateway tests**

Use two `MockSourceServer`s + `LocalGateway` on port 0 + `URLSession`/`AsyncHTTPClient` client hitting `http://127.0.0.1:<gw>/v1/chat/completions`. Assert failover 500→second, `/v1/models` lists pool names, unknown model → 404.

- [ ] **Step 2: Implement Hummingbird router**

Pattern after `MockSourceServer` (Application + `onServerRunning` for port) and NovaMLXAPI's JSON error shape (`error.message` / `error.type`).

- [ ] **Step 3: `./build.sh test --filter LocalGatewayTests`**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(peer): LocalGateway OpenAI port with pool dispatch"
```

---

### Task 7: PeerService — gateway always on; tunnel optional

**Files:**
- Modify: `Sources/NovaMLXTknetPeer/PeerService.swift`
- Modify: `Tests/NovaMLXTknetPeerTests/PeerServiceTests.swift`

**Interfaces:**
- Extend `PeerService` so `start(options:)` (or separate `startGateway()` + existing `start()`) can:
  1. Always start `LocalGateway` when `gatewayPort` configured.
  2. Dial the tunnel only when `peerId != nil` **and** caller asked for tunnel (CLI flag).
- Today `start()` throws `TunnelError.notRegistered` without `peerId` — that blocks Segment 1's "serve without tunnel". Change: missing peerId skips tunnel (log info), still serves gateway.
- `stop()` tears down gateway + tunnel + forwarder/relay shutdown order (HTTPClient trap).
- `applyConfig` updates forwarder + gateway's view of pools without restart when possible.
- Status: extend `PeerStatus` with `gatewayListening: Bool` / `gatewayPort: Int?` (optional but useful for CLI prints).

- [ ] **Step 1: Failing tests** — start without peerId → gateway up, connection stays `.idle`; start with peerId + mock tunnel → both paths work; stop is idempotent.

- [ ] **Step 2: Implement**

Share one `PoolForwarder` between gateway and relay. `Relay` may wrap the same forwarder instance.

- [ ] **Step 3: `./build.sh test --filter 'PeerServiceTests|TunnelClientTests|EndToEndTests'`**

Expected: existing tunnel suites still pass.

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(peer): PeerService serves local gateway without requiring tunnel"
```

---

### Task 8: CLI — member/pool CRUD + serve listens on 6600

**Files:**
- Modify: `Sources/tknet-peer/TknetPeerCLI.swift`
- Modify: `Sources/NovaMLXTknetPeer/CLIConfig.swift` if default paths/helpers needed
- Create: `Tests/NovaMLXTknetPeerTests/CLIPoolCommandTests.swift` (unit-test pure helpers in the module; ArgumentParser command parsing can stay light)

**Subcommands to add** (under `tknet-peer`, alongside `setup` / `serve` / `earnings`):

```text
tknet-peer member add    --id --name --endpoint --upstream-model [--api-key] [--weight] [--free/--paid] [--type]
tknet-peer member remove --id
tknet-peer member list
tknet-peer member update --id [fields...]

tknet-peer pool add      --id/--name --strategy/--max-retries --member <id> (repeatable)
tknet-peer pool remove   --id
tknet-peer pool list
tknet-peer pool update   --id [fields...] [--add-member/--remove-member]
```

Store keys via `FileSecretStore` at `src/<memberId>` (existing pattern from `setup`). Never print the key back.

**`serve` changes:**

```text
tknet-peer serve [--config PATH] [--listen-host 127.0.0.1] [--listen-port 6600] [--tunnel/--no-tunnel]
```

- Default: start gateway always.
- `--tunnel` (default **on** when `peerId` present, **off** when absent): dial tknet.
- `--no-tunnel`: gateway only (Segment 1 acceptance path).
- Remove the hard `Not registered — run setup first` failure when `--no-tunnel` (or when no peerId); still require registration when dialing the tunnel.
- Print `gateway listening on http://127.0.0.1:6600` and existing status lines.

- [ ] **Step 1: Implement member/pool commands** writing through `ConfigStore.save` (0600).

- [ ] **Step 2: Wire `Serve` to `PeerService` gateway + optional tunnel.**

- [ ] **Step 3: Manual smoke (document in commit body; optional script)**

```bash
./build.sh --product tknet-peer
# in a temp config dir:
tknet-peer member add --id m1 --endpoint http://127.0.0.1:$PORT_A/v1 --upstream-model u1 ...
tknet-peer member add --id m2 --endpoint http://127.0.0.1:$PORT_B/v1 --upstream-model u2 ...
tknet-peer pool add --name coding --member m1 --member m2
tknet-peer serve --no-tunnel --config ...
curl -s localhost:6600/v1/models
```

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(peer): CLI member/pool CRUD; serve local gateway without tunnel"
```

---

### Task 9: Segment 1 acceptance suite + regression

**Files:**
- Create: `Tests/NovaMLXTknetPeerTests/GatewayAcceptanceTests.swift`
- Modify: any broken EndToEnd fixtures that still assume single-source-only config (ensure migration fills pools)

**Acceptance tests (must exist and pass):**

1. **Failover:** two fake HTTP upstreams in one pool; first returns 500 → request served by second (`GatewayAcceptanceTests` / `PoolForwarderTests`).
2. **MLX-free:** document + scriptable check that `tknet-peer` dependency graph has no MLX (run in Step 3).
3. **Tunnel regression:** `./build.sh test --filter NovaMLXTknetPeerTests` — all existing suites green.

- [ ] **Step 1: Write `GatewayAcceptanceTests`** end-to-end through `LocalGateway` (not only PoolForwarder unit).

- [ ] **Step 2: Full Peer suite**

```bash
./build.sh test --filter NovaMLXTknetPeerTests
```

- [ ] **Step 3: MLX-free product build**

```bash
swift build -c release --product tknet-peer
# Confirm Package.swift NovaMLXTknetPeer / tknet-peer deps list contains no MLX* / NovaMLXEngine / NovaMLXCore
```

- [ ] **Step 4: Commit**

```bash
git commit -m "test(peer): Segment 1 gateway acceptance — failover, models list, suite green"
```

---

### Task 10: Verification gate (no new features)

**Files:** none required (checklist only). Optionally one-line note in `docs/superpowers/plans/2026-10-11-tknet-peer-gateway.md` that Segment 1 plan lives at this file — **only if Lucas wants**; default leave the overview untouched.

- [ ] **Step 1: Zero-warning debug build**

`./build.sh -c debug` → no `warning:` lines from Peer changes.

- [ ] **Step 2: Peer tests**

`./build.sh test --filter NovaMLXTknetPeerTests`

- [ ] **Step 3: Product MLX-free**

`swift build -c release --product tknet-peer`

- [ ] **Step 4: Manual failover demo** (two MockSourceServer ports or `nc`/`python -m http.server` fakes) via `tknet-peer serve --no-tunnel`.

- [ ] **Step 5: Stop** — do not start Segment 2 (NovaMLX LB page removal) in this branch of work.

---

## Acceptance criteria mapping

| Design / overview criterion | Covered by |
|---|---|
| Members + pools + selection in Peer library | Tasks 1–5 |
| CLI add/remove/update members and pools | Task 8 |
| `serve` listens on `127.0.0.1:6600` without tunnel | Tasks 6–8 |
| `/v1/chat/completions` + `/v1/models` | Task 6 |
| Strategies: tiered, round_robin, weighted, lowest_latency, random (default tiered) | Task 2 |
| Retry connect/timeout/5xx; pass through 4xx; 404/503/502 | Tasks 5–6 |
| `X-No-Autoload: 1` for NovaMLX loopback members | Task 5 |
| Secrets via SecretStore | Tasks 5, 8 (existing `FileSecretStore`) |
| Stay in-repo as NovaMLXTknetPeer + tknet-peer | Global Constraints |
| Two fake upstreams, first 500 → second | Tasks 5, 9 |
| `tknet-peer` dependency graph has no MLX | Tasks 4, 9, 10 |
| Existing tunnel tests still pass | Tasks 7, 9 |

## Out of scope (do not implement in Segment 1)

- Migrating NovaMLX LB/Tokenhub pages or deleting `lb:` routes (Segment 2)
- Mac menu-bar Peer-only app / auto-dial policy UI (Segment 3)
- Extracting to a separate git repo / Linux-Windows packaging (Segment 4)
- Active health probes, Anthropic native upstream, embeddings/image/audio routes, per-API-key pools
- Re-registering peer identity; copying inference code into Peer

## Ambiguities / risks for Lucas

1. **Capability ↔ pool cardinality on migration.** Today many capabilities can share one `sourceId`. Design text says each old source becomes a one-member pool named after the **published model**. If two capabilities share one source but different `model` strings, migration should create **two pools** (different names) that both list the same member id — or one pool and leave the second capability's `sourceId` retargeted. **Recommendation in Task 1:** one pool per capability (`id` = `capability.sourceId + "-" + capability.model` when collision, else `sourceId`), both referencing the same member. Confirm if you prefer strict `id == old sourceId` for hello stability (hello sends `model` + `sourceId`; tknet may not key on sourceId).

2. **`X-No-Autoload` behaviour change.** Current `Relay` always sends the header (and `RelayTests` assert it). Design restricts it to NovaMLX/loopback. Non-loopback OpenAI-compatible members will stop receiving it — intentional; tests must be updated.

3. **`LBProxy` treats `status < 500` as success (including 4xx) for stats.** Peer should match for failover decisions; whether 4xx increments `successCount` is cosmetic for Segment 1 but affects `lowest_latency` / successRate. Plan follows LBProxy.

4. **`serve` without registration.** Product decision: gateway-only mode needs no peer token. Tunnel mode still needs `setup`. Status stream currently assumes a tunnel client — keep publishing `.idle` when tunnel not started.

5. **Hummingbird in the library** slightly widens the Peer dependency surface but stays MLX-free. Alternative (HTTP server only in the executable target) would force duplicating types; prefer library placement so Mac Segment 3 can embed the same gateway.

6. **`ServerMemory.sources` migration** must not drop per-server bindings when users switch servers in the Mac UI (still on main app until Segment 3). Decode both old and new keys.

7. **Do not import `NovaMLXCore.LBRouter`.** It pulls Core → DB/Tokenhub types and would poison the MLX-free graph.
