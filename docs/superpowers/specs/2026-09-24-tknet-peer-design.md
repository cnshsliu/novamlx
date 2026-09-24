# Tknet Node — Design Spec

**Date:** 2026-09-24
**Status:** Approved (all design sections user-confirmed 2026-09-23/24)
**Owner:** lucasliu
**Companion map:** `~/Documents/SwiftMind/Tknet Node Architecture.swiftmind.html`

## Context

tknet.ai (the platform) wants a decentralized inference supply network. It publishes a **demand list** — large models it needs, with name/version and modality (language, image, audio/video). Anyone can run a **Tknet Node**: register, pick demand entries they can serve, and map each to a **source** — NovaMLX local inference, another local service (ollama, llama.cpp), or a public cloud API. When an end user on tknet.ai requests a model, the platform picks a node by reputation and tunnels the request to it; the node forwards to its source and streams the response back.

The node must ship on macOS (inside NovaMLX) and on Windows/Linux (standalone), from **one repository**.

### Decisions locked during brainstorming

1. **One repo, two entry points** (not "same project with features disabled", not a separate project). A new `NovaMLXTknetNode` module with **zero MLX/UI dependencies**; macOS embeds it in the menu-bar app, other platforms build a standalone CLI product. Rejected: platform-`#if` shredding the app; a second app Mac users would have to run alongside NovaMLX.
2. **Outbound WebSocket tunnel.** The node dials tknet.ai and keeps a long-lived connection; requests are pushed down it. This works behind NAT (home Macs) and adds zero latency. Rejected: server→node dialing (needs public reachability), polling (adds seconds of latency and *more* server load than idle sockets), connect-per-request (no way to notify a NAT'd node — degenerates to polling).
3. **Source API keys never leave the node.** tknet.ai stores only node identity, auth token, and capability declarations. The platform cannot bypass a node to hit its source.
4. **Demand-list checkbox mapping.** The node fetches tknet.ai's demand list; the operator selects entries and maps each to `(source endpoint, upstream model name)`. One node may declare multiple sources (e.g. NovaMLX serves Qwen, a cloud API serves GPT).
5. **Scale target: tens of thousands of nodes.** Idle degradation is in the v1 protocol, not deferred.
6. **Reputation-based dispatch.** Which node gets a request is decided by a score built from speed, success rate, and price. Scoring runs on tknet.ai; the node's obligation is to supply the telemetry.

## Goals

- macOS: one NovaMLX app with both local inference and node functionality; a Tknet page in the menu-bar UI.
- Windows/Linux amd64: `swift build -c release --product tknet-node` yields a single node binary with interactive `setup`; nothing else from the repo builds on those platforms.
- Streaming pass-through (SSE) with cancellation, backpressure, per-node concurrency caps, and full request telemetry for billing and reputation.
- Protocol carries everything reputation needs from day one (price declaration, TTFT, token counts, upstream status), so the scheduler can be built server-side later without a node update.

## Non-Goals

- The tknet.ai server side (tunnel gateway cluster, scheduler, probe service) — separate repo, Phase 3. This spec defines only the contract the node must honor.
- Deep anti-gaming (cross-node collusion detection) — v2+. v1 ships cheap countermeasures only.
- Incentive/billing settlement between tknet.ai and node operators — price is declared and metered; payout mechanics out of scope.
- AgentJev port, Laya numeric verification — unrelated, remain open as before.

## Architecture

### New module: `Sources/NovaMLXTknetNode/` (no MLX, no AppKit/SwiftUI)

Dependencies: Foundation, Hummingbird (WebSocket), AsyncHTTPClient, lightweight persistence (SQLite or JSON file with atomic writes). Nothing from `NovaMLXEngine`/`NovaMLXImage`/`NovaMLXAudio`/`NovaMLXMenuBar`.

| Component | Responsibility |
|---|---|
| `NodeConfig` | Registration state: nodeId, token ref, capability declarations (demand entry → source + upstream model + price), concurrency cap. Persisted locally (macOS: Keychain for secrets + config in app DB; others: one `0600` JSON file). |
| `FrameCodec` | Encode/decode tunnel frames (JSON). Transport-agnostic — used by both WebSocket and slow-poll transports. |
| `TunnelClient` | Outbound connection lifecycle: dial, authenticate, heartbeat with jitter, exponential-backoff reconnect with jitter, idle degradation to slow-poll after T minutes without traffic, immediate WS re-establishment on first polled request. |
| `Relay` | Request handling: map model → source, inject the source key, forward via AsyncHTTPClient (streaming), relay SSE chunks back, enforce per-request timeout and the node-wide concurrency cap, propagate cancellation. |
| `LocalSource` | `local://novamlx` — macOS only. Calls NovaMLX `InferenceService` in-process (no HTTP loopback). On other platforms this source type is not offered. |
| `NodeService` | Glue + observable status (connection state, active requests, cumulative tokens) for both the Mac UI and CLI output. |

### Executable: `Sources/tknet-node/`

- `tknet-node setup` — interactive wizard: register with tknet.ai (returns nodeId + token), fetch demand list, select entries, configure sources, test each source with a live request, save.
- `tknet-node serve` — run the node; prints status to stdout, suitable for systemd/launchd/NSSM.

### macOS integration

`TknetNodePageView` in `NovaMLXMenuBar` — register/connect flow, demand list with checkboxes, source editor (with `local://novamlx` pre-filled option), live status. `NodeService` runs in the app process.

### Cross-platform build

Package.swift: MLX/UI targets guarded by platform conditions so Linux/Windows resolve only the node product's dependency graph. `swift build --product tknet-node` builds the node alone on any platform. CI (Phase 2): GitHub Actions runners for linux-amd64 and windows-amd64 producing release binaries.

## Tunnel protocol (v1)

One outbound connection per node: `wss://tknet.ai/api/node/tunnel`, authenticated by the node token from registration. JSON frames. Companion REST: `POST /api/node/register` (issues nodeId + token), `GET /api/node/demand` (demand list).

| Direction | Frame | Payload |
|---|---|---|
| node→server | `hello` | `{nodeId, token, capabilities: [{demandId, model, sourceType, priceIn, priceOut}]}` |
| node→server | `heartbeat` | `{activeReq, queueDepth, rolling averages}` — health + scheduling weight |
| node→server | `response.chunk` | `{reqId, SSE chunk}` |
| node→server | `response.end` | `{reqId, status, ttftMs, totalMs, promptTokens, completionTokens, upstreamStatus}` — billing + reputation feed |
| server→node | `request` | `{reqId, model, apiFormat (openai\|anthropic), body}` |
| server→node | `request.cancel` | `{reqId}` |
| server→node | `demand.update` | Current demand list (added/removed/changed entries); the `hello` response also carries the full list for reconciliation after reconnect |
| node→server | `capabilities.update` | Node-side declaration changes (price, retire an entry, source swap) without re-registering |

**Demand lifecycle.** When tknet.ai retires a demand entry, the scheduler simply stops dispatching it (eligibility is computed live from demand list × capabilities; stale demandIds are ignored server-side). On the node: the mapping is marked **retired** (greyed in UI) but **not deleted** — source configs are reusable assets. The Mac app notifies the operator that any model loaded solely for that demand can be unloaded, but never auto-unloads. New demand entries arrive on the same `demand.update` frame and surface as opportunities ("new demand X — one of your sources may serve it").

Forwarding rules: the node strips all tknet.ai credentials and injects its own source key before contacting the source. WebSocket backpressure applies naturally. Every reconnect/heartbeat/poll interval carries jitter to prevent thundering herds at scale.

## Scaling (tens of thousands of nodes)

- **Server shape (contract note):** tunnel gateway is a separate, horizontally scalable process; nodeId→gateway assignment via consistent hashing; presence published to Redis. Main API never holds tunnels. Node is unaware of this.
- **Idle degradation (v1):** WS while active; after 5 minutes idle (tunable), drop to 60 s slow-poll carrying the same frames through `FrameCodec`; any polled request triggers immediate WS re-establishment.
- **Node protection:** operator-declared concurrency cap (default 1). The scheduler must not over-dispatch; declining work must be preferable to failing it (feeds reputation).

## Reputation & dispatch (tknet.ai side — contract only)

`score = w₁·latency + w₂·success + w₃·price (+ w₄·uptime)`, computed per node×model.

- latency: EWMA of TTFT and tokens/s, normalized per model class (a 27B is not compared to a 4B).
- success: decayed ratio with Bayesian smoothing (low-volume new nodes neither die on one failure nor top the board on three successes).
- price: operator-declared per-model price, normalized against the pool median. Reneging (timeout/abort after accepting) penalizes success heavily — undercutting then failing loses to slow-but-reliable.
- cold start: neutral score + ~5% exploration traffic; selection by softmax/ε-greedy over top-K eligible nodes, not pure argmax.
- v1 anti-gaming: periodic real probe requests cross-check self-reported metrics.

## Security

- Node token: macOS Keychain; other platforms `0600` config file.
- Source keys never leave the node (no field for them exists in the protocol).
- TLS everywhere; tknet.ai credentials never forwarded to sources.
- Per-node rate limits and quotas protect the operator's machine and keys.

## Testing

1. `FrameCodec` unit tests (pure, run on all platforms).
2. Relay integration tests: mock source (Hummingbird test server) + mock tunnel server; streaming pass-through, cancellation, timeout, concurrency cap, header hygiene.
3. Idle-degradation test: WS→poll→WS transition under a fake timer.
4. macOS UI smoke: Tknet page appears; register against mock server; status reaches connected.

## Phasing

- **Phase 1 (this repo):** `NovaMLXTknetNode` module + full frame protocol (all telemetry fields) + `tknet-node` CLI + Mac `TknetNodePageView`; end-to-end against a mock tknet.ai server.
- **Phase 2 (this repo):** Linux/Windows CI builds; idle degradation transport.
- **Phase 3 (tknet.ai repo):** gateway clustering, reputation scheduler, probe service.

## Risks

- **Swift on Windows is the roughest edge.** The node's dependency surface is minimal (WS + HTTP relay); worst case, a Go re-implementation of the node preserves the protocol without touching this repo's architecture.
- **Latency is the product's death sentence.** Active nodes must see zero added dispatch latency; any future "server savings" scheme must not introduce polling delay for active traffic.

## Reference

- Design dialogue: 2026-09-23/24 (WebSocket-vs-polling cost analysis; notification channel as the scarce resource; reputation requirements).
- SwiftMind map (status tracker): `~/Documents/SwiftMind/Tknet Node Architecture.swiftmind.html`, node ids `tkn_*` for CLI status updates.
