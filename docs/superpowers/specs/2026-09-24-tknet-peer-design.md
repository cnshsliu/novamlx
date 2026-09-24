# Tknet Peer — Design Spec

**Date:** 2026-09-24 (renamed Node→Peer 2026-09-24; protocol not yet deployed, no compat concerns)
**Status:** Approved (all design sections user-confirmed 2026-09-23/24)
**Owner:** lucasliu
**Companion map:** `~/Documents/SwiftMind/Tknet Peer Architecture.swiftmind.html`

## Context

tknet.ai (the platform) wants a decentralized inference supply network. It publishes a **demand list** — large models it needs, with name/version and modality (language, image, audio/video). Anyone can run a **Tknet Peer**: register, pick demand entries they can serve, and map each to a **source** — NovaMLX local inference, another local service (ollama, llama.cpp), or a public cloud API. When an end user on tknet.ai requests a model, the platform picks a peer by reputation and tunnels the request to it; the peer forwards to its source and streams the response back.

The peer must ship on macOS (inside NovaMLX) and on Windows/Linux (standalone), from **one repository**.

### Decisions locked during brainstorming

1. **One repo, two entry points** (not "same project with features disabled", not a separate project). A new `NovaMLXTknetPeer` module with **zero MLX/UI dependencies**; macOS embeds it in the menu-bar app, other platforms build a standalone CLI product. Rejected: platform-`#if` shredding the app; a second app Mac users would have to run alongside NovaMLX.
2. **Outbound WebSocket tunnel.** The peer dials tknet.ai and keeps a long-lived connection; requests are pushed down it. This works behind NAT (home Macs) and adds zero latency. Rejected: server→peer dialing (needs public reachability), polling (adds seconds of latency and *more* server load than idle sockets), connect-per-request (no way to notify a NAT'd peer — degenerates to polling).
3. **Source API keys never leave the peer.** tknet.ai stores only peer identity, auth token, and capability declarations. The platform cannot bypass a peer to hit its source.
4. **Demand-list checkbox mapping.** The peer fetches tknet.ai's demand list; the operator selects entries and maps each to `(source endpoint, upstream model name)`. One peer may declare multiple sources (e.g. NovaMLX serves Qwen, a cloud API serves GPT).
5. **Scale target: tens of thousands of peers.** Idle degradation is in the v1 protocol, not deferred.
6. **Reputation-based dispatch.** Which peer gets a request is decided by a score built from speed, success rate, and price. Scoring runs on tknet.ai; the peer's obligation is to supply the telemetry.

## Goals

- macOS: one NovaMLX app with both local inference and peer functionality; a Tknet page in the menu-bar UI.
- Windows/Linux amd64: `swift build -c release --product tknet-peer` yields a single peer binary with interactive `setup`; nothing else from the repo builds on those platforms.
- Streaming pass-through (SSE) with cancellation, backpressure, per-peer concurrency caps, and full request telemetry for billing and reputation.
- Protocol carries everything reputation needs from day one (price declaration, TTFT, token counts, upstream status), so the scheduler can be built server-side later without a peer update.

## Non-Goals

- The tknet.ai server side (tunnel gateway cluster, scheduler, probe service) — separate repo, Phase 3. This spec defines only the contract the peer must honor.
- Deep anti-gaming (cross-peer collusion detection) — v2+. v1 ships cheap countermeasures only.
- Incentive/billing settlement between tknet.ai and peer operators — price is declared and metered; payout mechanics out of scope.
- AgentJev port, Laya numeric verification — unrelated, remain open as before.

## Architecture

### New module: `Sources/NovaMLXTknetPeer/` (no MLX, no AppKit/SwiftUI)

Dependencies: Foundation, Hummingbird (WebSocket), AsyncHTTPClient, lightweight persistence (SQLite or JSON file with atomic writes). Nothing from `NovaMLXEngine`/`NovaMLXImage`/`NovaMLXAudio`/`NovaMLXMenuBar`.

| Component | Responsibility |
|---|---|
| `PeerConfig` | Registration state: peerId, token ref, capability declarations (demand entry → source + upstream model + price), concurrency cap. Persisted locally (macOS: Keychain for secrets + config in app DB; others: one `0600` JSON file). |
| `FrameCodec` | Encode/decode tunnel frames (JSON). Transport-agnostic — used by both WebSocket and slow-poll transports. |
| `TunnelClient` | Outbound connection lifecycle: dial, authenticate, heartbeat with jitter, exponential-backoff reconnect with jitter, idle degradation to slow-poll after T minutes without traffic, immediate WS re-establishment on first polled request. |
| `Relay` | Request handling: map model → source, inject the source key, forward via AsyncHTTPClient (streaming), relay SSE chunks back, enforce per-request timeout and the peer-wide concurrency cap, propagate cancellation. |
| `LocalSource` | `local://novamlx` — macOS only. Calls NovaMLX `InferenceService` in-process (no HTTP loopback). On other platforms this source type is not offered. |
| `PeerService` | Glue + observable status (connection state, active requests, cumulative tokens) for both the Mac UI and CLI output. |

### Executable: `Sources/tknet-peer/`

- `tknet-peer setup` — interactive wizard: register with tknet.ai (returns peerId + token), fetch demand list, select entries, configure sources, test each source with a live request, save.
- `tknet-peer serve` — run the peer; prints status to stdout, suitable for systemd/launchd/NSSM.

### macOS integration

`TknetPeerPageView` in `NovaMLXMenuBar` — register/connect flow, demand list with checkboxes, source editor (with `local://novamlx` pre-filled option), live status. `PeerService` runs in the app process.

### Cross-platform build

Package.swift: MLX/UI targets guarded by platform conditions so Linux/Windows resolve only the peer product's dependency graph. `swift build --product tknet-peer` builds the peer alone on any platform. CI (Phase 2): GitHub Actions runners for linux-amd64 and windows-amd64 producing release binaries.

## Tunnel protocol (v1)

One outbound connection per peer: `wss://tknet.ai/api/peer/tunnel`, authenticated by the peer token from registration. JSON frames. Companion REST: `POST /api/peer/register` (issues peerId + token), `GET /api/peer/demand` (demand list).

| Direction | Frame | Payload |
|---|---|---|
| peer→server | `hello` | `{peerId, token, capabilities: [{demandId, model, sourceType, priceIn, priceOut}]}` |
| peer→server | `heartbeat` | `{activeReq, queueDepth, rolling averages}` — health + scheduling weight |
| peer→server | `response.chunk` | `{reqId, SSE chunk}` |
| peer→server | `response.end` | `{reqId, status, ttftMs, totalMs, promptTokens, completionTokens, upstreamStatus}` — billing + reputation feed |
| server→peer | `request` | `{reqId, model, apiFormat (openai\|anthropic), body}` |
| server→peer | `request.cancel` | `{reqId}` |
| server→peer | `demand.update` | Current demand list (added/removed/changed entries); the `hello` response also carries the full list for reconciliation after reconnect |
| peer→server | `capabilities.update` | Peer-side declaration changes (price, retire an entry, source swap) without re-registering |

**Demand lifecycle.** When tknet.ai retires a demand entry, the scheduler simply stops dispatching it (eligibility is computed live from demand list × capabilities; stale demandIds are ignored server-side). On the peer: the mapping is marked **retired** (greyed in UI) but **not deleted** — source configs are reusable assets. The Mac app notifies the operator that any model loaded solely for that demand can be unloaded, but never auto-unloads. New demand entries arrive on the same `demand.update` frame and surface as opportunities ("new demand X — one of your sources may serve it").

Forwarding rules: the peer strips all tknet.ai credentials and injects its own source key before contacting the source. WebSocket backpressure applies naturally. Every reconnect/heartbeat/poll interval carries jitter to prevent thundering herds at scale.

## Scaling (tens of thousands of peers)

- **Server shape (contract note):** tunnel gateway is a separate, horizontally scalable process; peerId→gateway assignment via consistent hashing; presence published to Redis. Main API never holds tunnels. The peer is unaware of this.
- **Idle degradation (v1):** WS while active; after 5 minutes idle (tunable), drop to 60 s slow-poll carrying the same frames through `FrameCodec`; any polled request triggers immediate WS re-establishment.
- **Peer protection:** operator-declared concurrency cap (default 1). The scheduler must not over-dispatch; declining work must be preferable to failing it (feeds reputation).

## Reputation & dispatch (tknet.ai side — contract only)

`score = w₁·latency + w₂·success + w₃·price (+ w₄·uptime)`, computed per peer×model.

- latency: EWMA of TTFT and tokens/s, normalized per model class (a 27B is not compared to a 4B).
- success: decayed ratio with Bayesian smoothing (low-volume new peers neither die on one failure nor top the board on three successes).
- price: operator-declared per-model price, normalized against the pool median. Reneging (timeout/abort after accepting) penalizes success heavily — undercutting then failing loses to slow-but-reliable.
- cold start: neutral score + ~5% exploration traffic; selection by softmax/ε-greedy over top-K eligible peers, not pure argmax.
- v1 anti-gaming: periodic real probe requests cross-check self-reported metrics.

## Security

- Peer token: macOS Keychain; other platforms `0600` config file.
- Source keys never leave the peer (no field for them exists in the protocol).
- TLS everywhere; tknet.ai credentials never forwarded to sources.
- Per-peer rate limits and quotas protect the operator's machine and keys.

## Testing

1. `FrameCodec` unit tests (pure, run on all platforms).
2. Relay integration tests: mock source (Hummingbird test server) + mock tunnel server; streaming pass-through, cancellation, timeout, concurrency cap, header hygiene.
3. Idle-degradation test: WS→poll→WS transition under a fake timer.
4. macOS UI smoke: Tknet page appears; register against mock server; status reaches connected.

## Phasing

- **Phase 1 (this repo):** `NovaMLXTknetPeer` module + full frame protocol (all telemetry fields) + `tknet-peer` CLI + Mac `TknetPeerPageView`; end-to-end against a mock tknet.ai server.
- **Phase 2 (this repo):** Linux/Windows CI builds; idle degradation transport.
- **Phase 3 (tknet.ai repo):** gateway clustering, reputation scheduler, probe service.

## Risks

- **Swift on Windows is the roughest edge.** The peer's dependency surface is minimal (WS + HTTP relay); worst case, a Go re-implementation of the peer preserves the protocol without touching this repo's architecture.
- **Latency is the product's death sentence.** Active peers must see zero added dispatch latency; any future "server savings" scheme must not introduce polling delay for active traffic.

## Reference

- Design dialogue: 2026-09-23/24 (WebSocket-vs-polling cost analysis; notification channel as the scarce resource; reputation requirements).
- SwiftMind map (status tracker): `~/Documents/SwiftMind/Tknet Peer Architecture.swiftmind.html`, peer ids `tkn_*` for CLI status updates.
