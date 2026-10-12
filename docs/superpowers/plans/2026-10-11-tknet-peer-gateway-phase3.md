# Tknet Peer Gateway — Segment 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a **Peer-only Mac menu-bar app** that owns members, pools, the local OpenAI gateway (`127.0.0.1:6600`), and the tknet tunnel toggle — without requiring NovaMLX to be open. Gateway listens on every launch; tunnel auto-dials unless the user previously pressed Stop (max 3 launch retries); after Stop, next launch keeps the port up but does not dial. One Peer process at a time. Reuse the existing peer identity and connection keys in `~/.config/tknet-peer/`.

**Architecture:** Keep all networking in `NovaMLXTknetPeer` (`PeerService` + `LocalGateway` + `TunnelClient`). Extend `PeerService` so the **gateway and tunnel have independent lifetimes** (today `stop()` tears both down — that breaks Segment 3 Stop semantics). Add a **process lock** shared by `tknet-peer serve` and the app. New SwiftUI executable target **`TknetPeerApp`** (product name for the bundle: `TknetPeer.app`) depends only on `NovaMLXTknetPeer` + SwiftUI/AppKit — **no MLX, no NovaMLXCore/DB/Engine/MenuBar**. Move the usable bits of `Sources/NovaMLXMenuBar/TknetPeerState.swift` into the Peer app (strip NovaMLXDB / NovaMLXUtils / `APIKeyStore` coupling). Delete the leftover state file from NovaMLXMenuBar once the app owns it.

**Tech Stack:** Swift 6 (StrictConcurrency), SwiftUI `MenuBarExtra` + optional settings window, Foundation, existing Peer library (Hummingbird / AsyncHTTPClient already linked). Build via `./build.sh`; Peer app product must stay MLX-free: `swift build -c release --product TknetPeerApp` (or the product name chosen in Task 4). Tests: Swift Testing in `NovaMLXTknetPeerTests` for service/lock/frame contracts; UI smoke is manual / checklist.

**Spec:** `docs/superpowers/specs/2026-10-11-tknet-peer-gateway-design.md`  
**Phase overview:** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway.md` (Segment 3 section — leave the four-segment outline intact)  
**Segment 1 plan:** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway-phase1.md`  
**Segment 2 plan:** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway-phase2.md`  
**Sources of truth:** `Sources/NovaMLXTknetPeer/PeerService.swift`, `TunnelClient.swift`, `LocalGateway.swift`, `Frames.swift`, `Sources/NovaMLXMenuBar/TknetPeerState.swift` (tunnelWanted / launch retries — adapt, do not keep as NovaMLX dependency), `Sources/NovaMLXApp/main.swift` (MenuBarExtra pattern), `Scripts/package.sh` (bundle layout)

## Global Constraints

- `NovaMLXTknetPeer` / `tknet-peer` / **`TknetPeerApp`** MUST NOT import: `MLX*`, `NovaMLXEngine`, `NovaMLXInference`, `NovaMLXImage`, `NovaMLXAudio`, `NovaMLXDB`, `NovaMLXCore`, `NovaMLXMenuBar`, `NovaMLXAPI`, `NovaMLXUtils` (Keychain helper lives in Utils — if the app needs Keychain, copy a tiny helper into the Peer app target or Peer library; do **not** pull Utils).
- Segment 4 (repo split / Linux-Windows packaging) is **out of scope**. Stay in this git repo.
- Reuse existing peer registration (`peerId` + connection key in SecretStore). Do **not** force re-registration. Do not change tknet website scheduling/billing.
- Secrets: never log keys; never put member endpoints or API keys in tunnel frames, status UI copy that is screenshotted for support is OK for **pool names** / connection state only — member **URLs and keys** stay off any frame and off any clipboard export unless the user explicitly copies from a member edit field.
- UI copy: claimed Tknet Peer is a **「token 工厂」** / "token factory", never "我的机器" / "my machine". Connection key = **连接密钥**, never "token" in user-facing Peer UI (collides with LLM billing language).
- Single Peer process: app and `tknet-peer serve` must refuse to start a second owner of the gateway/tunnel for the same config directory.
- Stop (UI) = **stop tunnel only**; gateway stays listening. Quit app = stop everything (gateway + tunnel) and release the process lock.
- All new library tests use Swift Testing. Prefer `./build.sh test --filter …`. Clean debug build: zero new `warning:` from Peer / PeerApp targets.
- Do not commit unrelated untracked local files (`speech.wav`, `next_claude`, `.scratch/`, etc.).
- Do not reintroduce Load Balancers / Tokenhub / Tknet Peer pages into NovaMLX.

### Key behavioural contracts (Segment 3)

| Case | Behaviour |
|---|---|
| App launch | Acquire process lock → start **gateway** on config `gatewayHost`/`gatewayPort` (default `127.0.0.1:6600`) |
| `tunnelWanted == true` and `peerId` present | After gateway is up, auto-dial tunnel; on launch-path failures, retry at most **3** times, then leave gateway up and stop dialing until next launch or manual Start |
| User presses **Stop** | Set `tunnelWanted = false`; stop tunnel; **gateway keeps listening** |
| User presses **Start** | Set `tunnelWanted = true`; dial tunnel (TunnelClient’s own reconnect loop applies after a successful connect; launch-retry cap is for the auto-dial-after-launch path) |
| Next launch after Stop | Gateway up; **no** auto-dial |
| User never pressed Stop (flag absent / true) | Auto-dial after launch (same as today’s NovaMLX `TknetPeerState` default) |
| Second Peer process (app or `serve`) | Fail fast with a clear message; do not steal the lock |
| Members / pools UI | Editable without NovaMLX running; persists to `~/.config/tknet-peer/peer.json` + SecretStore; live `applyConfig` when service is running |
| Tunnel frames (`hello`, `capabilitiesUpdate`, …) | Must not include member `endpoint`, `apiKeyRef`, or upstream URLs — only Capability fields already on the wire (`demandId`, `model`, `sourceId` as **pool id**, prices, optional `api` contract) |

### Defaults for open questions (implement these unless Lucas overrides)

1. **UI scope this segment:** Members list + add/edit/delete, Pools list + add/edit/delete (strategy, members, retries, enabled), Tunnel Start/Stop + connection status + gateway listening badge, Register / show peer id + connection key (needed to dial), Quit. **Defer** earnings card, multi-server switcher polish, demand-marketplace binding UI, and account claim flow to a follow-up **unless** they already move cheaply with `TknetPeerState` — if ported, keep them behind a secondary “Account” section, not blocking acceptance.
2. **Secrets on Mac:** Keep **`FileSecretStore`** under `~/.config/tknet-peer/secrets` so the app and CLI share one identity without a Keychain migration. Document Keychain as a future hardening step (design mentioned Keychain; shared file store already works in production with CLI).
3. **Process lock path:** `~/.config/tknet-peer/peer.lock` (flock / PID file). Same path for CLI `serve` and the app.
4. **Bundle:** `dist/TknetPeer.app`, bundle id `ai.tknet.peer`, `LSUIElement` / menu-bar agent style like NovaMLX, menu title **Tknet Peer** (or 「token 工厂」 subtitle in the window — menu bar label stays short: `Tknet Peer`).
5. **NovaMLX leftover:** Delete `Sources/NovaMLXMenuBar/TknetPeerState.swift` after the Peer app owns equivalent state (or leave a one-line `#error` / remove from target). Do not leave a second auto-tunnel owner.

---

### Task 1: PeerService — independent gateway vs tunnel lifetime

**Files:**
- Modify: `Sources/NovaMLXTknetPeer/PeerService.swift`
- Modify: `Tests/NovaMLXTknetPeerTests/PeerServiceTests.swift` (extend)
- Create if needed: `Tests/NovaMLXTknetPeerTests/PeerServiceTunnelControlTests.swift`

**Interfaces (add):**
- `start(tunnel: Bool = true)` — keep; when `tunnel == false`, gateway only (already true).
- `startTunnel() async throws` — dial while gateway already running; no-op / idempotent if tunnel already up; error if gateway not started or no `peerId`.
- `stopTunnel() async` — tear down `TunnelClient` + related tasks; **leave** `LocalGateway` + `PoolForwarder` running; publish status with `connection: .idle`, `gatewayListening: true`.
- `stop() async` — full teardown (gateway + tunnel), as today; used on app Quit / CLI exit.
- Status: ensure `PeerStatus.gatewayListening` / `gatewayPort` remain accurate across `stopTunnel`.

**Launch-retry policy (library-side optional helper, or app-owned):** Prefer keeping the **3-retry abandon** in the **app state** (port of `TknetPeerState.noteLaunchConnection`), but `stopTunnel` must **not** call full `stop()`. Fix today’s bug relative to Segment 3: `abandonLaunch` → `tearDown` currently kills everything — new path must call `stopTunnel` only.

- [ ] **Step 1: Failing tests**

```swift
@Test("stopTunnel leaves gateway listening")
func stopTunnelKeepsGateway() async throws {
    // start(tunnel: false) with mock factory unused
    // assert gatewayListening
    // startTunnel with MockTunnel transport that connects
    // stopTunnel()
    // assert gatewayListening == true, connection == .idle
    // curl or LocalGateway still answers /v1/models
}

@Test("startTunnel after gateway-only start dials once")
func startTunnelAfterGateway() async throws { /* … */ }

@Test("full stop shuts gateway")
func fullStopShutsGateway() async throws { /* … */ }
```

- [ ] **Step 2: Implement `startTunnel` / `stopTunnel`** without breaking existing `PeerServiceTests` / CLI `serve`.

Refactor `start(tunnel:)` so tunnel install is shared with `startTunnel()`. `detachTunnel()` separate from `detach()` (full).

- [ ] **Step 3: `./build.sh test --filter 'PeerService'`**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(peer): startTunnel/stopTunnel keep local gateway up"
```

---

### Task 2: Single-process lock

**Files:**
- Create: `Sources/NovaMLXTknetPeer/PeerProcessLock.swift`
- Create: `Tests/NovaMLXTknetPeerTests/PeerProcessLockTests.swift`
- Modify: `Sources/tknet-peer/TknetPeerCLI.swift` (`serve` acquires lock before `PeerService.start`)
- Modify: Peer app launch (Task 5) to acquire the same lock

**Behaviour:**
- Lock file default: `~/.config/tknet-peer/peer.lock` (overridable for tests via URL).
- Exclusive non-blocking acquire; on failure throw `PeerProcessLockError.alreadyRunning(ownerHint:)` with pid if available.
- Hold for process lifetime; release on `stop()` / deinit / CLI exit (defer / `atexit` / explicit `unlock` in `PeerService.stop` or CLI defer).
- Stale lock: if PID is dead, allow steal (document Linux/macOS `kill(pid,0)` check). Do not steal a live PID.

- [ ] **Step 1: Unit tests** — two acquires on same path → second fails; unlock then succeed; dead-PID recovery if easy to simulate.

- [ ] **Step 2: Wire CLI `serve`** — fail with a one-line stderr if app already holds the lock.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer): process lock so app and serve cannot both own the gateway"
```

---

### Task 3: Frame hygiene — no member addresses on the wire

**Files:**
- Modify / extend: `Tests/NovaMLXTknetPeerTests/FrameCodecTests.swift` or new `TunnelFrameHygieneTests.swift`
- Audit: `Sources/NovaMLXTknetPeer/Frames.swift` `Capability`, `TunnelClient` hello path, `Relay` (must not send endpoints upstream to tknet)

**Acceptance test idea:** Build a `PeerConfig` with a member endpoint `http://10.0.0.9:8080/v1` and secret ref; encode `hello` / `capabilitiesUpdate`; assert the JSON/binary payload **does not contain** `10.0.0.9`, `8080`, or the secret value. `sourceId` may equal a pool id string (opaque).

- [ ] **Step 1: Write failing/affirming hygiene tests**

- [ ] **Step 2: Fix any leak if found** (should already be clean — this is a guardrail)

- [ ] **Step 3: Commit**

```bash
git commit -m "test(peer): tunnel frames must not include member endpoints or secrets"
```

---

### Task 4: Package.swift — `TknetPeerApp` product (MLX-free)

**Files:**
- Modify: `Package.swift`
- Create: `Sources/TknetPeerApp/main.swift` (stub `@main` MenuBarExtra that compiles)
- Optional: `Sources/TknetPeerUI/` library target if you want views testable separately — **not required**; a single executable target is fine for Segment 3

**Product:**
```swift
.executable(name: "TknetPeerApp", targets: ["TknetPeerApp"])
```

**Target deps:** `NovaMLXTknetPeer` only (+ system SwiftUI/AppKit). Linker: no MLX metallib.

- [ ] **Step 1: Add target + stub main that shows MenuBarExtra("Tknet Peer") { Button("Quit") { NSApp.terminate(nil) } }`**

- [ ] **Step 2: `swift build -c release --product TknetPeerApp`** and confirm link map / `nm` / `.build` object list has **no** `MLX`, `NovaMLXCore`, `NovaMLXDB`, `GRDB`, `NovaMLXMenuBar`.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer): add MLX-free TknetPeerApp executable target"
```

---

### Task 5: PeerAppState — config, gateway-on-launch, tunnelWanted

**Files:**
- Create: `Sources/TknetPeerApp/PeerAppState.swift` (port from `TknetPeerState`, strip NovaMLX deps)
- Create: `Sources/TknetPeerApp/PeerAppDelegate.swift` or fold into `main.swift`
- Config path: `~/.config/tknet-peer/peer.json` + `FileSecretStore` (same as CLI)
- Preference key: keep `tknetPeer.tunnelWanted` in `UserDefaults` for continuity with prior NovaMLX installs **or** move to a file under the config dir (prefer **UserDefaults key unchanged** so users who already pressed Stop keep that preference)

**Launch sequence:**
1. Acquire `PeerProcessLock`
2. Load config / migrate legacy pools if needed
3. Construct `PeerService` with `WSTransport.factory` when registered
4. `try await service.start(tunnel: false)` — gateway always
5. If `tunnelWanted && peerId != nil && token present` → `startTunnel()` with launch-retry watcher (max 3), on abandon → `stopTunnel()` only + set error string (do **not** clear `tunnelWanted` on retry exhaustion — only Stop clears it; next launch may try again). **Confirm with design:** design says “失败最多再试 3 次，然后停到下次启动或用户再按开始” — so retry exhaustion does **not** flip `tunnelWanted` to false; only Stop does.

**Quit:** `stop()` full + unlock.

- [ ] **Step 1: Port state machine; unit-test pure helpers if extracted** (tunnelWanted read/write, retry counter) in Peer tests or a tiny non-UI type in the Peer library: `TunnelLaunchPolicy`

Recommended extract to library for testability:
```swift
public struct TunnelLaunchPolicy: Sendable {
    public static let maxLaunchRetries = 3
    public mutating func note(_ status: TunnelStatus) -> TunnelLaunchDecision
}
public enum TunnelLaunchDecision: Sendable { case keepWatching, connected, abandon, upgradeRequired }
```

- [ ] **Step 2: Wire app delegate launch / terminate**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer-app): gateway on launch + tunnelWanted auto-dial policy"
```

---

### Task 6: Members + Pools UI

**Files:**
- Create: `Sources/TknetPeerApp/MembersView.swift`
- Create: `Sources/TknetPeerApp/PoolsView.swift`
- Create: `Sources/TknetPeerApp/PeerRootView.swift` (window or popover content)
- Modify: `main.swift` — MenuBarExtra items: Open… / Tunnel Start|Stop / Status / Quit; open a `Window` or `Settings` scene for members/pools

**Minimum fields:**
- Member: id, name, type (`openaiCompatible` / `localNovaMLX`), endpoint URL, upstream model, api key (write-only to SecretStore via `apiKeyRef`), weight, enabled, free/paid
- Pool: id, name (OpenAI `model` string clients use), strategy picker (five strategies), member multi-select, maxRetries, enabled

Edits call `PeerAppState.commit` → `ConfigStore.save` + `service.applyConfig`.

- [ ] **Step 1: Build CRUD forms** (SwiftUI; reuse Nova theme only if you **copy** colors — do not import `NovaTheme`)

- [ ] **Step 2: Manual smoke** — edit pool without NovaMLX running; `curl http://127.0.0.1:6600/v1/models` lists pool names

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer-app): members and pools CRUD without NovaMLX"
```

---

### Task 7: Tunnel toggle + connection status UI

**Files:**
- Create: `Sources/TknetPeerApp/TunnelStatusView.swift`
- Modify: MenuBarExtra label or submenu to show connection glyph (idle / connecting / connected / backingOff / upgradeRequired)

**Controls:**
- Start → `setTunnelWanted(true)` + `startTunnel()`
- Stop → `setTunnelWanted(false)` + `stopTunnel()` (gateway badge stays green)
- Show: peer id (if any), gateway `host:port`, last error, active request count (from `PeerStatus`)
- Register affordance: if no `peerId`, prompt to run registration (thin UI wrapping `TknetREST.register` — port from `TknetPeerState.register`) **or** deep-link copy instructions to `tknet-peer setup`. Prefer in-app register so users are not forced to CLI.

- [ ] **Step 1: Wire Start/Stop to Task 1 APIs**

- [ ] **Step 2: Verify Stop then Quit then relaunch → gateway up, no dial**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer-app): tunnel Start/Stop with connection status"
```

---

### Task 8: Packaging `dist/TknetPeer.app`

**Files:**
- Create: `Scripts/package-peer.sh` (mirror `Scripts/package.sh` but only Peer binary, no Worker/MLX metallib)
- Modify: `build.sh` post-build sync **optionally** — either always sync Peer app when product exists, or document `Scripts/package-peer.sh` as the Segment 3 path (prefer a small sync block so `./build.sh` leaves a runnable `dist/TknetPeer.app`)

**Info.plist essentials:**
- `CFBundleIdentifier` = `ai.tknet.peer`
- `CFBundleName` / display name = `Tknet Peer`
- `LSUIElement` = true (agent / menu-bar)
- `CFBundleExecutable` = `TknetPeerApp`

Ad-hoc or existing Developer ID sign same as NovaMLX if `build.sh` already signs — reuse identity when present; otherwise unsigned local debug is OK for Lucas’s machine.

- [ ] **Step 1: package script + open `dist/TknetPeer.app`**

- [ ] **Step 2: Confirm menu bar icon appears with NovaMLX **quit**; members visible

- [ ] **Step 3: Commit**

```bash
git commit -m "build: package dist/TknetPeer.app menu-bar agent"
```

---

### Task 9: Remove Peer ownership from NovaMLX

**Files:**
- Delete or gut: `Sources/NovaMLXMenuBar/TknetPeerState.swift`
- Grep purge: any remaining `TknetPeerState` / `restoreTunnelAfterLaunch` references in `MenuBarAppState` / `main.swift` / Settings (Segment 2 already removed pages and auto-connect — verify none remain)
- Settings copy may keep the pointer to Peer app / `tknet-peer serve`

- [ ] **Step 1: `rg TknetPeerState` → zero references outside Peer app / docs**

- [ ] **Step 2: `./build.sh -c debug` NovaMLX still builds**

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: drop NovaMLX-embedded TknetPeerState; Peer app owns the tunnel"
```

---

### Task 10: Acceptance suite + verification gate

**Automated:**
- PeerService tunnel/gateway split tests (Task 1)
- Process lock tests (Task 2)
- Frame hygiene tests (Task 3)
- Full `./build.sh test --filter NovaMLXTknetPeerTests`
- `swift build -c release --product tknet-peer`
- `swift build -c release --product TknetPeerApp` — MLX-free

**Manual (Lucas Mac):**
1. Quit NovaMLX and any `tknet-peer serve`
2. Open `dist/TknetPeer.app` — gateway listens on 6600; members/pools from migrated config visible
3. With `tunnelWanted` true and registration present: tunnel connects (or fails ≤3 retries without killing :6600)
4. Press Stop — tunnel down, `curl :6600/v1/models` still works
5. Quit + relaunch — gateway up, no auto-dial
6. Press Start — tunnel dials
7. Attempt `tknet-peer serve` while app runs — lock error
8. Optional: with NovaMLX on 6590 and a local member, completions via Peer match direct (regression from Segment 2)

- [ ] **Step 1: Automated filters green**

- [ ] **Step 2: Manual checklist documented in commit body**

- [ ] **Step 3: Commit**

```bash
git commit -m "test(peer): Segment 3 acceptance — menu-bar app, tunnel policy, process lock"
```

- [ ] **Step 4: Stop** — do not start Segment 4 (repo split) in this plan

---

## Acceptance criteria mapping

| Design / overview criterion | Covered by |
|---|---|
| Peer-only menu-bar app: members, pools, tunnel toggle, connection status | Tasks 4–7 |
| Local port open on launch | Tasks 1, 5 |
| Auto-dial unless user previously stopped; max 3 launch retries | Tasks 5, 7 |
| After Stop, next launch no auto-dial; port still up | Tasks 1, 5, 7, 10 |
| Reuse existing peer identity / keys | Task 5 (same config dir + FileSecretStore) |
| Single Peer process | Task 2 |
| View members/pools without NovaMLX | Tasks 6, 9, 10 |
| Tunnel frames never expose member addresses | Task 3 |
| `tknet-peer` stays MLX-free; Peer app MLX-free | Tasks 4, 10 |

## Out of scope (do not implement in Segment 3)

- Extracting Peer to a separate git repository / Windows-Linux GUI (Segment 4)
- Active health probes, Anthropic-native upstream, embeddings/image/audio routes on the gateway
- Re-registering peer identity; changing tknet website billing
- Re-adding LB/Tokenhub pages to NovaMLX
- Full Keychain migration for secrets (default remains FileSecretStore shared with CLI)
- Polished earnings / multi-server / claim-account UX (optional stretch only)

## Ambiguities / risks for Lucas

1. **Stop vs Quit.** Design is clear that Stop leaves :6600 up. Confirm Quit should fully stop gateway (recommended) rather than leave an orphan listener without UI.

2. **Retry exhaustion and `tunnelWanted`.** Plan does **not** clear `tunnelWanted` after 3 failed launch retries (so the next cold start tries again). Only Stop clears it. Confirm that matches your intent (“停到下次启动或用户再按开始”).

3. **Keychain vs file secrets.** Design text mentioned Keychain for the Mac app; CLI already uses files. Plan keeps **FileSecretStore** for one shared identity. Confirm OK, or require Keychain + migration before merge.

4. **UI breadth.** Spec lists members / pools / tunnel / status. Old Mac page also had earnings, claim, demand binding, multi-server. Plan ships the spec minimum first; say if claim/earnings must be in the same PR.

5. **App name / icon.** Using `Tknet Peer` / `ai.tknet.peer`. Prefer a custom icon later; Segment 3 can ship system symbol in MenuBarExtra.

6. **`sourceId` on the wire.** Still sent as the **pool id** (opaque). If tknet ever treated `sourceId` as a hostname, that would be a protocol issue outside this segment — hygiene tests only guard against member URLs/secrets.

7. **NovaMLX and Peer both in menu bar.** Expected. Peer must not auto-start a second tunnel from NovaMLX (Task 9). Users may run both apps; only Peer owns :6600.

8. **Code signing / notarization.** Local `dist/TknetPeer.app` may be ad-hoc signed. Notarization can follow NovaMLX’s existing pipeline in a later chore.

9. **`TknetPeerState` local-source helper used `APIKeyStore`.** Peer app must not import NovaMLX API keys; local members can use empty key or a key pasted into the member form (stored under `apiKeyRef`).
