# Tknet Peer Gateway — Segment 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Leave this Mac / this NovaMLX checkout. Publish an **MLX-free** `tknet-peer` CLI whose library, command line, and tests build on a machine that does not have NovaMLX source. Linux and Windows get the **CLI first**. The Mac menu-bar app stays in novamlx and keeps working. No Windows or Linux GUI.

**Architecture:** Move `Sources/NovaMLXTknetPeer`, `Sources/tknet-peer`, and `Tests/NovaMLXTknetPeerTests` into a standalone Swift package at `tknet-peer/` (the tree that becomes `github.com/cnshsliu/tknet-peer`). Rename the library module **`NovaMLXTknetPeer` → `TknetPeer`** so the published package does not pretend to be a NovaMLX target. novamlx depends on that directory with `.package(path: "tknet-peer")` and links it only from **`TknetPeerApp`**. This agent cannot create the GitHub repository (`gh` is read-only here), so the split ships as an extractable package plus `Scripts/publish-repo.sh`. After the remote exists, novamlx can switch the path dependency to a URL pin; until then the path dependency is what keeps `dist/TknetPeer.app` building.

**Tech Stack:** Swift 6 (StrictConcurrency), Foundation, Hummingbird, AsyncHTTPClient, ArgumentParser, Swift Testing. CLI build: `swift build -c release --product tknet-peer` **inside `tknet-peer/`** (or `./build.sh peer -c release` from novamlx, which only invokes that package). Mac app: `./build.sh` still builds `TknetPeerApp` and `Scripts/package-peer.sh` still writes `dist/TknetPeer.app`.

**Spec:** `docs/superpowers/specs/2026-10-11-tknet-peer-gateway-design.md` (Mac first, then Windows/Linux; no Win/Linux GUI this phase)  
**Phase overview:** `docs/superpowers/plans/2026-10-11-tknet-peer-gateway.md` — section 「第四段：离开这台 Mac」. Leave that four-segment outline intact.  
**What shipped before this segment:**  
- Segment 1 — members, pools, `:6600` gateway, `tknet-peer` CRUD (`phase1` plan)  
- Segment 2 — Nova LB/Tokenhub migration, NovaMLX `lb:` / `tknet:` 404 (`phase2` plan)  
- Segment 3 — `TknetPeerApp` menu-bar agent, process lock, tunnel Start/Stop (`phase3` plan, merged as PR #6)

## What moves vs what stays

| Piece | Segment 4 |
|---|---|
| Library (`PeerService`, gateway, tunnel, pools, migrator, process lock) | **Moves** to `tknet-peer/Sources/TknetPeer`. Module name `TknetPeer`. |
| `tknet-peer` CLI | **Moves** to `tknet-peer/Sources/tknet-peer`. |
| Peer tests | **Move** to `tknet-peer/Tests/TknetPeerTests`. |
| `Sources/TknetPeerApp` + `Scripts/package-peer.sh` + `dist/TknetPeer.app` | **Stay** in novamlx. Mac-only. Depend on the local package. |
| NovaMLX app, MLX, NovaMLXCore, NovaMLXDB, menu-bar NovaMLX UI | **Stay.** No new dependency from those targets onto Peer. |
| Earnings / multi-server / claim UX, demand-marketplace UI, Keychain migration, notarization, Win/Linux GUI | **Not this segment** (Segment 3 leftovers / design “later”). |

Product rules that already shipped and must survive the move:

- Sell **pools**, not members. Tunnel `Capability.sourceId` is the **pool id**.
- UI copy: **token 工厂** / token factory. Never “我的机器” / “my machine”.
- One pool per published model name (gateway looks up `model` as a pool name).

## Global Constraints

- The `tknet-peer` package MUST NOT import or depend on: `MLX*`, `NovaMLXEngine`, `NovaMLXInference`, `NovaMLXImage`, `NovaMLXAudio`, `NovaMLXDB`, `NovaMLXCore`, `NovaMLXMenuBar`, `NovaMLXUtils`, `GRDB`, `SwiftUI` / `AppKit`. Allowed: Foundation, Hummingbird, HummingbirdRouter, HummingbirdWebSocket, AsyncHTTPClient, Logging, ArgumentParser (CLI target only).
- A clean copy of **only** the `tknet-peer/` directory must `swift build -c release --product tknet-peer` and `swift test` with no NovaMLX checkout beside it.
- `TknetPeerApp` stays MLX-free: its only package dependency is `TknetPeer`. Do not break `./build.sh` packaging of `dist/TknetPeer.app` on Mac.
- Do not re-register peers, do not copy inference into Peer, do not change tknet website scheduling or billing.
- Secrets stay in `FileSecretStore`. No Keychain migration in this segment.
- Linux CLI is the acceptance platform for this agent. Windows gets a real lock implementation (`LockFileEx`) and a CI job, but that job may be partial (`continue-on-error`) until a Windows Swift toolchain is proven. No Win/Linux GUI target.
- New or moved tests stay Swift Testing. Do not commit unrelated untracked files (`vendors/`, `mlx-swift-lm/`, `speech.wav`, `.scratch/`).
- Do not reintroduce Load Balancers / Tokenhub / Tknet Peer pages into NovaMLX.

### Key behavioural contracts (unchanged by the split)

| Case | Behaviour |
|---|---|
| `tknet-peer serve --no-tunnel` | Gateway on `127.0.0.1:6600` (or `--listen-port`), no tknet dial |
| `POST /v1/chat/completions` with `model` = pool **name** | Forward to a member; rewrite `model` to `upstreamModel` |
| Unknown pool | HTTP 404 |
| Build graph of the standalone package | No MLX, NovaMLXCore, NovaMLXDB, GRDB, SwiftUI |
| Second Peer process | Process lock still refuses (`peer.lock` beside the config) |
| Mac `TknetPeer.app` | Still opens from novamlx; same `~/.config/tknet-peer` as the CLI |

---

### Task 1: Standalone package layout

**Files:**
- Create: `tknet-peer/Package.swift`, `tknet-peer/README.md`, `tknet-peer/.gitignore`, `tknet-peer/LICENSE` (Apache-2.0, same as novamlx)
- Move: `Sources/NovaMLXTknetPeer/` → `tknet-peer/Sources/TknetPeer/`
- Move: `Sources/tknet-peer/` → `tknet-peer/Sources/tknet-peer/`
- Move: `Tests/NovaMLXTknetPeerTests/` → `tknet-peer/Tests/TknetPeerTests/`
- Modify: every `import NovaMLXTknetPeer` / `@testable import NovaMLXTknetPeer` → `TknetPeer`

**Package products:** library `TknetPeer`, executable `tknet-peer`. Tools version `6.0`. Same Hummingbird / AsyncHTTPClient / swift-log / ArgumentParser lower bounds as the old novamlx targets. `StrictConcurrency` on every target.

- [ ] **Step 1: Move sources and rename the module**

- [ ] **Step 2: `swift build -c release --package-path tknet-peer --product tknet-peer` resolves and links**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer): extract TknetPeer library and tknet-peer CLI into a standalone package"
```

---

### Task 2: novamlx keeps the Mac app, drops the in-tree Peer targets

**Files:**
- Modify: `Package.swift` — remove targets `NovaMLXTknetPeer`, `tknet-peer`, `NovaMLXTknetPeerTests` and their products. Add `.package(path: "tknet-peer")`. `TknetPeerApp` depends on `.product(name: "TknetPeer", package: "tknet-peer")`.
- Modify: `Sources/TknetPeerApp/*.swift` imports.
- Modify: `Scripts/package-peer.sh` version grep path → `tknet-peer/Sources/TknetPeer/Version.swift`.
- Modify: `build.sh` — `./build.sh peer` and `./build.sh test-peer` run the standalone package and return before Metal/MLX steps. Default `./build.sh` still builds `TknetPeerApp` via the root package.
- Modify: `README.md` Tknet Peer section so it no longer says the CLI is `swift build --product tknet-peer` from the NovaMLX root, and no longer points at a removed in-app Tknet Peer page.
- Modify: `AGENTS.md` module table so the library lives under `tknet-peer/`.

- [ ] **Step 1: Root manifest no longer lists Peer library/CLI/test targets**

- [ ] **Step 2: `TknetPeerApp` imports `TknetPeer`**

- [ ] **Step 3: Commit**

```bash
git commit -m "build(peer): novamlx links TknetPeer.app to the standalone package"
```

---

### Task 3: Linux + Windows CLI packaging

**Files:**
- Create: `tknet-peer/Scripts/package-cli.sh` — release build, tarball `dist/tknet-peer-<os>-<arch>.tar.gz` (binary name `tknet-peer` or `tknet-peer.exe`).
- Create: `tknet-peer/Scripts/check-mlx-free.sh` — `swift package show-dependencies` identities must not contain `mlx`, `novamlx`, `grdb`.
- Create: `tknet-peer/Scripts/accept-lan-pool.sh` — fake OpenAI server on a LAN (or loopback if this host has no LAN address) address, one member, one pool, `serve --no-tunnel`, `curl` the pool name, assert the upstream saw `upstreamModel`.
- Create: `tknet-peer/Scripts/verify-standalone.sh` — copy the package to an empty temp directory (no NovaMLX sources), then build, test, MLX-free check, and LAN acceptance.
- Create: `tknet-peer/Scripts/publish-repo.sh` — `git subtree split --prefix=tknet-peer` and the push command for `github.com/cnshsliu/tknet-peer`. Does not create the GitHub repo.
- Create: `tknet-peer/.github/workflows/cli.yml` — workflow that runs when this directory **is** the repository root.
- Create: `.github/workflows/tknet-peer-cli.yml` — same jobs against `--package-path tknet-peer` so novamlx CI does not need MLX or `vendors/`. Ubuntu job is required. Windows job builds the CLI and may `continue-on-error: true`.
- Modify: `Sources/TknetPeer/PeerProcessLock.swift` — `#if os(Windows)` exclusive lock via `LockFileEx`, same public behaviour (second acquire fails, dead pid replaced, live foreign pid not stolen). POSIX path unchanged.
- Modify: CLI `terminate` / shutdown signal so the executable target has a Windows exit path (`WinSDK`), plus Musl next to Glibc.

- [ ] **Step 1: Windows lock is behind `os(Windows)` and does not change the Linux/macOS flock path**

- [ ] **Step 2: Packaging + acceptance scripts are executable and documented in `tknet-peer/README.md`**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(peer): Linux and Windows CLI packaging without MLX"
```

---

### Task 4: Acceptance — build without NovaMLX, route a pool at a LAN inference service

**Automated (this agent, Linux):**

```bash
bash tknet-peer/Scripts/verify-standalone.sh
```

That script:

1. Copies only `tknet-peer/` to a temp directory.
2. `swift build -c release --product tknet-peer`
3. `swift test`
4. `Scripts/check-mlx-free.sh`
5. `Scripts/accept-lan-pool.sh` — one pool, one HTTP member on a non-loopback address when the host has one, completion comes back through the gateway.

**Not run here:** `./build.sh` / `TknetPeerApp` (needs macOS + Metal). The root `Package.swift` change is reviewable: `TknetPeerApp`’s only dependency is the local `TknetPeer` product, and `Scripts/package-peer.sh` still assembles `dist/TknetPeer.app`.

**Manual (Lucas’s Mac, after merge):**

1. `./build.sh -c release` — `dist/TknetPeer.app` still launches, gateway on :6600, existing config still loads.
2. `./build.sh peer -c release` — produces `tknet-peer` with no MLX link.
3. Quit the app, run the CLI against the same config, confirm the process lock still conflicts if both are up.

- [ ] **Step 1: `verify-standalone.sh` passes on Linux**

- [ ] **Step 2: Stop.** Do not start earnings UI, Keychain, notarization, or a Win/Linux GUI.

---

## Acceptance criteria mapping

| Overview 「第四段」 criterion | Covered by |
|---|---|
| Peer library, CLI, and tests live in a separate tree | Tasks 1–2 |
| MLX-free `tknet-peer` CLI | Tasks 1, 3, 4 |
| Linux CLI first, Windows CLI path (GUI later) | Task 3 |
| Build on a machine with no NovaMLX source | Task 4 `verify-standalone.sh` |
| Route one pool to a LAN inference service | Task 3 `accept-lan-pool.sh`, Task 4 |
| Mac menu-bar app still works from novamlx | Task 2 |

## Out of scope (do not implement in Segment 4)

- Creating `github.com/cnshsliu/tknet-peer` from this agent (no repo-create permission). Publish script only.
- Switching novamlx from the path dependency to a URL dependency before that remote exists.
- Windows or Linux GUI, `TknetPeer.app` port, notarization.
- Earnings / multi-server / claim-account UX, demand-marketplace binding UI.
- Full Keychain migration (Mac app and CLI keep `FileSecretStore`).
- Re-registering peer identity. Copying MLX or Nova inference into the package.
- Changing tknet website scheduling or billing.
- Deleting NovaDB load-balancer / Tokenhub tables.

## Ambiguities / risks for Lucas

1. **Module rename.** Call sites inside the package and `TknetPeerApp` now `import TknetPeer`. Nothing else in novamlx imported `NovaMLXTknetPeer` after Segment 3 (MenuBar no longer links it). External forks that imported the old module name must update.

2. **Path dependency until the GitHub repo exists.** `TknetPeer.app` builds only while `tknet-peer/` is inside the novamlx checkout. `Scripts/publish-repo.sh` is the cut. After the first push, a follow-up can replace `.package(path:)` with `.package(url: "https://github.com/cnshsliu/tknet-peer", from: "0.2.0")` and delete the vendored tree. Not in this PR, because the remote is not there yet.

3. **Windows CI is partial.** The workflow builds `tknet-peer` on `windows-latest` with `continue-on-error`. The lock uses `LockFileEx`, but this agent cannot compile that branch. Linux flock behaviour is what the existing process-lock tests cover.

4. **Two `.build` directories.** Root `./build.sh` compiles `TknetPeer` into novamlx’s `.build` for the app. `./build.sh peer` compiles the CLI into `tknet-peer/.build`. That avoids pulling MLX into the CLI build. It does compile the library twice if you run both.

5. **LAN acceptance on a host with no routable IPv4.** The script uses the source address of the default route. If that fails, it binds `127.0.0.1` and says so. The forward path is the same; only the member URL’s host changes.

6. **`migrate-from-novamlx` stays in the CLI.** It shells out to `sqlite3` and does not link NovaMLXDB. A standalone checkout can still migrate a copied `nova_config.db`. It does not need NovaMLX source.

7. **Nested GitHub workflow.** `tknet-peer/.github/workflows/cli.yml` does not run inside the novamlx repository (Actions only reads the root `.github/workflows`). The root workflow `.github/workflows/tknet-peer-cli.yml` is the one that runs here. The nested file is for the day the prefix is its own repo.
