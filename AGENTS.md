# Repository Guidelines

## Project Structure & Module Organization

The project is organized as a Swift 6.0 package with three executable targets and several library targets:

| Path | Purpose |
|------|---------|
| `Sources/NovaMLX{Core,Utils,Engine,…}/` | Library modules (see below) |
| `Sources/NovaMLXApp/` | Host process — menu bar app entry point |
| `Sources/NovaMLXWorker/` | Worker subprocess spawned by the host |
| `Sources/NovaMLXCLI/` | `nova` CLI tool |
| `Tests/NovaMLX{Core,Engine,Inference,…}Tests/` | Unit & integration tests |
| `Scripts/` | Build helpers, patchers, test runners, packaging |
| `vendors/mlx-swift/` | Vendored MLX Swift bindings (read-only) |
| `mlx-swift-lm/` | Local package dependency for MLX LLM / VLM support |
| `dist/NovaMLX.app/` | Packaged app bundle (auto-synced by `./build.sh`) |

Key library targets:

- **NovaMLXCore** — foundational types, version, logging setup
- **NovaMLXUtils** — helpers, resource bundle locator, Security framework integration
- **NovaMLXEngine** — MLX model loading, inference engine, chat templates
- **NovaMLXInference** — high-level inference orchestration, streaming, distributed pipelines
- **NovaMLXModelManager** — model loading/unloading, lifecycle
- **NovaMLXAPI** — HTTP API (Hummingbird-based, port 6591)
- **NovaMLXMCP** — MCP protocol support
- **NovaMLXDistributed** — multi-host distributed inference
- **NovaMLXPrefixCache** — KV-cache prefix caching
- **NovaMLXAudio** / **NovaMLXImage** — modality support
- **NovaMLXMenuBar** — macOS menu bar UI

## Build, Test, and Development Commands

**Always use `./build.sh`** instead of raw `swift build`. The wrapper applies dependency patches, compiles MLX Metal shaders, and syncs built binaries to the app bundle. See the script header for details.

```bash
# Build (debug is default)
./build.sh
./build.sh -c debug
./build.sh -c release

# Skip dist sync (CI / clean-room)
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh -c release

# Package the app bundle + DMG + tarball (first time or after major changes)
Scripts/package.sh

# Run tests
swift test
swift test --filter <TestName>

# Run E2E model tests (loads all downloaded models, runs 4 API tests each)
Scripts/test-all-models.sh

# Restart the host after rebuilding (so it spawns the new worker)
killall NovaMLX; sleep 2; open dist/NovaMLX.app
```

Runtime config lives at `~/.nova/config.json` and logs at `~/.nova/novamlx.log`. Log levels can be changed at runtime via the admin API (port 6591).

## Coding Style & Naming Conventions

- **Language**: Swift 6.0 with `StrictConcurrency` enabled for all targets.
- **Indentation**: 4 spaces per level. No tabs.
- **Naming**: Follow [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).
  - Types, protocols, enums: `UpperCamelCase`
  - Properties, methods, parameters: `lowerCamelCase`
  - Booleans: use `is`/`has` prefixes (`isLoaded`, `hasShards`)
- **Imports**: Group by (1) own modules, (2) MLX ecosystem, (3) third-party, (4) system frameworks. Sort within groups.
- **Concurrency**: Prefer `async`/`await` over completion handlers. Use `MainActor` for UI-bound code.
- **No formatter** configured — maintain consistency with the surrounding code.

## Testing Guidelines

- **Framework**: XCTest (SwiftPM default).
- **Test targets** mirror library targets: `NovaMLXCoreTests`, `NovaMLXEngineTests`, `NovaMLXInferenceTests`, etc.
- **Test naming**: `test<description>`, e.g. `testLoadsModelFromHub`, `testPrefillTokenBatch`.
- **E2E tests** (`NovaMLXE2ETests`) load real MLX models and exercise the HTTP API. Run with `Scripts/test-all-models.sh`.
- **Benchmarks** live in `NovaMLXBenchTests`.
- Metal shader support is hooked up automatically when running via `./build.sh test` (the script copies `mlx.metallib` into the test bundle).

```bash
swift test                          # full suite
swift test --filter testLoadsModel  # single test
./build.sh test --filter testLoadsModel  # includes metallib setup
```

## Commit & Pull Request Guidelines

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) with an optional scope:

```markdown
<type>(<scope>): <description>
```

Types used in this repo: `feat`, `fix`, `docs`, `refactor`, `chore`, `test`. Scopes reflect the affected module (e.g., `distributed`, `inference`, `api`, `engine`).

Examples from the project history:

```
feat(distributed): bfloat16 transport — 2x smaller tensors over TCP
fix(distributed): coordinator must skip head during prefill
refactor(engine): extract attention mask builder
```

Pull requests should:

- Have a descriptive title matching the commit style.
- Reference related issues or tracking items (e.g., `todo.markdown §2.7`).
- Include a summary of changes and any relevant design trade-offs.
- Keep changes focused on a single concern — prefer multiple small PRs over one large one.

## Architecture Overview

NovaMLX uses a **two-process architecture**:

1. **NovaMLX** (host) — macOS menu bar app that owns the HTTP API, model manager, and distributed coordinator.
2. **NovaMLXWorker** (subprocess) — spawned by the host, handles MLX inference in a separate process for isolation.

The host spawns the worker from the app bundle (`dist/NovaMLX.app/Contents/MacOS/NovaMLXWorker`), which is why `./build.sh` auto-syncs binaries after each build. If the app bundle falls out of sync, the host silently runs stale worker code.

Distributed inference uses a **coordinator-worker** model over TCP with sharded model layers, supporting both simple and speculative decoding pipelines.

## Security & Configuration

- **API authentication**: All HTTP endpoints require a `Bearer` token. The admin API (`/admin/api/*`) uses a separate key.
- **Configuration**: `~/.nova/config.json` — plain JSON, should not contain secrets in plaintext.
- **Runtime secrets**: Use the system Keychain (Security framework) via `NovaMLXUtils` rather than config files.
- **Sandboxing**: The worker process is spawned without entitlement inheritance — treat it as untrusted for security-sensitive data.

## Agent-Specific Instructions

- **Never modify** files under `vendors/` directly. Patches to `mlx-swift` go through `Scripts/patch-mlx-complex.py` and `Scripts/patch-fused-sdpa.py`.
- **`mlx-swift-lm/`** is a local package dependency (not a submodule or vendored copy) — edit directly if needed, but prefer contributing upstream.
- **Resource bundles** (`.bundle` dirs) produced by SPM must be copied into `dist/NovaMLX.app/Contents/Resources/` — the `ResourceBundleLocator` searches there at runtime.
- When adding a new library target, add a corresponding test target and update `Package.swift` with the `StrictConcurrency` swift setting.
