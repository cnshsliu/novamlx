# NovaMLX — Agent Instructions

## Build & Deploy

**Always use `./build.sh`, never raw `swift build`.**

`./build.sh` is a thin wrapper around `swift build` that adds critical pre/post steps:

1. **Pre-build patches** — applies `mlx-swift` complex-number patch and `mlx-swift-lm` fused quantized SDPA patch to checked-out dependencies (these are required for the project to compile correctly).
2. **Metal shader compilation** — builds `mlx.metallib` from the MLX Metal sources if missing.
3. **Post-build dist sync** — keeps `dist/NovaMLX.app/Contents/MacOS/` in lockstep with `.build/<arch>/<config>/` by copying any binary whose Mach-O `LC_UUID` changed (`NovaMLX`, `NovaMLXWorker`, `nova`) and re-signing the bundle.

**Why this matters**: `NovaMLX` (host) launches `NovaMLXWorker` as a subprocess from the app bundle. Running raw `swift build` updates `.build/` but leaves the app bundle stale → the host silently runs old worker code (this was a real bug tracked as `todo.markdown` §2.7).

### Common invocations

```bash
./build.sh -c release          # release build + auto-sync to dist/
./build.sh -c debug            # debug build + auto-sync to dist/
./build.sh                     # default (debug)
NOVAMLX_SKIP_DIST_SYNC=1 ./build.sh -c release   # skip sync (CI / clean-room)
```

### After build

The host process must be restarted for the new worker to be spawned:

```bash
killall NovaMLX; sleep 2; open dist/NovaMLX.app
```

### First-time package

If `dist/NovaMLX.app` does not yet exist, run `Scripts/package.sh` to create the bundle + DMG + tarball.

## Tests

```bash
swift test                                 # full suite
swift test --filter <TestName>             # single test or pattern
```

Tests live under `Tests/NovaMLX*Tests/`.

## Logs & Config

- Runtime log: `~/.nova/novamlx.log`
- Config: `~/.nova/config.json`

## Project conventions

- Source: `Sources/NovaMLX{Core,Engine,Inference,API,Utils,MenuBar,ModelManager,...}/`
- Vendored deps: `vendors/mlx-swift/`, `vendors/mlx-swift-lm/` (treat as read-only — modifications go via `Scripts/patch-*.py`)
- Active diagnostic todos: `todo.markdown`
