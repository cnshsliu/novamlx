# Regression + performance gate

**Pitch:** Mac-local correctness and decode tok/s gate so core models cannot silently regress after engine or Metal changes.

A small suite that loads one model at a time against a running NovaMLX, warms it, measures streaming `decode_tps`, checks a trivial correctness assertion, unloads, then compares against a committed baseline. Improvements are fine; drops beyond tolerance fail the run.

## Who it is for

Maintainers on Apple Silicon who want a fast gate before merging engine/Metal changes that touch Flint, Qwen, or Unlimited OCR.

## What it does

| Piece | Role |
|-------|------|
| `Scripts/regression-suite.yaml` | Core vs optional models, ports, auth fallback, tolerance |
| `Scripts/regression-perf-gate.py` | Driver: health → unload-all → load → warm → measure → correctness → unload |
| `Scripts/regression-baselines.json` | Per-model `decode_tps` baselines (`version` + `models`) |
| `Scripts/regression-perf-gate.sh` | Wrapper; defaults to `--suite core` |

### Suites

- **core** (default tok/s gate): `RecursiveSI/flint-int4`, `mlx-community/Qwen3-8B-4bit`, `baidu/Unlimited-OCR-2`
- **optional**: larger Qwen (`Qwen3.8-27B-8bit`, `Qwen3.6-35B-A3B-4bit`) and `pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit` — not part of the default gate
- **all**: core + optional

OCR uses the chat path with `document parsing.` plus an inline `image_url` data URL (tiny generated PNG fixture with the digits `391`).

### Auth and ports

- Inference `http://127.0.0.1:6590`, admin `http://127.0.0.1:6591` (overridable in the YAML)
- API key: `NOVA_API_KEY` → else enabled `primary` key in `~/.nova/nova_config.db` → else `abcd1234`

### Pass / fail

- **Correctness fail** → non-zero exit (expected substrings missing, empty OCR text, load/chat errors)
- **decode_tps regression** → non-zero when measured tok/s is more than **8%** below baseline (`tolerance.default`). `--strict` tightens to **5%**
- **Higher tok/s** always passes
- **No baseline yet** → warning only (unless correctness failed); seed with `--seed-baseline`

Results land in `~/.nova/regression/results-<gitsha>-<timestamp>.json`.

## How to try

NovaMLX must already be running on this Mac.

```bash
# Default core gate
./Scripts/regression-perf-gate.sh

# First time (or after intentional speedups): write baselines from this machine
./Scripts/regression-perf-gate.sh --seed-baseline

# Optional / all / strict
SUITE=optional ./Scripts/regression-perf-gate.sh
./Scripts/regression-perf-gate.sh --suite all
./Scripts/regression-perf-gate.sh --strict
```

Model weight paths in the YAML point at the maintainer Mac layout (`/Users/lucas/Models/...`, `/Volumes/Samsung768/Models/...`). Use `--skip-missing-path` to skip entries whose directory is absent.

## Notes

Patterned on `Scripts/post_reboot_8bit_bench.py` (same stream chat decode_tps math, same admin load/unload). This gate does not start NovaMLX and does not wait for Apple Silicon CI.
