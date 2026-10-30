# Regression + performance gate

**Pitch:** Mac-local correctness and decode tok/s gate so core models cannot silently regress after engine or Metal changes.

A small suite that loads one model at a time against a running NovaMLX, warms it, measures `decode_tps` (stream for measure; non-stream warm/correctness), checks a trivial correctness assertion, unloads, then compares against a committed baseline. Improvements are fine; drops beyond tolerance fail the run.

## Who it is for

Maintainers on Apple Silicon who want a fast gate before merging engine/Metal changes that touch Flint, Qwen, or Unlimited OCR.

## What it does

| Piece | Role |
|-------|------|
| `Scripts/regression-suite.yaml` | Core vs optional models, ports, auth fallback, tolerance |
| `Scripts/regression-perf-gate.py` | Driver: health → unload loaded-only → load → ready → warm → measure → correctness → unload |
| `Scripts/regression-baselines.json` | Per-model `decode_tps` baselines (`version` + `models`) |
| `Scripts/regression-perf-gate.sh` | Wrapper; picks PATH/Homebrew `python3` with PyYAML; defaults to `--suite core` |

### Suites

- **core** (default tok/s gate): `RecursiveSI/flint-int4`, `mlx-community/Qwen3-8B-4bit`, `baidu/Unlimited-OCR-2`
- **optional**: larger Qwen (`Qwen3.8-27B-8bit`, `Qwen3.6-35B-A3B-4bit`) and `pipenetwork/GLM-5.3-Flash-REAP50-MLX-mixed-4_8bit` — not part of the default gate
- **all**: core + optional

LLM correctness prompts ask for `17*23` and expect substring `391`. Chat requests send `enable_thinking: false` (and matching `chat_template_kwargs`) so Qwen-style CoT cannot burn `max_tokens` before the answer.

OCR uses the chat path with `document parsing.` plus an inline `image_url` data URL (tiny generated PNG fixture with the digits `391`). Unlimited-OCR-2 typically returns PAGE/BLOCK markup rather than plain digits; the suite sets `expect_nonempty` + `expect_has_digit` and `gate_decode_tps: false`. Decode tok/s is **not** gated for OCR when the decode wall is ≈0 (that previously produced absurd tok/s like hundreds of thousands).

### Auth and ports

- Inference `http://127.0.0.1:6590`, admin `http://127.0.0.1:6591` (overridable in the YAML)
- API key: `NOVA_API_KEY` → else enabled `primary` key in `~/.nova/nova_config.db` → else `abcd1234`

### Load / unload / readiness

- **Unload** only non-companion models with `loaded: true` from `GET /admin/models`. Companion draft ids (MTP / DFlash / DSpark — same idea as `ResourceLimits.isCompanionDraftModelId`) are skipped: admin currently reports them `loaded:true` via on-disk `companionReady` while `/health.loadedModels` stays 0. Never storm-unload those virtual companions.
- Unload **HTTP 400** (pinned models, e.g. `rufang/laya-hknews-tags-v4`) is logged once and not retried; a fully empty GPU is not required.
- After `POST /admin/models/load`, wait until **`/health.loadedModels >= 1` and** the target id is in the filtered admin loaded set — admin `loaded` alone is not trustworthy.
- **Warm / correctness** use **non-stream** chat by default (Flint stream has returned empty SSE then `404 model_not_loaded`, and can crash the app).
- **Measure** prefers stream for `decode_tps` math, with non-stream fallback after empty SSE / 404. Retries log HTTP bodies and **abort if `/health` dies**.

### Pass / fail

- **Correctness fail** → non-zero exit (expected substrings missing, empty OCR text, no digit in OCR markup, load/chat errors)
- **decode_tps regression** → non-zero when measured tok/s is more than **8%** below baseline (`tolerance.default`). `--strict` tightens to **5%**. Skipped for OCR / unreliable decode walls (`decode_wall < 50ms`)
- **Higher tok/s** always passes
- **No baseline yet** → warning only (unless correctness failed); seed with `--seed-baseline` (LLM only)
- **`--seed-baseline`** refuses suspiciously low `decode_tps` (contention artifacts): will not overwrite an existing baseline with a drop beyond ~40%, and applies soft floors for Flint / Qwen3-8B first seeds (e.g. rejects Qwen ~33 tok/s when quiet runs are ~70+). **Do not commit** machine-local baselines from a contended gate run — keep `Scripts/regression-baselines.json` as the empty seed in git until a quiet seed is intentional.

Results land in `~/.nova/regression/results-<gitsha>-<timestamp>.json`.

## How to try

NovaMLX must already be running on this Mac.

```bash
# Default core gate (uses Homebrew/PATH python3 with PyYAML)
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

Patterned on `Scripts/post_reboot_8bit_bench.py` (stream chat decode_tps math for measure, admin load/unload, `enable_thinking: false` default). This gate does not start NovaMLX and does not wait for Apple Silicon CI.

**Engine follow-up (out of scope for the gate PR):** admin should eventually split `loaded` (resident container) vs `companionReady` / draft-on-disk so UI and scripts stop treating companions as chat-resident. Until then the gate filters companions client-side.
