# Changelog

All notable changes to NovaMLX will be documented in this file.

## [Unreleased]

### Fixed
- **Flint chat stop** — a role-header eos id other than the tokenizer's primary `eosTokenId` now ends fused generation and `/batch/completions` the same way the primary eos does, including when that token's text was scrubbed. After the assistant has already produced visible text, a new `<think>` (token 150006, or the decoded tag) ends the turn with `finish_reason=stop`. `/demo/chat` sends `max_tokens: 2048`.

### Added
- **Tested weights on Downloads** — the Downloads page lists every catalog entry with status `verified`: name, model version (family, quant, NovaMLX version that passed), and the exact weight id. One click downloads that weight.
- **Flint official INT4 is verified** — `RecursiveSI/flint-int4` is in that list (`testedOn` 1.8.1).

## [1.8.1] - 2026-10-26

**Headline: Flint packet width 16 — LOOP-RESULT 226.8 tok/s, FLINT-CHAT 217.3 tok/s.** MacMaxPro M4 Max, official `RecursiveSI/flint-int4`. #38 (`ef57be1`): greedy resident packet width **16**, PARITY match 16, LOOP-RESULT **226.8 tok/s** (kernel packet), SINGLE-RESULT **190.7 tok/s**. #39 (`313d92f`): FLINT-CHAT **217.3 tok/s** (`decodeTokens / wallDecodeSeconds`, prefill excluded; packetTokens=128, singleTokens=0). That same run's LOOP-RESULT was **223.3 tok/s** — 217.3 sits below the kernel line on that run; 226.8 vs 223.3 is run variance. Recursive's published Mac INT4 rec reference remains **196.8 tok/s** (1K context / 256 decoded tokens — a different window, not a re-run). llama.cpp on the same machine was ~89.7 tok/s in an earlier reference and was not re-run. The 1.8.0 packet=8 record (214.7) stays in the section below. Intro: [Flint](docs/features/flint.md).

### Added
- **Flint chat decode tok/s** — Greedy Flint chat (temperature 0, one sequence, no logprobs, frequency penalty 0, no grammar) uses the resident packet on the fused scheduler and on the serial engine path. Packet width is 16 (`FlintRecResident.width`, #38). `FLINT-CHAT` is `decodeTokens / wallDecodeSeconds` (prefill excluded; EOS excluded). Measured at `313d92f`: **217.3 tok/s** (packetTokens=128, singleTokens=0), below that run's LOOP-RESULT of **223.3 tok/s** (kernel packet only, no tokenizer). The width-16 kernel LOOP-RESULT at `ef57be1` is **226.8 tok/s**, with SINGLE-RESULT **190.7 tok/s** and PARITY match 16. The fused scheduler writes the `FLINT-CHAT` line to `~/.nova/novamlx.log` when the sequence finishes. `FlintSpeedTests` prints the same line after `LOOP-RESULT` for a chat-shaped window (one tokenizer decode per packet). Intro: [Flint](docs/features/flint.md).

## [1.8.0] - 2026-10-25

**Headline: Flint official INT4 packet decode, 214.7 tok/s.** MacMaxPro M4 Max, `RecursiveSI/flint-int4`, `./build.sh test --filter FlintSpeedTests` at `e8e7b27`. LOOP-RESULT is **214.7 tok/s** (128 rec steps, launch 76/64/64, greedy packet=8, argmax on device). Same run, one launch plus host argmax: **181.6 tok/s**. Recursive’s published Mac INT4 rec reference is **196.8 tok/s** (1K context / 256 decoded tokens — a different window, not a re-run of this suite). llama.cpp on the same machine was ~89.7 tok/s in an earlier reference and was not re-run. 214.7 is the shader timing window after compile, not HTTP end-to-end. Intro: [Flint](docs/features/flint.md), [brief](docs/feature-briefs/2026-10-25-flint.md).

**Core change: Flint (RecursiveSI) model support** — see the Flint entry below.

### Added
- **Running build on Status** — the Status page footer is one line: Debug or Release, version, build time to the second, and the `.app` path. That distinguishes `/Applications/NovaMLX.app` from a copy under the repo. The menu bar title includes Debug or Release, and the menu lists the same line and path. Settings → About shows configuration and location. `./build.sh` installs the newest host binary when SwiftPM has written more than one output directory. Intro: [menu bar](docs/features/menu-bar.md), [feature brief](docs/feature-briefs/2026-10-24-build-identity.md).
- **Flint (RecursiveSI) full inference** — 39.8B MoE agent model (~2.5B active) ported from mlx-lm to a native Swift/MLX `FlintModel`: GQA + per-head QK RMSNorm + learned attention sinks, hybrid 35 sliding-window(513)+RoPE / 14 full-attention-NoPE layers (RotatingKVCache mix), sigmoid-routed MoE (6/256 + ungated shared expert, dense layers 0-1), expert-stacking sanitize with uint8→uint32 views for flint-int4. Chat layer: `ChatTemplateFormat.flint` detection, FlintProcessor (role-header eos stop + scrub, think tags semantic), GLM4-dialect tool calls normalized to standard OpenAI tool_calls (Flint's syntax is exactly glm4), defensive eos ids. Flint routes through ContinuousBatcher — the fused streaming path has no ToolCallProcessor. Greedy output verified token-identical to Python mlx-lm with the same weights; reasoning + tool calls on OpenAI and Anthropic APIs, streaming and non-streaming; ~24 tok/s end-to-end (823 tok / 34.9 s) on M4 Max at `mlx-community/flint-OptiQ-4bit` (~26 GB resident). The implicit-thinking heuristic now matches its documented semantics (an open `<think>` only counts as an injection inside the add_generation_prompt block; Qwen3.5 unaffected). Intro: [feature page](docs/features/flint.md), [marketing report](marketing/why-novamlx/flint.md).
- **Flint default weights → official `RecursiveSI/flint-int4`** — catalog/docs/marketing now point at Recursive's official INT4 build (~22.6 GB, 4-bit lm_head and projections; the community OptiQ-4bit build carried 8-bit lm_head/projections) instead of `mlx-community/flint-OptiQ-4bit` used for early verification. Official config carries `rope_parameters.rope_type: "default"` (a string enum), so `FlintConfiguration` now decodes rope_parameters tolerantly, reading only numeric `rope_theta`. Gate 0 baselines on the official build (M4 Max, short-context pure decode): 22.7 tok/s bare decode eager vs llama.cpp Q4_K_XL 89.7 tok/s on the same machine.
- **Verified badge on Hub search results** — with "Allow unverified downloads" on, the Downloads page tags Hub results that are in the NovaMLX verified catalog with a green Verified badge, so unsupported repos are visible before downloading. Default catalog-gated search is unchanged.
- **3D generation (Hunyuan3D-2.1)** — task-style image-to-3D on a local Swift/MLX pipeline (vendored `hunyuan3d-swift`, no Python). `POST /v1/3d/generations` accepts one image (data URI / base64 / URL / local path) and returns 202; poll `GET /v1/3d/generations/{id}` or the shared `GET /v1/tasks/{id}` (`task = "3d.generation"`) through stable phases `queued → loading → shape → (paint) → export → done`; download the GLB from `/content`. One 3D job at a time (serial queue, like image tasks), cancellable mid-denoise. Shape weights stay resident until unload (`NOVA_HY3D_QUANT=0|4|8` cuts the resident DiT from ~3.6 GB to ~1.4 GB); optional `texture: true` loads a paint model per job and peaks ~25–39 GB (64 GB+ Macs). Weights are MLX conversions of official `tencent/Hunyuan3D-2.1` under the Tencent Hunyuan3D Community License — catalog ids `zimengxiong/hunyuan3d-mlx-shape-small` (~3.8 GB), `-shape-large` (~4.9 GB), `-paint-large` (~8.7 GB). `usage.meshes`/`usage.size_tier` are stable metering fields for tknet/peer billing (Level-2); `cost` stays 0. Intro: [feature page](docs/features/hunyuan3d.md), [feature brief](docs/feature-briefs/2026-10-22-hunyuan3d.md).
- **Pin on Local Inference** — each loaded model has a Pin switch. While it is on, Unload is hidden, admin unload/delete refuse, exclusive auto-unload and memory eviction skip it, and restart restores it even if it dropped off the last-loaded list. Turn Pin off to unload.
- **Adaptive verification length for DFlash2 and DSpark** — speculative verify no longer always scores a fixed draft block. Each round drafts up to the block budget, then verifies the prefix that maximizes expected accepted tokens per unit of target-verify cost. DSpark uses its confidence head. DFlash2 uses the top-two draft-score margin. Default mode is adaptive (min 1, max = draft budget, fixed cost 2, marginal cost 0.40 inside the fast SDPA window and 0.55 past 4 drafts). `"adaptive_verify": "fixed"` or `NOVAMLX_ADAPTIVE_VERIFY=fixed` restores the previous block, including DSpark’s 4-draft vector-SDPA cap. Greedy accept is unchanged, so a shorter prefix follows the same tokens as a fixed block on that prefix. Companion attach (known-faster auto-attach, known-slower skip, uncertain ask-once) is unchanged. `nova adaptive-verify --confidence …` prints the scheduled length without loading a model. Intro: [feature page](docs/features/speculative-decoding.md), [feature brief](docs/feature-briefs/2026-10-21-adaptive-verify.md).
- **MOSS-Image text-to-image** — `POST /v1/images/generations` loads OpenMOSS MOSS-Image through the existing image service. Catalog ids: `OpenMOSS-Team/MOSS-Image-SFT` (generation, ~27 GB) and `OpenMOSS-Team/MOSS-Image-Base` (same loader, research pretrain). Defaults are Euler, 50 steps, CFG 3, shift 5, empty negative, negative-branch skip of layer 22. Sides snap to multiples of 16 (minimum 256, area at most 1456×1456). Unified memory under 48 GiB quantizes the DiT and text encoder to int8 on load; `NOVA_MOSS_IMAGE_QUANT=bf16` or `int8` overrides. Edits and variations stay unavailable for this family. Intro: [feature page](docs/features/image-generation.md), [feature brief](docs/feature-briefs/2026-10-20-moss-image.md).
- **Model list filters** — On Local Inference, the “Downloaded, not loaded” header has a disk menu (All, plus each configured model root, labeled by volume or folder name) and a “Show companions” checkbox. DSpark, DFlash / DFlash2, and MTP companion weights stay hidden by default in both that list and the loaded-models list. The checkbox is remembered. Hiding only changes the lists; files stay on disk.
- **Supported models list** — [`docs/features/supported-models.md`](docs/features/supported-models.md) lists each catalog weight id (`org/repo`) from `catalog/models.json`. The root README links that page. Regenerate with `python3 Scripts/generate-supported-models-doc.py`.

## [1.6.0] - 2026-10-20

### Added
- **Feature catalog** — user-facing docs are indexed at [`docs/features/README.md`](docs/features/README.md) (one markdown page per feature). The root README links that index. `features.md` and `features.zh-CN.md` now point at it instead of keeping a second full catalog. Promo briefs remain in `docs/feature-briefs/`.
- **Text-only Nemotron 3.5 Bolt** (`nemotron_h_puzzle`) — load the Rapid-MLX 4-bit and 3-bit catalog previews as a text LLM (~43 GB disk / 64 GB unified memory, ~32 GB / 48 GB). Thinking defaults on (`enable_thinking`); tool calls use the checkpoint chat template (`xmlFunction`). Load refuses before Metal allocation when unified memory is below the indexed weight size plus 16 GiB (distributed workers pass `skipMemoryGate`). Intro: [feature page](docs/features/nemotron-bolt-text.md), [feature brief](docs/feature-briefs/2026-10-20-nemotron-bolt-text.md).
- **Request Log page** (new **Requests** sidebar entry) — scrolling record of every HTTP request the server handles. Upper section shows **live in-flight requests** (model, kind badge, API key used, elapsed time, live tok/s, auto-refreshing every 0.5s); lower section shows **completed requests** (method, endpoint, model, kind, API key, status, duration, TPS, error) with an "Errors only" filter and a "Clear" button. Backed by a new `RequestLogStore` (`NovaMLXUtils`), a `RequestLogMiddleware` (records start; errors finalize from HTTP status), and per-backend finalization hooks: LLM/VLM via `MLXEngine.generate`/`generateWithSession`/`generateWithProcessor` + a stream-tracking shim wrapping the fused/cluster/continuous-batcher/worker streams, ASR/TTS/image-gen at their route handlers. Capped at 300 recent entries. The HTTP request id is threaded from the `x-request-id` header into `InferenceRequest.httpRequestId` so the engine finalizes the exact matching log row.

### Fixed
- **Request Log: streaming ASR entries no longer get stuck as "active" forever.** Root cause was an id-correlation bug: `RequestIDMiddleware` and `RequestLogMiddleware` each generated their *own* request id, while the handler's `HTTPHelpers.requestID(from:)` read the *unmodified inbound* request and got back `nil` — so the streaming completion callback could never match its log entry, leaving the row pinned in the active list. Fixed by having `RequestIDMiddleware` propagate the resolved id back **onto the request** (rebuilds the `Request` with the `x-request-id` header stamped on it) before calling `next`, so the log middleware and every downstream handler read one consistent id. Also wired `cancelStale(olderThan: 120)` into the Requests page 0.5s tick as a safety net so any future uncaught orphan entries self-prune.
- **"Realtime Inference Speed" panel + dashboard speed card now reflect ALL model kinds**, not just LLM tok/s. ASR (transcription), TTS (speech synthesis), and image generation (SD/FLUX) now report live activity (model name + kind badge + kind-appropriate unit: `×RT`, `img/s`, `tok/s`) to the shared `MetricsStore`. The Status page hero icon and active-model line switch per kind (`waveform`/`speaker.wave.2`/`photo`/`eye`/`bolt`), so you always see what is running. Backed by a new `LiveActivity` record + `InferenceKind` enum, plumbed through `InferenceService` stats → `MenuBarAppState`.
- **VLM prefill now uses `model.prepare()` instead of `callAsFunction`**, so image features are actually merged into token embeddings during prefill. Previously, calling the model callable directly on raw `<|image_pad|>` token IDs made the LM decode placeholder tokens with no visual information — producing garbage logits that collapsed to EOS in a single step, so every VLM described pasted images wrongly (e.g. a red square → "Yellow", a man-on-horse hallucination). Fixed in both streaming and non-streaming paths in `MLXEngine.swift`; requires boxing `LMInput` payload into `SendableBox` for the `@Sendable` perform closure.

### ⚠ Breaking Changes
- **TokenHub is now remote-providers-only.** Local models no longer appear in the TokenHub provider list. Local inference remains in the renamed **Local Inference** page (formerly "Models").
- **`includeInLoadBalance` toggle removed** from provider cards. Multi-provider routing is now expressed via named Load Balancers, not via the per-provider LB flag.
- **Bare `tknet` dispatch rejected.** Requests with `model = "tknet"` now return 400 with a migration hint. Use `lb:<slug>` for load-balanced dispatch or `tknet:<provider-name>` for direct provider dispatch.
- **Implicit global LB removed.** Requests that previously fell through to the implicit global LB (every provider with `includeInLoadBalance=true`) now return 404. Create an explicit LB via the new **Load Balancers** page or `/admin/load-balancers` API.
- **`is_managed` column dropped.** Cloud-managed status is now expressed via the `"managed"` tag on the provider.

### Added
- **First-class Load Balancer entity** with its own SQLite tables (`load_balancers`, `lb_members`, `lb_member_stats`) and admin REST API at `/admin/load-balancers`.
- **Named LB dispatch** via `model = "lb:<slug>"` prefix. LBs support 5 selection strategies: `tiered` (default), `round_robin`, `weighted`, `lowest_latency`, `random`.
- **Per-member statistics**: request/success/failure counts, 5xx counter, latency average, last error — exposed via `/admin/load-balancers/:id`.
- **Auto-retry on member failure**: LBProxy retries up to `maxRetries` (default 3) on timeout/5xx, only before first byte for streaming requests.
- **Load Balancers page** in sidebar (sibling of TokenHub) with list + edit views, member picker (Local + Remote tabs), and per-strategy weight input.
- **`/admin/load-balancers/:id/test`** endpoint for invoking an LB with a sample payload and inspecting the chosen candidate trace.
- Localized "Local Inference" + "Load Balancers" sidebar/menu labels across all 9 languages.

### Changed
- Sidebar entry "Models" renamed to **"Local Inference"**.
- `LBStrategy` JSON wire format uses snake_case (`round_robin`, `lowest_latency`) to match the admin API contract; Swift enum case names stay camelCase.

### Removed
- Dead `pickRetryProvider` retry branches from tokenhub passthrough handlers (superseded by LBProxy).
- `provisionLocalProviders()` virtual-provider wrapping (locals are now referenced directly by model_id from LB members).
- Priority-tiered `TokenhubManager.resolve()` LB dispatch path.

### Migration
- On startup, `v4_load_balancers` migration creates the new tables, deletes rows where `is_managed = 1` from `tokenhub_providers`, and drops the legacy columns. Local model files in `~/.nova/models/` are untouched.
- Users who relied on the implicit global LB must create explicit LBs to restore multi-provider routing.

## [1.0.8] - 2026-05-08

### Added
- Pre-emptive memory feasibility check in model list endpoint
- Memory feasibility data per model in admin API

### Fixed
- Prevent double-finish race in SSE keep-alive continuation
- Scheduler concurrency regression tests

## [1.0.7] - 2026-05-07

### Fixed
- Eliminate 4 concurrency races in FusedBatchScheduler
- Add FinishGuard for atomic continuation lifecycle

## [1.0.6] - 2026-05-06

### Fixed
- Preserve `tool_calls` and `tool_call_id` in OpenAI incoming message mapping
- Preserve `tool_use`/`tool_result` blocks in Anthropic message mapping

## [1.0.5] - 2026-05-05

### Added
- Prefix cache: async write + async eviction in SSDCacheStore
- Prefix cache: safetensors header-only reader (replaces full-file scan)
- Prefix cache: skip fetch/store for VLM paths
- Prefix cache: pre-flight RotatingKVCache probe before SSD fetch
- Prefix cache: repeated-prefix TTFT benchmark

### Fixed
- E2E test: skip VLMs in core API suite, accept reasoning-only Harmony output

## [1.0.4] - 2026-05-04

### Added
- Audio transcription (`/v1/audio/transcriptions`) — Qwen3-ASR (Swift/MLX)
- Image generation (`/v1/images/generations`) — SDXL-Turbo
- Modelfile system — user-authored model recipes with system prompt and sampling overrides
- Per-request `keep_alive` — override model TTL per request
- Harmony streaming protocol — GPT-OSS channel-aware format
- ThinkingBudgetProcessor — per-request thinking token budget control
- Strict-FSM JSON logit processor — structured output with JSON schema
- Chat template library — three-level template resolution (user > registry > downloaded)
- `isImplicitThinkingModel` rewrite — auto-detect implicit thinking models at load time
- TokenMaskBuilder cache — pre-decoded vocabulary for fast masking
- VLM LogitProcessor chain — thinking detection for vision-language models
- DeepSeek-V4 lite regression test suite

### Fixed
- ThinkingParser regression tests
- Build script sync for worker binary

## [1.0.3] - 2026-05-02

### Added
- Cloud auth gate with subscription validation
- WebUI dashboard (SPA with status, models, chat pages)
- CLI login/logout/account commands
- GUI settings auth integration

### Changed
- Tagline updated from "fastest" to "blazing fast"

## [1.0.2] - 2026-05-01

### Added
- Homebrew tap distribution (`brew install novamlx`)
- Full OpenAI and Anthropic tools/function calling support
- Dynamic suggested searches from GitHub config
- Generic control token filtering for streaming output
- Agent-aware context scaling (ClientDetector)

### Fixed
- Buffer partial control tokens in streaming to prevent leaked fragments
- CI: patch mlx-swift-lm StrictConcurrency error
- CI: recurse submodules when cloning mlx-swift

## [1.0.1] - 2026-04-30

### Added
- Worker subprocess isolation for crash recovery
- TurnStopProcessor for Qwen3.6 turn separator handling
- ProcessMemoryEnforcer with soft/hard limits
- OCROptimizer for OCR model parameter tuning
- N-gram speculative decoding in FusedBatchScheduler
- Draft-model speculative decoding (SpeculativeTokenIterator)
- Full i18n system — 9 languages
- Web chat with input history and parameter controls
- Settings page with collapsible config.json editor
- Cloud model support — remote inference proxy with streaming

### Changed
- UI overhaul across all views

### Fixed
- Streaming deadlock fix

## [1.0.0] - 2025-04-09

### Added
- Pure Swift SPM project with modular targets
- Paged in-memory KV Cache with LRU eviction
- SSD-backed KV Cache for persistent caching
- Continuous batching for concurrent request processing
- MLX engine with model loading and tokenization
- LLM and VLM inference pipelines
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Anthropic-compatible API (`/v1/messages`)
- SSE streaming support for both API formats
- Model manager with download, registry, and versioning
- Auto-registration of popular MLX community models
- Native macOS menu bar app with SwiftUI
- System monitoring (CPU, memory, GPU)
- Health check and stats endpoints
- Comprehensive test suite
- MIT License

### Performance
- Zero Python dependencies
- Direct Metal acceleration via MLX
- Paged KV Cache with configurable memory limits
- SSD overflow for cold cache entries
- Continuous batching for maximum throughput
