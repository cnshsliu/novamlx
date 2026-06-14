# Changelog

All notable changes to NovaMLX will be documented in this file.

## [Unreleased] - 2026-06-14

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
