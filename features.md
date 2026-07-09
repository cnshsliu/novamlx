# NovaMLX — Feature Reference

> **Production-grade, pure-Swift LLM / VLM / Audio / Image inference server for Apple Silicon.**
> OpenAI-, Anthropic-, and Responses-API-compatible. Native macOS menu bar app. Built on MLX. Multi-node distributed inference, cloud proxy, load balancing, and a full key-management system.

---

## Table of Contents

- [Inference Engine](#inference-engine)
- [Worker Subprocess Isolation](#worker-subprocess-isolation)
- [Multi-Modal Support](#multi-modal-support)
- [Model Architectures](#model-architectures)
- [API Compatibility](#api-compatibility)
  - [OpenAI-Compatible (Port 6590)](#openai-compatible-port-6590)
  - [Anthropic-Compatible](#anthropic-compatible)
  - [Responses API](#responses-api)
  - [Admin Endpoints (Port 6591)](#admin-endpoints-port-6591)
- [Structured Output](#structured-output)
- [Tool Calling](#tool-calling)
- [Control Token & Thinking Filtering](#control-token--thinking-filtering)
- [KV Cache & Prefix Share](#kv-cache--prefix-share)
- [TurboQuant KV Compression](#turboquant-kv-compression)
- [Continuous Batching](#continuous-batching)
- [Speculative Decoding](#speculative-decoding)
  - [N-gram Spec Decoding](#n-gram-spec-decoding)
  - [Draft-Model Spec Decoding](#draft-model-spec-decoding)
- [Distributed Inference](#distributed-inference)
- [Audio (ASR / TTS / Voice Cloning)](#audio-asr--tts--voice-cloning)
- [Image Generation](#image-generation)
- [Embeddings & Reranking](#embeddings--reranking)
- [Session Management](#session-management)
- [MCP — Model Context Protocol](#mcp--model-context-protocol)
- [Agent Integration](#agent-integration)
- [Tokenhub Cloud Proxy](#tokenhub-cloud-proxy)
- [Anthropic↔OpenAI Translation Bridge](#anthropicopenai-translation-bridge)
- [Load Balancers](#load-balancers)
- [API Key Management](#api-key-management)
- [Model Management](#model-management)
- [HuggingFace Integration](#huggingface-integration)
- [Modelfile System](#modelfile-system)
- [Memory Management](#memory-management)
- [Per-Model Settings](#per-model-settings)
- [Observability & Benchmarking](#observability--benchmarking)
- [macOS Menu Bar App](#macos-menu-bar-app)
- [Internationalization](#internationalization)
- [Security & Middleware](#security--middleware)
- [Homebrew Distribution](#homebrew-distribution)
- [TCC Watcher](#tcc-watcher)
- [Configuration](#configuration)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)

---

## Inference Engine

NovaMLX runs LLM, VLM, audio, and image inference directly on Apple Silicon GPU via MLX, with zero external dependencies on Python or remote services.

| Feature | Details |
|---------|---------|
| **Backends** | MLX (Apple Silicon GPU), lazy evaluation, unified memory |
| **Model formats** | SafeTensors — 4-bit, 8-bit, FP16, NVFP4 pre-quantized |
| **Sampling** | `temperature`, `top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed` |
| **Stop control** | Stop strings, stop token IDs, max-token cap |
| **Streaming** | SSE for OpenAI/Anthropic/Responses; raw bytes for audio/image |
| **Staged evaluation** | Batched `eval` prevents Metal OOM for models >500 arrays (e.g., Ling-2.6-flash 61 GB) |
| **CompiledSampler** | Precompiled sampling graph for hot-path token selection |
| **MLXEngine.perform** | Thread-isolated `model()+eval()+sample()` wrapper for safe concurrency |

## Worker Subprocess Isolation

Each inference model runs in a dedicated worker subprocess supervised by `WorkerSupervisor` (`Sources/NovaMLXInference/WorkerSupervisor.swift`).

| Feature | Details |
|---------|---------|
| **Process isolation** | Crash in one model cannot take down the server |
| **Auto-restart** | Crash recovery with 2-second cooldown, request re-routing |
| **Memory stats** | Tracks current / soft / hard RSS limits per worker |
| **GPU memory** | Live Metal allocator pressure reporting |
| **Request tracking** | In-flight request bookkeeping with auto-cleanup on crash |
| **Progress callbacks** | Phased model-loading progress (init / weights / kv-cache / ready) |

## Multi-Modal Support

| Modality | Architectures | Notes |
|----------|---------------|-------|
| **Vision (VLM)** | Llava, LlavaNext, LlavaQwen2, Qwen2VL, Qwen2.5VL, Qwen3VL, Mllama, Gemma3, Gemma4, InternVLChat, Idefics3, PaliGemma, Phi3V, Pixtral, Molmo, Florence2, Mistral3 | 3D-mRoPE state requires `ContinuousBatcher` (not fused-decode) |
| **Audio ASR** | Whisper, Qwen3-ASR | 48 kHz recording via Core Audio |
| **Audio TTS** | Qwen3-TTS (legacy), Dots TTS (current) | Dots voice cloning supported |
| **Image generation** | FLUX.1 (4-bit, 8-bit, FP16) | Vendored `flux.swift` |
| **Embeddings** | BERT, XLM-Roberta, ModernBert, Qwen3-ForTextEmbedding, Siglip, NomicBert | Pooling: mean / cls / last-token |
| **Reranking** | Cross-encoder rerankers | Scored re-order of candidate passages |

## Model Architectures

`ModelFamily` registry (`Sources/NovaMLXCore/Types.swift`): `llama`, `mistral`, `phi`, `qwen`, `gemma`, `starcoder`, `claude`, `bailing`, `gptOss`, `whisper`, `qwen3Asr`, `qwen3Tts`, `dotsTts`, `stableDiffusion`, `flux`, `other`.

**LLM families include:** Llama 3 / 3.1 / 3.2 / 3.3, Mistral / Mixtral, Qwen 2 / 2.5 / 3 / 3.5 / 3.6, Gemma 2 / 3 / 4, Phi 3.5 / 4, StarCoder2, GPT-OSS (Harmony), Bailing-Hybrid (Ling series — MLA + GLA + MoE).

**Family-specific defaults** (in `ModelFamilyRegistry`):

| Family | KV precision | Prefill | Context |
|--------|--------------|---------|---------|
| Llama / Mistral | 4-bit | 512 | 8192 |
| Phi | 32-bit | 256 | 4096 |
| Qwen | 4-bit | 512 | 8192 |
| Gemma | 32-bit | 256 | 8192 |

**Chat-template processor** — `ChatTemplateProcessorRegistry` routes by `(ModelFamily, ChatTemplateFormat)` tuple. Family-specific processors handle control tokens, implicit thinking detection, and stop-sequence injection.

## API Compatibility

Three wire formats share a single inference engine and tool-calling layer.

### OpenAI-Compatible (Port 6590)

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/v1/models` | GET | List loaded + downloaded models |
| `/v1/models/{id}` | GET | Model metadata |
| `/v1/chat/completions` | POST | Streaming + non-streaming |
| `/v1/completions` | POST | Legacy text completion |
| `/v1/embeddings` | POST | Text → vector |
| `/v1/audio/transcriptions` | POST | Whisper / Qwen3-ASR |
| `/v1/audio/speech` | POST | Dots / Qwen3-TTS |
| `/v1/images/generations` | POST | FLUX.1 |
| `/v1/images/edits` | POST | FLUX.1 with mask |
| `/v1/images/variations` | POST | FLUX.1 variations |
| `/v1/rerank` | POST | Cross-encoder reranking |

### Anthropic-Compatible

| Endpoint | Method | Notes |
|----------|--------|-------|
| `/v1/messages` | POST | Streaming + non-streaming; `messages`, `system`, `tools`, `tool_choice`, `thinking_budget` |
| `/v1/messages/count_tokens` | POST | Pre-flight token count |

Auth: `x-api-key` or `Authorization: Bearer`. `anthropic-version: 2023-06-01` header required.

### Responses API

Codex-compatible `/v1/responses` implementation.

| Feature | Details |
|---------|---------|
| **Endpoint** | `POST /v1/responses` (stream + non-stream), `GET/DELETE /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` |
| **P0** | `tool_choice` passthrough (auto / required / none / specific tool) |
| **P1** | 17-field response echo (status, output, usage, completed_at, etc.) |
| **P2** | SSE `seq` field for client-side gap detection |
| **ConversationStore** | `previous_response_id` resolves stored conversation history |
| **Reasoning aliases** | `reasoning.effort` → `thinking_budget` |
| **Compact** | `POST /v1/responses/compact` truncates stored conversation |
| **Input tokens** | `POST /v1/responses/input_tokens` exact tokenization count |

### Admin Endpoints (Port 6591)

Bearer-token auth via admin key. Full catalog:

- **Models** — list / detail / download / cancel / load / unload / forget / benchmark / perplexity / cache stats
- **API Keys** — CRUD, rotate, usage stats, per-key metrics
- **Tokenhub providers** — CRUD, test, metrics
- **Load balancers** — CRUD, member management, dry-run routing
- **Modelfiles** — CRUD for custom model definitions
- **Stats / Info / Reset** — server metrics, system info, reset
- **Log level** — runtime log severity control

## Structured Output

| Mode | Wire field | Notes |
|------|------------|-------|
| **JSON object** | `response_format: {type: "json_object"}` | Forces valid JSON output |
| **JSON schema** | `response_format: {type: "json_schema", json_schema: {schema: ...}}` | Strict-schema constrained generation |
| **Regex** | `response_format: {type: "regex", regex: "..."}` | Regex-anchored output |
| **GBNF grammar** | `response_format: {type: "gbnf", gbnf: "..."}` | llama.cpp grammar format |
| **Choice / Enum** | via schema `enum` | One-of constrained selection |

## Tool Calling

| Format | Field | Response shape |
|--------|-------|----------------|
| **OpenAI** | `tools: [{type:"function", function:{name, parameters}}]` | `choices[0].message.tool_calls[]` |
| **Anthropic** | `tools: [{name, description, input_schema}]` | `content[].tool_use` blocks |
| **Responses** | via modelfile | Extracted via regex + JSON parse |

`tool_choice`: `auto` / `required` / `none` / specific tool / `any` (Anthropic). Bidirectional translation preserves tool id, name, and arguments across formats.

## Control Token & Thinking Filtering

| Subsystem | Behavior |
|-----------|----------|
| **TurnStopProcessor** | Model-specific stop-token sets (`<\|turn\|>`, `<\|end\|>`, etc.); excludes channel tokens for channel-thinking models |
| **ThinkingParser** | Implicit `<think>` tag handling; explicit thinking-budget pass-through |
| **Semantic vs. protocol tokens** | Semantic tags (`<think>`) pass through; protocol tokens (`<\|turn\|>`) filtered |
| **Per-model detection** | `isImplicitThinkingModel` flag per model ID |
| **Finalize-before-stop** | Flushes parsed thinking before emitting stop chunk (Qwen3.6 streaming fix) |
| **Channel-aware** | GPT-OSS Harmony `<\|channel\|>` tokens excluded from stop set |

## KV Cache & Prefix Share

| Feature | Details |
|---------|---------|
| **SSD cache store** | Persistent cross-session KV cache at `~/.nova/cache/` |
| **Block hashing** | Content-addressable blocks via `BlockHasher` |
| **Paged block pool** | Memory-efficient block management via `PagedBlockPool` |
| **Auto-load on startup** | Rehydrates cache for warm models (hybrid models currently disabled due to Mamba+KV mix) |
| **Hit/miss tracking** | Per-model cache statistics exposed via `/admin/models/{id}/cache` |
| **Session reuse** | `session_id` request field pins a request to a KV-cache lineage; same session = same KV across requests. Works on `/v1/chat/completions`, `/v1/messages`, `/v1/responses`. |

### Clarification: Anthropic `cache_control` is intentionally not parsed

Anthropic's `cache_control: { type: "ephemeral" }` field exists for **billing** — it tells Anthropic's cloud which content blocks to charge cache-write pricing for. NovaMLX is a **local inference server with no billing layer**, so the field serves no purpose here: KV cache is reused automatically on every request that shares a prefix.

How we handle `cache_control` per routing path:

| Path | Behavior |
|------|----------|
| **Local inference** (`/v1/messages` → MLX) | Field silently dropped by Codable decode (no `cache_control` key on `AnthropicContentBlock`). KV cache reuse still happens automatically via prefix matching + `session_id`. **No client action needed.** |
| **`tknet:` / `lb:` → native Anthropic upstream** (raw passthrough) | Field preserved verbatim in body. `anthropic-version` header always forwarded; `anthropic-beta` header forwarded when client sent it (so 1-hour cache TTL reaches the provider instead of silently degrading to 5 minutes). |
| **`tknet:` / `lb:` → OpenAI-format upstream** (Anthropic↔OpenAI bridge) | Field dropped during translation (OpenAI chat/completions has no equivalent). Server emits a `[WARN] [TokenhubBridge]` log line so operators can see the silent downgrade. |

## TurboQuant KV Compression

4-bit affine-quantized KV cache with dynamic group sizing.

| Feature | Details |
|---------|---------|
| **Compression** | 4-bit affine quantization for K and V tensors |
| **Group sizing** | Dynamic per-layer group size based on head dim |
| **Transparency** | Hidden from sampler — operates below sampling layer |
| **Quality trade** | ~75% KV memory reduction with minimal quality loss on long context |

## Continuous Batching

| Feature | Details |
|---------|---------|
| **Default batch** | 8 sequences (configurable) |
| **Preemption** | Priority-based eviction under memory pressure |
| **Specialized queues** | Separate paths for VLM, hybrid-attention, sessions, grammar-guided |
| **Async streaming** | Priority-queued token emission per request |
| **Metrics** | `BatcherMetrics` exposes queue depth, preemption rate, mean wait |

## Speculative Decoding

### N-gram Spec Decoding

Free speculative decoding via n-gram token prediction (no draft model required). Lookahead window + acceptance sampling.

### Draft-Model Spec Decoding

| Feature | Details |
|---------|---------|
| **DraftModelRegistry** | Auto-injects recommended draft model per target model family |
| **Built-in drafts** | Qwen3-0.6B-4bit, Llama-3.2-1B-4bit, Gemma-2-2B-4bit |
| **API** | `draft_model` + `num_draft_tokens` request fields |
| **EOS suppression** | Draft-only EOS tokens filtered to prevent premature stop |
| **Limits** | Hybrid-attention (Mamba) targets unsupported; cross-vocab drafts unsupported |
| **Status API** | `SpecBoostStatus`: `.eligible` / `.active` / `.ineligible` per model |

## Distributed Inference

Pipeline-parallel sharding across multiple Apple Silicon nodes over TCP / Thunderbolt.

| Feature | Details |
|---------|---------|
| **Transport** | Raw TCP binary data plane; scale-adaptive control plane |
| **Shard policy** | `SlicedForwardPolicy` — reflection-based layer slicing |
| **ShardEngine** | Coordinates per-node forward pass |
| **ClusterModelManager** | Cluster-wide activation / deactivation; `ClusterModelState` (idle / activating / ready / failed) |
| **Remote sampling** | Argmax on worker; 4-byte token ID instead of full logits tensor |
| **WorkerSupervisor** | Per-node worker lifecycle with heartbeat / health tracking |
| **Auto-fallback** | Degrades to local inference on cluster failure |
| **Backends compiled** | Ring (TCP) live; JACCL (RDMA) ready behind `rdma_ctl` flag |
| **Profiling (measured)** | coord 31.8 ms, worker 33.7 ms, TCP 9 ms, tokenizer 0.3 ms; ~14 tok/s sequential ceiling; ~13.8 tok/s baseline with remote sampling |
| **Real-world test** | Qwen3.6-27B across M4 Max + M4 Mac Mini via Thunderbolt → 1.8 tok/s pipeline-parallel |

## Audio (ASR / TTS / Voice Cloning)

| Feature | Details |
|---------|---------|
| **Whisper ASR** | 48 kHz recording, language auto-detect, `TranscriptionContainer` for hot-swap |
| **Qwen3-ASR** | Alternate ASR backend, same `/v1/audio/transcriptions` surface |
| **Dots TTS** | Vendored `mlx-swift-dots-tts`; `DotsTTSPipeline` neural voice |
| **Qwen3-TTS** | Legacy path; superseded by Dots |
| **System voices** | macOS `NSSpeechSynthesizer` fallback |
| **Voice cloning** | `VoiceProfile` manager at `~/.nova/voices/`; multi-speaker; reference-audio cloning |
| **VoiceCloneSheet UI** | Record / pick reference; preview; persist as named profile |
| **Mic permission** | `audio-input` entitlement + `NSMicrophoneUsageDescription` injected via `build.sh` |

## Image Generation

| Feature | Details |
|---------|---------|
| **Model** | FLUX.1 (4-bit, 8-bit, FP16) |
| **Pipeline** | Vendored `flux.swift` `FluxPipeline` |
| **Container** | `ImageGenerationContainer` for hot-load / unload |
| **Endpoints** | `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations` |
| **Output** | Base64 PNG; configurable height / width / steps / guidance |
| **Service** | `ImageGenerationService` async API |

## Embeddings & Reranking

| Feature | Details |
|---------|---------|
| **EmbeddingContainer** | Hot-loadable embedding models |
| **Architectures** | BERT, XLM-Roberta, ModernBert, Qwen3-ForTextEmbedding, Siglip, NomicBert |
| **Pooling** | mean / cls / last-token |
| **Endpoint** | `POST /v1/embeddings` |
| **RerankerContainer** | Cross-encoder rerankers |
| **Endpoint** | `POST /v1/rerank` (candidates → scored candidates) |

## Session Management

| Feature | Details |
|---------|---------|
| **Session ID** | `session_id` request field pins request to KV-cache lineage |
| **Fork** | Branch a session into a new ID without copying KV |
| **TTL** | Idle sessions evicted under memory pressure (LRU) |
| **Cross-endpoint** | Same session works across OpenAI / Anthropic / Responses |

## MCP — Model Context Protocol

| Feature | Details |
|---------|---------|
| **Transports** | stdio, SSE, streamable-HTTP |
| **Tools** | `MCPTool` with JSON-schema input; namespaced `server__tool` naming |
| **Resources** | Exposed via `MCPServerConfig` |
| **Server status** | disconnected / connecting / connected / error |
| **Tool exec** | `MCPExecuteRequest` / `Response`; timeout + headers configurable |
| **Validation** | Input-schema enforcement before tool invocation |

## Agent Integration

| Agent | Built-in support |
|-------|------------------|
| **OpenClaw** | Tool-using agent |
| **Hermes Agent** | Long-running reasoning agent |
| **OpenCode** | Coding agent |
| **Plugin system** | Extensible agent frameworks |

`AgentsPageView` provides install / launch / configure / view-config UX.

## Tokenhub Cloud Proxy

Proxy remote API providers through NovaMLX so all clients (Codex, Claude Code, Continue, etc.) can hit one local endpoint.

| Feature | Details |
|---------|---------|
| **Provider catalog** | 20+ pre-configured: OpenAI, Anthropic, DeepSeek, GLM/Zhipu, Qwen/DashScope, Groq, Mistral, Moonshot, Yi, Together, Fireworks, OpenRouter, and more |
| **Provider kinds** | Cloud-managed (tknet.ai session) vs. BYO-key |
| **Routing** | `tknet:<provider-id>` model prefix → resolves provider |
| **Passthrough** | Raw body forward; swaps `model` to `provider.remoteModel` |
| **Vision backends** | Tier 1: local VLM; Tier 2: provider's `anthropicEndpoint` (e.g., GLM anthropic-proxy); Tier 3: `visionCompanionModel`; with image preprocessing + description injection |
| **Provider metrics** | Success count, request count, avg latency, per-provider stats |
| **Auth resolution** | Managed providers use session token; BYO-key uses `provider.apiKey` |
| **Endpoint field** | `anthropicEndpoint` opt-in for providers that natively expose Anthropic format |

## Anthropic↔OpenAI Translation Bridge

When a client sends `/v1/messages` (Anthropic format) but the resolved provider speaks OpenAI only (DeepSeek, GLM, Qwen-compat, etc.), the bridge translates the request to OpenAI `/chat/completions`, forwards it, and rebuilds an `AnthropicResponse`.

| Subsystem | Behavior |
|-----------|----------|
| **Discriminator** | `needsAnthropicBridge(provider, path)`: true when path is `messages` AND provider's `anthropicEndpoint` is unset |
| **Inbound** | Decodes `AnthropicRequest`, maps via existing `mapAnthropicMessages` |
| **Outbound body** | Builds OpenAI chat/completions: model, messages, tools, tool_choice, sampling, stop |
| **Response** | Decodes `OpenAIResponse`, builds `AnthropicResponse` with text / thinking / tool_use blocks |
| **Streaming** | Event-by-event state machine: OpenAI chunks → Anthropic `message_start` / `content_block_start` / `content_block_delta(text_delta\|thinking_delta\|input_json_delta)` / `content_block_stop` / `message_delta(stop_reason, usage)` / `message_stop` |
| **Stop reason map** | `stop → end_turn`, `tool_calls → tool_use`, `length → max_tokens`, `stop_sequence → stop_sequence` |
| **LB-transparent** | LB dispatcher routes `lb:` + messages through the same passthrough — bridge kicks in automatically |
| **Files** | `Sources/NovaMLXAPI/APIServer+TokenhubAnthropicBridge.swift` |

## Load Balancers

Route requests across pools of local + remote models via `lb:<slug>` model prefix.

| Feature | Details |
|---------|---------|
| **LBRouter strategies** | Tiered (default), round-robin, weighted, least-latency |
| **Member kinds** | `.local` (inference service) and `.remote` (tokenhub provider) |
| **LBProxy** | Per-request actor: picks member, tries candidates in order, retries on failure |
| **Admin API** | 9 endpoints: CRUD for LBs + members, dry-run routing, stats |
| **Per-member stats** | Success / fail / avg latency; surfaces in UI |
| **UI** | `LoadBalancersPageView`: accordion rows, member picker, strategy config, per-LB play button |
| **API formats** | Works across OpenAI / Anthropic / Responses (Anthropic via translation bridge; Responses via `tknet:` rewrite) |

## API Key Management

SQLite-backed API key system with hashing, rate limits, and whitelists.

| Feature | Details |
|---------|---------|
| **Storage** | SQLite via `APIKeyStore` (replaces former `api_keys.json`) |
| **Hashing** | SHA-256 (plaintext retained for reveal feature; opt-in DB access) |
| **CRUD** | Create, read, update, delete via `/admin/keys` and UI |
| **Rotate** | `/admin/keys/{id}/rotate` mints a new plaintext, invalidates old |
| **Rate limits** | Per-key `rateLimitPerSecond`, `maxTokensPerPeriod`, `maxRequestsPerPeriod` with reset period (minute / hour / day) |
| **Whitelists** | Per-key `allowedModels[]`, `allowedEndpoints[]` |
| **Usage tracking** | Total + period tokens / requests; per-model breakdown; last-used timestamp |
| **Open-mode bypass** | When no keys configured, auth disabled (dev mode) |
| **UI** | `APIKeysPageView`: whole-row accordion, eye-reveal toggle, copy-revealed button, usage stats with progress bars, whitelists |
| **Admin vs user key** | Separate middleware: `AdminAuthMiddleware` (port 6591) vs `APIKeyAuthMiddleware` (port 6590) |

## Model Management

| Feature | Details |
|---------|---------|
| **Model directory** | `~/.nova/models/<repo_id>/` |
| **Discovery** | `modelManager.downloadedModels()` + `inferenceService.listLoadedModels()` |
| **Auto-load** | Request to unloaded model triggers load (configurable via `ensureModelReady`) |
| **Auto-eviction** | LRU under memory pressure; never evicts mid-request |
| **Restore on restart** | `restoreModels()` rehydrates last session's loaded set |
| **loaded_models persistence** | SQLite table (was JSON, fixed wipe-on-restart bug) |
| **Model settings** | Per-model overrides persisted in SQLite (`model_settings` table) |
| **Model cards** | Fetch metadata from HuggingFace via `/admin/api/hf/model-info` |
| **Forgetting** | `/admin/models/{id}/forget` — drops KV cache + container state |

## HuggingFace Integration

| Feature | Details |
|---------|---------|
| **Download** | `POST /admin/api/hf/download` with `repo_id` + optional `endpoint` |
| **Cancel** | `POST /admin/api/hf/cancel` by task ID; kills in-flight `Task` |
| **Status polling** | `GET /admin/api/hf/tasks` returns per-file progress + speed + stall detection |
| **Mirror support** | Configurable HF endpoint (default `huggingface.co`; CN users often set `hf-mirror.com`) |
| **Idempotent resume** | `cancelTasksForRepo` kills any in-flight task before starting new (prevents click-spam races) |
| **HEAD probe** | 3-second reachability check before each file download (fail-fast on blocked endpoints) |
| **Phase UI** | Client shows: Connecting / Downloading (MB/s) / Stalled (Ns since last byte) / Endpoint unreachable |
| **Partial-scan** | Detects `*.download` temp files on launch → surfaces as `.failed` tasks for manual Resume |
| **Xet CDN** | Auto-follows `cas-bridge.xethub.hf.co` redirects |

## Modelfile System

Custom model definitions analogous to Ollama Modelfiles.

| Feature | Details |
|---------|---------|
| **Storage** | SQLite `modelfiles` table |
| **Fields** | `name`, `base_model`, `system`, `template`, `parameters`, `adapter`, `tools` |
| **Admin API** | CRUD at `/admin/modelfiles` |
| **Resolution** | Modelfile name resolves like a model ID |
| **Tool defs** | Static tool definitions embedded in modelfile |

## Memory Management

| Subsystem | Behavior |
|-----------|----------|
| **ProcessMemoryEnforcer** | Hard RSS cap via `ProcessInfo.memoryPressure` + proactive trim |
| **MemoryBudgetTracker** | GPU memory budget per active model |
| **WiredMemoryTicket** | Reserves wired Metal memory before allocation |
| **Auto mode** | Process monitors system pressure, evicts when needed |
| **Disabled mode** | No cap (dev / benchmark) |
| **Percent mode** | Cap = N% of system RAM |
| **Fixed mode** | Explicit GB cap |
| **Staged eval** | Large models load in batches to avoid Metal OOM |

## Per-Model Settings

Persisted per model ID, override defaults at request time:

- Sampling defaults (temperature, maxTokens, topP, repeatPenalty)
- KV precision (4-bit / 8-bit / 32-bit)
- Context window override
- TurboQuant toggle
- Vision strategy
- Companion vision model
- Thinking defaults (`enableThinking`, `thinkingBudget`, `preserveThinking`)
- Keep-alive interval

## Observability & Benchmarking

| Tool | Endpoint | Notes |
|------|----------|-------|
| **Benchmark** | `/admin/models/{id}/benchmark` | Measures TPS, TTFT, memory, peak |
| **Perplexity** | `/admin/models/{id}/perplexity` | Standard perplexity on test set |
| **Cache stats** | `/admin/models/{id}/cache` | KV cache hit / miss / size |
| **Server stats** | `/admin/stats` | Aggregate: total tokens, active reqs, uptime |
| **System info** | `/admin/info` | Chip, cores, RAM, GPU, macOS version |
| **Logging** | `~/.nova/novamlx.log` | Rotating file log; runtime level via admin |
| **Log levels** | debug / info / warning / error | `POST /admin/log/level` |
| **Metrics headers** | `X-Tokenhub-Provider`, `X-Model-Cold-Load`, `X-Model-Load-Time-Ms` | Per-response introspection |
| **InferenceStats** | Live TPS, peak TPS, tokens generated, active requests, worker CPU | Polled every 2 s by UI |

## macOS Menu Bar App

Native SwiftUI menu bar app. Status icon + dropdown + popout window.

### Pages

| Page | Highlights |
|------|------------|
| **Status** | Live TPS chart (90-sample window, peak tracking, zero-trim), CPU/mem/GPU grid, device info, peak TPS |
| **Dashboard** | One-screen overview: loaded models, active requests, memory, uptime, quick-load |
| **Local Inference** | Active models with unload + copy-name + play-in-Playground; downloaded models with type tabs (All / LLM / VLM / Embed / Audio / Image); model card |
| **Downloads** | Category tabs, suggested-model cards, phase-aware progress, stall detection, mirror config |
| **Playground (Chat)** | Unified LLM / ASR / TTS / Image; model-type auto-detect; section-header picker (LOCAL DIRECT IN-PROC vs TOKENHUB HTTP); parameter sliders; Disable-Thinking toggle; sticky auto-scroll; copy cURL (OpenAI/Anthropic/Responses); ASR mic + TTS speaker + voice clone |
| **Tokenhub** | 20+ provider catalog; CRUD; per-provider API models with copy + play buttons; endpoint testing |
| **Load Balancers** | Accordion rows; member picker (local + remote); strategy + stats; play button per row |
| **API Keys** | Whole-row accordion; eye reveal; copy revealed; usage bars; rate-limit display; whitelists |
| **Cluster** | Network scan (Thunderbolt + ARP); worker monitoring with health states; per-node model readiness |
| **Settings** | HF endpoint + mirror; memory mode; log level; cluster toggle; TurboQuant; tknet.ai account link |
| **Audio** | Dedicated ASR / TTS surface (separate from Playground) |
| **Agents** | Install / launch / configure OpenClaw, Hermes, OpenCode |

### Cross-page UX

| Feature | Details |
|---------|---------|
| **Pick-to-Playground** | `play.circle` button on Active Models / Tokenhub API models / Load Balancers → jumps to Playground + pre-selects model |
| **Copy cURL buttons** | In Playground Parameters: OpenAI / Anthropic / Responses — emit `${NOVA_API_KEY}` placeholder so secret never enters clipboard |
| **Sticky auto-scroll** | 80 pt threshold via `MessageListBottomOffsetKey` PreferenceKey |
| **Model picker** | `inferModeFromName` + `autoDetectMode` switch mode (LLM / ASR / TTS / Image) automatically |
| **HF Endpoint settings** | Mirror swap without restart — propagated via `NovaMLXConfiguration.shared` |

## Internationalization

9 languages, auto-detected from system locale with English fallback.

`English`, `Simplified Chinese (zh-Hans)`, `Traditional Chinese HK (zh-Hant-HK)`, `Traditional Chinese TW (zh-Hant-TW)`, `Japanese`, `Korean`, `French`, `German`, `Russian`.

## Security & Middleware

| Layer | Behavior |
|-------|----------|
| **CORS** | Configurable allowed origins; preflight handling |
| **Rate limiter** | Token bucket per API key + per IP; global + per-route limiters |
| **Error middleware** | `NovaMLXErrorMiddleware` normalizes errors to OpenAI/Anthropic shape; `Retry-After` header on 429 |
| **Admin auth** | `AdminAuthMiddleware` on port 6591 — admin key required |
| **API auth** | `APIKeyAuthMiddleware` on port 6590 — Bearer or `x-api-key`; open-mode bypass when no keys |
| **Strict headers** | `X-Content-Type-Options`, `Strict-Transport-Security`, `X-Frame-Options`, `Referrer-Policy` |
| **Request ID** | `x-request-id` per request for tracing |
| **Cloud session** | `AuthCache` for tknet.ai session token; cloud validation endpoint |

## Homebrew Distribution

| Channel | Details |
|---------|---------|
| **Formula** | `Formula/novamlx.rb` |
| **Build script** | `./build.sh` — UUID sync, codesigning, MLX shader compile, bundle assembly |
| **DMG** | Script-based disk image packaging |
| **Codesigning** | `codesign --force --deep --sign -` required post-deploy (else macOS SIGKILLs worker) |
| **Entitlements** | `audio-input`, `com.apple.security.device.camera` (where applicable), `NSMicrophoneUsageDescription` injected via PlistBuddy |

## TCC Watcher

| Feature | Details |
|---------|---------|
| **Purpose** | Auto-dismiss macOS TCC privacy prompts that block automation |
| **Bundle** | `TCCWatcher.app` (CFBundleIdentifier `com.novamlx.TCCWatcher`) |
| **Install** | `Scripts/install-tcc-watcher.sh` — sets up LaunchAgent |
| **Mechanism** | Watches for TCC prompts via `System Events`; inlined `inspectWindow` in main tell block (fixes context-prop bug) |
| **One-time setup** | User adds `TCCWatcher.app` via `+` button in System Settings → Privacy → Accessibility (macOS 14+ no auto-prompt) |

## Configuration

### `ServerConfig` (`~/.nova/config.json` or SQLite)

| Field | Default | Notes |
|-------|---------|-------|
| `server.port` | 6590 | Public API port |
| `server.adminPort` | 6591 | Admin API port |
| `server.cluster` | null | Enables distributed mode |
| `server.apiKey` | null | Open-mode if null |
| `server.corsOrigins` | `*` | CORS allow-list |
| `huggingface.endpoint` | `huggingface.co` | Mirror swap |
| `autoLoad` | enabled | Pre-load models on startup |
| `memory.mode` | `auto` | auto / disabled / percent / fixed |
| `memory.limitGB` | null | Fixed cap |
| `scaleTokenCount` | enabled | Scale reported tokens for context accounting |
| `logLevel` | `info` | debug / info / warning / error |

### `NOVA_DIR`

| Precedence | Source |
|------------|--------|
| 1 (highest) | `~/.config/novamlx/path` file contents |
| 2 | `NOVA_DIR` environment variable |
| 3 (default) | `~/.nova` |

Multi-instance supported. Only `models/` is shareable across instances.

### Engine Configuration

Per-family defaults in `ModelFamilyRegistry` — KV precision, prefill chunk, context window, draft-model recommendation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Clients: Codex / Claude Code / Continue / OpenAI SDK /     │
│           Anthropic SDK / curl                              │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │   Hummingbird HTTP/2    │  Port 6590 (api) + 6591 (admin)
            │   + CORS + RateLimit    │
            │   + Auth Middleware     │
            └────────────┬────────────┘
                         │
   ┌─────────────────────┼──────────────────────────┐
   │                     │                          │
┌──▼─────────┐  ┌────────▼─────────┐  ┌────────────▼───────────┐
│  Local     │  │  Tokenhub Proxy  │  │  LBProxy (lb:<slug>)   │
│  Inference │  │  + Anthropic     │  │  → routes to local or  │
│  (Worker)  │  │    Bridge        │  │    tokenhub member     │
└──┬─────────┘  └────────┬─────────┘  └────────────────────────┘
   │                     │
┌──▼─────────────────────▼──┐
│  MLX Engine               │
│  - ContinuousBatcher      │
│  - TurboQuant KV Cache    │
│  - Speculative Decoding   │
│  - Tool calling           │
│  - Structured output      │
│  - Chat template proc     │
└──┬────────────────────────┘
   │
┌──▼──────────────────────┐  ┌──────────────────────────┐
│  WorkerSupervisor       │  │  Distributed (optional)  │
│  (subprocess isolation) │  │  Ring TCP / Thunderbolt  │
└─────────────────────────┘  └──────────────────────────┘
```

Data persistence: **NovaDB** (SQLite + GRDB) — 14 tables covering api_keys, providers, load_balancers, modelfiles, loaded_models, model_settings, metrics, cluster_policy, conversations (Responses), and more. Auto-migrates legacy JSON on first launch.

## Quick Start

```bash
# Install (Homebrew)
brew install --head novamlx

# Or build from source
git clone https://github.com/novamlx/novamlx && cd novamlx
./build.sh
open dist/NovaMLX.app

# Set your API key (or run in open mode without one)
export NOVA_API_KEY=$(openssl rand -hex 24)

# List models
curl http://localhost:6590/v1/models \
  -H "Authorization: Bearer $NOVA_API_KEY"

# Chat
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role":"user","content":"Hello"}]
  }'

# Anthropic format
curl http://localhost:6590/v1/messages \
  -H "x-api-key: $NOVA_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"Hello"}]
  }'

# Proxy a cloud model (set up via Tokenhub UI first)
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tknet:deepseek-v4-flash",
    "messages": [{"role":"user","content":"Hello"}]
  }'

# Load balancer (create via UI first, then use lb:<slug>)
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lb:my-pool",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

---

**NovaMLX** — *Built by hlky and contributors. Released under the MIT License.*
