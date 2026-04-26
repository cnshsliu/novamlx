# NovaMLX — Feature Reference

> **Production-grade, pure-Swift LLM/VLM inference server for Apple Silicon.**
> OpenAI- and Anthropic-compatible APIs. Native macOS menu bar app. Built on MLX.

---

## Table of Contents

- [Inference Engine](#inference-engine)
- [Worker Subprocess Isolation](#worker-subprocess-isolation)
- [Multi-Modal Support](#multi-modal-support)
- [API Compatibility](#api-compatibility)
- [Structured Output](#structured-output)
- [Tool Calling](#tool-calling)
- [Control Token & Thinking Filtering](#control-token--thinking-filtering)
- [KV Cache & Prefix Sharing](#kv-cache--prefix-sharing)
- [TurboQuant KV Compression](#turboquant-kv-compression)
- [Continuous Batching](#continuous-batching)
- [Speculative Decoding](#speculative-decoding)
- [Session Management](#session-management)
- [Embeddings & Reranking](#embeddings--reranking)
- [Audio — Speech-to-Text & Text-to-Speech](#audio--speech-to-text--text-to-speech)
- [MCP — Model Context Protocol](#mcp--model-context-protocol)
- [Agent Integration](#agent-integration)
- [Model Management](#model-management)
- [Memory Management](#memory-management)
- [Per-Model Settings](#per-model-settings)
- [HuggingFace Integration](#huggingface-integration)
- [Cloud Model Proxy](#cloud-model-proxy)
- [Observability & Benchmarking](#observability--benchmarking)
- [Web Interfaces](#web-interfaces)
- [macOS Menu Bar App](#macos-menu-bar-app)
- [Internationalization](#internationalization)
- [Security & Middleware](#security--middleware)
- [Homebrew Distribution](#homebrew-distribution)
- [Configuration](#configuration)

---

## Inference Engine

NovaMLX runs LLM and VLM inference directly on Apple Silicon GPU via MLX, with zero external dependencies on Python or remote services.

| Feature | Details |
|---------|---------|
| **Backends** | MLX (Apple Silicon GPU), lazy evaluation, unified memory |
| **Model formats** | SafeTensors (4-bit, 8-bit, FP16 quantization) |
| **50+ architectures** | Llama 3/3.1, Mistral/Mixtral, Qwen 2/2.5/3, Gemma 2/3, Phi 3.5/4, StarCoder2, and more |
| **Sampling** | Temperature, Top-P, Top-K, Min-P, Frequency/Presence/Repetition penalty, Seed |
| **Streaming** | SSE token-by-token streaming for all generation endpoints |
| **Max tokens** | Configurable per request or per model |
| **Fused batch decode** | Concurrent multi-sequence decode on shared KV cache |
| **Speculative decoding** | N-gram pattern-based and draft-model token prediction for faster generation |

---

## Worker Subprocess Isolation

The inference worker runs as an independent subprocess, communicating with the main API server via JSON messages over stdin/stdout.

| Feature | Details |
|---------|---------|
| **Crash isolation** | Worker crash doesn't bring down the API server — auto-restart by supervisor |
| **Memory isolation** | Worker gets its own process memory budget, independent of main app |
| **Communication** | Bidirectional JSON message protocol over stdin/stdout |
| **Health reporting** | Periodic memory stats pushed to parent every 5 seconds |
| **Code signing** | Ad-hoc signed for macOS process integrity compliance |
| **Supervisor** | Monitors worker health, restarts on unexpected exit, handles backpressure |

---

## Multi-Modal Support

Full vision-language model (VLM) pipeline for image understanding.

| Feature | Details |
|---------|---------|
| **Image inputs** | Base64 data URIs, HTTP URLs, local file paths |
| **VLM architectures** | Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA, Gemma3, Phi-3-Vision, Pixtral, Molmo, Idefics3, InternVL, PaliGemma, DeepSeek-VL2, and more |
| **Vision feature cache** | In-memory LRU (20 entries) + optional SSD persistence, SHA-256 hashing, per-model isolation |
| **API** | Standard OpenAI `image_url` content parts in chat messages |

---

## API Compatibility

### OpenAI-Compatible Endpoints (Port 8080)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
| `POST` | `/v1/completions` | Legacy text completions (streaming + non-streaming) |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/rerank` | Document reranking |
| `POST` | `/v1/responses` | OpenAI Responses API |
| `GET` | `/v1/responses/{id}` | Retrieve stored response |
| `DELETE` | `/v1/responses/{id}` | Delete stored response |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text |
| `POST` | `/v1/audio/speech` | Text-to-speech |
| `GET` | `/v1/audio/voices` | List TTS voices |
| `GET` | `/v1/audio/languages` | List supported languages |

### Anthropic-Compatible Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/messages` | Anthropic Messages API (streaming + non-streaming) |
| `POST` | `/v1/messages/count_tokens` | Token counting |

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (status, GPU memory, loaded models, MCP) |
| `GET` | `/v1/stats` | Session and all-time inference metrics |
| `POST` | `/v1/mcp/execute` | Execute MCP tool call |

---

## Structured Output

### JSON Mode (`response_format: json_object`)

Character-level finite-state machine constraining output to valid JSON. Tracks 12 internal states for objects, arrays, strings, numbers, and literals.

### Schema-Guided Mode (`response_format: json_schema`)

Full JSON Schema parsing into a type tree supporting:
- `object` (with required field tracking)
- `array`, `string`, `integer`, `number`, `boolean`, `null`
- `enum` (string value restriction)
- `anyOf`, `oneOf`, `allOf` composition
- Schema state tracked alongside JSON state for field-level enforcement
- Token-level logit masking based on allowed characters per state

---

## Tool Calling

Automatic detection and parsing of tool call output across 7 format families:

| Format | Pattern |
|--------|---------|
| **XML** | `<tool>{"name": "...", "arguments": {...}}</tool>` or `<function=name>...` |
| **Bracket** | `[TOOL_CALLS] [{"name": ...}]` |
| **Marker** | `<\|tool_call\|>...<\|/tool_call\|>` |
| **Namespaced XML** | `<ns:tool_call><invoke name="...">...` |
| **GLM** | `☐function_name☐key■value■` delimiter pairs |
| **Gemma** | `call:functionName {key: value}` |
| **Thinking fallback** | Extracts tool calls from within `<think...>` blocks |

A stream filter suppresses tool markup during streaming so end users see clean content.

---

## Control Token & Thinking Filtering

Real-time control token detection and filtering to keep generation output clean and prevent runaway loops.

### TurnStopProcessor

Detects turn separator patterns in generated text and forces EOS token emission. Critical for models (e.g., Qwen3.6) that use semantic turn markers but rarely emit configured EOS tokens.

| Feature | Details |
|---------|---------|
| **Pattern matching** | Configurable stop patterns (e.g., `<|turn|>`, `<|im_end|>`) |
| **State machine** | Active → StopDetected → Done lifecycle |
| **EOS forcing** | Masks all non-EOS logits to `-inf` on pattern detection |
| **Accumulation** | Tracks full decoded sequence, not just individual tokens |

### Thinking Tag Architecture

Semantic thinking tags pass through to end users while protocol-level control tokens are stripped:

| Behavior | Example |
|----------|---------|
| **Pass through** | `<think>Reasoning here</think>` — user sees thinking content |
| **Filtered** | `<|turn|>`, `<|im_end|>` — stripped before streaming output |
| **Implicit open** | `ThinkingParser` handles responses that close `</think>` without opening tag |
| **Partial buffering** | Buffers incomplete control tokens in SSE streams to prevent leaked fragments |

### Control Token Filtering Pipeline

- Generic streaming filter scrubs control tokens across all agent output formats
- Multi-step agent conversation protection — no control token leakage between turns
- Regex-only token matching option for custom token patterns

---

## KV Cache & Prefix Sharing

Block-level paged KV cache inspired by vLLM, enabling cross-session prefix reuse.

| Feature | Details |
|---------|---------|
| **Block size** | 64 tokens (configurable) |
| **Hash algorithm** | SHA-1 chain hashing (parent hash + tokens + model name) |
| **Block pool** | Doubly-linked free list, reference counting, copy-on-write forking |
| **SSD persistence** | SafeTensors format, 16-bucket sharded directories, async GCD write queue |
| **SSD capacity** | 100 GB default with LRU eviction |
| **Cache types** | KVCacheSimple, RotatingKVCache, QuantizedKVCache, ChunkedKVCache, MambaCache, ArraysCache, CacheList |
| **Stats** | Hits, misses, tokens saved, evictions, shared blocks, SSD block count/size |

---

## TurboQuant KV Compression

Configurable per-model KV cache quantization for memory-constrained scenarios.

| Bits | Compression Ratio | Use Case |
|------|-------------------|----------|
| 2-bit | **8.0×** | Extreme memory pressure |
| 3-bit | **5.33×** | High memory pressure |
| 4-bit | **4.0×** | Balanced (recommended default) |
| 6-bit | **2.67×** | Quality-sensitive |
| 8-bit | **2.0×** | Minimal quality loss |

- **Group size**: 32, 64 (default), 128
- **Auto-recommendation**: Based on model size / available memory ratio and context length
- **Admin endpoint**: `GET /admin/api/turboquant` — view active configurations
- **Leverages**: mlx-swift-lm's `QuantizedKVCache` with optimized Metal kernels

---

## Continuous Batching

Priority-aware request batching for high-throughput concurrent inference.

| Feature | Details |
|---------|---------|
| **Max batch size** | 8 (configurable) |
| **Priority levels** | Low, Normal, High |
| **Preemption** | Lower-priority requests preempted for higher-priority |
| **Metrics** | Active requests, queue depth, total queued/completed/preempted, peak active, average wait time |
| **Abortion** | Per-request cancellation support |

---

## Speculative Decoding

Two complementary approaches accelerate generation by predicting and verifying multiple tokens in parallel.

### N-gram Speculative Decoding

Pattern-based draft token generation built into `FusedBatchScheduler`:

| Feature | Details |
|---------|---------|
| **Method** | N-gram pattern matching from recent context window |
| **Draft length** | Configurable number of candidate tokens per step |
| **Verification** | Target model verifies all draft tokens in parallel |
| **Overhead** | Zero secondary model — purely algorithmic |

### Draft-Model Speculative Decoding

Dedicated smaller draft model via `SpeculativeTokenIterator`:

| Feature | Details |
|---------|---------|
| **Draft model** | Smaller, faster model generates candidate tokens |
| **Target model** | Full-size model verifies candidates in a single forward pass |
| **Throughput gain** | Significant speedup for memory-bandwidth-bound generation |
| **Use case** | Large models where each token generation is memory-limited |

---

## Session Management

Persistent conversational sessions with KV cache reuse across turns.

| Feature | Details |
|---------|---------|
| **Max sessions** | 64 concurrent |
| **Session TTL** | 1800 seconds (auto-eviction) |
| **Persistence** | Save/restore KV cache to SafeTensors files |
| **Forking** | Deep-copy a session's KV cache into a new independent session |
| **Admin API** | List, delete, save, fork sessions |

---

## Embeddings & Reranking

### Embedding Service
- **Models**: BERT, XLM-RoBERTa, ModernBERT, SigLIP, ColQwen2.5, NomicBERT, Jina v3
- **Batching**: Pad-and-stack with attention masking
- **Output**: Normalized pooled embeddings
- **Endpoint**: `POST /v1/embeddings` (OpenAI-compatible)

### Reranker Service
- **Method**: Cross-encoder pair scoring (query `</s></s>` document)
- **Normalization**: Min-max score normalization across documents
- **Top-N filtering**: Return only top N results
- **Endpoint**: `POST /v1/rerank` (Cohere/Jina-compatible)

---

## Audio — Speech-to-Text & Text-to-Speech

On-device audio processing using Apple's native Speech and AVFoundation frameworks.

### Speech-to-Text
- **Engine**: Apple `SFSpeechRecognizer` (on-device)
- **Input**: Base64-encoded audio data
- **Language**: Auto-detect or specify (`en-US`, `zh-CN`, etc.)
- **Output**: Transcribed text, language, duration, timestamped segments
- **Endpoint**: `POST /v1/audio/transcriptions`

### Text-to-Speech
- **Engine**: Apple `AVSpeechSynthesizer`
- **Voices**: System voices (default, enhanced, premium quality tiers)
- **Parameters**: Voice selection, speed control, language
- **Output**: 16-bit PCM WAV audio
- **Endpoint**: `POST /v1/audio/speech`

### Discovery
- `GET /v1/audio/voices` — list all available TTS voices with identifiers and quality
- `GET /v1/audio/languages` — list supported language codes

---

## MCP — Model Context Protocol

Multi-server MCP client with tool discovery and execution.

| Feature | Details |
|---------|---------|
| **Transports** | `stdio` (subprocess), `sse` (HTTP SSE), `streamable-http` (HTTP POST) |
| **Protocol version** | `2024-11-05` |
| **Tool discovery** | Automatic `tools/list` on connection |
| **Namespacing** | Tools namespaced as `{server}__{tool}` to avoid collisions |
| **Timeouts** | Per-server configurable |
| **Admin API** | Configure, enable/disable, list tools and server status |

---

## Agent Integration

Built-in support for external AI coding agents via auto-generated configuration. Agents connect through NovaMLX's OpenAI-compatible API as their LLM backend.

| Agent | Description | Config Format |
|-------|-------------|---------------|
| **OpenClaw** | Open-source AI agent framework with plugin system and tool use | JSON (`~/.openclaw/openclaw.json`) |
| **Hermes Agent** | Autonomous AI agent with multi-step reasoning and tool execution | YAML (`~/.hermes/config.yaml`) |
| **OpenCode** | Terminal-based AI programming assistant with code editing | Environment variable |

| Feature | Details |
|---------|---------|
| **Auto-detection** | Scans `$PATH` and common install locations for agent binaries |
| **Config generation** | One-click config with correct port, API key, and model settings |
| **Copy to clipboard** | Quick copy for pasting into agent config files |
| **Install link** | Direct link to each agent's GitHub repository |
| **GUI** | Dedicated Agents tab in the menu bar app |

---

## Model Management

### Admin Endpoints (Port 8081, Bearer auth)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/models` | List all models (downloaded, loaded status) |
| `POST` | `/admin/models/download` | Download from HuggingFace |
| `POST` | `/admin/models/load` | Load model into GPU memory |
| `POST` | `/admin/models/unload` | Evict from GPU memory |
| `DELETE` | `/admin/models/{id}` | Delete model files |
| `POST` | `/admin/models/discover` | Scan filesystem for new models |
| `GET` | `/admin/models/{id}/settings` | Get per-model settings |
| `PUT` | `/admin/models/{id}/settings` | Update per-model settings |
| `GET` | `/admin/api/memory` | Memory state and pressure monitoring |

### Model Eviction
- **LRU eviction** of unpinned models when memory budget exceeded
- **TTL-based auto-unload** (configurable per model)
- **OS memory pressure handler**: Critical → evict all unpinned; Warning → evict to 1 unpinned
- **Periodic monitoring** (60s): GPU memory >80% triggers eviction

---

## Memory Management

System-level memory pressure monitoring with automatic model eviction via `ProcessMemoryEnforcer`.

### ProcessMemoryEnforcer

Actor-based memory pressure manager with configurable policies:

| Feature | Details |
|---------|---------|
| **Polling** | 1-second memory pressure monitoring loop |
| **Soft limit** | Triggers warning log when exceeded |
| **Hard limit** | Forces immediate eviction of unpinned models |
| **Auto mode** | Reserves `max(4GB, min(8GB, physMem/5))` for system, rest available to NovaMLX |
| **Percent mode** | User-defined percentage of physical RAM (e.g., `80%`) |
| **Fixed mode** | Absolute byte limit (e.g., `24GB`, `4096MB`) |
| **OS integration** | Responds to macOS memory pressure notifications |
| **GUI config** | Mode picker in Settings with conditional inputs for percent/fixed values |

### Memory Budget Tracking

Per-sequence KV cache budget with admission control:

| Feature | Details |
|---------|---------|
| **Reservation** | Each sequence reserves KV cache budget before admission |
| **Release** | Budget freed on sequence completion |
| **Admission control** | Rejects requests when available budget insufficient |
| **TurboQuant boost** | KV quantization effectively doubles admission capacity |

---

## Per-Model Settings

Every inference parameter can be overridden per model, persisted to `model_settings.json`:

| Setting | Description |
|---------|-------------|
| `max_context_window` | Override context length |
| `max_tokens` | Max generation tokens |
| `temperature` | Sampling temperature |
| `top_p` / `top_k` / `min_p` | Sampling strategy |
| `frequency_penalty` / `presence_penalty` / `repetition_penalty` | Repetition control |
| `seed` | Deterministic generation |
| `ttl_seconds` | Auto-unload after idle |
| `model_alias` | Short alias for model ID |
| `is_pinned` | Exempt from eviction |
| `is_default` | Default model for API requests |
| `kv_bits` / `kv_group_size` | TurboQuant configuration |
| `thinking_budget` | Max thinking/reasoning tokens |
| `display_name` / `description` | Human-readable metadata |

---

## HuggingFace Integration

Browse, search, and download models directly from HuggingFace.

| Feature | Details |
|---------|---------|
| **Search** | Query HF Hub with optional MLX-only filter, sortable by trending/downloads |
| **Model info** | Architecture, tags, file listing, license |
| **Downloads** | Async file-by-file with progress tracking and cancellation |
| **Dashboard UI** | Search box, results table, download progress with cancel buttons |
| **Auto-register** | Downloaded models automatically discovered and registered |

---

## Cloud Model Proxy

Remote inference proxy that makes cloud-hosted models appear alongside local models.

| Feature | Details |
|---------|---------|
| **Protocol** | OpenAI-compatible API passthrough |
| **Streaming** | Full SSE streaming passthrough with token-by-token relay |
| **Model discovery** | Cloud models listed in `/v1/models` alongside local models |
| **Configuration** | Remote endpoint URL and API key per cloud model |
| **Use case** | Access models too large for local GPU, or add remote fallback capacity |

---

## Observability & Benchmarking

### Performance Benchmark
- **Prompt lengths**: Configurable (default: 1024, 4096)
- **Metrics**: TTFT (ms), generation tok/s, prefill tok/s, peak memory (GB), end-to-end latency (s)
- **Endpoints**: `POST /admin/api/bench/start`, `GET /admin/api/bench/status`, `POST /admin/api/bench/cancel`

### Perplexity Evaluation
- **Metrics**: Per-text perplexity, token count, average perplexity
- **Endpoints**: `POST /admin/api/ppl/start`, `GET /admin/api/ppl/status`, `POST /admin/api/ppl/cancel`

### Inference Metrics
- Session + persistent all-time counters
- Total requests, tokens generated, average tok/s
- Per-model request counts, cache hit rate
- GPU memory tracking

### System Monitoring
- Device info: chip, variant, memory, GPU/CPU cores, OS version
- Update checker: GitHub Releases API
- Stats clearing: Session and all-time

---

## Web Interfaces

### Chat UI (`/chat`)
Full-featured web chat application — no admin auth required.
- Dark theme with purple accent
- SSE streaming responses with stop button
- Model selector
- System prompt configuration modal
- Conversation history (localStorage, max 50)
- Markdown rendering with code syntax highlighting and copy buttons
- Image upload (drag-drop, paste, file picker) for VLM
- Thinking/reasoning bubble display
- Image lightbox modal
- API key input with persistent storage
- Mobile-responsive sidebar

### Admin Dashboard (`/admin/dashboard`)
Operational dashboard — requires admin auth.
- Status cards (health, GPU memory)
- Device info table
- MCP servers status
- Benchmark runner
- Perplexity runner
- HuggingFace model browser (search, download, progress)
- Download task management with cancel
- Stats management (clear session / all-time)
- Auto-refresh every 5 seconds
- Responsive layout

---

## macOS Menu Bar App

Native SwiftUI menu bar extra with brain icon.

| Feature | Details |
|---------|---------|
| **Status tab** | Running indicator, server address, loaded models, GPU memory, active requests, uptime, tok/s |
| **Models tab** | Loaded model list with indicators, downloaded count, disk usage |
| **Agents tab** | External agent detection (OpenClaw, Hermes, OpenCode), config generation, install links |
| **Settings tab** | View/Edit config toggle, memory limit configuration, model settings, server management |
| **Polling** | 2-second stats refresh |
| **Window** | 280×200pt floating panel |

---

## Internationalization

Full i18n system supporting 9 languages across all GUI views and web interfaces.

| Language | Code |
|----------|------|
| English | `en` |
| 简体中文 | `zh-Hans` |
| 繁體中文 (香港) | `zh-Hant-HK` |
| 繁體中文 (台灣) | `zh-Hant-TW` |
| 日本語 | `ja` |
| 한국어 | `ko` |
| Français | `fr` |
| Deutsch | `de` |
| Русский | `ru` |

| Feature | Details |
|---------|---------|
| **Pattern** | `L10n.tr("key.path")` with compile-time key validation |
| **Coverage** | Settings, Agents, Chat, Menu Bar, Dashboard — all user-facing strings |
| **Fallback** | English fallback for missing translations |
| **Web UI** | Chat and Dashboard also fully internationalized |

---

## Security & Middleware

| Layer | Implementation |
|-------|---------------|
| **API auth** | Bearer token on port 8080 (`APIKeyAuthMiddleware`) |
| **Admin auth** | Bearer token or `X-Admin-Key` header on port 8081 (`AdminAuthMiddleware`) |
| **CORS** | `Access-Control-Allow-Origin: *` with full method/header support |
| **Request ID** | `x-request-id` header pass-through or auto-generation |
| **Error handling** | OpenAI-compatible JSON error bodies with proper HTTP status codes |
| **Admin isolation** | Admin API disabled entirely when no API keys configured |
| **Worker signing** | Ad-hoc code signing for worker subprocess to satisfy macOS process integrity |

---

## Homebrew Distribution

Install NovaMLX via Homebrew with automatic updates.

```bash
brew tap cnshsliu/nova
brew install --cask novamlx
```

| Feature | Details |
|---------|---------|
| **Tap** | `cnshsliu/nova` Homebrew tap |
| **Cask** | `novamlx` — installs `.app` bundle to `/Applications` |
| **Updates** | `brew upgrade --cask novamlx` fetches latest GitHub Release |
| **Version check** | Built-in update checker queries GitHub Releases API |

---

## Configuration

### Server Configuration (`ServerConfig`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `127.0.0.1` | Bind address |
| `port` | `8080` | Inference API port |
| `adminPort` | `8081` | Admin API port |
| `apiKeys` | `[]` | API keys (empty = auth disabled) |
| `maxConcurrentRequests` | `16` | Max concurrent requests |
| `requestTimeout` | `300s` | Request timeout |
| `contextScalingTarget` | `nil` | Scale reported token counts for billing compatibility |
| `maxProcessMemory` | `auto` | Process memory limit: `auto`, percent (`80%`), or absolute (`24GB`, `4096MB`) |
| `maxRequestSizeMB` | `100` | Maximum request body size in megabytes |

### Engine Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `kvCacheCapacity` | `1024` | KV cache block capacity |
| `maxMemoryMB` | `2048` | Max GPU memory for models |
| `maxConcurrent` | `8` | Max concurrent inference tasks |

### Prefix Cache Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `blockSize` | `64` | Tokens per cache block |
| `maxBlocks` | `4096` | Maximum cache blocks |
| `ssdMaxSizeBytes` | `100 GB` | Max SSD cache size |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    macOS Menu Bar App                     │
│     Status • Models • Agents • Settings • Dashboard      │
├──────────────────────────────────────────────────────────┤
│                   NovaMLX API Server                      │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │  Inference API   │  │        Admin API              │  │
│  │  (port 8080)     │  │        (port 8081)            │  │
│  │  • OpenAI        │  │  • Model Management           │  │
│  │  • Anthropic     │  │  • Per-Model Settings         │  │
│  │  • Embeddings    │  │  • Benchmarking               │  │
│  │  • Reranking     │  │  • HuggingFace Browser        │  │
│  │  • Audio STT/TTS │  │  • TurboQuant Config          │  │
│  │  • MCP Tools     │  │  • Memory Monitoring          │  │
│  │  • Cloud Proxy   │  │  • Dashboard UI               │  │
│  └────────┬─────────┘  └──────────────┬───────────────┘  │
│           └──────────┬────────────────┘                   │
│                      ▼                                    │
│  ┌──────────────────────────────────────────────────┐    │
│  │            Inference Service                      │    │
│  │  • Continuous Batcher  • Session Manager          │    │
│  │  • Speculative Decode  • Engine Pool              │    │
│  │  • Model Settings      • MCP Client Manager       │    │
│  └────────────────────────┬─────────────────────────┘    │
│                           ▼                               │
│  ┌──────────────────────┬──────────────────────────┐     │
│  │   MLX Engine Process │  Worker Subprocess        │     │
│  │  • LLM/VLM Generation│  • stdin/stdout JSON IPC  │     │
│  │  • Structured Output  │  • ProcessMemoryEnforcer  │     │
│  │  • TurboQuant         │  • WorkerSupervisor       │     │
│  │  • Prefix/SSD Cache   │  • GPU-Isolated Memory    │     │
│  │  • Vision Feature     │                           │     │
│  │  • TurnStopProcessor  │                           │     │
│  │  • ControlTokenFilter │                           │     │
│  └────────────────────┬─┴──────────────────────────┘     │
│                       ▼                                   │
│  ┌──────────────────────────────────────────────────┐    │
│  │          MLX / Apple Silicon GPU                  │    │
│  │  • Lazy Evaluation  • Unified Memory  • Metal    │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Build
swift build -c release

# Run with API key
./build/release/NovaMLX --api-key sk-your-key

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```
