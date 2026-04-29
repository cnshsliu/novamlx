# Development Guide

> This document is for contributors and developers who want to build, modify, or extend NovaMLX.

---

## Architecture

NovaMLX is a modular Swift package with 11 targets:

```
┌──────────────────────────────────────────────────────┐
│                    NovaMLXApp                         │
│              (Entry Point + Menu Bar)                 │
├──────────┬───────────┬───────────┬───────────────────┤
│ MenuBar  │    API    │ Inference │   ModelManager     │
│(SwiftUI) │(Hummingbird)│(Batcher)│ (Download/Registry)│
├──────────┴─────┬─────┴─────┬─────┴───────────────────┤
│                │  Engine   │                         │
│                │   (MLX)   │                         │
├────────────────┼───────────┤                         │
│                │PrefixCache│                         │
│                │(Paged+SSD)│                         │
├────────────────┴───────────┴─────────────────────────┤
│                  Core + Utils                         │
└──────────────────────────────────────────────────────┘
```

| Module | Purpose |
|--------|---------|
| **NovaMLXCore** | Foundation types, protocols, configuration |
| **NovaMLXUtils** | Logging, monitoring, tool call parsing, device info |
| **NovaMLXPrefixCache** | Paged KV cache with SSD persistence (64-token blocks, SHA-1 hashing) |
| **NovaMLXEngine** | MLX model loading, generation, structured output, custom Metal kernels |
| **NovaMLXInference** | Inference service, continuous batching, streaming |
| **NovaMLXModelManager** | Model registry, HuggingFace download, discovery |
| **NovaMLXAPI** | HTTP server — OpenAI + Anthropic API, admin endpoints |
| **NovaMLXMCP** | Model Context Protocol client (stdio, SSE, streamable-http) |
| **NovaMLXMenuBar** | Native SwiftUI menu bar app + dashboard |
| **NovaMLXCLI** | Command-line management tool (`nova`) |
| **NovaMLXApp** | Application entry point |

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| [mlx-swift](https://github.com/ml-explore/mlx-swift) | 0.31.3 | Apple MLX framework (GPU compute, Metal) |
| [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) | 2.31.3 | LLM model implementations |
| [swift-transformers](https://github.com/huggingface/swift-transformers) | 1.1.0 | HuggingFace model/tokenizer loading |
| [hummingbird](https://github.com/hummingbird-project/hummingbird) | 2.0.0 | HTTP server framework |
| [swift-log](https://github.com/apple/swift-log) | 1.6.0 | Structured logging |
| [swift-async-algorithms](https://github.com/apple/swift-async-algorithms) | 1.0.0 | Async stream algorithms |

---

## Building

```bash
# Clone
git clone https://github.com/novamlx/novamlx.git
cd novamlx

# Build (auto-patches dependencies and compiles Metal shaders)
./build.sh -c release

# Zero-warning build (CI standard)
swift build -c release -Xswiftc -warnings-as-errors

# Build only the CLI
swift build -c release --target NovaMLXCLI
```

### Build Script

`build.sh` performs:
1. `swift package resolve`
2. Patches `mlx-swift` C headers (replaces `_Complex*` with `void*`)
3. Patches `mlx-swift-lm` `AttentionUtils.swift` (injects fused SDPA hook)
4. Compiles MLX Metal shaders
5. `swift build`

---

## Testing

```bash
# All tests
swift test

# Specific target
swift test --filter NovaMLXCoreTests

# Benchmark suite (requires GPU + model)
.build/release/NovaMLXBenchmarkRunner
```

### Test Targets

| Target | Coverage |
|--------|----------|
| NovaMLXCoreTests | Types, config, protocols |
| NovaMLXEngineTests | Model loading, generation |
| NovaMLXInferenceTests | Batching, streaming |
| NovaMLXModelManagerTests | Download, registry |
| NovaMLXAPITests | HTTP endpoints, E2E |
| NovaMLXPrefixCacheTests | Paged cache, SSD |
| NovaMLXBenchTests | Performance benchmarks |

---

## API Reference

### Inference API (Port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI chat completions (streaming + non-streaming) |
| `/v1/completions` | POST | OpenAI text completions |
| `/v1/embeddings` | POST | Text embeddings |
| `/v1/rerank` | POST | Document reranking |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/responses/{id}` | GET/DELETE | Retrieve/delete stored responses |
| `/v1/messages` | POST | Anthropic Messages API |
| `/v1/messages/count_tokens` | POST | Token counting |
| `/v1/audio/transcriptions` | POST | Speech-to-text |
| `/v1/audio/speech` | POST | Text-to-speech |
| `/v1/audio/voices` | GET | List TTS voices |
| `/v1/audio/languages` | GET | List supported languages |
| `/v1/models` | GET | List loaded models |
| `/v1/stats` | GET | Inference metrics |
| `/v1/mcp/execute` | POST | MCP tool execution |
| `/v1/batch/completions` | POST | Fused batch generation |
| `/chat` | GET | Built-in web chat UI |
| `/health` | GET | Health check |

### Admin API (Port 8081, Bearer auth when apiKeys configured)

| Group | Endpoints |
|-------|-----------|
| Models | `GET/POST/DELETE /admin/models/*` |
| Sessions | `GET/DELETE /admin/sessions/*` |
| Cache | `GET/DELETE /admin/cache/{modelId}/*` |
| TurboQuant | `GET/PUT/DELETE /admin/api/turboquant/*` |
| Adapters | `GET/POST /admin/adapters/*` |
| HuggingFace | `GET/POST /admin/api/hf/*` |
| Benchmarking | `POST/GET /admin/api/bench/*` |
| Perplexity | `POST/GET /admin/api/ppl/*` |
| Device Info | `GET /admin/api/device-info` |
| Grammar | `POST /admin/api/grammar/validate` |
| Dashboard | `GET /admin/dashboard` |

---

## Project Structure

```
Sources/
├── NovaMLXApp/            # Entry point
├── NovaMLXAPI/            # HTTP server (Hummingbird)
│   ├── APIServer.swift    # All routes + inline dashboard HTML
│   ├── OpenAITypes.swift  # OpenAI request/response types
│   ├── AnthropicTypes.swift
│   ├── ProductionMiddleware.swift  # Rate limiting, security
│   └── ChatHTML.swift     # Built-in chat UI
├── NovaMLXCLI/            # nova command-line tool
├── NovaMLXCore/           # Types, protocols, config
├── NovaMLXEngine/         # MLX inference engine
│   ├── MLXEngine.swift    # Core: model loading, generation
│   ├── ContinuousBatcher.swift  # Priority-aware batching
│   ├── FusedQuantizedSDPA.swift # Custom Metal kernel
│   ├── JSONLogitProcessor.swift # Structured output
│   ├── ToolCallParser.swift     # 7-format tool detection
│   └── ...
├── NovaMLXInference/      # Batching, streaming, sessions
├── NovaMLXMenuBar/        # Native macOS UI
├── NovaMLXMCP/            # MCP client
├── NovaMLXModelManager/   # Model download/registry
├── NovaMLXPrefixCache/    # Paged KV cache
└── NovaMLXUtils/          # Logging, monitoring
Tests/                     # 7 test targets
Scripts/
├── patch-mlx-complex.py   # MLX compatibility patches
├── patch-fused-sdpa.py    # Fused SDPA kernel injection
├── package-dmg.sh         # DMG packaging
└── cli-placeholder.swift  # Legacy placeholder
```

---

## Creating a Release

Releases are built automatically via GitHub Actions when you push a tag:

```bash
git tag v1.0.0
git push origin v1.0.0
```

This triggers `.github/workflows/release.yml` which:
1. Builds on macOS 15 (Apple Silicon)
2. Runs tests
3. Creates a tarball + DMG
4. Creates a GitHub Release with attached binaries

Manual DMG:

```bash
./Scripts/package-dmg.sh
```

---

## Key Technical Details

### Custom Metal Kernels

NovaMLX includes a fused quantized SDPA Metal kernel (`FusedQuantizedSDPA.swift`) that fuses quantized attention decode into a single GPU kernel. This is JIT-compiled via `MLXFast.metalKernel()` and injected into `mlx-swift-lm` at build time.

Data format for affine KV quantization:
- Weights: uint32 packed (32/bits values per word)
- Scales/biases: float16 (same as model dtype)
- Dequantization: `value * scale + bias`

### Paged KV Cache

Block-level (64 tokens/block) KV cache with:
- SHA-1 chain hashing for deduplication
- Doubly-linked free list with reference counting
- Copy-on-write session forking
- SSD persistence (SafeTensors format, 16-bucket sharding)

### Thinking Model Detection & Streaming

NovaMLX automatically detects thinking/reasoning models at load time by inspecting the model's `chat_template` (from `tokenizer_config.json` or `chat_template.jinja`). If the template contains any of `<think`, `</think`, `<thinking`, `</thinking`, the model is flagged as a thinking model — no manual configuration or model name matching needed. New thinking models are supported automatically as long as their chat template includes these tags.

**How it works:**
- `ModelContainer.detectThinkingModel(for:)` in `MLXEngine.swift` reads the chat template and checks for thinking tags
- `ThinkingParser` in `ThinkingParser.swift` receives `expectImplicitThinking=true` for detected thinking models, which causes it to stream thinking tokens immediately (rather than buffering until `</think` is found)
- The API layer (`APIServer.swift`) calls `detectThinkingModel` to initialize the correct `ThinkingParser` for each request
- Both OpenAI and Anthropic streaming endpoints send `reasoning_content` / `thinking_delta` events in real-time as thinking tokens are generated

**Two thinking modes:**
- **Implicit** (Qwen3, DeepSeek-R1): chat template injects `<think...>` into the prompt, model only generates `</think...>`. The parser treats all content before the first close tag as thinking.
- **Explicit**: model generates both `<think...>` and `</think...>` tags. Standard parsing applies.

**Web UI:** The built-in chat page shows a Neural Pulse animation during thinking (real-time token count, speed, ghost preview of latest tokens), collapsing to a "Thought for Xs · N words" badge after completion.

### Security

- API key authentication (Bearer token) on both ports
- Rate limiting (token bucket, per-API-key)
- Security headers middleware
- Request size limits (100MB default)

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Ensure `swift build -c release -Xswiftc -warnings-as-errors` passes with zero warnings
4. Add tests
5. Submit a pull request
