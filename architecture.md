# NovaMLX Software Architecture

> A blazing-fast local LLM/VLM inference server for Apple Silicon, written in pure Swift.
> Exposes OpenAI-compatible and Anthropic-compatible APIs with a native macOS menu bar app.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Module Dependency Graph](#module-dependency-graph)
- [Module Deep Dives](#module-deep-dives)
  - [NovaMLXCore](#novamlxcore--shared-foundation)
  - [NovaMLXUtils](#novamlxutils--cross-cutting-utilities)
  - [NovaMLXPrefixCache](#novamlxprefixcache--kv-cache-reuse)
  - [NovaMLXEngine](#novamlxengine--inference-engine)
  - [NovaMLXInference](#novamlxinference--orchestration-layer)
  - [NovaMLXModelManager](#novamlxmodelmanager--model-lifecycle)
  - [NovaMLXMCP](#novamlxmcp--model-context-protocol)
  - [NovaMLXAPI](#novamlxapi--http-api-layer)
  - [NovaMLXMenuBar](#novamlxmenubar--macos-gui)
  - [Executables](#executables)
- [Request Lifecycle](#request-lifecycle)
- [Memory Management Architecture](#memory-management-architecture)
- [Scheduling Architecture](#scheduling-architecture)
- [Chat Template Processing](#chat-template-processing)
  - [File-level reconciliation (rendering source of truth)](#file-level-reconciliation-rendering-source-of-truth)
  - [Load-time health check (`runChatTemplateSanityCheck`)](#load-time-health-check-runchattemplatesanitycheck)
  - [Family-level interpretation (output processor selection)](#family-level-interpretation-output-processor-selection)
  - [Format detection — multi-marker scan](#format-detection--multi-marker-scan)
  - [Diagnostic CLI](#diagnostic-cli)
- [Configuration Files Reference](#configuration-files-reference)
- [Structured Output Pipeline](#structured-output-pipeline)
- [Cloud Backend](#cloud-backend)
- [Configuration and Data Storage](#configuration-and-data-storage)
- [Diagnostic Playbook](#diagnostic-playbook)

---

## Project Overview

NovaMLX is a production-grade local inference server optimized for Apple Silicon. It transforms any Mac into a private, high-performance LLM/VLM API endpoint compatible with the OpenAI and Anthropic SDKs.

**Key capabilities:**
- Local inference via Apple MLX framework (GPU-accelerated on Metal)
- OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/embeddings`, etc.)
- Anthropic-compatible REST API (`/v1/messages`)
- Streaming SSE support for both API formats
- VLM (Vision-Language Model) support with image input
- Multi-model loading with LRU eviction and memory pressure handling
- Fused batch decoding for concurrent request serving
- Prefix KV-cache with SSD persistence
- Structured output (JSON schema, regex, GBNF grammar)
- Thinking/reasoning model support (Qwen3, DeepSeek, etc.)
- Tool calling with multi-format parser
- LoRA adapter management
- Cloud proxy for remote models
- Worker subprocess mode for crash isolation
- MCP (Model Context Protocol) integration
- Native macOS menu bar app + web dashboard
- CLI tool (`nova`) for terminal-based management
- 9-language localization

**Technology stack:** Swift 6.0, Swift Concurrency (actors, async/await), Apple MLX, Hummingbird HTTP server, SwiftUI, HuggingFace Hub.

---

## Module Dependency Graph

The project is organized into 12 Swift modules (8 libraries + 4 executables) with a clean layered dependency structure:

```
                         ┌─────────────┐
                         │  NovaMLXApp │  (macOS menu bar application)
                         └──────┬──────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                   │
     ┌────────┴───────┐  ┌─────┴──────┐  ┌────────┴────────┐
     │ NovaMLXMenuBar │  │ NovaMLXAPI │  │ NovaMLXInference │
     └────────┬───────┘  └─────┬──────┘  └────────┬────────┘
              │                │                    │
              │          ┌─────┼──────────┐         │
              │          │     │          │         │
              │    ┌─────┴──┐  │   ┌──────┴───┐    │
              │    │NovaMLXMCP│  │   │NovaMLX   │    │
              │    └─────┬──┘  │   │ModelManager│   │
              │          │     │   └──────┬───┘    │
              │          │     │          │         │
              └──────────┼─────┼──────────┼─────────┘
                         │     │          │
                    ┌────┴─────┴──────────┴──────┐
                    │       NovaMLXEngine         │
                    └────┬──────────────┬─────────┘
                         │              │
              ┌──────────┴──┐    ┌──────┴──────────┐
              │NovaMLXUtils  │    │NovaMLXPrefixCache│
              └──────┬───────┘    └──────┬──────────┘
                     │                   │
              ┌──────┴───────────────────┴──────┐
              │          NovaMLXCore             │
              └──────────────────────────────────┘

  External Dependencies:
    MLX, MLXNN, MLXRandom, MLXLLM, MLXVLM, MLXLMCommon, MLXEmbedders  (Apple MLX)
    Tokenizers, Hub                                                        (swift-transformers)
    Hummingbird, HummingbirdRouter                                         (HTTP server)
    swift-log, swift-async-algorithms                                      (Apple core libs)
    CryptoKit, CoreImage, Charts, SwiftUI, AppKit                          (Apple platforms)
```

**Layer descriptions:**

| Layer | Modules | Role |
|-------|---------|------|
| Foundation | NovaMLXCore | Domain types, config, auth, paths, worker protocol |
| Utilities | NovaMLXUtils | Logging, metrics, streaming detokenizer, thinking parser, tool call parser |
| Cache | NovaMLXPrefixCache | Prefix KV-cache with SSD persistence |
| Engine | NovaMLXEngine | MLX inference engine, batch scheduling, logit processing, model implementations |
| Orchestration | NovaMLXInference | Request routing, worker subprocess management, cloud proxy, memory pressure |
| Services | NovaMLXModelManager, NovaMLXMCP | Model download/registry, MCP tool integration |
| API | NovaMLXAPI | HTTP server (OpenAI + Anthropic), web UI, chat page |
| Presentation | NovaMLXMenuBar | macOS native GUI (menu bar popover + full window) |
| Entry Points | NovaMLXApp, NovaMLXWorker, NovaMLXCLI, NovaMLXBenchmarkRunner | Executables |

---

## Module Deep Dives

### NovaMLXCore -- Shared Foundation

The bedrock module. Every other module depends on this. It defines the domain model types, configuration, authentication, file paths, worker IPC protocol, and localization. No inference or UI logic lives here.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **Types.swift** | Central type definitions for the entire project | `InferenceRequest`, `InferenceResult`, `ChatMessage`, `Token`, `FinishReason`, `ModelIdentifier`, `ModelFamily`, `ModelType`, `ModelConfig`, `ServerConfig`, `ResponseFormat`, `ToolCallResult`, `ProcessMemoryLimit`, `InferenceEngineProtocol`, `ModelRegistryProtocol`, `TokenizerProtocol`, `NovaMLXError` |
| **NovaMLXPaths.swift** | All filesystem path resolution. Three-tier base dir: `~/.config/novamlx/path` > `NOVA_DIR` env > `~/.nova` | `NovaMLXPaths` (static enum with `baseDir`, `modelsDir`, `logFile`, `configFile`, `metricsFile`, `sessionsDir`, `chatHistoryDir`, `prefixCacheDir(for:)`, etc.) |
| **Configuration.swift** | Singleton actor managing runtime configuration state | `NovaMLXConfiguration.shared` (actor: models dir, server config, default model; JSON persistence) |
| **AuthClient.swift** | Cloud subscription authentication and validation | `AuthClient` (HTTP client), `CloudAuth.validate()` (cached session check), `AuthCache` (local token persistence with 5-min TTL) |
| **ModelSettings.swift** | Per-model override settings (sampling defaults, TTL, pinning, aliases) | `ModelSettings` (all-optional struct with `applySamplingOverrides(to:)` that merges into an `InferenceRequest`) |
| **WorkerProtocol.swift** | IPC contract between main process and worker subprocess | `WorkerMessage` (JSON envelope), `CodableInferenceRequest` (Codable-safe mirror of InferenceRequest), `WorkerMessageType` (string constants for load/unload/generate/stream/result/token/done/error/abort/ping/pong/memoryStats) |
| **Localization.swift** | Runtime localization manager | `L10n.shared` (ObservableObject, `tr(_:)` lookup, 9 languages, auto-detect from OS) |
| **LocalizationStrings.swift** | ~2200 lines of translated UI strings for 9 languages | `LocalizationStrings.all` (dictionary of language code to key-value string pairs) |

#### Internal Dependencies

```
Types.swift ───────────┐
NovaMLXPaths.swift ────┤   (leaf nodes, no internal deps)
LocalizationStrings.swift
                       │
        ┌──────────────┼──────────────┬──────────────────┐
        │              │              │                  │
AuthClient.swift  Configuration.swift  Localization.swift  ModelSettings.swift
(forks paths)    (paths + types)     (paths + strings)   (types)
                       │
                 WorkerProtocol.swift (types)
```

---

### NovaMLXUtils -- Cross-Cutting Utilities

Shared utility library used by nearly every module. Provides infrastructure for logging, metrics, streaming token decoding, thinking-block parsing, tool-call parsing, system monitoring, and device info.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **Logging.swift** | Centralized logging facade writing to both swift-log and a rotating log file | `NovaMLXLog` (debug/info/warning/error/request + `rotateLogFile()`) |
| **MetricsStore.swift** | Persistent metrics accumulator with auto-save | `MetricsStore` (request/token counts, per-model breakdown, cache hits, TPS; JSON persistence to `metrics.json`) |
| **StreamingDetokenizer.swift** | Incremental token-to-text decoder handling partial Unicode sequences | `StreamingDetokenizer` (addToken -> lastSegment pattern, buffers incomplete multi-byte chars) |
| **ThinkingParser.swift** | Streaming state machine for `<think...</think >` block parsing | `ThinkingParser` (feed token -> ParsedToken with .thinking/.content type; handles implicit open tags, Qwen3.6 `<|begin_of_thought|>` markers) |
| **ToolCallParser.swift** | Multi-format tool-call parser + streaming markup filter | `ToolCallParser.parse()` (7 formats: XML, bracket, marker, namespaced, GLM, Gemma, thinking-embedded), `ToolCallStreamFilter` (suppresses raw tool-call markup during streaming) |
| **DeviceInfo.swift** | Apple Silicon hardware detection | `DeviceInfo.current()` (chip, GPU cores, memory, OS version) |
| **SystemMonitor.swift** | Live CPU/memory metrics | `SystemMonitor.shared.currentStats()` (CPU usage via task_threads, physical memory via task_info) |
| **Lock.swift** | Thin `os_unfair_lock` wrapper | `NovaMLXLock` (withLock scope-based locking) |
| **Extensions.swift** | Convenience extensions on Foundation types | String.nilIfEmpty, Int.bytesFormatted, URL.fileName, Date.iso8601String, etc. |
| **UpdateChecker.swift** | GitHub Releases API checker with caching | `UpdateChecker.checkForUpdates()` (cached result, semver comparison) |
| **ChatTemplateRegistry.swift** | Singleton actor that loads `Resources/template-registry.json` (bundled) and overlays `~/.nova/templates/registry.json` (user). Drives template overrides, family detection extensions, and per-family hallucination/leakage patterns. Hot-reloadable via `reload()`. Lives in Utils (not Engine) so both Engine and ModelManager can consult it without circular dependencies. | `ChatTemplateRegistry.shared`, `ChatTemplateRegistry.FamilyConfig`, `templateOverride(...)`, `familyConfig(for:)`, `familyByModelType(_:)`, `familyByArchitecture(_:)` |
| **ChatTemplateProfileDrafter.swift** | Auto-drafts a test profile (`profiles/_drafts/<modelId>.json`) the first time an unknown model is loaded. Inspects `tokenizer_config.json` for thinking Jinja vars, modality markers (vision/audio/video), and EOS token; uses `ChatTemplateRegistry` family config for sampling/leakage seeds. Marked `_review_required: true` — operators promote drafts to permanent profiles by reviewing and moving out of `_drafts/`. | `ChatTemplateProfileDrafter.draftIfNeeded(modelId:modelDir:family:registryHasOverride:)` |
| **ChatTemplateUpstreamCheck.swift** | Optional SHA-256 fingerprint comparison of the local chat template against the upstream HuggingFace copy. Catches the case where `tokenizer_config.json.chat_template` itself is corrupt (something `ensureChatTemplate` cannot detect locally). **Off by default** — enable by setting `NOVAMLX_TEMPLATE_UPSTREAM_CHECK=1`. Results cached 24h per modelId. | `ChatTemplateUpstreamCheck.shared`, `check(modelId:localTemplate:)`, `isEnabled` |

#### Why These Exist Here (Not in Engine)

`ThinkingParser`, `ToolCallParser`, and `StreamingDetokenizer` live in Utils (not Engine) because they're consumed by both the engine (for inference-time processing) and the API layer (for response formatting and client delivery). This avoids a circular Engine -> API dependency.

---

### NovaMLXPrefixCache -- KV-Cache Reuse

Implements prefix KV-cache: when two requests share the same prompt prefix (e.g. system prompt + conversation history), the already-computed KV-cache blocks are reused instead of recomputed. This dramatically reduces TTFT for multi-turn conversations.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **PrefixCacheTypes.swift** | Core data types for the cache system | `BlockHash` (SHA-1 digest), `PrefixCacheConfig` (block size 64 tokens, max 4096 blocks, SSD 5GB, TTL 24h), `CacheBlock` (ref-counted, free-list node), `BlockTable` (per-request block mapping, supports forking), `PrefixCacheStats` |
| **BlockHasher.swift** | Content-addressable hashing with parent chaining | `BlockHasher.computeChainHashes()` (chained SHA-1: each block depends on parent hash, model name, and token IDs) |
| **PagedBlockPool.swift** | Paged block allocator with free-list, hash map, and ref counting | `PagedBlockPool` (allocate/free, `findSharedPrefix()` walks hash chain, `forkBlockTable()` for session forking, LRU eviction) |
| **CacheBlockExtractor.swift** | Extracts and reconstructs KV-cache data per block | `CacheBlockExtractor.extractBlockSlices()` (handles KVCacheSimple, RotatingKVCache, ChunkedKVCache, QuantizedKVCache, MambaCache, ArraysCache with per-type slicing), `reconstructKVCache()` (merges blocks by concatenating sequence dims) |
| **SSDCacheStore.swift** | Persists cache blocks to SSD as safetensors files | `SSDCacheStore` (atomic write via .tmp + rename, LRU index with TTL, background dispatch queue, crash-safe) |
| **PrefixCacheManager.swift** | Top-level orchestrator | `PrefixCacheManager.fetchPrefix()` (lookup + load), `storeCache()` (hash + allocate + save), `clear()`, `getStats()` |

#### Data Flow

```
Store path:
  tokenIds -> BlockHasher.computeChainHashes() -> PagedBlockPool.registerBlock()
           -> CacheBlockExtractor.extractBlockSlices() -> SSDCacheStore.saveBlock()

Fetch path:
  tokenIds -> PagedBlockPool.findSharedPrefix() -> (blockIds, remainingCount)
           -> SSDCacheStore.loadBlock() for each hit
           -> CacheBlockExtractor.reconstructKVCache() -> PrefixResult(cachedKV, cachedTokenCount, remainingTokenIds)
```

---

### NovaMLXEngine -- Inference Engine

The heart of NovaMLX. Manages model loading/unloading, tokenization, inference (generate + stream), batch scheduling, KV-cache management, chat template processing, logit manipulation, memory budgeting, and custom model architectures.

This is the largest module (~35 Swift files) organized into several subsystems.

#### Core Engine

| File | Purpose | Key Types |
|------|---------|-----------|
| **MLXEngine.swift** | Central inference engine. Model lifecycle, generate/stream, control tokens, memory | `MLXEngine` (loadModel, unloadModel, generate, stream, abort, getContainer), `ModelContainer` (loaded model + tokenizer + metadata), `Tokenizer` (Sendable wrapper) |
| **EnginePool.swift** | Thread-safe model pool with LRU eviction and pinning | `EnginePool` (add/remove/get, evictLRU, pin/unpin), `PooledModel` (container + load time + last access + pinned + estimated size) |
| **MLXSerializer.swift** | Global lock for serializing MLX GPU operations | `MLXSerializer.shared.perform()` (single lock ensuring one MLX operation at a time) |
| **ModelFamilyRegistry.swift** | Per-family optimization defaults and architecture mapping | `ModelFamilyRegistry.shared` (KV cache settings, prefill step size, context length, head dims, sampling defaults; maps architecture class names to ModelFamily) |
| **LocalTokenizerLoader.swift** | Loads tokenizers from local model directories | `TokenizerBridge` (adapts Tokenizers.Tokenizer to MLXLMCommon.Tokenizer), `LocalTokenizerLoader` |
| **ChatTemplateLibrary.swift** | Three-level template resolution: (1) user `.jinja` file at `~/.nova/templates/<id>.jinja`, (2) `ChatTemplateRegistry`-driven exact / repo-prefix / family+arch override (data-driven, hot-reloadable), (3) downloaded template fallback. Adding a fix for a problematic quant is a JSON edit, no recompile. | `ChatTemplateLibrary.resolve(modelId:family:architecture:downloadedTemplate:)` |
| **MLXEngine.swift `ensureChatTemplate`** | Reconciles `tokenizer_config.json.chat_template` (HF canonical) and standalone `chat_template.jinja`. Detects disagreement, quarantines stale `.jinja` (which swift-transformers would otherwise prefer), promotes orphaned `.jinja` content into `tokenizer_config.json`, applies `ChatTemplateLibrary` overrides, and emits a one-shot cleanup notice when a quarantine backup exists alongside a healthy config. **Refuses to inject a guessed fallback** — corrupting the prompt format causes silent inference failures that look like model hallucination. | `ensureChatTemplate`, `injectTemplate`, `quarantineJinjaIfPresent`, `promoteJinjaToConfig`, `readConfigChatTemplate` |
| **MLXEngine.swift `runChatTemplateSanityCheck`** | Load-time prompt-rendering health check. Sends a probe message (`ping_test_42`) through the loaded tokenizer and verifies: (a) `applyChatTemplate` doesn't throw, (b) output is non-empty, (c) no unrendered Jinja literals remain, (d) the user-content sentinel survives, (e) common control-token pairs are balanced, (f) only one family format is present (multi-family markers indicate corruption). All findings logged with `[ChatTemplateHealth]` prefix. | (private static helper) |
| **ChatTemplateDiagnostics.swift** | Read-only diagnostic snapshot for a model's chat-template state. Used by `nova chat-template diagnose <id>` and `/admin/api/chat-template/diagnose/{id}`. Returns a structured report: file presence + sizes, agreement status, format detection (multi-marker), registry override, family interpretation, and a list of health issues. | `ChatTemplateDiagnostics.diagnose(...)`, `ChatTemplateDiagnostics.Report` |

#### Scheduling

| File | Purpose | Key Types |
|------|---------|-----------|
| **FusedBatchScheduler.swift** | Production scheduler. Fuses multiple sequences into shared GPU forward passes. Memory-aware admission, auto-concurrency tuning, N-gram speculative decoding | `FusedBatchScheduler` (submit, submitStream, abort, metrics; ~1350 lines), `ActiveStreamSequence`, `FusedSchedulerMetrics` |
| **ContinuousBatcher.swift** | Non-fused scheduler for VLM, hybrid attention, sessions, grammar-constrained requests | `ContinuousBatcher` (submit, submitStream, abort; routes to engine.generate/stream directly), `BatcherMetrics` |
| **BatchScheduler.swift** | Simpler batch scheduler (prefill individually, then fused decode) | `BatchScheduler`, `PendingBatchRequest`, `ActiveBatchSequence` |
| **FusedBatchDecode.swift** | Fused batch KV-cache infrastructure for multi-sequence decode | `FusedBatchKVCache` (implements KVCache; pads/stacks per-sequence caches for batched attention), `FusedBatchDecoder` |
| **FusedQuantizedSDPA.swift** | Custom Metal kernel for fused quantized scaled dot-product attention | Generates parameterized Metal shaders for different head dims (64/96/128/256), group sizes, bit widths (4/8) |
| **FusedSDPARegistration.swift** | Legacy registration (now no-op; MLX handles fused SDPA internally) | -- |

#### Memory Management

| File | Purpose | Key Types |
|------|---------|-----------|
| **MemoryBudgetTracker.swift** | Actor tracking GPU memory budget across active inference sequences | `MemoryBudgetTracker` (canAdmit, reserve, release, updateActual; per-model ModelBudget with KV bytes estimation) |
| **ProcessMemoryEnforcer.swift** | Actor polling MLX memory and enforcing soft/hard limits | `ProcessMemoryEnforcer` (start/stop, status; evicts LRU unpinned models on soft limit, aborts + clears cache when only one model) |
| **TurboQuantCache.swift** | Per-model KV cache quantization configuration | `TurboQuantService` (setConfig, applyToGenerateParameters, autoConfigure based on model size/memory) |

#### Chat Sessions

| File | Purpose | Key Types |
|------|---------|-----------|
| **ChatSessionManager.swift** | Multi-turn conversation sessions with KV-cache persistence | `ChatSessionManager` (getOrCreate, saveSession, fork, removeSessions, listSessions), `SessionBox` |
| **VisionFeatureCache.swift** | In-memory + disk cache for VLM image embeddings | `VisionFeatureCache` (get/put with SHA256 image hashing, LRU eviction, custom binary format) |

#### Chat Template Processing

| File | Purpose | Key Types |
|------|---------|-----------|
| **ChatTemplateProcessor.swift** | Protocol defining control token behavior per model family | `ChatTemplateProcessor` (refineControlTokens, isThinkingModel, isImplicitThinkingModel, hallucinationPatterns, scrubControlTokens, trimControlTokens, shouldStopForHallucination) |
| **ChatTemplateProcessorRegistry.swift** | Factory mapping (ModelFamily, ChatTemplateFormat) -> processor | `ChatTemplateProcessorRegistry` (Qwen->ChatML/TurnRole, Gemma, Bailing, GPT-OSS->Harmony, default) |
| **ChatTemplateProcessors/SharedControlTokenLogic.swift** | Shared regex-based scrub/trim/filter utilities | `SharedControlTokenLogic` (regex scrubbing, close-variant generation, template token extraction) |
| **ChatTemplateProcessors/QwenChatMLProcessor.swift** | Qwen2/3/3.5/3.6 ChatML format. Stable when prompted in canonical ChatML; no hallucination detection needed. *Caveat:* if a stale `chat_template.jinja` corrupts the rendered prompt (see "Chat Template Processing"), the model may emit Bailing-style `<\|turn\|>` markers — this is **not** a processor bug, fix the file-layer template instead. | `refineControlTokens` (also lists `<\|turn\|>` / `<\|end_turn\|>` as defensive stop tokens), `hallucinationPatterns: []` |
| **ChatTemplateProcessors/QwenTurnRoleProcessor.swift** | Qwen3.6+ turn-role format | Turn refinement, bare multi-turn hallucination detection after 20+ tokens |
| **ChatTemplateProcessors/GemmaProcessor.swift** | Gemma3/4 series | Extra start/end_of_turn scrubbing, `<\|channel\>thought` thinking detection |
| **ChatTemplateProcessors/BailingProcessor.swift** | Bailing (Ling) series | Turn refinement, hallucination detection |
| **ChatTemplateProcessors/HarmonyProcessor.swift** | GPT-OSS (OpenAI Harmony format) | Multi-token structure scrubbing (`<\|channel\|>word<\|message\|>`), structural token exclusion |
| **ChatTemplateProcessors/DefaultProcessor.swift** | Fallback for unknown families | Format-based behavior, turn refinement for .turnRole format |

#### Logit Processing (Structured Output)

| File | Purpose | Key Types |
|------|---------|-----------|
| **CompiledSampler.swift** | MLX-compiled sampling + N-gram speculative decoding | `CompiledSampler` (sample with greedy/topP/temperature), `NGramSpeculator` (records sequences, proposes draft tokens), `SpeculativeDecoder` (orchestrates speculation, tracks acceptance rate) |
| **ComposedLogitProcessor.swift** | Chains grammar + penalty + turn-stop processors | `ComposedLogitProcessor` (applies in sequence: grammar -> penalty -> turn stop) |
| **JSONLogitProcessor.swift** | State machine (12 states) constraining output to valid JSON | `JSONLogitProcessor` (precomputes token masks per state; handles objects, arrays, strings, numbers, literals) |
| **GBNFLogitProcessor.swift** | Full GBNF grammar parser and token mask builder | `GBNFParser`, `GBNFLogitProcessor` (recursive matching: literals, char sets, rule references, sequences, repetitions, optionals) |
| **RegexLogitProcessor.swift** | Constrains output to match a regex pattern | `RegexLogitProcessor` (tests each character for partial/full match; forces EOS on completion) |
| **SchemaGuidedProcessor.swift** | JSON Schema-driven output constraint | `SchemaNode` (recursive schema parser: object/array/string/integer/number/boolean/null/anyOf/stringEnum), `SchemaGuidedProcessor` (state machine driven by schema) |
| **TokenMaskBuilder.swift** | Pre-decoded vocabulary for fast token masking | `TokenMaskBuilder` (buildMask from allowed chars, buildEOSMask; applies masks to logits via MLX.where) |
| **TurnStopProcessor.swift** | Detects turn separator and hallucination patterns, forces EOS | `TurnStopProcessor` (prevents infinite generation on models that never emit EOS) |

#### Services

| File | Purpose | Key Types |
|------|---------|-----------|
| **AdapterService.swift** | LoRA adapter lifecycle (load, unload, fuse, list, discover) | `AdapterService` (thread-safe, wraps MLXLLM LoRAContainer) |
| **EmbeddingService.swift** | Text embedding generation | `EmbeddingService` (load embedding model, compute embeddings with padding/masking/pooling) |
| **RerankerService.swift** | Document reranking by relevance | `RerankerService` (load reranker model, compute scores, normalize to [0,1], return top-N) |

#### Custom Model Architectures

| File | Purpose | Key Types |
|------|---------|-----------|
| **Models/CustomModelRegistration.swift** | Registers custom model types with MLX at first use | Registers `bailing_hybrid` and `deepseek_v4` with LLMTypeRegistry |
| **Models/BailingHybridModel.swift** | Bailing (Ling) hybrid: MLA attention + GLA + MoE (~860 lines) | `BailingHybridModel` (LLMModel, KVCacheDimensionProvider, LoRAModel); internal: MultiLinear, BailingHybridMLA, BailingHybridGLA (recurrent), BailingHybridGate (group-limited top-K), BailingHybridSparseMoeBlock; custom sanitize() for KV weight splitting |
| **Models/DeepseekV4Model.swift** | DeepSeek-V4: MLA + hash-routed MoE + HyperConnections (~720 lines) | `DeepseekV4Model` (LLMModel, KVCacheDimensionProvider, LoRAModel); internal: HyperConnection (Sinkhorn), Compressor (learned KV compression), Indexer (load-only stub), Attention (MLA with inverse RoPE), Gate (hash + score routing), SwitchGLU (routed experts with swiglu_limit) |

#### Internal Dependency Graph

```
MLXEngine (central hub)
  ├── EnginePool                    (model storage + LRU)
  ├── MemoryBudgetTracker           (KV memory budget)
  ├── ProcessMemoryEnforcer         (memory limit enforcement)
  ├── TurboQuantService             (KV quantization)
  ├── ChatSessionManager            (multi-turn sessions)
  ├── AdapterService                (LoRA adapters)
  ├── PrefixCacheManager            (prefix cache)
  ├── ModelFamilyRegistry           (per-family defaults)
  ├── LocalTokenizerLoader          (tokenizer bridge)
  ├── ChatTemplateLibrary           (template resolution)
  ├── CompiledSampler               (sampling + N-gram speculation)
  │     └── SpeculativeDecoder
  ├── ChatTemplateProcessorRegistry → per-family processors
  │     └── SharedControlTokenLogic
  ├── CustomModelRegistration
  │     ├── BailingHybridModel
  │     └── DeepseekV4Model
  ├── FusedBatchScheduler (production scheduler)
  │     ├── FusedBatchDecoder → FusedBatchKVCache
  │     ├── CompiledSampler + SpeculativeDecoder
  │     ├── MemoryBudgetTracker
  │     └── ChatTemplateProcessor
  ├── ContinuousBatcher (non-fused scheduler)
  │     └── MemoryBudgetTracker
  └── BatchScheduler (simple scheduler)
        └── FusedBatchKVCache, CompiledSampler

ComposedLogitProcessor (grammar chain)
  ├── JSONLogitProcessor ──┐
  ├── GBNFLogitProcessor  ─┤ all use TokenMaskBuilder
  ├── RegexLogitProcessor ─┤
  ├── SchemaGuidedProcessor┘
  └── TurnStopProcessor
```

---

### NovaMLXInference -- Orchestration Layer

The public-facing API through which the rest of the application accesses inference. Routes requests between three execution modes: local engine, worker subprocess, and cloud proxy.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **InferenceService.swift** | Central orchestrator. Routes to cloud/worker/batcher based on model type and features | `InferenceService` (generate, stream, abort, loadModel, unloadModel, stats, resolveModelId, forkSession, countTokens), `InferenceStats` |
| **WorkerSupervisor.swift** | Manages worker subprocess lifecycle with crash recovery | `WorkerSupervisor` (start/stop/ensureRunning, sendLoad/sendUnload/sendGenerate/sendStream/sendAbort; auto-restart with 2s cooldown), `WorkerMemoryStats` |
| **CloudBackend.swift** | Proxies to remote cloud API (OpenAI + Anthropic format) | `CloudBackend.shared` (actor; proxy/proxyStream/proxyAnthropic/proxyAnthropicStream; model discovery with 10-min cache) |
| **Pipelines.swift** | Convenience wrappers for direct engine access (bypasses InferenceService) | `LLMInferencePipeline` (complete/chat/streamChat), `VLMInferencePipeline` (analyze/streamAnalyze) |
| **BenchmarkService.swift** | Parameterized benchmarking across prompt lengths | `BenchmarkService` (startBenchmark, getActiveRun, cancelActiveRun), `BenchmarkResult` (TTFT, TPS, memory, latency) |
| **PerplexityService.swift** | Text perplexity evaluation | `PerplexityService` (startEvaluation, getActiveRun, cancelActiveRun), `PerplexityResult` |
| **MemoryPressureHandler.swift** | OS memory pressure monitoring + auto-eviction | `MemoryPressureHandler` (start/stop; DispatchSourceMemoryPressure + 60s periodic timer; evicts unpinned models on warning/critical) |

#### Request Routing Logic (in InferenceService)

```
Incoming Request
  │
  ├── Model ID ends with ":cloud"? → CloudBackend
  │
  ├── workerMode == true? → WorkerSupervisor
  │
  ├── VLM model? ──────────────┐
  ├── Hybrid linear attention? ─┤→ ContinuousBatcher
  ├── Has session ID? ─────────┤  (per-sequence decode)
  ├── Has grammar/regex/GBNF? ─┘
  │
  └── Standard LLM → FusedBatchScheduler
     (fused batch decode)
```

---

### NovaMLXModelManager -- Model Lifecycle

Handles model discovery, download, registration, deletion, and per-model settings persistence.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **ModelManager.swift** | Central model registry and download manager | `ModelManager` (register/unregister, startDownload with per-file progress, deleteModel, totalDiskUsage, discoverModels), `ModelRecord`, `DownloadStatus` |
| **ModelDiscovery.swift** | Scans local directories for MLX-compatible models | `ModelDiscovery.discover()` (classifies LLM/VLM/embedding by architecture; detects family, completeness, adapter status), `DiscoveredModel` |
| **HuggingFaceService.swift** | HuggingFace Hub client with parallel streaming downloads | `HuggingFaceService` (searchModels, getModelDetail, startDownload with resume/retry up to 20 attempts), `HFDownloadTask`, `FileProgress` |
| **ModelSettingsManager.swift** | Per-model settings persistence (default, pinned, alias) | `ModelSettingsManager` (getSettings/setSettings, getDefaultModelId, getPinnedModelIds, resolveAlias, resolveModelId) |

---

### NovaMLXMCP -- Model Context Protocol

Implements MCP client and server management for tool integration with external AI agents.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **MCPTypes.swift** | All MCP data types | `MCPTransport` (stdio/sse/streamableHTTP), `MCPServerConfig`, `MCPConfig`, `MCPTool` (with namespacedName "server__name"), `MCPToolResult`, `MCPServerState`, API request/response types |
| **MCPClient.swift** | Connects to a single MCP server via stdio or HTTP/SSE | `MCPClient` (connect, disconnect, execute; JSON-RPC message framing; initialization handshake + tool discovery) |
| **MCPManager.swift** | Manages multiple MCP clients with namespaced tool routing | `MCPManager` (loadConfig, getAllTools, getToolsForOpenAI, executeTool by "server__tool" name), `MCPServerStatus` |

---

### NovaMLXAPI -- HTTP API Layer

Exposes OpenAI-compatible and Anthropic-compatible REST APIs, admin endpoints, and a web-based dashboard/chat UI. Uses Hummingbird as the HTTP server framework.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **APIServer.swift** | Central server (~2700 lines). All routing, middleware, request handling | `NovaMLXAPIServer` (starts two Hummingbird instances: main API + admin API), `APIKeyAuthMiddleware`, `AdminAuthMiddleware`, `CORSMiddleware`, `RequestIDMiddleware`, `NovaMLXErrorMiddleware` |
| **OpenAITypes.swift** | All OpenAI API request/response types (~1070 lines) | `OpenAIRequest`, `OpenAIChatMessage`, `OpenAIResponse`, `OpenAIStreamChunk`, `OpenAIModelsResponse`, `EmbeddingInput/Request/Response`, `OpenAICompletionRequest/Response`, `OpenAIResponseRequest/Object`, `AdminModelStatus`, `AnyCodableDict/Value` |
| **AnthropicTypes.swift** | All Anthropic API request/response types (~415 lines) | `AnthropicRequest`, `AnthropicMessage`, `AnthropicContent/ContentBlock`, `AnthropicResponse`, `AnthropicStreamEvent` (with factory methods for SSE events), `AnyCodable` |
| **GrammarTypes.swift** | Structured output constraint types | `StructuredOutputOptions`, `JSONSchemaFormat`, `ResponseFormatJsonSchema` |
| **RerankTypes.swift** | Document reranking API types | `RerankRequest`, `RerankResponse`, `RerankResult`, `RerankUsage` |
| **ChatHistoryStore.swift** | File-system chat history persistence | `ChatHistoryStore.shared` (list/get/save/delete/search), `ChatRecord`, `ChatSummary` |
| **ChatHTML.swift** | Legacy standalone chat page HTML/CSS/JS generator (~874 lines) | `ChatHTML.render()` (full SPA with sidebar, markdown, syntax highlighting, SSE streaming, thinking panel, image upload, settings panel) |
| **ClientDetector.swift** | Detects agent vs chat client from HTTP headers | `ClientDetector.detect()` -> `ClientType` (.agentTool or .generalChat; agent tools get context scaling) |
| **OCROptimizer.swift** | Auto-optimizes parameters for OCR models (DeepSeekOCR, DotsOCR, GLM-OCR) | `OCROptimizer` (model detection, default prompt injection, stop sequences, sampling defaults) |
| **ProductionMiddleware.swift** | Rate limiting, request size limits, timeouts, security headers | `RateLimitMiddleware` (token bucket per API key or IP), `RequestSizeLimitMiddleware`, `TimeoutMiddleware`, `SecurityHeadersMiddleware` |
| **ResponseStore.swift** | In-memory store for OpenAI Responses API with LRU eviction (1024 entries) | `ResponseStore.shared` (put/get/delete) |

#### WebUI Subdirectory

Server-rendered SPA dashboard assembled from HTML fragments.

| File | Purpose |
|------|---------|
| **WebUIBuilder.swift** | Assembles the full SPA HTML page by combining all fragments + CSS + JS |
| **SharedHTML.swift** | Shared CSS (dark theme), navigation bar, and JavaScript utilities (API key management, authenticated fetch, formatting) |
| **StatusHTML.swift** | Status page with real-time TPS chart (Chart.js), metrics grid, device info |
| **ModelsHTML.swift** | Model management page with load/unload/delete, HuggingFace search, download progress |
| **ChatPageHTML.swift** | Chat page placeholder (currently iframes the legacy `/chat` page) |
| **AgentsHTML.swift** | Agent detection and config template generation (OpenClaw, Hermes, OpenCode) |
| **SettingsHTML.swift** | Server config, session management, device info |

#### API Routes

**Main API (default port 6590):**
- `POST /v1/chat/completions` -- OpenAI chat completions (streaming + non-streaming)
- `POST /v1/messages` -- Anthropic messages API (streaming + non-streaming)
- `POST /v1/completions` -- OpenAI text completions
- `POST /v1/embeddings` -- Embedding generation
- `POST /v1/responses` -- OpenAI Responses API
- `POST /v1/rerank` -- Document reranking
- `POST /v1/mcp/execute` -- MCP tool execution
- `POST /v1/batch/completions` -- Batch inference
- `GET /v1/models` -- List loaded models
- `GET /v1/stats` -- Inference metrics
- `GET /health` -- Health check
- `GET /chat` -- Legacy chat page
- `GET /` -- SPA dashboard

**Admin API (default port 6591):**
- `/admin/models/*` -- Model management (list, download, load, unload, delete, discover, settings)
- `/admin/sessions/*` -- Session management (list, delete, clear, save, fork)
- `/admin/cache/{modelId}/*` -- Prefix cache stats and clearing
- `/admin/api/device-info` -- Hardware info
- `/admin/api/bench/*` -- Benchmark control
- `/admin/api/ppl/*` -- Perplexity evaluation
- `/admin/adapters/*` -- LoRA adapter management
- `/admin/api/hf/*` -- HuggingFace search/download
- `/admin/api/turboquant/*` -- KV quantization config
- `/admin/api/model-family/*` -- Family optimization overrides
- `/admin/api/config` -- Server config read/write
- `/admin/api/grammar/validate` -- Grammar validation

---

### NovaMLXMenuBar -- macOS GUI

Native macOS menu bar application with a popover and full window, built with SwiftUI.

#### Files

| File | Purpose | Key Types |
|------|---------|-----------|
| **MenuBarController.swift** | Entry point creating the MenuBarExtra scene | `MenuBarContentView` (TabView with Status/Models tabs + Open Window/Quit buttons), notification names (.openNovaAppWindow, .restartNovaMLXServer, .novaMLXModelsChanged) |
| **MenuBarAppState.swift** | Central shared observable state with 2-second polling | `MenuBarAppState` (ObservableObject; published: isServerRunning, loadedModels, systemStats, tpsHistory, downloadTasks, cloudAuth; 2s timer refreshes all data) |
| **NovaAppView.swift** | Main full-window layout with sidebar + detail split | `NovaAppView` (sidebar: Status/Models/Downloads/Chat/Agents/Settings nav items; detail: corresponding page view), `AppPage` enum |
| **NovaTheme.swift** | Design token namespace (colors, spacing, radii) | `NovaTheme.Colors` (13 tokens with dark/light adaptation), `NovaTheme.Spacing`, `NovaTheme.Radius` |
| **NovaComponents.swift** | Reusable SwiftUI components | `sectionCard()`, `rowCard()` view modifiers, `StatusBadge`, `MetricCard`, `CopyIDButton`, `FlowLayout` |
| **StatusPageView.swift** | Status/dashboard with TPS chart and metrics grid | `StatusPageView` (server status hero, Chart.js-style SwiftUI Chart, 8 MetricCards, device info) |
| **StatusMenuView.swift** | Compact Status tab for menu bar popover | `StatusMenuView` (condensed status: server address, model count, memory, uptime, TPS) |
| **ModelsPageView.swift** | Model management with loaded/cloud/downloaded sections | `ModelsPageView` (load/unload/delete actions, model card detail sheet, cloud auth gate) |
| **ModelsMenuView.swift** | Compact Models tab for menu bar popover | `ModelsMenuView` (loaded model list with status dots, disk usage summary) |
| **ChatPageView.swift** | Full chat playground with three display modes | `ChatPageView` (Pretty/Raw JSON/Raw SSE modes; model picker; parameter sliders; streaming via InferenceService; input history) |
| **SettingsPageView.swift** | Settings: server config, cloud account, CLI, language, TurboQuant, sessions | `SettingsPageView` (inline config editor, AuthClient login, CLI symlink, language picker, per-model KV quantization) |
| **DownloadsPageView.swift** | HuggingFace model browser and downloader | `DownloadsPageView` (search with suggested queries, manual URL/repo download, progress tracking, model card sheets) |
| **AgentsPageView.swift** | Agent tool detection and config generation | `AgentsPageView` (detects OpenClaw/Hermes/OpenCode, shows installed status, generates config templates) |
| **DashboardView.swift** | Legacy single-page dashboard (not used in current navigation) | `DashboardView` (stats grid, device info, loaded/available models) |

---

### Executables

#### NovaMLXApp (main.swift)

The macOS application entry point. Initializes the entire system:

1. **Path validation** -- validates configured paths, migrates from legacy `Application Support` directory
2. **Configuration loading** -- loads `config.json` via `NovaMLXConfiguration`
3. **Model discovery** -- scans models directory, registers found models
4. **Engine setup** -- creates `MLXEngine`, `ModelSettingsManager`, `InferenceService`
5. **Worker subprocess** -- starts `NovaMLXWorker` for crash isolation
6. **API server** -- starts `NovaMLXAPIServer` on configured ports
7. **Model restoration** -- re-loads models from persisted `loaded_models.json`
8. **Memory enforcement** -- starts `ProcessMemoryEnforcer` and `MemoryPressureHandler`
9. **Cloud discovery** -- fetches available cloud models
10. **GUI** -- creates menu bar popover (`MenuBarController`) and optional full window (`NovaAppView`)

#### NovaMLXWorker (WorkerMain.swift)

Inference worker subprocess. Reads JSON commands from stdin, dispatches to MLX engine, writes JSON responses to stdout. Provides crash isolation: if inference crashes, only the worker dies, not the main app.

- Routes VLM/hybrid/grammar requests to `ContinuousBatcher`, others to `FusedBatchScheduler`
- Reports memory/CPU stats to parent every 5 seconds
- Auto-cleans MLX cache between requests

#### NovaMLXCLI (main.swift + CLIClient.swift + LaunchCommand.swift)

Command-line tool (`nova`) for terminal-based NovaMLX management.

**Subcommands:** models, download, load, unload, delete, search, status, chat (interactive REPL), config, sessions, cache, adapters, turboquant, bench, launch (agent launcher), login, logout, account.

`CLIClient` is a lightweight HTTP client that reads ports/API keys from the config file and calls the NovaMLX REST API.

`LaunchCommand` writes agent-specific config files (OpenClaw JSON, Hermes YAML, OpenCode env vars) and launches the agent binary.

#### NovaMLXBenchmarkRunner (main.swift + BenchmarkHarness.swift + BenchmarkInfrastructure.swift + FusedSDPABench.swift)

Standalone benchmark runner for performance testing. Measures TTFT, generation TPS, processing TPS, peak memory, and end-to-end latency across configurable prompt lengths. Includes Fused SDPA micro-benchmarks.

---

## Request Lifecycle

The complete path of an inference request from arrival to response:

### 1. HTTP Request Arrival

```
Client (curl/SDK/agent)
  │
  POST /v1/chat/completions
  │
  ▼
Hummingbird HTTP Server (NovaMLXAPIServer)
  ├── CORSMiddleware (CORS headers + OPTIONS preflight)
  ├── RequestIDMiddleware (x-request-id)
  ├── APIKeyAuthMiddleware (bearer token validation)
  ├── RateLimitMiddleware (token bucket)
  ├── RequestSizeLimitMiddleware (max body size)
  ├── NovaMLXErrorMiddleware (error → JSON)
  │
  ▼
handleChat / handleStreamChat (APIServer.swift)
  ├── Parse OpenAIRequest / AnthropicRequest
  ├── ClientDetector.detect() → scale context for agent tools
  ├── OCROptimizer.optimize() → adjust params for OCR models
  ├── Convert to InferenceRequest (ChatMessage array, sampling params, tools, images)
  │
  ▼
InferenceService.generate() / .stream()
```

### 2. Request Routing

```
InferenceService
  │
  ├── Resolve model alias (settingsManager.resolveModelId)
  ├── Apply per-model sampling overrides (ModelSettings.applySamplingOverrides)
  │
  ├── ":cloud" suffix? → CloudBackend.proxy() → remote API
  │
  ├── workerMode? → WorkerSupervisor.sendGenerate/sendStream()
  │                    │
  │                    ▼ (stdin/stdout JSON pipe)
  │                 WorkerMain.swift
  │                    │
  │                    ▼
  │                 FusedBatchScheduler / ContinuousBatcher
  │
  ├── VLM / hybrid / session / grammar? → ContinuousBatcher
  │
  └── Standard LLM → FusedBatchScheduler
```

### 3. Admission Control

```
FusedBatchScheduler.submit()
  │
  ├── MemoryBudgetTracker.canAdmit()
  │     └── Check: estimated KV bytes < remaining budget
  │         (factoring TurboQuant compression ratio)
  │
  ├── per-model concurrency check (optimalConcurrency, auto-tuned)
  │
  ├── If admitted → create ActiveStreamSequence
  └── If not → queue with priority, may preempt newer sequences
```

### 4. Prefill

```
prefillSequence()
  │
  ├── Resolve ModelContainer from EnginePool
  ├── Build UserInput from messages
  │     (template chosen by ensureChatTemplate at load time —
  │      tokenizer_config.json wins over .jinja by default; see Chat Template Processing)
  ├── Tokenize via mlxContainer.prepare(input:)
  ├── Check context window limit
  │
  ├── Prefix cache lookup:
  │     PrefixCacheManager.fetchPrefix(tokenIds:)
  │     ├── PagedBlockPool.findSharedPrefix() → matched block IDs
  │     ├── SSDCacheStore.loadBlock() → load cached KV arrays
  │     ├── CacheBlockExtractor.reconstructKVCache() → merge blocks
  │     └── Return: (cachedKV, cachedTokenCount, remainingTokenIds)
  │
  └── Full or partial prefill (inside MLXSerializer.shared.perform):
        ├── Create KV caches via model.newCache()
        ├── If prefix cache hit: prefill only remaining tokens
        ├── Chunked forward pass (prefillStepSize, typically 512)
        ├── Validate logits (NaN/Inf check)
        └── CompiledSampler.sample() → first token
```

### 5. Fused Batch Decode Loop

```
FusedBatchScheduler.runLoop() (repeats)
  │
  ├── admitQueued() — admit pending requests sorted by priority
  │
  ├── For each model with active sequences:
  │     │
  │     ├── Speculative decoding:
  │     │     NGramSpeculator.speculate(context) → 0-5 draft tokens
  │     │
  │     ├── Fused decode step (inside MLXSerializer.shared.perform):
  │     │     ├── FusedBatchDecoder: combine per-sequence KV caches
  │     │     ├── Single forward pass across all sequences
  │     │     ├── If draft: verify against main model greedy argmax
  │     │     ├── Frequency penalty (scatter_add on token histogram)
  │     │     └── CompiledSampler.sample() → next token per sequence
  │     │
  │     ├── Decode token via StreamingDetokenizer
  │     ├── Scrub control tokens (ChatTemplateProcessor)
  │     ├── Parse thinking blocks (ThinkingParser)
  │     ├── Parse tool calls (ToolCallParser)
  │     │
  │     ├── Check termination:
  │     │     ├── EOS token? → done
  │     │     ├── Max tokens reached? → done
  │     │     ├── Custom stop sequence? → done
  │     │     └── Hallucination detected? → done
  │     │
  │     └── On completion:
  │           ├── Release MemoryBudgetTracker allocation
  │           ├── Store prefix cache blocks
  │           └── Record metrics
```

### 6. Response Delivery

```
For streaming:
  Token → scrub → thinking parse → tool call parse
    → SSE event (OpenAI or Anthropic format)
    → HTTP chunked transfer

For non-streaming:
  All tokens collected → final scrub → assemble InferenceResult
    → JSON response
```

---

## Memory Management Architecture

NovaMLX employs a multi-layer memory management strategy to prevent OOM on Apple Silicon's shared memory architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Process Memory                            │
│                                                             │
│  ┌─────────────┐  ┌──────────────────────┐                 │
│  │ Model        │  │ KV Cache (per-sequence)│                │
│  │ Weights      │  │ ┌────┐ ┌────┐ ┌────┐ │                │
│  │ (MLX arrays) │  │ │Seq1│ │Seq2│ │Seq3│ │                │
│  │              │  │ └────┘ └────┘ └────┘ │                │
│  │              │  │                        │                │
│  │              │  │ Budget tracked by       │                │
│  │              │  │ MemoryBudgetTracker     │                │
│  └─────────────┘  └──────────────────────┘                 │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ProcessMemoryEnforcer (polls every 1s)              │   │
│  │ - Soft limit: evict LRU unpinned models              │   │
│  │ - Hard limit: abort requests + clear cache           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ MemoryPressureHandler (OS signals + 60s timer)       │   │
│  │ - Critical: evict all unpinned models                 │   │
│  │ - Warning: evict all but most recent unpinned         │   │
│  │ - GPU > 80%: evict unpinned models                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**TurboQuant** reduces KV cache memory via quantization (2/4/8-bit), configured per model. Auto-configuration considers model size, context length, and available memory.

**EnginePool** manages model lifecycle with LRU eviction and pinning support. Models can be pinned to prevent eviction.

---

## Scheduling Architecture

Three schedulers serve different use cases:

```
                    ┌────────────────────┐
                    │ InferenceService   │ (router)
                    └────────┬───────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
    ┌─────────┴──────┐ ┌────┴─────────┐ ┌──────┴────────┐
    │ FusedBatch     │ │ Continuous   │ │ Batch         │
    │ Scheduler      │ │ Batcher      │ │ Scheduler     │
    │ (production)   │ │ (specialized)│ │ (simple)      │
    └─────────┬──────┘ └────┬─────────┘ └──────┬────────┘
              │              │                  │
    Fused batch decode    Direct engine     Simple fused
    + N-gram speculation  generate/stream  decode only
    + auto-concurrency    (per-sequence)   (no speculation)
    + prefix cache
    + Harmony filtering

    Used for:            Used for:        Used for:
    Standard LLMs        VLM models       Legacy/testing
                         Hybrid attention
                         Session requests
                         Grammar-constrained
```

---

## Chat Template Processing

Different model families use different chat template formats and control tokens. NovaMLX has **four cooperating subsystems** for this:

1. **Data-driven configuration** — `ChatTemplateRegistry` (in NovaMLXUtils) loads bundled `template-registry.json` and merges `~/.nova/templates/registry.json` (user). All family-specific behavior — hallucination patterns, leakage blacklist, expected EOS tokens, thinking markers, template overrides, and family-detection extensions — comes from JSON. Adding a new family or fixing a problematic quant is a JSON edit, not a recompile.
2. **File-level reconciliation** — `MLXEngine.ensureChatTemplate` selects which on-disk template the tokenizer actually uses for prompt rendering, quarantining stale `.jinja` files that disagree with `tokenizer_config.json`.
3. **Load-time health check** — `runChatTemplateSanityCheck` renders a probe message through the tokenizer at load time and validates the output against six failure modes (unrendered Jinja, dropped user content, unbalanced control tokens, multi-format markers, etc.).
4. **Family-level interpretation** — `ChatTemplateProcessorRegistry` picks a per-family processor (Qwen ChatML, Bailing turn-role, Harmony, etc.) that scrubs control tokens, detects hallucinations, and stops turns. Each processor pulls hallucination patterns from `ChatTemplateRegistry`, so additions don't require touching processor code.

### File-level reconciliation (rendering source of truth)

A model directory can contain up to three places that store a chat template:

| Source | Convention | Priority in `swift-transformers/Hub.swift:280` |
|--------|-----------|------------------------------------------------|
| `tokenizer_config.json` → `chat_template` field | HuggingFace canonical, post-2023 | (lower) |
| `chat_template.jinja` standalone file | HuggingFace standard for large templates | **(higher) — silently overrides config** |
| `chat_template.json` (array form) | Used by older repos with multiple named templates | lowest |

**This priority is a footgun.** swift-transformers prefers the `.jinja` file and silently overwrites the value from `tokenizer_config.json`. A stale or wrong `.jinja` corrupts the prompt format. The model then **faithfully continues** that wrong format, emitting tokens like `<|turn>`, bare `user\n` / `model\n` separators, or other family-foreign markers. These look like control-token leakage or hallucination but are **the model behaving exactly as told**.

`ensureChatTemplate` runs every time a model is loaded and:

```
                    tokenizer_config.json.chat_template     chat_template.jinja
                              │                                       │
                              ▼                                       ▼
                        readConfigChatTemplate                    file read
                              │                                       │
                              └─────────────┬─────────────────────────┘
                                            ▼
                          ChatTemplateLibrary.resolve(...)?
                                  ┌─────────┴─────────┐
                                  ▼                   ▼
                            yes (override)         no (use disk)
                                  │                   │
                  injectTemplate to config            │
                  quarantineJinjaIfPresent            │
                                                      ▼
                                  ┌──────────────────────────────────────┐
                                  │  Reconcile by case:                  │
                                  │  config + jinja agree → keep config  │
                                  │  config + jinja DISAGREE → WARN +    │
                                  │       quarantine .jinja              │
                                  │  no config + jinja → promote jinja   │
                                  │       into config + quarantine       │
                                  │  config only → no-op                 │
                                  │  neither → ERROR (do NOT guess)      │
                                  └──────────────────────────────────────┘
```

Quarantine renames `chat_template.jinja` → `chat_template.jinja.disabled-by-novamlx` so the user can inspect/restore. It is **never deleted** without an existing backup.

> **Lesson — never inject a guessed default chat template.** Earlier versions of NovaMLX shipped a `defaultChatTemplate(for: .gemma)` fallback (a Bailing/turn-role template mislabeled as Gemma) that was written into `chat_template.jinja` whenever a model lacked one. Combined with `findChatTemplateInSiblings` (which copied the first sibling's `.jinja` into a new model dir), this contaminated unrelated model directories. The contamination was invisible at the API level — the model would still respond to chat completions, but with garbled output that looked like a model bug. The fix is to refuse to guess: if no maintained template is in `ChatTemplateLibrary` and no template is on disk, log an explicit error and fail chat requests with a clear message.

### Load-time health check (`runChatTemplateSanityCheck`)

After `ensureChatTemplate` runs and the tokenizer is loaded, the engine renders a sentinel probe message through `applyChatTemplate(...)` and validates the result. Six checks:

| # | Check | Failure indicates |
|---|-------|-------------------|
| 1 | `applyChatTemplate` doesn't throw | Missing or malformed Jinja template |
| 2 | Rendered output is non-empty | Template silently filters all input |
| 3 | No `{{`, `{%-`, `%}` literals | Template wasn't compiled (Jinja syntax error or runtime failure) |
| 4 | Sentinel `ping_test_42` appears in output | Template drops user content |
| 5 | `<\|im_start\|>` ↔ `<\|im_end\|>` (and equivalents) balanced | Template opens turns it never closes |
| 6 | Only one family format detected | Multi-family markers = corruption (e.g. ChatML template containing both `<\|im_start\|>` and `<\|turn\|>`) |

All findings emit a single `[ChatTemplateHealth]` log line per model load. The model still loads (we don't fail-fast) so non-chat models like embeddings aren't blocked, but operators have a clear signal that chat requests will misbehave.

### Family-level interpretation (output processor selection)

Once the tokenizer is producing prompts in the correct format, `ChatTemplateProcessorRegistry` routes to a per-family processor based on `(ModelFamily, ChatTemplateFormat)`:

```
Model loaded
  │
  ├── Extract control tokens from tokenizer added_tokens
  ├── Detect ChatTemplateFormat from template content (.turnRole, .imStartEnd, .startOfTurn, .deepSeek, .harmony, .unknown)
  │
  └── Lookup processor:
        Qwen + .imStartEnd     → QwenChatMLProcessor (stable, simple)
        Qwen + .turnRole       → QwenTurnRoleProcessor (hallucination detection)
        Gemma + *              → GemmaProcessor (extra scrubbing, channel thought)
        Bailing + *            → BailingProcessor (turn refinement)
        GPT-OSS + .harmony     → HarmonyProcessor (multi-token scrubbing)
        * + *                  → DefaultProcessor (format-based fallback)
```

Each processor provides: control-token lists, thinking-model detection, hallucination patterns, scrubbing rules, and turn-stop behavior. **Hallucination patterns are merged from the registry** at runtime, so adding a new locale-specific bare-text pattern (e.g. `\n\n用户\n`) is a JSON edit in `template-registry.json` — no Swift recompile.

### Format detection — multi-marker scan

`ChatTemplateFormat.detectAll(...)` scans the template for the markers of every known format and returns all hits, sorted by confidence (= number of distinct evidence markers seen). Used by:
- `runChatTemplateSanityCheck` (multi-format = corruption signal)
- `ChatTemplateDiagnostics` (operator-facing report)
- The legacy single-shot `detect(...)` API (returns `.first?.format`)

A historical bug (`<|im_start>` vs `<|im_start|>` typo) made the legacy detector silently miss canonical Qwen ChatML; the multi-marker scan replaces brittle first-substring-wins with explicit evidence enumeration.

### Diagnostic CLI

```
nova chat-template diagnose <model-id>
```

Pretty-prints the `ChatTemplateDiagnostics.Report`: family + architecture, on-disk template file inventory (sizes, agreement, quarantine status), multi-marker format detection with evidence, registry override status, family interpretation (EOS, hallucination, leakage, thinking markers), and a numbered list of health issues. First thing to run on any "model is hallucinating" report.

> **Diagnostic order when output looks wrong.**
> 1. **First** — `nova chat-template diagnose <id>`. The report tells you which template the tokenizer will use, whether files agree, and what format markers are present. If `chat_template.jinja` and `tokenizer_config.json.chat_template` disagree, fix the file layer before touching processor code.
> 2. **Then** — verify the family + format detection picked the right processor.
> 3. **Last** — investigate the processor's `hallucinationPatterns()` / `shouldStopForHallucination()` / `scrubControlTokens()` logic.
>
> Roughly 80% of "model is hallucinating" reports are actually file-layer template bugs.

---

## Configuration Files Reference

The following data files drive runtime behavior without recompilation:

| File | Purpose | Hot-reload? |
|------|---------|-------------|
| `Sources/NovaMLXUtils/Resources/template-registry.json` | Bundled defaults: family configs, template overrides, family-detection extensions | No (rebuild required) |
| `~/.nova/templates/registry.json` | User overlay for `template-registry.json` (per-key precedence) | Yes (call `ChatTemplateRegistry.shared.reload()`) |
| `~/.nova/templates/<sanitized-modelId>.jinja` | Raw `.jinja` override for a single model | Yes (next model load) |
| `~/.nova/profiles/_drafts/<sanitized-modelId>.json` | Auto-drafted test profile, awaiting operator review | Yes (the test skill picks it up by file path) |
| `Sources/NovaMLXEngine/ChatTemplates/*.jinja` | Bundled .jinja templates referenced by `template-registry.json` overrides | No (rebuild required) |

When upstream model authors fix a bad template, the user can drop the corrected version into `~/.nova/templates/<id>.jinja` and reload — no need to wait for a NovaMLX release.

---

## Structured Output Pipeline

When a request specifies `response_format`, `json_schema`, `regex`, or `gbnf_grammar`:

```
Request with grammar constraint
  │
  ▼
MLXEngine generates with ComposedLogitProcessor
  │
  ├── 1. Grammar processor builds token mask:
  │     JSONLogitProcessor      → 12-state machine, precomputed masks
  │     GBNFLogitProcessor      → recursive grammar matching
  │     RegexLogitProcessor     → per-character regex testing
  │     SchemaGuidedProcessor   → JSON Schema-driven state machine
  │     (all use TokenMaskBuilder for mask construction)
  │
  ├── 2. Repetition penalty processor adjusts logits
  │
  ├── 3. TurnStopProcessor checks for turn boundaries
  │
  └── 4. CompiledSampler samples from constrained logits
```

---

## Cloud Backend

For users with cloud subscriptions, NovaMLX transparently proxies requests to a remote API:

```
InferenceService
  │
  ├── Model ID has ":cloud" suffix? → CloudBackend (actor)
  │     │
  │     ├── CloudAuth.validate() → check subscription (5-min cached)
  │     │
  │     ├── OpenAI format:
  │     │     proxy() / proxyStream() → https://chat.baystoneai.com/v1
  │     │     SSE parsing handles content + reasoning fields
  │     │
  │     └── Anthropic format:
  │           proxyAnthropic() / proxyAnthropicStream()
  │           SSE handles content_block_delta + message_delta events
  │
  └── Model discovery:
        fetchModels() → GET /v1/models (10-min cache)
        Returns: [CloudModelInfo] with local names like "Qwen3-235B-4bit:cloud"
```

Cloud models appear alongside local models in `/v1/models` and are selected by appending `:cloud` to the model name.

---

## Configuration and Data Storage

```
~/.nova/                              (or NOVA_DIR, or ~/.config/novamlx/path)
  ├── config.json                     (server config: ports, API keys, limits, language)
  ├── metrics.json                    (persistent metrics: request counts, TPS, cache stats)
  ├── novamlx.log                     (rotating log file)
  ├── loaded_models.json              (persisted loaded model list for restart recovery)
  ├── auth_cache.json                 (cloud auth cache with 5-min TTL)
  ├── session                         (auth session token)
  ├── models/                         (downloaded model directories)
  ├── sessions/                       (chat session KV-cache files)
  ├── chat_history/                   (persistent chat conversations)
  └── prefix_cache/                   (per-model prefix KV-cache blocks as safetensors)
      └── <model-id>/
          ├── ab/
          │   ├── <hash-1>.safetensors
          │   └── <hash-2>.safetensors
          └── ...
```

Inside each downloaded model directory:

```
<models-dir>/<repo>/<name>/
  ├── config.json                     (architectures[], model_type — drives family detection)
  ├── tokenizer_config.json           (CANONICAL chat_template per HF spec — preferred)
  ├── tokenizer.json                  (BPE/SPM tables, added_tokens with `special` flags)
  ├── chat_template.jinja             (optional — wins over tokenizer_config in
  │                                    swift-transformers; reconciled by ensureChatTemplate)
  ├── chat_template.jinja.disabled-by-novamlx   (created by quarantine — see playbook)
  ├── chat_template.json              (legacy multi-template form, lowest priority)
  ├── generation_config.json          (sampling defaults: temperature, top_p, etc.)
  ├── preprocessor_config.json        (VLM only — image preprocessing settings)
  ├── processor_config.json           (VLM only — vision/audio token IDs)
  ├── video_preprocessor_config.json  (VLM with video — frame sampling)
  ├── model.safetensors.index.json    (sharding map)
  └── model-NNNNN-of-MMMMM.safetensors (weight shards)
```

---

## Diagnostic Playbook

When inference output looks wrong (control-token leakage, infinite generation, hallucinated turns, model rambling, gibberish reasoning), follow this order. Each layer's bugs masquerade as the next layer's bugs, so jumping ahead wastes time.

### Layer 1 — Verify the rendered prompt format

Symptoms: model emits family-foreign tokens (e.g. `<|turn|>` from a ChatML-only Qwen, `<start_of_turn>` from a non-Gemma, `<|channel|>` from a non-Harmony model); model echoes the user prompt back; bare `user\n` / `assistant\n` / `model\n` separators appear after the answer.

```
1. nova chat-template diagnose <model-id>
   The report shows file presence + sizes, agreement, multi-marker format
   detection, registry overrides, and a numbered list of issues.
2. If "templates agree: NO", the .jinja and tokenizer_config disagree.
   ensureChatTemplate will quarantine the .jinja on next model load.
3. If "multiple format markers detected", template is corrupted (mixed family
   markers). Either the upstream is wrong, or a previous bug contaminated it.
4. If "no recognized markers", the model uses a brand-new format. Add a profile
   draft in ~/.nova/templates/registry.json (or wait for the auto-drafter at
   ~/.nova/profiles/_drafts/<id>.json) and route it to a processor.
5. Force a reload via:
     POST /admin/models/unload   {"modelId": "..."}
     POST /admin/models/load     {"modelId": "..."}
6. Re-test. If the symptoms vanish, this was a file-layer bug — DONE.
```

Why this is layer 1: swift-transformers (`Hub.swift:280`) prefers `chat_template.jinja` over the canonical `tokenizer_config.json.chat_template` and silently overwrites it. A wrong `.jinja` makes the model receive a different chat format than its training, and the model **correctly continues that wrong format**. This looks like a model bug or NovaMLX scrubbing bug but is neither.

Optional belt-and-suspenders: enable `NOVAMLX_TEMPLATE_UPSTREAM_CHECK=1` to SHA-256 compare the local template against the upstream HF copy. Catches the case where `tokenizer_config.json` itself is corrupt (so local-only diff can't see the problem).

### Layer 2 — Verify family + format detection

Symptoms: prompt format is correct but stop tokens aren't recognized, output runs to `max_tokens`, hallucination patterns aren't caught.

```
1. Tail novamlx.log — find lines like:
     [ChatTemplateLibrary] <model>: level N (exact: <template>)
     [ControlTokens] <model>: [...]
     Detected hybrid linear attention model — will use ContinuousBatcher
2. Confirm:
   - Which ModelFamily was assigned (via config.json architectures[]).
   - Which ChatTemplateFormat was detected (.imStartEnd, .turnRole, …).
   - Which ChatTemplateProcessor was selected (in
     ChatTemplateProcessorRegistry).
   - Which scheduler took the request (FusedBatchScheduler vs ContinuousBatcher).
3. If family/format mapping is wrong, fix in ModelFamilyRegistry or
   ChatTemplateFormat.detect().
```

Why this is layer 2: even with a correct prompt, the wrong processor will fail to scrub model-specific control tokens or detect family-specific hallucination patterns.

### Layer 3 — Inspect processor and scheduler logic

Symptoms: family is correct, prompt is correct, but specific stop conditions still aren't triggered.

```
1. Read the processor's:
   - refineControlTokens()           — what counts as a stop token
   - hallucinationPatterns()         — bare-text patterns to catch
   - shouldStopForHallucination()    — when to enforce them
   - scrubControlTokens()            — what to strip from output
2. Read the scheduler's stop-check loop. Note: FusedBatchScheduler and
   ContinuousBatcher have separate stop-check paths. ContinuousBatcher
   relies on TurnStopProcessor + per-token EOS check; FusedBatchScheduler
   does its own text-level pattern check after each step.
3. If a model takes ContinuousBatcher (VLM, hybrid attention, sessions,
   grammar), all hallucination/stop logic must come through TurnStopProcessor
   or the family processor — there is no scheduler-level safety net there.
```

### Layer 4 — Verify request-time parameters

Symptoms: thinking content missing or duplicated, `enable_thinking` ignored, sampling defaults wrong.

```
1. Trace the request through the API → InferenceRequest pipeline:
   - OpenAI/Anthropic body parsed in APIServer.swift handlers
   - ModelSettings.applySamplingOverrides(to:) merges per-model defaults
   - InferenceService.{generate,stream}() forwards to scheduler
   - WorkerProtocol.CodableInferenceRequest — Codable mirror; every new
     field MUST be added in three places (struct + init + toInferenceRequest)
   - Final render via Tokenizer.applyChatTemplate(... additionalContext:)
     — this is where `enable_thinking`, `preserve_thinking`, `chat_template_kwargs`
     are passed as Jinja variables.
2. For Qwen3.6+ models specifically:
   - `enable_thinking` accepted at top-level (Dashscope/Alibaba style) AND
     inside `chat_template_kwargs` (vLLM/SGLang style). Both must thread through.
   - `preserve_thinking` only via `chat_template_kwargs`.
   - Greedy decoding (temperature=0) is officially DISCOURAGED — causes
     repetition loops.
```

### Common root-cause map

| Symptom | Most likely layer | First file to inspect |
|---------|-------------------|------------------------|
| `<\|turn\|>` in output of Qwen ChatML model | Layer 1 (file) | model dir's `chat_template.jinja` vs `tokenizer_config.json` |
| Bare `user\n / model\n / assistant\n` after answer | Layer 1 → 2 | template diff first; if same, processor's `hallucinationPatterns` |
| `enable_thinking=false` ignored | Layer 4 | `OpenAITypes.swift` → `InferenceRequest` → `CodableInferenceRequest` (worker) |
| Reasoning content empty when thinking should be on | Layer 4 → 1 | template's `enable_thinking` Jinja branch; then how the API forwards the flag |
| Model never stops, hits `max_tokens` | Layer 3 → 2 | `TurnStopProcessor`, then `eos_token_id`s in `tokenizer_config.json` |
| Streaming SSE missing terminal events | API layer | `APIServer.swift` SSE handlers |
| VLM gives text-only response | VLM path | `InferenceService` routing → `ContinuousBatcher` (FusedBatchScheduler does NOT support VLM) |

### Test-driven validation

The skill `novamlx-full-api-test` (under `.opencode/skills/`) ships a profile registry (`profiles/<key>.json`) that adapts each test to the loaded model's actual capabilities. After any fix to chat-template handling, processors, or schedulers, re-run the suite for every loaded family — leakage and hallucination patterns are family-specific, and a fix for one family can regress another.
