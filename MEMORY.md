# NovaMLX Project Memory

## Critical Rules (MUST follow every session)

### Rule 0: After Every Task — Self-Reflect and Record
After completing any task, ask yourself: "Is this a one-time fix, or a recurring pattern that could happen again?" If it could recur, add it to this Critical Rules section. This includes:
- Bugs caused by framework pitfalls (e.g., SwiftUI locale formatting)
- Non-obvious constraints (e.g., metallib must be colocated with binary)
- Project conventions (e.g., CLI tool named `nova`, not `novamlx`)
- User preferences expressed during the session

Do NOT wait for the user to ask. Do this proactively after every non-trivial change.

### Rule 0a: ALWAYS Kill NovaMLX Before Build, Add Timeout
Before EVERY `swift build`, run `pkill -9 -f NovaMLX` first. Residual NovaMLX processes block ports 8080/8081 and can hang the shell session after build completes. Also add a reasonable timeout to build commands.

**Pattern:**
```
pkill -9 -f NovaMLX; swift build -c release -Xswiftc -warnings-as-errors
```

This is NOT optional. Forgetting this wastes the user's time. Think ahead — if a command might leave background processes, clean them up BEFORE running the next command.

### Rule 0b: SwiftUI String Interpolation — Never Use Bare Int/Double in Text
SwiftUI `Text` interpolation applies locale formatting to numeric types (Int gets comma separators like `8,080`, Double gets decimal separators). This breaks URLs, port numbers, IDs, and any technical string.

**Always wrap with `String()`:**
- ✅ `Text("http://127.0.0.1:\(String(port))")`
- ✅ `"http://127.0.0.1:\(String(adminPort))/admin/api/..."`
- ❌ `Text("http://127.0.0.1:\(port)")` — produces `8,080` in en_US locale
- ❌ `Text("admin:\(port)")` — produces `admin:8,081`

This applies to ALL string contexts where numbers must appear as plain digits: URLs, file paths, identifiers, size values. When in doubt, wrap with `String()`.

### Rule 0c: MLX Metallib Must Be Colocated With Binary
MLX searches for `mlx.metallib` in this order: (1) same directory as binary, (2) `Resources/` subdirectory, (3) SwiftPM bundle, (4) framework bundle. When packaging for distribution, place `mlx.metallib` in `Contents/MacOS/` alongside the executable in the `.app` bundle.

### Rule 0d: macOS App Distribution Must Be a Proper .app Bundle
Users expect DMG → drag to Applications → done. Never distribute bare binaries. Use `Scripts/package.sh` to build a proper `.app` bundle with `Info.plist`, `Contents/MacOS/`, and `Contents/Resources/`. The DMG must include an `Applications` symlink for drag-to-install.

### Rule 0e: LSUIElement Hides App From Cmd+Tab and Dock
`LSUIElement=true` in Info.plist makes the app invisible in Dock and application switcher. Only use this for pure menu-bar agents with no main window. Since NovaMLX has a full GUI window, do NOT set LSUIElement.

### Rule 0f: CLI Tool Installs to ~/.local/bin/, Not /usr/local/bin
No sudo required. Create symlink at `~/.local/bin/nova` pointing to `Contents/MacOS/nova` inside the app bundle. GUI Settings page has an "Install CLI" button for one-click setup.

### Rule 0g: SwiftUI Window Must Be Opened From a Persistent View
`@Environment(\.openWindow)` only works from views that are already in the view hierarchy. Put `.onReceive` on the `MenuBarExtra` content (always alive), NOT on a `Window` that hasn't been opened yet.

### Rule 1: Decisions Must Survive Future Competitive Analysis
Every architecture change must be deeply thought through so that it will NOT be overturned in the next competitive analysis round. This is NOT about blindly locking decisions — it's about making robust, well-reasoned decisions NOW that stand up to scrutiny.

**Mandatory before any architecture update suggestion:**
1. Check this file's "Architecture Decisions Log" for prior decisions and WHY they were made
2. Run `git log --oneline -20` to review recent changes
3. For the proposed change, explicitly answer: "Will a future competitive analysis find this inferior? Why not?"
4. If you cannot confidently defend the change against future scrutiny, do NOT make it — refine it first
5. Prefer **additive improvements** over replacing existing architectures

### Rule 2: Every Change Must Be Wired In and Tested
Every code change MUST:
1. Be integrated into an actual running request path — no dead code
2. Have corresponding test code that verifies the change actually works
3. Be verified by running `swift test` after the change
4. If a change cannot be tested directly (e.g., requires GPU/model), add a unit test that validates the code path is reachable (e.g., tests for initialization, configuration, metrics output)

**Before finishing any feature:**
- [ ] Code is called from a real endpoint or internal path (not just instantiated)
- [ ] At least one test covers the new functionality
- [ ] `swift build -c release -Xswiftc -warnings-as-errors` passes
- [ ] `swift test` passes

---

## Architecture Decisions Log

Track major architecture decisions here so future competitive analysis rounds don't accidentally reverse them.

### 2026-04-12: BatchScheduler + Fused Batch Decode
- **Decision**: Implemented fused batch decode with `FusedBatchKVCache` wrapping per-layer `KVCacheSimple` instances
- **Rationale**: Batches multiple concurrent decode steps into single GPU ops for throughput gain
- **API**: `POST /v1/batch/completions` on inference port (8080), NOT admin port
- **Metrics**: Fused batch metrics merged into `GET /v1/stats` under `fusedBatch` key
- **Competitive edge**: omlx has no fused batch decode; this is a NovaMLX differentiator

### 2026-04-12: Endpoint Organization
- **Decision**: Inference endpoints on port 8080 (`/v1/*`), admin/management on port 8081 (`/admin/*`)
- **Rule**: Batch generation, completions, streaming = inference port. Model CRUD, adapter management, cache clearing, benchmarking = admin port.

### 2026-04-12: CompiledSampler + Speculative Decoding
- **Decision**: `CompiledSampler` uses `mx.compile()` to fuse softmax+topP+categorical into single Metal kernel. `NGramSpeculator` + `SpeculativeDecoder` for n-gram speculative decoding.
- **Status**: Infrastructure created, wired into `MLXEngine` as properties. Not yet in hot path for standard generate calls.

### 2026-04-12: WiredMemoryTicket Integration
- **Decision**: Per-generation `WiredMemoryTicket` via `WiredSumPolicy`, model weight reservation tickets, preflight memory check before generation.
- **Rationale**: Prevents OOM by capping wired memory to GPU recommended working set.

### 2026-04-12: LoRA/Adapter Support
- **Decision**: Full `AdapterService` with load/unload/fuse/list/discover lifecycle, 5 admin endpoints.

### 2026-04-12: InferenceResult is Codable
- **Decision**: Added `Codable` conformance to `InferenceResult` for batch response serialization.

### 2026-04-12: Preflight Memory Check — Fully Wired
- **Decision**: Auto-calculate `kvMemoryBytesPerToken` from model's `KVCacheDimensionProvider.kvHeads` at load time. Admin can override via `PUT /admin/models/{id}/settings` with `kv_memory_bytes_per_token_override`.
- **Formula**: `numLayers * 2 * sum(kvHeads) * headDim(128) * sizeof(Float16)`. Fallback 262144 (256KB) if no model info available.
- **Wired into**: All 6 generate paths (generate, stream, generateWithSession, streamWithSession, generateWithProcessor, streamWithProcessor)
- **Logging**: Rejection events log model, estimated vs available MB, bytesPerToken. OK events at info level.
- **Competitive survival**: This is a better approach than omlx (which uses fixed estimates) because we auto-calculate per-model. Admin override adds flexibility. Will NOT be overturned by future analysis.

### 2026-04-12: IndexCache (DeepSeek DSA) — DECISION: Skip
- **Decision**: Not implementing IndexCache for DeepSeek DSA models.
- **Reason**: mlx-swift-lm has NO DSA models (deepseek_v32 / glm_moe_dsa). Without a loadable DSA model, the code would be untestable dead code. Violates Rule 2.
- **Risk if implemented**: 200-300 lines of invasive model-layer code, monkey-patching not possible in Swift, precision loss (0-30% token mismatch), maintenance burden for 2 niche models.
- **When to revisit**: If/when mlx-swift-lm adds DSA model support. At that point, implement directly in Swift referencing omlx's `index_cache.py`.
- **Competitive survival**: This is a model-specific optimization for 2 niche architectures. omlx's advantage is theoretical until DSA models are widely used. Our preflight check, fused batch decode, LoRA support, and prefix cache are more impactful across all models.

### 2026-04-12: Framework-Level Optimizations (Gap 6/15/16/17/18) — DECISION: Use MLX Framework, No Custom Implementation

**Conclusion:** All 5 optimizations are fully handled by MLX framework's production Metal kernels. Custom implementations would provide zero performance benefit and risk introducing bugs.

| Gap | Optimization | MLX Framework Implementation | Auto? |
|-----|-------------|------------------------------|:-----:|
| 6 | Quantized fused MatMul | `qdot()` + `QuantizedBlockLoader` + NAX path, 2-8bit | ✅ |
| 15 | SIMD group matmul | `simdgroup_multiply_accumulate` 8×8 tile, ALL matmuls | ✅ |
| 16 | Function constant specialization | GEMM alignment / attention mask / RoPE direction | ✅ |
| 17 | mmap zero-copy | Apple Silicon unified memory `StorageModeShared` | ✅ |
| 18 | Metal RoPE kernel | `rope.metal` + `metal::fast::cos/sin`, all models via `MLXFast.RoPE` | ✅ |

**Source evidence:**
- Gap 6: `mlx-swift/Source/Cmlx/mlx/backend/metal/kernels/quantized.h` (qdot, QuantizedBlockLoader)
- Gap 15: `mlx-swift/Source/Cmlx/mlx/backend/metal/kernels/steel/gemm/mma.h` (BlockMMA → simdgroup_multiply_accumulate)
- Gap 16: `mlx-swift/Source/Cmlx/mlx/backend/metal/device.cpp` (MTLFunctionConstantValues pipeline specialization)
- Gap 17: `mlx-swift/Source/Cmlx/mlx/backend/metal/allocator.cpp` (StorageModeShared — hardware zero-copy on Apple Silicon)
- Gap 18: `mlx-swift/Source/Cmlx/mlx/backend/metal/kernels/rope.metal` (metal::fast::cos/sin)

**Competitive survival:** This is NOT "we don't do it" — it's "MLX already does it as well as llama.cpp". Future competitive analysis MUST NOT list these as gaps.

### 2026-04-12: Quantized Flash Attention (Gap 5) — DECISION: No Custom Implementation Needed

**NovaMLX's KV quantization path (affine only):**
- Enabled by `GenerateParameters.kvBits` → `QuantizedKVCache` → `quantizedScaledDotProductAttention()`
- Source: `mlx-swift-lm/Libraries/MLXLMCommon/KVCache.swift:1532`
- This function calls `quantizedMM()` twice (Q*K and attn*V), which uses `qdot()` Metal kernel — **KV is NEVER dequantized**
- Equivalent to omlx's `decode_attention()` — fused quantized attention without materializing full dequantized KV
- **Automatic when kvBits is set** — now wired through TurboQuantService via `settingsProvider`

**Why no custom implementation needed:**
1. MLX framework provides fused quantized attention out of the box
2. Codebook VQ path was deleted (dead code, see "Codebook Quantization Cleanup" above)
3. omlx's `decode_attention` comes from `mlx_vlm.turboquant` (external Python package), specifically for codebook mode which we don't use

**Competitive survival:** MLX's `quantizedScaledDotProductAttention()` + `quantizedMM()` is equivalent to omlx's fused decode_attention. Future analysis MUST NOT list this as a gap.

### 2026-04-12: Codebook Quantization Cleanup — DELETED

- **Decision**: Deleted `CodebookQuantizer`, `CodebookMetalShaders`, `CodebookTests` (23 tests). Removed `QuantizationScheme.codebook` case. Removed `kvScheme` from `ModelSettings`, `OpenAITypes`, APIServer update handler.
- **Rationale**: Codebook VQ path was dead code — `CodebookQuantizer` could not integrate with MLX's `QuantizedKVCache` attention pipeline. Would require 500-800 lines of custom Metal attention kernels to make it work. Affine path already provides fused quantized attention via MLX framework.
- **Competitive survival**: omlx's TurboQuant codebook mode uses `mlx_vlm.turboquant` (Python-only, not portable). Our affine path achieves the same fused quantized attention through MLX. No gap.

### 2026-04-12: TurboQuant Service Rewired Into Generate Path

- **Decision**: TurboQuantService is now wired into all generate paths through `MLXEngine.buildGenerateParameters()` via `settingsProvider` callback.
- **Architecture**: `InferenceService.init()` sets `engine.settingsProvider = { settingsManager.getSettings($0) }`. `buildGenerateParameters` calls `turboQuantService.applyToGenerateParameters()` with resolved settings.
- **Flow**: Settings → `settingsProvider` → `TurboQuantService.applyToGenerateParameters()` → sets `GenerateParameters.kvBits/kvGroupSize` → MLX's `QuantizedKVCache` → fused quantized attention.
- **Auto-configure**: `autoConfigure(modelId:modelSizeGB:contextLength:availableMemoryGB:)` selects optimal bit width based on memory pressure.
- **Admin endpoints**:
  - `PUT /admin/api/turboquant/{modelId}` — enable (with optional `bits`, `groupSize`, or auto-select), or disable
  - `DELETE /admin/api/turboquant/{modelId}` — remove config
  - `GET /admin/api/turboquant` — list all configs with stats
- **Competitive survival**: This is a cleaner architecture than omlx's monkey-patching approach. Settings provider pattern decouples engine from settings manager. Will NOT be overturned.

### 2026-04-12: Gap 12 — Boolean Mask Grammar Constraints

- **Decision**: Replaced Float32 mask arrays (`-inf`/`0.0`, 512KB per mask) with boolean MLXArray masks (16KB per mask) across all 4 logit processors.
- **New file**: `TokenMaskBuilder.swift` — pre-computes token-to-first-char mapping at init time, builds boolean masks via `[Bool]` arrays.
- **Files updated**: `JSONLogitProcessor`, `GBNFLogitProcessor`, `RegexLogitProcessor`, `SchemaGuidedProcessor` — all use `TokenMaskBuilder.buildMask()` + `TokenMaskBuilder.applyMask()`.
- **Memory savings**: 32x per mask (16KB vs 512KB for 128K vocab). JSONLogitProcessor pre-computes 12 masks → 192KB vs 6MB.
- **Performance**: Boolean `MLX.where` is faster than Float32 comparison on GPU. Token-to-char mapping precomputed once vs decoded every step.
- **Dead code removed**: `RegexLogitProcessor.isCompatibleWithAllowed`, `GBNFLogitProcessor.isCompatible`, `RegexLogitProcessor.precomputedMask` (was never populated).

### 2026-04-12: Gap 13 — Composed Grammar + Penalty Processors

- **Decision**: Created `ComposedLogitProcessor` that chains penalty processing with grammar constraints. Grammar-constrained paths now include repetition/frequency/presence penalties.
- **Problem**: In the grammar path (`generateWithProcessor`/`streamWithProcessor`), only the grammar processor ran on logits — penalties were silently bypassed. `GenerateParameters.processor()` was never called.
- **Solution**: `ComposedLogitProcessor` runs `penaltyProcessor.process()` then `grammarProcessor.process()` on each step. Penalty processor is created from `GenerateParameters.processor()` which uses GPU-side scatter-write/scatter-add.
- **Wired into**: Both `generateWithProcessor()` and `streamWithProcessor()` via `buildGenerateParameters` + `parameters.processor()`.
- **Order**: Penalties first (modify raw logits), then grammar mask (zero out disallowed tokens). This ensures penalties affect allowed token selection within grammar constraints.
- **Competitive survival**: omlx has no grammar-constrained generation with penalty support. This is a NovaMLX differentiator.

### 2026-04-12: Gap 20 — Model Family Optimization Registry

- **Decision**: `ModelFamilyRegistry` provides per-model-family inference defaults (kvBits, prefillStepSize, repeatLastN, recommendedContextLength).
- **Architecture**: Registry with 6 family defaults (llama, mistral, phi, qwen, gemma, starcoder) + architecture string mapping (e.g., `LlamaForCausalLM` → `.llama`) + per-model overrides.
- **Wired into**: `buildGenerateParameters()` uses `ModelFamilyRegistry.shared.optimization()` for `prefillStepSize` and fallback `kvBits` when no TurboQuant/settings config exists.
- **Admin endpoints**:
  - `PUT /admin/api/model-family/{modelId}` — set per-model override
  - `DELETE /admin/api/model-family/{modelId}` — remove override
  - `GET /admin/api/model-family` — list all overrides
- **Competitive survival**: llama.cpp uses per-model defaults too. This brings NovaMLX to parity with sensible family-specific tuning. Override mechanism adds admin flexibility.

### 2026-04-12: Performance Benchmark Test Suite

- **Decision**: Dedicated `NovaMLXBenchTests` test target with comprehensive multi-dimensional performance measurement. Not a modification of existing `BenchmarkService` — that remains the production admin benchmark. The test suite is for development-time regression testing and competitive analysis.
- **Architecture**: Standalone test target that directly uses `MLXEngine`, `ModelManager`, and `InferenceService`. Downloads models via `HubApi` if not already cached. Measures streaming token timing for accurate TTFT/decode TPS.
- **Model coverage**: 6 LLM families (llama, mistral, phi, qwen, gemma, starcoder) + VLM models. Uses `mlx-community/` 4-bit quantized models from HuggingFace (~4-5GB each for 7-8B).

**Performance dimensions measured:**

| Dimension | What | Why |
|-----------|------|-----|
| TTFT | Time from request start to first token yielded | Latency perception, prefill speed |
| Decode TPS | Tokens/sec after first token | Core throughput metric |
| Prefill TPS | Prompt tokens / time-to-first-token | How fast the model processes context |
| Model load time | Time to load + first inference | Cold-start latency |
| Peak GPU memory | `MLX.Memory.activeMemory` peak during generation | Memory efficiency |
| Memory per token | (peak - baseline) / completion tokens | KV cache efficiency |
| Streaming interval | Std dev of inter-token latency | Smoothness of streaming |
| Concurrent throughput | 2/4 parallel requests, total TPS | Scheduling efficiency |
| KV quantization impact | Same model with kvBits=2/4/8 vs none | TurboQuant effectiveness |
| Grammar overhead | JSON-constrained vs unconstrained TPS | Logit processor cost |
| Warmup effect | 1st vs 2nd vs 3rd generation TPS | Metal kernel compilation cache |

- **Competitive survival**: omlx has no benchmark suite. This is a NovaMLX differentiator for performance regression testing and competitive claims.

### 2026-04-13: Fused Quantized SDPA Metal Kernel — Working for 4-bit and 8-bit

- **Decision**: Custom fused quantized SDPA Metal kernel via `MLXFast.metalKernel()` that replaces MLX's unfused `quantizedScaledDotProductAttention()` decode path.
- **Architecture**: JIT-compiled Metal kernel that fuses quantized attention decode into a single kernel launch. Uses online softmax pattern with SIMD group reduction (same as MLX's `sdpa_vector`). Each SIMD group (32 threads) handles one (batch, head, q_idx).
- **Integration**: Build-time dependency patching via `Scripts/patch-fused-sdpa.py` which injects `_novamlxFusedSDPA` hook into `AttentionUtils.swift`. NovaMLXEngine registers its kernel at init time via `FusedSDPARegistration.register()`.
- **Data format** (critical — got this wrong twice):
  - Weights: **uint32** packed (NOT uint8!). Each uint32 holds `32/bits` values. For 8-bit: 4 vals/word, for 4-bit: 8 vals/word.
  - Scales: **float16** (same dtype as model weights, NOT float32 or fp8_e8m0)
  - Biases: **float16** (same as scales, NOT optional for affine mode)
  - Dequantization formula: `float((word >> shift) & mask) * float(scale) + float(bias)`
  - Weight stride: `D * bits / 32` uint32 elements per position
  - Scale stride: `D / groupSize` half elements per position
- **Benchmark results** (Phi-3.5-mini, head_dim=96, group_size=32):

| Config | Decode TPS | vs Baseline | vs Unfused |
|--------|-----------|-------------|------------|
| No KV quant | ~172.4 t/s | baseline | — |
| KV-8bit FUSED | ~161.7 t/s | -6.2% | **+2.6%** |
| KV-4bit FUSED | ~162.0 t/s | -6.0% | **+5.9%** |
| KV-8bit UNFUSED | ~157.6 t/s | -8.6% | — |
| KV-4bit UNFUSED | ~153.0 t/s | -11.3% | — |

- **Key results**: Fused 4-bit kernel is **5.9% faster** than unfused. Fused 8-bit is **2.6% faster**. Both produce correct output.
- **Files**: `FusedQuantizedSDPA.swift` (kernel), `FusedSDPARegistration.swift` (hook registration), `Scripts/patch-fused-sdpa.py` (dependency patching)
- **Supported configs**: head_dim 64/96/128/256, bits 4/8, any group_size dividing head_dim evenly. Decode only (L==1).
- **Competitive survival**: This is a custom Metal kernel that omlx does not have (they rely on Python-level TurboQuant). The fused kernel eliminates multi-launch overhead in the quantized attention decode path.

### 2026-04-16: Dual API — OpenAI + Anthropic (Native, No Proxy)

- **Decision**: NovaMLX natively supports both OpenAI and Anthropic API formats. Both are implemented directly in `Sources/NovaMLXAPI/APIServer.swift`.
- **OpenAI API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/rerank` — auth via `Authorization: Bearer xxx`
- **Anthropic API**: `/v1/messages`, `/v1/messages/count_tokens` — auth via `x-api-key: xxx` (Anthropic standard)
- **Auth middleware**: `APIKeyAuthMiddleware` accepts both header formats
- **Types**: `OpenAITypes.swift` + `AnthropicTypes.swift` — separate type definitions for each API format
- **Streaming**: Both APIs have SSE streaming support — OpenAI uses `data: {...}\n\n` + `data: [DONE]`, Anthropic uses typed events (`message_start`, `content_block_delta`, `message_stop`)
- **Why both**: Claude Code uses Anthropic Messages API. By supporting both natively, Claude Code can connect directly to NovaMLX via `ANTHROPIC_BASE_URL=http://127.0.0.1:6590`. No external proxy needed.
- **Competitive survival**: omlx only supports OpenAI format. Native dual-API is a NovaMLX differentiator.
