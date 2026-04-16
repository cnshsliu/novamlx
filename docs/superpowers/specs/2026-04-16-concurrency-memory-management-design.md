# NovaMLX Concurrency & Memory Management Architecture

> **Status**: Design Spec
> **Date**: 2026-04-16
> **Goal**: World-class concurrent request handling for local LLM inference on Apple Silicon

---

## 1. Problem Statement

NovaMLX currently crashes or errors when a single Claude Code client sends 2 concurrent requests to the same model. The root causes:

1. **Count-based admission**: `ContinuousBatcher` admits up to 8 requests by count, ignoring GPU memory. Two 11K-token requests on a 27B model consume ~32 GB KV cache and crash.
2. **No memory budget tracking**: No component tracks how much KV memory is committed across all active requests. `preflightCheck()` is per-request, not per-batch.
3. **TurboQuant savings invisible to admission**: 4-bit KV quantization saves 4x memory, but preflight uses raw FP16 estimates, rejecting requests that would fit.
4. **No request queuing based on memory**: When memory is insufficient, requests are rejected outright instead of queued.
5. **True continuous batching exists but is dead code**: `BatchScheduler` + `FusedBatchKVCache` implement fused decode but are not wired into production.

### What Claude Code Sends

Claude Code sends **2 requests per user message** (confirmed by GitHub issues #21969, #30193):
- Request 1: `stream: true, max_tokens: 32000` — main streaming inference
- Request 2: `stream: false, max_tokens: 21333, temperature: 1` — background/speculative request

This is by design, not a bug. Any local inference engine must handle this gracefully.

---

## 2. How the Best Systems Do It

### vLLM (Production Grade, Most Adopted)

- **PagedAttention**: KV cache managed in fixed-size blocks (16-32 tokens/block) with page tables. Blocks allocated on demand, not pre-allocated. Eliminates fragmentation.
- **Memory-aware scheduler**: Three queues: `waiting` (need prefill), `running` (in decode), `swapped` (evicted to CPU). Admission based on available KV block count, not request count.
- **Prefix caching**: Content-addressed KV blocks shared across requests with same system prompt. Copy-on-Write semantics.
- **Preemption**: When memory runs out, preempt requests. V1 defaults to **recompute** (discard KV, re-prefill later). Legacy option: swap KV to CPU RAM.
- **Continuous batching**: Iteration-level scheduling — at each decode step, completed requests exit and waiting requests enter.

### Ollama (Mac-Friendly, Most Popular on Apple Silicon)

- **Slot-based parallelism**: `OLLAMA_NUM_PARALLEL=1-4` slots per model. Requests fill available slots.
- **Prefix sharing**: `CopyPrefix` method references same physical KV cells across sequences. Zero-copy, O(cells) metadata operation.
- **FIFO queuing**: `OLLAMA_MAX_QUEUE=512`. When all slots full, requests wait in FIFO queue. When queue full, returns 503.
- **Memory-aware auto-config**: Auto-adjusts `numParallel` based on available memory and model size.
- **Continuous batching via llama.cpp**: Underlying llama.cpp engine handles iteration-level batching within slots.

### Key Insight

Both systems share the same fundamental architecture:

```
Request → Admission Controller (memory check) → Ready Queue → Scheduler → GPU Execution
                ↓ (memory insufficient)                ↑
            Wait Queue (FIFO)                   Completion frees memory
```

The scheduler is the central component. It knows the memory budget, tracks committed memory, and decides when to admit or queue requests.

---

## 3. Architecture Design

### 3.1 Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                         API Layer (APIServer)                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                      InferenceService                                │
│                    (unified routing)                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                    Scheduler (NEW)                                    │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐     │
│  │  AdmissionCtrl  │  │  ReadyQueue  │  │  MemoryBudgetTrack  │     │
│  │  - preflight    │  │  - FIFO      │  │  - total committed  │     │
│  │  - budget check │  │  - priority  │  │  - per-model budget │     │
│  │  - TurboQuant   │  │  - backpress │  │  - actual vs est    │     │
│  └────────┬────────┘  └──────┬───────┘  └──────────┬──────────┘     │
│           │                  │                      │                │
│  ┌────────▼──────────────────▼──────────────────────▼──────────┐    │
│  │              RunLoop (iteration-level)                       │    │
│  │  1. Check memory budget                                      │    │
│  │  2. Admit waiting requests if budget allows                  │    │
│  │  3. Prefill one new request (interleaved with decode)        │    │
│  │  4. Fused decode step for all active sequences               │    │
│  │  5. Check completions, free memory                           │    │
│  │  6. Repeat                                                   │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                    GPU Execution Layer                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │  FusedBatchDecode │  │  TurboQuantCache  │  │  PrefixCache     │   │
│  │  (batched KV)     │  │  (4/8-bit KV)     │  │  (shared prefix) │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Budget Tracker

**File**: `Sources/NovaMLXEngine/MemoryBudgetTracker.swift` (NEW)

The central memory accounting system. Every component queries this before making decisions.

```swift
public actor MemoryBudgetTracker {
    // Hardware limit
    private let gpuLimitBytes: UInt64  // GPU.maxRecommendedWorkingSetBytes()

    // Current state
    private var weightsMemoryBytes: UInt64 = 0          // all loaded model weights
    private var committedKVMemoryBytes: UInt64 = 0      // KV cache for all active sequences
    private var reservedSystemBytes: UInt64 = 0          // OS/system reserved

    // Per-model tracking
    private var modelBudgets: [String: ModelMemoryBudget] = [:]

    struct ModelMemoryBudget {
        let weightsBytes: UInt64
        let bytesPerToken: UInt64         // actual bytes/token (with TurboQuant factored in)
        var activeSequences: [UUID: SequenceBudget]
    }

    struct SequenceBudget {
        let estimatedTokens: Int          // promptTokens + maxTokens
        let estimatedBytes: UInt64
        var actualBytes: UInt64?          // filled in after generation starts
    }

    // Available budget for new sequences
    var availableKVBudget: UInt64 {
        gpuLimitBytes - weightsMemoryBytes - committedKVMemoryBytes - reservedSystemBytes
    }

    // Can we admit a new sequence?
    func canAdmit(modelId: String, estimatedTokens: Int, bytesPerToken: Int) -> Bool {
        let needed = UInt64(estimatedTokens) * UInt64(bytesPerToken)
        // Leave 10% headroom
        return needed <= availableKVBudget * 9 / 10
    }

    // Reserve memory for a sequence (called at admission)
    func reserve(modelId: String, sequenceId: UUID, tokens: Int, bytesPerToken: Int) { ... }

    // Release memory when a sequence completes or is preempted
    func release(sequenceId: UUID) { ... }

    // Update actual memory usage (feedback loop)
    func updateActual(modelId: String, sequenceId: UUID, actualBytes: UInt64) { ... }
}
```

**Key behaviors**:
- Tracks memory at the **sequence level**, not just the model level
- Factors in TurboQuant compression (4-bit KV = bytesPerToken / 4)
- Leaves 10% headroom to avoid OOM
- Provides a feedback loop: after generation, compare estimated vs actual memory

### 3.3 Scheduler

**File**: `Sources/NovaMLXEngine/Scheduler.swift` (NEW, replaces both `ContinuousBatcher` and `BatchScheduler`)

The unified, memory-aware request scheduler. Replaces `ContinuousBatcher` as the production admission path and incorporates `BatchScheduler`'s fused decode.

```swift
public final class Scheduler: @unchecked Sendable {
    private let engine: MLXEngine
    private let budgetTracker: MemoryBudgetTracker
    private let lock = NovaMLXLock()

    // Request queues (per model)
    private var waitingQueue: [ScheduledRequest] = []      // FIFO, priority-sorted within
    private var activeSequences: [UUID: ActiveSequence] = [:]

    // Configuration
    private let maxConcurrentPerModel: Int   // default: 4 (like OLLAMA_NUM_PARALLEL)
    private let maxQueueDepth: Int           // default: 512 (like OLLAMA_MAX_QUEUE)

    struct ScheduledRequest {
        let request: InferenceRequest
        let estimatedTokens: Int
        let estimatedBytes: UInt64
        let enqueuedAt: Date
        let priority: RequestPriority
        let continuation: RequestContinuation  // either stream or generate
    }

    enum RequestContinuation {
        case generate(CheckedContinuation<InferenceResult, Error>)
        case stream(AsyncThrowingStream<Token, Error>.Continuation)
    }

    // Main entry points (called from InferenceService)
    public func submit(_ request: InferenceRequest) async throws -> InferenceResult { ... }
    public func submitStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> { ... }
}
```

**Admission flow**:

```
submit(request)
  │
  ├─ 1. Estimate memory: tokens × bytesPerToken (with TurboQuant ratio)
  │
  ├─ 2. budgetTracker.canAdmit(model, tokens, bytesPerToken)?
  │     │
  │     ├─ YES → budgetTracker.reserve() → start execution
  │     │
  │     └─ NO  → enqueue in waitingQueue (sorted by priority, then FIFO)
  │              log: "Request queued: insufficient memory (need X MB, available Y MB)"
  │
  └─ 3. On completion:
        budgetTracker.release()
        → check waitingQueue for admittable requests
        → admit if budget allows
```

**Run loop (fused decode)**:

```
while activeSequences not empty or waitingQueue not empty:
    1. Check waitingQueue:
       - For each waiting request, check budgetTracker.canAdmit()
       - Admit up to N new requests (N = available budget / estimated bytes)
       - Prefill ONE new request (interleaved, not batched)

    2. Fused decode step:
       - Collect all active sequences' latest tokens
       - Stack into single batch tensor
       - One forward pass through model
       - Split logits, sample per-sequence
       - Yield tokens to stream continuations

    3. Check completions:
       - Finished sequences → budgetTracker.release()
       - More budget freed → go back to step 1
```

### 3.4 Admission Controller

**Integrated into `Scheduler`** (not a separate file).

The admission logic replaces the current `ContinuousBatcher`'s count-based check:

**Before (broken)**:
```swift
guard activeRequests < maxBatchSize else {
    return try await enqueueAndWait(...)
}
```

**After (correct)**:
```swift
let (canAdmit, reason) = await budgetTracker.canAdmit(
    modelId: request.model,
    estimatedTokens: estimateTokenCount(request),
    bytesPerToken: effectiveBytesPerToken(model: request.model)
)

if canAdmit && activeCountForModel(request.model) < maxConcurrentPerModel {
    await budgetTracker.reserve(...)
    startExecution(request)
} else {
    enqueue(request, reason: reason)
}
```

The check has TWO conditions:
1. **Memory budget**: enough GPU memory for the estimated KV cache
2. **Concurrency limit**: max N concurrent sequences per model (configurable, like `OLLAMA_NUM_PARALLEL`)

### 3.5 TurboQuant-Aware Memory Estimation

**Modified**: `MLXEngine.preflightCheck()` and new `MemoryBudgetTracker`

Currently, `preflightCheck()` uses raw FP16 `bytesPerToken`. With TurboQuant, the actual memory can be 4x less.

```swift
// Compute effective bytesPerToken with TurboQuant compression
func effectiveBytesPerToken(modelId: String) -> Int {
    let raw = container.config.kvMemoryBytesPerToken ?? 262_144
    let turboConfig = turboQuantService.config(for: modelId)
    if let bits = turboConfig?.bits {
        // 4-bit quantization: FP16 (16 bits) → 4 bits = 4x compression
        let compressionRatio = Double(16) / Double(bits)
        return Int(Double(raw) / compressionRatio)
    }
    return raw
}
```

This ensures:
- Preflight and admission use the **same** memory estimate
- When TurboQuant is configured, more requests can be admitted (4x more for 4-bit)
- The estimate is conservative (FP16) when TurboQuant is not configured

### 3.6 Prefix Cache Integration

**File**: `Sources/NovaMLXPrefixCache/PrefixCacheManager.swift` (MODIFIED)

The prefix cache already exists with block-level hashing and SSD persistence. It needs to be **read-active**, not just write-only:

**Current** (broken): `MLXEngine.generate()` calls `storeCache()` after generation but never calls `fetchPrefix()` to reuse cached KV.

**Fix**:
1. Before prefill, call `fetchPrefix()` to check if the system prompt KV is already cached
2. If hit: skip prefill for cached tokens, only prefill the remaining user tokens
3. Multiple requests sharing the same system prompt share the same cached blocks

```swift
// In the prefill phase:
let prefixResult = prefixCacheManager.fetchPrefix(
    modelId: modelId,
    tokenIds: allTokenIds
)

if let cachedKV = prefixResult.cache, prefixResult.cachedTokenCount > 0 {
    // Initialize KV cache from cached prefix — skip prefill for these tokens
    kvCache.restore(from: cachedKV, count: prefixResult.cachedTokenCount)
    // Only prefill remaining tokens
    remainingTokens = allTokenIds[prefixResult.cachedTokenCount...]
}
```

**Memory sharing**: When multiple requests share a prefix, they reference the same physical cache blocks (reference-counted). Only diverging tokens allocate new memory.

### 3.7 Backpressure and Client Communication

**New**: HTTP-level backpressure signals

When the scheduler queues a request, the client should know:

1. **503 Service Unavailable** when queue is full (like Ollama)
2. **202 Accepted** with `X-Queue-Position: N` header for queued requests (optional)
3. **Retry-After** header for rate limiting

```swift
// In InferenceService:
if scheduler.queueDepth(for: modelId) >= maxQueueDepth {
    throw NovaMLXError.serviceUnavailable(
        "Queue full for model \(modelId). Retry after: \(estimatedWaitSeconds)s"
    )
}
```

### 3.8 MemoryPressureHandler Integration

**Modified**: `Sources/NovaMLXInference/MemoryPressureHandler.swift`

The pressure handler becomes a **listener** on the `MemoryBudgetTracker` instead of polling `MLX.Memory.activeMemory`:

1. **Proactive**: When `budgetTracker.availableKVBudget` drops below 20%, signal the scheduler to stop admitting new sequences.
2. **Reactive**: When OS sends `critical` pressure, the scheduler preempts active sequences (newest first) to free memory.
3. **No more blind model eviction**: Instead of evicting entire models, first try evicting KV caches. Only evict model weights as a last resort.

---

## 4. Data Flow (After)

```
Claude Code sends 2 concurrent requests
  │
  ├─ Request 1 (stream=true):  "What is Svelte?"
  │   └→ InferenceService → Scheduler.submitStream()
  │       ├→ Estimate: 11K tokens × 64 KB/token (4-bit TurboQuant) = 704 MB
  │       ├→ budgetTracker.canAdmit? → YES (107 GB available)
  │       ├→ budgetTracker.reserve(704 MB)
  │       └→ Start execution → prefill → fused decode → stream tokens
  │
  └─ Request 2 (stream=false): "What is Svelte?" (background)
      └→ InferenceService → Scheduler.submit()
          ├→ Estimate: 11K tokens × 64 KB/token = 704 MB
          ├→ budgetTracker.canAdmit? → YES (106 GB remaining after Request 1)
          ├→ budgetTracker.reserve(704 MB)
          └→ Queue in waitingQueue (same model, 2nd slot)
              → Prefill after Request 1's current decode step
              → Both run in fused decode (2 sequences, 1 forward pass)
              → Complete → budgetTracker.release(704 MB each)
```

Both requests run. No crash. No "I/O on closed channel". Memory tracked accurately.

---

## 5. Implementation Phases

### Phase 1: Memory-Aware Admission (Highest Priority, Unblocks Claude Code)

**Goal**: Replace count-based admission with memory-based admission. Queue requests when memory insufficient.

**Files**:
- Create: `Sources/NovaMLXEngine/MemoryBudgetTracker.swift`
- Modify: `Sources/NovaMLXEngine/ContinuousBatcher.swift` — add memory check before admission
- Modify: `Sources/NovaMLXEngine/MLXEngine.swift` — expose TurboQuant-aware bytesPerToken
- Modify: `Sources/NovaMLXInference/InferenceService.swift` — route ALL requests through batcher

**Key changes**:
1. `MemoryBudgetTracker` tracks committed KV memory across all active sequences
2. `ContinuousBatcher` admission checks `budgetTracker.canAdmit()` instead of `activeCount < maxBatchSize`
3. Preflight check factors in TurboQuant compression ratio
4. All request paths (session, grammar, regular) go through the batcher

**Result**: Two concurrent Claude Code requests are handled correctly — admitted if memory allows, queued if not.

### Phase 2: Fused Decode Production Path

**Goal**: Activate the existing `BatchScheduler` + `FusedBatchDecode` infrastructure as the primary execution path.

**Files**:
- Modify: `Sources/NovaMLXEngine/BatchScheduler.swift` — add streaming support, interleaved prefill
- Modify: `Sources/NovaMLXEngine/FusedBatchDecode.swift` — add compaction, reduce padding waste
- Modify: `Sources/NovaMLXInference/InferenceService.swift` — route through new scheduler

**Key changes**:
1. `BatchScheduler` supports streaming (yield tokens per decode step)
2. Prefill interleaved with decode (one new request prefilled per decode step, not all at once)
3. FusedBatchKVCache compacts finished sequences
4. Wire into `InferenceService` as primary path

**Result**: Multiple concurrent requests share a single GPU forward pass. Throughput doubles+ for concurrent workloads.

### Phase 3: Prefix Cache Read-Path Activation

**Goal**: Make the existing prefix cache read-active so shared system prompts are not re-prefilled.

**Files**:
- Modify: `Sources/NovaMLXEngine/MLXEngine.swift` — call `fetchPrefix()` before prefill
- Modify: `Sources/NovaMLXPrefixCache/PrefixCacheManager.swift` — support KV restoration
- Modify: `Sources/NovaMLXEngine/FusedBatchDecode.swift` — shared prefix blocks

**Key changes**:
1. Before prefill, check prefix cache for matching blocks
2. Restore cached KV instead of computing from scratch
3. Multiple requests sharing system prompt reference same cache blocks
4. Reference-counted blocks freed when all referencing sequences complete

**Result**: Second Claude Code request with same system prompt skips ~90% of prefill. TTFT (Time To First Token) drops dramatically.

### Phase 4: Advanced Scheduling

**Goal**: Preemption, dynamic quantization, and predictive memory management.

**Features**:
1. **Preemption**: When memory runs out, preempt newest sequence, recompute later
2. **Dynamic KV quantization**: Active sequences at 4-bit, preempted sequences at 2-bit
3. **Predictive scaling**: Estimate future memory needs based on request rate
4. **Per-model concurrency tuning**: Auto-adjust `maxConcurrentPerModel` based on model size and available memory (like Ollama's auto `NUM_PARALLEL`)

---

## 6. Configuration

New configuration in `~/.nova/config.json`:

```json
{
  "scheduler": {
    "maxConcurrentPerModel": 2,
    "maxQueueDepth": 128,
    "memoryHeadroomPercent": 10,
    "defaultKVBits": 4,
    "prefixCacheEnabled": true
  }
}
```

Environment variable overrides (Ollama-style):
- `NOVA_NUM_PARALLEL` — max concurrent sequences per model (default: 2, auto-scale if memory allows)
- `NOVA_MAX_QUEUE` — max queued requests (default: 128)
- `NOVA_KV_BITS` — default KV cache quantization bits (default: 4)

---

## 7. Success Criteria

| Metric | Before | After (Phase 1) | After (Phase 2-3) |
|--------|--------|------------------|---------------------|
| Claude Code 2 concurrent requests | Crash (I/O error) | Both complete | Both complete, faster |
| Memory tracking accuracy | ~32x off (buggy) | Accurate ±10% | Accurate ±5% |
| 3 clients simultaneously | Crash | All queued and served | All served concurrently |
| TTFT for shared-prefix request | Full prefill | Full prefill | Skip cached prefix |
| Throughput (concurrent requests) | 1x (serialized by crash) | 1-2x (queued) | 2-4x (fused decode) |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Memory estimation inaccuracy | Feedback loop: compare estimated vs actual, adjust over time |
| TurboQuant not always available | Fallback to FP16 estimate (conservative) |
| MLX unified memory behavior differs from discrete GPU | Use `GPU.maxRecommendedWorkingSetBytes()` as the authoritative limit |
| Prefix cache hash collisions | Content hash uses SHA256 truncated to 128 bits |
| Preemption loses work | Only preempt sequences in prefill phase (cheap to recompute) |

---

## Appendix A: Claude Code Request Pattern

Per user message, Claude Code sends:
1. `POST /v1/messages` — `stream: true, max_tokens: 32000` — main response
2. `POST /v1/messages` — `stream: false, max_tokens: 21333, temperature: 1` — background

Additional lightweight requests for metadata (title, topic) may also be sent.

With `ANTHROPIC_BASE_URL` set, ALL requests route to NovaMLX. The model names may vary (e.g., `claude-haiku-4-5`, `claude-sonnet-*`). NovaMLX should map all unknown model names to the loaded model (already supported via model resolution).

## Appendix B: Reference Implementations

| System | Key Innovation | What We Adopt |
|--------|---------------|---------------|
| vLLM | PagedAttention (block-level KV) | Memory budget tracking, prefix caching concept |
| Ollama | Slot-based parallelism + FIFO queue | Per-model concurrency limit, queue depth limit |
| TGI | Router + continuous batching | Unified admission path |
| LMDeploy | Persistent batch + INT8 KV | KV quantization in admission estimate |

## Appendix C: Bug Fixes Included

These existing bugs are fixed as part of this work:

1. **KV cache double-counting** (MLXEngine.swift:134) — FIXED (removed `numLayers *`)
2. **Hardcoded head_dim=128** (MLXEngine.swift:135) — FIXED (reads from config.json)
3. **Wired memory ticket hardcoded** (MLXEngine.swift:265) — FIXED (uses actual bytesPerToken)
4. **EnginePool eviction bug** (EnginePool.swift:106) — `pool[evicted]` returns nil after `evictLRU()`, causing all models to be evicted
5. **Unused init parameters** (MLXEngine init) — `kvCacheCapacity` and `maxConcurrent` accepted but never used
