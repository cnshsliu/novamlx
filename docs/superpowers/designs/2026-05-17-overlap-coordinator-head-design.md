# Design: Coordinator-Head Speculation + Compute/Communication Overlap (Double Buffering)

**Date**: 2026-05-17  
**Status**: Design Phase (2.2) → Prototype (2.3)  
**Related TODO items**: Direction 1 (1.1–1.5) + Direction 2 (2.1–2.6)

## 1. Background and Strategic Context

From the May 2026 performance analysis (`t1.html`) and subsequent GLM/Claude discussion:

- On the current 2-node hardware (M4 Max 128GB coordinator + M4 24GB worker over Thunderbolt 4), **network is only ~3%** of per-token time. The real bottleneck is **imbalanced GPU compute** (strong early layers on coordinator, weak tail + head on worker).
- Classic speculative decoding using small external Transformer draft models has **diminishing returns** on modern hybrid-attention models (Mamba + Transformer blocks, long context, etc.).
- Two architecture-specific directions offer much higher leverage:
  1. **Coordinator-head based low-cost multi-token verification** — The strong coordinator can cheaply propose high-quality drafts using its own layers + head (or a few extra worker layers with throwaway caches).
  2. **Compute/Communication Overlap (Double Buffering / Token-level Pipelining)** — While the worker is verifying one batch, the coordinator starts the next.

These two directions are synergistic: good cheap drafts from the Coordinator head make overlap far more valuable (fewer wasted round-trips on bad drafts).

## 2. Current State (as of 2026-05-17)

### Working Today
- Full K>1 speculative decode loop (n-gram based) in both streaming and non-streaming paths (`DistributedInferenceRunner`).
- `speculativeVerify` + `verifiedTokens` protocol on the worker.
- `rollbackCache` on both sides (with Mamba snapshot support).
- `sendCompute` / `recvResult` split primitives in `RemoteShardPolicy` (used for pipelined prefill, **unused in decode**).

### Already Implemented but Not Integrated
- `SlicedForwardPolicy.draftTokens(from:activation:count:extraPredictionLayers:)` → `DraftBundle`
  - Takes a hidden state after the coordinator's layers.
  - Optionally runs a few extra "worker" layers (throwaway caches) for better first prediction.
  - Autoregressively predicts tokens using the coordinator's norm + head.
  - Returns both `predictedTokens` **and** `precomputedStates` (hidden states ready to send to the worker).
  - Captures Mamba snapshot for safe rollback.

- `SlicedForwardPolicy.rollbackCache(...)` with Mamba support.

### Not Yet Done
- The main speculative loop in `DistributedInferenceRunner` still only calls `specDecoder.speculate` (n-gram). `draftTokens()` on the sliced coordinator policy is never used in the remote-sampling path.
- No overlap / double-buffering in the decode loop (decode is still strictly "compute then wait").

## 3. Design Principles

1. **Coordinator-head drafts are preferred** when available (they are cheap on the strong node and produce better acceptance rates than pure n-gram).
2. **Overlap is built on top of speculation**, not instead of it. The unit of pipelining is a *speculative batch*, not a single token.
3. **Use existing infrastructure** (`sendCompute`/`recvResult`, `DraftBundle`, `speculativeVerify`, `rollbackCache`).
4. **Keep rollback simple and safe** — never overlap more than one speculative batch ahead without a clear flush point.
5. **Graceful degradation** — if Coordinator-head accuracy is low, fall back to n-gram or even single-token remote sampling.

## 4. Proposed Architecture (Combined Direction 1 + 2)

### 4.1 Two-Level Drafting (Coordinator Head + N-gram Fallback)

In `DistributedInferenceRunner`, the speculative proposer becomes:

```swift
func proposeDrafts(context: [Int], k: Int) -> [Int] {
    // Preferred: Coordinator-head (if SlicedForwardPolicy supports it and we have recent activation)
    if let sliced = currentSlicedCoord, let bundle = try? await sliced.draftTokens(from: lastActivation, count: k) {
        return bundle.predictedTokens
    }
    // Fallback: n-gram
    return specDecoder.speculate(context: context).prefix(k)
}
```

`DraftBundle` already gives us precomputed hidden states — on acceptance we can sometimes avoid re-sending the full activation for accepted positions (future optimization).

### 4.2 Overlap Model: Speculative-Batch Double Buffering

**Core idea**:
- While the Worker is running `speculativeVerify` on speculative batch N (last token + K drafts),
- the Coordinator immediately starts proposing and partially computing speculative batch N+1 (using its head on the already-accepted prefix).

This overlaps:
- Worker compute (verification of batch N)
- Coordinator compute (proposal + forward for batch N+1)

We use the existing split primitives:

- `workerPolicy.sendCompute(activationForBatchN)` — fire and forget
- Coordinator runs `draftTokens(...)` + prepares the next activation
- Later: `workerPolicy.recvResult()` or a new batched `speculativeVerify` variant that can be split

For the first prototype we can keep `speculativeVerify` synchronous for one batch while the next proposal runs in a `Task`.

### 4.3 Pipeline Sketch (Non-Streaming, Simplified)

```swift
// After first token
var lastActivation = ... // hidden after coordinator layers for the last real token

while ... {
    let drafts = proposeDrafts(...) // Coordinator-head preferred
    let k = drafts.count

    let seq = [lastRealToken] + drafts
    let activationBatch = try await slicedCoord.computeLayersOnly(input: MLXArray(seq))

    // Fire verification to worker (can later become non-blocking)
    let verifiedTask = Task {
        try await workerPolicy.speculativeVerify(input: activationBatch)
    }

    // === OVERLAP WINDOW: While worker verifies, prepare next batch ===
    // We can already start proposing the *next* set of drafts from the accepted prefix
    // (we don't know acceptance yet, so we propose conservatively from the common prefix)

    let verified = await verifiedTask.value
    let accepted = computeAcceptance(drafts, verified)

    // Apply accepted tokens + possible bonus
    ... update lastRealToken, lastActivation, generated list ...

    if rejected > 0 {
        try await workerPolicy.rollbackCache(...)
        // Coordinator also rolls back its caches using DraftBundle.mambaSnapshot if available
    }
}
```

A more aggressive version would use `sendCompute` + `recvResult` to truly overlap the *verification* of batch N with the *forward* of batch N+1.

## 5. Detailed Design for 2.3 Prototype (Minimal Viable Overlap)

**Scope for first prototype (non-streaming only)**:

1. Add a config flag `enableCoordinatorHeadSpeculation` (default true when remote sampling + 2 nodes).
2. In the speculative round:
   - First try `slicedCoord.draftTokens(...)`.
   - If it returns a usable `DraftBundle`, use those tokens + precomputed states.
   - Fall back to n-gram if the head path fails or returns low quality.
3. Keep the current `speculativeVerify` call synchronous for one batch.
4. While waiting for `speculativeVerify`, run a background `Task` that prepares the *next* proposal using the currently accepted prefix (conservative).
5. On the first successful integration, add timing logs:
   - `draftMs`
   - `verifyMs`
   - `overlapMs` (time the coordinator spent usefully working while worker was verifying)

**Rollback handling**:
- `DraftBundle` already carries `mambaSnapshot`.
- On rejection we call `slicedCoord.rollbackCache(position: ..., speculatedCount: ..., mambaSnapshot: ...)` (method already exists).

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Coordinator-head accuracy lower than expected on some models | Keep n-gram as strong fallback; add runtime acceptance rate tracking per proposer type |
| Cache rollback complexity with overlapped batches | Prototype only overlaps **one batch ahead**; never more than two in flight |
| Streaming latency regression | Overlap logic only in non-streaming path for 2.3; streaming gets the integrated Coordinator-head proposer without aggressive pipelining yet |
| Mamba models | `DraftBundle` and `rollbackCache` already carry snapshots — must be exercised in tests |

## 7. Success Metrics (to be measured on real hardware)

- Acceptance rate of Coordinator-head drafts vs n-gram (target: >45–50% on Qwen3-8B class).
- Effective tokens per speculative round.
- Wall-clock time per speculative round (with overlap) vs without.
- Overall tok/s improvement on the 58/8 Thunderbolt setup (target for combined work: 30–50%+ over current n-gram speculative baseline).

## 8. Next Steps (Immediate)

1. **2.2 Complete** — This document.
2. **Integrate Coordinator-head proposer** (TODO 1.2) into `DistributedInferenceRunner` speculative loop (both paths).
3. **2.3 Prototype** — Minimal non-streaming overlap using `Task` + current `speculativeVerify` while preparing next drafts.
4. Add rich logging (`[CoordHead]`, `[Overlap]`, acceptance by proposer).
5. Measure on real 2-node cluster.

---

This design deliberately builds on the excellent `draftTokens()` + `DraftBundle` work that already exists, rather than starting from scratch. The overlap is a natural evolution once we have cheap, high-quality drafts coming from the strong coordinator node.