# Distributed Inference Performance Optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve distributed inference from 7.2 tok/s to 18+ tok/s via shard rebalancing, bfloat16 transport, continuous async pipeline, and improved draft accuracy.

**Architecture:** Move lm_head from worker to coordinator so the worker becomes a pure transformer-layer node (~17ms vs ~33ms). Send bfloat16 hidden states over TCP (~7KB vs ~14KB). Rewrite decode loop with double-buffer async pipeline where coord and worker compute simultaneously.

**Tech Stack:** Swift, MLX (bfloat16 native on Apple Silicon), TCP sockets, async/await.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `NovaMLXDistributed/ShardableModel.swift` | Add `computeHeadOnly()` to `SlicedForwardPolicy` — coord runs norm+head+argmax |
| `NovaMLXDistributed/RemoteShardPolicy.swift` | bfloat16 quantize before send, dequantize after recv; remove `computeAndSample` |
| `NovaMLXDistributed/WorkerShardService.swift` | Worker always returns hidden state, never runs head; remove `handleComputeAndSample` |
| `NovaMLXDistributed/DistributedInferenceRunner.swift` | Rewrite decode loop with double-buffer async pipeline |
| `NovaMLXDistributed/ClusterModelManager.swift` | Adjust shard plan: coord gets `isLast=true`, worker gets `isLast=false` |
| `NovaMLXDistributed/WorkerShardService.swift` | Remove `handleComputeAndSample`, simplify `handleCompute` to always return hidden state |
| `NovaMLXDistributed/ClusterAdminRoutes.swift` | Add decode timing breakdown to inference stats |
| `NovaMLXDistributed/DistributedTypes.swift` | Add `DistributedDecodeStats` with per-component timing |

---

### Task 1: Add `computeHeadOnly()` to SlicedForwardPolicy

The coordinator needs to run norm + lm_head + argmax on a hidden state received from the worker. Currently `compute()` runs layers + head (when `isLast=true`). We need a separate method that ONLY runs head.

**Files:**
- Modify: `Sources/NovaMLXDistributed/ShardableModel.swift:322-380` (SlicedForwardPolicy)
- Test: `Tests/NovaMLXDistributedTests/ShardEngineTests.swift`

- [ ] **Step 1: Add `computeHeadOnly()` method to SlicedForwardPolicy**

Add after the `compute()` method (after line ~380):

```swift
/// Run norm + lm_head + argmax on a hidden state tensor.
/// Used by the coordinator after receiving the worker's output.
/// Returns (sampledTokenId, logits).
func computeHeadOnly(_ hidden: MLXArray) -> (tokenId: Int, logits: MLXArray) {
    let logits = shardable.head(hidden)
    let squeezed = logits.ndim > 2 ? logits.squeezed(dim: 0) : logits
    let lastLogits = squeezed.ndim > 1 ? squeezed[-1..., 0...] : squeezed
    let tokenId = argmaxToken(lastLogits)
    return (tokenId, logits)
}
```

- [ ] **Step 2: Build and verify compilation**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXDistributed/ShardableModel.swift
git commit -m "feat(distributed): add computeHeadOnly for coordinator head+argmax"
```

---

### Task 2: Shard Rebalancing — Move lm_head to Coordinator

Change the shard assignment so the coordinator gets `isLast=true` and the worker gets `isLast=false`. This means:
- Coordinator: layers 0-52 + norm + lm_head (isFirst=true, isLast=true)
- Worker: layers 53-65 only (isFirst=false, isLast=false)

**Files:**
- Modify: `Sources/NovaMLXDistributed/ClusterModelManager.swift:315-390` (performActivation)
- Modify: `Sources/NovaMLXDistributed/WorkerShardService.swift:280-315` (handleBindWeights)

- [ ] **Step 1: Change shard assignment in ClusterModelManager**

In `performActivation()`, after creating the `ShardPlan` (around line 310), modify the loop that creates policies. For 2-node setups, the coordinator gets both `isFirst=true` AND `isLast=true`:

Find the loop around line 324:
```swift
for (index, assignment) in plan.assignments.enumerated() {
    let isFirst = index == 0
    let isLast = index == plan.assignments.count - 1
```

Replace with:
```swift
for (index, assignment) in plan.assignments.enumerated() {
    // In 2-node pipeline, coordinator owns head (isLast=true) so worker
    // is a pure transformer-layer node. This cuts worker compute in half.
    let isFirst = index == 0
    let isLast = plan.assignments.count <= 2 || index == plan.assignments.count - 1
```

This gives the coordinator `isLast=true` in a 2-node setup, so it runs embedding + layers + norm + head. The worker only runs its layers.

- [ ] **Step 2: Update ShardAssignmentPayload sent to worker**

In the same loop, when sending `ShardAssignmentPayload` to the worker, the `isLast` flag is already taken from the loop variable. With the change above, the worker gets `isLast=false` — so `SlicedForwardPolicy` on the worker will NOT run head. The coordinator gets `isLast=true` and WILL run head.

No additional changes needed — the `ShardAssignmentPayload.isLast` is already passed through.

- [ ] **Step 3: Update WorkerShardService handleBindWeights**

In `handleBindWeights()` (line 309), the worker creates `SlicedForwardPolicy` with:
```swift
isLast: isShardLast
```

`isShardLast` comes from the `ShardAssignmentPayload.isLast` field. Since we now send `isLast=false` for the worker in 2-node setups, this is automatically correct. No code change needed here.

- [ ] **Step 4: Verify the change compiles**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/ClusterModelManager.swift
git commit -m "feat(distributed): rebalance shards — coord owns lm_head, worker is pure layers"
```

---

### Task 3: bfloat16 Transport

Add bfloat16 quantization before TCP send and dequantization after TCP recv. The worker computes directly in bfloat16 (native on Apple Silicon).

**Files:**
- Modify: `Sources/NovaMLXDistributed/RemoteShardPolicy.swift:96-175` (compute and sendCompute/recvResult)
- Modify: `Sources/NovaMLXDistributed/WorkerShardService.swift:363-388` (handleCompute response)

- [ ] **Step 1: Add bfloat16 conversion helpers**

At the top of `RemoteShardPolicy.swift`, add two private helper methods:

```swift
/// Quantize activation to bfloat16 for transport (~2x compression).
private func quantizeForTransport(_ array: MLXArray) -> MLXArray {
    if array.dtype == .bfloat16 { return array }
    return array.asType(.bfloat16)
}

/// Dequantize bfloat16 back to float32 for head computation.
private func dequantizeFromTransport(_ array: MLXArray) -> MLXArray {
    if array.dtype == .float32 { return array }
    return array.asType(.float32)
}
```

- [ ] **Step 2: Apply bfloat16 in RemoteShardPolicy.compute()**

In the TCP transport path (the `else` branch starting around line 147), replace:

```swift
try conn.sendTensor(input)
```
with:
```swift
try conn.sendTensor(quantizeForTransport(input))
```

And for the response, replace:
```swift
return try conn.recvTensor()
```
with:
```swift
let raw = try conn.recvTensor()
return dequantizeFromTransport(raw)
```

- [ ] **Step 3: Apply bfloat16 in RemoteShardPolicy.sendCompute() and recvResult()**

In `sendCompute()` (around line 230), change the TCP path:
```swift
try conn.sendTensor(input)
```
→
```swift
try conn.sendTensor(quantizeForTransport(input))
```

In `recvResult()` (around line 290), change the TCP path:
```swift
return try conn.recvTensor()
```
→
```swift
return dequantizeFromTransport(try conn.recvTensor())
```

- [ ] **Step 4: Worker sends bfloat16 response**

In `WorkerShardService.handleCompute()` (around line 380), the worker already returns whatever `policy.compute()` produces. Since the input is bfloat16, and MLX transformer layers compute natively in bfloat16 on Apple Silicon, the output will naturally be bfloat16. No change needed on the worker side — the data flows through as-is.

However, to ensure the response is bfloat16 (in case the model upcasts internally), add before sending:

```swift
let outputToSend = output.dtype != .bfloat16 ? output.asType(.bfloat16) : output
```

Then send `outputToSend` instead of `output` in the TCP path.

- [ ] **Step 5: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 6: Commit**

```bash
git add Sources/NovaMLXDistributed/RemoteShardPolicy.swift Sources/NovaMLXDistributed/WorkerShardService.swift
git commit -m "feat(distributed): bfloat16 transport — 2x smaller tensors over TCP"
```

---

### Task 4: Continuous Async Pipeline — Double-Buffer Decode

Rewrite the decode loop so coordinator and worker compute simultaneously. The key: while the worker processes token N, the coordinator drafts token N+1 and precomputes its activation. This task subsumes the `computeAndSample` → `compute` transition — the new loop never uses `computeAndSample`.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` (both `generate()` and `stream()` decode loops)

- [ ] **Step 1: Rewrite the speculative decode loop in `generate()`**

Replace the decode loop (approximately lines 341-476) with the continuous pipeline version:

```swift
// === CONTINUOUS ASYNC PIPELINE ===
// Double-buffer: coord drafts N+1 while worker computes N.
// If draft hits, precomputed activation is used — zero idle time.

var activation = prefillOutput  // hidden state from prefill
var currentPosition = promptTokens.count

// First token: coord runs head on prefill output
let firstToken = slicedCoord.computeHeadOnly(activation).tokenId
if eosTokenIds.contains(firstToken) {
    // empty response
} else {
    generatedTokenIds.append(firstToken)
}

// Compute coord's activation for the first generated token
activation = try await coordPolicy.compute(input: MLXArray(Int32(firstToken)))
currentPosition += 1

var timingLogInterval = 0

while generatedTokenIds.count < maxTokens {
    let t0 = CFAbsoluteTimeGetCurrent()

    // Step 1: Fire worker compute async (token N)
    let activationBox = SendableBox(activation)
    let workerTask = Task {
        try await workerPolicy.compute(input: activationBox.value)
    }

    // Step 2: Draft token N+1 while worker is busy
    let draft = try? await slicedCoord.draftTokens(
        from: activation, count: draftLength, extraPredictionLayers: extraPredictionLayers
    )
    let tDraft = CFAbsoluteTimeGetCurrent()

    // Step 3: Await worker result (bfloat16 hidden state)
    let workerHidden = try await workerTask.value
    let (actualToken, _) = slicedCoord.computeHeadOnly(workerHidden)
    let tWorkerDone = CFAbsoluteTimeGetCurrent()

    // Step 4: Verify draft
    if let draft = draft, draft.count > 0, draft.predictedTokens[0] == actualToken {
        // HIT — use precomputed activation
        activation = draft.precomputedStates[0]
        recordPrediction(true, &recentPredictions, &correctPredictions, &totalPredictions)
    } else {
        // MISS — recompute with actual token
        recordPrediction(false, &recentPredictions, &correctPredictions, &totalPredictions)
        if let draft = draft {
            try? await slicedCoord.rollbackCache(
                position: currentPosition,
                speculatedCount: draft.count,
                mambaSnapshot: draft.mambaSnapshot
            )
        }
        activation = try await coordPolicy.compute(input: MLXArray(Int32(actualToken)))
    }
    currentPosition += 1

    if eosTokenIds.contains(actualToken) { break }
    generatedTokenIds.append(actualToken)

    let fullText = tokenizer.decode(generatedTokenIds)
    if stopTokens.contains(where: { fullText.hasSuffix($0) }) { break }

    // Timing log every 20 tokens
    timingLogInterval += 1
    if timingLogInterval % 20 == 1 {
        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let draftMs = (tDraft - t0) * 1000
        let workerMs = (tWorkerDone - tDraft) * 1000
        let accuracy = totalPredictions > 0 ? Double(correctPredictions) / Double(totalPredictions) : 0
        NovaMLXLog.info("[Pipeline] token \(generatedTokenIds.count): \(String(format: "%.1f", totalMs))ms draft=\(String(format: "%.1f", draftMs))ms wait=\(String(format: "%.1f", workerMs))ms acc=\(String(format: "%.0f%%", accuracy * 100))")
    }

    // Adaptive: disable speculation if accuracy too low
    if recentPredictions.count >= adaptiveWindow {
        let rolling = Double(recentPredictions.filter { $0 }.count) / Double(recentPredictions.count)
        if rolling < disableThreshold {
            speculationEnabled = false
            NovaMLXLog.info("[Pipeline] Speculation disabled: \(String(format: "%.0f%%", rolling * 100)) accuracy")
        }
    }
}

// Fallback: sequential path when speculation disabled
if !speculationEnabled {
    while generatedTokenIds.count < maxTokens {
        let workerHidden = try await workerPolicy.compute(input: activation)
        let (sampledId, _) = slicedCoord.computeHeadOnly(workerHidden)
        if eosTokenIds.contains(sampledId) { break }
        generatedTokenIds.append(sampledId)
        let fullText = tokenizer.decode(generatedTokenIds)
        if stopTokens.contains(where: { fullText.hasSuffix($0) }) { break }
        activation = try await coordPolicy.compute(input: MLXArray(Int32(sampledId)))
    }
}
```

Note: The `else` branch (non-speculative, starting at original line 477) and the `speculationCapable == false` branch are replaced by the single continuous loop above with the adaptive fallback at the end.

- [ ] **Step 2: Apply the same rewrite to the `stream()` method's decode loop**

The streaming method (around lines 760-891) has an identical decode loop structure. Apply the same continuous pipeline pattern, but replace `generatedTokenIds.append(sampledId)` with `if yieldToken(sampledId) { break }` and remove the stop token text check (already handled by yieldToken).

- [ ] **Step 3: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedInferenceRunner.swift
git commit -m "feat(distributed): continuous async pipeline — coord+worker compute simultaneously"
```

---

### Task 5: Add Decode Timing Breakdown to Stats

Add per-component timing to `DistributedInferenceStats` for profiling and the admin API.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` (stats struct and recording)
- Modify: `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift` (stats reporting)

- [ ] **Step 1: Extend DistributedInferenceStats**

In `DistributedInferenceStats` (around line 990), add timing fields:

```swift
public let coordComputeMs: Double?
public let workerComputeMs: Double?
public let transportMs: Double?
public let headMs: Double?
```

Update the `init` to accept these with defaults of `nil`.

- [ ] **Step 2: Record timing in the decode loop**

Add timing instrumentation in the continuous pipeline decode loop (from Task 4):

```swift
let tCoordStart = CFAbsoluteTimeGetCurrent()
let draft = try? await slicedCoord.draftTokens(...)
let tCoordEnd = CFAbsoluteTimeGetCurrent()
let coordMs = (tCoordEnd - tCoordStart) * 1000

let workerHidden = try await workerTask.value
let tWorkerEnd = CFAbsoluteTimeGetCurrent()
let workerMs = (tWorkerEnd - tCoordEnd) * 1000

let (_, _) = slicedCoord.computeHeadOnly(workerHidden)
let tHeadEnd = CFAbsoluteTimeGetCurrent()
let headMs = (tHeadEnd - tWorkerEnd) * 1000
```

Record in stats:
```swift
DistributedInferenceRunnerCache.shared.recordStats(DistributedInferenceStats(
    tokensPerSecond: tps,
    promptTokens: promptTokens.count,
    completionTokens: generatedTokenIds.count,
    elapsedSeconds: elapsed,
    speculationAccuracy: accuracy,
    coordComputeMs: coordMs,
    workerComputeMs: workerMs,
    transportMs: nil,  // not separately measurable in overlap mode
    headMs: headMs
))
```

- [ ] **Step 3: Report timing in ClusterAdminRoutes.encodeModelStatus()**

In `encodeModelStatus()`, extend the `inferenceStats` dict:

```swift
if let stats = DistributedInferenceRunnerCache.shared.lastStats {
    result["inferenceStats"] = [
        "tokensPerSecond": stats.tokensPerSecond,
        "promptTokens": stats.promptTokens,
        "completionTokens": stats.completionTokens,
        "elapsedSeconds": stats.elapsedSeconds,
        "speculationAccuracy": stats.speculationAccuracy as Any,
        "coordComputeMs": stats.coordComputeMs as Any,
        "workerComputeMs": stats.workerComputeMs as Any,
        "headMs": stats.headMs as Any,
        "timestampAgo": String(format: "%.0fs", Date().timeIntervalSince(stats.timestamp)),
    ] as [String: Any]
}
```

- [ ] **Step 4: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedInferenceRunner.swift Sources/NovaMLXDistributed/ClusterAdminRoutes.swift
git commit -m "feat(distributed): add per-component timing to inference stats"
```

---

### Task 6: Integration Test — Measure and Verify

Deploy to both machines, activate distributed model, and measure tok/s.

**Files:** No code changes — validation only.

- [ ] **Step 1: Build and deploy**

```bash
./build.sh
killall NovaMLX; sleep 2; open dist/NovaMLX.app
```

Wait for models to load.

- [ ] **Step 2: Activate distributed model via admin API**

```bash
curl -s -X POST -H "Authorization: Bearer abcd1234" \
  http://127.0.0.1:6591/admin/api/cluster/activate-model \
  -H "Content-Type: application/json" \
  -d '{"modelId":"mlx-community/Qwen3.6-27B-4bit"}'
```

Verify `"state": "ready"` and both nodes have readiness status.

- [ ] **Step 3: Run streaming inference test**

```bash
curl -s -N -H "Authorization: Bearer abcd1234" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:6590/v1/chat/completions \
  -d '{"model":"mlx-community/Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Explain quantum computing in 3 sentences."}],"stream":true,"max_tokens":200}'
```

- [ ] **Step 4: Check logs for tok/s and timing breakdown**

```bash
grep "Pipeline\|Stream completed\|Completed:" ~/.nova/novamlx.log | tail -5
```

Expected: tok/s > 15 (up from 7.2). Check draft accuracy, coord/worker/head timing.

- [ ] **Step 5: Check admin API for stats**

```bash
curl -s -H "Authorization: Bearer abcd1234" \
  http://127.0.0.1:6591/admin/api/cluster/model-status | python3 -m json.tool
```

Verify `inferenceStats` has timing breakdown.

- [ ] **Step 6: Run non-streaming comparison**

```bash
curl -s -H "Authorization: Bearer abcd1234" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:6590/v1/chat/completions \
  -d '{"model":"mlx-community/Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Say hello."}],"max_tokens":200}'
```

Compare tok/s between streaming and non-streaming.

- [ ] **Step 7: Commit final validation**

If performance target (18+ tok/s) is met, commit any log-level tweaks:

```bash
git add -A
git commit -m "feat(distributed): continuous async pipeline delivers 18+ tok/s"
```

If performance is below target, investigate the timing breakdown and iterate.
