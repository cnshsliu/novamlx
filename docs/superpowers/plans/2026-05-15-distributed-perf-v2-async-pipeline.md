# Distributed Perf v2: 2-Stage Async Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat single-node speed (18+ tok/s) by replacing the speculative decode loop with a 2-stage async pipeline where coord and worker compute simultaneously.

**Architecture:** Fire worker compute at the bottom of each decode iteration, await it at the top of the next. This overlaps coord's 33ms `computeLayersOnly()` with the worker's ~29ms compute. No speculation needed. Also rebalance layers using GPU compute ratio so the M4 base Mini (4.3x slower per layer) gets fewer layers (11 instead of 13).

**Tech Stack:** Swift, MLX, async/await, TCP sockets (existing).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `Sources/NovaMLXDistributed/DistributedTypes.swift` | Add `computeRatio(for cpuModel:)` helper; update `ShardPlan` compute init to weight by ratio |
| `Sources/NovaMLXDistributed/ClusterModelManager.swift` | Pass `cpuModel` to coordinator's `NodeSpec` |
| `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` | Rewrite generate() and stream() decode loops with 2-stage overlap; remove speculation; add overlap stats |
| `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift` | Add overlap stats to admin API reporting |

---

### Task 1: Add Compute Ratio Lookup

Add a function that maps Apple Silicon CPU model strings to relative GPU compute ratios.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedTypes.swift`

- [ ] **Step 1: Add `computeRatio` function after the `NodeSpec` struct (after line ~95)**

Add a free function near the `NodeSpec` definition:

```swift
/// Map Apple Silicon CPU model string to relative GPU compute ratio.
/// Used by ShardPlan to weight layer assignment by GPU capability.
/// M4 Max = 1.0 baseline (40 GPU cores).
public func computeRatio(for cpuModel: String) -> Double {
    if cpuModel.contains("M4 Max") { return 1.0 }
    if cpuModel.contains("M4 Pro") { return 0.50 }
    if cpuModel.contains("M4") { return 0.23 }      // base M4 (10 cores)
    if cpuModel.contains("M3 Max") { return 0.75 }
    if cpuModel.contains("M3 Pro") { return 0.38 }
    if cpuModel.contains("M3") { return 0.19 }
    if cpuModel.contains("M2 Max") { return 0.75 }
    if cpuModel.contains("M2 Pro") { return 0.38 }
    if cpuModel.contains("M2") { return 0.19 }
    return 0.2  // conservative default for unknown chips
}
```

- [ ] **Step 2: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 3: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedTypes.swift
git commit -m "feat(distributed): add computeRatio helper for GPU capability detection"
```

---

### Task 2: Update ShardPlan to Weight by Compute Ratio

Modify the `ShardPlan` compute initializer so the `.spread` strategy weights layer assignment by compute ratio instead of just memory.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedTypes.swift` (ShardPlan init, lines 185-275)

- [ ] **Step 1: Read the ShardPlan compute initializer**

Read `Sources/NovaMLXDistributed/DistributedTypes.swift` lines 185-275 to understand the current `.spread` strategy implementation.

- [ ] **Step 2: Update the `.spread` strategy to use compute ratio**

In the `.spread` case (around line 235), the current code distributes layers proportional to `memoryWeight`. Add a compute-ratio weighting factor.

Find the memory weight calculation (around line 248-252):
```swift
let totalMemory = nodes.reduce(Int64(0)) { $0 + Int64($1.totalMemoryBytes) }
```

After the `totalMemory` line, add compute ratio weighting:
```swift
let totalMemory = nodes.reduce(Int64(0)) { $0 + Int64($1.totalMemoryBytes) }
let computeRatios = nodes.map { computeRatio(for: $0.cpuModel) }
let totalCompute = computeRatios.reduce(0.0, +)
```

Then find where `layersForNode` is calculated (the proportional split). It currently uses `memoryWeight = Double(node.totalMemoryBytes) / Double(totalMemory)`. Replace it with a combined weight:

```swift
let memoryWeight = totalMemory > 0 ? Double(node.totalMemoryBytes) / Double(totalMemory) : equalShare
let computeWeight = totalCompute > 0 ? computeRatios[i] / totalCompute : equalShare
// Blend: 30% memory, 70% compute — compute capability matters more for throughput
let combinedWeight = 0.3 * memoryWeight + 0.7 * computeWeight
```

Then use `combinedWeight` instead of `memoryWeight` for the `layersForNode` calculation.

- [ ] **Step 3: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedTypes.swift
git commit -m "feat(distributed): weight layer assignment by compute ratio in ShardPlan"
```

---

### Task 3: Pass cpuModel to Coordinator's NodeSpec

The coordinator currently creates its `NodeSpec` with `cpuModel: ""` (default). This means the compute ratio lookup returns the conservative default (0.2). Fix it to detect the local machine's CPU.

**Files:**
- Modify: `Sources/NovaMLXDistributed/ClusterModelManager.swift`

- [ ] **Step 1: Add local CPU model detection**

In `performActivation()`, before the `coordinatorSpec` construction (around line 245), add:

```swift
let localCpuModel = Sysctl.string("machdep.cpu.brand_string") ?? "Apple M4"
```

This requires adding an import for the Sysctl helper. Check if it's already available via `import Foundation` or needs a helper. If not available, use:

```swift
var size = 0
sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
var cpu = [CChar](repeating: 0, count: size)
sysctlbyname("machdep.cpu.brand_string", &cpu, &size, nil, 0)
let localCpuModel = String(cString: cpu)
```

- [ ] **Step 2: Update coordinator NodeSpec construction**

Change the `coordinatorSpec` (around line 250) to include cpuModel:

```swift
let coordinatorSpec = NodeSpec(
    nodeId: "local-coordinator",
    totalMemoryBytes: localMemory,
    computeCapability: 1.0,
    hostname: "127.0.0.1",
    port: clusterConfig?.coordinatorPort ?? 6591,
    cpuModel: localCpuModel
)
```

- [ ] **Step 3: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXDistributed/ClusterModelManager.swift
git commit -m "feat(distributed): detect local CPU model for compute-ratio-aware sharding"
```

---

### Task 4: Rewrite generate() Decode Loop with 2-Stage Async Pipeline

Replace the speculative decode loop in `generate()` with the 2-stage overlap pipeline. Remove all speculation variables and logic.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` (lines 307-536)

- [ ] **Step 1: Read the current decode loop**

Read `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` lines 307-536 to understand the full scope of changes needed.

- [ ] **Step 2: Replace the speculation-capable decode path (lines 340-494)**

Replace the entire speculation-capable section (from `if speculationCapable {` at line 340 through its closing `}` before the non-speculative path at line 496) with the new 2-stage pipeline.

The new code:

```swift
        // === 2-STAGE ASYNC PIPELINE ===
        // Coord and worker compute simultaneously.
        // Fire worker at bottom of iteration, await at top of next.
        if remoteSamplingEnabled && shardEngines.count == 2 {
            let coordPolicy = shardEngines[0].policy
            guard let slicedCoord = coordPolicy as? SlicedForwardPolicy else {
                throw DistributedInferenceError.shardPlanFailed("Coord policy is not SlicedForwardPolicy")
            }
            let workerPolicy = shardEngines[1].policy as! RemoteShardPolicy

            // Prefill already ran, activation holds the worker's hidden state
            // First token: run head on prefill output
            guard let firstHeadResult = await slicedCoord.computeHeadOnly(activation) else {
                throw DistributedInferenceError.shardPlanFailed("computeHeadOnly returned nil on prefill output")
            }
            let firstToken = firstHeadResult.tokenId
            if !eosTokenIds.contains(firstToken) {
                generatedTokenIds.append(firstToken)
            } else {
                // EOS on first token — empty response
                if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                return InferenceResult(text: "", promptTokens: promptTokens.count, completionTokens: 0)
            }

            // Compute coord activation for first generated token
            activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(firstToken)))

            // Warmup: fire first worker compute
            let warmupBox = SendableBox(activation)
            var pendingWorker = Task {
                SendableBox(try await workerPolicy.compute(input: warmupBox.value))
            }

            let decodeStart = CFAbsoluteTimeGetCurrent()
            var timingLogCounter = 0

            while generatedTokenIds.count < maxTokens {
                let t0 = CFAbsoluteTimeGetCurrent()

                // Stage A: Await worker result (was computing during previous iteration)
                let workerHidden = try await pendingWorker.value.value
                let tAwait = CFAbsoluteTimeGetCurrent()

                // Stage B: Run head to get token
                guard let headResult = await slicedCoord.computeHeadOnly(workerHidden) else { break }
                let actualToken = headResult.tokenId

                if eosTokenIds.contains(actualToken) { break }
                generatedTokenIds.append(actualToken)

                let fullText = tokenizer.decode(generatedTokenIds)
                if stopTokens.contains(where: { fullText.hasSuffix($0) }) { break }

                // Stage C: Compute coord activation for this token
                activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(actualToken)))
                let tCoordDone = CFAbsoluteTimeGetCurrent()

                // Stage D: Fire worker for next iteration (overlaps with next loop iteration's await)
                let activationBox = SendableBox(activation)
                pendingWorker = Task {
                    SendableBox(try await workerPolicy.compute(input: activationBox.value))
                }

                // Timing log every 20 tokens
                timingLogCounter += 1
                if timingLogCounter % 20 == 1 {
                    let totalMs = (tCoordDone - t0) * 1000
                    let awaitMs = (tAwait - t0) * 1000
                    let headMs = (tCoordDone - tAwait) * 1000
                    let overlapPct = awaitMs < 1.0 ? 100.0 : max(0, (1.0 - awaitMs / totalMs) * 100)
                    NovaMLXLog.info("[Pipeline] token \(generatedTokenIds.count): \(String(format: "%.1f", totalMs))ms await=\(String(format: "%.1f", awaitMs))ms head+coord=\(String(format: "%.1f", headMs))ms overlap=\(String(format: "%.0f%%", overlapPct))")
                }
            }

            let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
            let decodeTps = Double(generatedTokenIds.count) / decodeElapsed
            NovaMLXLog.info("[Distributed] Pipeline done: \(generatedTokenIds.count) tokens, \(String(format: "%.1f", decodeTps)) tok/s")
        } else {
```

The `else` block at the end connects to the existing non-speculative decode loop (lines 496-536) which remains unchanged.

- [ ] **Step 3: Remove speculation variables from setup**

In the setup section (around lines 319-336), remove the speculation-related variables:
- Delete `let speculationCapable = ...`
- Delete `var speculationEnabled = speculationCapable`
- Delete `var recentPredictions: [Bool] = []`
- Delete `var correctPredictions = 0, totalPredictions = 0`
- Delete `let adaptiveWindow = 20`
- Delete `let disableThreshold = 0.30`
- Delete `let draftLength = 1`
- Delete `let extraPredictionLayers = 5`

Keep `remoteSamplingEnabled` since it's used by the new pipeline.

Also remove the `recordPrediction` helper function if it exists as a local function.

- [ ] **Step 4: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedInferenceRunner.swift
git commit -m "feat(distributed): 2-stage async pipeline for generate() — overlap coord+worker compute"
```

---

### Task 5: Rewrite stream() Decode Loop with 2-Stage Async Pipeline

Apply the same 2-stage pipeline pattern to the `stream()` method's decode loop.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` (lines 748-960)

- [ ] **Step 1: Read the current stream() decode loop**

Read `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` lines 748-960.

- [ ] **Step 2: Replace the speculation-capable decode path in stream()**

The stream() method has an identical structure to generate() but uses `yieldToken()` instead of `generatedTokenIds.append()`. Replace the speculation-capable section with:

```swift
                    // === 2-STAGE ASYNC PIPELINE (streaming) ===
                    if remoteSamplingEnabled && shardEngines.count == 2 {
                        let coordPolicy = shardEngines[0].policy
                        guard let slicedCoord = coordPolicy as? SlicedForwardPolicy else {
                            throw DistributedInferenceError.shardPlanFailed("Coord policy is not SlicedForwardPolicy")
                        }
                        let workerPolicy = shardEngines[1].policy as! RemoteShardPolicy

                        // First token from prefill
                        guard let firstHeadResult = await slicedCoord.computeHeadOnly(activation) else {
                            throw DistributedInferenceError.shardPlanFailed("computeHeadOnly returned nil on prefill output")
                        }
                        let firstToken = firstHeadResult.tokenId
                        if !eosTokenIds.contains(firstToken) {
                            if yieldToken(firstToken) {
                                if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                                continuation.finish()
                                return
                            }
                        } else {
                            if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                            continuation.finish()
                            return
                        }

                        // Compute coord activation for first token
                        activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(firstToken)))

                        // Warmup: fire first worker compute
                        let warmupBox = SendableBox(activation)
                        var pendingWorker = Task {
                            SendableBox(try await workerPolicy.compute(input: warmupBox.value))
                        }

                        var timingLogCounter = 0

                        while generatedCount < maxTokens {
                            let t0 = CFAbsoluteTimeGetCurrent()

                            // Await worker result
                            let workerHidden = try await pendingWorker.value.value
                            let tAwait = CFAbsoluteTimeGetCurrent()

                            // Run head to get token
                            guard let headResult = await slicedCoord.computeHeadOnly(workerHidden) else { break }
                            let actualToken = headResult.tokenId

                            if eosTokenIds.contains(actualToken) { break }

                            generatedCount += 1
                            if yieldToken(actualToken) {
                                if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                                break
                            }

                            // Compute coord activation
                            activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(actualToken)))
                            let tCoordDone = CFAbsoluteTimeGetCurrent()

                            // Fire worker for next iteration
                            let activationBox = SendableBox(activation)
                            pendingWorker = Task {
                                SendableBox(try await workerPolicy.compute(input: activationBox.value))
                            }

                            // Timing log every 20 tokens
                            timingLogCounter += 1
                            if timingLogCounter % 20 == 1 {
                                let totalMs = (tCoordDone - t0) * 1000
                                let awaitMs = (tAwait - t0) * 1000
                                let overlapPct = awaitMs < 1.0 ? 100.0 : max(0, (1.0 - awaitMs / totalMs) * 100)
                                NovaMLXLog.info("[Pipeline-Stream] token \(generatedCount): \(String(format: "%.1f", totalMs))ms await=\(String(format: "%.1f", awaitMs))ms overlap=\(String(format: "%.0f%%", overlapPct))")
                            }
                        }
                    } else {
```

- [ ] **Step 3: Remove speculation variables from stream() setup**

Same as Task 4 — remove `speculationCapable`, `speculationEnabled`, `recentPredictions`, `correctPredictions`, `totalPredictions`, `adaptiveWindow`, `disableThreshold`, `draftLength`, `extraPredictionLayers` from the stream() setup section.

- [ ] **Step 4: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 5: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedInferenceRunner.swift
git commit -m "feat(distributed): 2-stage async pipeline for stream() — overlap coord+worker compute"
```

---

### Task 6: Add Overlap Stats to DistributedInferenceStats

Add `overlapPct` and `workerWaitMs` fields to the stats struct and admin API reporting.

**Files:**
- Modify: `Sources/NovaMLXDistributed/DistributedInferenceRunner.swift` (DistributedInferenceStats struct)
- Modify: `Sources/NovaMLXDistributed/ClusterAdminRoutes.swift` (stats reporting)

- [ ] **Step 1: Add fields to DistributedInferenceStats**

In the `DistributedInferenceStats` struct (near line 1063), add after `headMs`:

```swift
public let workerWaitMs: Double?
public let overlapPct: Double?
```

Update the `init` to accept these with defaults of `nil`.

- [ ] **Step 2: Report in ClusterAdminRoutes**

In `ClusterAdminRoutes.swift`, find where `inferenceStats` is reported (search for `headMs`). Add after the `headMs` line:

```swift
"workerWaitMs": stats.workerWaitMs as Any,
"overlapPct": stats.overlapPct as Any,
```

- [ ] **Step 3: Build and verify**

Run: `./build.sh 2>&1 | tail -5`
Expected: Build complete

- [ ] **Step 4: Commit**

```bash
git add Sources/NovaMLXDistributed/DistributedInferenceRunner.swift Sources/NovaMLXDistributed/ClusterAdminRoutes.swift
git commit -m "feat(distributed): add overlap stats to DistributedInferenceStats"
```

---

### Task 7: Integration Test — Measure and Verify

Deploy to both machines, activate distributed model, and measure tok/s.

**Files:** No code changes — validation only.

- [ ] **Step 1: Build release and deploy**

```bash
./build.sh -c release
./Scripts/package.sh
scp dist/NovaMLX-1.0.8-arm64.tar.gz lucass-mac-mini.local:/tmp/
ssh lucass-mac-mini.local "killall NovaMLX 2>/dev/null; sleep 1; cd /tmp && rm -rf NovaMLX.app && tar -xzf NovaMLX-1.0.8-arm64.tar.gz && rm -rf /Applications/NovaMLX.app && mv NovaMLX.app /Applications/ && codesign --force --deep --sign - /Applications/NovaMLX.app && open /Applications/NovaMLX.app"
killall NovaMLX; sleep 2; open dist/NovaMLX.app
```

Wait for both to start and worker to connect.

- [ ] **Step 2: Check layer split in activation logs**

```bash
curl -s -X POST -H "Authorization: Bearer abcd1234" \
  http://127.0.0.1:6591/admin/api/cluster/activate-model \
  -H "Content-Type: application/json" \
  -d '{"modelId":"mlx-community/Qwen3.6-27B-4bit"}'
```

Check logs for layer assignment. Expected: coord gets ~55 layers, worker gets ~11 (not 53/13).

- [ ] **Step 3: Run non-streaming test (200 tokens)**

```bash
curl -s -H "Authorization: Bearer abcd1234" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:6590/v1/chat/completions \
  -d '{"model":"mlx-community/Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Count from 1 to 30."}],"max_tokens":200}'
```

Check tok/s in logs: `grep "Completed\|tok/s" ~/.nova/novamlx.log | tail -5`

- [ ] **Step 4: Check overlap logs**

```bash
grep "Pipeline" ~/.nova/novamlx.log | tail -5
```

Expected: `await=0.xms` (near zero), `overlap=90%+`. This confirms the pipeline is overlapping.

- [ ] **Step 5: Run streaming test**

```bash
curl -s -N -H "Authorization: Bearer abcd1234" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:6590/v1/chat/completions \
  -d '{"model":"mlx-community/Qwen3.6-27B-4bit","messages":[{"role":"user","content":"Explain quantum computing in 3 sentences."}],"stream":true,"max_tokens":200}'
```

Verify streaming tok/s matches non-streaming.

- [ ] **Step 6: Verify target met**

Target: **18+ tok/s** (must beat single-node ~12-15 tok/s).

If below target, check:
- Layer split (should be ~55/11, not 53/13)
- Worker wait time in logs (should be <5ms)
- coordLayersOnly timing (should be ~33ms)
- Worker compute timing (should be ~29ms)

- [ ] **Step 7: Commit validation**

If performance target is met:

```bash
git add -A
git commit -m "feat(distributed): 2-stage async pipeline delivers 18+ tok/s"
```

If not, investigate timing breakdown and iterate.
