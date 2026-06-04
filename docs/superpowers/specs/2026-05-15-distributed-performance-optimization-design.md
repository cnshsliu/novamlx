# Distributed Inference Performance Optimization

## Problem

Distributed inference over Thunderbolt (M4 Max + M4 Mac Mini) runs at **7.1-7.3 tok/s** with Qwen3.6-27B-4bit — slower than single-node M4 Max (~12-15 tok/s). The two-node setup adds compute capacity but serialization overhead dominates.

Root cause: serial pipeline latency. Coord computes (31.8ms) → TCP send (4ms) → Worker computes (33.7ms) → TCP return (4ms) = ~73ms per token.

## Target

Faster than single-node (~12-15 tok/s). Expected outcome: **18+ tok/s**.

## Design: Continuous Async Pipeline

Four interlocking optimizations, each building on the previous.

### 1. Shard Rebalancing

**Current**: Coord runs layers 0-52 (31.8ms). Worker runs layers 53-65 + norm + lm_head (33.7ms).

**Proposed**: Coord runs layers 0-52 + norm + lm_head + argmax (~34ms). Worker runs layers 53-65 only (~17ms).

The lm_head is a large matrix multiply (~0.5B params for 27B model) that takes ~16ms. Moving it to the coord:
- Worker becomes a pure transformer-layer node — 2x faster (33.7ms → ~17ms)
- Coord only adds ~2ms (head compute) since it already runs embedding + 53 layers
- Worker always returns a hidden state tensor (same size as what it receives)
- Eliminates the separate `computeAndSample` message type — one uniform protocol

**Files**: `ClusterModelManager.swift`, `SlicedForwardPolicy.swift`, `WorkerShardService.swift`

### 2. bfloat16 Transport

**Current**: Hidden states sent as float32. Shape [1, 3584] = 14,336 bytes per direction.

**Proposed**: Send bfloat16 = 7,168 bytes. Apple Silicon supports native bfloat16 compute, so the worker can process bfloat16 activations directly without dequantization.

Protocol:
- Coord → Worker: quantize activation to bfloat16, send via TCP
- Worker: receives bfloat16, runs 13 transformer layers in bfloat16 (0.1% precision loss, negligible)
- Worker → Coord: sends output as bfloat16
- Coord: dequantizes to float32 for lm_head (head needs float32 for argmax accuracy)

Savings: ~4ms per direction on Thunderbolt TCP.

**Files**: `RemoteShardPolicy.swift`, `WorkerShardService.swift`, `TensorTransport.swift`

### 3. Continuous Async Pipeline (Core)

Replace the serial decode loop with a double-buffered async pipeline where coord and worker compute simultaneously.

**Current serial flow** (per token):
```
coord compute → TCP → worker compute → TCP → coord head+argmax
[  31.8ms   ] [4ms ] [   33.7ms     ] [4ms ] [   2ms      ]
Total: 75.5ms = 13.2 tok/s ceiling, actual 7.2 tok/s (spec misses)
```

**Proposed continuous flow**:
```
Time →   0    10   20   30   40   50   60   70   80

Coord:   [compute N, send N]  [draft N+1]  [recv N, head→token N]  [compute N+1, send N+1]  [draft N+2]  ...
Worker:         [compute N]            [compute N+1]            [compute N+2]
```

Mechanism:
1. Coord computes token N's activation, sends bfloat16 to worker (~4ms TCP)
2. Coord immediately starts drafting token N+1 (uses extraPredictionLayers)
3. Worker computes token N's 13 layers in parallel (~17ms)
4. Worker returns bfloat16 hidden state (~4ms TCP)
5. Coord runs head+argmax on worker's result → actual token N
6. If draft predicted == actual → use precomputed activation (HIT, zero idle time)
7. If miss → recompute with actual token (MISS, ~20ms penalty)

**Double-buffer state**:
- Buffer A: coord writes, worker reads (activation for token N)
- Buffer B: worker writes, coord reads (result for token N)
- Swap after each token

**Performance math**:
- Draft HIT: effective token time = max(34ms coord, 21ms worker+TCP) ≈ 34ms → ~29 tok/s
- Draft MISS: effective token time ≈ 34ms + 20ms penalty ≈ 54ms → ~18 tok/s
- At 20% hit rate: 0.2×29 + 0.8×18 ≈ **20 tok/s** (conservative)

### 4. Decode Loop Implementation

**Prefill**: Unchanged — existing pipelined prefill already overlaps coord/worker chunks.

**Post-prefill setup**:
```
lastHidden = prefill output from worker (bfloat16, [1, 3584])
firstToken = coord runs head(lastHidden) → argmax → token 0
```

**Decode loop** (per token N ≥ 1):
```
Step 1: Send activation (bfloat16) to worker                    [fire-and-forget TCP]
Step 2: Start draft prediction for token N+1                    [~5ms, overlaps with worker]
Step 3: Await worker result (bfloat16 hidden state)
Step 4: Run head+argmax on worker result → actualToken           [~2ms]
Step 5: If predictedToken == actualToken → use precomputed state (HIT)
        Else → recompute with actualToken (MISS)
Step 6: Emit token, update activation, loop
```

**Draft accuracy improvement**: With the faster worker (17ms vs 33ms), coord has more time to run a better draft:
- Increase extraPredictionLayers from 5 to 8 (more context for prediction)
- Consider K=3 draft (predict 3 tokens ahead) for batch verification

## Files to Modify

| File | Change |
|------|--------|
| `NovaMLXDistributed/ClusterModelManager.swift` | Adjust shard plan: coord gets head, worker doesn't |
| `NovaMLXDistributed/SlicedForwardPolicy.swift` | Add `computeHeadOnly(hidden)` method for coord to run lm_head |
| `NovaMLXDistributed/RemoteShardPolicy.swift` | bfloat16 quantize before send, dequantize after recv; unified compute protocol |
| `NovaMLXDistributed/WorkerShardService.swift` | Simplify: always return hidden state, never run head |
| `NovaMLXDistributed/DistributedInferenceRunner.swift` | Rewrite decode loop with double-buffer async pipeline |
| `NovaMLXDistributed/TensorTransport.swift` | Add bfloat16 support to sendTensor/recvTensor |

## Verification

1. Single-node baseline: measure Qwen3.6-27B-4bit tok/s on M4 Max alone
2. Distributed: activate model, run streaming inference, measure tok/s
3. Compare: distributed should exceed single-node by 20%+ (target: 18+ tok/s)
4. Draft accuracy: log hit/miss ratio (target: >30%)
5. Latency breakdown: log coord compute, TCP, worker compute, head times per token
6. Stress test: 3+ consecutive requests to verify pipeline stability
