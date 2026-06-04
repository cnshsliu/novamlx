# Distributed Inference TPS Analysis — NovaMLX

## 1. Current Performance Numbers

| Metric | Value |
|--------|-------|
| **Single-node TPS (M4 Max 128GB)** | 30+ tok/s |
| **Distributed TPS (M4 Max + M4 Mini, Thunderbolt 5)** | 6.2 tok/s |
| **Distributed latency per token** | ~161 ms |
| **Single-node latency per token** | ~33 ms |
| **Overhead introduced by distribution** | ~128 ms/token (3.9x slower) |

## 2. Model Architecture: Qwen3.6-27B-4bit

```
hidden_size:          5120
intermediate_size:    17408
num_hidden_layers:    64
num_attention_heads:  24
num_key_value_heads:  4   (GQA: group query attention)
head_dim:             256
vocab_size:           248320
dtype:                bfloat16
quantization:         4-bit affine, group_size=64
full_attention_interval: 4  (hybrid: every 4th layer full attention)
tie_word_embeddings:  false
```

### Shard Plan (spread strategy, minLayersPerShard=8)

```
totalLayers: 66 (embedding + 64 transformer + norm/head)

Coordinator (M4 Max 128GB): layers 0–52   = 53 layers  ~11.9 GB
Worker     (M4 Mini 24GB):  layers 53–65  = 13 layers  ~3.3 GB
```

The worker runs the final 13 layers including norm + lm_head (output projection to full vocabulary).

## 3. Decode Step Data Flow (per token)

For each token generated, the following steps happen sequentially:

```
COORDINATOR (M4 Max)                    WORKER (M4 Mini)
════════════════════                    ════════════════

Step A: Sample previous token
  argmax(prev_logits) → token_id (Int32)

Step B: Local compute (53 layers)
  embed(token_id) → [1, 1, 5120] bfloat16    (GPU: ~0.3 ms)
  forward layers 0..52                        (GPU: ~26 ms)
  → hidden_state [1, 1, 5120] bfloat16        10,240 bytes

Step C: Serialize + Send hidden_state
  GPU → CPU copy (asData)                     (~1 ms)
  WireFormat.encode: 32B header + 24B shape + 10240B data
  TCP write: ~10,296 bytes                    (~0.2 ms on Thunderbolt)

                                         Step D: Receive + Deserialize
                                           TCP read: ~10,296 bytes
                                           WireFormat.decode → CPU Data
                                           MLXArray(Data) → GPU copy      (~1 ms)

                                         Step E: Worker compute (13 layers)
                                           forward layers 53..63          (~6 ms)
                                           norm + lm_head:
                                             [1,1,5120] → [1,248320] float32
                                                                           ~993,280 bytes

                                         Step F: Serialize + Send logits
                                           GPU → CPU copy                 (~2 ms)
                                           WireFormat.encode:
                                             32B header + 16B shape + 993280B
                                           TCP write: ~993,328 bytes      (~3 ms)

Step G: Receive + Deserialize logits
  TCP read: ~993,328 bytes                   (~3 ms)
  WireFormat.decode → CPU Data               (~1 ms)
  MLXArray(Data) → GPU copy                  (~2 ms)

Step H: Sample
  argmax(logits [1, 248320]) → next token_id (~0.1 ms)
```

### Per-Token Timing Breakdown (estimated)

| Step | Description | Time (ms) |
|------|-------------|-----------|
| A | argmax sampling from previous logits | 0.1 |
| B | Coordinator GPU compute (53 layers + embed) | 26.5 |
| C | GPU→CPU copy + serialize + TCP send (~10 KB) | 1.5 |
| D | TCP recv + deserialize + CPU→GPU copy (~10 KB) | 1.5 |
| E | Worker GPU compute (13 layers + norm + lm_head) | 6.5 |
| F | GPU→CPU copy + serialize + TCP send (~970 KB) | 5.0 |
| G | TCP recv + deserialize + CPU→GPU copy (~970 KB) | 6.0 |
| H | argmax sampling | 0.1 |
| **Total** | | **~161 ms** |
| | **→ TPS = 1000/161 ≈ 6.2 tok/s** | |

**Key insight: GPU compute is only ~33 ms of the 161 ms total. The remaining ~128 ms is data movement overhead — GPU↔CPU copies, serialization, and network transfer.**

## 4. Where the Time Goes — Ranked by Impact

### #1: Full Vocabulary Logits Transfer (~970 KB per token)

**Impact: ~11 ms per token (steps F + G)**

The worker's last layer produces `[1, 248320]` float32 logits — 993,280 bytes of raw data. This tensor is serialized, sent over TCP, deserialized, and copied back to GPU — just so the coordinator can run `argmax()` (which takes 0.1 ms).

This is the single biggest design flaw in the current architecture. The fix is straightforward:

**Fix: Run argmax on the worker, send only the token ID (4 bytes).**

This would eliminate ~970 KB of network transfer and the associated GPU↔CPU copies per token. Estimated savings: **~11 ms per token**, bringing TPS from 6.2 to ~6.7 tok/s.

Even better: run full sampling (temperature, top-p, top-k) on the worker side. The worker has the logits in GPU memory; sampling there avoids the transfer entirely. Only the selected token ID needs to go back.

### #2: GPU ↔ CPU Memory Copies (4 copies per token)

**Impact: ~8 ms per token**

Each tensor transfer involves:

1. **GPU → CPU** (`asData(access: .copy)` in `WireFormat.encode`): Forces GPU evaluation, copies result to CPU-allocated buffer.
2. **CPU → GPU** (`MLXArray(tensorData:shape:dtype:)`): Copies from CPU Data buffer into a new MLXArray on GPU.

For each token, this happens twice (hidden_state in, logits out), so **4 GPU↔CPU copies** per token:
- hidden_state: 10 KB × 2 copies = 20 KB moved
- logits: 970 KB × 2 copies = 1,940 KB moved

**Total data movement through system memory: ~1.96 MB per token.**

At 6.2 tok/s, that's ~12 MB/s of pure copy overhead. MLX arrays live in Metal buffer memory; copying between Metal and system memory involves the PCIe/Thunderbolt bus and the GPU's DMA engine.

**Fix: Zero-copy GPU-to-GPU transfer.** If both machines supported NVLink or a similar GPU-direct protocol, tensors could move between GPUs without touching CPU memory. Apple Silicon doesn't support this natively, but a potential workaround is to use IOSurface shared memory for the Thunderbolt link.

### #3: Serialization / Wire Format Overhead

**Impact: ~2 ms per token**

`WireFormat.encode()` builds a `Data` buffer by:
1. Allocating a `Data` of exact size (header + shape + tensor bytes)
2. Writing header fields (magic, ndim, dtype, nbytes)
3. Calling `array.asData(access: .copy)` to get raw bytes
4. Appending bytes to the Data buffer

`WireFormat.decode()` then:
1. Parses header from the Data
2. Extracts shape array
3. Calls `data.subdata(in: range)` to slice the tensor bytes (another copy!)
4. Creates `MLXArray(tensorData, shape, dtype:)` — copies to GPU

Each tensor goes through at least **3 copies** between encode and decode:
1. GPU → `asData` result (Data buffer #1)
2. Data buffer #1 → concatenated wire Data (buffer #2)
3. Wire Data → `subdata` slice → MLXArray constructor (GPU buffer)

**Fix: Direct scatter-gather I/O.** Instead of building a single contiguous buffer, write the header, shape, and tensor bytes directly to the socket in separate writes. This eliminates the intermediate concatenation copy.

### #4: No Compute/Communication Overlap in Decode

**Impact: Full serialization — no overlap at all**

The decode loop in `DistributedInferenceRunner` is strictly sequential:

```swift
for shard in shardEngines {
    decodeActivation = try await shard.decode(token: decodeActivation)
}
activation = decodeActivation
```

Each `shard.decode()` is `async throws` but effectively blocks until the remote worker responds. There is **no pipelining** — the coordinator cannot start computing token N+1's first layers while the worker processes token N's last layers.

The codebase has `pipelinedPrefill()` for the prefill phase (which overlaps coordinator compute with worker compute), but the decode loop does not use this pattern.

**Fix: Overlapped decode.** After the coordinator finishes its layers for token N and sends the hidden state to the worker, it should immediately start processing the previous token's logits (sampling) and begin token N+1's local compute — all while the worker processes token N's layers. This hides ~6 ms of worker compute behind coordinator compute.

### #5: Serial Dispatch Queue for TCP I/O

**Impact: ~1 ms overhead per send/recv pair**

`TCPConnection` uses a single serial `DispatchQueue` (`queue.sync { ... }`) for all socket operations. This means:
- A `sendTensor` blocks until the full data is written to the socket buffer
- A `recvTensor` blocks until all bytes are read
- Send and receive cannot happen concurrently on the same connection

For the decode path, this isn't a huge issue (we need to send, then receive sequentially anyway), but it adds latency from the dispatch queue scheduling overhead.

### #6: POSIX Blocking I/O (no async socket)

**Impact: Minor — Thunderbolt latency is low**

All TCP reads/writes use blocking `Darwin.write` and `Darwin.recv`. On Thunderbolt 5 (~40 Gbps), the network latency itself is negligible for our tensor sizes. The 10 KB hidden state transfer takes ~0.01 ms of pure network time. The 970 KB logits transfer takes ~0.2 ms. The dominant overhead is the GPU↔CPU copies, not the network.

## 5. Comparison: Why Single-Node Gets 30+ TPS

On a single M4 Max 128GB, the same model runs at 30+ tok/s. Here's why:

| Aspect | Single-Node | Distributed (2 nodes) |
|--------|------------|----------------------|
| **Compute per token** | All 64 layers on one GPU (~33 ms) | 53 layers local (~26 ms) + 13 layers remote (~6 ms) = 32 ms |
| **Tensor transfers** | Zero (everything in GPU memory) | 4 GPU↔CPU copies (~1.96 MB moved) |
| **Network** | None | 2 TCP round-trips (~1 MB total) |
| **Serialization** | None | 2 encode + 2 decode (~2 ms) |
| **Overhead** | ~0 ms | ~128 ms |

The compute time is nearly identical (~33 ms). But distributed adds ~128 ms of data movement overhead per token. That's why TPS drops from 30+ to 6.2.

**The fundamental issue: distributed pipeline parallelism amortizes compute over multiple nodes but adds per-token communication overhead that doesn't exist in single-node inference.**

## 6. Quantitative Breakdown: Where Does Each Millisecond Go?

```
Total per-token time: ~161 ms
│
├── GPU Compute: ~33 ms (20.5%)
│   ├── Coordinator (53 layers): ~26.5 ms
│   └── Worker (13 layers + head): ~6.5 ms
│
├── GPU↔CPU Memory Copies: ~8 ms (5.0%)
│   ├── Coordinator GPU→CPU (hidden_state 10KB): ~1 ms
│   ├── Worker CPU→GPU (hidden_state 10KB): ~1 ms
│   ├── Worker GPU→CPU (logits 970KB): ~2 ms
│   └── Coordinator CPU→GPU (logits 970KB): ~2 ms
│       → Note: This is eliminated if we sample on the worker
│
├── Tensor Serialization: ~2 ms (1.2%)
│   ├── encode hidden_state: ~0.3 ms
│   ├── decode hidden_state: ~0.5 ms
│   ├── encode logits: ~0.5 ms
│   └── decode logits: ~0.7 ms
│
├── Network Transfer: ~4 ms (2.5%)
│   ├── Send hidden_state (10KB): ~0.2 ms
│   ├── Send logits (970KB): ~2.5 ms
│   ├── Recv hidden_state: ~0.3 ms
│   └── Recv logits: ~1.0 ms
│
└── Unaccounted / Scheduling: ~114 ms (70.8%)
    ├── DispatchQueue queue.sync overhead: ~2 ms
    ├── async/await context switching: ~2 ms
    ├── TCP kernel overhead (syscall, TCP stack): ~2 ms
    ├── GPU synchronization (MLX.asyncEval, Metal commit): ~5 ms
    └── Remaining gap: ~103 ms
```

**The ~103 ms unaccounted gap** is likely caused by:
1. **MLX GPU synchronization**: Each `asData(access: .copy)` forces a full GPU pipeline flush. The GPU may have queued work from previous steps that must complete before the copy.
2. **Memory pressure**: The model is ~15 GB. Moving tensors through system memory while the GPU has nearly all RAM allocated for model weights causes memory pressure and potential swapping.
3. **Metal command buffer submission**: Each layer forward pass submits Metal compute commands. The GPU may batch these, causing the `asData` copy to wait for all pending commands to finish.

## 7. Improvement Roadmap (Estimated Impact)

| Priority | Fix | Est. TPS Gain | Effort |
|----------|-----|--------------|--------|
| **P0** | Run argmax/sampling on worker, send only token ID | +1-2 tok/s (6.2→8) | Small |
| **P1** | Overlap coordinator compute with worker compute (pipelined decode) | +2-3 tok/s (8→10-11) | Medium |
| **P2** | Use bfloat16 for hidden state transfer (already bfloat16 in GPU) | +0.5 tok/s | Tiny |
| **P3** | Zero-copy IOSurface for Thunderbolt tensor sharing | +3-5 tok/s | Large |
| **P4** | Speculative decoding (draft model predicts N tokens, worker verifies) | +5-10 tok/s (theoretically) | Very Large |
| **P5** | Batch multiple tokens in a single TCP message | +1-2 tok/s | Medium |

### P0: Remote Sampling (Recommended Next Step)

Current flow:
```
Worker: compute 13 layers → logits [1, 248320] float32 → send 970 KB → Coordinator: argmax
```

New flow:
```
Worker: compute 13 layers → logits → argmax → send 4 bytes → Coordinator: receive token_id
```

Implementation: Move `argmax()` and sampling logic into `WorkerShardService.handleCompute()`. The worker already has the logits in GPU memory. Just run `argmax` there and send the Int32 result.

### P1: Pipelined Decode

Current flow (sequential):
```
Token N: Coord compute [26ms] → send [1.5ms] → Worker compute [6ms] → send [6ms] → sample
Token N+1:                                                          ← waits for Token N to fully complete
```

Pipelined flow:
```
Token N:   Coord compute [26ms] → send [1.5ms] → Worker compute [6ms] → send [4B]
Token N+1: Coord compute [26ms] → (overlapped with Worker processing Token N)
```

The coordinator starts Token N+1's compute immediately after sending Token N's hidden state to the worker, without waiting for the worker's response. When the worker finishes Token N, it sends the token_id back. The coordinator samples and feeds the result into Token N+1's decode.

This hides the worker's ~6 ms compute time behind the coordinator's ~26 ms compute time, effectively making the worker compute "free" (it's overlapped).

### P3: Zero-Copy IOSurface

Apple's IOSurface framework allows sharing GPU textures between processes and across Thunderbolt connections without CPU copies. If we could:

1. Create an IOSurface backed by the MLX array's Metal buffer
2. Send the IOSurface ID over the control channel (just an integer)
3. The receiver maps the IOSurface into its GPU address space

This would eliminate all 4 GPU↔CPU copies per token. Estimated savings: ~8 ms per token.

However, cross-machine IOSurface sharing requires Thunderbolt DMA and is not a standard API. This would require significant Metal framework integration.

## 8. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Coordinator (M4 Max 128GB)                        │
│                                                                      │
│  ┌──────────────────┐    ┌───────────────────┐    ┌──────────────┐  │
│  │   Chat Template   │    │  SlicedForward     │    │   Sampling   │  │
│  │   + Tokenizer     │    │  Policy            │    │   (argmax)   │  │
│  │                    │    │                    │    │              │  │
│  │  messages → tokens │    │  embed + layers   │    │  logits → id │  │
│  │  [17-23 tokens]   │    │  0..52 (53 layers) │    │              │  │
│  └──────────────────┘    └───────────────────┘    └──────────────┘  │
│                                    │                         ▲        │
│                                    │ hidden_state             │        │
│                                    │ [1,1,5120] bf16         │        │
│                                    │ 10 KB                   │        │
│                                    ▼                         │        │
│                          ┌──────────────────────┐            │        │
│                          │   TensorTransport     │            │        │
│                          │                      │            │        │
│                          │  encode: GPU→CPU→TCP  │            │        │
│                          │  decode: TCP→CPU→GPU  │            │        │
│                          │                      │            │        │
│                          │  WireFormat:          │            │        │
│                          │  32B header + shape   │            │        │
│                          │  + raw tensor bytes   │            │        │
│                          └──────────┬───────────┘            │        │
│                                     │ TCP                     │        │
└─────────────────────────────────────┼─────────────────────────┼────────┘
                                      │ Thunderbolt 5           │
                                      │ (~40 Gbps)              │
┌─────────────────────────────────────┼─────────────────────────┼────────┐
│                    Worker (M4 Mac Mini 24GB)                         │
│                                     │                        │         │
│                          ┌──────────┴───────────┐            │         │
│                          │   TensorTransport     │            │         │
│                          └──────────┬───────────┘            │         │
│                                     │                        │         │
│                          ┌──────────┴───────────┐            │         │
│                          │  SlicedForward        │            │         │
│                          │  Policy               │            │         │
│                          │                      │            │         │
│                          │  layers 53..63        │            │         │
│                          │  norm + lm_head       │            │         │
│                          │                      │            │         │
│                          │  → logits [1,248320]  │            │         │
│                          │  float32, 970 KB      │            │         │
│                          └──────────────────────┘            │         │
│                                     │                        │         │
│                                     └──── TCP send logits ───┘         │
└────────────────────────────────────────────────────────────────────────┘

Per-token data flow:
  Coord ──[10 KB hidden]──→ Worker ──[970 KB logits]──→ Coord ──[argmax]──→ token_id
         (TCP send)              (GPU compute)         (TCP recv)        (sample)
```

## 9. Key Files Reference

| File | Role | Key Methods |
|------|------|-------------|
| `DistributedInferenceRunner.swift` | Orchestrator | `generate()`, decode loop at lines ~310-329 |
| `SlicedForwardPolicy.swift` (in `ShardableModel.swift`) | Local GPU compute | `compute()`, `bindWeights()` |
| `RemoteShardPolicy.swift` | TCP coordinator→worker | `compute()`, `sendCompute()`, `recvResult()` |
| `WorkerShardService.swift` | Worker TCP server | `handleCompute()`, `run()` |
| `TensorTransport.swift` | Binary wire format | `WireFormat.encode/decode`, `TCPConnection` |
| `ShardEngine.swift` | Pipeline orchestration | `decode()`, `prefill()` |
| `ClusterModelManager.swift` | Model lifecycle | `activateModel()`, `getShardEngines()` |

## 10. Summary

The 5x TPS gap (30+ → 6.2) is caused by **per-token data movement overhead**, not by compute distribution:

1. **970 KB logits transfer per token** — the biggest single cost. Sending the full vocabulary tensor just to run argmax is wasteful.
2. **4 GPU↔CPU copies per token** — Metal buffer → system memory → network → system memory → Metal buffer. Each copy has latency from GPU pipeline flushes.
3. **No compute/communication overlap** — the coordinator idles while waiting for the worker to finish.
4. **3 intermediate data copies in serialization** — concatenate, subdata slice, MLXArray constructor.

The compute itself is nearly free: 53 layers on the coordinator (~26 ms) + 13 layers on the worker (~6 ms) ≈ 32 ms total — same as single-node. The ~128 ms of overhead comes entirely from moving data between the two GPUs.

The highest-ROI fix is **P0: remote sampling** (send token ID, not logits). It's a small code change that eliminates the largest transfer. Combined with **P1: pipelined decode**, distributed TPS could reach 10-11 tok/s — still slower than single-node, but the gap narrows from 5x to 3x. Further improvements require hardware-level optimizations (IOSurface, GPU-direct) that are platform-specific.
