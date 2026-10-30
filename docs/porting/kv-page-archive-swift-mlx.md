# KV page archive for long agent sessions (Swift + MLX)

Status: design only. This document does not change inference, Metal, or model code.

Product direction: long-running agent sessions (the Agent Infer direction: OpenClaw, Hermes, OpenCode, Claude Code, and the same `session_id` clients NovaMLX already serves). This note does not rename NovaMLX.

Companion field name stays `companionReady`.

## Decision

Borrow the KVMem idea: keep history as paged KV, attend a bounded working set, and later retrieve relevant middle blocks into that set.

Leave NInfer, the `.ninfer` format, Windows CUDA launchers, and MiniCPM-OCR where they are. Speed and MTP stay a separate track. This repo already registers Qwen MTP (`QwenMtpRegistration` in `Sources/NovaMLXEngine/QwenMtpDecode.swift`, `DraftModelRegistry`, `nativeMtpEnabled`). Virtualizing KV does not make decode faster and must not be described as MTP.

## Product sentence

The logical workspace is addressable history. Each model step attends a bounded working set: attention sinks, a recent tail, a generation reserve, and (from phase P2) a retrieved subset of middle pages.

That working set is the attention the model actually runs. Addressable history can be larger than the set. It is a paged archive plus a selector. It is a different mechanism from dense attention over every stored token, and a different mechanism from `contextScalingTarget`, which only rescales **reported** usage so agent auto-compact fires earlier (`ServerConfig.scaleTokenCount` in `Sources/NovaMLXCore/Types.swift`; README section “Agent Context Scaling”).

Wording that matches the implementation:

- “addressable workspace”
- “paged KV archive”
- “bounded active attention window”
- “retrieved blocks” once P2 selection exists

Wording that over-claims, and should stay out of UI, logs, and release notes:

- “million-token context” or “full history in context” when the live cache is a window
- “unlimited context”
- paper or NInfer throughput and task-success figures as if they were Apple Silicon results

KVMem’s own measurements stay citations, not acceptance targets. The paper (arXiv:2609.04852) reports a DeepSWE lift from 43.8% with compaction-only context management to 48.4% on Qwen3.8-27B, workspaces up to 1M tokens against a 256K native window, and about 50 tokens/s for Qwen3.6/3.8-27B NVFP4 with MTP on a 24 GB RTX 5090 laptop. Those runs are the CUDA QW3 stack. A NovaMLX acceptance test has to be measured here.

## What we borrow

Sources, read for ideas only:

- Paper: [KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU](https://arxiv.org/abs/2609.04852) (arXiv:2609.04852v1, 4 Sep 2026). Paper code pointer is `kvmem-qw3`.
- Multi-backend tree: [github.com/kvmem/kvmem](https://github.com/kvmem/kvmem) (README budgets, LRFU, sink/recent).
- llama.cpp adapter notes: [kvmem-llama.cpp `docs/architecture.md`](https://github.com/kvmem/kvmem-llama.cpp/blob/master/docs/architecture.md).

Mechanics worth mapping onto Swift:

| Idea | Meaning here |
|------|----------------|
| Repository vs execution view | History pages live in an archive. The model forward sees one bounded execution view. |
| Page | A fixed token span (KVMem recipes use 128; our prefix-cache block size of 64 is a different store and stays 64). Each page records original token positions. |
| Budget `B` | Tokens of **selected history** in the execution view, including sink, recent, and any mandatory region (for example an image span). |
| Generation reserve `R` | Tokens of **this turn’s** new KV. Selected history stays pinned while decoding. New tokens consume `R`. |
| Sink + recent | Always resident. They are not retrieval candidates. They spend `B`. |
| Step-level recall | Choose the working set once per agent step, after the new query is known and before decode. Keep that set fixed for the rest of the turn. |
| Mean-K index | A compact, position-independent summary of each page’s K, scored in attention space against the query. The paper averages layer/head softmax scores. The llama.cpp notes capture mean-K in F32, pre-RoPE, at first write. |
| LRFU | Optional score memory across turns. `lambda` decays history. `top-frac` is the hit fraction allowed to accumulate a positive update (KVMem’s own definition). Window size stays budget `B`. Sink and recent are reserved inside `B`. Default off until a fixture shows a gain. |
| Host then SSD | Completed pages leave the execution view. A later tier may spill cold pages to SSD. SSD stays off until the RAM archive is honest. |
| Accepted tokens only | MTP drafts, speculative rejects, and probe prefills do not become archive pages. The llama.cpp notes say the same. |

Two upstream position policies exist. They are different designs, and this stack has to pick one on purpose.

1. **Paper (QW3).** The execution view is a compact window. Raw K (pre-RoPE) is the source of truth outside the active cache. Entering the window applies RoPE at the block’s compact position. A block that only shifts position can delta-re-RoPE. V is position-independent. Page-table aliasing avoids a dense gather in their CUDA executor.
2. **llama.cpp product path.** Slot `pos` is the original monotonic token position. Restore is a packed memcpy of GPU-format K/V at that original position. That path does no raw-K store and no re-RoPE, and it does not modify Flash Attention. Dropped blocks stay dropped.

Recommendation for MLX, phases P1–P2: assemble a **dense** `[KVCache]` the existing model forward already accepts. Do not patch `vendors/mlx-swift` and do not add a Metal page table in these phases.

RoPE makes a naive truncate unsafe. K in the live cache was written at the token’s original position. Deleting the middle and concatenating sink with recent leaves that recent K at the wrong index. P1 seals pages and repairs the recent tail, and still does not retrieve:

- The sink is a prefix at positions `0 ..< sink`. Its K stays as stored.
- A page is sealed by unrotating K with the position it currently has, then storing that pre-RoPE K and V. The archive copy is the source of truth.
- The recent tail is rewritten at compact positions `sink ..< sink+recent` when it is installed in the active window. V is copied. Tokens generated after that are written by the model at the new tail, so they do not need a second repair.
- Middle pages sit only in the archive. P1 does not score them and does not attend them. The active window is sink + recent. It is a truncated window, and the logs should say that.

P2 uses the same pre-RoPE payload. Selected middle pages are re-RoPE’d into the compact gap between sink and recent, in original-position order. That is ordinary MLX math on K. It stays outside the attention kernel.

The llama.cpp “keep original positions, leave holes” policy needs a cache and mask that understand gaps. MLX `KVCacheSimple` is a dense sequence axis. Padding the holes spends the memory the archive was meant to free. Treat that policy as a later alternative if compact re-RoPE fails a quality fixture, not as the P2 default.

Generation that fills `R` stops or returns a clear budget error. It does not evict a pinned retrieved page to keep going. That matches the llama.cpp failure (`no free GPU slot` once `gen_reserve` is full). NInfer can archive mid-generation and continue; that behavior is deliberately absent here. A ring buffer inside `R` (spill this turn’s oldest completed generation block, never the in-progress block, never a pinned page) is the llama.cpp follow-up note. It can be a later design. It is not P1 or P2.

Numbers such as sink 12288 / recent 1024, `B` 36864 / `R` 16384, and a 12 GiB host quota are NInfer Windows recipes. They are not Mac defaults. Archive budgets are configuration. A conservative example may ship as a commented profile. The first measured profile is allowed to replace it. No phase may present an untuned example as a quality result.

Query span for scoring starts as the latest user or tool message, capped (KVMem bounds the query at 512 tokens). Sink and recent are masked out of the candidate denominator.

## Mac unified memory

KVMem’s discrete-GPU story is three address spaces: a VRAM page pool, host RAM, and NVMe, with PCIe copies between them. Apple Silicon running MLX has one unified memory pool. “Resident” and “archive” are software roles, not devices.

| Discrete GPU + host | This stack |
|---------------------|------------|
| GPU pool of `B + R` slots | Token budgets `B` and `R` on the **active** cache that the forward pass reads. |
| Host RAM quota `H` | Archive pages that are MLX arrays (or compressed blobs) **outside** that active cache. They still occupy unified memory and RSS. |
| NVMe quota `D` | Optional later spill through a durable store. Default off, same as KVMem SSD. |
| Pinned H2D / D2H | `MLXArray` slice and copy. No PCIe. A gather into a dense cache still peaks memory for the copy. |
| VRAM OOM | Process RSS, the memory enforcer, and Metal’s recommended working set. `WiredReservationPlanner` refuses a reservation that would pass `GPU.maxRecommendedWorkingSetBytes()`. `MemoryPressureHandler` evicts unpinned models and then calls `MLXMemoryGate.clearCache()`. |
| Separate GPU heap for KV | `MemoryBudgetTracker` already charges `effectiveBytesPerToken * (prompt + maxTokens)` on the fused path. An archive has to be in that accounting or it will look free while RSS climbs. |

Consequences:

- Moving a page “to host” does not free unified memory. The memory win is dropping that page from the **active** attention tensors (and, later, spilling the archive copy out of RAM).
- A large RAM archive can trip `MemoryPressureHandler` and unload the chat model. The archive quota sits under the same enforcer as weights.
- `MLXMemoryGate.clearCache()` calls `MLX.Memory.clearCache()`. That drops the MLX temporary-buffer cache. It does not drop a conversation, a prefix block, or an archive page. Callers today include `MemoryPressureHandler` after model eviction, `TranscriptionService`, and image pipelines. Archive code must not use this function as a KV evictor, and must not run it while `MLXEngine`’s lock is held (the gate’s own comment documents the deadlock).
- Quantized live KV (request `kvBits` / `kvGroupSize`, and TurboQuant’s 4-bit affine cache) stays the in-window format the sampler already ignores. A sealed archive page needs K that can be re-RoPE’d, so the seal dequantizes to pre-RoPE K (or stores the quantized values with their scale and the original position, and dequants at install). TurboQuant is a storage format under the sampler. It is not the archive index.
- Weight tiering (TIE: wired / LRU / SSD weights) is a weight path. KV pages do not join the weight tier manager.

## What already exists (file map)

Symbols below are from the tree this design was written against. P0 reconfirms them before editing.

### Two KV mechanisms, plus a third that only trims allocator memory

1. **Session lineage** (`session_id`). `MLXEngine.generate` (`Sources/NovaMLXEngine/MLXEngine.swift`, the `request.sessionId` branch) and `MLXEngine.stream` send the request to `generateWithSession` / `streamWithSession`. Those build an mlx-swift-lm `ChatSession` via `ChatSessionManager.getOrCreate` and stream **only the latest user message** (`SessionBox.streamDetails(to:)`). The session object holds the growing cache. Logs already print `promptTokenCount`, `cachedPromptTokenCount`, and `cacheEfficiency` from the generation `.info` event.
2. **Exact-prefix blocks** (`NovaMLXPrefixCache`). Used when the client resends a full token sequence and there is no `session_id` dispatch into `ChatSession`. `FusedBatchScheduler` calls `PrefixCacheManager.fetchPrefix` before prefill and `storeCache` after the sequence finishes. A hit restores a contiguous prefix and prefills the suffix. Identity is a block-hash chain (`BlockHasher`), not a session position. Flint never takes this manager (`flintFamily ? nil` at the fetch site). `storeCache` is also skipped for Flint, and `CacheBlockExtractor.admitsLongTermKV` refuses any cache that contains `RotatingKVCache` or `BoundedSlidingKVCache`.
3. **Allocator `clearCache`.** `MLXMemoryGate.clearCache`. See above. Unrelated to (1) and (2).

The prefix store answers “have we already prefilled this exact token prefix?”. It does not select a non-contiguous middle of an agent trace. The session store answers “continue this `ChatSession`.” It grows one dense cache until memory or the context window stops it. Neither one is a query-dependent working set.

`ChatSession` itself lives in the local package `mlx-swift-lm` (`MLXLMCommon`). The in-tree wrapper is `Sources/NovaMLXEngine/ChatSessionManager.swift`. P1 wraps that manager from the outside. It does not fork `ChatSession` inside `mlx-swift-lm`.

### Hook table

| Concern | Where |
|---------|--------|
| `session_id` on the request | `InferenceRequest.sessionId` in `Sources/NovaMLXCore/Types.swift`. OpenAI JSON key `session_id` in `Sources/NovaMLXAPI/OpenAITypes.swift`. Threaded through `Sources/NovaMLXAPI/APIServer+InferenceHandlers.swift` and the chat handler in `APIServer.swift`. Worker copy in `Sources/NovaMLXCore/WorkerProtocol.swift`. |
| Session object, fork, TTL, disk | `ChatSessionManager` (`Sources/NovaMLXEngine/ChatSessionManager.swift`). Defaults: 64 sessions, TTL 1800 s. Disk under `NovaMLXPaths.sessionsDir` (`~/.nova/sessions/{percent-encoded-id}.safetensors` plus `.json`). `fork` writes the source cache and loads a full copy into the new id. `evictOldest` persists only when `persistToDisk` is true. |
| Hybrid / recurrent sessions | `ChatSessionManager.persistToDisk(hasLinearAttention:)` is false when `container.config.hasLinearAttention` is true. Those sessions are memory-only. Save and disk-fork throw `NovaMLXError.cacheError`. Recurrent state is not a KV page. The paper leaves recurrent state on the original path; we do the same. |
| Session generate / stream | `MLXEngine.generateWithSession`, `MLXEngine.streamWithSession`. Tool calls on this path become `ToolCallResult` and `finishReason = .toolCalls`. The engine does not run the tool. The **next** request is the continuation. |
| Admin sessions | Admin router (`APIServer.swift`, admin port, default 6591): `GET /admin/sessions`, `DELETE /admin/sessions/{id}`, `DELETE /admin/sessions`, `POST /admin/sessions/{id}/save`, `POST /admin/sessions/fork`. |
| Prefix manager | `Sources/NovaMLXPrefixCache/PrefixCacheManager.swift` (`fetchPrefix`, `storeCache`, `PrefixCacheStats`). Pool: `PagedBlockPool.swift`. Hot RAM: `HotPrefixTier.swift`. SSD: `SSDCacheStore.swift`, `PrefixCacheDurableStore.swift`. Types: `PrefixCacheTypes.swift` (`PrefixCacheConfig.blockSize` default 64, hot default 0 in the struct, engine default 256 MiB when unset, TTL 24 h, SSD cap 5 GiB). |
| Prefix wiring | `Sources/NovaMLXEngine/FusedBatchScheduler.swift` (fetch before manual prefill; `storeCache` at sequence completion). Creation: `MLXEngine.getOrCreatePrefixCacheManager`. Kill switch: `ServerConfig.prefixCacheEnabled`. Roots and budgets: `Sources/NovaMLXCore/PrefixCacheLocation.swift`, `NovaMLXPaths.prefixCacheBaseDir` (`~/.nova/prefix_cache`). |
| Prefix admin | Live routes: `GET /admin/cache/{modelId}/stats`, `DELETE /admin/cache/{modelId}`. The feature page `docs/features/kv-cache.md` says `/admin/models/{id}/cache`. The live group is `admin/cache`. Archive work adds a **new** route group. It does not rename this one and does not change the stats JSON keys. |
| Sliding rings that must stay out | `CacheBlockExtractor.usesRotatingKV` / `admitsLongTermKV` (`Sources/NovaMLXPrefixCache/CacheBlockExtractor.swift`). Flint: `FlintSlidingKVCache` and `BoundedSlidingKVCache` in `Sources/NovaMLXEngine/Models/FlintSWAReplay.swift`; in-memory exact prefix `FlintPrefixEngine.select` in `Sources/NovaMLXEngine/Models/FlintPrefix.swift` (the file header states this is `select_prefix`, not `NovaMLXPrefixCache`). Unlimited OCR: `UnlimitedOCRModel.makeCache` and `slidingWindow` in `Sources/NovaMLXEngine/Models/UnlimitedOCRModel.swift` and `UnlimitedOCRConfiguration.swift`. |
| Memory | `Sources/NovaMLXEngine/MLXMemoryGate.swift`, `Sources/NovaMLXEngine/WiredReservationPlanner.swift`, `Sources/NovaMLXInference/MemoryPressureHandler.swift`. Feature overview: `docs/features/memory-management.md`. |
| Reported agent context | `ServerConfig.contextScalingTarget` and `scaleTokenCount`. Detection is header / User-Agent based (README). Catalog pointer: `docs/features/agents.md`. |
| Tool surface | `docs/features/tool-calling.md`. OpenAI, Anthropic, and Responses all reach the same engine. A continuation after `tool_calls` is another HTTP request, with the same `session_id` or with the full transcript resent. |
| MTP (do not fold in) | `QwenMtpRegistration.ensure` registers `Qwen35TextMTPRegistration` and `Qwen35VLMMTPRegistration`. `qwenMtpCompanionReady` requires an on-disk MTP config (`NovaMLXEngine.isMtpDraftConfig` in `Sources/NovaMLXEngine/SpeculativeDecoding.swift`: `model_type` `qwen3_5_mtp` or suffix `_mtp`, or an architecture containing `mtp`) **and** safetensors weights. `InferenceService` skips MTP auto-inject when temperature is above 0 (Qwen MTP is greedy-only) and refuses to serve an MTP head as chat. |
| `companionReady` | Property on `AdminModelStatus` in `Sources/NovaMLXAPI/OpenAITypes.swift`. Populated from `InferenceService.isCompanionReady` (DSpark, DFlash, Qwen MTP on disk). `/health` reports `companionReadyCount`. `loaded` stays resident-only. Archive state must not flip either field. A backbone with an archive is `loaded` only when its weights are resident, as today. |
| Regression floors | `Scripts/regression-suite.yaml`, `Scripts/regression-perf-gate.py`, `docs/features/regression-perf-gate.md`. Core suite: `RecursiveSI/flint-int4` (`min_seed_decode_tps` 100), `mlx-community/Qwen3-8B-4bit` (`min_seed_decode_tps` 50), `baidu/Unlimited-OCR-2` (non-empty text, contains a digit, `gate_decode_tps: false`). Default tolerance is 8% below baseline (`--strict` is 5%). |

Worker mode runs the engine in `NovaMLXWorker`. Archive arrays have to live in the process that owns the `ChatSession`. The menu-bar host does not hold a second copy.

`FusedBatchKVCache` (continuous batching) is a batch-slot layout. It is not a workspace archive. P0–P2 leave it alone. The first consumer is the `session_id` path.

Distributed inference is out of this design. An archive is single-process.

## Proposed module boundaries

New library target `NovaMLXKVArchive` (sibling of `NovaMLXPrefixCache`, not a mode flag inside it). Prefix identity is a content hash shared across requests. Archive identity is `(sessionId, originalPosition)`. Mixing them in one pool will make Flint’s “do not use the SSD prefix store” rule and the hash chain unreadable.

Allowed dependencies: `NovaMLXCore`, `NovaMLXUtils`, `MLX`, `MLXLMCommon` (for `KVCache` only). The module must not import `NovaMLXEngine`, SwiftUI, or the API server. The engine calls the archive; the archive does not call the engine. Tests live in `Tests/NovaMLXKVArchiveTests` and use synthetic `KVCache` values, not downloaded weights. `Package.swift` gets the library and the test target with `StrictConcurrency`, matching the other library targets.

Types:

| Type | Role |
|------|------|
| `KVArchiveConfig` | `enabled` (default **false**), `blockTokens`, `sinkTokens`, `recentTokens`, `budgetTokens` (`B`), `genReserveTokens` (`R`), `querySpanTokens`, optional `hostQuotaBytes`. SSD fields exist and default off. |
| `KVPage` | `sessionId`, `pageIndex`, `origPosStart`, `tokenCount`, tier (`.active`, `.archiveRAM`, later `.ssd`), optional mean-K, payload handle. |
| `KVArchiveStore` | Append completed pages, drop them from the active set, lookup by session and position. P1 has no score. Quota eviction of **archive** pages is oldest-first and must not delete sink or the current generation block. |
| `KVWorkingSetPlanner` | P1 plan: sink range + recent range, middle listed as archived and not attended. P2 plan: same mandatory ranges, then fill remaining `B` by score, then sort the selected pages into chronological order. |
| `KVWorkingSetAssembler` | Build the dense cache the model already accepts. P1: sink prefix kept, recent tail re-RoPE’d to sit against the sink, middle stored as pre-RoPE K plus V and left unattended. P2: same, plus selected middle pages re-RoPE’d into the gap in chronological order. |
| `KVArchiveMetrics` | Counters below. Separate from `PrefixCacheStats`. |

Engine integration, behind `enabled == false` unless a test or config turns it on:

- After `ChatSessionManager.getOrCreate` and before `streamDetails`, on `generateWithSession` / `streamWithSession` only.
- Enrollment check: full-attention KV only. Refuse when `hasLinearAttention` is true or `CacheBlockExtractor.usesRotatingKV` is true. Refusal leaves today’s path untouched and logs one line.
- On turn end, append newly **accepted** completion tokens as pages. Tool-call tokens that were actually emitted count. Speculative draft tokens that the target rejected do not.
- `ChatSessionManager.fork` while the archive is on copies that session’s pages into the new id (today’s fork is already a full copy). P1 does not share page storage across sessions.
- `remove` / TTL eviction drops that session’s pages. It does not call `MLX.Memory.clearCache`.

Admin, new group on the admin port, for example `GET /admin/kv-archive/{sessionId}` and `DELETE` for that session. Do not extend `/admin/cache/{modelId}/stats`.

Config lives next to prefix-cache settings (`ServerConfig` plus `~/.nova` / `~/.config/novamlx`), default disabled. A request field is unnecessary in P1. Per-request opt-in can wait until the flag is safe.

Client that never sent the old tokens cannot be reconstructed. If the client compacted the transcript to a summary and resent it **without** `session_id`, the server never stored those KV pages. The archive does not invent them. `contextScalingTarget` still only changes reported counts so the client compacts sooner. The two features answer different questions and stay independent.

## Metrics

Record these as soon as the archive can see a session. P0 may count today’s session and prefix behavior with the archive flag off, so a later phase has a baseline.

| Metric | What it measures |
|--------|------------------|
| `resident_tokens` / `archived_tokens` | Active window versus archive for one session. |
| `sink_tokens` / `recent_tokens` / `retrieved_tokens` | Composition of `B`. P1 leaves `retrieved_tokens` at 0. |
| `gen_tokens` versus `R` | How close the turn came to the reserve. A turn that stops on the reserve increments `gen_reserve_stops`. |
| Tool-continuation KV hit | On the next request with the same `session_id`, tokens whose KV was still resident or restored from the archive, versus tokens prefilled again. |
| Re-prefill time | Wall time to prefill the query span (and any hole) after the working set is assembled. |
| Prefix hit, unchanged | Existing `PrefixCacheStats.hits` / `tokensSaved` and the session log `cacheEfficiency`. Archive metrics do not replace them. |
| Task success versus summary compaction | P3 only, on a fixed fixture. Report both arms. Do not cite the paper’s 43.8 → 48.4 as this run. |

## Out of scope

- NInfer, `.ninfer` checkpoints, Windows `start-ninfer.ps1` launchers, and the NInfer behavior that refreshes selection mid-generation so output can exceed `R`.
- MiniCPM-OCR, and any change to the Unlimited OCR gate model or its sliding-window cache.
- Porting CUDA kernels, llama.cpp `llama_memory_i`, or GGUF KV dtypes (`q8_0` recipes). Live KV dtypes stay whatever `kvBits` / TurboQuant already use.
- PolyStrata and expert-affinity routing. The KVMem README mentions PolyStrata as another surface on that Windows stack. It is related reading only. This design does not select experts.
- Replacing `FlintPrefixEngine.select`, `select_prefix`, or Flint’s sliding ring.
- Changing MTP, DFlash, DSpark, `DraftModelRegistry`, or speculative acceptance.
- Renaming `companionReady`, `loaded`, or product branding.
- Editing `vendors/mlx-swift`. MLX patches still go through `Scripts/patch-mlx-complex.py` and `Scripts/patch-fused-sdpa.py`, and this work should not need a new patch.
- Fused-batch slot compaction, distributed KV, and TIE weight tiers.
- A Metal page table or a modified SDPA kernel.
- Claiming dense attention over the archived workspace.

## Related pages in this repo

- `docs/features/kv-cache.md` — prefix share and `session_id` (shipped behavior).
- `docs/features/sessions.md` — fork and idle TTL.
- `docs/features/turboquant-kv.md` — quantized KV layout under the sampler.
- `docs/features/regression-perf-gate.md` — floors this work is not allowed to lower.
- `docs/features/companion-ready.md` — why `companionReady` is not chat-ready.
- `docs/porting/kv-page-archive-IMPLEMENTATION.md` — phased checklist for the implementation PRs.
