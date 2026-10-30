# KV page archive — peer checklist (P0–P3)

Implementation checklist for a follow-up engineer. “Peer” here means that reviewer, not the Tknet Peer gateway.

Spec: [`kv-page-archive-swift-mlx.md`](kv-page-archive-swift-mlx.md).

This file is the plan. Implementation PRs come after it. Each phase is its own PR. Default remains **archive off**, so the shipped chat path stays the path this tree has today.

## Global constraints

- Swift 6, `StrictConcurrency`, 4-space indent. New library target `NovaMLXKVArchive` plus `Tests/NovaMLXKVArchiveTests` in `Package.swift`.
- Build and test with `./build.sh` and `./build.sh test --filter <name>`. Do not drive this work with a raw `swift build`.
- Do not edit `vendors/`. Do not add an MLX Metal patch for paging.
- Do not rename `companionReady` (`AdminModelStatus` in `Sources/NovaMLXAPI/OpenAITypes.swift`). Archive state must not change `loaded` or `companionReady`.
- Do not rename NovaMLX. Agent Infer is the session workload, not a rebrand.
- `MLXMemoryGate.clearCache` stays an allocator flush (`MLX.Memory.clearCache`). New archive APIs use names like `dropSession` or `evictArchivePage`.
- Sliding-window models stay out: `CacheBlockExtractor.usesRotatingKV` (Flint `FlintSlidingKVCache` / `BoundedSlidingKVCache`, `RotatingKVCache`, Unlimited OCR rings). Hybrid `hasLinearAttention` sessions stay memory-only and unenrolled.
- Flint `FlintPrefixEngine.select` stays the Flint prefix. `FusedBatchScheduler` keeps passing `nil` for the prefix manager on Flint.
- MTP, DFlash, and DSpark stay on their current registration path (`QwenMtpDecode.swift`, `DraftModelRegistry`, `InferenceService` SpecBoost). Accepted target tokens may be archived later. Draft tokens that were rejected are not pages.
- `contextScalingTarget` / `scaleTokenCount` stays a reported-usage scale. It is not `B` or `R`.
- Core regression floors stay at the values in `Scripts/regression-suite.yaml`:
  - `RecursiveSI/flint-int4` — correctness substring `391`, `min_seed_decode_tps` **100**
  - `mlx-community/Qwen3-8B-4bit` — correctness substring `391`, `min_seed_decode_tps` **50**
  - `baidu/Unlimited-OCR-2` — `expect_nonempty`, `expect_has_digit`, `gate_decode_tps: false`
  - Tolerance stays 8% under baseline, 5% with `--strict`. Do not lower `min_seed_decode_tps`. Do not weaken the OCR assertions. Do not put the archive suite on the default `core` gate.
- NInfer, `.ninfer`, Windows CUDA launchers, MiniCPM-OCR, and PolyStrata / expert-affinity are not tasks in any phase.
- Feature flag default `false` through P3. A core-gate run is an archive-off run.

## P0 — Inventory and smoke metrics

Goal: confirm the file map in the design still matches the tree, and record the metrics the later phases will be judged against. P0 does not change which tokens are attended.

### Touch

- Design doc hook table, if a symbol moved.
- A small metrics sink that can count **existing** signals with the archive flag absent or false:
  - session `.info` already logged in `MLXEngine.generateWithSession` / `streamWithSession` (`cachedPromptTokenCount`, `cacheEfficiency`)
  - `PrefixCacheStats` already exposed at `GET /admin/cache/{modelId}/stats`
- Tests that lock those routes and field names.

### Leave alone

- `ChatSession` growth, `fetchPrefix` / `storeCache`, Flint `select_prefix`, OCR `makeCache`, MTP registration, `companionReady`, `MLXMemoryGate.clearCache`.
- JSON keys of `/admin/cache/{modelId}/stats`.
- `Scripts/regression-suite.yaml` floors.

### Tasks

- [ ] Re-read the hook table in the design against `MLXEngine.generate`, `MLXEngine.stream`, `ChatSessionManager`, `FusedBatchScheduler` prefix fetch/store, `CacheBlockExtractor.admitsLongTermKV`, and the admin session and cache routes. Update the design doc in the same PR if a symbol moved.
- [ ] Add `KVArchiveMetrics` (or a P0-shaped subset) with the counters in the design’s metrics table. With the flag off, `retrieved_tokens` stays 0 and archive counters stay 0. Session and prefix counters may mirror the existing logs. They must not change control flow.
- [ ] Smoke, no weights required:
  - [ ] Unit test: metrics object starts at zeros; recording a session info event stores `cachedPromptTokenCount` without creating pages.
  - [ ] Unit test: a sliding-window cache (`usesRotatingKV == true`) and a `hasLinearAttention` flag both report “not eligible”. Eligibility is a pure function in P0. It does not load a model.
  - [ ] If a local server is already up, one `session_id` two-turn request still logs `ChatSession` reuse, and `GET /admin/cache/{modelId}/stats` still returns `hits`, `misses`, `tokensSaved`. Missing weights skip this live check. The unit tests are the required bar.
- [ ] Document the live prefix route versus the catalog shorthand (`docs/features/kv-cache.md` says `/admin/models/{id}/cache`; the router group is `admin/cache`). Do not “fix” the route in P0.

### Done when

- A reader can point at the symbols that own session KV, prefix KV, and allocator `clearCache`, and those three are still separate.
- New counters exist and are inert when the archive is off.
- `companionReady` still means “on-disk speculative draft”, and `loaded` still means resident.

### Non-claims

P0 does not report a KV hit rate for tool continuations beyond the session `cacheEfficiency` number that already exists. It does not report task success.

## P1 — Sink + recent, page archive, no retrieval

Goal: on an enrolled full-attention session, completed middle pages move to the archive and the active cache keeps a sink prefix plus a recent tail, inside budget `B`, with new tokens charged to `R`. Sealing a page stores pre-RoPE K. Installing the recent tail rewrites that K at the compact index directly after the sink. There is no score, no retrieved middle page, and no quality comparison to full history.

### Touch

- New `Sources/NovaMLXKVArchive/` and its test target.
- `Package.swift` registration only.
- A single call site in `generateWithSession` / `streamWithSession`, guarded by `enabled`.
- Config defaulting to off (`ServerConfig` or the same style of file `PrefixCacheLocation` uses). Off must be a pure no-op: `getOrCreate` and `streamDetails` run as they do now.
- Session `remove`, TTL cleanup, and `fork` so pages do not outlive the session or get silently shared.

### Leave alone

- `PrefixCacheManager`, Flint prefix, fused-batch KV, MTP verify, OCR cache construction, `vendors/`.
- Retrieval scoring, mean-K, LRFU, SSD.
- Any user-facing sentence that says the model still sees the archived middle. P1’s middle is stored and **not** attended.

### Tasks

- [ ] `KVArchiveConfig` with `enabled = false`, `blockTokens`, `sinkTokens`, `recentTokens`, `budgetTokens`, `genReserveTokens`. Reject a config where sink + recent exceeds `B`.
- [ ] `KVPage` identity is `(sessionId, pageIndex, origPosStart)`. Payload is pre-RoPE K plus V. Seal a page by unrotating the live K with the position it was written at. Quantized live KV is dequantized for that seal, or the page stores the quantized values plus the scale and the original position so a later install can dequant and re-RoPE. State which one the test used.
- [ ] Planner output for a history longer than sink + recent:
  - [ ] active token ids = the sink span concatenated with the recent span
  - [ ] middle pages present in the store, absent from the active cache
  - [ ] chronological order of each span preserved
  - [ ] sink K left at positions `0 ..< sink` (it was already written there)
  - [ ] recent K re-RoPE’d to positions `sink ..< sink+recent` on install
- [ ] Assembler does not score and does not gather an arbitrary middle page. The only position rewrite in P1 is that recent-tail repair. A test checks a known RoPE pair: K sealed at a high original position and installed at the compact recent index matches a fresh RoPE at that compact index within a stated tolerance.
- [ ] Decode accounting: accepted tokens append into `R`. When the next page would pass `R`, the turn stops with a typed error (or the engine’s existing length stop, if that is the only clean hook) and **does not** evict a sink page or an archived page to free a slot. Increment `gen_reserve_stops`.
- [ ] Enrollment: full-attention session only. Flint, Unlimited OCR, and hybrid linear-attention models log one refusal and take the current path.
- [ ] `fork` copies pages into the new session id. `remove` and TTL eviction drop them. Neither calls `MLXMemoryGate.clearCache`.
- [ ] Archive bytes are visible to the memory budget (the same `effectiveBytesPerToken` style of charge the fused path already uses, or an explicit byte counter the enforcer can read). A unit test can assert the counter moves when a page is archived and falls when the session is dropped.
- [ ] Flag-off test: planner and assembler are not invoked from `MLXEngine` (call-site test or a seam that records invocations).

### Tests (synthetic caches, no checkpoint)

- [ ] History of N blocks, sink = 1 block, recent = 1 block: active length is 2 blocks; archive holds the middle; token order inside sink and inside recent matches the source.
- [ ] `R` exhausted: error or length stop; sink page still in the active set; archive middle still stored.
- [ ] Sliding-window cache is refused and the source cache object is unchanged.
- [ ] `enabled == false` leaves token counts and cache identity unchanged.
- [ ] Forked session id has its own page list equal to the source at fork time. Mutating one does not mutate the other.

### Done when

- An enabled test session can grow past `B`, keep sink + recent resident, and show the middle only in the archive.
- A disabled engine test matches the pre-P1 session path.
- Docs and logs call this a truncated window plus an archive. They do not call it retrieval.

### Non-claims

P1 must not publish a retrieval hit rate, a task-success delta, or a tok/s claim. Re-prefill time is not meaningful yet because dropped middle tokens are not brought back.

## P2 — Select a working set and assemble it

Goal: once per agent step, after the new user or tool message is known and before decode, score archived middle pages and assemble sink + selected middle + recent into one dense cache the current model forward can run. The selected set stays pinned for the whole turn.

### Touch

- `KVWorkingSetPlanner` scoring and `KVWorkingSetAssembler` gather, inside `NovaMLXKVArchive`.
- The same session call site, still behind the flag.
- Mean-K captured when a page is sealed, from the pre-RoPE K that P1 already stores.
- Metrics: `retrieved_tokens`, tool-continuation restored-versus-reprefilled token counts, re-prefill wall time.

### Leave alone

- LRFU default **off**. If a flag exists, `top-frac` updates scores only. It does not change `B`.
- SSD tier off.
- Metal SDPA, `vendors/mlx-swift`, Flint, OCR, MTP control flow.
- Paper percentages and laptop tok/s figures. Do not copy them into assertions.

### Tasks

- [ ] Candidate set = archived middle pages. Sink and recent are mandatory, consume `B`, and are masked out of the score denominator.
- [ ] Score = attention-space mean-K against the query span (latest user or tool message, capped by `querySpanTokens`, default 512). Average across the layers and heads the fixture configures. Sort selected pages by original position before assembly.
- [ ] If the mandatory sink + recent already fill `B`, select nothing else. That degenerate plan matches P1.
- [ ] Assembler writes a dense cache: V copied, K re-RoPE’d to compact positions. Positions are the compact index, recorded for the next turn’s unrotate. Numerical test: a page assembled, dropped, and assembled again at the same compact position matches within a fixed tolerance (state the tolerance in the test, do not leave it implicit).
- [ ] Pin: during decode the planner does not run again. New KV only occupies `R`. Filling `R` still refuses to evict a selected page.
- [ ] Tool continuation: a scripted second request with the same `session_id` and a new tool result. Assert some archived page can return to the active set when it is the unique high score, and a low-score page stays archived. This is a selector test on synthetic K, not an agent benchmark.
- [ ] Restored tokens increment the continuation counter. Tokens of the new tool message increment the re-prefill counter and record a duration. Both counters are zero when the flag is off.
- [ ] Rejected speculative tokens are absent from new pages (feed the assembler a list of accepted positions and a longer draft list; only accepted positions appear).
- [ ] Query-dependent sanity: two different query vectors can select different middle pages when their mean-K dots differ. The same query selects the same pages.

### Done when

- A unit fixture shows selection changing with the query, chronological reassembly, and a stable re-RoPE round trip.
- Continuation metrics distinguish restored history from the new tool span.
- Flag off still matches P1’s no-op engine path.

### Non-claims

P2 does not claim the working set matches full dense attention, and it does not claim a task-success lift over summary compaction. LRFU stays off unless a follow-up fixture, in a later PR, shows a measured gain on the P3 task.

## P3 — Regression gate and agent-session acceptance

Goal: prove the core floors still hold with the archive **off**, and add an opt-in acceptance check for the archive **on**. Acceptance is allowed to be small. It is not allowed to replace the core gate.

### Touch

- `docs/features/regression-perf-gate.md` — one paragraph that the core suite is an archive-off run, and that a new opt-in script exists.
- A new script or Swift test target that is **not** in `Scripts/regression-suite.yaml` `core:`.
- Optional SSD store, default off, only after P2’s RAM tests are green. Reuse the durability pattern in `SSDCacheStore` (TTL, size cap, async read that does not block the turn). Use archive page ids, not `BlockHasher` prefix hashes.

### Leave alone

- `min_seed_decode_tps` for Flint and Qwen3-8B.
- Unlimited OCR expectations (`expect_nonempty`, `expect_has_digit`) and `gate_decode_tps: false`.
- `companionReady` wait rules in `Scripts/regression-perf-gate.py` (wait on `loaded` plus `/health.loadedModels`, never on `companionReady` alone).
- Default suite selection (`core`).

### Tasks

- [ ] Run the existing core gate with the archive flag off (or unset). Record the command and the result path (`~/.nova/regression/results-*.json`) in the PR.
  - [ ] Flint correctness `391` and decode tok/s within tolerance of baseline, and not below the seed floor 100 if a baseline is being seeded.
  - [ ] Qwen3-8B-4bit correctness `391` and the seed floor 50 under the same rule.
  - [ ] Unlimited OCR non-empty, contains a digit, decode tok/s not gated.
- [ ] Any PR that touches shared KV types (`KVCache`, `RotatingKVCache`, `FlintSlidingKVCache`, `QuantizedKVCache`, `CacheBlockExtractor`) re-runs that core gate even if the archive flag is off. A failure there blocks the PR. Fix the regression. Do not edit the floor to make it pass.
- [ ] Add an opt-in acceptance entry, excluded from `core` and from `optional` unless the owner of the gate agrees in a later PR. Suggested name: `Scripts/kv-archive-acceptance.sh`, documented next to the gate page.
  - [ ] Arm A: archive on, scripted multi-turn `session_id` conversation that includes a tool-result continuation. Record KV/prefix continuation hit rate (restored tokens / history tokens the server had already seen), re-prefill time of the new span, and task success on a frozen prompt.
  - [ ] Arm B: the same prompt with summary-only compaction (the middle turns replaced by a short text summary, archive off). Record task success on the same frozen answer key.
  - [ ] The script prints both arms. It does not fail the core gate. Thresholds, if any, are whatever the first quiet Mac run justifies, written into the script’s comment with the machine and the git sha. They are not the paper’s 43.8% and 48.4%.
- [ ] Eligibility check on the three core model ids, as a unit test of the pure function: Flint ineligible, Unlimited OCR ineligible, Qwen3-8B-class full attention eligible. This test does not load weights.
- [ ] SSD, if implemented: default off; a test writes a page, drops the RAM copy, reads it back, and checks original positions. A failed disk read falls back to re-prefill of that span and increments a counter. It does not crash the turn.
- [ ] Product strings in any new log or admin JSON use “addressable workspace” and “active window”. Search the diff for “million-token context”, “unlimited context”, and “full history in context” and remove those phrases.

### Done when

- Core gate evidence is in the PR, archive off, floors unchanged.
- The opt-in script can run on a machine that has the weights and can be skipped where the weights are absent, without failing `./Scripts/regression-perf-gate.sh`.
- `companionReady` field name and semantics are unchanged in the admin model JSON.

### Non-claims

A green P3 script is evidence about one fixture on one Mac. It is not a claim that NovaMLX matches KVMem’s CUDA evaluation, and it is not a license to turn the flag on by default.

## Suggested PR split

| PR | Contents |
|----|----------|
| Design (this change) | The two docs under `docs/porting/` and the related link on `docs/features/kv-cache.md`. No Swift. |
| P0 | Metrics types, eligibility pure function, hook-table refresh if needed. Flag absent or off. |
| P1 | `NovaMLXKVArchive` sink+recent window with pre-RoPE archive pages, session call site, flag default off. |
| P2 | Mean-K select, re-RoPE assemble, continuation counters. |
| P3 | Core-gate evidence plus the opt-in acceptance script. SSD only inside this PR if P2 has already landed. |

Each implementation PR links the spec and ticks the boxes it actually finished. A phase that needs a design change updates `kv-page-archive-swift-mlx.md` in that PR and says what changed.
