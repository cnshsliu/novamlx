# Changelog

All notable changes to NovaMLX will be documented in this file.

## [Unreleased]

### Added
- **Long-term prefix cache root, TTL, and hot RAM** — prefix KV files resolve to the first mounted, writable root. `NOVA_PREFIX_CACHE_ROOT`, `~/.config/novamlx/prefix-cache-path`, or `ServerConfig.prefixCacheRoots` can list an external disk (for example `/Volumes/Samsung768/NovaMLX/prefix_cache`) and then an internal path (`~/Models/prefix_cache`). A missing volume is skipped. The last resort is `~/.nova/prefix_cache`, which is created when used. Entries expire from `created_at` (`NOVA_PREFIX_CACHE_TTL`, `prefix-cache.json` `ttlSeconds`, or `prefixCacheTTLSeconds`; default 24h). A fetch serves resident blocks from a RAM tier (`NOVA_PREFIX_CACHE_HOT_BYTES` / `hotCacheMaxBytes`, default 256 MiB) and promotes SSD misses on a background queue, so prefill does not wait on disk. The tier drops the least recently used block when it is over budget; a block larger than the budget is not kept in RAM. The long-term files remain. Flint's in-memory `select_prefix` is unchanged, and `storeCache` still refuses a sliding ring (`FlintSlidingKVCache`, `RotatingKVCache`, including through the rec proxy), so that window is not flattened to SSD. Intro: [KV cache](docs/features/kv-cache.md).
- **Flint sliding-window replay** — sliding layers keep a window-sized ring (`FlintSlidingKVCache`) instead of a growing `RotatingKVCache`. A later prompt that diverges after the ring has overwritten a token restores the last 65536-token checkpoint and prefills the suffix. Full-attention layers still grow. `num_kv_shared_layers` (absent on official Flint) makes the last N layers reuse KV from the last earlier layer of the same kind. `maxKVSize`, `kvBits`, `kvCache`, and `kvScheme` keep the previous hybrid caches. Intro: [Flint](docs/features/flint.md).
- **Segmented prompt encode and incremental tool calls** — long prompts can be encoded in segments when the specials are known at render time: each special is one id, and plain regions are encoded on their own, so a special spelling inside user text is not a special id. Flint's chat template is now built from those segments; the rendered string is unchanged, and the generation hot path still uses `prepare`. An untemplated fallback uses the segmented encode only when every non-empty message has its role special in the vocabulary; otherwise it keeps the one-shot join. On the fused stream, a tool name is emitted when it is known and argument JSON follows in pieces. Those pieces concatenate to the finished call. OpenAI streams that already received the pieces do not emit the finished call again, so clients that append `delta.tool_calls` still see one arguments string. A call the incremental parser does not own stays one OpenAI delta. Responses streams emit `response.output_item.added`, `response.function_call_arguments.delta`, and `response.function_call_arguments.done` for both incremental and whole calls. Intro: [tool calling](docs/features/tool-calling.md).
- **Host grammar on fused decode** — JSON object, JSON schema, regex, and GBNF stay on `FusedBatchScheduler` when the request is otherwise a `KVCacheSimple` chat. The mask is applied per row after the frequency penalty and before the Flint filter, then the row is sampled. Session, VLM, hybrid attention, draft-model verify, native MTP, and `use_mtp=false` still use `ContinuousBatcher`. Before this, any of those grammar flags left fused for the engine iterator, and the first `process(logits:)` scanned the vocabulary inside the decode step. The initial mask is now compiled on the submitter (or a detached admit task) before the sequence joins the decode set. JSON's full table is cached once per model, EOS set, and vocab width. Schema, regex, and GBNF cache each state; a miss clears the ready mask, compiles on a detached task, and the sequence skips forwards until the mask is installed, so a waiting partner does not pull a lone Flint sequence off packet-16. A logits width wider than the compiled mask is recompiled from the real dimension before the first sample. The log line is `[Grammar] compiled kind=… vocab=… cache=hit|miss elapsedMs=…`. A second identical request logs `cache=hit`. Intro: [structured output](docs/features/structured-output.md).
- **Online calibration for DFlash2 / DSpark adaptive verify** — sequential temperature scaling now updates each draft position from recent accept/reject labels and stores the fit per draft+target pair in `~/.nova/adaptive_verify_calibration.json` (override with `NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION_PATH`). Adaptive mode uses those temperatures when it picks the verify prefix. Each round only records a label; the fit and the file write run when the request finishes, after decode timing, and the next request for that pair loads the result. Fixed mode keeps the historical block and still records labels, so a fixed warmup can teach the next adaptive request. `NOVAMLX_ADAPTIVE_VERIFY_CALIBRATION=off` leaves the previous uncalibrated scheduler. Non-streaming chat completions include `usage.nova_speculation` (`proposed`, `accepted`, `accept_rate`, `decode_tokens_per_second`, `temperatures`) when a DFlash2 or DSpark round verified drafts. `python3 Scripts/adaptive-verify-ab.py` runs the same prompts in both modes and writes tok/s and accept rate. The comparison to chase on a Mac is the DeepInfer band of about +4% to +7% decode tok/s versus fixed verify; this change supplies the measurement, it does not record one. Intro: [speculative decoding](docs/features/speculative-decoding.md).

### Fixed
- **Qwen MTP greedy verify** — companion MTP heads (`qwen3_5_mtp` / `qwen3_8_mtp`) are no longer treated as a chat draft that skips verification. Temperature 0 loads the head with `MTPDrafterModelFactory.shared` (text and VLM registrations share that registry) and checks drafts with `MTPSpeculativeTokenIterator`, so accepted tokens match plain greedy decode. The loaded head must be `Qwen35MTPDraftModel` for a text target or `Qwen35VLMNextNDraftModel` for a VLM target; any other head stays on serial decode. Temperature above 0 stays on serial decode. The Engine `MtpTokenIterator` stub (`MtpTarget` / `MtpDrafter`), which sampled with an ordinary `TokenIterator` while logging MTP, now throws instead of running. DeepSeek's in-graph head still uses those Nova protocols and stays on serial decode until it moves to `MTPDrafterModel`.

## [1.8.2] - 2026-10-27

**Headline: Flint chat follows official rec (P0–P2), and think-phase stops stay on the filtered packet.** Family `.flint` matches Recursive `rec` through the P0–P2 alignment: every eos id and a new `<think>` after visible text stop the turn (#42), an omitted `thinking_budget` leaves the chain of thought under `max_tokens` (#44), structural grammar allows one `<think>` block (#45), grammar and recommended sampling stay on the filtered resident packet at width 16 (#46), HTTP defaults match `rec` (#47), `enforce_tool_schemas` can constrain tool arguments (#48), and the official chat template, session prefix, seed, and schema-aware tool calls are in place (#49). #50 keeps the grammar stops sampleable inside an open `<think>`, including `<|developer|>` as a checkpoint EOS, so a temperature-0 grammar-on hello finishes, and that default chat stays on the filtered packet. Intro: [Flint](docs/features/flint.md).

### Fixed
- **Flint omitted `max_tokens`** (#47) — when the client leaves `max_tokens` unset, Flint now fills the remaining context (`context window − prompt`), matching Recursive `rec` `server.py` `handle_completion`. The old path used ModelConfig's 2048. An explicit `max_tokens` is unchanged, including a value saved on the model and `/demo/chat`'s 2048. There is no extra 1024/2048/8192 cap under the window. A prompt that already fills the window is `contextWindowExceeded`. The memory preflight still applies to the resolved budget. Other families keep 2048. Intro: [Flint](docs/features/flint.md).
- **Flint sampling fallbacks** (#47) — when `generation_config.json` omits temperature, top-p, or top-k, family `.flint` now uses the official `rec` `Sampling` defaults (1.0 / 0.95 / 20) instead of ModelConfig 0.7 / 0.9 / top-k 0. A field present in `generation_config.json` still wins. Other families are unchanged.
- **Flint prompt sanitization** (#47) — Flint message normalize now matches `rec` `chat.py` `normalize_messages`, independent of Jinja. Non-assistant text drops NUL and `<think>` / `</think>`, then escapes `<|begin_of_text|>`, `<|end_of_text|>`, `<|system|>`, `<|developer|>`, `<|user|>`, `<|assistant|>`, and `<|observation|>`. Assistant text is unchanged. History tool-call argument strings must be JSON objects; blank becomes `{}`.
- **Flint custom stop** (#47) — a custom `stop` while structural grammar would be on is HTTP 400, `Flint grammar cannot be combined with custom stop strings`, the same rejection as `rec`. The request does not run, so `ThinkReentryGuard` is not the product behavior for that case. JSON schema, regex, GBNF, and JSON mode still turn grammar off and still allow a custom stop; the guard covers those. An empty stop list is not a custom stop. Grammar still returns nil if a custom stop reaches it, so the two cannot run together.
- **Flint thinking budget** (#44) — an omitted `thinking_budget` no longer smart-defaults a cap (temperature 0 used `min(1024, max(256, max_tokens / 2))`) that can force `</think>` before a long chain of thought finishes. Official Recursive `rec` and Hugging Face Flint have no separate reasoning budget, only `max_tokens`. An explicit `thinking_budget` is still honored. Other thinking families keep the automatic cap.
- **Flint chat stop** (#42) — a role-header eos id other than the tokenizer's primary `eosTokenId` now ends fused generation and `/batch/completions` the same way the primary eos does, including when that token's text was scrubbed. After the assistant has already produced visible text, a new `<think>` (token 150006, or the decoded tag) ends the turn with `finish_reason=stop`. `/demo/chat` sends `max_tokens: 2048`.
- **Flint structural grammar** (#45, #49, #50) — Flint chat and agent turns use the official rec flint shape: one optional `<think>…</think>`, then answer or tool-call text that masks think tags and protocol markers. Grammar stops (`<|end_of_text|>`, `<|user|>`, `<|observation|>`) stay sampleable in the think body and finish the turn there. `<|developer|>` is masked in the answer and is not a grammar stop. Sampling it during think finishes the turn because it is a checkpoint EOS id. Grammar off keeps the checkpoint multi-EOS set. A second `<think>` cannot be sampled. `ThinkReentryGuard` remains only when that grammar is off. `max_tokens` is unchanged and is not the stop. Intro: [Flint](docs/features/flint.md).
- **Flint hello finish and filtered packet** (#50) — a temperature-0 grammar-on hello opens `<think>` (token 150006). The think-phase mask had been setting the three grammar stops to `-inf`, and the grammar-on host finish set had dropped `<|developer|>`, so that body had no sampleable end token and ran until `max_tokens` (`finish_reason=length`). Those stops are terminals inside the open think. A sampled checkpoint EOS, including `<|developer|>` during think, finishes the turn. Default Flint chat (grammar on, temperature 0, top-p 0.95, top-k 20, one sequence) stays on the filtered resident packet, width 16. The serial iterator takes that packet when the structural grammar is installed, including when the device packet is vetoed. The fused scheduler logs `[Flint] resident route` once per sequence. Intro: [Flint](docs/features/flint.md).

### Added
- **Flint schema-aware tool calls** (#47, #49) — the fused Flint path parses `<tool_call>` bodies the way `rec` `parse_tool_call` does: string-typed arguments stay raw (including padding), other values are JSON, and `null` is null only when the schema accepts it. Call ids are UUID version 4 from `sha256(seed ‖ sampling JSON ‖ prompt token ids ‖ call index)`, matching `seeded_tool_parser`. An omitted Flint seed is 0, the same default as `Sampling.seed`. An explicit request seed wins over the model config, and that seed is also the sampler seed. Session, JSON, regex, and GBNF turns still use the stock parser. Intro: [Flint](docs/features/flint.md).
- **Flint `enforce_tool_schemas`** (#48, #49) — optional, default off. The chat, Anthropic, and Responses fields `enforce_tool_schemas` win; otherwise `NOVA_ENFORCE_TOOL_SCHEMAS` (`1` / `true` / `yes`); otherwise server config (`enforce_tool_schemas`, default false). Tool names are constrained whenever tools are present. Argument schemas are enforced when the flag is on or `strict` is not `false` (omitted `strict` enforces). `strict: false` is left false on the client tool and is still enforced when the flag is on; with the flag off, that body stays loose after the name is accepted. The host FSM follows xgrammar `glm_4_7` (`any_order=False`): `pattern`, `format`, string length, numeric bounds, nested object properties, and schema key order. Every tool needs a function name and an object parameter schema. Other families ignore the flag. Intro: [Flint](docs/features/flint.md).
- **Flint filtered resident packet** (#46, #49) — structural grammar and recommended sampling (temperature 1.0, top-p 0.95, top-k 20 from `generation_config.json`) stay on the resident packet. Official `rec` does this with `FilteredResidentSampler` at width 8: each step applies the grammar mask and the presence mask, then samples. Nova keeps width **16**. Temperature 0 with no grammar and no presence still uses the device argmax packet, also width 16. `frequency_penalty != 0`, `min_p != 0`, a repetition penalty other than 0 or 1, and streaming logprobs are HTTP 400 with the `rec` `validate_request` messages. Non-stream logprobs and a second sequence stay single-step. A second `<think>` is masked before the packet accepts it. Intro: [Flint](docs/features/flint.md).
- **Flint chat template and session prefix** (#49) — Flint load overwrites the checkpoint chat template with the official `rec` `chat_template.jinja` and quarantines a conflicting `chat_template.jinja`. The log line is `chat_template_sha256=44fd1b0f94b78d9cf554c5a207945e48fef698b279d65863a33a77d787ca411b`. `select_prefix` reuses a token-level common prefix (`cached_prompt_tokens` / `common_prefix_tokens`). `prepare_prefix` chunks a long generation (65536) and a cancelled chunk does not replace the anchor. This is separate from the block-hash `NovaMLXPrefixCache`, which Flint does not use for this protocol. `tool_choice` is `auto` or `none`, `parallel_tool_calls=false` is rejected, `store` must be omitted or false, and `grammar` is `none` or `flint`. Intro: [Flint](docs/features/flint.md).
- **Verified models on Downloads** — one A–Z list. Each model shows its exact catalog weights (one or more). The Models-page regex filter sits under the title. Family globs stay in their own section.
- **Flint official INT4 is verified** — `RecursiveSI/flint-int4` is in that list (`testedOn` 1.8.2).

## [1.8.1] - 2026-10-26

**Headline: Flint packet width 16 — LOOP-RESULT 226.8 tok/s, FLINT-CHAT 217.3 tok/s.** MacMaxPro M4 Max, official `RecursiveSI/flint-int4`. #38 (`ef57be1`): greedy resident packet width **16**, PARITY match 16, LOOP-RESULT **226.8 tok/s** (kernel packet), SINGLE-RESULT **190.7 tok/s**. #39 (`313d92f`): FLINT-CHAT **217.3 tok/s** (`decodeTokens / wallDecodeSeconds`, prefill excluded; packetTokens=128, singleTokens=0). That same run's LOOP-RESULT was **223.3 tok/s** — 217.3 sits below the kernel line on that run; 226.8 vs 223.3 is run variance. Recursive's published Mac INT4 rec reference remains **196.8 tok/s** (1K context / 256 decoded tokens — a different window, not a re-run). llama.cpp on the same machine was ~89.7 tok/s in an earlier reference and was not re-run. The 1.8.0 packet=8 record (214.7) stays in the section below. Intro: [Flint](docs/features/flint.md).

### Added
- **Flint chat decode tok/s** — Greedy Flint chat (temperature 0, one sequence, no logprobs, frequency penalty 0, no grammar) uses the resident packet on the fused scheduler and on the serial engine path. Packet width is 16 (`FlintRecResident.width`, #38). `FLINT-CHAT` is `decodeTokens / wallDecodeSeconds` (prefill excluded; EOS excluded). Measured at `313d92f`: **217.3 tok/s** (packetTokens=128, singleTokens=0), below that run's LOOP-RESULT of **223.3 tok/s** (kernel packet only, no tokenizer). The width-16 kernel LOOP-RESULT at `ef57be1` is **226.8 tok/s**, with SINGLE-RESULT **190.7 tok/s** and PARITY match 16. The fused scheduler writes the `FLINT-CHAT` line to `~/.nova/novamlx.log` when the sequence finishes. `FlintSpeedTests` prints the same line after `LOOP-RESULT` for a chat-shaped window (one tokenizer decode per packet). Intro: [Flint](docs/features/flint.md).

## [1.8.0] - 2026-10-25

**Headline: Flint official INT4 packet decode, 214.7 tok/s.** MacMaxPro M4 Max, `RecursiveSI/flint-int4`, `./build.sh test --filter FlintSpeedTests` at `e8e7b27`. LOOP-RESULT is **214.7 tok/s** (128 rec steps, launch 76/64/64, greedy packet=8, argmax on device). Same run, one launch plus host argmax: **181.6 tok/s**. Recursive’s published Mac INT4 rec reference is **196.8 tok/s** (1K context / 256 decoded tokens — a different window, not a re-run of this suite). llama.cpp on the same machine was ~89.7 tok/s in an earlier reference and was not re-run. 214.7 is the shader timing window after compile, not HTTP end-to-end. Intro: [Flint](docs/features/flint.md), [brief](docs/feature-briefs/2026-10-25-flint.md).

**Core change: Flint (RecursiveSI) model support** — see the Flint entry below.

### Added
- **Running build on Status** — the Status page footer is one line: Debug or Release, version, build time to the second, and the `.app` path. That distinguishes `/Applications/NovaMLX.app` from a copy under the repo. The menu bar title includes Debug or Release, and the menu lists the same line and path. Settings → About shows configuration and location. `./build.sh` installs the newest host binary when SwiftPM has written more than one output directory. Intro: [menu bar](docs/features/menu-bar.md), [feature brief](docs/feature-briefs/2026-10-24-build-identity.md).
- **Flint (RecursiveSI) full inference** — 39.8B MoE agent model (~2.5B active) ported from mlx-lm to a native Swift/MLX `FlintModel`: GQA + per-head QK RMSNorm + learned attention sinks, hybrid 35 sliding-window(513)+RoPE / 14 full-attention-NoPE layers (RotatingKVCache mix), sigmoid-routed MoE (6/256 + ungated shared expert, dense layers 0-1), expert-stacking sanitize with uint8→uint32 views for flint-int4. Chat layer: `ChatTemplateFormat.flint` detection, FlintProcessor (role-header eos stop + scrub, think tags semantic), GLM4-dialect tool calls normalized to standard OpenAI tool_calls (Flint's syntax is exactly glm4), defensive eos ids. Flint routes through ContinuousBatcher — the fused streaming path has no ToolCallProcessor. Greedy output verified token-identical to Python mlx-lm with the same weights; reasoning + tool calls on OpenAI and Anthropic APIs, streaming and non-streaming; ~24 tok/s end-to-end (823 tok / 34.9 s) on M4 Max at `mlx-community/flint-OptiQ-4bit` (~26 GB resident). The implicit-thinking heuristic now matches its documented semantics (an open `<think>` only counts as an injection inside the add_generation_prompt block; Qwen3.5 unaffected). Intro: [feature page](docs/features/flint.md), [marketing report](marketing/why-novamlx/flint.md).
- **Flint default weights → official `RecursiveSI/flint-int4`** — catalog/docs/marketing now point at Recursive's official INT4 build (~22.6 GB, 4-bit lm_head and projections; the community OptiQ-4bit build carried 8-bit lm_head/projections) instead of `mlx-community/flint-OptiQ-4bit` used for early verification. Official config carries `rope_parameters.rope_type: "default"` (a string enum), so `FlintConfiguration` now decodes rope_parameters tolerantly, reading only numeric `rope_theta`. Gate 0 baselines on the official build (M4 Max, short-context pure decode): 22.7 tok/s bare decode eager vs llama.cpp Q4_K_XL 89.7 tok/s on the same machine.
- **Verified badge on Hub search results** — with "Allow unverified downloads" on, the Downloads page tags Hub results that are in the NovaMLX verified catalog with a green Verified badge, so unsupported repos are visible before downloading. Default catalog-gated search is unchanged.
- **3D generation (Hunyuan3D-2.1)** — task-style image-to-3D on a local Swift/MLX pipeline (vendored `hunyuan3d-swift`, no Python). `POST /v1/3d/generations` accepts one image (data URI / base64 / URL / local path) and returns 202; poll `GET /v1/3d/generations/{id}` or the shared `GET /v1/tasks/{id}` (`task = "3d.generation"`) through stable phases `queued → loading → shape → (paint) → export → done`; download the GLB from `/content`. One 3D job at a time (serial queue, like image tasks), cancellable mid-denoise. Shape weights stay resident until unload (`NOVA_HY3D_QUANT=0|4|8` cuts the resident DiT from ~3.6 GB to ~1.4 GB); optional `texture: true` loads a paint model per job and peaks ~25–39 GB (64 GB+ Macs). Weights are MLX conversions of official `tencent/Hunyuan3D-2.1` under the Tencent Hunyuan3D Community License — catalog ids `zimengxiong/hunyuan3d-mlx-shape-small` (~3.8 GB), `-shape-large` (~4.9 GB), `-paint-large` (~8.7 GB). `usage.meshes`/`usage.size_tier` are stable metering fields for tknet/peer billing (Level-2); `cost` stays 0. Intro: [feature page](docs/features/hunyuan3d.md), [feature brief](docs/feature-briefs/2026-10-22-hunyuan3d.md).
- **Pin on Local Inference** — each loaded model has a Pin switch. While it is on, Unload is hidden, admin unload/delete refuse, exclusive auto-unload and memory eviction skip it, and restart restores it even if it dropped off the last-loaded list. Turn Pin off to unload.
- **Adaptive verification length for DFlash2 and DSpark** — speculative verify no longer always scores a fixed draft block. Each round drafts up to the block budget, then verifies the prefix that maximizes expected accepted tokens per unit of target-verify cost. DSpark uses its confidence head. DFlash2 uses the top-two draft-score margin. Default mode is adaptive (min 1, max = draft budget, fixed cost 2, marginal cost 0.40 inside the fast SDPA window and 0.55 past 4 drafts). `"adaptive_verify": "fixed"` or `NOVAMLX_ADAPTIVE_VERIFY=fixed` restores the previous block, including DSpark’s 4-draft vector-SDPA cap. Greedy accept is unchanged, so a shorter prefix follows the same tokens as a fixed block on that prefix. Companion attach (known-faster auto-attach, known-slower skip, uncertain ask-once) is unchanged. `nova adaptive-verify --confidence …` prints the scheduled length without loading a model. Intro: [feature page](docs/features/speculative-decoding.md), [feature brief](docs/feature-briefs/2026-10-21-adaptive-verify.md).
- **MOSS-Image text-to-image** — `POST /v1/images/generations` loads OpenMOSS MOSS-Image through the existing image service. Catalog ids: `OpenMOSS-Team/MOSS-Image-SFT` (generation, ~27 GB) and `OpenMOSS-Team/MOSS-Image-Base` (same loader, research pretrain). Defaults are Euler, 50 steps, CFG 3, shift 5, empty negative, negative-branch skip of layer 22. Sides snap to multiples of 16 (minimum 256, area at most 1456×1456). Unified memory under 48 GiB quantizes the DiT and text encoder to int8 on load; `NOVA_MOSS_IMAGE_QUANT=bf16` or `int8` overrides. Edits and variations stay unavailable for this family. Intro: [feature page](docs/features/image-generation.md), [feature brief](docs/feature-briefs/2026-10-20-moss-image.md).
- **Model list filters** — On Local Inference, the “Downloaded, not loaded” header has a disk menu (All, plus each configured model root, labeled by volume or folder name) and a “Show companions” checkbox. DSpark, DFlash / DFlash2, and MTP companion weights stay hidden by default in both that list and the loaded-models list. The checkbox is remembered. Hiding only changes the lists; files stay on disk.
- **Supported models list** — [`docs/features/supported-models.md`](docs/features/supported-models.md) lists each catalog weight id (`org/repo`) from `catalog/models.json`. The root README links that page. Regenerate with `python3 Scripts/generate-supported-models-doc.py`.

## [1.6.0] - 2026-10-20

### Added
- **Feature catalog** — user-facing docs are indexed at [`docs/features/README.md`](docs/features/README.md) (one markdown page per feature). The root README links that index. `features.md` and `features.zh-CN.md` now point at it instead of keeping a second full catalog. Promo briefs remain in `docs/feature-briefs/`.
- **Text-only Nemotron 3.5 Bolt** (`nemotron_h_puzzle`) — load the Rapid-MLX 4-bit and 3-bit catalog previews as a text LLM (~43 GB disk / 64 GB unified memory, ~32 GB / 48 GB). Thinking defaults on (`enable_thinking`); tool calls use the checkpoint chat template (`xmlFunction`). Load refuses before Metal allocation when unified memory is below the indexed weight size plus 16 GiB (distributed workers pass `skipMemoryGate`). Intro: [feature page](docs/features/nemotron-bolt-text.md), [feature brief](docs/feature-briefs/2026-10-20-nemotron-bolt-text.md).
- **Request Log page** (new **Requests** sidebar entry) — scrolling record of every HTTP request the server handles. Upper section shows **live in-flight requests** (model, kind badge, API key used, elapsed time, live tok/s, auto-refreshing every 0.5s); lower section shows **completed requests** (method, endpoint, model, kind, API key, status, duration, TPS, error) with an "Errors only" filter and a "Clear" button. Backed by a new `RequestLogStore` (`NovaMLXUtils`), a `RequestLogMiddleware` (records start; errors finalize from HTTP status), and per-backend finalization hooks: LLM/VLM via `MLXEngine.generate`/`generateWithSession`/`generateWithProcessor` + a stream-tracking shim wrapping the fused/cluster/continuous-batcher/worker streams, ASR/TTS/image-gen at their route handlers. Capped at 300 recent entries. The HTTP request id is threaded from the `x-request-id` header into `InferenceRequest.httpRequestId` so the engine finalizes the exact matching log row.

### Fixed
- **Request Log: streaming ASR entries no longer get stuck as "active" forever.** Root cause was an id-correlation bug: `RequestIDMiddleware` and `RequestLogMiddleware` each generated their *own* request id, while the handler's `HTTPHelpers.requestID(from:)` read the *unmodified inbound* request and got back `nil` — so the streaming completion callback could never match its log entry, leaving the row pinned in the active list. Fixed by having `RequestIDMiddleware` propagate the resolved id back **onto the request** (rebuilds the `Request` with the `x-request-id` header stamped on it) before calling `next`, so the log middleware and every downstream handler read one consistent id. Also wired `cancelStale(olderThan: 120)` into the Requests page 0.5s tick as a safety net so any future uncaught orphan entries self-prune.
- **"Realtime Inference Speed" panel + dashboard speed card now reflect ALL model kinds**, not just LLM tok/s. ASR (transcription), TTS (speech synthesis), and image generation (SD/FLUX) now report live activity (model name + kind badge + kind-appropriate unit: `×RT`, `img/s`, `tok/s`) to the shared `MetricsStore`. The Status page hero icon and active-model line switch per kind (`waveform`/`speaker.wave.2`/`photo`/`eye`/`bolt`), so you always see what is running. Backed by a new `LiveActivity` record + `InferenceKind` enum, plumbed through `InferenceService` stats → `MenuBarAppState`.
- **VLM prefill now uses `model.prepare()` instead of `callAsFunction`**, so image features are actually merged into token embeddings during prefill. Previously, calling the model callable directly on raw `<|image_pad|>` token IDs made the LM decode placeholder tokens with no visual information — producing garbage logits that collapsed to EOS in a single step, so every VLM described pasted images wrongly (e.g. a red square → "Yellow", a man-on-horse hallucination). Fixed in both streaming and non-streaming paths in `MLXEngine.swift`; requires boxing `LMInput` payload into `SendableBox` for the `@Sendable` perform closure.

### ⚠ Breaking Changes
- **TokenHub is now remote-providers-only.** Local models no longer appear in the TokenHub provider list. Local inference remains in the renamed **Local Inference** page (formerly "Models").
- **`includeInLoadBalance` toggle removed** from provider cards. Multi-provider routing is now expressed via named Load Balancers, not via the per-provider LB flag.
- **Bare `tknet` dispatch rejected.** Requests with `model = "tknet"` now return 400 with a migration hint. Use `lb:<slug>` for load-balanced dispatch or `tknet:<provider-name>` for direct provider dispatch.
- **Implicit global LB removed.** Requests that previously fell through to the implicit global LB (every provider with `includeInLoadBalance=true`) now return 404. Create an explicit LB via the new **Load Balancers** page or `/admin/load-balancers` API.
- **`is_managed` column dropped.** Cloud-managed status is now expressed via the `"managed"` tag on the provider.

### Added
- **First-class Load Balancer entity** with its own SQLite tables (`load_balancers`, `lb_members`, `lb_member_stats`) and admin REST API at `/admin/load-balancers`.
- **Named LB dispatch** via `model = "lb:<slug>"` prefix. LBs support 5 selection strategies: `tiered` (default), `round_robin`, `weighted`, `lowest_latency`, `random`.
- **Per-member statistics**: request/success/failure counts, 5xx counter, latency average, last error — exposed via `/admin/load-balancers/:id`.
- **Auto-retry on member failure**: LBProxy retries up to `maxRetries` (default 3) on timeout/5xx, only before first byte for streaming requests.
- **Load Balancers page** in sidebar (sibling of TokenHub) with list + edit views, member picker (Local + Remote tabs), and per-strategy weight input.
- **`/admin/load-balancers/:id/test`** endpoint for invoking an LB with a sample payload and inspecting the chosen candidate trace.
- Localized "Local Inference" + "Load Balancers" sidebar/menu labels across all 9 languages.

### Changed
- Sidebar entry "Models" renamed to **"Local Inference"**.
- `LBStrategy` JSON wire format uses snake_case (`round_robin`, `lowest_latency`) to match the admin API contract; Swift enum case names stay camelCase.

### Removed
- Dead `pickRetryProvider` retry branches from tokenhub passthrough handlers (superseded by LBProxy).
- `provisionLocalProviders()` virtual-provider wrapping (locals are now referenced directly by model_id from LB members).
- Priority-tiered `TokenhubManager.resolve()` LB dispatch path.

### Migration
- On startup, `v4_load_balancers` migration creates the new tables, deletes rows where `is_managed = 1` from `tokenhub_providers`, and drops the legacy columns. Local model files in `~/.nova/models/` are untouched.
- Users who relied on the implicit global LB must create explicit LBs to restore multi-provider routing.

## [1.0.8] - 2026-05-08

### Added
- Pre-emptive memory feasibility check in model list endpoint
- Memory feasibility data per model in admin API

### Fixed
- Prevent double-finish race in SSE keep-alive continuation
- Scheduler concurrency regression tests

## [1.0.7] - 2026-05-07

### Fixed
- Eliminate 4 concurrency races in FusedBatchScheduler
- Add FinishGuard for atomic continuation lifecycle

## [1.0.6] - 2026-05-06

### Fixed
- Preserve `tool_calls` and `tool_call_id` in OpenAI incoming message mapping
- Preserve `tool_use`/`tool_result` blocks in Anthropic message mapping

## [1.0.5] - 2026-05-05

### Added
- Prefix cache: async write + async eviction in SSDCacheStore
- Prefix cache: safetensors header-only reader (replaces full-file scan)
- Prefix cache: skip fetch/store for VLM paths
- Prefix cache: pre-flight RotatingKVCache probe before SSD fetch
- Prefix cache: repeated-prefix TTFT benchmark

### Fixed
- E2E test: skip VLMs in core API suite, accept reasoning-only Harmony output

## [1.0.4] - 2026-05-04

### Added
- Audio transcription (`/v1/audio/transcriptions`) — Qwen3-ASR (Swift/MLX)
- Image generation (`/v1/images/generations`) — SDXL-Turbo
- Modelfile system — user-authored model recipes with system prompt and sampling overrides
- Per-request `keep_alive` — override model TTL per request
- Harmony streaming protocol — GPT-OSS channel-aware format
- ThinkingBudgetProcessor — per-request thinking token budget control
- Strict-FSM JSON logit processor — structured output with JSON schema
- Chat template library — three-level template resolution (user > registry > downloaded)
- `isImplicitThinkingModel` rewrite — auto-detect implicit thinking models at load time
- TokenMaskBuilder cache — pre-decoded vocabulary for fast masking
- VLM LogitProcessor chain — thinking detection for vision-language models
- DeepSeek-V4 lite regression test suite

### Fixed
- ThinkingParser regression tests
- Build script sync for worker binary

## [1.0.3] - 2026-05-02

### Added
- Cloud auth gate with subscription validation
- WebUI dashboard (SPA with status, models, chat pages)
- CLI login/logout/account commands
- GUI settings auth integration

### Changed
- Tagline updated from "fastest" to "blazing fast"

## [1.0.2] - 2026-05-01

### Added
- Homebrew tap distribution (`brew install novamlx`)
- Full OpenAI and Anthropic tools/function calling support
- Dynamic suggested searches from GitHub config
- Generic control token filtering for streaming output
- Agent-aware context scaling (ClientDetector)

### Fixed
- Buffer partial control tokens in streaming to prevent leaked fragments
- CI: patch mlx-swift-lm StrictConcurrency error
- CI: recurse submodules when cloning mlx-swift

## [1.0.1] - 2026-04-30

### Added
- Worker subprocess isolation for crash recovery
- TurnStopProcessor for Qwen3.6 turn separator handling
- ProcessMemoryEnforcer with soft/hard limits
- OCROptimizer for OCR model parameter tuning
- N-gram speculative decoding in FusedBatchScheduler
- Draft-model speculative decoding (SpeculativeTokenIterator)
- Full i18n system — 9 languages
- Web chat with input history and parameter controls
- Settings page with collapsible config.json editor
- Cloud model support — remote inference proxy with streaming

### Changed
- UI overhaul across all views

### Fixed
- Streaming deadlock fix

## [1.0.0] - 2025-04-09

### Added
- Pure Swift SPM project with modular targets
- Paged in-memory KV Cache with LRU eviction
- SSD-backed KV Cache for persistent caching
- Continuous batching for concurrent request processing
- MLX engine with model loading and tokenization
- LLM and VLM inference pipelines
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Anthropic-compatible API (`/v1/messages`)
- SSE streaming support for both API formats
- Model manager with download, registry, and versioning
- Auto-registration of popular MLX community models
- Native macOS menu bar app with SwiftUI
- System monitoring (CPU, memory, GPU)
- Health check and stats endpoints
- Comprehensive test suite
- MIT License

### Performance
- Zero Python dependencies
- Direct Metal acceleration via MLX
- Paged KV Cache with configurable memory limits
- SSD overflow for cold cache entries
- Continuous batching for maximum throughput
