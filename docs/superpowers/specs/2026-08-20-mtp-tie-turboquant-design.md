# Generic MTP, TIE (large MoE), TurboQuant 8

Date: 2026-08-20

## Scope correction

Not in scope:

- Switching `ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit` to a community MTP quant
- Converting Ornith to TIE `.tiered` (it is small enough to stay eager)
- Exclusive-load / unload-other as a product policy (operator already unloads by hand)

In scope, in this order:

1. **T1 — common MTP** for any checkpoint that *has MTP weights*
2. **T3 — TIE `.tiered` + expert prefetch** for large MoE that need SSD streaming (e.g. DeepSeek-V4-Flash, Qwen3.6-35B-A3B), not Ornith
3. **T4 — persist 8-bit TurboQuant** on the active large model

Measure decode tok/s after each slice. Revert a slice if it is slower.

## T1 Common MTP

Two on-disk shapes count as “has MTP weights”:

| Shape | How to detect | Example |
|---|---|---|
| Split drafter | `model_type == qwen3_5_mtp` and tensors `layers.0.*`, `fc.*`, `pre_fc_norm_*` | `mlx-community/Qwen3.8-27B-MTP-4bit` (~250 MB) |
| In-backbone | weight keys contain `mtp.` | DeepSeek-V4 / Bailing when not stripped |

Configs that only set `mtp_num_hidden_layers` **without** those tensors (current Ornith 8-bit, Qwen3.8-27B-8bit backbone) are **not** MTP-capable. Do not invent a head.

Reference: mlx-vlm `Qwen3_5MTPDraftModel` — concat(norm(embed(token)), norm(hidden)) → `fc` → 1 decoder layer → `lm_head` of the **target** model. The drafter has no `embed_tokens`.

### Behavior

- Register `qwen3_5_mtp` in `LLMTypeRegistry`. Load via LLM factory, never as a VLM.
- Refuse using an MTP repo as the chat `model` id. It is a draft only.
- Pair a loaded backbone (`qwen3_5` / `qwen3_5_moe`) with a same-vocab `*-MTP-*` (or any on-disk `qwen3_5_mtp`) and auto-inject it even if the backbone is hybrid.
- Generate uses an MTP iterator (hidden-state draft + target verify), not `SpeculativeTokenIterator` with a fake second LM.
- Stop dropping `mtp.*` in Qwen3.5 sanitize when those tensors will be consumed as a split drafter or in-model head.
- Vocab must match (Qwen3.8 is 248320; do not pair Qwen3-0.6B).

### Files (T1)

- `mlx-swift-lm/Libraries/MLXLLM/Models/Qwen35MTP.swift` — drafter
- `mlx-swift-lm/Libraries/MLXLMCommon/` — `MtpTarget` / `MtpDrafter` + iterator
- `LLMModelFactory` register `qwen3_5_mtp`
- `DraftModelRegistry` MTP pairing
- `InferenceService.autoInjectDraftModel` — allow hybrid when draft is MTP
- `MLXEngine` generate/stream — MTP iterator when draft implements `MtpDrafter`
- Tests: sanitize, config, pairing, refuse-as-chat

## T3 TIE (after T1)

- Keep existing `TieredOffloadPolicy.bindIfTiered` + expert prefetch.
- Convert only large MoE that are not fully resident: `Qwen3.6-35B-A3B-4bit` if needed; `DeepSeek-V4-Flash-4bit.tiered` already exists — verify prefetch on load.
- Do not convert Ornith.

## T4 TurboQuant 8 (after T1/T3)

- Persist `ModelSettings.kvBits = 8`, `kvGroupSize = 64` for the active large model id.
- `TurboQuantService.applyToGenerateParameters` already honors settings. Not a global default.

## Success

- `nova load mlx-community/Qwen3.8-27B-8bit` with MTP-4bit on disk auto-loads the drafter and decode uses MTP (log: `MTP draft`).
- `nova load …-MTP-4bit` as the only model is rejected for chat.
- Ornith 8-bit is unchanged (no TIE, no fake MTP).
- 8-bit KV is on for the active large model after T4.
