# Speculative Decoding

**Pitch:** Draft several tokens ahead — with an n-gram window or a small draft model — and let the target model accept or reject them.

An n-gram window or a small draft model proposes several tokens first. The target model then accepts or rejects them.

## Who it is for

People who already run a target model locally and want to verify more than one token at a time. The n-gram path does not need a separate draft model download.

## What it does

### N-gram

No draft model. N-gram token prediction, with a lookahead window and acceptance sampling.

### Draft models

| Feature | Details |
|---------|---------|
| **DraftModelRegistry** | Injects the recommended draft model for the target model family |
| **Built-in drafts** | Qwen3-0.6B-4bit, Llama-3.2-1B-4bit, Gemma-2-2B-4bit |
| **API** | Request fields `draft_model` and `num_draft_tokens` |
| **EOS suppression** | Only draft-side EOS tokens are filtered, so generation does not stop early |
| **Limits** | Hybrid-attention (Mamba) targets are not supported. Cross-vocabulary drafts are not supported |
| **Status** | `SpecBoostStatus`: `.eligible` / `.active` / `.ineligible`, per model |

Draft recommendations in the family defaults live in `ModelFamilyRegistry`. See [Model architectures](model-architectures.md).

## How to try

The n-gram path does not need a draft model. The catalog has no separate n-gram request field. It only describes the lookahead window and acceptance sampling.

The draft path uses the request fields `draft_model` and `num_draft_tokens`. Built-in drafts are Qwen3-0.6B-4bit, Llama-3.2-1B-4bit, and Gemma-2-2B-4bit. `DraftModelRegistry` injects the recommended draft for the target family. Per-model `SpecBoostStatus` is `.eligible`, `.active`, or `.ineligible`. The catalog does not give a default for `num_draft_tokens`.

### DFlash2 and DSpark: verification length from confidence

By default, DFlash2 and DSpark no longer hand the whole draft block to the target model on every round. The draft model still proposes up to the budget in one shot. The scheduler sends only the prefix with the highest expected throughput for verification.

Each position has a conditional confidence: the probability that this position is also accepted, given that every draft before it was accepted. DSpark uses the confidence head in the checkpoint. DFlash2 has no such head. It uses the gap between the first- and second-place draft scores, mapped through a sigmoid onto 0–1. The prefix's expected accepted length is the sum of the cumulative products of those probabilities. Verification cost is a fixed cost for one target forward, plus a marginal cost per draft token. Past the fast SDPA window (default 4 drafts, which keeps query length at most 5) the marginal cost is higher. High confidence can stretch the prefix out to the draft budget. Low confidence stops before the tail.

The accept rule is unchanged: the target model takes argmax over this prefix, the longest matching draft is kept, and the target token at the first mismatch is appended. Under the same sampling parameters, this rule matches a fixed block on that prefix. Temperature must be 0. Above temperature 0, DFlash2 / DSpark still use ordinary single-token decode.

Turn adaptive off and return to a fixed block with the request field `"adaptive_verify": "fixed"`, or the process environment variable `NOVAMLX_ADAPTIVE_VERIFY=fixed` (`off` / `0` are also accepted). `"adaptive"`, `on`, and `1` turn it on explicitly. When unset, the mode is adaptive.

| Knob | Request field | Environment variable | Default |
|------|---------------|----------------------|---------|
| Mode | `adaptive_verify` | `NOVAMLX_ADAPTIVE_VERIFY` | adaptive |
| Minimum prefix | `adaptive_verify_min` | `NOVAMLX_ADAPTIVE_VERIFY_MIN` | 1 |
| Maximum prefix | `adaptive_verify_max` | `NOVAMLX_ADAPTIVE_VERIFY_MAX` | draft budget (DSpark fixed mode is still the vector SDPA cap of 4) |
| Fixed cost | — | `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST` | 2 |
| Marginal cost inside the window | — | `NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST` | 0.40 |
| Marginal cost outside the window | — | `NOVAMLX_ADAPTIVE_VERIFY_SLOW_MARGINAL_COST` | 0.55 |
| Fast window (draft count) | — | `NOVAMLX_ADAPTIVE_VERIFY_FAST_DRAFTS` | 4 |
| DFlash2 score-gap scale | — | `NOVAMLX_ADAPTIVE_VERIFY_MARGIN_SCALE` | 2 |
| confidence head temperature | — | `NOVAMLX_ADAPTIVE_VERIFY_TEMPERATURE` | 1 |

A larger fixed cost makes the prefix longer, which suits a single generation. Lowering it truncates earlier, which suits the case where verify width itself is expensive. This does not change companion attach: a known-faster companion still attaches automatically, a known-slower one still stays unloaded, and an uncertain one is still asked once at load and remembered.

The scheduled length can be printed without loading a model:

```bash
nova adaptive-verify --confidence 0.92,0.88,0.84,0.70,0.25,0.15,0.10
nova adaptive-verify --mode fixed --confidence 0.92,0.88,0.84,0.70,0.25
```

On a Mac, A/B the same greedy request: set `"temperature": 0` both times, once with `"adaptive_verify": "adaptive"` and once with `"fixed"`. In the log, `draft` is the draft length, `verified` is the prefix sent to the target model, and `accepted` is the number of matching draft tokens. `nova chat <model> --adaptive-verify adaptive` sends temperature 0 and the matching field. `--adaptive-verify fixed` uses the fixed block.

DFlash2 and DSpark schedule how many drafted tokens the target model scores. The default is adaptive. `"adaptive_verify": "fixed"` or `NOVAMLX_ADAPTIVE_VERIFY=fixed` restores the previous block. Acceptance stays the greedy prefix plus the target bonus token, and it still requires temperature 0. Companion attach policy is unchanged.

## Limits / notes

- Hybrid-attention (Mamba) targets are not supported.
- Cross-vocabulary draft models are not supported.
- Draft-side EOS is filtered so the target model is not cut off early.
- Adaptive verification applies only to DFlash2 and DSpark. N-gram and ordinary small draft models still use `num_draft_tokens`.
- When the verify width passes the fast SDPA window, a near-tie argmax can differ from token-by-token decode. That is the existing multi-token verification limit. Fixed mode keeps DSpark inside that window (at most 4 drafts). Adaptive mode crosses it only when confidence is high enough.
- At low concurrency a shorter verify is not necessarily faster. Use `adaptive_verify: fixed` or raise `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST` to get a longer block again. The companion is not forced on because of that.

## Related

- [Inference engine](inference-engine.md)
- [Model architectures](model-architectures.md)
- [Configuration](configuration.md) (engine family defaults include draft recommendations)
