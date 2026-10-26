# Adaptive verification length (DFlash2 / DSpark)

## Title

DFlash2 and DSpark: draft confidence decides how far the target model verifies

## One-line pitch

DFlash2 and DSpark verify a confidence-scheduled prefix instead of a fixed draft block, and greedy acceptance is unchanged.

## Who it is for / when to care

People already running DFlash2 or DSpark in NovaMLX on a Mac, where a fixed block still sends the low-confidence tail through a full target-model verify. People comparing accepted length and tok/s against a fixed block can A/B the same switch.

## What shipped (user-visible behavior)

DFlash2 and DSpark now default to an adaptive verification length. The draft model still proposes up to the budget in one shot. The scheduler picks the prefix with the highest expected throughput and sends only that prefix to the target model.

- DSpark uses the checkpoint's own confidence head (sigmoid). If the head is missing, or every confidence is 0, it falls back to the previous fixed length.
- DFlash2 has no confidence head. The gap between the first- and second-place draft scores goes through a sigmoid (default scale 2) to a confidence in 0–1.
- The accept rule is unchanged: the target model takes argmax over this prefix, the longest matching draft is kept, and the target token at the first mismatch is appended. Temperature must be 0. Otherwise both paths stay on ordinary single-token decode.
- The fixed block is still available. Request `"adaptive_verify": "fixed"`, or set the process environment variable `NOVAMLX_ADAPTIVE_VERIFY=fixed` (`off` / `0` also turn it off). When unset, the mode is adaptive.
- Defaults: minimum prefix 1, maximum equal to the draft budget; fixed cost 2; marginal cost 0.40 per draft inside the fast SDPA window and 0.55 outside it; the fast window is 4 drafts. DSpark's fixed mode still stops at that window (at most 4 drafts). Adaptive mode crosses it only when confidence is high enough.
- Companion attach is unchanged: known-faster attaches automatically, known-slower stays unloaded, and uncertain asks once at load and is remembered. Adaptive verification does not turn speculative decoding on forever.
- Each round's log includes `draft` (draft length), `verified` (length actually verified), and `accepted` (draft tokens accepted).

Request fields `adaptive_verify_min` / `adaptive_verify_max` override the minimum and maximum prefix. Cost and window use environment variables: `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST`, `NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST`, `NOVAMLX_ADAPTIVE_VERIFY_SLOW_MARGINAL_COST`, `NOVAMLX_ADAPTIVE_VERIFY_FAST_DRAFTS`, `NOVAMLX_ADAPTIVE_VERIFY_MARGIN_SCALE`, `NOVAMLX_ADAPTIVE_VERIFY_TEMPERATURE`.

The default is adaptive. The fixed block remains available. Companion attach policy is unchanged.

## How to try it

Without loading a model, print the length the scheduler picks:

```bash
nova adaptive-verify --confidence 0.92,0.88,0.84,0.70,0.25,0.15,0.10
nova adaptive-verify --mode fixed --confidence 0.92,0.88,0.84,0.70,0.25
```

On a Mac, A/B a loaded target model with a greedy request. Attach the DFlash2 / DSpark companion with the existing policy. Do not start a second setup. Keep the two requests the same except for `adaptive_verify`, with `temperature` 0.

```bash
nova chat <model-id> --adaptive-verify adaptive
nova chat <model-id> --adaptive-verify fixed
```

Or:

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","temperature":0,"max_tokens":128,"adaptive_verify":"adaptive","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}]}'
```

Replace `adaptive_verify` with `fixed` for the fixed block. Compare `draft`, `verified`, and `accepted` in the log. To turn it off for the process, set `NOVAMLX_ADAPTIVE_VERIFY=fixed` when launching NovaMLX.

## Limits / out of scope

- No Mac GPU throughput comparison was completed on this machine. The schedule has unit tests. End-to-end tok/s needs the A/B above on Apple Silicon.
- `num_draft_tokens` for n-gram or ordinary small draft models is unchanged.
- No new confidence head is trained, and Wide-EP, cluster, and CUDA are not ported.
- When verify width crosses the fast SDPA window, a near-tie argmax can still differ from token-by-token decode. That is the existing multi-token verification limit.
- At low concurrency a shorter verify can be slower. Use `fixed`, or raise `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST`.

## Suggested video beats

- A fixed block also sends the low-confidence draft tail into the target model.
- Adaptive mode verifies only a prefix chosen from confidence. The accept rule matches the fixed block.
- On one temperature 0 request, switch `adaptive` and `fixed`, and read `draft`, `verified`, and `accepted` in the log.
- Whether a companion attaches still follows the existing known-faster / known-slower policy. This switch does not force it on forever.

## Links

- [Speculative decoding](../features/speculative-decoding.md)
- `nova adaptive-verify`
