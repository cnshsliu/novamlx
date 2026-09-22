# DFlash2 one-wait block

Date: 2026-09-22

## Goal

Make one DFlash2 block wait for the GPU once. A short greedy reply on this Mac must not be slower than DFlash2 off. A longer prompt's decode must be faster than DFlash2 off. The accepted tokens stay the same.

## Why

`DFlashCandidateSelector.walkGreedy` calls `.item()` twice per guessed token. `DFlashTokenIterator.speculateRound` then calls `.item()` once per checked token. Each call waits for the GPU. On Qwen3.8-27B that made short replies slower with the helper on (about 22 tokens/s) than with it off (about 30 tokens/s).

## Out of scope

- New Metal kernels
- MLX `compile` around the block
- A new model format
- Sampling changes (`speculateRound` stays greedy argmax)
- Changes to `DFlashAccept.greedy` or to `dflashRollback`'s math
- Prompt-cap or Gated DeltaNet capture changes

## Files

| File | Change |
|---|---|
| `mlx-swift-lm/Libraries/MLXLLM/Models/DFlash.swift` | `walkGreedy` stays on the GPU |
| `mlx-swift-lm/Libraries/MLXLLM/DFlashTokenIterator.swift` | `speculateRound` waits once |
| `mlx-swift-lm/Libraries/MLXLMCommon/DFlashSupport.swift` | `DFlashBlockCheck.decide` |
| `Tests/NovaMLXEngineTests/DFlashTests.swift` | Length-check tests; existing walk test still passes |

`mlx-swift-lm` is the package `Package.swift` already depends on. In this worktree that directory is absent. The sources are at `/Users/lucas/dev/novamlx/mlx-swift-lm`. Implementation edits that checkout.

## Data flow

For each new block:

1. The helper runs. `walkGreedy` walks the path on the GPU and returns guessed token ids still tied to the helper's scores. It does not call `.item()` or `eval`.
2. The anchor token (the last accepted token, a Swift integer) is joined to those ids on the GPU. The big model checks that block, still on the GPU.
3. One `eval` realizes three arrays: the check scores (logits), the hidden state the next block needs (`fused`), and the drafted ids.
4. On the CPU, the existing per-token logit processor runs on the realized rows, then `DFlashAccept.greedy` decides the match. `dflashRollback` drops the rest.

`walkGreedyFollowsChain` still expects `[12, 20, 31]` after that single `eval`. The first step of the walk is `scores[0, 0]`. Each later step indexes the previous winner with `take` and `argMax`, and the index stays an `MLXArray`.

## Failure behavior

`DFlashBlockCheck.decide(draftCount:logitLength:fusedLength:accepted:)` returns:

| Fields agree? | `acceptNormally` | `specWidth` | `nRejected` |
|---|---|---|---|
| Yes: `logitLength == fusedLength == draftCount + 1` and `accepted` is in `0...draftCount` | true | `draftCount + 1` | `draftCount - accepted` |
| No | false | `draftCount + 1` | `draftCount` |

`specWidth` is the token width submitted to verify. That is the width the cache update wrote. `draftCount` is the realized draft-id length, not the cap.

When the decision is normal, rollback uses those two numbers and the round commits `DFlashAccept.greedy`'s tokens. `pendingCtx` is the fused prefix of length `accepted + 1`.

When the decision is not normal, rollback drops every drafted token and keeps the anchor. The round adds no output. `next()` returns nil. The iterator does not try another width and does not fall back to the per-token `.item()` walk.

An empty block is unchanged: when no tokens remain, the round returns before the helper runs. `decide` is not called.

If building the GPU walk fails before `eval`, the cache has not been written, so there is no rollback. The iterator stops.

If the one `eval` throws, do not catch it and do not call rollback with a guessed width. The request fails. This generation is not written to the prefix cache.

`DFlashTokenIterator` failing to start still falls back to ordinary one-token decode. That fallback is unchanged and is not used after a block has started.

## Tests

### Gate 1 — same tokens, no model weights

Run `swift test --filter DFlash`.

- `walkGreedyFollowsChain` returns `[12, 20, 31]` after one `eval`.
- `greedyAcceptPrefix` stays as it is.
- `DFlashBlockCheck` tests: matching lengths with `accepted == 2` and `draftCount == 4` give `acceptNormally`, `specWidth == 5`, `nRejected == 2`. A logit length that differs gives `acceptNormally == false`, `specWidth == draftCount + 1`, `nRejected == draftCount`. An `accepted` value outside `0...draftCount` does the same.

### Gate 2 — speed on this Mac

Qwen3.8-27B with companion `incoai/Qwen3.8-27B-DFlash2`. One checkpoint id for every run. Record the id. Greedy (`temperature` 0), `thinking_budget` 0, `max_tokens` 128.

Decode tokens/s = completion tokens / (time of last token − time of first token). Three runs with the helper off and three with it on, interleaved (off, on, off, on, off, on). Compare medians.

- Short. User message: `Explain how an apple grows, in plain sentences.` Median with the helper is greater than or equal to the median without it. The completion text matches the helper-off run.
- Long. Repeat the sentence `A repository agent reads source files, edits them, and runs the tests before it answers.` until the tokenized prompt is at least 2048 tokens and under 4096. Median decode with the helper is greater than the median without it. This measures decode, not time to first token.

Keep the change only when both gates pass. If the short median is slower, or the long median is not faster, revert the block change.
