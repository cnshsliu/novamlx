# Laya-MLX on NovaMLX

Laya-MLX is a local typed-decision runtime, not a chat model. NovaMLX runs the multilingual checkpoint natively in Swift.

## What it is

Source article: https://aiidelist.com/blog/what-is-laya-mlx  
Port: https://github.com/mizorewww/laya-mlx (Apache-2.0, independent of Convai Innovations)  
Default weights: `aac6fef/laya-multilingual-mlx` (mmBERT, about 322M, 1024-token context, FP16)

```
state + typed questions
        ↓
bidirectional encoder
        ↓
decision head + scorer + action head
        ↓
choice / score / noul probabilities
```

There is no token decoding. A question is one forward pass.

| Kind | Result |
|---|---|
| `choice` | One label plus a probability for each option |
| `score` | Expected zero-based rubric level, plus per-level probabilities |
| `noul` | `P(true)` for a proposition |

English-only `aac6fef/laya-mlx` (421M, 512 context) is the same architecture. NovaMLX catalogs the multilingual checkpoint because the local workload is Chinese as well as English.

Published M3 Max numbers (load excluded): multilingual short question P50 7.39 ms, English 13.42 ms. Those are decision latencies, not an application SLA.

## What NovaMLX does

- `LayaDecisionModel` in `Sources/NovaMLXEngine/Laya/` follows the MLX port: ModernBERT local/global attention, decision Transformer, scorer, action head.
- Prompt layout matches upstream: `[CLS] type instructions [SEP] [MASK] option ... [SEP] state [SEP]`.
- `DecisionService` loads a checkpoint that contains `rl_agent_config.json`, `encoder/config.json`, `tokenizer/`, and `model.safetensors`.
- Loading a decision model does not exclusive-unload a chat LLM. A chat load does not unload Laya.
- `POST /v1/decisions` with Bearer auth. Auto-load works when the model is downloaded.

Request:

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "state": "发票被重复扣款，请退款。",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this?",
      "criteria": ["billing", "technical", "sales"]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the customer ask for money back?"
    }
  }
}
```

`choice` criteria must be a list of unique labels so option order is stable. `score` criteria is an ordered list of level names. `noul` may omit criteria.

Response `answers` includes `confidence`, `probabilities`, and `action.act_probability`. `usage.output_tokens` is 0.

## Limits

- Context is 1024 tokens for the multilingual checkpoint, shared by instructions, options, and state.
- A probability is not permission. Thresholds belong in application code and need a labeled set.
- This does not write text, code, or summaries. Keep Qwen / DeepSeek for generation.
- Choice criteria sent as a JSON object is rejected. Use a list.
- Weight parity with the Python port is structural (same module names and prompt). A numeric gold test against `laya-mlx` on a downloaded checkpoint is still the right next check after the first local load.
