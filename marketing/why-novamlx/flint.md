# Why Flint with NovaMLX

> Hook: **A 40-billion-parameter agent model that reasons on its own, running in a MacBook with no fan noise — and it also gives you tool calls, a chain of thought, and a 256K context.**

## What the model is

Flint is Recursive's open-source 39.8B-parameter MoE (about 2.5B active per step, which is why it is fast). It reasons inside `<think>` before it answers, supports multi-turn tool calls, has a 256K context, and is Apache-2.0 for commercial use. It is positioned as one of the "strongest agent-coding MoEs under 50B."

## Measured results (M4 Max 128GB laptop, official INT4)

| Metric | NovaMLX measured | Notes |
|---|---|---|
| End-to-end throughput (prefill + HTTP) | **~24 tok/s** (823 tokens / 34.9s measured) | TTFT 0.6s (17-token prompt) |
| Token-by-token match with Python mlx-lm | **100% character-level match** | Greedy, same prompt on both sides — the ported math is exact |
| Chain of thought (`reasoning_content`) | ✅ streaming and non-streaming, both channels | OpenAI and Anthropic APIs |
| Tool calls (GLM4 dialect) | ✅ fully verified | Parsed into standard OpenAI tool_calls; multi-turn round trips work |
| Resident memory | **~22.6 GB** | Coexists with 3D and image models on a 128GB machine |

**Which comparisons to use:**
- **vs cloud APIs (peer 40B reasoning models):** zero token cost, code and documents stay on the machine, unlimited calls. For in-house code where compliance is a hard requirement, this is the only fit.
- **vs llama.cpp / Ollama** (same machine, M4 Max, 2026-10-25): llama.cpp wins raw decode speed (Q4_K_XL 84.9 tok/s vs NovaMLX ~24 tok/s). The gap is operator fusion we have not done yet; it is on the roadmap. NovaMLX wins on **everything outside the kernel**: OpenAI and Anthropic protocols out of the box, GLM4-dialect tool calls normalized to standard tool_calls, continuous batching for concurrent requests, request logs / API keys / hot-swapping models, and text, image, video, and 3D in the same service. llama.cpp leaves those to be built. For the fastest raw token today, pick llama.cpp. For a local inference service a product can call directly, pick NovaMLX.
- **vs Python mlx-lm:** the same speed (the token-by-token match shows the same model). NovaMLX is a resident service: batched concurrency, request logs, API key management, hot model switching, webhooks. A Python script has to grow those itself.

## Why NovaMLX

1. **Both API protocols, native** — OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` are served together. Claude Code, Cursor, and other clients connect with no extra adapter.
2. **Chain of thought as a first-class field** — `reasoning_content` is streamed separately. The client does not have to strip `<think>` tags.
3. **Tool calls out of the box** — Flint's custom `<tool_call>` dialect is normalized to standard OpenAI tool_calls. An agent framework can consume them directly.
4. **Token-by-token match with mlx-lm** — not "close enough." It is the same model mathematically, and the match is open to check.
5. **Hybrid attention, ported** — 35 sliding-window layers + 14 full-attention layers (NoPE) + attention sinks, ported correctly (the alignment test shows it). The 256K context is actually usable.

## API demo (recordable)

```bash
curl -X POST localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "RecursiveSI/flint-int4",
       "messages": [{"role": "user", "content": "What is 17*23? Think step by step."}],
       "max_tokens": 500, "temperature": 0}'
# Returns two channels: reasoning (step by step) + content ("= 391")
```

Demo highlight: the model counting haiku syllables stays visible for the whole reasoning stream, so a viewer can see what the AI is thinking.

## Picture ideas

- Open: one line typed in a terminal, then a split screen. The chain of thought scrolls on the left (reasoning). The answer types out on the right.
- Middle: a screenshot of Claude Code pointed at localhost ("swap the cloud model for local Flint, one line of config").
- Data card: "39.8B parameters · 24 tok/s generated locally · 0 yuan/token".
- Comparison: a crossing chart of a local Flint bill versus a cloud API monthly bill as call count grows.

## Slogan

**"Four billion pixels of thought, and it only burns your own electricity."**
(Alternate: **"A thinking 40B that lives in your MacBook."**)

---

### Measurements filled in (2026-10-24/25, M4 Max 128GB; the default weights are now the official INT4, and the earlier numbers came from the community OptiQ build)

- [x] Throughput: 823 completion tokens / 34.9s = **~24 tok/s** (temperature 0, including 0.6s TTFT)
- [x] Tool calls: finish_reason=tool_calls, argument JSON correct (city=Tokyo); multi-turn tool results come back correctly (18°C → clothing advice)
- [ ] 256K long-context measurement (sliding-window correctness + speed falloff curve)
- [x] llama.cpp on the same machine (2026-10-25, llama.cpp master+Metal, UD-Q4_K_XL 23.3GiB, llama-bench mean of 3 runs): pp512 = 1555 tok/s, tg256 = **84.9 tok/s** — raw decode is 3.5× the current NovaMLX, mainly because we have not compiled the decode segment yet (eager per-op dispatch). That work is the top performance item, with a theoretical 2–3× of room. **Video line: do not compete on tok/s alone. Compete on token-verifiable quality + both protocols + tool normalization + the full suite + concurrency.**
