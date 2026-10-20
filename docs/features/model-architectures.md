# Model Architectures

**Pitch:** A `ModelFamily` registry with per-family defaults for KV precision, prefill, and context.

`ModelFamily` 注册表决定家族、默认 KV 精度、prefill 与上下文。聊天模板按家族与格式分发。

## Catalog

`ModelFamily`（`Sources/NovaMLXCore/Types.swift`）：`llama`、`mistral`、`phi`、`qwen`、`gemma`、`starcoder`、`claude`、`bailing`、`gptOss`、`whisper`、`qwen3Asr`、`qwen3Tts`、`dotsTts`、`stableDiffusion`、`flux`、`other`。

**LLM 家族：** Llama 3 / 3.1 / 3.2 / 3.3、Mistral / Mixtral、Qwen 2 / 2.5 / 3 / 3.5 / 3.6、Gemma 2 / 3 / 4、Phi 3.5 / 4、StarCoder2、GPT-OSS（Harmony）、Bailing-Hybrid（Ling 系列 — MLA + GLA + MoE）。

`ModelFamilyRegistry` 中的默认：

| 家族 | KV 精度 | Prefill | 上下文 |
|------|---------|---------|--------|
| Llama / Mistral | 4-bit | 512 | 8192 |
| Phi | 32-bit | 256 | 4096 |
| Qwen | 4-bit | 512 | 8192 |
| Gemma | 32-bit | 256 | 8192 |

**Chat 模板：** `ChatTemplateProcessorRegistry` 按 `(ModelFamily, ChatTemplateFormat)` 路由。家族处理器负责控制令牌、隐式思考检测、停止序列注入。

草稿模型推荐也写在该注册表里。见 [推测解码](speculative-decoding.md)。Nemotron Bolt 文本使用单独的 `nemotron_h_puzzle` 注册，不在上表默认行中。见 [Nemotron 3.5 Bolt 文本](nemotron-bolt-text.md)。

正文来自 `features.zh-CN.md`「模型架构」；标题与 pitch 为英文。
