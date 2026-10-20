# Speculative Decoding

**Pitch:** Draft several tokens ahead — with an n-gram window or a small draft model — and let the target model accept or reject them.

用 n-gram 窗口或小草稿模型先提出若干 token，再由目标模型接受或拒绝。

## Who it is for

已经在本地跑目标模型，并希望一次验证多个 token 的人。n-gram 路径不需要另下一个草稿模型。

## What it does

### N-gram

无需草稿模型。通过 n-gram token 预测，配合 lookahead 窗口与接受采样。

### 草稿模型

| 功能 | 详情 |
|------|------|
| **DraftModelRegistry** | 按目标模型家族自动注入推荐草稿模型 |
| **内置草稿** | Qwen3-0.6B-4bit、Llama-3.2-1B-4bit、Gemma-2-2B-4bit |
| **API** | 请求字段 `draft_model` 与 `num_draft_tokens` |
| **EOS 抑制** | 仅草稿侧的 EOS token 被过滤，防止提前停止 |
| **限制** | 不支持 hybrid-attention（Mamba）目标；不支持跨词表草稿 |
| **状态** | `SpecBoostStatus`：`.eligible` / `.active` / `.ineligible`，按模型 |

家族默认里的草稿推荐写在 `ModelFamilyRegistry`。见 [模型架构](model-architectures.md)。

## How to try

n-gram 路径不需要草稿模型。目录没有单独的 n-gram 请求字段，只说明 lookahead 窗口与接受采样。

草稿路径使用请求字段 `draft_model` 与 `num_draft_tokens`。内置草稿为 Qwen3-0.6B-4bit、Llama-3.2-1B-4bit、Gemma-2-2B-4bit。`DraftModelRegistry` 会按目标家族自动注入推荐草稿。按模型的 `SpecBoostStatus` 为 `.eligible`、`.active` 或 `.ineligible`。目录没有给出 `num_draft_tokens` 的默认值。

## Limits / notes

- hybrid-attention（Mamba）目标不支持。
- 跨词表的草稿模型不支持。
- 草稿侧 EOS 会被滤掉，避免目标模型被提前截断。

## Related

- [推理引擎](inference-engine.md)
- [模型架构](model-architectures.md)
- [配置](configuration.md)（引擎按家族默认含草稿推荐）

正文来自 `features.zh-CN.md`「推测解码」（n-gram 与草稿模型两节）；标题与 pitch 为英文。
