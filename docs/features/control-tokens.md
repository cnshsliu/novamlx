# Control Tokens & Thinking Filter

**Pitch:** Strip protocol stop tokens, and keep or flush `<think>` content according to the model.

按模型过滤协议停止符，并处理隐式 `<think>` 与 thinking budget。语义标签保留，协议 token 去掉。

## Catalog

| 子系统 | 行为 |
|--------|------|
| **TurnStopProcessor** | 按模型定义停止 token 集（`<\|turn\|>`、`<\|end\|>` 等）；channel-thinking 模型排除 channel token |
| **ThinkingParser** | 隐式 `<think>` 标签处理；显式 thinking_budget 透传 |
| **语义 vs. 协议 token** | 语义标签（`<think>`）透传；协议 token（`<\|turn\|>`）过滤 |
| **按模型检测** | 每个 model ID 有 `isImplicitThinkingModel` 标记 |
| **stop 前刷新** | 在发出 stop chunk 前先 flush 解析出的思考（Qwen3.6 流式修复） |
| **Channel 感知** | GPT-OSS Harmony 的 `<\|channel\|>` token 排除出停止集合 |

按模型还可配置 `enableThinking`、`thinkingBudget`、`preserveThinking`。见 [按模型配置](per-model-settings.md)。

正文来自 `features.zh-CN.md`「控制令牌与思考过滤」；标题与 pitch 为英文。
