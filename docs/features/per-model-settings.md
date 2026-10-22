# Per-Model Settings

**Pitch:** Override sampling, KV precision, context length, TurboQuant, vision, and thinking per model id.

每个模型 ID 一份持久设置，请求时盖过默认值。

## Catalog

按 model ID 保存，请求时覆盖默认：

- 采样默认：temperature、maxTokens、topP、repeatPenalty
- KV 精度：4-bit / 8-bit / 32-bit
- 上下文窗口
- TurboQuant 开关
- 视觉策略
- 配套视觉模型
- 思考：`enableThinking`、`thinkingBudget`、`preserveThinking`
- Keep-alive 间隔
- 盯住（Pin）：重启仍加载，关掉之前不能卸载

存储在 SQLite `model_settings`。见 [模型管理](model-management.md)。家族级默认（未单独覆盖时）见 [模型架构](model-architectures.md)。

正文来自 `features.zh-CN.md`「按模型独立配置」；标题与 pitch 为英文。
