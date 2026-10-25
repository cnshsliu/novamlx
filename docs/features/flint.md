# Flint（Recursive）

**Pitch:** Recursive 的 40B agent 推理模型——思考链、多轮工具调用、256K 上下文，原生跑在本地 MLX 上。

本地推理 RecursiveSI Flint（39.8B MoE，~2.5B 激活），Swift/MLX 原生移植（无 Python）。与 Python mlx-lm **逐 token 对齐**（greedy 同题字符级一致）。

## Who it is for

要在本地跑带推理和工具调用的 agent/coding 模型、且要求 OpenAI/Anthropic 双协议标准接口的人。

## What it does

| Feature | Details |
|---------|---------|
| **架构** | 49 层：GQA（32q/4kv，head 128）+ per-head QK RMSNorm + learned attention sinks；35 层滑窗(513)+RoPE / 14 层全注意力 NoPE；sigmoid 路由 MoE（6/256 专家 + 无门控共享专家），前 2 层 dense |
| **能力** | `<think>` 推理（`reasoning_content` 独立通道）、多轮工具调用（GLM4 方言自动归一为标准 OpenAI tool_calls）、256K 上下文（滑窗 KV） |
| **API** | OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`，流式/非流式；eos = `<|end_of_text|>` 及角色头 token（自动停止+清洗，不泄漏） |
| **调度** | ContinuousBatcher（连续批处理路径；fused 路径暂无工具解析） |
| **量化** | 官方 `RecursiveSI/flint-int4`（~22.6 GB；社区 OptiQ-4bit 曾用于早期验证，现已弃用为默认） |
| **性能参考** | M4 Max 128GB：裸 decode 基线见 Gate 0 报告（eager 路径 ~20 tok/s 量级，优化进行中）；此前 OptiQ 权重上的端到端 ~34 tok/s 口径已随默认权重切换作废 |

## How to try

```bash
nova download RecursiveSI/flint-int4
# 加载后在 Local Inference 页选择，或：
curl -X POST localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "RecursiveSI/flint-int4",
       "messages": [{"role": "user", "content": "What is 17*23? Think step by step."}],
       "max_tokens": 500, "temperature": 0}'
# → reasoning_content（分步推理）+ content（答案）双通道
```

推荐采样：temperature 1.0、top-p 0.95、top-k 20（官方推荐）；精确复现用 temperature 0。

## Limits / notes

- 工具调用走 ContinuousBatcher 路径（fused 批处理暂不解析工具协议）。
- 纯 text-to-3D 之类与 Flint 无关；Flint 是文本 agent 模型。
- TurboQuant KV 缓存会丢弃 attention sinks（与 GPT-OSS 相同的取舍），默认不启用。

## Related

- [API 兼容 / API compatibility](api-compatibility.md)
- [工具调用 / Tool calling](tool-calling.md)
- `FlintModel.swift`、`FlintProcessor.swift`（`NovaMLXEngine`）
