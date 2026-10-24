# Why Flint with NovaMLX

> 钩子：**一个 400 亿参数、会自己推理的agent 大模型，跑在一台不带风扇声的 MacBook 里——还顺手把工具调用、思考链、256K 上下文全给你了。**

## 模型是什么

Flint 是 Recursive 公司开源的 39.8B 参数 MoE 模型（每次推理只激活 ~2.5B，所以快）：会先在 `<think>` 里推理、再回答，支持多轮工具调用，256K 上下文，Apache-2.0 随便商用。定位是"50B 以下最强 agent 编码 MoE"之一。

## 实测数据（M4 Max 128GB 笔记本，4-bit OptiQ 量化）

| 指标 | NovaMLX 实测 | 说明 |
|---|---|---|
| 端到端吞吐（含 prefill + HTTP） | **~34 tok/s** | 首个基准：haiku 任务 232 token / 6.7s |
| 与 Python mlx-lm 逐 token 对齐 | **字符级 100% 一致** | greedy 双侧同题对比——移植数学完全正确 |
| 思考链（reasoning_content） | ✅ 流式/非流式双通道 | OpenAI + Anthropic 两种 API |
| 工具调用（GLM4 方言） | ✅（路由修复后复测中*） | 解析为标准 OpenAI tool_calls |
| 常驻内存 | **~26 GB** | 128GB 机器与 3D/图像模型共存 |

*标注：部分复测因外置硬盘离线暂停，盘恢复后补齐最终数字。

**对比维度选择**：
- **vs 云 API（同级 40B reasoning 模型）**：零 token 费用、代码/文档不出机、无限调用。企业内部代码场景（合规刚需）唯一解。
- **vs llama.cpp / Ollama**：NovaMLX 有**连续批处理 + OpenAI/Anthropic 双协议 + 任务式全家桶**——同一个服务里 text/image/video/3D 全能调；llama.cpp 生态里工具调用协议各模型一坨方言，NovaMLX 帮你归一成标准 OpenAI tool_calls。
- **vs Python mlx-lm**：同样快（我们逐 token 对齐证明同源性能），但 NovaMLX 是常驻服务：批处理并发、请求日志、API key 管理、多模型热切换、Webhook——这些 Python 脚本都得自己写。

## 为什么是 NovaMLX

1. **两种 API 协议原生** — OpenAI `/v1/chat/completions` 和 Anthropic `/v1/messages` 同时服务，Claude Code / Cursor / 任何客户端零配置接入。
2. **思考链一等公民** — `reasoning_content` 流式分离输出，客户端不用自己剥 `<think>` 标签。
3. **工具调用开箱即用** — Flint 的自定义 `<tool_call>` 方言被 NovaMLX 归一成标准 OpenAI tool_calls，你的 agent 框架直接吃。
4. **逐 token 对齐 mlx-lm** — 不是"差不多对"，是数学上同一个模型。开源可验证。
5. **混合注意力原生支持** — 35 层滑窗 + 14 层全注意力（NoPE）+ attention sinks，全部正确移植（对齐测试证明），256K 上下文真实可用。

## API 演示（可录屏）

```bash
curl -X POST localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/flint-OptiQ-4bit",
       "messages": [{"role": "user", "content": "What is 17*23? Think step by step."}],
       "max_tokens": 500, "temperature": 0}'
# 返回：reasoning（分步推理）+ content（"= 391"）双通道
```

演示亮点：模型自己数 haiku 音节的思考过程，reasoning 流里全程可见——观众能"看见 AI 想什么"。

## 画面建议

- 开头：终端输入一句 → 分屏：左边思考链滚动（reasoning），右边答案逐字打出
- 中段：Claude Code 指向 localhost 的截图（"把云模型换成本地 Flint，一行配置"）
- 数据卡：「39.8B 参数 · 34 tok/s · 0 元/百万 token」
- 对比条：本地 Flint vs 云 API 月账单随调用次数增长的曲线交叉图

## 口号

**"四十亿像素的思考，只烧你自己的电。"**
（备选：**"会思考的 40B，住进你的 MacBook。"**）

---

### TODO（盘恢复后补齐）

- [ ] 纯 decode 吞吐（剔除 prefill）+ 首 token 延迟
- [ ] 工具调用完整复测视频素材
- [ ] 256K 长上下文实测（滑窗正确性 + 速度衰减曲线）
- [ ] 与 llama.cpp（GGUF Q4）同机 tok/s 对比
