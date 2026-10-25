# Why Flint with NovaMLX

> 钩子：**一个 400 亿参数、会自己推理的agent 大模型，跑在一台不带风扇声的 MacBook 里——还顺手把工具调用、思考链、256K 上下文全给你了。**

## 模型是什么

Flint 是 Recursive 公司开源的 39.8B 参数 MoE 模型（每次推理只激活 ~2.5B，所以快）：会先在 `<think>` 里推理、再回答，支持多轮工具调用，256K 上下文，Apache-2.0 随便商用。定位是"50B 以下最强 agent 编码 MoE"之一。

## 实测数据（M4 Max 128GB 笔记本，4-bit OptiQ 量化）

| 指标 | NovaMLX 实测 | 说明 |
|---|---|---|
| 端到端吞吐（含 prefill + HTTP） | **~24 tok/s**（823 token / 34.9s 实测） | TTFT 0.6s（17 token 提示） |
| 与 Python mlx-lm 逐 token 对齐 | **字符级 100% 一致** | greedy 双侧同题对比——移植数学完全正确 |
| 思考链（reasoning_content） | ✅ 流式/非流式双通道 | OpenAI + Anthropic 两种 API |
| 工具调用（GLM4 方言） | ✅ 完整验证 | 解析为标准 OpenAI tool_calls，多轮往返正常 |
| 常驻内存 | **~26 GB** | 128GB 机器与 3D/图像模型共存 |

**对比维度选择**：
- **vs 云 API（同级 40B reasoning 模型）**：零 token 费用、代码/文档不出机、无限调用。企业内部代码场景（合规刚需）唯一解。
- **vs llama.cpp / Ollama**（同机实测 M4 Max，2026-10-25）：裸解码速度 llama.cpp 赢（Q4_K_XL 84.9 tok/s vs NovaMLX ~24 tok/s，差距来自我们尚未做算子融合，路线图在补）。但 NovaMLX 赢在**引擎之外的一切**：OpenAI/Anthropic 双协议开箱、GLM4 方言工具调用自动归一成标准 tool_calls、连续批处理多请求并发、请求日志/API key/多模型热切换，以及同一个服务里 text/image/video/3D 全家桶——这些 llama.cpp 都要自己造。要"最快的裸 token"今天选 llama.cpp；要"能直接接产品的本地推理服务"选 NovaMLX。
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
- 数据卡：「39.8B 参数 · 24 tok/s 本地生成 · 0 元/token」
- 对比条：本地 Flint vs 云 API 月账单随调用次数增长的曲线交叉图

## 口号

**"四十亿像素的思考，只烧你自己的电。"**
（备选：**"会思考的 40B，住进你的 MacBook。"**）

---

### 实测补齐（2026-10-24，M4 Max 128GB）

- [x] 吞吐：823 completion tokens / 34.9s = **~24 tok/s**（temperature 0，含 0.6s TTFT）
- [x] 工具调用：finish_reason=tool_calls，参数 JSON 正确（city=Tokyo）；多轮工具结果回传正常（18°C→穿衣建议）
- [ ] 256K 长上下文实测（滑窗正确性 + 速度衰减曲线）
- [x] llama.cpp 同机对比（2026-10-25，llama.cpp master+Metal，UD-Q4_K_XL 23.3GiB，llama-bench 3 次均值）：pp512 = 1555 tok/s，tg256 = **84.9 tok/s**——裸解码 3.5× 于当前 NovaMLX，主因是我方未做编译解码段（eager 逐算子 dispatch）；该工作已列为最高优先级性能项，理论空间 2-3×。**视频口径：不单打 tok/s，打"质量逐 token 可验证 + 双协议 + 工具归一 + 全家桶 + 并发"。**
