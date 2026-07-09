# NovaMLX — 功能参考

> **面向 Apple Silicon 的生产级纯 Swift LLM / VLM / 音频 / 图像推理服务器。**
> 兼容 OpenAI、Anthropic、Responses 三套 API。原生 macOS 菜单栏应用。基于 MLX 构建。支持多节点分布式推理、云模型代理、负载均衡，以及完整的 API Key 管理体系。

---

## 目录

- [推理引擎](#推理引擎)
- [Worker 子进程隔离](#worker-子进程隔离)
- [多模态支持](#多模态支持)
- [模型架构](#模型架构)
- [API 兼容性](#api-兼容性)
  - [OpenAI 兼容（端口 6590）](#openai-兼容端口-6590)
  - [Anthropic 兼容](#anthropic-兼容)
  - [Responses API](#responses-api)
  - [管理端口（6591）](#管理端口6591)
- [结构化输出](#结构化输出)
- [工具调用](#工具调用)
- [控制令牌与思考过滤](#控制令牌与思考过滤)
- [KV Cache 与前缀共享](#kv-cache-与前缀共享)
- [TurboQuant KV 压缩](#turboquant-kv-压缩)
- [连续批处理](#连续批处理)
- [推测解码](#推测解码)
  - [N-gram 推测解码](#n-gram-推测解码)
  - [草稿模型推测解码](#草稿模型推测解码)
- [分布式推理](#分布式推理)
- [音频（ASR / TTS / 语音克隆）](#音频asr--tts--语音克隆)
- [图像生成](#图像生成)
- [嵌入与重排](#嵌入与重排)
- [会话管理](#会话管理)
- [MCP — 模型上下文协议](#mcp--模型上下文协议)
- [Agent 集成](#agent-集成)
- [Tokenhub 云模型代理](#tokenhub-云模型代理)
- [Anthropic↔OpenAI 翻译桥](#anthropicopenai-翻译桥)
- [负载均衡器](#负载均衡器)
- [API Key 管理](#api-key-管理)
- [模型管理](#模型管理)
- [HuggingFace 集成](#huggingface-集成)
- [Modelfile 系统](#modelfile-系统)
- [内存管理](#内存管理)
- [按模型独立配置](#按模型独立配置)
- [可观测性与基准测试](#可观测性与基准测试)
- [macOS 菜单栏应用](#macos-菜单栏应用)
- [国际化](#国际化)
- [安全与中间件](#安全与中间件)
- [Homebrew 分发](#homebrew-分发)
- [TCC 监控器](#tcc-监控器)
- [配置](#配置)
- [架构总览](#架构总览)
- [快速开始](#快速开始)

---

## 推理引擎

NovaMLX 通过 MLX 直接在 Apple Silicon GPU 上运行 LLM、VLM、音频与图像推理，完全脱离 Python 或任何远端服务。

| 功能 | 详情 |
|---------|---------|
| **后端** | MLX（Apple Silicon GPU）、惰性求值、统一内存 |
| **模型格式** | SafeTensors — 4-bit、8-bit、FP16、NVFP4 预量化 |
| **采样参数** | `temperature`、`top_p`、`top_k`、`min_p`、`frequency_penalty`、`presence_penalty`、`repetition_penalty`、`seed` |
| **停止控制** | 停止字符串、停止 token ID、max_tokens 上限 |
| **流式输出** | OpenAI/Anthropic/Responses 使用 SSE；音频/图像使用原始字节流 |
| **分阶段求值** | 分批 `eval` 防止大模型（>500 数组，如 Ling-2.6-flash 61 GB）出现 Metal OOM |
| **CompiledSampler** | 预编译采样图，加速热路径 token 选择 |
| **MLXEngine.perform** | 线程隔离的 `model()+eval()+sample()` 包裹器，保证并发安全 |

## Worker 子进程隔离

每个推理模型运行在独立的 Worker 子进程中，由 `WorkerSupervisor` 监管（`Sources/NovaMLXInference/WorkerSupervisor.swift`）。

| 功能 | 详情 |
|---------|---------|
| **进程隔离** | 单个模型崩溃不会拖垮整个服务 |
| **自动重启** | 崩溃恢复（2 秒冷却），请求自动重路由 |
| **内存统计** | 跟踪每个 Worker 的当前 / soft / hard RSS 上限 |
| **GPU 内存** | 实时 Metal 分配器压力报告 |
| **请求追踪** | 飞行中请求账目，崩溃时自动清理 |
| **进度回调** | 分阶段的模型加载进度（init / weights / kv-cache / ready） |

## 多模态支持

| 模态 | 支持架构 | 备注 |
|----------|---------------|-------|
| **视觉（VLM）** | Llava、LlavaNext、LlavaQwen2、Qwen2VL、Qwen2.5VL、Qwen3VL、Mllama、Gemma3、Gemma4、InternVLChat、Idefics3、PaliGemma、Phi3V、Pixtral、Molmo、Florence2、Mistral3 | 3D-mRoPE 状态需走 `ContinuousBatcher`（不能用 fused-decode） |
| **音频 ASR** | Whisper、Qwen3-ASR | 48 kHz Core Audio 录音 |
| **音频 TTS** | Qwen3-TTS（旧）、Dots TTS（现行） | Dots 支持语音克隆 |
| **图像生成** | FLUX.1（4-bit、8-bit、FP16） | 内嵌 `flux.swift` |
| **嵌入** | BERT、XLM-Roberta、ModernBert、Qwen3-ForTextEmbedding、Siglip、NomicBert | 池化：mean / cls / last-token |
| **重排** | 交叉编码器 reranker | 对候选段落重新打分排序 |

## 模型架构

`ModelFamily` 注册表（`Sources/NovaMLXCore/Types.swift`）：`llama`、`mistral`、`phi`、`qwen`、`gemma`、`starcoder`、`claude`、`bailing`、`gptOss`、`whisper`、`qwen3Asr`、`qwen3Tts`、`dotsTts`、`stableDiffusion`、`flux`、`other`。

**LLM 家族包括：** Llama 3 / 3.1 / 3.2 / 3.3、Mistral / Mixtral、Qwen 2 / 2.5 / 3 / 3.5 / 3.6、Gemma 2 / 3 / 4、Phi 3.5 / 4、StarCoder2、GPT-OSS（Harmony）、Bailing-Hybrid（Ling 系列 — MLA + GLA + MoE）。

**按家族默认配置**（`ModelFamilyRegistry`）：

| 家族 | KV 精度 | Prefill | 上下文 |
|--------|--------------|---------|---------|
| Llama / Mistral | 4-bit | 512 | 8192 |
| Phi | 32-bit | 256 | 4096 |
| Qwen | 4-bit | 512 | 8192 |
| Gemma | 32-bit | 256 | 8192 |

**Chat 模板处理器** — `ChatTemplateProcessorRegistry` 按 `(ModelFamily, ChatTemplateFormat)` 元组路由。家族特定处理器负责控制令牌、隐式思考检测、停止序列注入。

## API 兼容性

三种线上协议共用同一套推理引擎与工具调用层。

### OpenAI 兼容（端口 6590）

| 端点 | 方法 | 备注 |
|----------|--------|-------|
| `/v1/models` | GET | 列出已加载 + 已下载模型 |
| `/v1/models/{id}` | GET | 模型元数据 |
| `/v1/chat/completions` | POST | 流式 + 非流式 |
| `/v1/completions` | POST | 旧式文本补全 |
| `/v1/embeddings` | POST | 文本转向量 |
| `/v1/audio/transcriptions` | POST | Whisper / Qwen3-ASR |
| `/v1/audio/speech` | POST | Dots / Qwen3-TTS |
| `/v1/images/generations` | POST | FLUX.1 |
| `/v1/images/edits` | POST | FLUX.1 带遮罩 |
| `/v1/images/variations` | POST | FLUX.1 变体 |
| `/v1/rerank` | POST | 交叉编码器重排 |

### Anthropic 兼容

| 端点 | 方法 | 备注 |
|----------|--------|-------|
| `/v1/messages` | POST | 流式 + 非流式；支持 `messages`、`system`、`tools`、`tool_choice`、`thinking_budget` |
| `/v1/messages/count_tokens` | POST | 预先 token 计数 |

鉴权：`x-api-key` 或 `Authorization: Bearer`。需带 `anthropic-version: 2023-06-01` 头。

### Responses API

与 Codex 兼容的 `/v1/responses` 实现。

| 功能 | 详情 |
|---------|---------|
| **端点** | `POST /v1/responses`（流式 + 非流式）、`GET/DELETE /v1/responses/{id}`、`POST /v1/responses/{id}/cancel` |
| **P0** | `tool_choice` 透传（auto / required / none / 指定工具） |
| **P1** | 17 字段响应回显（status、output、usage、completed_at 等） |
| **P2** | SSE `seq` 字段，便于客户端断点检测 |
| **ConversationStore** | `previous_response_id` 解析已存储的对话历史 |
| **Reasoning 别名** | `reasoning.effort` → `thinking_budget` |
| **Compact** | `POST /v1/responses/compact` 截断存储的对话 |
| **Input tokens** | `POST /v1/responses/input_tokens` 精确 tokenization 计数 |

### 管理端口（6591）

通过 admin key 走 Bearer 鉴权。完整端点清单：

- **Models** — 列表 / 详情 / 下载 / 取消 / 加载 / 卸载 / forget / benchmark / perplexity / 缓存统计
- **API Keys** — CRUD、轮换、用量统计、按 key 指标
- **Tokenhub providers** — CRUD、测试、指标
- **Load balancers** — CRUD、成员管理、试跑路由
- **Modelfiles** — 自定义模型定义的 CRUD
- **Stats / Info / Reset** — 服务指标、系统信息、重置
- **Log level** — 运行时日志级别控制

## 结构化输出

| 模式 | 线上字段 | 备注 |
|------|------------|-------|
| **JSON object** | `response_format: {type: "json_object"}` | 强制输出合法 JSON |
| **JSON schema** | `response_format: {type: "json_schema", json_schema: {schema: ...}}` | 严格 schema 约束生成 |
| **Regex** | `response_format: {type: "regex", regex: "..."}` | 正则锚定输出 |
| **GBNF 语法** | `response_format: {type: "gbnf", gbnf: "..."}` | llama.cpp 语法格式 |
| **Choice / Enum** | 通过 schema `enum` | 枚举单选约束 |

## 工具调用

| 格式 | 字段 | 响应结构 |
|--------|-------|----------------|
| **OpenAI** | `tools: [{type:"function", function:{name, parameters}}]` | `choices[0].message.tool_calls[]` |
| **Anthropic** | `tools: [{name, description, input_schema}]` | `content[].tool_use` 块 |
| **Responses** | 通过 modelfile | 正则 + JSON 解析提取 |

`tool_choice`：`auto` / `required` / `none` / 指定工具 / `any`（Anthropic）。跨格式双向翻译会保留 tool id、name、arguments。

## 控制令牌与思考过滤

| 子系统 | 行为 |
|-----------|----------|
| **TurnStopProcessor** | 按模型定义停止 token 集（`<\|turn\|>`、`<\|end\|>` 等）；channel-thinking 模型排除 channel token |
| **ThinkingParser** | 隐式 `<think>` 标签处理；显式 thinking_budget 透传 |
| **语义 vs. 协议 token** | 语义标签（`<think>`）透传；协议 token（`<\|turn\|>`）过滤 |
| **按模型检测** | 每个 model ID 有 `isImplicitThinkingModel` 标记 |
| **stop 前刷新** | 在发出 stop chunk 前先 flush 解析出的思考（Qwen3.6 流式修复） |
| **Channel 感知** | GPT-OSS Harmony 的 `<\|channel\|>` token 排除出停止集合 |

## KV Cache 与前缀共享

| 功能 | 详情 |
|---------|---------|
| **SSD 缓存存储** | 持久化跨会话 KV cache，位于 `~/.nova/cache/` |
| **Block 哈希** | 内容寻址的 block，通过 `BlockHasher` |
| **分页 block 池** | 通过 `PagedBlockPool` 高效管理内存 |
| **启动自动加载** | 为热模型重建缓存（hybrid 模型当前禁用，因 Mamba+KV 混合） |
| **命中/缺失追踪** | 通过 `/admin/models/{id}/cache` 暴露每模型缓存统计 |
| **Session 复用** | 请求里的 `session_id` 字段把请求钉到对应 KV-cache 谱系；同一 session = 同一份 KV 跨请求复用。OpenAI / Anthropic / Responses 三套 API 都支持。 |

### 澄清：Anthropic `cache_control` 字段为什么不被解析

Anthropic 的 `cache_control: { type: "ephemeral" }` 字段是为**计费**设计的——告诉 Anthropic 云端哪些内容块要按 cache write 价格计费。NovaMLX 是**没有计费层的本地推理服务器**，这个字段对我们没意义：KV cache 在每次共享前缀的请求里都会自动复用。

我们按路由路径处理 `cache_control`：

| 路径 | 行为 |
|------|------|
| **本地推理**（`/v1/messages` → MLX） | Codable 解码时静默丢弃（`AnthropicContentBlock` 没有这个字段）。KV cache 仍然通过前缀匹配 + `session_id` 自动复用。**客户端无需做任何事。** |
| **`tknet:` / `lb:` → 原生 Anthropic 上游**（raw passthrough） | body 原样保留。`anthropic-version` header 总是转发；`anthropic-beta` header 客户端传了就转发（保证 1 小时 cache TTL 真到达 provider，不会静默降级成 5 分钟）。 |
| **`tknet:` / `lb:` → OpenAI 格式上游**（Anthropic↔OpenAI 翻译桥） | 翻译时丢弃（OpenAI chat/completions 没有对应概念）。服务器会打 `[WARN] [TokenhubBridge]` 日志，让运维知道发生了静默降级。 |

## TurboQuant KV 压缩

4-bit 仿射量化 KV cache，支持动态 group size。

| 功能 | 详情 |
|---------|---------|
| **压缩** | 对 K、V 张量做 4-bit 仿射量化 |
| **group size** | 按 head dim 动态决定每层 group size |
| **透明性** | 在采样层之下运作，sampler 无感知 |
| **质量权衡** | KV 内存减少约 75%，长上下文质量损失极小 |

## 连续批处理

| 功能 | 详情 |
|---------|---------|
| **默认 batch** | 8 序列（可配） |
| **抢占** | 内存压力下按优先级驱逐 |
| **专用队列** | VLM、hybrid-attention、会话、语法引导分别走独立路径 |
| **异步流** | 每请求按优先级队列发射 token |
| **指标** | `BatcherMetrics` 暴露队列深度、抢占率、平均等待 |

## 推测解码

### N-gram 推测解码

无需草稿模型的免费推测解码，通过 n-gram token 预测 + 接受采样实现。

### 草稿模型推测解码

| 功能 | 详情 |
|---------|---------|
| **DraftModelRegistry** | 按目标模型家族自动注入推荐草稿模型 |
| **内置草稿** | Qwen3-0.6B-4bit、Llama-3.2-1B-4bit、Gemma-2-2B-4bit |
| **API** | `draft_model` + `num_draft_tokens` 请求字段 |
| **EOS 抑制** | 仅草稿侧的 EOS token 被过滤，防止提前停止 |
| **限制** | 不支持 hybrid-attention（Mamba）目标；不支持跨词表草稿 |
| **状态 API** | `SpecBoostStatus`：`.eligible` / `.active` / `.ineligible`，按模型 |

## 分布式推理

通过 TCP / Thunderbolt 在多台 Apple Silicon 节点间做 pipeline-parallel 切片。

| 功能 | 详情 |
|---------|---------|
| **传输层** | 数据面走原始 TCP 二进制；控制面按规模自适应 |
| **切片策略** | `SlicedForwardPolicy` — 基于反射的层级切片 |
| **ShardEngine** | 协调每节点前向传播 |
| **ClusterModelManager** | 全集群激活/去激活；`ClusterModelState`（idle / activating / ready / failed） |
| **远端采样** | Worker 侧做 argmax；只回传 4 字节 token ID，不回完整 logits |
| **WorkerSupervisor** | 每节点生命周期、心跳/健康追踪 |
| **自动降级** | 集群失败时回落到本地推理 |
| **已编译后端** | Ring（TCP）已启用；JACCL（RDMA）在 `rdma_ctl` flag 后待启用 |
| **实测性能** | coord 31.8 ms、worker 33.7 ms、TCP 9 ms、tokenizer 0.3 ms；顺序上限约 14 tok/s；远端采样基线 13.8 tok/s |
| **真实场景** | Qwen3.6-27B 跨 M4 Max + M4 Mac Mini 经 Thunderbolt → 1.8 tok/s pipeline-parallel |

## 音频（ASR / TTS / 语音克隆）

| 功能 | 详情 |
|---------|---------|
| **Whisper ASR** | 48 kHz 录音、语言自动检测、`TranscriptionContainer` 热切换 |
| **Qwen3-ASR** | 备选 ASR 后端，复用 `/v1/audio/transcriptions` 入口 |
| **Dots TTS** | 内嵌 `mlx-swift-dots-tts`；`DotsTTSPipeline` 神经语音 |
| **Qwen3-TTS** | 旧路径，已被 Dots 取代 |
| **系统语音** | macOS `NSSpeechSynthesizer` 兜底 |
| **语音克隆** | `VoiceProfile` 管理器位于 `~/.nova/voices/`；多说话人；参考音频克隆 |
| **VoiceCloneSheet UI** | 录音/选取参考音、预览、保存为命名 profile |
| **麦克风权限** | `audio-input` entitlement + `NSMicrophoneUsageDescription`，由 `build.sh` 注入 |

## 图像生成

| 功能 | 详情 |
|---------|---------|
| **模型** | FLUX.1（4-bit、8-bit、FP16） |
| **Pipeline** | 内嵌 `flux.swift` 的 `FluxPipeline` |
| **容器** | `ImageGenerationContainer` 支持热加载/卸载 |
| **端点** | `/v1/images/generations`、`/v1/images/edits`、`/1/images/variations` |
| **输出** | Base64 PNG；可配 height / width / steps / guidance |
| **服务** | `ImageGenerationService` async API |

## 嵌入与重排

| 功能 | 详情 |
|---------|---------|
| **EmbeddingContainer** | 可热加载的嵌入模型 |
| **架构** | BERT、XLM-Roberta、ModernBert、Qwen3-ForTextEmbedding、Siglip、NomicBert |
| **池化** | mean / cls / last-token |
| **端点** | `POST /v1/embeddings` |
| **RerankerContainer** | 交叉编码器 reranker |
| **端点** | `POST /v1/rerank`（候选 → 打分后的候选） |

## 会话管理

| 功能 | 详情 |
|---------|---------|
| **Session ID** | 请求里的 `session_id` 字段把请求钉到对应 KV-cache 谱系 |
| **Fork** | 不复制 KV，把会话分叉到新 ID |
| **TTL** | 内存压力下按 LRU 驱逐空闲会话 |
| **跨端点** | 同一会话在 OpenAI / Anthropic / Responses 都能用 |

## MCP — 模型上下文协议

| 功能 | 详情 |
|---------|---------|
| **传输** | stdio、SSE、streamable-HTTP |
| **工具** | `MCPTool` 带 JSON-schema 输入；命名空间化 `server__tool` |
| **资源** | 通过 `MCPServerConfig` 暴露 |
| **服务状态** | disconnected / connecting / connected / error |
| **工具执行** | `MCPExecuteRequest` / `Response`；超时 + headers 可配 |
| **校验** | 工具调用前强制 input-schema 校验 |

## Agent 集成

| Agent | 内置支持 |
|-------|------------------|
| **OpenClaw** | 工具调用型 agent |
| **Hermes Agent** | 长链路推理 agent |
| **OpenCode** | 编程 agent |
| **插件系统** | 可扩展的 agent 框架 |

`AgentsPageView` 提供 安装/启动/配置/查看配置 的 UX。

## Tokenhub 云模型代理

把远端 API 提供商通过 NovaMLX 代理出来，让所有客户端（Codex、Claude Code、Continue 等）只需要打一个本地入口。

| 功能 | 详情 |
|---------|---------|
| **提供商目录** | 20+ 预置：OpenAI、Anthropic、DeepSeek、GLM/智谱、Qwen/通义、Groq、Mistral、Moonshot、Yi、Together、Fireworks、OpenRouter 等 |
| **提供商类型** | 云托管（tknet.ai 会话）vs. 自带 Key（BYO-key） |
| **路由** | `tknet:<provider-id>` 模型前缀 → 解析到对应 provider |
| **透传** | 原样转发 body；只把 `model` 换成 `provider.remoteModel` |
| **视觉后端** | 层级一：本地 VLM；层级二：provider 的 `anthropicEndpoint`（如 GLM anthropic-proxy）；层级三：`visionCompanionModel`；附带图像预处理 + 描述注入 |
| **Provider 指标** | 成功数、请求数、平均延迟、按 provider 统计 |
| **Key 解析** | 托管 provider 用会话 token；BYO-key 用 `provider.apiKey` |
| **Endpoint 字段** | `anthropicEndpoint` 用于原生支持 Anthropic 格式的 provider |

## Anthropic↔OpenAI 翻译桥

当客户端发送 `/v1/messages`（Anthropic 格式），但解析到的 provider 只会说 OpenAI（DeepSeek、GLM、Qwen-compat 等），翻译桥会把请求翻译成 OpenAI `/chat/completions`，转发后重新组装出 `AnthropicResponse`。

| 子系统 | 行为 |
|-----------|----------|
| **判别条件** | `needsAnthropicBridge(provider, path)`：path 是 `messages` 且 provider 没设 `anthropicEndpoint` 时为 true |
| **入站** | 解码 `AnthropicRequest`，通过既有 `mapAnthropicMessages` 映射 |
| **出站 body** | 构造 OpenAI chat/completions：model、messages、tools、tool_choice、采样参数、stop |
| **响应** | 解码 `OpenAIResponse`，重建带 text / thinking / tool_use 块的 `AnthropicResponse` |
| **流式** | 逐事件状态机：OpenAI chunk → Anthropic `message_start` / `content_block_start` / `content_block_delta(text_delta\|thinking_delta\|input_json_delta)` / `content_block_stop` / `message_delta(stop_reason, usage)` / `message_stop` |
| **stop reason 映射** | `stop → end_turn`、`tool_calls → tool_use`、`length → max_tokens`、`stop_sequence → stop_sequence` |
| **对 LB 透明** | LB dispatcher 把 `lb:` + messages 走同一套 passthrough — 翻译桥自动生效 |
| **代码位置** | `Sources/NovaMLXAPI/APIServer+TokenhubAnthropicBridge.swift` |

## 负载均衡器

通过 `lb:<slug>` 模型前缀，把请求路由到本地 + 远端混合的模型池。

| 功能 | 详情 |
|---------|---------|
| **LBRouter 策略** | Tiered（默认）、round-robin、weighted、least-latency |
| **成员类型** | `.local`（推理服务）和 `.remote`（tokenhub provider） |
| **LBProxy** | 每请求独立 actor：选成员、按序尝试、失败重试 |
| **管理 API** | 9 个端点：LB + 成员 CRUD、试跑路由、统计 |
| **按成员统计** | 成功/失败/平均延迟；UI 中可视化 |
| **UI** | `LoadBalancersPageView`：手风琴行、成员选择器、策略配置、每行 play 按钮 |
| **API 格式** | 跨 OpenAI / Anthropic / Responses 全支持（Anthropic 走翻译桥；Responses 走 `tknet:` 重写） |

## API Key 管理

基于 SQLite 的 API key 体系，带哈希、速率限制、白名单。

| 功能 | 详情 |
|---------|---------|
| **存储** | SQLite 通过 `APIKeyStore`（取代了原 `api_keys.json`） |
| **哈希** | SHA-256（明文为了 reveal 功能保留；DB 访问受控） |
| **CRUD** | 通过 `/admin/keys` 与 UI 创建/读/更新/删除 |
| **轮换** | `/admin/keys/{id}/rotate` 生成新明文，作废旧 key |
| **速率限制** | 按 key 的 `rateLimitPerSecond`、`maxTokensPerPeriod`、`maxRequestsPerPeriod`，按周期（分/时/日）重置 |
| **白名单** | 按 key 的 `allowedModels[]`、`allowedEndpoints[]` |
| **用量追踪** | 总计 + 周期 token/请求数；按模型分解；最近使用时间 |
| **开放模式旁路** | 未配置任何 key 时关闭鉴权（开发模式） |
| **UI** | `APIKeysPageView`：整行手风琴、eye 揭示、copy 已揭示、用量进度条、白名单 |
| **Admin vs. 普通 key** | 中间件分离：`AdminAuthMiddleware`（端口 6591）vs. `APIKeyAuthMiddleware`（端口 6590） |

## 模型管理

| 功能 | 详情 |
|---------|---------|
| **模型目录** | `~/.nova/models/<repo_id>/` |
| **发现** | `modelManager.downloadedModels()` + `inferenceService.listLoadedModels()` |
| **自动加载** | 请求未加载模型时触发加载（通过 `ensureModelReady` 可配） |
| **自动驱逐** | 内存压力下 LRU；绝不在请求中途驱逐 |
| **重启恢复** | `restoreModels()` 重建上次会话的加载集合 |
| **loaded_models 持久化** | SQLite 表（原 JSON，已修复"重启即被擦"的 bug） |
| **模型设置** | 按模型覆盖项持久化到 SQLite（`model_settings` 表） |
| **模型卡** | 通过 `/admin/api/hf/model-info` 从 HuggingFace 拉取元数据 |
| **Forgetting** | `/admin/models/{id}/forget` — 清空 KV cache + container 状态 |

## HuggingFace 集成

| 功能 | 详情 |
|---------|---------|
| **下载** | `POST /admin/api/hf/download`，带 `repo_id` + 可选 `endpoint` |
| **取消** | `POST /admin/api/hf/cancel` 按 task ID；杀掉飞行中的 `Task` |
| **状态轮询** | `GET /admin/api/hf/tasks` 返回按文件的进度 + 速度 + 卡顿检测 |
| **镜像支持** | HF endpoint 可配（默认 `huggingface.co`；国内常切 `hf-mirror.com`） |
| **幂等续传** | `cancelTasksForRepo` 在启动新任务前先杀飞行中任务（防点击竞争） |
| **HEAD 探测** | 每个文件下载前做 3 秒连通性探测（被墙时快速失败） |
| **阶段 UI** | 客户端显示：Connecting / Downloading（MB/s）/ Stalled（已 N 秒无字节）/ Endpoint unreachable |
| **断点扫描** | 启动时检测 `*.download` 临时文件 → 转成 `.failed` 任务供手动 Resume |
| **Xet CDN** | 自动跟随 `cas-bridge.xethub.hf.co` 重定向 |

## Modelfile 系统

类似 Ollama Modelfile 的自定义模型定义。

| 功能 | 详情 |
|---------|---------|
| **存储** | SQLite `modelfiles` 表 |
| **字段** | `name`、`base_model`、`system`、`template`、`parameters`、`adapter`、`tools` |
| **管理 API** | `/admin/modelfiles` 的 CRUD |
| **解析** | Modelfile 名称像 model ID 一样被解析 |
| **工具定义** | 静态 tool 定义嵌在 modelfile 里 |

## 内存管理

| 子系统 | 行为 |
|-----------|----------|
| **ProcessMemoryEnforcer** | 通过 `ProcessInfo.memoryPressure` + 主动 trim 硬性 RSS 上限 |
| **MemoryBudgetTracker** | 每个激活模型的 GPU 内存预算 |
| **WiredMemoryTicket** | 分配前预留 wired Metal 内存 |
| **Auto 模式** | 进程监控系统压力，按需驱逐 |
| **Disabled 模式** | 不设上限（开发/基准） |
| **Percent 模式** | 上限 = 系统内存的 N% |
| **Fixed 模式** | 显式 GB 上限 |
| **分阶段 eval** | 大模型按批次加载，避免 Metal OOM |

## 按模型独立配置

按 model ID 持久化，请求时覆盖默认值：

- 默认采样（temperature、maxTokens、topP、repeatPenalty）
- KV 精度（4-bit / 8-bit / 32-bit）
- 上下文窗口覆盖
- TurboQuant 开关
- 视觉策略
- 配套视觉模型
- 思考默认值（`enableThinking`、`thinkingBudget`、`preserveThinking`）
- Keep-alive 间隔

## 可观测性与基准测试

| 工具 | 端点 | 备注 |
|------|----------|-------|
| **Benchmark** | `/admin/models/{id}/benchmark` | 测 TPS、TTFT、内存、峰值 |
| **Perplexity** | `/admin/models/{id}/perplexity` | 标准困惑度测试集 |
| **缓存统计** | `/admin/models/{id}/cache` | KV cache 命中/缺失/大小 |
| **服务统计** | `/admin/stats` | 聚合：总 token、活跃请求、uptime |
| **系统信息** | `/admin/info` | 芯片、核心、RAM、GPU、macOS 版本 |
| **日志** | `~/.nova/novamlx.log` | 轮转文件日志；运行时级别通过 admin 控制 |
| **日志级别** | debug / info / warning / error | `POST /admin/log/level` |
| **指标响应头** | `X-Tokenhub-Provider`、`X-Model-Cold-Load`、`X-Model-Load-Time-Ms` | 按响应内省 |
| **InferenceStats** | 实时 TPS、峰值 TPS、已生成 token、活跃请求、Worker CPU | UI 每 2 秒轮询 |

## macOS 菜单栏应用

原生 SwiftUI 菜单栏应用。状态图标 + 下拉 + 弹出窗口。

### 页面

| 页面 | 亮点 |
|------|------------|
| **Status** | 实时 TPS 曲线（90 采样窗口、峰值追踪、零值裁剪）、CPU/内存/GPU 栅格、设备信息、峰值 TPS |
| **Dashboard** | 单屏概览：已加载模型、活跃请求、内存、uptime、快速加载 |
| **Local Inference** | 已激活模型支持 卸载 + 复制名 + play 跳 Playground；已下载模型按类型 tabs（全部 / LLM / VLM / Embed / Audio / Image）；模型卡 |
| **Downloads** | 分类 tabs、推荐模型卡、阶段感知进度、卡顿检测、镜像配置 |
| **Playground（Chat）** | LLM / ASR / TTS / Image 统一入口；按模型类型自动识别；带分段的 picker（本地直连 vs TOKENHUB HTTP）；参数滑块；Disable-Thinking 开关；粘性自动滚动；cURL 复制（OpenAI/Anthropic/Responses）；ASR 麦克风 + TTS 扬声器 + 语音克隆 |
| **Tokenhub** | 20+ 提供商目录；CRUD；每个 provider 的 API 模型支持复制 + play；端点连通性测试 |
| **Load Balancers** | 手风琴行；成员选择器（本地 + 远端）；策略 + 统计；每行 play 按钮 |
| **API Keys** | 整行手风琴；eye 揭示；复制已揭示；用量进度条；速率限制展示；白名单 |
| **Cluster** | 网络扫描（Thunderbolt + ARP）；Worker 健康状态监控；按节点模型就绪度 |
| **Settings** | HF endpoint + 镜像；内存模式；日志级别；集群开关；TurboQuant；tknet.ai 账号绑定 |
| **Audio** | 独立的 ASR / TTS 入口（与 Playground 分离） |
| **Agents** | 安装/启动/配置 OpenClaw、Hermes、OpenCode |

### 跨页面 UX

| 功能 | 详情 |
|---------|---------|
| **Pick-to-Playground** | Active Models / Tokenhub API 模型 / Load Balancers 上的 `play.circle` 按钮 → 跳到 Playground 并预选该模型 |
| **复制 cURL 按钮** | Playground Parameters 里：OpenAI / Anthropic / Responses — 输出 `${NOVA_API_KEY}` 占位符，密钥不入剪贴板 |
| **粘性自动滚动** | 通过 `MessageListBottomOffsetKey` PreferenceKey 做 80 pt 阈值 |
| **模型 picker** | `inferModeFromName` + `autoDetectMode` 自动切换模式（LLM / ASR / TTS / Image） |
| **HF Endpoint 设置** | 镜像切换免重启 — 通过 `NovaMLXConfiguration.shared` 传播 |

## 国际化

9 种语言，从系统 locale 自动检测，英文兜底。

`英文`、`简体中文（zh-Hans）`、`香港繁体中文（zh-Hant-HK）`、`台湾繁体中文（zh-Hant-TW）`、`日文`、`韩文`、`法文`、`德文`、`俄文`。

## 安全与中间件

| 层 | 行为 |
|-------|----------|
| **CORS** | 可配 origins；处理 preflight |
| **限流** | 按 API key + 按 IP 的 token bucket；全局 + 按路由限流器 |
| **错误中间件** | `NovaMLXErrorMiddleware` 把错误统一为 OpenAI/Anthropic 形态；429 带 `Retry-After` |
| **Admin 鉴权** | 端口 6591 上的 `AdminAuthMiddleware` — 需要 admin key |
| **API 鉴权** | 端口 6590 上的 `APIKeyAuthMiddleware` — Bearer 或 `x-api-key`；无 key 时开放模式旁路 |
| **严格响应头** | `X-Content-Type-Options`、`Strict-Transport-Security`、`X-Frame-Options`、`Referrer-Policy` |
| **请求 ID** | 每请求 `x-request-id` 用于追踪 |
| **云端会话** | `AuthCache` 持有 tknet.ai 会话 token；云端校验端点 |

## Homebrew 分发

| 渠道 | 详情 |
|---------|---------|
| **Formula** | `Formula/novamlx.rb` |
| **构建脚本** | `./build.sh` — UUID 同步、codesign、MLX shader 编译、bundle 组装 |
| **DMG** | 脚本化磁盘映像打包 |
| **Codesigning** | 部署后必须 `codesign --force --deep --sign -`（否则 macOS 会 SIGKILL worker） |
| **Entitlements** | `audio-input`、`com.apple.security.device.camera`（按需）、通过 PlistBuddy 注入 `NSMicrophoneUsageDescription` |

## TCC 监控器

| 功能 | 详情 |
|---------|---------|
| **目的** | 自动关闭阻断自动化的 macOS TCC 隐私弹窗 |
| **Bundle** | `TCCWatcher.app`（CFBundleIdentifier `com.novamlx.TCCWatcher`） |
| **安装** | `Scripts/install-tcc-watcher.sh` — 配置 LaunchAgent |
| **机制** | 通过 `System Events` 监视 TCC 弹窗；`inspectWindow` 内联到主 tell 块（修复上下文传播 bug） |
| **一次性配置** | 用户在 系统设置 → 隐私 → 辅助功能 里通过 `+` 添加 `TCCWatcher.app`（macOS 14+ 不会自动弹窗） |

## 配置

### `ServerConfig`（`~/.nova/config.json` 或 SQLite）

| 字段 | 默认 | 备注 |
|-------|---------|-------|
| `server.port` | 6590 | 公共 API 端口 |
| `server.adminPort` | 6591 | 管理 API 端口 |
| `server.cluster` | null | 启用分布式模式 |
| `server.apiKey` | null | 为 null 时开放模式 |
| `server.corsOrigins` | `*` | CORS 白名单 |
| `huggingface.endpoint` | `huggingface.co` | 镜像切换 |
| `autoLoad` | 启用 | 启动时预加载模型 |
| `memory.mode` | `auto` | auto / disabled / percent / fixed |
| `memory.limitGB` | null | 固定上限 |
| `scaleTokenCount` | 启用 | 对上下文核算做 token 数缩放 |
| `logLevel` | `info` | debug / info / warning / error |

### `NOVA_DIR`

| 优先级 | 来源 |
|------------|--------|
| 1（最高） | `~/.config/novamlx/path` 文件内容 |
| 2 | `NOVA_DIR` 环境变量 |
| 3（默认） | `~/.nova` |

支持多实例。仅 `models/` 可跨实例共享。

### 引擎配置

`ModelFamilyRegistry` 里的按家族默认值 — KV 精度、prefill chunk、上下文窗口、草稿模型推荐。

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│  客户端: Codex / Claude Code / Continue / OpenAI SDK /      │
│         Anthropic SDK / curl                                │
└────────────────────────┬────────────────────────────────────┘
                         │
            ┌────────────▼────────────┐
            │   Hummingbird HTTP/2    │  端口 6590 (api) + 6591 (admin)
            │   + CORS + 限流         │
            │   + 鉴权中间件          │
            └────────────┬────────────┘
                         │
   ┌─────────────────────┼──────────────────────────┐
   │                     │                          │
┌──▼─────────┐  ┌────────▼─────────┐  ┌────────────▼───────────┐
│  本地推理  │  │  Tokenhub 代理   │  │  LBProxy (lb:<slug>)   │
│  (Worker)  │  │  + Anthropic 桥  │  │  → 路由到本地或        │
│            │  │                  │  │    tokenhub 成员        │
└──┬─────────┘  └────────┬─────────┘  └────────────────────────┘
   │                     │
┌──▼─────────────────────▼──┐
│  MLX 引擎                 │
│  - ContinuousBatcher      │
│  - TurboQuant KV Cache    │
│  - 推测解码               │
│  - 工具调用               │
│  - 结构化输出             │
│  - Chat 模板处理器        │
└──┬────────────────────────┘
   │
┌──▼──────────────────────┐  ┌──────────────────────────┐
│  WorkerSupervisor       │  │  分布式 (可选)           │
│  (子进程隔离)           │  │  Ring TCP / Thunderbolt  │
└─────────────────────────┘  └──────────────────────────┘
```

数据持久化：**NovaDB**（SQLite + GRDB）— 14 张表，覆盖 api_keys、providers、load_balancers、modelfiles、loaded_models、model_settings、metrics、cluster_policy、conversations（Responses）等。首次启动时自动从旧 JSON 迁移。

## 快速开始

```bash
# 通过 Homebrew 安装
brew install --head novamlx

# 或从源码构建
git clone https://github.com/novamlx/novamlx && cd novamlx
./build.sh
open dist/NovaMLX.app

# 配置 API Key（或不配置任何 key 走开放模式）
export NOVA_API_KEY=$(openssl rand -hex 24)

# 列出模型
curl http://localhost:6590/v1/models \
  -H "Authorization: Bearer $NOVA_API_KEY"

# 聊天
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role":"user","content":"Hello"}]
  }'

# Anthropic 格式
curl http://localhost:6590/v1/messages \
  -H "x-api-key: $NOVA_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"Hello"}]
  }'

# 代理云端模型（先在 Tokenhub UI 里配置 provider）
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tknet:deepseek-v4-flash",
    "messages": [{"role":"user","content":"Hello"}]
  }'

# 负载均衡器（先在 UI 里创建，再用 lb:<slug>）
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lb:my-pool",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

---

**NovaMLX** — *由 hlky 与贡献者构建。MIT License 发布。*
