# 功能目录 / Feature index

本目录是面向用户的功能说明，一项一页。

This directory is the user-facing documentation, one page per feature.

正文尽量来自 `features.zh-CN.md`，标题与一句话为英文。某节只有英文目录（视频）或只有根 README（TIE）时，页面上会注明。`stable` 表示从既有目录拆出；`preview` 表示产品自己标成预览。

<a id="inference-performance"></a>

## 推理与性能 / Inference & performance

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [NovaMLX-TIE](tie.md) | 权重按 wired / LRU / SSD 三层供数，模型可以大于统一内存。 | stable |
| [推理引擎 / Inference engine](inference-engine.md) | 在 Apple Silicon 上用 MLX 跑本地 LLM、VLM、音频与图像。 | stable |
| [Worker 隔离 / Worker isolation](worker-isolation.md) | 每个模型一个子进程，崩溃不会拖垮服务。 | stable |
| [连续批处理 / Continuous batching](continuous-batching.md) | 并发序列共享 GPU，内存紧时按优先级抢占。 | stable |
| [推测解码 / Speculative decoding](speculative-decoding.md) | n-gram 或小草稿模型先提议 token，目标模型再接受。 | stable |
| [TurboQuant KV](turboquant-kv.md) | KV 以 4-bit 仿射量化存放，采样器无感知。 | stable |
| [KV cache 与前缀共享 / KV cache & prefix share](kv-cache.md) | 可配置目录复用前缀（外置盘不在则回落），`session_id` 钉住谱系。 | stable |
| [分布式推理 / Distributed inference](distributed-inference.md) | 多台 Apple Silicon 上做流水线并行，失败则回到本机。 | stable |

<a id="multi-modal"></a>

## 多模态 / Multi-modal

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [多模态总表 / Multi-modal overview](multi-modal.md) | 视觉、语音、FLUX、MOSS-Image、嵌入与重排的架构一览。 | stable |
| [视觉 / VLM](vision.md) | Qwen-VL、LLaVA、Gemma3、Pixtral 等本地看图。 | stable |
| [音频 ASR / TTS / Audio](audio.md) | Whisper 或 Qwen3-ASR 转写，Dots TTS 与语音克隆。 | stable |
| [图像生成 / Image generation](image-generation.md) | FLUX、Qwen-Image、Z-Image 与 MOSS-Image 文生图，输出 Base64 PNG。 | stable |
| [抠图去背景 / Image matting](matting.md) | BiRefNet 通用与人像版抠图，`POST /v1/images/matting` 出 mask 与透明底。 | preview |
| [视频生成 / Video generation](video-generation.md) | 本地 h3.c 的异步 `/v1/videos` 任务。 | stable |
| [3D 生成 / 3D generation](hunyuan3d.md) | 本地 Hunyuan3D-2.1 的图生 3D 任务，`POST /v1/3d/generations` 出 GLB。 | preview |
| [Flint（Recursive）](flint.md) | 40B agent 推理模型：思考链 + 工具调用 + 256K，本地 MLX。 | preview |
| [嵌入与重排 / Embeddings & rerank](embeddings-rerank.md) | `POST /v1/embeddings` 与 `POST /v1/rerank`。 | stable |

<a id="apis"></a>

## API 与协议 / APIs & protocols

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [API 兼容 / API compatibility](api-compatibility.md) | OpenAI、Anthropic、Responses 共用一个引擎。 | stable |
| [OpenAI 兼容 / OpenAI-compatible](api-compatibility.md#openai) | 聊天、补全、嵌入、音频、图像、视频与重排。 | stable |
| [Anthropic 兼容 / Anthropic-compatible](api-compatibility.md#anthropic) | `/v1/messages` 与 token 计数，需要 `anthropic-version`。 | stable |
| [Responses API](api-compatibility.md#responses) | Codex 形态的 `/v1/responses`，含 `previous_response_id`。 | stable |
| [管理端口 / Admin API](api-compatibility.md#admin) | 端口 6591：模型、Key、Tokenhub、LB、日志级别。 | stable |
| [结构化输出 / Structured output](structured-output.md) | JSON、JSON schema、正则、GBNF 或枚举。 | stable |
| [工具调用 / Tool calling](tool-calling.md) | OpenAI、Anthropic、Responses 三种工具形态。 | stable |
| [控制令牌与思考 / Control tokens & thinking](control-tokens.md) | 滤掉协议停止符，并处理 `<think>`。 | stable |

<a id="routing"></a>

## 路由与云 / Routing & cloud

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [Tokenhub](tokenhub.md) | `tknet:<provider>` 把请求转到远端 API。 | stable |
| [Anthropic↔OpenAI 翻译桥 / Translation bridge](anthropic-openai-bridge.md) | `/v1/messages` 译成 OpenAI chat completions 再译回。 | stable |
| [负载均衡 / Load balancers](load-balancers.md) | `lb:<slug>` 在本地模型与 Tokenhub provider 之间选择。 | stable |

<a id="model-ops"></a>

## 模型运维 / Model ops

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [支持的模型 / Supported models](supported-models.md) | 目录中每条具体权重的 `org/repo`，与 `nova download` 相同。 | stable |
| [模型架构 / Model architectures](model-architectures.md) | `ModelFamily` 与按家族的 KV、prefill、上下文默认值。 | stable |
| [模型管理 / Model management](model-management.md) | 多根目录、按需加载、重启恢复。 | stable |
| [Hugging Face](huggingface.md) | 下载、取消、续传，以及镜像端点。 | stable |
| [Modelfile](modelfile.md) | 命名一套系统提示、模板、参数与静态工具。 | stable |
| [内存 / Memory](memory-management.md) | auto、关闭、百分比或固定 GB 上限。 | stable |
| [按模型配置 / Per-model settings](per-model-settings.md) | 采样、KV 精度、上下文、TurboQuant、视觉与思考。 | stable |

<a id="app-ops"></a>

## 应用与运维 / App & ops

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [菜单栏 / Menu bar](menu-bar.md) | 状态、下载、Playground、Key、集群与设置。 | stable |
| [国际化 / Internationalization](internationalization.md) | 九种界面语言，跟随系统，英文兜底。 | stable |
| [API Key](api-keys.md) | SQLite key、限额、白名单与轮换。 | stable |
| [可观测性 / Observability](observability.md) | 基准、困惑度、统计与运行时日志级别。 | stable |
| [Homebrew](homebrew.md) | Formula、DMG，以及 worker 的 codesign。 | stable |
| [MCP](mcp.md) | stdio、SSE、streamable-HTTP 与 schema 校验。 | stable |
| [Agent](agents.md) | 菜单栏安装 OpenClaw、Hermes、OpenCode。 | stable |
| [会话 / Sessions](sessions.md) | `session_id` 复用 KV，分叉不复制缓存。 | stable |
| [安全 / Security](security.md) | 用户 key 与 admin key、CORS、限流。 | stable |
| [TCC Watcher](tcc-watcher.md) | 辅助功能授权后，关掉挡住自动化的隐私窗。 | stable |
| [配置 / Configuration](configuration.md) | `~/.nova/config.json` 中的端口、内存与日志。 | stable |
| [架构 / Architecture](architecture.md) | 双端口 HTTP、本地 Worker、Tokenhub 与 LB。 | stable |
| [快速开始 / Quick start](quick-start.md) | 安装并调用 `/v1/chat/completions`。 | stable |

<a id="new-models"></a>

## 新模型 / New model support

| 功能 / Feature | 一句话 / Pitch | 状态 |
|----------------|----------------|------|
| [Nemotron 3.5 Bolt 文本 / Nemotron 3.5 Bolt text](nemotron-bolt-text.md) | 本地 74B 级异构 MoE，仅文本，思考默认开启。 | preview |
| [GLM-5.3-Flash 文本 / GLM-5.3-Flash text](glm-5-3-flash.md) | 本地 glm5_next 文本对话。混合 2-bit 约 128 GB，4-bit 需要 256 GB。 | preview |
