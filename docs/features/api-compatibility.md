# API Compatibility

**Pitch:** One inference engine behind OpenAI, Anthropic, and Responses APIs, plus a separate admin port.

OpenAI、Anthropic、Responses 三种协议共用同一套推理引擎与工具调用层。功能目录记载公共端口 `6590`、管理端口 `6591`。

## Who it is for

已经使用 OpenAI SDK、Anthropic SDK、Codex 或自有 HTTP 客户端，希望把 base URL 指到本机的人。

## What it does

三种格式共享引擎与工具层。工具字段的对应关系见 [工具调用](tool-calling.md)。结构化约束见 [结构化输出](structured-output.md)。

<a id="openai"></a>

### OpenAI-compatible（端口 6590）

| 端点 | 方法 | 备注 |
|------|------|------|
| `/v1/models` | GET | 列出已加载与已下载模型 |
| `/v1/models/{id}` | GET | 模型元数据 |
| `/v1/chat/completions` | POST | 流式与非流式 |
| `/v1/completions` | POST | 旧式文本补全 |
| `/v1/embeddings` | POST | 文本转向量 |
| `/v1/audio/transcriptions` | POST | Whisper / Qwen3-ASR |
| `/v1/audio/speech` | POST | Dots / Qwen3-TTS |
| `/v1/images/generations` | POST | FLUX、Qwen-Image、Z-Image、MOSS-Image 文生图 |
| `/v1/images/edits` | POST | FLUX.1 带遮罩 |
| `/v1/images/variations` | POST | FLUX.1 变体 |
| `/v1/videos` | POST | 异步视频任务（OpenRouter 形态，h3.c），返回 202 |
| `/v1/videos/{id}` | GET | 查询视频任务 |
| `/v1/videos/{id}/content` | GET | 下载生成的 mp4 |
| `/v1/videos/models` | GET | 视频模型能力 |
| `/v1/rerank` | POST | 交叉编码器重排 |

视频路由来自英文 `features.md`。中文目录的 OpenAI 表没有这些行；细节见 [视频生成](video-generation.md)。英文目录另有 `POST /v1/videos/{id}/cancel`（仅排队中的任务），写在视频一节而不是上表。

<a id="anthropic"></a>

### Anthropic-compatible

| 端点 | 方法 | 备注 |
|------|------|------|
| `/v1/messages` | POST | 流式与非流式。支持 `messages`、`system`、`tools`、`tool_choice`、`thinking_budget` |
| `/v1/messages/count_tokens` | POST | 预先 token 计数 |

鉴权：`x-api-key` 或 `Authorization: Bearer`。需要头 `anthropic-version: 2023-06-01`。

<a id="responses"></a>

### Responses API

与 Codex 兼容的 `/v1/responses`。

| 功能 | 详情 |
|------|------|
| **端点** | `POST /v1/responses`（流式与非流式）、`GET` / `DELETE /v1/responses/{id}`、`POST /v1/responses/{id}/cancel` |
| **P0** | `tool_choice` 透传（auto / required / none / 指定工具） |
| **P1** | 17 个字段的响应回显（status、output、usage、completed_at 等） |
| **P2** | SSE 的 `seq`，供客户端做断档检测 |
| **ConversationStore** | `previous_response_id` 取回已存对话 |
| **Reasoning** | `reasoning.effort` 对应 `thinking_budget` |
| **Compact** | `POST /v1/responses/compact` 截断已存对话 |
| **Input tokens** | `POST /v1/responses/input_tokens` 精确 token 计数 |

<a id="admin"></a>

### 管理端口（6591）

Bearer admin key。目录列出的能力：

- **Models** — 列表、详情、下载、取消、加载、卸载、forget、benchmark、perplexity、缓存统计
- **API Keys** — CRUD、轮换、用量、按 key 的指标
- **Tokenhub providers** — CRUD、测试、指标
- **Load balancers** — CRUD、成员、试跑路由
- **Modelfiles** — 自定义模型定义的 CRUD
- **Stats / Info / Reset** — 服务指标、系统信息、重置
- **Log level** — 运行时日志级别

用户 key 与 admin key 的中间件是分开的。见 [API Key](api-keys.md) 与 [安全](security.md)。

## How to try

聊天（OpenAI）：

```bash
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

Anthropic：

```bash
curl http://localhost:6590/v1/messages \
  -H "x-api-key: $NOVA_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

列出模型：`GET /v1/models`。未配置 key 时目录记载为开放模式。根 README 的同类示例使用端口 `8080`。

## Limits / notes

- Responses 的 P1 写的是「17 字段回显」，目录没有把这 17 个字段逐一命名。
- `cache_control` 在本地 Anthropic 路径上不解析。见 [KV cache](kv-cache.md)。
- 视频端点的执行队列、webhook 与配置文件不在本页，见 [视频生成](video-generation.md)。

## Related

- [工具调用](tool-calling.md)
- [结构化输出](structured-output.md)
- [Tokenhub](tokenhub.md)
- [负载均衡](load-balancers.md)
- [安全与中间件](security.md)
- 仓库笔记：`docs/superpowers/designs/2026-06-02-responses-api-implementation-notes.md`

正文来自 `features.zh-CN.md`「API 兼容性」，视频行补自 `features.md`（中文目录的 OpenAI 表未列视频）。标题与 pitch 为英文。
