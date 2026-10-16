# OpenRouter 视频生成 API 完整规范

> 调研日期：2026-10-16
> 来源：
> - [OpenRouter 官方文档 — Video Generation](https://openrouter.ai/docs/guides/overview/multimodal/video-generation)
> - [OpenRouter 官方教程 — Video Generation API: A Code-First Guide](https://openrouter.ai/blog/tutorials/video-generation-api/)
> - [OpenRouter 公告 — Announcing Video Generation](https://openrouter.ai/blog/announcements/video-generation/)

## 1. 核心设计：异步任务（Job）模式

视频生成通常需要 30 秒到数分钟，不能让客户端挂在一个 HTTP 请求上干等。OpenRouter 因此采用**异步 Job 模式**，生命周期如下：

```
提交 POST /api/v1/videos
      │
      ▼  立即返回 202 + job id + polling_url
轮询 GET /api/v1/videos/{jobId}
      │
      ▼  直到 status == "completed"
下载 GET /api/v1/videos/{jobId}/content?index=0
      │
      ▼  MP4 文件
```

关键点：

- **提交与完成解耦**：提交后立即返回 job ID，进程重启后凭 job ID 可恢复跟踪，不必重新付费生成。
- **鉴权**：所有端点（包括下载）都走 `Authorization: Bearer $OPENROUTER_API_KEY`。
- **统一接口**：Seedance、Veo、Wan 等模型共用同一端点，换模型只需改 `model` 字段。
- **ZDR 不适用**：异步模式要求平台短暂保存输出供下载，因此开启 Zero Data Retention 的账号/请求不会被路由到视频生成。

## 2. 模型发现

### 2.1 视频模型专用端点

```bash
curl "https://openrouter.ai/api/v1/videos/models" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

响应示例（`data` 数组）：

```json
{
  "data": [
    {
      "id": "google/veo-3.1",
      "canonical_slug": "google/veo-3.1",
      "name": "Google: Veo 3.1",
      "description": "...",
      "created": 1719792000,
      "supported_durations": [4, 6, 8],
      "supported_resolutions": ["720p", "1080p"],
      "supported_aspect_ratios": ["16:9", "9:16", "1:1"],
      "supported_sizes": ["1280x720", "1920x1080"],
      "pricing_skus": {
        "per-video-second": "0.50",
        "per-video-second-1080p": "0.75"
      },
      "allowed_passthrough_parameters": ["output_config"]
    }
  ]
}
```

| 字段 | 说明 |
|---|---|
| `id` | 提交请求时使用的模型 slug |
| `supported_durations` | 支持的时长（秒）列表 |
| `supported_resolutions` | 支持的分辨率（如 `720p`、`1080p`） |
| `supported_aspect_ratios` | 支持的宽高比（如 `16:9`、`9:16`） |
| `supported_sizes` | 支持的像素尺寸（如 `1280x720`） |
| `pricing_skus` | 按 SKU 计费信息 |
| `allowed_passthrough_parameters` | 可经 `provider.options` 透传的厂商参数 |

**重要**：提交前应对照 `supported_*` 字段校验 `duration` / `resolution` / `aspect_ratio`，不支持的组合会返回 400 并列出合法值。各模型能力差异很大（例如 Veo 3.1 只支持 4/6/8 秒，Seedance 2.0 支持 4–15 秒）。

### 2.2 通用 Models API

```bash
curl "https://openrouter.ai/api/v1/models?output_modalities=video"
```

## 3. 提交生成请求

**`POST https://openrouter.ai/api/v1/videos`**

```bash
curl "https://openrouter.ai/api/v1/videos" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bytedance/seedance-2.0",
    "prompt": "A paper boat drifting down a rain-slicked gutter at night, neon reflections, slow tracking shot, cinematic lighting",
    "duration": 4,
    "resolution": "720p",
    "aspect_ratio": "16:9",
    "generate_audio": false
  }'
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model` | string | 是 | 模型 slug，如 `google/veo-3.1` |
| `prompt` | string | 是* | 文本描述（*支持纯图生视频的模型可省略） |
| `duration` | int | 否 | 时长（秒） |
| `resolution` | string | 否 | 分辨率：`480p`/`720p`/`768p`/`1080p`/`1K`/`2K`/`4K` |
| `aspect_ratio` | string | 否 | 宽高比：`16:9`、`9:16`、`1:1`、`4:3`、`3:4`、`3:2`、`2:3`、`21:9`、`9:21` |
| `size` | string | 否 | 精确像素 `WIDTHxHEIGHT`，与 `resolution`+`aspect_ratio` 二选一 |
| `frame_images` | array | 否 | 首/尾帧图片（image-to-video），每项须带 `frame_type: first_frame` 或 `last_frame` |
| `input_references` | array | 否 | 风格/内容参考图（reference-to-video） |
| `generate_audio` | bool | 否 | 是否生成音频；支持音频的模型默认 `true` |
| `seed` | int | 否 | 随机种子（不保证所有厂商支持） |
| `callback_url` | string | 否 | Webhook 回调地址（必须 HTTPS），优先于工作区默认回调 |
| `provider` | object | 否 | 厂商透传参数（见 §6） |

两种图片输入同时提供时，`frame_images` 优先，按图生视频处理。

图生视频示例：

```json
{
  "model": "alibaba/wan-2.7",
  "prompt": "A character walking through a forest",
  "frame_images": [
    {
      "type": "image_url",
      "image_url": { "url": "https://example.com/first-frame.png" },
      "frame_type": "first_frame"
    }
  ],
  "resolution": "1080p"
}
```

### 提交响应（202 Accepted）

```json
{
  "id": "abc123",
  "polling_url": "https://openrouter.ai/api/v1/videos/abc123",
  "status": "pending"
}
```

**拿到响应立即持久化 `id`**——它是你的应用与后台生成任务之间唯一的持久链接。

## 4. 轮询进度

**`GET https://openrouter.ai/api/v1/videos/{jobId}`**（即提交响应里的 `polling_url`）

- 建议轮询间隔 **30 秒**；视频生成一般 30 秒到数分钟，频繁轮询不会让生成变快。
- 必须设置**整体超时上限**（如 1 小时），防止卡死的 job 挂住进程。

### Job 状态机

| 状态 | 含义 | 终态？ |
|---|---|---|
| `pending` | 已受理，排队中 | 否 |
| `in_progress` | 正在生成 | 否 |
| `completed` | 完成，可下载 | 是（成功） |
| `failed` | 生成失败，看 `error` 字段 | 是 |
| `cancelled` | 被取消 | 是 |
| `expired` | 超过最大存活时间 | 是 |

### 轮询响应（完成时）

```json
{
  "id": "abc123",
  "generation_id": "gen-1234567890-abcdef",
  "polling_url": "https://openrouter.ai/api/v1/videos/abc123",
  "status": "completed",
  "unsigned_urls": [
    "https://openrouter.ai/api/v1/videos/abc123/content?index=0"
  ],
  "usage": {
    "cost": 0.25,
    "is_byok": false
  }
}
```

- `usage.cost`：本次生成的实际费用，建议存入内部 job 记录用于成本核算。
- 若 OpenRouter 保存了输出副本，还会带 `urls` 字段：免鉴权的签名下载 URL（到 `expires_at` 过期）。**注意：持有该 URL 的任何人都可下载，不要记录日志或外发。**

### Python 轮询参考实现（含完整终态处理）

```python
import time
from urllib.parse import urljoin

TERMINAL_ERROR_STATES = {"failed", "cancelled", "expired"}

def poll_video(initial_job: dict, interval: float = 30.0, timeout: float = 3600.0) -> dict:
    polling_url = urljoin("https://openrouter.ai", initial_job["polling_url"])
    deadline = time.monotonic() + timeout
    job = initial_job

    while True:
        status = job["status"]
        if status == "completed":
            return job
        if status in TERMINAL_ERROR_STATES:
            raise RuntimeError(f"Video generation ended with '{status}': {job.get('error')}")
        if status not in {"pending", "in_progress"}:
            raise RuntimeError(f"Unexpected job status: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Job {job['id']} did not complete within {timeout}s")

        time.sleep(interval)
        response = requests.get(polling_url,
                                headers={"Authorization": f"Bearer {API_KEY}"},
                                timeout=30)
        response.raise_for_status()
        job = response.json()
```

**区分两类失败**：轮询*请求*失败（网络超时等）≠ 生成*任务*失败。前者用原 job ID 重试查询；只有 job 本身进入 `failed`/`cancelled`/`expired` 才考虑重新提交（重新提交 = 新的付费任务）。

## 5. 下载结果

**`GET https://openrouter.ai/api/v1/videos/{jobId}/content?index=0`**

```bash
curl "https://openrouter.ai/api/v1/videos/abc123/content?index=0" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  --output video.mp4
```

- `index` 默认 0，仅在模型一次产出多个视频时需要变更。
- `unsigned_urls` 名字里虽有 "unsigned"，但**不是预签名 URL**，下载仍需带 API Key。
- 建议流式写盘，避免整段 MP4 进内存。
- 下载后尽快转存到自己的存储——生成端点不是永久文件托管。

## 6. 厂商透传参数

通过 `provider.options` 按厂商 slug 传递模型特有参数，只有实际命中厂商的参数会被转发，未知 key 被丢弃：

```json
{
  "model": "google/veo-3.1",
  "prompt": "A time-lapse of a flower blooming",
  "provider": {
    "options": {
      "google-vertex": {
        "parameters": {
          "personGeneration": "allow",
          "negativePrompt": "blurry, low quality"
        }
      }
    }
  }
}
```

各模型允许的透传 key 见视频模型端点的 `allowed_passthrough_parameters` 字段（如 Veo 的 `negativePrompt`/`enhancePrompt`，Wan 的 `negative_prompt`/`prompt_extend`）。

## 7. Webhook（轮询的替代方案）

任务量大时（数百并发 job），轮询效率低，可改用 Webhook：

1. **按请求**：提交时传 `callback_url`（优先）。
2. **工作区默认**：在 workspace 设置里配置默认回调地址。

Job 到达终态时，OpenRouter 向回调地址 POST 事件信封，事件类型：`video.generation.completed` / `.failed` / `.cancelled` / `.expired`。

`video.generation.completed` 载荷示例：

```json
{
  "type": "video.generation.completed",
  "created_at": "2026-04-24T12:00:00.000Z",
  "data": {
    "id": "abc123",
    "status": "completed",
    "generation_id": "gen-xyz789",
    "model": "google/veo-3.1",
    "unsigned_urls": ["https://openrouter.ai/api/v1/videos/abc123/content?index=0"],
    "usage": { "cost": 0.5, "is_byok": false }
  }
}
```

### 幂等与验签

- 每次投递带 **`X-OpenRouter-Idempotency-Key`** 头，形如 `<job_id>-<status>`（如 `job-abc123-completed`）。处理前先存 key，重复投递直接跳过。
- 配置 signing secret 后带 **`X-OpenRouter-Signature`** 头：`t=<timestamp>,v1=<hmac>`。验证方法：
  1. 从头部取出 `t` 和 `v1`；
  2. 拼接 `{timestamp},{raw_request_body}`（逗号连接）；
  3. 用 signing secret 计算 HMAC-SHA256，hex 后与 `v1` 比对；
  4. 拒绝超过 5 分钟的时间戳防重放。
  **必须用原始请求体字节**，解析再序列化会破坏 key 顺序/数字格式导致验签失败。
- 生产 handler 应：验签 → 存状态 → 快速返回 2xx → 把下载/转码/入库丢给后台 worker。
- 注意：job 在早期校验失败时，`generation_id` 和 `model` 可能为 `null`。

## 8. 生产实践清单

1. **提交即持久化 job ID**（连同 model、prompt 参数、提交时间、attempt 次数等），进程重启后恢复跟踪而不是重新付费提交。
2. **轮询重试与生成重试分开**：状态查询失败重试查询；只有终态失败才按策略重新生成，并给生成重试设上限，保留同一内部请求的所有 job ID。
3. **并发控制**：不要为每个 job 起一个无限轮询循环，用有界 worker 池 / 队列限制并发轮询与下载数。
4. **成本预估**：不要硬编码统一计价公式；提交前查 `/api/v1/videos/models` 的 `pricing_skus`，完成后用 `usage.cost` 对账。
5. **及时转存**：完成后立刻把视频搬到自己的存储。
6. **轮询间隔 30 秒**，整体超时上限（如 1 小时）。
7. **校验参数组合**：`duration`/`resolution`/`aspect_ratio` 对照模型的 `supported_*` 字段，避免 400。

## 9. 端点速查表

| 操作 | 方法 & 路径 | 说明 |
|---|---|---|
| 列出视频模型 | `GET /api/v1/videos/models` | 含能力、定价、透传参数 |
| 提交生成 | `POST /api/v1/videos` | 返回 202 + `id` + `polling_url` |
| 查询进度 | `GET /api/v1/videos/{jobId}` | 状态机：`pending`→`in_progress`→终态 |
| 下载视频 | `GET /api/v1/videos/{jobId}/content?index=0` | 需 API Key，流式下载 |
| Webhook | 请求体 `callback_url` 或工作区默认 | 终态推送，带幂等 key 和 HMAC 签名 |
