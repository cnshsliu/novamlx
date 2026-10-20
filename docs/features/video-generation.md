# Video Generation

**Pitch:** Submit an OpenRouter-style video job to the local h3.c binary, then poll and download the mp4.

本地 h3.c（MiniMax-H3，音视频一起出）通过 `H3VideoService` 跑异步任务。口播画面走 `TalkingHeadService`。

## Who it is for

要用 OpenRouter 形态的 `/v1/videos` 在本机排队生成视频的人。

## What it does

此节只在英文 `features.md` 中。中文目录没有对应章节。

| Feature | Details |
|---------|---------|
| **Backend** | Local h3.c CLI (MiniMax-H3, joint audio+video) via `H3VideoService`; talking-head pipeline via `TalkingHeadService` |
| **API shape** | OpenRouter-compatible async jobs: submit → poll → download |
| **Endpoints** | `POST /v1/videos` (202 + job id), `GET /v1/videos/{id}` (poll), `GET /v1/videos/{id}/content` (mp4 download), `GET /v1/videos/models`, `POST /v1/videos/{id}/cancel` (queued jobs) |
| **Talking-head** | `POST /v1/video/talking-head`, `GET /v1/video/progress` |
| **Execution** | Bounded FIFO queue (full → 429), one h3 job at a time, `SafeProcess` with deadline |
| **Persistence** | File-based jobs under `~/.nova/videos/<id>/`; restart recovery; TTL expiry |
| **Webhooks** | `callback_url` POST on terminal state; `X-OpenRouter-Idempotency-Key` + optional HMAC-SHA256 signature |
| **Config** | `~/.nova/h3_video.json` + `NOVA_H3_*` env overrides (binary path, weights dir, steps/layers/reuse, timeouts) |
| **Notes** | `usage.cost` always 0 (local); `generate_audio` ignored (h3 always emits AAC); spec in `docs/research/openrouter-video-generation-api.md` |

## How to try

`POST /v1/videos` 接受任务并返回 202 与 job id，再 `GET /v1/videos/{id}` 轮询，完成后 `GET /v1/videos/{id}/content` 下载 mp4。排队中的任务可用 `POST /v1/videos/{id}/cancel` 取消。

口播：`POST /v1/video/talking-head`，进度 `GET /v1/video/progress`。

配置文件 `~/.nova/h3_video.json`，环境变量前缀 `NOVA_H3_*`（二进制路径、权重目录、steps / layers / reuse、超时）。

## Limits / notes

- 队列有上界，满了返回 429。同一时刻只跑一个 h3 任务。
- `usage.cost` 恒为 0。`generate_audio` 被忽略，因为 h3 总会出 AAC。
- 任务文件在 `~/.nova/videos/<id>/`，进程重启后可恢复，并有 TTL。

## Related

- [API 兼容性](api-compatibility.md)
- [图像生成](image-generation.md)
- `docs/research/openrouter-video-generation-api.md`
- `H3VideoService`、`TalkingHeadService`（`NovaMLXEngine`）

语言：英文 `features.md`「Video Generation」。中文功能目录未单列此节。
