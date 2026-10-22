# 3D Generation (Hunyuan3D)

**Pitch:** Submit an image-to-3D job to the local Hunyuan3D-2.1 pipeline, then poll and download a GLB mesh (optionally textured).

本地 Hunyuan3D-2.1（Swift + MLX，纯本机无 Python）通过 `Hy3DService` 跑异步任务：一张图 → mesh（可选 PBR 纹理）→ GLB。

## Who it is for

要在本机把一张图片变成 3D 模型（GLB）的人；或想用任务式 API（与 image / video 同一套）做 3D 生成接入的人。

## What it does

| Feature | Details |
|---------|---------|
| **Backend** | Local Swift/MLX port (vendored `hunyuan3d-swift`: Hy3DMLX shape + HunyuanPaintMLX paint); no Python subprocess |
| **Weights** | MLX-format conversions of official `tencent/Hunyuan3D-2.1` (Tencent Hunyuan3D Community License): `zimengxiong/hunyuan3d-mlx-shape-small` / `-shape-large` / `-paint-large` |
| **API shape** | Task-style async jobs, same family as `/v1/videos`: submit (202) → poll → download |
| **Endpoints** | `POST /v1/3d/generations` (202 + job id), `GET /v1/3d/generations/{id}` (poll), `GET /v1/3d/generations/{id}/content` (GLB download), `GET /v1/3d/models`, `POST /v1/3d/generations/{id}/cancel`; every job also answers `GET /v1/tasks/{id}` as `task = "3d.generation"` |
| **Execution** | Serial queue — one 3D job at a time (same philosophy as image tasks); cancel polled per denoise step |
| **Artifacts** | `~/.nova/3d/<job id>/output.glb`; finished jobs queryable 24 h, then `expired` |
| **Progress** | `phase`: `queued → loading → shape → (paint) → export → done`, with `percent` |
| **Usage** | `usage.meshes = 1`, `usage.size_tier = "shape" \| "shape+texture"`, `cost` always 0 (local) — stable metering fields for tknet/peer billing |
| **Config** | `NOVA_HY3D_QUANT=0|4|8` — DiT weight quantization at load (fp16 ≈3.6 GB resident, 4-bit ≈1.4 GB) |

## How to try

1. 下载并加载模型（catalog 搜 `hunyuan3d`，或在 Downloads 页）：

```bash
nova download zimengxiong/hunyuan3d-mlx-shape-small
# 然后在 Local Inference 页加载 zimengxiong/hunyuan3d-mlx-shape-small
```

2. 提交任务（`image` 接受 data URI、裸 base64、http(s) URL 或本地路径）：

```bash
curl -X POST http://localhost:6590/v1/3d/generations \
  -H "Authorization: Bearer $NOVA_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "zimengxiong/hunyuan3d-mlx-shape-small",
    "image": "data:image/png;base64,....",
    "texture": false,
    "seed": 42,
    "octree_resolution": 256,
    "num_inference_steps": 30,
    "guidance_scale": 5.0
  }'
# → 202 {"id": "h3d-...", "status": "pending", "polling_url": "/v1/3d/generations/h3d-..."}
```

3. 轮询并下载：

```bash
curl -H "Authorization: Bearer $NOVA_API_KEY" http://localhost:6590/v1/tasks/h3d-...
# status: pending → in_progress (progress.phase = shape) → completed
curl -H "Authorization: Bearer $NOVA_API_KEY" \
  -o out.glb http://localhost:6590/v1/3d/generations/h3d-.../content
```

## Texture (paint)

- `texture: true` 需要同时加载一个 paint 模型（如 `zimengxiong/hunyuan3d-mlx-paint-large`），或请求里显式传 `paint_model`。
- paint-large 走 PBR 路径（albedo + metallic-roughness 贴图）；paint-small 走 RGB 路径。
- 峰值内存 ~25–39 GB：建议 64 GB 以上机型，内存不足会干净地失败。
- shape 模型常驻（直到 unload）；paint 权重每个任务现加载、跑完即释放。

## Limits / notes

- 本轮只做 image-to-3D；`prompt` 字段接受但忽略；`output_format` 只支持 `glb`。
- shape-small 实测参考（上游 README，M 系列芯片）：30 步 + octree 256 约 21 s、峰值 ~5.6 GB。
- `usage.cost` 恒为 0（本机生成）；`meshes` / `size_tier` 为上游计量预留。
- 权重遵循 Tencent Hunyuan3D Community License（由官方 tencent/Hunyuan3D-2.1 转换而来），不入 git。

## Related

- [图像生成 / Image generation](image-generation.md)
- [视频生成 / Video generation](video-generation.md)
- [API 兼容性 / API compatibility](api-compatibility.md)
- `Hy3DService`（`NovaMLXEngine`）、`APIServer+3D.swift`（`NovaMLXAPI`）
