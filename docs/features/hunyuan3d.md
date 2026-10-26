# 3D Generation (Hunyuan3D)

**Pitch:** Submit an image-to-3D job to the local Hunyuan3D-2.1 pipeline, then poll and download a GLB mesh (optionally textured).

Local Hunyuan3D-2.1 (Swift + MLX, on this machine, no Python) runs async jobs through `Hy3DService`: one image → a mesh (optional PBR texture) → a GLB.

## Who it is for

People who want to turn one image into a 3D model (GLB) on this machine, or who want to call 3D generation through the same task-style API as image and video.

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

1. Download and load a model (search the catalog for `hunyuan3d`, or use the Downloads page):

```bash
nova download zimengxiong/hunyuan3d-mlx-shape-small
# Then load zimengxiong/hunyuan3d-mlx-shape-small on the Local Inference page
```

2. Submit a job (`image` accepts a data URI, raw base64, an http(s) URL, or a local path):

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

3. Poll and download:

```bash
curl -H "Authorization: Bearer $NOVA_API_KEY" http://localhost:6590/v1/tasks/h3d-...
# status: pending → in_progress (progress.phase = shape) → completed
curl -H "Authorization: Bearer $NOVA_API_KEY" \
  -o out.glb http://localhost:6590/v1/3d/generations/h3d-.../content
```

## Texture (paint)

- `texture: true` needs a paint model loaded as well (for example `zimengxiong/hunyuan3d-mlx-paint-large`), or an explicit `paint_model` on the request.
- paint-large takes the PBR path (albedo + metallic-roughness maps). paint-small takes the RGB path.
- Peak memory is ~25–39 GB. 64 GB and above is the suggested machine size. If memory is short, the job fails cleanly.
- The shape model stays resident until unload. Paint weights load for each job and are released when it finishes.

## Limits / notes

- This round is image-to-3D only. The `prompt` field is accepted and ignored. `output_format` supports `glb` only.
- shape-small measured reference (upstream README, M-series chips): 30 steps + octree 256 is about 21 s, peak ~5.6 GB.
- `usage.cost` is always 0 (generated on this machine). `meshes` / `size_tier` are reserved for upstream metering.
- Weights follow the Tencent Hunyuan3D Community License (converted from official tencent/Hunyuan3D-2.1) and are not in git.

## Related

- [Image generation](image-generation.md)
- [Video generation](video-generation.md)
- [API compatibility](api-compatibility.md)
- `Hy3DService` (`NovaMLXEngine`), `APIServer+3D.swift` (`NovaMLXAPI`)
