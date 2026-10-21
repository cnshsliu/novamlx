# Image Matting

**Pitch:** Remove the background from any image on this Mac. BiRefNet gives a soft-alpha matte (通用物体) or cleanest person edges (portrait 微调), returned as a grayscale mask and a transparent cutout.

同一套本地接口跑通用 BiRefNet 和人像微调 BiRefNet-portrait，输出灰度 mask 与透明底 cutout。

## Who it is for

要在本机抠图、去背景、出透明底 PNG 的人。拍人像用 portrait 版，边缘（发丝、轮廓）最干净；物品、商品、混合内容用通用版。

## What it does

| 功能 | 详情 |
|------|------|
| **端点** | `POST /v1/images/matting`，body 为 JSON：`{"model": "...", "image": "<base64>", "include": ["mask","cutout"], "response_format": "b64_json"}` |
| **输出** | `data[]` 里每项带 `kind`（`mask` 灰度白=前景 / `cutout` 原图+alpha 透明底），`b64_json` 或 `url` |
| **加载** | 下载后 `nova load <id>`，或第一次 matting 请求时自动加载 |
| **分辨率** | 模型在 1024×1024 上推理，mask/cutout 双线性回缩到原图分辨率 |
| **首次加载** | 官方 PyTorch 权重一次性转换为 MLX fp16 缓存（`model.mlx-fp16.safetensors`，与原文件同目录），之后直接加载缓存 |

已接入的权重：

| 家族 | 权重 | 说明 |
|------|------|------|
| 通用 | `ZhengPeng7/BiRefNet` | 任意显著物体。MIT。~440 MB（fp16） |
| 人像 | `ZhengPeng7/BiRefNet-portrait` | 人像微调，脸/发丝/身体边缘最干净。MIT。~885 MB（fp32，加载时转 fp16） |

两者架构相同（swin_v1_l 主干 + 定位/重建模块 + ASPP-Deformable 解码器，内置 DCNv2 Metal kernel），加载与推理路径完全一致，按内容选模型即可。

内存：权重约 420 MB（fp16），1024² 推理峰值约 5 GB，8 GB 统一内存即可。

## How to try

```bash
nova download ZhengPeng7/BiRefNet-portrait
```

```bash
IMG=$(base64 -i photo.jpg | tr -d '\n')
curl http://127.0.0.1:6590/v1/images/matting \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"ZhengPeng7/BiRefNet-portrait\",\"image\":\"$IMG\"}" \
  | python3 -c "import json,sys,base64; d=json.load(sys.stdin); [open(x['kind']+'.png','wb').write(base64.b64decode(x['b64_json'])) for x in d['data']]"
```

得到 `mask.png`（灰度）与 `cutout.png`（透明底）。只要其中一个时传 `"include": ["cutout"]`。

## Limits / notes

- 只做抠图。`/v1/images/generations` 传入 BiRefNet 模型会被拒绝（提示改用 matting 端点）。
- 输出是软 alpha matte；不做颜色去污（color decontamination）与边缘细化后处理。
- 输入是 RGB 图；CMYK/调色板图会先经 CoreGraphics 转 RGB。
- 目录状态是 preview（两个检查点的加载与出图已在本地真权重上验证）。

## Related

- [支持的模型](supported-models.md)
- [图像生成](image-generation.md)
- [API 兼容性](api-compatibility.md)
- 权重：[ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)、[ZhengPeng7/BiRefNet-portrait](https://huggingface.co/ZhengPeng7/BiRefNet-portrait)
- 核心移植来自 [mnmly/mlx-swift-BiRefNet](https://github.com/mnmly/mlx-swift-BiRefNet)（MIT，vendored 于 `vendors/birefnet-swift`）
