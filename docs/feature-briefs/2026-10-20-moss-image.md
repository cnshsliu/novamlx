# MOSS-Image 文生图

## Title

OpenMOSS MOSS-Image — Apple Silicon 上的本地文生图（SFT）

## One-line pitch

Generate images from a sentence on this Mac with MOSS-Image-SFT, through the same local image API as FLUX.

在 Mac 上用 MOSS-Image-SFT 把一句话画成图，走和 FLUX 同一套本地图像接口。

## Who it is for / when to care

已经在 NovaMLX 里跑本地模型，并希望用 OpenMOSS MOSS-Image 做文生图的人。中英文文字上屏、或把一段 JSON 版式当作提示词，是它相对现有 FLUX / Qwen-Image 路径的差别。统一内存大约 64 GB 时走 bf16；大约 32 GB 时 Nova 会在加载时把 DiT 和文本编码器量化成 int8。

For people running NovaMLX on a Mac who want OpenMOSS MOSS-Image as a local text-to-image model. Use bf16 around 64 GB of unified memory. Around 32 GB, Nova quantizes the DiT and the text encoder to int8 on load.

## What shipped (user-visible behavior)

目录新增两条 preview，id 与 `nova download` 相同：

| 仓库 | 用途 | 磁盘 | 目录建议统一内存 |
| --- | --- | --- | --- |
| `OpenMOSS-Team/MOSS-Image-SFT` | 生成用这一条 | ~27 GB（safetensors 27,199,664,580 字节） | 32 GB |
| `OpenMOSS-Team/MOSS-Image-Base` | 同一加载路径上的预训练权重 | ~27 GB | 32 GB |

下载的是 bf16 diffusers 快照（`transformer/`、`text_encoder/`、`tokenizer/`、`vae/`）。加载规则：

- 统一内存 ≥ 48 GiB：DiT 与 Qwen3.5-9B 文本编码器保持 bf16。接入的 Swift 端口公布的峰值大约 27 GB（编码阶段）。
- 统一内存低于 48 GiB：加载时把 DiT 和文本编码器量化为 int8。同一端口公布的峰值大约 16 GB。
- `NOVA_MOSS_IMAGE_QUANT=bf16` 或 `int8` 覆盖上面的自动选择。名字里带 `int8` / `8bit` 的 id 默认 int8。

这些峰值是 moss-image-swift 0.3.0 公布的测量，不是这次在 NovaMLX 里重新采样的数字。

生成走已有的 `POST /v1/images/generations`。未传 `steps` 时：Euler、50 步、CFG 3、shift 5、空负面提示、负面分支跳过第 22 层。宽和高落到 16 的倍数，每边至少 256，面积不超过 1456×1456。提示词可以是自然语言，也可以是 JSON 版式字符串。Playground 图像页（`/demo/image`）会把 MOSS-Image 的步数预填成 50。

编辑和变体接口会拒绝 MOSS-Image，并说明改用 generations。

## How to try it

需要已运行的 NovaMLX（Apple Silicon）。推理 API 默认端口 6590。

```bash
nova download OpenMOSS-Team/MOSS-Image-SFT
nova load OpenMOSS-Team/MOSS-Image-SFT
```

菜单栏 Downloads 的目录里也能看到 `MOSS-Image-SFT`，带 preview 标记。

```bash
curl http://127.0.0.1:6590/v1/images/generations \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"OpenMOSS-Team/MOSS-Image-SFT","prompt":"a red paper lantern on a wooden table, with the Chinese character 春 on the lantern","size":"1024x1024"}'
```

响应 `data[0].b64_json` 是 PNG。想少等可以在 JSON 里加 `"steps": 20`。32 GB 机器应自动走 int8；64 GB 机器应走 bf16。强制指定：启动前设置 `NOVA_MOSS_IMAGE_QUANT=int8` 或 `bf16`。

不要对这个模型调用 `/v1/images/edits` 或 `/v1/images/variations`。

## Limits / out of scope

- 不做图生图、遮罩编辑、变体。
- 没有提示词增强，也没有云端 API。
- 没有训练或 LoRA 界面。Base 权重只是可以加载，视频里不要把它讲成另一套产品。
- 不做视频生成。
- 请求里不能改 CFG。固定为 3。
- 目录状态是 preview。这次没有在 Nova 里重跑出图计时或内存曲线。

No image edits, no prompt enhancer, no training UI, no video.

## Suggested video beats

- 开场一句：MOSS-Image on this Mac, same image API as FLUX。本地、文生图、Apple Silicon。
- 在 Downloads 目录里指出 `OpenMOSS-Team/MOSS-Image-SFT`，大约 27 GB，preview。
- 下载并加载 SFT。一句带中文的提示词，1024 方图，步数 50。
- 点出结果是 PNG，文字在画面里。
- 收在内存上：48 GB 及以上 bf16，更小的机器 int8；编辑和视频不在这次。

## Links

- 用户文档：[docs/features/image-generation.md](../features/image-generation.md)
- 目录：[docs/features/supported-models.md](../features/supported-models.md)
- SFT: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
- Base: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- 上游: https://github.com/OpenMOSS/MOSS-Image
- Swift 端口: https://github.com/xocialize/moss-image-swift （MossImage core，不是 MLXEngine 包装）
- 目录数据：`catalog/models.json`
- 产品日志：根目录 `CHANGELOG.md` 的 `[Unreleased]`
