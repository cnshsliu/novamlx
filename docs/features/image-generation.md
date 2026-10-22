# Image Generation

**Pitch:** Generate images on this Mac from a text prompt. FLUX, Qwen-Image, Z-Image, and MOSS-Image share `POST /v1/images/generations` and return a base64 PNG.

同一套本地图像接口跑 FLUX、Qwen-Image、Z-Image 与 MOSS-Image。输出是 Base64 PNG。

## Who it is for

要用 OpenAI 图像接口在本机画图的人。MOSS-Image 适合需要中英文文字上屏、或用一句自然语言（也可以是一段 JSON 版式）出图的人。

## What it does

| 功能 | 详情 |
|------|------|
| **端点** | `POST /v1/images/generations`。编辑与变体是 `/v1/images/edits`、`/v1/images/variations` |
| **加载** | 下载后 `nova load <id>`，或第一次 generations 请求时自动加载 |
| **输出** | Base64 PNG。请求里可以带 `size`、`steps`、`seed`、`negative_prompt`、`n` |
| **Playground** | 选中图像模型后打开 `/demo/image` |

已接入的文生图家族：

| 家族 | 目录里的生成权重 | 说明 |
|------|------------------|------|
| FLUX.1 | `mzbac/flux1.dev.4bit.mlx` | 4-bit。`mzbac/flux1.schnell.4bit.mlx` 不加载 |
| FLUX.2 Klein | `black-forest-labs/FLUX.2-klein-4B` | 4-step |
| Z-Image-Turbo | `mzbac/Z-Image-Turbo-8bit` | 9-step |
| Qwen-Image | `Qwen/Qwen-Image`、`Qwen/Qwen-Image-2.1`、`mlx-community/Qwen-Image-2.1-MLX-4bit` | 2.1 另支持图生图 |
| MOSS-Image | `OpenMOSS-Team/MOSS-Image-SFT` | 生成用这一条。`OpenMOSS-Team/MOSS-Image-Base` 是同一套加载路径上的预训练权重 |

MOSS-Image 是 5B 单流 DiT，文本编码器是 Qwen3.5-9B，解码器是 FLUX.2 的 small VAE。权重 Apache-2.0。提示词是普通句子，也可以是模型自己的 JSON 版式字符串，原样送进编码器。

未写 `steps` 时，MOSS-Image 用官方采样：Euler flow matching、50 步、CFG 3、shift 5、空负面提示、负面分支跳过第 22 层。宽和高会落到 16 的倍数，每边至少 256，面积不超过 1456×1456。DiT 整步编译进一张 MLX 图，按文本长度、latent 格点和 skip 层缓存；CFG 负向分支单独一张图。该形状第一次前向会编译，后面的步复用。`NOVA_MOSS_IMAGE_COMPILE=0` 关掉编译。

下载约 27 GB（safetensors 合计 27,199,664,580 字节）。统一内存达到 48 GB 时按 bf16 加载，峰值大约 27 GB。低于 48 GB 时，Nova 在加载时把 DiT 和文本编码器量化成 int8，峰值大约 16 GB。目录建议内存是 32 GB。环境变量 `NOVA_MOSS_IMAGE_QUANT=bf16` 或 `int8` 可以改掉自动选择。

## Turbo LoRA（Qwen-Image-2.1 few-step）

Qwen-Image-2.1 支持挂 few-step 蒸馏 LoRA，把默认 40 步降到 6 步，走 viggle 采样节点表，guidance 固定 1。生成和编辑（`/v1/images/edits`）共用这条路，未写 `steps` 时 viggle 模式默认 6 步，普通 2.1 默认 40 步。

以 [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) 为例。这一条**不在目录里**：该仓库还带 ~60 GB 的 GGUF 与完整权重变体，而 `nova download` 会拉整个仓库，所以手动只下 LoRA 文件（如 `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors`，约 1.3 GB），放进任一模型根目录下的文件夹（例如 `<模型根>/Viggle/Qwen-Image-2.1-viggle-turbo/`）。

发现与加载规则：

- 文件夹里有文件名同时含 `qwen-image-2.1` 和 `lora` 的 `.safetensors`，且**没有** `vae/config.json` 和 `text_encoder/config.json`（有则视作完整底座），就注册成一条独立的 Qwen-Image-2.1 条目；多个匹配文件取最大的。
- LoRA 只是适配器，底座按顺序解析：环境变量 `NOVAMLX_QWEN21_TURBO_BASE` → `Qwen/Qwen-Image-2.1` → `mlx-community/Qwen-Image-2.1-MLX-4bit`。都没有时加载报错，提示先装底座。
- 底座量化跟随 `NOVAMLX_QWEN_IMAGE_21_QUANT`（`4` / `8`），与普通 2.1 一致。

## How to try

生成用 SFT：

```bash
nova download OpenMOSS-Team/MOSS-Image-SFT
nova load OpenMOSS-Team/MOSS-Image-SFT
```

```bash
curl http://127.0.0.1:6590/v1/images/generations \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"OpenMOSS-Team/MOSS-Image-SFT","prompt":"a red paper lantern on a wooden table, with the Chinese character 春 on the lantern","size":"1024x1024"}'
```

响应里 `data[0].b64_json` 是 PNG。菜单栏 Downloads 的目录里也能看到这条 preview。Playground 的图像页会把 MOSS-Image 的步数预填成 50，可以改小做更快的预览。

## Limits / notes

- MOSS-Image 这一版只做文生图。`/v1/images/edits` 和 `/v1/images/variations` 不接受它。
- 请求里没有 guidance 字段。MOSS-Image 固定 CFG 3。
- 面积大于 1456×1456，或某一边小于 256，会直接拒绝。
- 目录状态是 preview。
- `OpenMOSS-Team/MOSS-Image-Base` 可以加载，生成仍建议用 SFT。

## Related

- [支持的模型](supported-models.md)
- [多模态](multi-modal.md)
- [API 兼容性](api-compatibility.md)
- [模型架构](model-architectures.md)
- 权重：[OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)、[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)、[Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)（LoRA，手动导入）
