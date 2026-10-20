<!-- Generated from catalog/models.json by Scripts/generate-supported-models-doc.py. Do not edit the tables by hand. -->

# Supported models

**Pitch:** Download an exact catalog weight id (`org/repo`), the same string as `nova download`.

「支持」指「精确权重」一节中的 id。该 id 与 `nova download <id>` 相同。名字相近而 id 不同的仓库可能无法加载。

本页与应用内目录一致，数据日期 2026-10-20。`verified` 为已测条目；`preview` 在目录中，尚未标成 verified。`unsupported` 列在文末，NovaMLX 不会下载或加载。

## 下载 / Download

```bash
nova download <id>
nova search "qwen"
```

镜像在应用的下载页选择：Official（huggingface.co）、ModelScope，或自定义端点。ModelScope 与 Hugging Face 使用同一个 id。

Hugging Face 上的可选链接由 id 拼出：`https://huggingface.co/<id>`。下载、取消与续传见 [Hugging Face](huggingface.md)。文件落盘见 [模型管理](model-management.md)。

## 精确权重 / Exact weights

下列 id 各对应一个仓库。

<a id="llm"></a>

### 文本 LLM / Text LLM

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| Nemotron 3.5 Bolt 74B-A9B 4-bit | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit` | ~43 GB (4-bit) | preview | 64 GB | `nova download rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit` | [Hugging Face](https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit) |
| Nemotron 3.5 Bolt 74B-A9B 3-bit | `rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit` | ~32 GB (3-bit) | preview | 48 GB | `nova download rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit` | [Hugging Face](https://huggingface.co/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-3bit) |
| DeepSeek-V4.1-Flash DSpark drafter | `mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter` | ~4.5 GB (2-bit) | verified | 16 GB | `nova download mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter` | [Hugging Face](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter) |
| Hy4-preview | `mlx-community/Hy4-preview-4bit` | ~404 GB (4-bit) | preview | 128 GB | `nova download mlx-community/Hy4-preview-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Hy4-preview-4bit) |
| Gemma-4-12B | `mlx-community/gemma-4-12B-it-OptiQ-4bit` | ~7 GB | preview | — | `nova download mlx-community/gemma-4-12B-it-OptiQ-4bit` | [Hugging Face](https://huggingface.co/mlx-community/gemma-4-12B-it-OptiQ-4bit) |
| Qwen3.5-9B | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | ~5 GB | preview | — | `nova download mlx-community/Qwen3.5-9B-OptiQ-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit) |
| GPT-OSS-20B | `mlx-community/gpt-oss-20b-MXFP4-Q8` | ~13 GB | preview | — | `nova download mlx-community/gpt-oss-20b-MXFP4-Q8` | [Hugging Face](https://huggingface.co/mlx-community/gpt-oss-20b-MXFP4-Q8) |
| Qwen3.8-27B DFlash2 drafter | `incoai/Qwen3.8-27B-DFlash2` | ~1.5 GB | verified | — | `nova download incoai/Qwen3.8-27B-DFlash2` | [Hugging Face](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) |
| Qwen3.6-27B | `mlx-community/Qwen3.6-27B-OptiQ-4bit` | 15.2 GB | preview | — | `nova download mlx-community/Qwen3.6-27B-OptiQ-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-27B-OptiQ-4bit) |
| Gemma-4-26B-A4B | `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | ~14 GB | preview | — | `nova download mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | [Hugging Face](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit) |
| GLM-4.7-Flash | `mlx-community/GLM-4.7-Flash-4bit` | ~3 GB | preview | — | `nova download mlx-community/GLM-4.7-Flash-4bit` | [Hugging Face](https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit) |
| Gemma-4-31B | `mlx-community/gemma-4-31B-it-OptiQ-4bit` | ~17 GB | preview | — | `nova download mlx-community/gemma-4-31B-it-OptiQ-4bit` | [Hugging Face](https://huggingface.co/mlx-community/gemma-4-31B-it-OptiQ-4bit) |
| Qwen3.6-35B-A3B | `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` | ~18 GB | preview | — | `nova download mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit) |
| Nemotron-Cascade-2-30B | `mlx-community/Nemotron-Cascade-2-30B-A3B-4bit` | ~16 GB | preview | — | `nova download mlx-community/Nemotron-Cascade-2-30B-A3B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Nemotron-Cascade-2-30B-A3B-4bit) |
| MiniMax-M3 | `mlx-community/MiniMax-M3-4bit` | ~20 GB | preview | — | `nova download mlx-community/MiniMax-M3-4bit` | [Hugging Face](https://huggingface.co/mlx-community/MiniMax-M3-4bit) |
| Granite-4.1-8B | `mlx-community/granite-4.1-8b-8bit` | ~8 GB | preview | — | `nova download mlx-community/granite-4.1-8b-8bit` | [Hugging Face](https://huggingface.co/mlx-community/granite-4.1-8b-8bit) |
| Kimi-K2.7-Code | `mlx-community/Kimi-K2.7-Code-4bit` | ~10 GB | preview | — | `nova download mlx-community/Kimi-K2.7-Code-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Kimi-K2.7-Code-4bit) |
| GPT-OSS-120B | `mlx-community/gpt-oss-120b-MXFP4-Q8` | ~75 GB | preview | — | `nova download mlx-community/gpt-oss-120b-MXFP4-Q8` | [Hugging Face](https://huggingface.co/mlx-community/gpt-oss-120b-MXFP4-Q8) |
| DeepSeek-V4-Flash | `mlx-community/DeepSeek-V4-Flash-4bit` | ~30 GB | preview | — | `nova download mlx-community/DeepSeek-V4-Flash-4bit` | [Hugging Face](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit) |

<a id="vlm"></a>

### 视觉语言 / Vision-language

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| DeepSeek-V4.1-Flash | `mlx-community/DeepSeek-V4.1-Flash-MLX-2bit` | ~239 GB (2-bit) | verified | 128 GB | `nova download mlx-community/DeepSeek-V4.1-Flash-MLX-2bit` | [Hugging Face](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-MLX-2bit) |
| Qwythos-27B-v1 | `Shiftedx/Qwythos-27B-v1-MXFP4-Vision-MLX` | ~15.2 GB (MXFP4) | verified | 24 GB | `nova download Shiftedx/Qwythos-27B-v1-MXFP4-Vision-MLX` | [Hugging Face](https://huggingface.co/Shiftedx/Qwythos-27B-v1-MXFP4-Vision-MLX) |
| Qwen3.8-Flash-Next Uncensored | `orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX` | ~163 GB (4-bit) | preview | 128 GB | `nova download orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX` | [Hugging Face](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX) |
| Qwen3.8-27B Uncensored | `orcarouter/Qwen3.8-27B-Uncensored-MLX` | ~15 GB (4-bit) | verified | 24 GB | `nova download orcarouter/Qwen3.8-27B-Uncensored-MLX` | [Hugging Face](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX) |
| Qwen3.8-27B 8-bit | `mlx-community/Qwen3.8-27B-8bit` | ~28 GB (8-bit) | verified | 32 GB | `nova download mlx-community/Qwen3.8-27B-8bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit) |
| Qwen3.8-27B 4-bit | `mlx-community/Qwen3.8-27B-4bit` | ~16 GB (4-bit) | verified | 24 GB | `nova download mlx-community/Qwen3.8-27B-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) |
| Gemma-4-26B-A4B-VL | `mlx-community/gemma-4-26b-a4b-it-4bit` | 14.5 GB | preview | — | `nova download mlx-community/gemma-4-26b-a4b-it-4bit` | [Hugging Face](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) |
| Qwen3-VL-4B | `lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit` | ~2.5 GB | preview | — | `nova download lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit` | [Hugging Face](https://huggingface.co/lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit) |
| Qwen3-VL-8B | `lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit` | ~5 GB | preview | — | `nova download lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit` | [Hugging Face](https://huggingface.co/lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit) |
| Qwen3-VL-30B-A3B | `lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit` | ~17 GB | preview | — | `nova download lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit` | [Hugging Face](https://huggingface.co/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-MLX-4bit) |

<a id="audio"></a>

### 音频 / Audio

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| Whisper-Large-V3-Turbo | `mlx-community/whisper-large-v3-turbo` | 0.8 GB | preview | — | `nova download mlx-community/whisper-large-v3-turbo` | [Hugging Face](https://huggingface.co/mlx-community/whisper-large-v3-turbo) |
| Whisper-Small | `mlx-community/whisper-small` | 0.24 GB | preview | — | `nova download mlx-community/whisper-small` | [Hugging Face](https://huggingface.co/mlx-community/whisper-small) |
| Dots.TTS | `smcleod/dots.tts-soar-mlx` | 4.0 GB | preview | — | `nova download smcleod/dots.tts-soar-mlx` | [Hugging Face](https://huggingface.co/smcleod/dots.tts-soar-mlx) |
| Qwen3-TTS 1.7B Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | 3.1 GB | verified | — | `nova download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |
| Qwen3-ASR-1.7B | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3.2 GB | preview | — | `nova download mlx-community/Qwen3-ASR-1.7B-8bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) |

<a id="image"></a>

### 图像 / Image

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| FLUX.1-dev-4bit | `mzbac/flux1.dev.4bit.mlx` | ~6 GB | preview | — | `nova download mzbac/flux1.dev.4bit.mlx` | [Hugging Face](https://huggingface.co/mzbac/flux1.dev.4bit.mlx) |
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4B` | ~16 GB | preview | — | `nova download black-forest-labs/FLUX.2-klein-4B` | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| Z-Image-Turbo-8bit | `mzbac/Z-Image-Turbo-8bit` | ~7.5 GB | preview | — | `nova download mzbac/Z-Image-Turbo-8bit` | [Hugging Face](https://huggingface.co/mzbac/Z-Image-Turbo-8bit) |
| Qwen-Image | `Qwen/Qwen-Image` | ~40 GB | preview | — | `nova download Qwen/Qwen-Image` | [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` | ~33 GB | preview | 64 GB | `nova download Qwen/Qwen-Image-2.1` | [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| Qwen-Image-2.1 4-bit | `mlx-community/Qwen-Image-2.1-MLX-4bit` | ~10.5 GB / 4bit | preview | 32 GB | `nova download mlx-community/Qwen-Image-2.1-MLX-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit) |

<a id="embedding"></a>

### 嵌入 / Embedding

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| Qwen3-Embedding-0.6B | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | ~335 MB / 4bit | verified | 4 GB | `nova download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) |

<a id="decision"></a>

### 决策 / Decision

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| Laya Multilingual | `aac6fef/laya-multilingual-mlx` | ~700 MB / fp16 | verified | 4 GB | `nova download aac6fef/laya-multilingual-mlx` | [Hugging Face](https://huggingface.co/aac6fef/laya-multilingual-mlx) |

## 家族浏览 / Family browse

这些条目的 id 以 `*` 结尾，或对应 Hugging Face 搜索页，表示一族权重。用 `nova search` 列出具体 `org/repo`，再 `nova download` 那个 id。「精确权重」一节只收录具体 id。

| 名称 | Catalog id | 大小 / quant | 状态 | 最低内存 | 浏览 |
|------|------------|--------------|------|----------|------|
| Ornith-1.5-35B-A3B family | `ornith-ai/Ornith-1.5-35B-A3B-MLX*` | ~34 GB / 8bit | preview | 48 GB | `nova search Ornith-1.5-35B-A3B-MLX` |
| Qwen3.8-Flash-Next | `pipenetwork/Qwen3.8-Flash-Next-*` | ~104 GB (4-bit) | preview | 128 GB | `nova search Qwen3.8-Flash-Next` |
| Qwen3.8-27B MTP heads | `mlx-community/Qwen3.8-27B-MTP-*` | ~0.3 GB | preview | — | `nova search Qwen3.8-27B-MTP` |

## 不加载 / Not loaded

状态为 `unsupported` 的 id 留在目录里，便于辨认。NovaMLX 不下载、不加载。

| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |
|------|-----------|--------------|------|----------|------|--------------|
| FLUX.1-schnell-4bit | `mzbac/flux1.schnell.4bit.mlx` | ~6 GB | unsupported | — | 不下载 | [Hugging Face](https://huggingface.co/mzbac/flux1.schnell.4bit.mlx) |

- `mzbac/flux1.schnell.4bit.mlx` — Not supported. Quantized FLUX.1 schnell crashes in VAE decode on NovaMLX. Use FLUX.2 Klein instead.

表格由 [`catalog/models.json`](../../catalog/models.json) 生成（`updatedAt` 2026-10-20）。重新生成：`python3 Scripts/generate-supported-models-doc.py`。
