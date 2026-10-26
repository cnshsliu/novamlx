<!-- Generated from catalog/models.json by Scripts/generate-supported-models-doc.py. Do not edit the tables by hand. -->

# Supported models

**Pitch:** Download an exact catalog weight id (`org/repo`), the same string as `nova download`.

"Supported" means an id in [Exact weights](#exact-weights). That id is the same string as `nova download <id>`. A repo with a similar name and a different id may fail to load.

This page matches the in-app catalog. Data date 2026-10-26. `verified` entries have been tested; `preview` is in the catalog and is not marked verified. `unsupported` is listed at the end, and NovaMLX will not download or load it.

## Download

```bash
nova download <id>
nova search "qwen"
```

Pick a mirror on the app's Downloads page: Official (huggingface.co), ModelScope, or a custom endpoint. ModelScope and Hugging Face use the same id.

Optional Hugging Face links are built from the id: `https://huggingface.co/<id>`. Download, cancel, and resume are covered in [Hugging Face](huggingface.md). Where files land is covered in [Model management](model-management.md).

<a id="exact-weights"></a>

## Exact weights

Each id below is one repository.

<a id="llm"></a>

### Text LLM

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
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
| Flint 40B A2.5B (Official INT4) | `RecursiveSI/flint-int4` | ~22.6 GB / int4 | verified | 48 GB | `nova download RecursiveSI/flint-int4` | [Hugging Face](https://huggingface.co/RecursiveSI/flint-int4) |

<a id="vlm"></a>

### Vision-language

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
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

### Audio

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
| Whisper-Large-V3-Turbo | `mlx-community/whisper-large-v3-turbo` | 0.8 GB | preview | — | `nova download mlx-community/whisper-large-v3-turbo` | [Hugging Face](https://huggingface.co/mlx-community/whisper-large-v3-turbo) |
| Whisper-Small | `mlx-community/whisper-small` | 0.24 GB | preview | — | `nova download mlx-community/whisper-small` | [Hugging Face](https://huggingface.co/mlx-community/whisper-small) |
| Dots.TTS | `smcleod/dots.tts-soar-mlx` | 4.0 GB | preview | — | `nova download smcleod/dots.tts-soar-mlx` | [Hugging Face](https://huggingface.co/smcleod/dots.tts-soar-mlx) |
| Qwen3-TTS 1.7B Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | 3.1 GB | verified | — | `nova download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit) |
| Qwen3-ASR-1.7B | `mlx-community/Qwen3-ASR-1.7B-8bit` | 3.2 GB | preview | — | `nova download mlx-community/Qwen3-ASR-1.7B-8bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit) |

<a id="image"></a>

### Image

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
| FLUX.1-dev-4bit | `mzbac/flux1.dev.4bit.mlx` | ~6 GB | preview | — | `nova download mzbac/flux1.dev.4bit.mlx` | [Hugging Face](https://huggingface.co/mzbac/flux1.dev.4bit.mlx) |
| FLUX.2-klein-4B | `black-forest-labs/FLUX.2-klein-4B` | ~16 GB | preview | — | `nova download black-forest-labs/FLUX.2-klein-4B` | [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) |
| Z-Image-Turbo-8bit | `mzbac/Z-Image-Turbo-8bit` | ~7.5 GB | preview | — | `nova download mzbac/Z-Image-Turbo-8bit` | [Hugging Face](https://huggingface.co/mzbac/Z-Image-Turbo-8bit) |
| Qwen-Image | `Qwen/Qwen-Image` | ~40 GB | preview | — | `nova download Qwen/Qwen-Image` | [Hugging Face](https://huggingface.co/Qwen/Qwen-Image) |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` | ~33 GB | preview | 64 GB | `nova download Qwen/Qwen-Image-2.1` | [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| Qwen-Image-2.1 4-bit | `mlx-community/Qwen-Image-2.1-MLX-4bit` | ~10.5 GB / 4bit | preview | 32 GB | `nova download mlx-community/Qwen-Image-2.1-MLX-4bit` | [Hugging Face](https://huggingface.co/mlx-community/Qwen-Image-2.1-MLX-4bit) |
| MOSS-Image-SFT | `OpenMOSS-Team/MOSS-Image-SFT` | ~27 GB / bf16 | preview | 32 GB | `nova download OpenMOSS-Team/MOSS-Image-SFT` | [Hugging Face](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) |
| MOSS-Image-Base | `OpenMOSS-Team/MOSS-Image-Base` | ~27 GB / bf16 | preview | 32 GB | `nova download OpenMOSS-Team/MOSS-Image-Base` | [Hugging Face](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) |
| BiRefNet | `ZhengPeng7/BiRefNet` | ~440 MB | preview | 8 GB | `nova download ZhengPeng7/BiRefNet` | [Hugging Face](https://huggingface.co/ZhengPeng7/BiRefNet) |
| BiRefNet-portrait | `ZhengPeng7/BiRefNet-portrait` | ~885 MB | preview | 8 GB | `nova download ZhengPeng7/BiRefNet-portrait` | [Hugging Face](https://huggingface.co/ZhengPeng7/BiRefNet-portrait) |

<a id="embedding"></a>

### Embedding

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
| Qwen3-Embedding-0.6B | `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | ~335 MB / 4bit | verified | 4 GB | `nova download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` | [Hugging Face](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ) |

<a id="decision"></a>

### Decision

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
| Laya Multilingual | `aac6fef/laya-multilingual-mlx` | ~700 MB / fp16 | verified | 4 GB | `nova download aac6fef/laya-multilingual-mlx` | [Hugging Face](https://huggingface.co/aac6fef/laya-multilingual-mlx) |

<a id="threed"></a>

### threed

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
| Hunyuan3D-2.1 Shape Small (MLX) | `zimengxiong/hunyuan3d-mlx-shape-small` | ~3.8 GB | preview | 16 GB | `nova download zimengxiong/hunyuan3d-mlx-shape-small` | [Hugging Face](https://huggingface.co/zimengxiong/hunyuan3d-mlx-shape-small) |
| Hunyuan3D-2.1 Shape Large (MLX) | `zimengxiong/hunyuan3d-mlx-shape-large` | ~4.9 GB | preview | 24 GB | `nova download zimengxiong/hunyuan3d-mlx-shape-large` | [Hugging Face](https://huggingface.co/zimengxiong/hunyuan3d-mlx-shape-large) |
| Hunyuan3D-2.1 Paint PBR Large (MLX) | `zimengxiong/hunyuan3d-mlx-paint-large` | ~8.7 GB | preview | 64 GB | `nova download zimengxiong/hunyuan3d-mlx-paint-large` | [Hugging Face](https://huggingface.co/zimengxiong/hunyuan3d-mlx-paint-large) |

## Family browse

These ids end in `*`, or point at a Hugging Face search page, and stand for a family of weights. Use `nova search` to list a concrete `org/repo`, then `nova download` that id. [Exact weights](#exact-weights) lists only concrete ids.

| Name | Catalog id | Size / quant | Status | Minimum memory | Browse |
|------|------------|--------------|--------|----------------|--------|
| Ornith-1.5-35B-A3B family | `ornith-ai/Ornith-1.5-35B-A3B-MLX*` | ~34 GB / 8bit | preview | 48 GB | `nova search Ornith-1.5-35B-A3B-MLX` |
| Qwen3.8-Flash-Next | `pipenetwork/Qwen3.8-Flash-Next-*` | ~104 GB (4-bit) | preview | 128 GB | `nova search Qwen3.8-Flash-Next` |
| Qwen3.8-27B MTP heads | `mlx-community/Qwen3.8-27B-MTP-*` | ~0.3 GB | preview | — | `nova search Qwen3.8-27B-MTP` |

## Not loaded

Ids with status `unsupported` stay in the catalog so they can be recognized. NovaMLX does not download or load them.

| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |
|------|-----------|--------------|--------|----------------|----------|--------------|
| FLUX.1-schnell-4bit | `mzbac/flux1.schnell.4bit.mlx` | ~6 GB | unsupported | — | will not download | [Hugging Face](https://huggingface.co/mzbac/flux1.schnell.4bit.mlx) |

- `mzbac/flux1.schnell.4bit.mlx` — Not supported. Quantized FLUX.1 schnell crashes in VAE decode on NovaMLX. Use FLUX.2 Klein instead.

Tables are generated from [`catalog/models.json`](../../catalog/models.json) (`updatedAt` 2026-10-26). Regenerate with `python3 Scripts/generate-supported-models-doc.py`.
