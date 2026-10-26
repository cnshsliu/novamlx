# Image Generation

**Pitch:** Generate images on this Mac from a text prompt. FLUX, Qwen-Image, Z-Image, and MOSS-Image share `POST /v1/images/generations` and return a base64 PNG.

The same local image API runs FLUX, Qwen-Image, Z-Image, and MOSS-Image. The output is a Base64 PNG.

## Who it is for

People who want to draw on this machine through the OpenAI image API. MOSS-Image fits people who need Chinese and English text rendered in the image, or a picture from one natural-language sentence (a JSON layout string works too).

## What it does

| Feature | Details |
|---------|---------|
| **Endpoint** | `POST /v1/images/generations`. Edits and variations are `/v1/images/edits` and `/v1/images/variations` |
| **Load** | After download, `nova load <id>`, or the first generations request loads it automatically |
| **Output** | Base64 PNG. The request can include `size`, `steps`, `seed`, `negative_prompt`, and `n` |
| **Playground** | Selecting an image model opens `/demo/image` |

Text-to-image families that are wired up:

| Family | Generation weights in the catalog | Notes |
|--------|-----------------------------------|-------|
| FLUX.1 | `mzbac/flux1.dev.4bit.mlx` | 4-bit. `mzbac/flux1.schnell.4bit.mlx` does not load |
| FLUX.2 Klein | `black-forest-labs/FLUX.2-klein-4B` | 4-step |
| Z-Image-Turbo | `mzbac/Z-Image-Turbo-8bit` | 9-step |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2.1`, `mlx-community/Qwen-Image-2.1-MLX-4bit` | 2.1 also supports image-to-image |
| MOSS-Image | `OpenMOSS-Team/MOSS-Image-SFT` | Use this one for generation. `OpenMOSS-Team/MOSS-Image-Base` is the pretrain on the same load path |

MOSS-Image is a 5B single-stream DiT. The text encoder is Qwen3.5-9B. The decoder is FLUX.2's small VAE. The weights are Apache-2.0. A prompt is an ordinary sentence, or the model's own JSON layout string, passed through to the encoder unchanged.

When `steps` is omitted, MOSS-Image uses the official sampler: Euler flow matching, 50 steps, CFG 3, shift 5, an empty negative prompt, and the negative branch skips layer 22. Width and height snap to multiples of 16, at least 256 on each side, with area at most 1456×1456. Denoising uses the per-layer `eval` path in moss-image-swift 0.3.0.

The download is about 27 GB (safetensors total 27,199,664,580 bytes). At 48 GB of unified memory and above, the load stays bf16, with a peak around 27 GB. Below 48 GB, Nova quantizes the DiT and the text encoder to int8 on load, with a peak around 16 GB. The catalog's suggested memory is 32 GB. `NOVA_MOSS_IMAGE_QUANT=bf16` or `int8` overrides the automatic choice.

## Turbo LoRA (Qwen-Image-2.1 few-step)

Qwen-Image-2.1 can take a few-step distilled LoRA that drops the default 40 steps to 6, uses the viggle sampler schedule, and fixes guidance at 1. Generation and edits (`/v1/images/edits`) share this path. When `steps` is omitted, viggle mode defaults to 6 steps, and plain 2.1 defaults to 40 steps.

Example: [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo). This repo is **not in the catalog**. It also carries ~60 GB of GGUF and full-weight variants, and `nova download` pulls the whole repository, so download only the LoRA file by hand (for example `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors`, about 1.3 GB) and place it in a folder under any model root (for example `<model root>/Viggle/Qwen-Image-2.1-viggle-turbo/`).

Discovery and load rules:

- If the folder contains a `.safetensors` whose filename includes both `qwen-image-2.1` and `lora`, and it does **not** contain `vae/config.json` and `text_encoder/config.json` (those mark a full base), it is registered as its own Qwen-Image-2.1 entry. When several files match, the largest is used.
- The LoRA is only an adapter. The base is resolved in order: the environment variable `NOVAMLX_QWEN21_TURBO_BASE`, then `Qwen/Qwen-Image-2.1`, then `mlx-community/Qwen-Image-2.1-MLX-4bit`. If none of them is present, load fails and asks for the base to be installed first.
- Base quantization follows `NOVAMLX_QWEN_IMAGE_21_QUANT` (`4` / `8`), the same as plain 2.1.

## How to try

Generate with the SFT:

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

In the response, `data[0].b64_json` is the PNG. The same preview row is in the menu-bar Downloads catalog. The Playground image page prefills MOSS-Image steps at 50. Lower that for a faster preview.

## Limits / notes

- This MOSS-Image build is text-to-image only. `/v1/images/edits` and `/v1/images/variations` do not accept it.
- The request has no guidance field. MOSS-Image fixes CFG at 3.
- Area above 1456×1456, or a side below 256, is rejected.
- Catalog status is preview.
- `OpenMOSS-Team/MOSS-Image-Base` can be loaded. Generation should still use the SFT.

## Related

- [Supported models](supported-models.md)
- [Multi-modal overview](multi-modal.md)
- [API compatibility](api-compatibility.md)
- [Model architectures](model-architectures.md)
- Weights: [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base), [Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) (LoRA, imported by hand)
