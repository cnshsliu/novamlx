# MOSS-Image text-to-image

## Title

OpenMOSS MOSS-Image — local text-to-image on Apple Silicon (SFT)

## One-line pitch

Generate images from a sentence on this Mac with MOSS-Image-SFT, through the same local image API as FLUX.

## Who it is for / when to care

People already running local models in NovaMLX who want OpenMOSS MOSS-Image for text-to-image. Chinese and English text rendered in the image, or a JSON layout used as the prompt, is how it differs from the existing FLUX / Qwen-Image path. Around 64 GB of unified memory it stays bf16. Around 32 GB, Nova quantizes the DiT and the text encoder to int8 on load.

## What shipped (user-visible behavior)

Two preview rows were added to the catalog. The ids match `nova download`:

| Repo | Role | Disk | Catalog suggested unified memory |
| --- | --- | --- | --- |
| `OpenMOSS-Team/MOSS-Image-SFT` | Use this one for generation | ~27 GB (safetensors 27,199,664,580 bytes) | 32 GB |
| `OpenMOSS-Team/MOSS-Image-Base` | Pretrain weights on the same load path | ~27 GB | 32 GB |

The download is a bf16 diffusers snapshot (`transformer/`, `text_encoder/`, `tokenizer/`, `vae/`). Load rules:

- Unified memory ≥ 48 GiB: the DiT and the Qwen3.5-9B text encoder stay bf16. The wired Swift port's published peak is about 27 GB (during encode).
- Unified memory below 48 GiB: the DiT and the text encoder are quantized to int8 on load. The same port's published peak is about 16 GB.
- `NOVA_MOSS_IMAGE_QUANT=bf16` or `int8` overrides that automatic choice. An id whose name contains `int8` / `8bit` defaults to int8.

Those peaks are measurements published by moss-image-swift 0.3.0. They were not resampled inside NovaMLX for this ship.

Generation uses the existing `POST /v1/images/generations`. When `steps` is omitted: Euler, 50 steps, CFG 3, shift 5, an empty negative prompt, and the negative branch skips layer 22. Width and height snap to multiples of 16, at least 256 on each side, with area at most 1456×1456. The prompt can be natural language or a JSON layout string. The Playground image page (`/demo/image`) prefills MOSS-Image steps at 50.

The edit and variation endpoints reject MOSS-Image and say to use generations instead.

## How to try it

NovaMLX must already be running (Apple Silicon). The inference API's default port is 6590.

```bash
nova download OpenMOSS-Team/MOSS-Image-SFT
nova load OpenMOSS-Team/MOSS-Image-SFT
```

`MOSS-Image-SFT` is also in the menu-bar Downloads catalog, marked preview.

```bash
curl http://127.0.0.1:6590/v1/images/generations \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"OpenMOSS-Team/MOSS-Image-SFT","prompt":"a red paper lantern on a wooden table, with the Chinese character 春 on the lantern","size":"1024x1024"}'
```

The response `data[0].b64_json` is a PNG. To wait less, add `"steps": 20` to the JSON. A 32 GB machine should take int8 automatically. A 64 GB machine should take bf16. To force it, set `NOVA_MOSS_IMAGE_QUANT=int8` or `bf16` before launch.

Do not call `/v1/images/edits` or `/v1/images/variations` on this model.

## Limits / out of scope

- No image-to-image, mask edits, or variations.
- No prompt enhancer, and no cloud API.
- No training or LoRA UI. The Base weights can be loaded. Do not describe them in a video as a second product.
- No video generation.
- CFG cannot be changed on the request. It is fixed at 3.
- Catalog status is preview. Image timing and memory curves were not re-run inside Nova for this ship. No image edits, no prompt enhancer, no training UI, and no video.

## Suggested video beats

- Opening line: MOSS-Image on this Mac, same image API as FLUX. Local, text-to-image, Apple Silicon.
- In the Downloads catalog, point at `OpenMOSS-Team/MOSS-Image-SFT`, about 27 GB, preview.
- Download and load the SFT. One prompt that includes Chinese text, a 1024 square, 50 steps.
- Point out that the result is a PNG and the text is in the picture.
- Close on memory: bf16 at 48 GB and above, int8 on smaller machines. Edits and video are not in this ship.

## Links

- PR: https://github.com/cnshsliu/novamlx/pull/28
- User docs: [docs/features/image-generation.md](../features/image-generation.md)
- Catalog: [docs/features/supported-models.md](../features/supported-models.md)
- SFT: https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT
- Base: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- Upstream: https://github.com/OpenMOSS/MOSS-Image
- Swift port: https://github.com/xocialize/moss-image-swift (MossImage core, not an MLXEngine wrapper)
- Catalog data: `catalog/models.json`
- Product log: the `[Unreleased]` section of the root `CHANGELOG.md`
