# MiniCPM-OCR 1.0 Swift port — Peer checklist

Design and the corrections to the architecture notes: [minicpm-ocr-swift-mlx.md](minicpm-ocr-swift-mlx.md).

Do this on a Mac with Metal. This checklist is the implementation. The planning PR does not add model code.

Oracle pin: `Blaizzy/mlx-vlm` @ `230fc308efb577d87446306bfb8a3e9f1a129558`, package `mlx_vlm/models/minicpm_ocr`.

Weights for every phase: official `openbmb/MiniCPM-OCR-1.0` (ModelScope mirror preferred, Hugging Face is the same tree), downloaded with aria2, then `mlx_vlm.convert` at the pin into MLX safetensors. `nativ-community/*` packs are an optional key diff, not the artifact Swift loads.

Leave `companionReady` spelled that way. Leave `docs/competitive/` untouched. Leave Flint, Qwen, and Unlimited OCR suite rows unchanged.

After every engine commit: `Scripts/regression-perf-gate.sh`. Core today is `RecursiveSI/flint-int4` (`min_seed_decode_tps: 100`), `mlx-community/Qwen3-8B-4bit` (`min_seed_decode_tps: 50`), and `baidu/Unlimited-OCR-2` (`kind: ocr`, `gate_decode_tps: false`). Those three must not regress.

---

## P0 — Load and forward smoke

Goal: bf16 weights load, one 4× image forward runs, greedy text matches the pin on one fixture.

- [ ] Download official weights with aria2 from the ModelScope mirror of `openbmb/MiniCPM-OCR-1.0` (Hugging Face `openbmb/MiniCPM-OCR-1.0` if that mirror is down). Convert at the pin, without quantizing:

```bash
mlx_vlm.convert \
  --hf-path /path/to/openbmb/MiniCPM-OCR-1.0 \
  --mlx-path /path/to/MiniCPM-OCR-1.0-mlx-bf16
```

- [ ] Check out the pin. Run `mlx_vlm.generate` on **that converted directory** (temperature 0, `downsample_mode=4x`, `max_slice_nums=9`, one image, one prompt such as `Text Recognition:`). Save decoded text, token ids, `tgt_sizes`, `image_bound`, crop count, and vision-token count next to the fixture.
- [ ] On that same converted directory, run the pin’s `Model.sanitize` twice. The second pass must be a no-op (keys and a sampled language-norm value). Do not hand-patch tensors. A `nativ-community` directory is not a substitute for this step.
- [ ] Add `MiniCPMOCRConfiguration` and a test that decodes the official (or converted) `config.json`: `model_type == minicpm_ocr`, 27 vision layers, `insert_layer_id == 6`, 24 language layers, `layer_types` is 18 linear + 6 full, `downsample_mode == 4x`, `scale_resolution` default 448, `max_slice_nums` default 9, `head_dim == 256`, `tie_word_embeddings == true`.
- [ ] Add `sanitize` as specified in the design (float32 norm suffixes, +1 only on original `model.language_model.*` keys, conv1d transpose, patch-embed OHWI, drop `mtp.` / `position_ids` / untied `lm_head`, remap table). Test: Swift key set equals the pin’s key set; one norm and `A_log` stay float32; `q_proj` for a full-attention layer has out-width 4096.
- [ ] Add vision, both mergers, and the language stack. 4× forward skips `vit_merger`. `vit_merger` GELU is tanh; post-tower merger GELU is precise. Cache is `ArraysCache(size=2)` on linear layers and a KV cache on full-attention layers.
- [ ] Log the pin’s `position_ids` rank on the fixture before writing RoPE. Follow `get_rope_index` called **without** image grids.
- [ ] Module tree under `Sources/NovaMLXEngine/Models/MiniCPMOCR{Configuration,Vision,Merger,Language,Model,Processor}.swift`. Register later (P2). P0 can call the module from a test.
- [ ] Acceptance: same fixture, greedy ids match the pin. If text differs, first-token id and prefill logits are the bug report. Stop.
- [ ] `Scripts/regression-perf-gate.sh` still passes for Flint, Qwen, and Unlimited OCR.

## P1 — Processor parity

Goal: Swift preprocessing matches the pin’s processor outputs. No adaptive upsize (those flags default off).

- [ ] Slice with `scale_resolution=448`, `max_slice_nums=9`, `slice_mode=true`, alignment 28 (4×). Mean/std 0.5.
- [ ] Copy image/slice/unk/image-id strings from `MiniCPMVImageProcessor` at the pin (the HF preprocessor does not override them).
- [ ] Placeholder counts use divisor 4. Each slice has its own count. Newline token between rows. `use_image_id` on.
- [ ] For the fixture and for a second image that actually slices: `image_bound` row count equals the crop count, and the summed `(end - start)` spans equal the vision feature rows.
- [ ] Position-bucket ids for the fixture match the pin (70×70 table).
- [ ] Tests in `Tests/NovaMLXEngineTests/MiniCPMOCRTests.swift` for config, sanitize, slice grid, and `image_bound` length. No DASH and no Unlimited OCR fixtures.
- [ ] Gate still green.

## P2 — Chat API parity

Goal: the Unlimited OCR chat shape serves this model. The weights and the prompt stay MiniCPM-OCR.

- [ ] `MiniCPMOCRRegistration.register()` from `CustomModelRegistration.ensureRegistered()`, beside `UnlimitedOCRRegistration.register()`.
- [ ] `VLMTypeRegistry` model type `minicpm_ocr`. Processor type `MiniCPMOcrProcessor`. A loading resolver so a directory without processor metadata still binds that processor.
- [ ] `ModelDiscovery`: architecture `MiniCPMOcrForConditionalGeneration`, model type `minicpm_ocr`, so `MLXEngine` uses `VLMModelFactory`.
- [ ] Confirm load sets `hasLinearAttention` from `text_config.layer_types` and runs on `ContinuousBatcher`.
- [ ] Chat template is the checkpoint jinja. Do not route the request through the Qwen3.5 thinking processor. Smoke with `enable_thinking` false if the template reads that flag.
- [ ] EOS behavior matches the pin: when both 248044 and 248046 are listed, stop on 248046.
- [ ] `POST /v1/chat/completions` with an `image_url` data URL, temperature 0, the same prompt as P0. Streamed text matches the Python smoke. `loaded: true` after admin load. This id is not `companionReady`.
- [ ] `prepare` does not quantize the language stack.
- [ ] Unlimited OCR chat (`document parsing.`, `baidu/Unlimited-OCR-2`) still returns non-empty markup on the regression fixture.
- [ ] Gate still green. Do not edit Flint/Qwen `min_seed_decode_tps`.

## P3 — Regression entry

Goal: the new model is in the suite without changing the existing floors.

- [ ] Add a suite row for the converted official id, `kind: ocr`, `gate_decode_tps: false`, `expect_nonempty: true`, plus `expect_has_digit` or a substring taken from the P0 fixture. Point `path` at the maintainer Mac `mlx_vlm.convert` directory (the official tree, not a third-party pack).
- [ ] Do not set `min_seed_decode_tps` on that row until a quiet run shows a real decode wall (Unlimited OCR leaves tok/s ungated for the same reason). If a floor is added later, it is only on this row.
- [ ] Run `Scripts/regression-perf-gate.sh` and, if the new weights are present, a pass that includes the new id (`--suite all` or a one-model invocation — follow the script’s flags). Flint, Qwen, and Unlimited OCR results stay inside tolerance.
- [ ] Write `docs/features/minicpm-ocr.md` and a catalog row only after P2 is green. State 4× as the supported compression. Mention 16× as unimplemented product surface.
- [ ] 16×, adaptive upsize, video, Jev, PP-DocLayoutV3, and DSpark are separate checklists. They are not leftovers of P3.

## Explicitly not in any phase

- Layout SDK, Jev, DSpark, GGUF/llama.cpp, AWQ, GPTQ.
- Requiring `nativ-community/*` or any other preconverted MLX repo. Optional key diff only.
- Copying Unlimited OCR Dash, sparse attention, or MoE quant.
- Renaming `companionReady` to `agentInfer` or anything else.
- Editing `docs/competitive/`.
- Lowering Flint or Qwen seed floors to absorb a shared-engine regression.
