# MiniCPM-OCR 1.0 → pure Swift MLX

Recognition model only: one chat turn with an image in, text out, inside `NovaMLXEngine`. Layout, the Jev scorer, and DSpark stay out of this port. No llama.cpp and no GGUF.

This document is the design. The executable task list is [minicpm-ocr-IMPLEMENTATION.md](minicpm-ocr-IMPLEMENTATION.md). This pull request does not add inference code.

## Product intent

Stop the competitive OCR comparison and port MiniCPM-OCR 1.0 as a native Swift MLX VLM. The first shippable path is the same shape Unlimited OCR already serves: `POST /v1/chat/completions` with a user message that contains text plus an `image_url`. PP-DocLayoutV3, the MiniCPM-OCR Jev package, and the DSpark draft head are later products.

`companionReady` stays the admin field name. It means an on-disk speculative draft, and it is not chat-ready. MiniCPM-OCR 1.0 is a chat model. Do not rename the field, and do not mark this checkpoint `companionReady`.

## Golden reference

Python MLX is the numerical source of truth:

| Item | Value |
|------|--------|
| Repository | [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) |
| Package | `mlx_vlm/models/minicpm_ocr` |
| Pin | `230fc308efb577d87446306bfb8a3e9f1a129558` |
| Parent the OCR class subclasses | `mlx_vlm/models/minicpmv4_6` |
| Language implementation | `mlx_vlm/models/qwen3_5/language.py` |
| Processor | `processing_minicpm_ocr.py`, which subclasses the V 4.7 processor and replaces the image processor |

Read those files at the pin. Later commits of mlx-vlm are not the oracle until someone re-pins them in this doc.

Weights Swift loads are a local MLX conversion of the official checkpoint. The path is in [Weight strategy](#weight-strategy). Third-party preconverted packs are optional and are not required.

The official card also lists GGUF, AWQ, GPTQ, and a DSpark draft. Those formats are not inputs to this port.

## Verified architecture

Checked against `openbmb/MiniCPM-OCR-1.0` `config.json` and `preprocessor_config.json`, and the pinned mlx-vlm sources. Where the Hugging Face card and the pin disagree, the pin wins. A `nativ-community` bf16 config was read only as an optional cross-check; the port does not depend on it.

### What the checkpoint is

- `model_type`: `minicpm_ocr`
- `architectures`: `MiniCPMOcrForConditionalGeneration`
- Vision `model_type`: `minicpm_ocr_vision`
- Text `model_type`: `qwen3_5_text`
- About 1.3B parameters (card: ~778M language + ~522M vision), bf16 on disk ~2.6 GB
- `tie_word_embeddings`: true
- `image_token_id`: 248056, `video_token_id`: 248057
- EOS on the official card: 248046. BOS: 248045. The pin’s `ModelConfig.__post_init__` keeps **248046** when a config lists both 248044 and 248046, because treating 248044 (`<|endoftext|>`) as EOS can stop on an empty generation. Match that after reading the official `generation_config.json`.
- BOS on the card: 248045
- `mtp_num_hidden_layers`: 0. Drop any `mtp.*` keys. There is no draft head in this checkpoint.

### Vision tower

SigLIP-style ViT, not the Qwen3.5-VL tower and not Unlimited OCR’s SAM+CLIP stack.

| Field | Value |
|-------|--------|
| Layers | 27 |
| Hidden | 1152 |
| Intermediate | 4304 |
| Heads | 16 (head dim 72) |
| Patch | 14 |
| Channels | 3 |
| Activation | `gelu_pytorch_tanh` |
| LayerNorm eps | 1e-6 |
| `image_size` in `vision_config` | 980 |

`image_size` 980 is the learned position table (980 / 14 = 70, a 70×70 table). It is not the resize target. Slices are sized from `scale_resolution`.

Embeddings: conv patch embed (bias on) plus the 70×70 position table, indexed by fractional buckets. The pin builds bucket boundaries in a way that keeps an exact boundary (the comment calls out 0.25 on this 70×70 table) in the right bin. Copy that construction. A float32 `arange` step will mis-bin.

The tower’s own `VisionModel.__call__` runs embeddings → all 27 layers → post LayerNorm. MiniCPM does **not** use that call for OCR. `get_vision_embedding` runs the layers itself so it can insert the window merger.

### Two mergers, and when each runs

This is the correction to “layer-6 window merger is always on.”

`insert_layer_id` is 6. `window_kernel_size` is `[2, 2]`. The window module is `VitMerger`:

- Group the grid into non-overlapping 2×2 windows (4 tokens, each 1152).
- LayerNorm + bias cross-attention used as self-attention (16 heads, dim 1152).
- Residual add.
- Mean over the 4 tokens as a second residual.
- Pre-norm, linear 4608 → `window_intermediate_size` (4304 × 4 = 17216), GELU, linear back to 1152, add the mean residual.
- Grid H and W are halved.

The V 4.6 parent constructs that GELU as `approx="precise"`. The OCR subclass **replaces** `vit_merger.act` with `GELU(approx="tanh")`. The post-tower `Merger` is left on precise GELU. Match both.

The post-tower `Merger` (`merger_times` 1, `merge_kernel_size` `[2, 2]`) packs another 2×2, LayerNorm, linear, precise GELU, linear, and projects 1152×4 → language hidden 1024.

Runtime rule in `minicpmv4_6.py` `get_vision_embedding`:

- `downsample_mode == "4x"` (the OCR default): run all 27 encoder layers, **skip** `vit_merger`, then run `merger`. Token count is divided by 4.
- `downsample_mode == "16x"`: after encoder layer index 6 (`layer_index == insert_layer_id`), run `vit_merger`, then finish the remaining layers, then run `merger`. Token count is divided by 16.

The official card says this release “ships 4× only” and also describes early compression at layer 6. The pin implements both modes and skips the layer-6 module on 4×. P0 and P1 implement **4×**. Keep the `vit_merger` parameters in the module so 16× can be switched on later without a second weight layout. The checkpoint still contains those tensors; 4× simply does not call the module.

Default preprocessing (`preprocessor_config.json`):

| Field | Value |
|-------|--------|
| `downsample_mode` | `4x` |
| `scale_resolution` | 448 |
| `max_slice_nums` | 9 |
| `slice_mode` | true |
| `use_image_id` | true |
| `patch_size` | 14 |
| mean / std | 0.5, 0.5, 0.5 |

OCR resize alignment is `patch_size * 2` in 4× (28 pixels) and `patch_size * 4` in 16× (56). The V 4.6 resize helper is not the OCR helper; use `MiniCPMOcrImageProcessor._find_best_resize`.

Placeholder count for one crop is `(grid_h * grid_w) / token_divisor`, with `token_divisor` 4 in 4× mode and 16 in 16× mode. Each crop is wrapped with the image or slice start/end strings from `MiniCPMVImageProcessor` defaults (the official preprocessor does not override them). Read the strings from the pin; they are MiniCPM control tokens and are easy to corrupt if copied out of rendered HTML. `use_image_id` prefixes `<image_id>{index}</image_id>`. The OCR override of `_build_placeholder_ids_for_image` gives **each slice its own token count** and inserts a newline token between rows. The V 4.6 parent reuses one slice length for every cell. Follow the OCR override.

`image_bound` is a row per crop: `[start, end)` covers the fill tokens inside that crop’s image or slice wrapper. The processor requires `image_bound` row count to equal the number of crops. The model requires the sum of `(end - start)` to equal the vision feature rows, and writes those rows into `inputs_embeds` by index. That write is the MiniCPM path (the pin’s `_apply_vision_delta`). Unlimited OCR concatenates local crops, then the global view, then a separator; this model does not.

`pack_image_by_patch` defaults true. The embedding path accepts both CHW and the packed `(C, patch, n * patch)` layout and converts to HWC before the conv. P1 must produce the same layout the pin’s processor emits.

Adaptive upsize (`adaptive_small_image_upsize` and the related `adaptive_*` knobs, including the OpenCV connected-component tier) defaults **off** in `ADAPTIVE_DEFAULTS`. P0/P1 leave it off. A later change can turn it on once a fixture needs it.

### Language half

Qwen3.5 hybrid decoder, 24 layers, hidden 1024, intermediate 3584, SiLU SwiGLU, RMSNorm eps 1e-6. `layer_types` is 18 × `linear_attention` and 6 × `full_attention`, in groups of three linear layers then one full layer. That matches `full_attention_interval` 4 and the pin’s rule `(layer_idx + 1) % 4 != 0` → linear. Trust `layer_types` when it is present.

Full attention (layers 3, 7, 11, 15, 19, 23):

| Field | Value |
|-------|--------|
| Heads | 8 |
| KV heads | 2 |
| `head_dim` | 256 (this is **not** `hidden_size / num_attention_heads`) |
| `q_proj` out | `8 * 256 * 2 = 4096` |
| `k_proj` / `v_proj` out | `2 * 256 = 512` |
| `o_proj` in | 2048 |
| Bias | false |
| Q/K RMSNorm | per-head, dim 256 |

`q_proj` is twice the query width because `attn_output_gate` is true. The pin’s `Qwen3_5Attention` always allocates that extra half and splits it into a gate. Copy the class, including the flag’s effect, from `qwen3_5/language.py`.

RoPE: `partial_rotary_factor` 0.25, so 64 rotary dimensions out of 256. `rope_theta` is 10_000_000. The HF `rope_parameters` object has no `mrope_section`. The pin’s `TextConfig.__post_init__` fills a default section table (`[11, 11, 10]`). Do not invent a different table. OCR `get_input_embeddings` calls `language_model.get_rope_index(input_ids, attention_mask=mask)` **without** image or video grids, so the no-grid branch runs (cumulative positions from the mask, plus `rope_deltas`). It then stores those tensors on the language model and on the returned embeddings. The V 4.6 parent’s dummy broadcast of a 1-D range to shape `(3, batch, seq)` is the path OCR replaces. P0 logs `position_ids` rank and the first/last values from the pin on the smoke example before the Swift RoPE is written.

Linear attention is `Qwen3_5GatedDeltaNet`, one module per linear layer:

- `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`
- One grouped causal conv1d over Q, K, and V concatenated. Kernel 4, `linear_conv_kernel_dim`.
- 16 key heads and 16 value heads, key/value head dim 128
- `A_log` shape `[16]`, `dt_bias` shape `[16]`
- Gated RMSNorm over the value head (`Qwen3_5RMSNormGated`), then `out_proj` back to 1024
- Cache: `ArraysCache(size=2)` — slot 0 is the conv window, slot 1 is the recurrent state. Full-attention layers use a KV cache. `make_cache` in the pin is the pattern.

`mamba_ssm_dtype` is `float32`. Recurrent math and the gated norm’s SiLU product run in float32 and cast back to the activation dtype. The decode depthwise conv casts its weight to float32 inside the kernel even when the stored weight is bf16.

This is a different stack from Unlimited OCR (DASH / R-SWA, MoE, SAM+CLIP) and a different stack from `Glm5NextModel` (gated-delta with separate `f_a_proj` / `b_proj` / `g_*` and DeepSeek sparse attention) and `BailingHybridModel` (MLA + GLA + MoE). Those two Swift files are the in-repo pattern for a hybrid `newCache`, a per-layer branch, and a `sanitize` that keeps recurrent tensors in the dtype the kernel expects. The equations to transcribe are `Qwen3_5GatedDeltaNet` and `Qwen3_5Attention` at the pin. `MLXVLM.Qwen35` is the packaged Qwen3.5 **VL** model (its own vision tower and `grid_thw` merge). Loading MiniCPM-OCR through that class would attach the wrong vision path.

`max_position_embeddings` is 262144. `MLXEngine` already copies that into `contextLength` from `text_config`. Leave the auto-detect in place for P0. A 256K window on six full-attention layers (2 KV heads, head dim 256) is a serving decision, recorded when the feature page is written, after a memory measurement. Do not change Flint, Qwen, or Unlimited OCR context defaults in the port.

### Registration pattern to copy

`UnlimitedOCRRegistration.register()` is the VLM hook, called from `CustomModelRegistration.ensureRegistered()`:

- `VLMTypeRegistry.shared.registerModelType` for the `model_type` string
- `VLMProcessorTypeRegistry` for the processor class name
- `VLMProcessorLoadingRegistry` so a checkpoint without `preprocessor_config` processor metadata still resolves

MiniCPM-OCR registers `minicpm_ocr` and processor class `MiniCPMOcrProcessor` the same way. It does not register on `LLMTypeRegistry`. `MLXEngine` sends `modelType == .vlm` through `VLMModelFactory`.

`ModelDiscovery` must learn the architecture and the type or the directory will load as a text model and drop the tower:

- `vlmArchitectures`: `MiniCPMOcrForConditionalGeneration`
- `vlmModelTypes`: `minicpm_ocr`

Family: the checkpoint’s own `chat_template` is the prompt authority. Do not send this model through the Qwen3.5 thinking-template processor. `enable_thinking` stays false on smoke and on the regression prompt. A `ModelFamily` entry is added only when a family processor would otherwise rewrite the template; `.qwen` is the wrong default if that processor injects Qwen control tokens.

Hybrid scheduling is already implemented. `MLXEngine` reads `text_config.layer_types`, and when it contains `linear_attention` it sets `hasLinearAttention` and selects `ContinuousBatcher`. This config has that array (18 linear, 6 full). Confirm the flag on first load. `FusedBatchScheduler` only supports `KVCacheSimple` and is the wrong scheduler for this model.

Element prompts from the model card, for later tests: `Text Recognition:`, `Table Recognition:`, `Formula Recognition:`. One of those, at temperature 0, is the smoke prompt. The official `generation_config.json` may set a non-zero temperature; parity runs stay at temperature 0.

## Porting method

1. **Freeze the oracle.** Check out mlx-vlm at `230fc308efb577d87446306bfb8a3e9f1a129558`. Download the official checkpoint and convert it with `mlx_vlm.convert` at that pin (see [Weight strategy](#weight-strategy)). Run `mlx_vlm.generate` on the converted directory at temperature 0, `downsample_mode=4x`, `max_slice_nums=9`, on one fixed image and one fixed prompt. Save the decoded text, the input ids, `tgt_sizes`, `image_bound`, and the first generated token id.
2. **Swift modules, bottom up.** Config decode → sanitize key map (no forward) → vision embeddings on one crop → one linear layer and one full-attention layer against a dumped hidden → full 4× forward → processor parity → `VLMTypeRegistry` → chat route.
3. **Registry.** `MiniCPMOCRRegistration.register()` from `CustomModelRegistration.ensureRegistered()`, next to `UnlimitedOCRRegistration.register()`.
4. **Weight load.** Swift loads the MLX safetensors produced from `openbmb/MiniCPM-OCR-1.0`. `sanitize` matches the pin, so loading that directory does not shift norms a second time. Third-party packs are not part of this step.
5. **Chat.** The Unlimited OCR chat path (image content part, VLM container, streaming token counts) is the product shell. The model, processor, and prompts are MiniCPM-OCR’s.
6. **Gate.** After any engine change, run `Scripts/regression-perf-gate.sh`. Flint, Qwen3-8B, and Unlimited OCR must stay green. MiniCPM-OCR joins the suite only in P3, as `kind: ocr`, with decode tok/s left ungated until a quiet measurement says otherwise.

Each step compares tensors to the pin before the next step starts. A green chat string with a wrong `image_bound` length is a failed step.

## File and module map

All of these land in `Sources/NovaMLXEngine/Models/`. Tests land in `Tests/NovaMLXEngineTests/MiniCPMOCRTests.swift`. No new Swift package target. `Package.swift` already compiles every file in that directory.

| Swift file | Responsibility | Python at the pin |
|------------|----------------|-------------------|
| `MiniCPMOCRConfiguration.swift` | Decode `config.json` / preprocessor fields. Nested `text_config` and `vision_config`. Defaults from the OCR config, not the V 4.6 16× default. | `minicpm_ocr/config.py`, `minicpmv4_6/config.py` |
| `MiniCPMOCRVision.swift` | SigLIP embeddings (conv + 70×70 buckets), 27 pre-norm encoder layers, tanh-GELU MLP, post LayerNorm. | `minicpmv4_6/vision.py` |
| `MiniCPMOCRMerger.swift` | `VitMerger` (tanh GELU) and post-tower `Merger` (precise GELU). 4× calls only the post-tower merger. | `VitMerger`, `Merger`, `MergerBlock` in `minicpmv4_6.py`; OCR overrides `vit_merger.act` |
| `MiniCPMOCRLanguage.swift` | Hybrid stack: gated-delta linear layers, gated full attention, partial RoPE, mixed cache. | `qwen3_5/language.py` (`Qwen3_5GatedDeltaNet`, `Qwen3_5Attention`, `LanguageModel.make_cache`) |
| `MiniCPMOCRModel.swift` | Owns language, tower, both mergers. `get_vision_embedding`, `image_bound` write, `sanitize`, `newCache`, `callAsFunction` / `prepare`. | `minicpm_ocr/minicpm_ocr.py` + `minicpmv4_6.py` `Model` |
| `MiniCPMOCRProcessor.swift` | Slice (max 9), 4× resize, mean/std 0.5, placeholders, `image_bound`, `UserInputProcessor`. Adaptive upsize left off. | `processing_minicpm_ocr.py` and the V 4.6 processor it subclasses |
| Registration | `MiniCPMOCRRegistration.register()` in the model file, same shape as `UnlimitedOCRRegistration`. One call added in `CustomModelRegistration.swift`. | `install_auto_processor_patch("minicpm_ocr", ...)` |

`ModelDiscovery.swift` gains the architecture and `model_type` entries in the same change as registration. That is discovery, not a new model module.

Do not add files under `vendors/`. Do not import a Python runtime on the request path.

## Weight strategy

Official weights are the source. Swift never requires a third-party conversion.

### Primary path

1. On the maintainer Mac, download `openbmb/MiniCPM-OCR-1.0` with aria2. Prefer the ModelScope mirror of that repo id (multi-connection). The Hugging Face repo [openbmb/MiniCPM-OCR-1.0](https://huggingface.co/openbmb/MiniCPM-OCR-1.0) is the same checkpoint when ModelScope is inconvenient. Apache-2.0.
2. Convert that directory with mlx-vlm **at the pin**. `mlx_lm.convert` does not know `minicpm_ocr`. P0 does not quantize:

```bash
mlx_vlm.convert \
  --hf-path /path/to/openbmb/MiniCPM-OCR-1.0 \
  --mlx-path /path/to/MiniCPM-OCR-1.0-mlx-bf16
```

3. Swift loads `/path/to/MiniCPM-OCR-1.0-mlx-bf16` (MLX safetensors plus the copied config, tokenizer, and preprocessor). The pin’s `Model.sanitize` is what `mlx_vlm.convert` applies. Swift’s `sanitize` reproduces it, so a second pass on the converted directory is a no-op: keys stay put, and a language norm scale does not move by +1 again.
4. Python smoke (`mlx_vlm.generate`) and the Swift loader use this same directory.

Do not hand-edit tensors. If a key is missing or a norm is off by 1, fix the Swift predicate until it matches the pin’s `sanitize` on this conversion.

### Optional convenience

`nativ-community/MiniCPM-OCR-1.0-bf16` and the bit-width packs under that org (`-4bit`, `-5bit`, `-6bit`, `-8bit`, and any others they publish) are someone else’s conversion. They are not required, and the port does not block on them. A developer may diff one against the official conversion (key set, one language-norm value, `A_log` dtype) when debugging a loader. Ship and regression paths point at the official conversion.

A `conversion.json` on a nativ bf16 pack notes that shifted language-norm scales and recurrent `A_log` stay in FP32. That sentence matches the pin’s `sanitize` / `cast_predicate`. It is a cross-check of the Python code, not a weight we have to download.

Later bit-width artifacts, if any, are produced from the official tree with `mlx_vlm.convert --quantize` at the pin. The pin’s `quant_predicate` leaves paths containing `vit_merger`, `merger`, `linear_attn`, `embed_tokens`, or `lm_head` dense. Swift does not quantize this model inside `prepare` in P0–P2 (Unlimited OCR’s load-time language quantizer stays on that model only).

### Sanitize rules to transcribe

OCR `sanitize` casts any tensor whose key ends in a norm suffix to float32, then calls the V 4.6 `sanitize`. Implement both, in that order.

Norm suffixes (`qwen3_5.py` `NORM_WEIGHT_SUFFIXES`):

- `.input_layernorm.weight`
- `.post_attention_layernorm.weight`
- `model.norm.weight`
- `.q_norm.weight`
- `.k_norm.weight`

The +1 shift applies only when the **original** key starts with `model.language_model.` and `should_offset_norm_weight` is true. `should_shift_norm_weights` is true when `mtp.` keys are present or a `conv1d.weight` still has a trailing axis other than 1 (PyTorch layout). After a previous sanitize, keys already start with `language_model.` and conv weights already end in 1, so the +1 does not run again. That is the idempotency test.

Key remap, after stripping one leading `model.` if present:

| Incoming prefix | Swift / MLX key |
|-----------------|-----------------|
| `language_model.model.` | unchanged |
| `language_model.lm_head.` | unchanged |
| `language_model.` (other) | `language_model.model.` |
| `lm_head.` | `language_model.lm_head.` |
| `vision_tower.vit_merger.` | `vit_merger.` |
| `vision_tower.` | unchanged |
| `vpm.` | `vision_tower.` |
| `vit_merger.` / `merger.` | unchanged |
| `llm.model.` / `llm.` | `language_model.model.` |
| `llm.lm_head.` | `language_model.lm_head.` |
| `visual.` | `vision_tower.` |

Unmapped keys are dropped. `position_ids` keys are dropped. `mtp.` keys are dropped. `tie_word_embeddings` drops `language_model.lm_head.weight`.

Layout fixes, guarded so a second pass is a no-op:

- `conv1d.weight` with rank 3 and a last axis other than 1: transpose `(0, 2, 1)`.
- `embeddings.patch_embedding.weight` still in PyTorch OIHW: transpose to MLX OHWI. Use the pin’s `check_array_shape` condition.

### Dtypes mlx-vlm keeps

| Tensor | Dtype behavior at the pin |
|--------|---------------------------|
| Norm scales in the suffix list | Forced to float32 in OCR `sanitize`. `cast_predicate` refuses to cast them back. The custom `RMSNorm` in `minicpm_ocr/precision.py` runs the norm and casts the **activation** back to the input dtype so a float32 scale does not promote the residual stream. |
| `A_log` | `cast_predicate` returns false. Stays at the converted checkpoint dtype. The pin keeps this recurrent parameter in FP32. |
| `dt_bias` | Created as float32 and consumed by `gated_delta_update` next to `A_log`. The cast predicate does **not** special-case it, so a loader that casts every other parameter may store it as bf16. Match the pin’s predicate. If the first-token logit moves, keep `dt_bias` in float32 and record the diff. |
| Conv weight | Stored at checkpoint dtype. The single-token decode kernel multiplies in float32. |
| Gated RMSNorm (`linear_attn.norm`) | Weight is not in the suffix list. The forward promotes the SiLU product to float32 and casts back. |
| Vision, mergers, ordinary linears | bf16, matching the official checkpoint dtype and an unquantized `mlx_vlm.convert`. |

`preserve_norm_precision` replaces every language `RMSNorm` with the casting subclass. Swift `RMSNorm` must do the same cast on the way out.

## Smoke acceptance

One image, one prompt, temperature 0, `downsample_mode=4x`, `max_slice_nums=9`, greedy decode.

1. Python, at the pin: `mlx_vlm.generate` on the official conversion from [Weight strategy](#weight-strategy). Record text, token ids, `tgt_sizes`, `image_bound`, crop count, and vision-token count.
2. Swift on the same directory: sanitized key set equals the pin’s key set; a language norm stays float32; `A_log` stays float32; patch-embed rank matches MLX; `image_bound` row count equals the crop count and the summed span lengths equal the vision rows; decoded text matches the Python text.
3. If the text diverges, compare the first generated token id, then the prefill logits at that position. Do not tune temperature or the prompt to hide a mismatch.
4. Chat smoke comes after that, on the Unlimited OCR request shape (`image_url` data URL, `temperature` 0). The prompt is the OCR prompt (`Text Recognition:` or whichever string the Python smoke used), not `document parsing.`
5. Regression `kind: ocr` is P3. It is not the definition of the first green run.

Suggested prompts once parity holds: the three card prompts on a text crop, a table crop, and a formula crop. P0 needs only one of them.

## Regression gate

`Scripts/regression-perf-gate.sh` is the check Peer runs after engine work. The default suite is `Scripts/regression-suite.yaml` `core`:

| Id | Kind | Seed floor |
|----|------|------------|
| `RecursiveSI/flint-int4` | llm | `min_seed_decode_tps: 100` |
| `mlx-community/Qwen3-8B-4bit` | llm | `min_seed_decode_tps: 50` |
| `baidu/Unlimited-OCR-2` | ocr | decode tok/s not gated (`gate_decode_tps: false`) |

Those three stay in `core` with their current floors, prompts, and paths. A MiniCPM-OCR change that moves their tok/s or their correctness strings fails the port. `min_seed_decode_tps` is per model in the YAML. Do not add a global floor and do not lower these two.

P3 adds a new core or optional entry whose `path` is the official `mlx_vlm.convert` directory, `kind: ocr`, `gate_decode_tps: false`, `expect_nonempty: true`, and a digit or a known substring from the fixture. OCR decode walls near 0 ms are why Unlimited OCR does not gate tok/s. Leave MiniCPM-OCR the same until a quiet Mac shows a real decode wall, then set `min_seed_decode_tps` on that row only. The gate waits on admin `loaded: true` and `/health.loadedModels`. It must not wait on `companionReady`.

## Out of scope

- PP-DocLayoutV3 and the `minicpmvocr` SDK (page detect, reading order, block crops, Markdown assembly).
- MiniCPM-OCR Jev (`mlx-vlm` documents it beside 1.0). Do not vendor that package.
- DSpark (`openbmb/MiniCPM-OCR-1.0-DSpark` and any third-party draft pack). Do not register a companion, do not set `companionReady` for this model, and do not rename `companionReady`.
- Treating `nativ-community/*` (or any other preconverted MLX repo) as the required weight. Optional diff only.
- llama.cpp, GGUF, AWQ, and GPTQ runtimes.
- 16× as a product default. The module can exist; the tested path is 4×.
- Adaptive small-image upsize / OpenCV tiering (defaults off).
- Video (`video_token_id` is in the config; no video processor work in this port).
- Vendor OCR routes (Baidu / Aliyun / Tencent shapes stay on Unlimited OCR).
- Speculative decoding on this VLM. `mtp_num_hidden_layers` is 0.
- Competitive scorecards under `docs/competitive`. Leave that tree alone.
- Unlimited OCR numerics, DASH, sparse attention, and its prompts.
- A user-facing `docs/features/` page. Write that when P2 chat parity is green, not in the planning PR.

## Risks

- **4× versus the layer-6 module.** Implementing the card’s “always merge at layer 6” sentence will not match the pin and will not match a 4× processor’s placeholder count.
- **Position buckets.** The 70×70 table is sensitive to how boundaries are rounded. Dump buckets for the smoke image before comparing vision hidden states.
- **RoPE rank.** OCR calls `get_rope_index` without grids. Confirm the tensor rank the attention module actually consumes.
- **Norm +1 applied twice.** A bf16 load that “almost” matches, with norms off by 1, is this bug.
- **`q_proj` width.** Forgetting the gate half (4096 vs 2048) fails the key shape check immediately. Treat that as a sanitize/config test, not a runtime surprise.
- **Scheduler.** A hybrid cache in `FusedBatchScheduler` will not track the conv window or the recurrent state.
- **Context 262144.** Auto-detect will advertise it. Measure before documenting a product default. Do not “fix” it by editing shared engine defaults that Flint and Qwen also read.
- **Chat template.** A Qwen family processor that injects thinking tags will desynchronize `image_bound`.
