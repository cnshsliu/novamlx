# Competitive OCR capture

Side-by-side JSONL for Unlimited OCR on local Nova and MiniCPM-OCR on an OpenAI-compatible endpoint. The scorecard is [docs/competitive/minicpm-ocr-vs-unlimited-ocr.md](../../docs/competitive/minicpm-ocr-vs-unlimited-ocr.md).

This directory does not download weights and does not start NovaMLX. CI can run `--self-check` and `--dry-run` with no GPU.

Unlimited OCR here is one local Mac Agent Infer feature (`baidu/Unlimited-OCR-2`). Repo paths stay `cnshsliu/novamlx`.

## Unlimited side on a Mac

NovaMLX must already be running, with `baidu/Unlimited-OCR-2` loaded. This CLI does not load or unload. Peer’s 2026-10-30 run left the model loaded.

The regression gate is the correctness smoke, not this comparison:

```bash
./Scripts/regression-perf-gate.sh
```

That gate’s OCR item is `baidu/Unlimited-OCR-2` in `Scripts/regression-suite.yaml`: kind `ocr`, prompt `document parsing.`, `expect_nonempty`, `expect_has_digit`, `gate_decode_tps: false`. Weight path on the maintainer Mac: `/Volumes/Samsung768/Models/baidu/Unlimited-OCR-2`. The gate generates its own 391 PNG in memory. It does not read this directory.

Auth and port match `Scripts/regression-perf-gate.py`:

- `NOVA_API_KEY`, else the enabled `primary` key in `~/.nova/nova_config.db`, else `abcd1234`
- Inference `http://127.0.0.1:6590` (`NOVA_BASE_URL` or `NOVA_INFERENCE_PORT`)
- Non-stream chat. User text `document parsing.`, then an `image_url` data URL. No `enable_thinking` field.
- Default `max_tokens` **512**, same as the gate entry. The feature-page curl uses 8192; pass `--max-tokens 8192` when a full page matters.

```bash
python3 Scripts/competitive-ocr/compare_ocr.py --backend unlimited
```

With no image arguments, the three paths below are the list.

## Default fixture list

Peer already captured Unlimited OCR on these three files. Prompt `document parsing.`, non-stream chat completions, aligned with the regression gate. End-to-end time is the whole call, not decode tok/s.

| # | Path | Latency | What Peer saw |
|---|------|---------|---------------|
| 1 | `/tmp/novamlx-ocr-compare/01_gate_fixture_391.png` | 0.781s | PAGE and BLOCK. The tiny “391” was tagged as an image block; the digits were not read |
| 2 | `/Users/lucas/Desktop/Scan.jpeg` | 11.91s | BLOCK, no PAGE. Chinese form structure is usable. Too many spaces |
| 3 | `/Users/lucas/Desktop/Scan 1.jpeg` | 2.986s | BLOCK. Short form is weak and misaligned. 「操作时间」 repeats as filler |

Full text is on the Mac, not in git:

- `/tmp/novamlx-ocr-compare/SUMMARY.md` and `results.json`
- Mirror: `/Users/lucas/dev/novamlx/.scratch/ocr-compare/`
- `01` / `02` / `03` `*.txt` and `.meta.json`

`fixtures/01_gate_fixture_391.png` is the same 108×36 bitmap as the gate’s in-memory PNG, for machines that do not have the `/tmp` copy. It is not the measured path. Pass it explicitly:

```bash
python3 Scripts/competitive-ocr/compare_ocr.py --backend unlimited --dry-run \
  --images Scripts/competitive-ocr/fixtures
```

The gate only checks that OCR text is non-empty and contains some digit. Coordinate digits can satisfy that. Peer’s note on image 1 is that the glyph was not transcribed. This scaffold does not change the gate.

Peer’s summary does not record `max_tokens`. To sit next to 0.781s / 11.91s / 2.986s, replay with `--max-tokens 512` (the CLI default for Unlimited). Run `--max-tokens 8192` as a second pass if you need to see whether the page was cut off.

## MiniCPM side

When a local or hosted OpenAI-compatible server exists:

```bash
export OCR_BASE_URL=http://127.0.0.1:18499/v1   # or a host that ends in /v1
export OCR_API_KEY=token-abc123                 # omit only if the server has no key
export OCR_MODEL=minicpm-ocr

python3 Scripts/competitive-ocr/compare_ocr.py --backend minicpm --task all
```

`OCR_BASE_URL` may be the origin or an origin that already ends in `/v1`. The client posts `{base}/v1/chat/completions` or `{base}/chat/completions` accordingly.

Official vLLM sampling is the default: image first, then the prompt, `temperature` 0.01, `top_p` 0.001, `top_k` 1, `repetition_penalty` 1, `no_repeat_ngram_size` 100, thinking off, `downsample_mode` 4x. `--minicpm-plain` drops the vLLM-only fields. Default `max_tokens` is 2048. Prompts:

- `--task text` → `Text Recognition:` (default)
- `--task table` → `Table Recognition:`
- `--task formula` → `Formula Recognition:`
- `--task all` → all three

The two desktop scans are forms. On the default list, a MiniCPM run with `--task text` prints a reminder to add `--task all`.

That call is region recognition, not `minicpmvocr parse`. The published 93.68 is the SDK pipeline (PP-DocLayoutV3 + region prompts + assembly). Do not treat these JSONL rows as an OmniDocBench score.

`--backend both` writes Unlimited, then MiniCPM, for each image. Set one `--max-tokens` if the two defaults (512 and 2048) would truncate differently.

## JSONL

One object per image, backend, and prompt:

```json
{"path":"/tmp/novamlx-ocr-compare/01_gate_fixture_391.png","backend":"unlimited","latency_ms":781.0,"text":"...","error":null,"model":"baidu/Unlimited-OCR-2","prompt":"document parsing.","max_tokens":512}
```

Required fields are `path`, `backend`, `latency_ms`, `text`, `error`. `model`, `prompt`, and `max_tokens` are there so a later row is still readable. `latency_ms` is null when the file is missing or the run is a dry run. A missing default path becomes `"error": "path missing: ..."` and does not stop the other rows.

```bash
python3 Scripts/competitive-ocr/compare_ocr.py --backend both --dry-run --out -
python3 Scripts/competitive-ocr/compare_ocr.py --self-check
python3 Scripts/competitive-ocr/compare_ocr.py --write-fixture
```

`--self-check` confirms the default path list, the prompt strings, the two request shapes, and that the committed PNG matches the generator. `--dry-run` writes JSONL and does not open a socket. Default output is `Scripts/competitive-ocr/out/compare.jsonl` (gitignored).

## What we still need

(a) Unlimited OCR on this fixture list: **done.** Table above. Regression suite and baselines were not touched. The model stayed loaded.

(b) MiniCPM on the **same three paths**, once `OCR_BASE_URL` exists, or official demo captures written into the same JSONL fields. Prefer `--task all` on the two forms.

(c) Optional later: an OmniDocBench subset on each product path. Not claimed until then.

Do not download the multi-GB checkpoints in a cloud agent. Point the CLI at a server that is already up, or paste demo text.
