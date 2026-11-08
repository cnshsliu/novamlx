#!/usr/bin/env python3
"""Convert oMLX-format MLX safetensors builds to NovaMLX tensor naming.

oMLX builds (oQ4e/oQ6e/oQ8e family, e.g. hakunokina/Huihui-...-oQ8e-mtp) use
`language_model.model.layers.*` / `language_model.lm_head.*` and a transposed
conv1d layout `[out, k, 1]`. NovaMLX loads the mlx-community layout:
`model.language_model.layers.*` / `lm_head.*` with conv1d `[out, 1, k]`.

This script rewrites shards key-by-key (quantized tensors pass through
untouched apart from renaming), transposes conv1d, and extracts any `mtp.*`
tensors into a separate sibling file.

Usage:
  python3 scripts/omlx_to_novamlx.py --src <raw_dir> --dst <out_dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx.core as mx

# Pure IO/rename work — stay off the GPU so the Metal watchdog can't kill
# long saves (kIOGPUCommandBufferCallbackErrorTimeout seen with 5GB shards).
mx.set_default_device(mx.cpu)

# NOTE: do NOT transpose conv1d. oMLX builds already store the runtime layout
# [out, k, 1]. Qwen35.sanitize() treats a trailing dim != 1 as an unsanitized
# HF checkpoint and shifts ALL layernorm weights by +1 (w+1), which corrupts
# oMLX absolute-form norms into garbage. Keeping [out, k, 1] makes sanitize a
# no-op, which is exactly what an already-cleaned checkpoint needs.

PREFIX_MAP = [
    ("language_model.lm_head.", "lm_head."),
    ("language_model.model.", "model.language_model."),
    ("language_model.mtp.", "mtp."),
    ("model.language_model.", "model.language_model."),  # already correct
    ("lm_head.", "lm_head."),  # already correct
]


def remap_key(key: str) -> str:
    for src, dst in PREFIX_MAP:
        if key.startswith(src):
            return dst + key[len(src):]
    return key


def convert_tensor(key: str, t: mx.array) -> mx.array:
    return t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    args = p.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    shards = sorted(args.src.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no model shards in {args.src}")

    mtp: list = []
    renamed = passthrough = 0
    for shard in shards:
        weights = dict(mx.load(str(shard)))
        out: dict[str, mx.array] = {}
        for key, t in weights.items():
            new_key = remap_key(key)
            t = convert_tensor(new_key, t)
            if new_key.startswith("mtp."):
                mtp.append((new_key, t))
                continue
            out[new_key] = t
            if new_key != key:
                renamed += 1
            else:
                passthrough += 1
        mx.save_safetensors(str(args.dst / shard.name), out, metadata={"format": "mlx"})
        print(f"wrote {shard.name} ({len(out)} tensors)", flush=True)
        del weights, out

    if mtp:
        mx.save_safetensors(str(args.dst / "mtp.safetensors"), dict(mtp), metadata={"format": "mlx"})
        print(f"extracted {len(mtp)} MTP tensors -> mtp.safetensors")

    # copy sidecar files as-is
    for f in args.src.iterdir():
        if f.is_file() and f.suffix not in (".safetensors", ".aria2") and f.name not in ("SHA256SUMS",):
            shutil.copy2(f, args.dst / f.name)

    print(f"done: {renamed} renamed, {passthrough} passthrough")


if __name__ == "__main__":
    main()
