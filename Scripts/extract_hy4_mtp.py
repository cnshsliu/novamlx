#!/usr/bin/env python3
"""Extract native Hy4 MTP tensors from official BF16 shards and merge into an MLX dir.

Official mlx-community/Hy4-preview-4bit strips `model.mtp_layers.*`. NovaMLX keeps
those weights and runs them as in-graph MTP. This script copies MTP tensors from
the Tencent BF16 shards into `mtp.safetensors` next to the 4-bit backbone.

Usage:
  python3 Scripts/extract_hy4_mtp.py \\
      --src /Volumes/WD/nova-models/tencent/Hy4-preview-mtp \\
      --dst /Volumes/WD/nova-models/mlx-community/Hy4-preview-4bit
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    args = p.parse_args()

    index_path = args.src / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"missing {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    mtp_keys = [k for k in weight_map if "mtp_layers" in k]
    if not mtp_keys:
        raise SystemExit("no mtp_layers keys in official index")

    files = sorted({weight_map[k] for k in mtp_keys})
    tensors = {}
    for fname in files:
        path = args.src / fname
        if not path.exists():
            raise SystemExit(f"missing shard {path}")
        print(f"reading {path.name}")
        with safe_open(str(path), framework="pt") as fh:
            for k in fh.keys():
                if "mtp_layers" in k:
                    tensors[k] = fh.get_tensor(k)

    out = args.dst / "mtp.safetensors"
    args.dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    print(f"wrote {len(tensors)} MTP tensors -> {out} ({out.stat().st_size / 1e9:.2f} GB)")

    dst_index = args.dst / "model.safetensors.index.json"
    if dst_index.exists():
        idx = json.loads(dst_index.read_text())
        wm = idx.setdefault("weight_map", {})
        for k in tensors:
            wm[k] = "mtp.safetensors"
        dst_index.write_text(json.dumps(idx, indent=2) + "\n")
        print(f"updated {dst_index} with {len(tensors)} mtp_layers keys")


if __name__ == "__main__":
    main()
