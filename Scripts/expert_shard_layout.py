#!/usr/bin/env python3
"""
NovaMLX-TIE ExpertShardLayout converter.

Splits an MLX model into a tiered layout for SSD streaming inference:
  - tier0.safetensors           : shared layers (attn, norm, embed, lm_head, shared experts, router)
  - expert.L{NN}.E{NNN}.safetensors : per-expert weights (one file per expert per layer)
  - tier-manifest.json          : layout index used by WeightTierManager

Handles two MoE weight layouts:
  A) Per-expert tensors (classic MoE):
        model.layers.0.mlp.experts.0.gate_proj.weight
        model.layers.0.mlp.experts.1.gate_proj.weight
        ...
     -> copied verbatim into expert.L00.E00.safetensors etc.

  B) Stacked expert tensors (DeepSeek-V4 SwitchLinear / vLLM batched):
        model.layers.0.switch_mlp.gate_proj.weight   shape [num_experts, out, in]
        model.layers.0.switch_mlp.up_proj.weight     shape [num_experts, out, in]
        model.layers.0.switch_mlp.down_proj.weight   shape [num_experts, out, in]
     -> sliced along axis 0 into per-expert files.

Usage:
  python3 scripts/expert_shard_layout.py --src ~/.nova/models/DeepSeek-V4-Flash-4bit \\
      --dst ~/.nova/models/DeepSeek-V4-Flash-4bit.tiered

The output directory can then be loaded by TieredOffloadPolicy (Phase 2+).
Phase 1 fallback: the output dir still contains a complete tier0.safetensors so
existing eager load paths work unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: pip install safetensors torch", file=sys.stderr)
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: pip install torch", file=sys.stderr)
    sys.exit(1)


# Regex patterns classifying tensors.
# Tier 0 = always resident (shared across all experts / all tokens).
TIER0_PATTERNS = [
    re.compile(r".*\.embed_tokens\..*"),
    re.compile(r".*\.embeddings\..*"),
    re.compile(r".*\.lm_head\..*"),
    re.compile(r"lm_head\..*"),
    re.compile(r".*\.shared_experts?\..*"),     # DeepSeek shared expert
    re.compile(r".*\.gate\..*"),                # DeepseekV4Gate router (computed every token)
    re.compile(r".*\.gate_proj\..*"),           # Conservative: treat gate_proj (non-stacked) as tier 0 unless matched as expert
    re.compile(r".*\.(input_layernorm|post_attention_layernorm|norm|final_norm|model_norm)\..*"),
    re.compile(r".*\.(q_proj|k_proj|v_proj|o_proj|q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|rope)\..*"),
    re.compile(r"model\.norm\..*"),
]

# Per-expert tensor patterns.
# Captures (layer, expert) so we can group/slice.
EXPERT_PATTERNS = [
    # Classic MoE: [prefix.]model.layers.{L}[.mlp].experts.{E}.<proj>.<suffix>
    re.compile(r"^.*model\.layers\.(?P<L>\d+)(?:\.mlp)?\.experts\.(?P<E>\d+)\.(?P<rest>.+)$"),
]

# Stacked expert patterns.
# Captures (layer, proj) — the expert dim is axis 0 of the weight.
# Quantization siblings (weight/biases/scales/qscales) are all matched;
# the suffix is preserved in `suffix` so we can slice each separately.
STACKED_PROJS = r"gate_proj|up_proj|down_proj|gate_up_proj|w1|w2|w3"
# Parent of switch_mlp can be `mlp`, `ffn`, or direct. Allow any single-segment
# parent (or none). Captures SwitchGLU's parent module variations across
# model architectures (Qwen3MoE uses `mlp.switch_mlp`, V4 uses `ffn.switch_mlp`).
STACKED_PATTERNS = [
    # DeepSeek-V4 style: [prefix.]model.layers.{L}[.parent].switch_mlp.{proj}.{suffix}
    re.compile(rf"^.*model\.layers\.(?P<L>\d+)(?:\.\w+)?\.switch_mlp\.(?P<proj>{STACKED_PROJS})\.(?P<suffix>[\w.]+)$"),
    # Generic batched: [prefix.]model.layers.{L}[.mlp].experts.{proj}.{suffix}
    re.compile(rf"^.*model\.layers\.(?P<L>\d+)(?:\.mlp)?\.experts\.(?P<proj>{STACKED_PROJS})\.(?P<suffix>[\w.]+)$"),
]


def is_tier0(name: str) -> bool:
    return any(p.match(name) for p in TIER0_PATTERNS)


def match_expert_tensor(name: str) -> Optional[Tuple[int, int, str]]:
    """Return (layer, expert, rest) if name is a per-expert tensor in classic layout."""
    for p in EXPERT_PATTERNS:
        m = p.match(name)
        if m:
            return int(m.group("L")), int(m.group("E")), m.group("rest")
    return None


def match_stacked_tensor(name: str) -> Optional[Tuple[int, str, str]]:
    """Return (layer, proj, suffix) if name is a stacked expert tensor.
    suffix is typically 'weight', 'biases', 'scales', etc. — all quantization
    siblings must be sliced along axis 0 together."""
    for p in STACKED_PATTERNS:
        m = p.match(name)
        if m:
            return int(m.group("L")), m.group("proj"), m.group("suffix")
    return None


def read_safetensors_index(model_dir: Path) -> Dict[str, Tuple[Path, int, int]]:
    """Build {tensor_name: (file, offset, nbytes)} by reading safetensors headers."""
    index: Dict[str, Tuple[Path, int, int]] = {}
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise SystemExit(f"No .safetensors files in {model_dir}")
    for f in st_files:
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                # safetensors slices aren't supported by safe_open directly,
                # so we record (file, 0, 0) and let the loader read the whole
                # tensor when needed. Offset tracking is for future mmap work.
                index[k] = (f, 0, 0)
    return index


def load_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"No config.json in {model_dir}")
    return json.loads(cfg_path.read_text())


def detect_layout(index: Dict[str, Tuple[Path, int, int]]) -> Tuple[str, Dict]:
    """
    Detect MoE layout. Returns (layout_kind, info).
    layout_kind in {"stacked", "classic", "none"}.
    """
    stacked_hits: Dict[Tuple[int, str, str], str] = {}
    classic_hits: Dict[Tuple[int, int], List[str]] = {}

    for name in index.keys():
        s = match_stacked_tensor(name)
        if s is not None:
            stacked_hits[s] = name
            continue
        e = match_expert_tensor(name)
        if e is not None:
            layer, expert, rest = e
            classic_hits.setdefault((layer, expert), []).append(rest)

    if stacked_hits:
        layers = {L for (L, _, _) in stacked_hits.keys()}
        projs = sorted({p for (_, p, _) in stacked_hits.keys()})
        return "stacked", {
            "layers": sorted(layers),
            "projs": projs,
            "sample": next(iter(stacked_hits.values())),
        }
    if classic_hits:
        return "classic", {
            "experts": classic_hits,
        }
    return "none", {}


def classify_all(index: Dict[str, Tuple[Path, int, int]]) -> Tuple[List[str], Dict[Tuple[int, int], List[str]], Dict[Tuple[int, str], Dict[str, str]]]:
    """
    Returns:
      tier0_names:        list of tensor names that go into tier0.safetensors
      classic_experts:    {(layer, expert): [tensor_name, ...]}
      stacked_tensors:    {(layer, proj): {suffix: tensor_name}}
    """
    tier0: List[str] = []
    classic: Dict[Tuple[int, int], List[str]] = {}
    stacked: Dict[Tuple[int, str], Dict[str, str]] = {}

    for name in index.keys():
        s = match_stacked_tensor(name)
        if s is not None:
            L, proj, suffix = s
            stacked.setdefault((L, proj), {})[suffix] = name
            continue
        e = match_expert_tensor(name)
        if e is not None:
            layer, expert, _ = e
            classic.setdefault((layer, expert), []).append(name)
            continue
        # Not an expert tensor — goes to tier 0
        tier0.append(name)

    # Sanity: explicitly tier0-patterned names that were caught above as expert
    # are NOT in tier0 (they're per-expert). Anything not matching expert patterns
    # goes to tier0 unconditionally — this is the safe default.
    return tier0, classic, stacked


def write_tier0(model_dir: Path, dst_dir: Path, names: List[str]) -> Path:
    """Write all tier-0 tensors into one file."""
    out = dst_dir / "tier0.safetensors"
    bucket: Dict[str, torch.Tensor] = {}
    # Group by source file to minimize opens
    src_map: Dict[str, List[str]] = {}
    for n in names:
        f = n  # we need to find which file holds n
        src_map.setdefault(f, []).append(n)
    # We need the actual file index
    # Re-scan to find each tensor's source file
    file_for: Dict[str, Path] = {}
    for st in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(st), framework="pt") as fh:
            for k in fh.keys():
                file_for[k] = st
    for n in names:
        with safe_open(str(file_for[n]), framework="pt") as fh:
            bucket[n] = fh.get_tensor(n)
    save_file(bucket, str(out))
    return out


def write_dense_layer_shards(
    dst_dir: Path,
    tier0_names: List[str],
    file_for: Dict[str, Path],
) -> List[dict]:
    """For dense (non-MoE) models: split per-layer weights into per-layer files.

    A "layer" is any tensor matching `model.layers.{N}.*`. These get pulled
    OUT of tier0_names and into layer.L{NN}.safetensors files. Tensors NOT
    matching (embed_tokens, final norm, lm_head) stay in tier0.safetensors.

    NovaMLX-TIE: small norm/layernorm weights (RMSNorm, LayerNorm) stay in
    tier0 because the sync hook only handles Linear/SwitchLinear. Without
    this, norm weights would be unpopulated (random init) → garbage output.
    Pattern heuristic: tensor name contains "norm" OR has 1D shape (typical
    norm weight is `[hidden]`).
    """
    import re
    layer_re = re.compile(r"^.*model\.layers\.(?P<L>\d+)\..+$")
    norm_re = re.compile(r"(?:norm|layernorm)", re.IGNORECASE)
    # Group tier0 tensors by layer index, but keep norm-like tensors in tier0
    by_layer: Dict[int, List[str]] = {}
    keep_in_tier0: List[str] = []
    for n in tier0_names:
        m = layer_re.match(n)
        if m and not norm_re.search(n):
            by_layer.setdefault(int(m.group("L")), []).append(n)
        else:
            keep_in_tier0.append(n)

    tier0_names.clear()
    tier0_names.extend(keep_in_tier0)

    entries: List[dict] = []
    for L in sorted(by_layer.keys()):
        names = by_layer[L]
        fname = f"layer.L{L:02d}.safetensors"
        out = dst_dir / fname
        bucket = {}
        for n in names:
            with safe_open(str(file_for[n]), framework="pt") as fh:
                bucket[n] = fh.get_tensor(n)
        save_file(bucket, str(out))
        entries.append({
            "layer": L,
            "file": fname,
            "bytes": out.stat().st_size,
            "tensors": names,
        })
        print(f"  wrote {fname} ({out.stat().st_size // 1024 // 1024}MB, {len(names)} tensors)")
    return entries


def write_classic_experts(
    dst_dir: Path,
    experts: Dict[Tuple[int, int], List[str]],
    file_for: Dict[str, Path],
) -> List[dict]:
    """One file per (layer, expert). Returns manifest entries."""
    entries = []
    for (L, E), names in sorted(experts.items()):
        fname = f"expert.L{L:02d}.E{E:03d}.safetensors"
        out = dst_dir / fname
        bucket = {}
        for n in names:
            with safe_open(str(file_for[n]), framework="pt") as fh:
                bucket[n] = fh.get_tensor(n)
        save_file(bucket, str(out))
        entries.append({
            "layer": L,
            "expert": E,
            "file": fname,
            "bytes": out.stat().st_size,
            "tensors": names,
        })
        print(f"  wrote {fname} ({out.stat().st_size // 1024 // 1024}MB)")
    return entries


def write_stacked_experts(
    dst_dir: Path,
    stacked: Dict[Tuple[int, str], Dict[str, str]],
    file_for: Dict[str, Path],
) -> List[dict]:
    """For each layer, load all projs * all suffixes (weight/biases/scales),
    slice each along axis 0, emit per-expert files. Returns manifest entries."""
    # Group by layer -> proj -> {suffix: tensor_name}
    by_layer: Dict[int, Dict[str, Dict[str, str]]] = {}
    for (L, proj), suffixes in stacked.items():
        by_layer.setdefault(L, {})[proj] = suffixes

    entries = []
    for L in sorted(by_layer.keys()):
        projs = by_layer[L]
        # Load all (proj, suffix) tensors for this layer
        # Key by (proj, suffix) so we can reconstruct output names.
        tensors: Dict[Tuple[str, str], torch.Tensor] = {}
        for proj, suffixes in projs.items():
            for suffix, name in suffixes.items():
                with safe_open(str(file_for[name]), framework="pt") as fh:
                    tensors[(proj, suffix)] = fh.get_tensor(name)
        # Determine num_experts from axis 0 of first tensor
        first = next(iter(tensors.values()))
        num_experts: int = first.shape[0]
        print(f"  layer {L}: {num_experts} experts, projs={sorted(set(p for p,_ in tensors.keys()))}, "
              f"suffixes={sorted(set(s for _,s in tensors.keys()))}")
        for E in range(num_experts):
            fname = f"expert.L{L:02d}.E{E:03d}.safetensors"
            out = dst_dir / fname
            bucket = {}
            for (proj, suffix), t in tensors.items():
                sliced = t[E]
                # Preserve original tensor name (with prefix + suffix) so MLX's
                # loader finds them at the same key path.
                # We reconstruct the original name from the source tensor name.
                src_name = projs[proj][suffix]
                bucket[src_name] = sliced.contiguous().clone()
            save_file(bucket, str(out))
            entries.append({
                "layer": L,
                "expert": E,
                "file": fname,
                "bytes": out.stat().st_size,
                "tensors": list(bucket.keys()),
                "stacked_source": True,
            })
        # Free per-layer tensors
        del tensors
    return entries


def copy_aux_files(src: Path, dst: Path) -> None:
    """Copy tokenizer.json, tokenization_config.json, etc. (not safetensors/config.json)."""
    SKIP = {".safetensors"}
    for f in src.iterdir():
        if f.is_dir() or f.suffix in SKIP or f.name == "config.json":
            continue
        shutil.copy2(f, dst / f.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="Source model directory")
    ap.add_argument("--dst", required=True, type=Path, help="Output .tiered directory")
    ap.add_argument("--force", action="store_true", help="Overwrite existing dst")
    args = ap.parse_args()

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not src.is_dir():
        print(f"ERROR: src not a directory: {src}", file=sys.stderr)
        return 2
    if dst.exists():
        if not args.force:
            print(f"ERROR: dst exists: {dst} (use --force)", file=sys.stderr)
            return 2
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print(f"[ExpertShardLayout] {src} -> {dst}")
    config = load_config(src)
    print(f"  arch={config.get('architectures', config.get('model_type', '?'))}")

    index = read_safetensors_index(src)
    print(f"  total tensors: {len(index)}")

    layout, info = detect_layout(index)
    print(f"  layout: {layout}")
    if layout == "stacked":
        print(f"    layers={len(info['layers'])} projs={info['projs']}")
    elif layout == "classic":
        print(f"    expert tensors={sum(len(v) for v in info['experts'].values())}")

    tier0_names, classic, stacked = classify_all(index)
    print(f"  tier0 tensors: {len(tier0_names)}")
    print(f"  classic expert tensors: {sum(len(v) for v in classic.values())}")
    print(f"  stacked tensors: {len(stacked)}")

    # Build file_for so we don't rescan repeatedly
    file_for: Dict[str, Path] = {}
    for st in sorted(src.glob("*.safetensors")):
        with safe_open(str(st), framework="pt") as fh:
            for k in fh.keys():
                file_for[k] = st

    # Dense path: split per-layer weights out of tier0 BEFORE writing tier0 file.
    # write_dense_layer_shards mutates tier0_names in place to remove tensors
    # that go into per-layer files (so tier0 ends up with embed/lm_head/norm only).
    layer_entries: List[dict] = []
    if not stacked and not classic:
        print("[layers] splitting per-layer weights (dense strategy)...")
        layer_entries = write_dense_layer_shards(dst, tier0_names, file_for)

    # Write tier0.safetensors (after potential dense split above)
    print(f"[tier0] writing {len(tier0_names)} shared tensors...")
    tier0_bucket: Dict[str, torch.Tensor] = {}
    for n in tier0_names:
        with safe_open(str(file_for[n]), framework="pt") as fh:
            tier0_bucket[n] = fh.get_tensor(n)
    save_file(tier0_bucket, str(dst / "tier0.safetensors"))
    print(f"  tier0.safetensors: {(dst / 'tier0.safetensors').stat().st_size // 1024 // 1024}MB")
    del tier0_bucket

    # Write per-expert files (MoE models)
    expert_entries: List[dict] = []
    if stacked:
        print("[experts] slicing stacked expert tensors...")
        expert_entries = write_stacked_experts(dst, stacked, file_for)
    elif classic:
        print("[experts] copying classic expert tensors...")
        expert_entries = write_classic_experts(dst, classic, file_for)

    # Copy tokenizer + aux
    copy_aux_files(src, dst)

    # Copy config.json (TieredOffloadPolicy reads it)
    shutil.copy2(src / "config.json", dst / "config.json")

    # Determine strategy for manifest
    if stacked or classic:
        strategy = "expert"
    elif layer_entries:
        strategy = "layer"
    else:
        strategy = "none"

    # Write manifest
    manifest = {
        "version": 1,
        "converter": "expert_shard_layout.py",
        "source_model": str(src),
        "architecture": config.get("model_type", "unknown"),
        "layout": layout,
        "strategy": strategy,
        "tier0_file": "tier0.safetensors",
        "tier0_tensor_count": len(tier0_names),
        "tier0_bytes": (dst / "tier0.safetensors").stat().st_size,
        "expert_count": len(expert_entries),
        "experts": expert_entries,
        "layers": layer_entries,
    }
    (dst / "tier-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {dst}")
    print(f"  strategy: {strategy}")
    print(f"  tier0:   {manifest['tier0_bytes'] // 1024 // 1024}MB")
    total_expert_bytes = sum(e["bytes"] for e in expert_entries)
    total_layer_bytes = sum(e["bytes"] for e in layer_entries)
    if total_expert_bytes:
        print(f"  experts: {len(expert_entries)} files, {total_expert_bytes // 1024 // 1024}MB")
    if total_layer_bytes:
        print(f"  layers:  {len(layer_entries)} files, {total_layer_bytes // 1024 // 1024}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
