#!/usr/bin/env python3
"""Stream baidu/ERNIE-5.5-Flash-Base FP8 safetensors into an MLX quant.

The official checkpoint is about 288GB of FP8 (`model_type: ernie5_5_moe`).
Stock `mlx_lm.convert` has no architecture for it and would also have to
materialize the whole model. This tool reads one tensor at a time, dequantizes
the FP8 block scales, and writes MLX affine `weight` / `scales` / `biases`.

Recipe `ernie55-128gb` (128GB unified memory, with headroom):

  routed experts     2-bit, group 32   (~104GB for the 277B expert params)
  attention / shared 6-bit, group 64   (higher than the experts)
  embed / lm_head    8-bit, group 64
  fp8 gates          8-bit, group 64
  bf16 gates, norms  kept as stored    (already above 8-bit)

3-bit experts do not fit the 90-110GB expert budget at MLX group sizes
32/64/128 (3-bit group 128 is ~113GB before attention and KV). See
`docs/features/ernie-5-5-flash-base.md`.

Usage:
  python3 Scripts/ernie55_fp8_to_mlx.py \\
      --src /path/to/ERNIE-5.5-Flash-Base \\
      --dest /path/to/ERNIE-5.5-Flash-Base-mlx-ernie55-128gb \\
      --recipe ernie55-128gb \\
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

RECIPE_NAME = "ernie55-128gb"
MODEL_TYPE = "ernie5_5_moe"
# 43 MoE layers * 512 experts * 3 projections * 2048 * 2048.
# Measured from the FP8 index: experts.{0..511}.w{1,2,3} on layers 1..43.
FLASH_ROUTED_EXPERT_PARAMS = 512 * 3 * 2048 * 2048 * 43
BLOCK_DEFAULT = (128, 128)
MLX_GROUP_SIZES = (32, 64, 128)
DEFAULT_SHARD_BYTES = 5 * 1024 * 1024 * 1024
SIDECAR_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "chat_template.jinja",
    "configuration.json",
    "LICENSE",
    "README.md",
}

_EXPERT = re.compile(r"\.experts\.\d+\.")
_SHARED_EXPERT = re.compile(r"\.shared_experts\.")


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    group_size: int
    mode: str = "affine"

    def as_dict(self) -> Dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class Recipe:
    name: str
    expert: QuantSpec
    attention: QuantSpec
    shared: QuantSpec
    embed: QuantSpec
    gate: QuantSpec

    def spec_for(self, role: str) -> Optional[QuantSpec]:
        return {
            "expert": self.expert,
            "attention": self.attention,
            "shared": self.shared,
            "embed": self.embed,
            "gate": self.gate,
        }.get(role)


RECIPES = {
    RECIPE_NAME: Recipe(
        name=RECIPE_NAME,
        expert=QuantSpec(bits=2, group_size=32),
        attention=QuantSpec(bits=6, group_size=64),
        shared=QuantSpec(bits=6, group_size=64),
        embed=QuantSpec(bits=8, group_size=64),
        gate=QuantSpec(bits=8, group_size=64),
    )
}


def tensor_role(name: str) -> str:
    """Classify a checkpoint key. Scale tensors are consumed with their weight."""
    if name.endswith("_scale_inv"):
        return "scale"
    if _SHARED_EXPERT.search(name):
        return "shared"
    if _EXPERT.search(name):
        return "expert"
    if name == "lm_head.weight" or name.endswith("embed_tokens.weight"):
        return "embed"
    if ".gate." in name or ".gate_proj" in name or ".gate.weight_" in name:
        return "gate"
    if ".self_attn." in name:
        return "attention"
    if (
        ".mlp." in name
        or "latent_proj" in name
        or name.endswith(".e_proj.weight")
        or "hyper_connection" in name
    ):
        return "shared"
    return "keep"


def affine_storage_bytes(numel: int, bits: int, group_size: int) -> int:
    """Packed uint32 payload plus bf16 scale and bf16 bias."""
    if group_size <= 0 or numel % group_size != 0:
        raise ValueError(f"numel {numel} is not divisible by group_size {group_size}")
    if (numel * bits) % 8 != 0:
        raise ValueError(f"bit packing is not byte-aligned for numel={numel} bits={bits}")
    groups = numel // group_size
    return numel * bits // 8 + groups * 4


def expert_storage_bytes(recipe: Recipe, numel: int = FLASH_ROUTED_EXPERT_PARAMS) -> int:
    return affine_storage_bytes(numel, recipe.expert.bits, recipe.expert.group_size)


def e4m3fn_to_fp32(raw: np.ndarray) -> np.ndarray:
    """Decode IEEE-style FP8 E4M3FN bytes. Exponent 15 is finite except mantissa 7 (NaN)."""
    u = np.asarray(raw, dtype=np.uint8).astype(np.int32)
    sign = (u >> 7) & 1
    exp = (u >> 3) & 0xF
    man = u & 0x7
    sub = exp == 0
    nan = (exp == 0xF) & (man == 0x7)
    normal = (1.0 + man.astype(np.float64) / 8.0) * np.exp2(exp.astype(np.float64) - 7.0)
    subnormal = (man.astype(np.float64) / 8.0) * np.exp2(-6.0)
    val = np.where(sub, subnormal, normal)
    val = np.where(sign == 1, -val, val)
    val = np.where(nan, np.nan, val)
    return val.astype(np.float32)


def decode_scale(raw: np.ndarray, dtype: str) -> np.ndarray:
    """weight_scale_inv. Float32 is already the multiplier. ue8m0 bytes are 2^(e-127)."""
    if dtype == "F32":
        return np.asarray(raw, dtype=np.float32)
    if dtype in ("F8_E8M0", "U8", "I8"):
        exp = np.asarray(raw, dtype=np.uint8).astype(np.int32) - 127
        return np.exp2(exp.astype(np.float32))
    raise ValueError(f"unsupported FP8 scale dtype {dtype}")


def dequant_fp8_block(
    weight_u8: np.ndarray, scale: np.ndarray, block: Tuple[int, int]
) -> np.ndarray:
    values = e4m3fn_to_fp32(weight_u8)
    n, k = values.shape
    bs0, bs1 = block
    expanded = np.repeat(np.repeat(scale, bs0, axis=0), bs1, axis=1)
    return values * expanded[:n, :k]


def bf16_to_fp32(raw: np.ndarray) -> np.ndarray:
    u16 = np.asarray(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def fp32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 bits."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + (np.uint32(0x7FFF) + lsb)
    return (rounded >> 16).astype(np.uint16)


def _pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack unsigned codes along the last axis into MLX uint32 words.

    Power-of-two widths match `pack_and_quantize`: value k sits at shift k*bits.
    Other widths (3, 5, 6) are a little-endian bitstream, which is the same
    byte layout as the MLX CPU quantize loop.
    """
    if codes.dtype != np.uint32:
        codes = codes.astype(np.uint32, copy=False)
    rows, width = codes.shape
    if bits in (1, 2, 4, 8):
        el = 32 // bits
        if width % el != 0:
            raise ValueError(f"width {width} is not divisible by {el} for {bits}-bit")
        view = codes.reshape(rows, width // el, el)
        packed = np.zeros((rows, width // el), dtype=np.uint32)
        for i in range(el):
            packed |= view[:, :, i] << np.uint32(i * bits)
        return packed
    out_cols = width * bits // 32
    if width * bits != out_cols * 32:
        raise ValueError(f"width {width} * {bits} is not a multiple of 32")
    packed = np.zeros((rows, out_cols), dtype=np.uint32)
    # Fill 32-bit words from the LSB stream without a full bit tensor.
    bit_pos = 0
    for col in range(width):
        value = codes[:, col]
        for b in range(bits):
            word = bit_pos // 32
            shift = bit_pos % 32
            packed[:, word] |= ((value >> np.uint32(b)) & np.uint32(1)) << np.uint32(shift)
            bit_pos += 1
    return packed


def affine_quantize(
    weights: np.ndarray, group_size: int, bits: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MLX affine quantize. Returns uint32 weight, float32 scales, float32 biases.

    `w ~= scale * q + bias`, with `q` in `[0, 2^bits - 1]`. The sign of `scale`
    follows MLX: negative when `|max| >= |min|`.
    """
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"unsupported bits {bits}")
    if group_size not in MLX_GROUP_SIZES:
        raise ValueError(f"unsupported group_size {group_size}")
    array = np.asarray(weights, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("affine quantize needs a matrix")
    k = array.shape[-1]
    if k % group_size != 0:
        raise ValueError(f"last dimension {k} is not divisible by {group_size}")
    rows = int(np.prod(array.shape[:-1]))
    grouped = array.reshape(rows, k // group_size, group_size)
    w_max = grouped.max(axis=-1, keepdims=True)
    w_min = grouped.min(axis=-1, keepdims=True)
    n_bins = float((1 << bits) - 1)
    eps = np.float32(1e-7)
    mask = np.abs(w_min) > np.abs(w_max)
    scales = np.maximum((w_max - w_min) / n_bins, eps).astype(np.float32)
    scales = np.where(mask, scales, -scales)
    edge = np.where(mask, w_min, w_max).astype(np.float32)
    q0 = np.rint(edge / scales)
    scales = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1.0), scales).astype(np.float32)
    biases = np.where(q0 == 0, np.float32(0), edge).astype(np.float32)
    q = np.rint((grouped - biases) / scales)
    q = np.clip(q, 0, n_bins).astype(np.uint32)
    packed = _pack_codes(q.reshape(rows, k), bits)
    out_shape = array.shape[:-1] + (packed.shape[-1],)
    scale_shape = array.shape[:-1] + (k // group_size,)
    return (
        packed.reshape(out_shape),
        scales.reshape(scale_shape),
        biases.reshape(scale_shape),
    )


def affine_dequantize(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int,
    bits: int,
) -> np.ndarray:
    """Inverse of `affine_quantize` for tests. Scales may be bf16-rounded."""
    flat = np.asarray(packed, dtype=np.uint32)
    rows = int(np.prod(flat.shape[:-1]))
    words = flat.reshape(rows, flat.shape[-1])
    in_features = int(scales.shape[-1]) * group_size
    codes = np.zeros((rows, in_features), dtype=np.uint32)
    if bits in (1, 2, 4, 8):
        el = 32 // bits
        mask = np.uint32((1 << bits) - 1)
        n_words = words.shape[-1]
        for i in range(el):
            codes[:, i::el] = (words[:, :] >> np.uint32(i * bits)) & mask
        if n_words * el != in_features:
            raise ValueError(
                f"packed width {n_words} * {el} != {in_features} for {bits}-bit"
            )
    else:
        bit_pos = 0
        for col in range(in_features):
            acc = np.zeros(rows, dtype=np.uint32)
            for b in range(bits):
                word = bit_pos // 32
                shift = bit_pos % 32
                acc |= ((words[:, word] >> np.uint32(shift)) & np.uint32(1)) << np.uint32(b)
                bit_pos += 1
            codes[:, col] = acc
    scales_f = np.asarray(scales, dtype=np.float32).reshape(rows, -1)
    biases_f = np.asarray(biases, dtype=np.float32).reshape(rows, -1)
    out = np.empty((rows, in_features), dtype=np.float32)
    for g in range(scales_f.shape[-1]):
        sl = slice(g * group_size, (g + 1) * group_size)
        out[:, sl] = scales_f[:, g : g + 1] * codes[:, sl] + biases_f[:, g : g + 1]
    return out.reshape(scales.shape[:-1] + (in_features,))


def choose_group_size(width: int, preferred: int) -> Optional[int]:
    if width % preferred == 0 and preferred in MLX_GROUP_SIZES:
        return preferred
    for candidate in MLX_GROUP_SIZES:
        if width % candidate == 0:
            return candidate
    return None


@dataclass
class TensorRef:
    name: str
    path: Path
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int
    data_start: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass
class PlannedTensor:
    ref: TensorRef
    scale: Optional[TensorRef]
    role: str
    action: str  # quantize | copy | dequant_bf16
    spec: Optional[QuantSpec]


class SafetensorSource:
    """Header index plus one-tensor reads. Never opens every shard together."""

    def __init__(self, root: Path):
        self.root = root
        self._headers: Dict[Path, Tuple[Dict[str, Any], int]] = {}
        self.tensors: Dict[str, TensorRef] = {}
        self._index()

    def _header(self, path: Path) -> Tuple[Dict[str, Any], int]:
        cached = self._headers.get(path)
        if cached is not None:
            return cached
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"short safetensors header in {path}")
            (nbytes,) = struct.unpack("<Q", raw_len)
            payload = handle.read(nbytes)
        header = json.loads(payload)
        cached = (header, 8 + nbytes)
        self._headers[path] = cached
        return cached

    def _index(self) -> None:
        files = sorted(self.root.glob("*.safetensors"))
        if not files:
            raise SystemExit(f"no .safetensors files in {self.root}")
        for path in files:
            header, data_start = self._header(path)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                offsets = meta["data_offsets"]
                self.tensors[name] = TensorRef(
                    name=name,
                    path=path,
                    dtype=meta["dtype"],
                    shape=tuple(int(x) for x in meta["shape"]),
                    begin=int(offsets[0]),
                    end=int(offsets[1]),
                    data_start=data_start,
                )

    def read(self, ref: TensorRef) -> np.ndarray:
        with ref.path.open("rb") as handle:
            handle.seek(ref.data_start + ref.begin)
            blob = handle.read(ref.nbytes)
        if len(blob) != ref.nbytes:
            raise ValueError(f"short read of {ref.name}")
        return _decode_blob(blob, ref.dtype, ref.shape)


def _decode_blob(blob: bytes, dtype: str, shape: Tuple[int, ...]) -> np.ndarray:
    if dtype == "F8_E4M3":
        return np.frombuffer(blob, dtype=np.uint8).reshape(shape).copy()
    if dtype == "BF16":
        return np.frombuffer(blob, dtype=np.uint16).reshape(shape).copy()
    if dtype == "F32":
        return np.frombuffer(blob, dtype="<f4").reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(blob, dtype="<f2").reshape(shape).copy()
    if dtype == "F8_E8M0":
        return np.frombuffer(blob, dtype=np.uint8).reshape(shape).copy()
    if dtype == "U8":
        return np.frombuffer(blob, dtype=np.uint8).reshape(shape).copy()
    if dtype == "I32":
        return np.frombuffer(blob, dtype="<i4").reshape(shape).copy()
    if dtype == "I64":
        return np.frombuffer(blob, dtype="<i8").reshape(shape).copy()
    if dtype == "BOOL":
        return np.frombuffer(blob, dtype=np.bool_).reshape(shape).copy()
    raise ValueError(f"unsupported safetensors dtype {dtype}")


def _to_float_matrix(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "F8_E4M3":
        raise ValueError("FP8 weights need a scale")
    if dtype == "BF16":
        return bf16_to_fp32(array)
    if dtype == "F16":
        return array.astype(np.float32)
    if dtype == "F32":
        return array.astype(np.float32, copy=False)
    raise ValueError(f"cannot quantize dtype {dtype}")


def plan_tensors(
    source: SafetensorSource, recipe: Recipe, block: Tuple[int, int]
) -> List[PlannedTensor]:
    names = set(source.tensors)
    planned: List[PlannedTensor] = []
    for name in sorted(names):
        role = tensor_role(name)
        if role == "scale":
            continue
        ref = source.tensors[name]
        scale = _scale_ref(name, source)
        spec = recipe.spec_for(role)
        action, chosen = _decide(ref, scale, role, spec)
        planned.append(
            PlannedTensor(ref=ref, scale=scale, role=role, action=action, spec=chosen)
        )
    _ = block
    return planned


def _scale_ref(name: str, source: SafetensorSource) -> Optional[TensorRef]:
    candidate = name + "_scale_inv"
    return source.tensors.get(candidate)


def _decide(
    ref: TensorRef,
    scale: Optional[TensorRef],
    role: str,
    spec: Optional[QuantSpec],
) -> Tuple[str, Optional[QuantSpec]]:
    # Only `*.weight` matrices become MLX QuantizedLinear tensors. Parameters
    # such as `gate.weight_1` and `vha_postmix_U` keep their checkpoint names.
    if spec is None or not name_is_weight(ref.name) or len(ref.shape) < 2:
        return _keep(ref)
    if role == "gate" and scale is None and ref.dtype != "F8_E4M3":
        return "copy", None
    group = choose_group_size(ref.shape[-1], spec.group_size)
    if group is None:
        return _keep(ref)
    if ref.dtype == "F8_E4M3" and scale is None:
        raise ValueError(f"{ref.name} is FP8 but has no weight_scale_inv")
    if ref.dtype not in ("F8_E4M3", "BF16", "F16", "F32"):
        return _keep(ref)
    return "quantize", QuantSpec(bits=spec.bits, group_size=group, mode=spec.mode)


def _keep(ref: TensorRef) -> Tuple[str, Optional[QuantSpec]]:
    if ref.dtype == "F8_E4M3":
        return "dequant_bf16", None
    return "copy", None


def name_is_weight(name: str) -> bool:
    return name.endswith(".weight")


def module_path(weight_name: str) -> str:
    if weight_name.endswith(".weight"):
        return weight_name[: -len(".weight")]
    return weight_name


def estimate_plan(planned: Iterable[PlannedTensor]) -> Dict[str, Any]:
    roles: Dict[str, Dict[str, int]] = {}
    total = 0
    for item in planned:
        bucket = roles.setdefault(item.role, {"count": 0, "bytes": 0, "params": 0})
        bucket["count"] += 1
        params = int(np.prod(item.ref.shape)) if item.ref.shape else 0
        bucket["params"] += params
        if item.action == "quantize" and item.spec is not None:
            nbytes = affine_storage_bytes(params, item.spec.bits, item.spec.group_size)
        elif item.action == "dequant_bf16":
            nbytes = params * 2
        else:
            nbytes = item.ref.nbytes
        bucket["bytes"] += nbytes
        total += nbytes
    return {"estimated_bytes": total, "roles": roles}


def quantize_matrix(matrix: np.ndarray, spec: QuantSpec, row_chunk: int = 2048):
    """Quantize along rows so a 1.6GB embedding is not a single fp32 allocation."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= row_chunk:
        return affine_quantize(array, spec.group_size, spec.bits)
    pieces_w = []
    pieces_s = []
    pieces_b = []
    for start in range(0, array.shape[0], row_chunk):
        chunk = array[start : start + row_chunk]
        weight, scales, biases = affine_quantize(chunk, spec.group_size, spec.bits)
        pieces_w.append(weight)
        pieces_s.append(scales)
        pieces_b.append(biases)
    return (
        np.concatenate(pieces_w, axis=0),
        np.concatenate(pieces_s, axis=0),
        np.concatenate(pieces_b, axis=0),
    )


def materialize_float(
    source: SafetensorSource, item: PlannedTensor, block: Tuple[int, int]
) -> np.ndarray:
    raw = source.read(item.ref)
    if item.ref.dtype == "F8_E4M3":
        if item.scale is None:
            raise ValueError(f"missing scale for {item.ref.name}")
        scale_raw = source.read(item.scale)
        scale = decode_scale(scale_raw, item.scale.dtype)
        return dequant_fp8_block(raw, scale, block)
    return _to_float_matrix(raw, item.ref.dtype)


class ShardWriter:
    def __init__(self, dest: Path, max_bytes: int):
        self.dest = dest
        self.max_bytes = max_bytes
        self._pending: List[Tuple[str, str, np.ndarray]] = []
        self._pending_bytes = 0
        self._files: List[Path] = []
        self.weight_map: Dict[str, str] = {}
        self.data_bytes = 0

    def add(self, name: str, dtype: str, array: np.ndarray) -> None:
        blob = _tensor_bytes(dtype, array)
        if self._pending and self._pending_bytes + len(blob) > self.max_bytes:
            self.flush()
        self._pending.append((name, dtype, array))
        self._pending_bytes += len(blob)

    def flush(self) -> None:
        if not self._pending:
            return
        index = len(self._files) + 1
        path = self.dest / f".ernie55-shard-{index:05d}.safetensors"
        written = _write_safetensors(path, self._pending)
        self.data_bytes += written
        self._files.append(path)
        for name, _, _ in self._pending:
            self.weight_map[name] = path.name
        self._pending.clear()
        self._pending_bytes = 0

    def finalize(self) -> List[Path]:
        self.flush()
        count = len(self._files)
        renamed: List[Path] = []
        new_map: Dict[str, str] = {}
        for index, path in enumerate(self._files, start=1):
            final_name = f"model-{index:05d}-of-{count:05d}.safetensors"
            final = self.dest / final_name
            path.rename(final)
            renamed.append(final)
        for name, old in self.weight_map.items():
            number = int(old.split("-")[2].split(".")[0])
            new_map[name] = f"model-{number:05d}-of-{count:05d}.safetensors"
        self.weight_map = new_map
        self._files = renamed
        return renamed


def _tensor_bytes(dtype: str, array: np.ndarray) -> bytes:
    if dtype == "U32":
        return np.ascontiguousarray(array, dtype="<u4").tobytes()
    if dtype == "BF16":
        if array.dtype == np.uint16:
            return np.ascontiguousarray(array, dtype="<u2").tobytes()
        return np.ascontiguousarray(fp32_to_bf16(array), dtype="<u2").tobytes()
    if dtype == "F32":
        return np.ascontiguousarray(array, dtype="<f4").tobytes()
    if dtype == "F16":
        return np.ascontiguousarray(array, dtype="<f2").tobytes()
    if dtype == "U8":
        return np.ascontiguousarray(array, dtype=np.uint8).tobytes()
    if dtype == "I32":
        return np.ascontiguousarray(array, dtype="<i4").tobytes()
    if dtype == "I64":
        return np.ascontiguousarray(array, dtype="<i8").tobytes()
    raise ValueError(f"cannot write dtype {dtype}")


def _write_safetensors(path: Path, tensors: Sequence[Tuple[str, str, np.ndarray]]) -> int:
    header: Dict[str, Any] = {"__metadata__": {"format": "mlx"}}
    blobs: List[bytes] = []
    cursor = 0
    for name, dtype, array in tensors:
        blob = _tensor_bytes(dtype, array)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [cursor, cursor + len(blob)],
        }
        blobs.append(blob)
        cursor += len(blob)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)
    return cursor


def load_config(src: Path) -> Dict[str, Any]:
    path = src / "config.json"
    if not path.exists():
        raise SystemExit(f"no config.json in {src}")
    return json.loads(path.read_text())


def block_size(config: Dict[str, Any]) -> Tuple[int, int]:
    quant = config.get("quantization_config") or {}
    raw = quant.get("weight_block_size") or list(BLOCK_DEFAULT)
    if len(raw) != 2:
        raise SystemExit(f"weight_block_size must have 2 entries, got {raw}")
    return int(raw[0]), int(raw[1])


def assert_ernie_config(config: Dict[str, Any]) -> None:
    model_type = config.get("model_type")
    if model_type != MODEL_TYPE:
        raise SystemExit(
            f"expected model_type {MODEL_TYPE}, found {model_type!r}. "
            "This converter is only for ERNIE-5.5-Flash-Base."
        )


def build_output_config(
    config: Dict[str, Any], recipe: Recipe, quant_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    out = dict(config)
    # The HF fp8 block is not an MLX quant. Replace it with the mlx-lm map.
    default = recipe.attention.as_dict()
    quantization: Dict[str, Any] = dict(default)
    for path, spec in sorted(quant_map.items()):
        quantization[path] = spec
    out["quantization"] = quantization
    out["quantization_config"] = quantization
    out["novamlx"] = {
        "recipe": recipe.name,
        "endpoint": "/v1/completions",
        "kernels": "hca-mla-pending",
        "source_quant": "fp8-e4m3-block",
        "note": (
            "Pretrained base checkpoint. Serve with POST /v1/completions. "
            "HCA and MLA kernels are not in this build."
        ),
    }
    out["torch_dtype"] = "bfloat16"
    return out


def copy_sidecars(src: Path, dest: Path) -> None:
    for name in SIDECAR_NAMES:
        source = src / name
        if source.is_file():
            shutil.copy2(source, dest / name)


def convert(
    src: Path,
    dest: Path,
    recipe: Recipe,
    dry_run: bool,
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> Dict[str, Any]:
    config = load_config(src)
    assert_ernie_config(config)
    block = block_size(config)
    source = SafetensorSource(src)
    planned = plan_tensors(source, recipe, block)
    summary = estimate_plan(planned)
    summary["recipe"] = recipe.name
    summary["tensors"] = len(planned)
    summary["source_tensors"] = len(source.tensors)
    if dry_run:
        summary["dry_run"] = True
        return summary

    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"destination {dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(dest, max_shard_bytes)
    quant_map: Dict[str, Dict[str, Any]] = {}
    written = 0
    for index, item in enumerate(planned, start=1):
        _emit_one(source, item, block, writer, quant_map)
        written += 1
        if index % 1024 == 0:
            print(f"[ernie55] converted {index}/{len(planned)}", file=sys.stderr)
    writer.finalize()
    index_doc = {
        "metadata": {"total_size": writer.data_bytes, "format": "mlx"},
        "weight_map": dict(sorted(writer.weight_map.items())),
    }
    (dest / "model.safetensors.index.json").write_text(json.dumps(index_doc, indent=2))
    output_config = build_output_config(config, recipe, quant_map)
    (dest / "config.json").write_text(json.dumps(output_config, indent=2))
    copy_sidecars(src, dest)
    summary["dry_run"] = False
    summary["output_bytes"] = writer.data_bytes
    summary["shards"] = len(writer._files)
    summary["quantized_modules"] = len(quant_map)
    return summary


def _emit_one(
    source: SafetensorSource,
    item: PlannedTensor,
    block: Tuple[int, int],
    writer: ShardWriter,
    quant_map: Dict[str, Dict[str, Any]],
) -> None:
    if item.action == "copy":
        raw = source.read(item.ref)
        writer.add(item.ref.name, item.ref.dtype, raw)
        del raw
        return
    if item.action == "dequant_bf16":
        values = materialize_float(source, item, block)
        writer.add(item.ref.name, "BF16", values)
        del values
        return
    if item.action != "quantize" or item.spec is None:
        raise ValueError(f"unhandled action {item.action} for {item.ref.name}")
    values = materialize_float(source, item, block)
    weight, scales, biases = quantize_matrix(values, item.spec)
    del values
    prefix = module_path(item.ref.name)
    writer.add(f"{prefix}.weight", "U32", weight)
    writer.add(f"{prefix}.scales", "BF16", scales)
    writer.add(f"{prefix}.biases", "BF16", biases)
    quant_map[prefix] = item.spec.as_dict()
    del weight, scales, biases


def _print_summary(summary: Dict[str, Any], dry_run: bool) -> None:
    tag = "ERNIE55-DRY-RUN" if dry_run else "ERNIE55-CONVERT"
    payload = {
        "recipe": summary["recipe"],
        "tensors": summary["tensors"],
        "estimated_bytes": summary["estimated_bytes"],
        "roles": summary["roles"],
    }
    if not dry_run:
        payload["output_bytes"] = summary.get("output_bytes")
        payload["shards"] = summary.get("shards")
        payload["quantized_modules"] = summary.get("quantized_modules")
    print(f"{tag} {json.dumps(payload, sort_keys=True)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path, help="FP8 checkpoint directory")
    parser.add_argument("--dest", required=True, type=Path, help="MLX output directory")
    parser.add_argument("--recipe", required=True, help="quant recipe name")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read headers only and print the size estimate",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_SHARD_BYTES,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    recipe = RECIPES.get(args.recipe)
    if recipe is None:
        known = ", ".join(sorted(RECIPES))
        print(f"unknown recipe {args.recipe!r}. known: {known}", file=sys.stderr)
        return 2
    if not args.src.is_dir():
        print(f"--src is not a directory: {args.src}", file=sys.stderr)
        return 2
    summary = convert(
        args.src,
        args.dest,
        recipe,
        dry_run=args.dry_run,
        max_shard_bytes=args.max_shard_bytes,
    )
    _print_summary(summary, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
