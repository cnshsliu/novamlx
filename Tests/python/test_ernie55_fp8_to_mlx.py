"""Streaming ERNIE-5.5 FP8 convert. Uses synthetic tensors, not the 288GB checkpoint."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts"))

import ernie55_fp8_to_mlx as conv  # noqa: E402


def _write_safetensors(path: Path, tensors: dict) -> None:
    """tensors: name -> (dtype, ndarray). ndarray for F8_E4M3 is uint8."""
    header = {"__metadata__": {"format": "pt"}}
    blobs = []
    cursor = 0
    for name, (dtype, array) in tensors.items():
        if dtype == "F8_E4M3":
            blob = np.ascontiguousarray(array, dtype=np.uint8).tobytes()
        elif dtype == "F32":
            blob = np.ascontiguousarray(array, dtype="<f4").tobytes()
        elif dtype == "BF16":
            blob = np.ascontiguousarray(array, dtype="<u2").tobytes()
        else:
            raise AssertionError(dtype)
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [cursor, cursor + len(blob)],
        }
        blobs.append(blob)
        cursor += len(blob)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _read_safetensors(path: Path) -> dict:
    blob = path.read_bytes()
    (nbytes,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8 : 8 + nbytes])
    data = blob[8 + nbytes :]
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        raw = data[begin:end]
        shape = tuple(meta["shape"])
        dtype = meta["dtype"]
        if dtype == "U32":
            array = np.frombuffer(raw, dtype="<u4").reshape(shape).copy()
        elif dtype == "BF16":
            array = np.frombuffer(raw, dtype="<u2").reshape(shape).copy()
        elif dtype == "F32":
            array = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
        else:
            array = raw
        out[name] = (dtype, array)
    return out


def _fp32_to_e4m3(values: np.ndarray) -> np.ndarray:
    """Coarse encoder so the fixture is real E4M3 bytes, not a random bit pattern."""
    x = np.clip(values.astype(np.float32), -448.0, 448.0)
    sign = (x < 0).astype(np.uint8) << 7
    ax = np.abs(x)
    exp = np.zeros(ax.shape, dtype=np.int32)
    man = np.zeros(ax.shape, dtype=np.int32)
    sub = ax < 2.0 ** -6
    # subnormal: value = (man/8) * 2^-6
    man_sub = np.rint(ax / (2.0 ** -6) * 8.0).astype(np.int32)
    man_sub = np.clip(man_sub, 0, 7)
    log = np.floor(np.log2(np.maximum(ax, 2.0 ** -6))).astype(np.int32)
    exp_n = np.clip(log + 7, 1, 15)
    scale = np.exp2((exp_n - 7).astype(np.float32))
    man_n = np.rint((ax / scale - 1.0) * 8.0).astype(np.int32)
    man_n = np.clip(man_n, 0, 6 if False else 7)
    # mantissa 7 with exp 15 is NaN; clamp those to 6.
    man_n = np.where((exp_n == 15) & (man_n == 7), 6, man_n)
    exp = np.where(sub, 0, exp_n)
    man = np.where(sub, man_sub, man_n)
    return (sign | (exp.astype(np.uint8) << 3) | man.astype(np.uint8)).astype(np.uint8)


class AffineTests(unittest.TestCase):
    def test_power_of_two_roundtrip_is_within_half_a_bin(self):
        rng = np.random.default_rng(0)
        weights = rng.normal(size=(4, 128)).astype(np.float32)
        packed, scales, biases = conv.affine_quantize(weights, group_size=64, bits=4)
        self.assertEqual(packed.dtype, np.uint32)
        self.assertEqual(packed.shape, (4, 16))
        self.assertEqual(scales.shape, (4, 2))
        restored = conv.affine_dequantize(packed, scales, biases, 64, 4)
        err = np.max(np.abs(restored - weights))
        # Scale is snapped so the edge hits an integer code, and clipping can
        # push one value past half a bin. A full bin still catches a bad pack.
        one_bin = np.max(np.abs(scales)) + 1e-4
        self.assertLessEqual(err, one_bin)

    def test_two_bit_and_six_bit_pack_roundtrip(self):
        rng = np.random.default_rng(1)
        for bits, group, width in ((2, 32, 64), (6, 64, 128), (8, 64, 64)):
            weights = rng.normal(size=(3, width)).astype(np.float32)
            packed, scales, biases = conv.affine_quantize(weights, group, bits)
            self.assertEqual(packed.shape[-1], width * bits // 32)
            restored = conv.affine_dequantize(packed, scales, biases, group, bits)
            one_bin = np.max(np.abs(scales)) + 1e-4
            self.assertLessEqual(np.max(np.abs(restored - weights)), one_bin)

    def test_expert_recipe_lands_in_the_128gb_budget(self):
        recipe = conv.RECIPES[conv.RECIPE_NAME]
        nbytes = conv.expert_storage_bytes(recipe)
        self.assertEqual(nbytes, 103_884_521_472)
        self.assertGreaterEqual(nbytes, 90 * 10**9)
        self.assertLessEqual(nbytes, 110 * 10**9)
        # 3-bit group 128 is the next MLX width and misses the expert budget.
        three_bit = conv.affine_storage_bytes(conv.FLASH_ROUTED_EXPERT_PARAMS, 3, 128)
        self.assertGreater(three_bit, 110 * 10**9)

    def test_roles_keep_shared_experts_out_of_the_routed_bucket(self):
        self.assertEqual(
            conv.tensor_role(
                "model.layers.1.block_sparse_moe.experts.7.w2.weight"
            ),
            "expert",
        )
        self.assertEqual(
            conv.tensor_role(
                "model.layers.1.block_sparse_moe.shared_experts.w1.weight"
            ),
            "shared",
        )
        self.assertEqual(conv.tensor_role("model.embed_tokens.weight"), "embed")
        self.assertEqual(conv.tensor_role("lm_head.weight"), "embed")
        self.assertEqual(
            conv.tensor_role("model.layers.1.block_sparse_moe.gate.weight"),
            "gate",
        )
        self.assertEqual(
            conv.tensor_role("model.layers.0.self_attn.gate_proj.weight"),
            "gate",
        )
        self.assertEqual(
            conv.tensor_role("model.layers.0.self_attn.o_proj.weight"),
            "attention",
        )
        self.assertEqual(
            conv.tensor_role("model.layers.1.block_sparse_moe.experts.0.w1.weight_scale_inv"),
            "scale",
        )
        recipe = conv.RECIPES[conv.RECIPE_NAME]
        self.assertGreater(recipe.attention.bits, recipe.expert.bits)
        self.assertGreater(recipe.shared.bits, recipe.expert.bits)
        self.assertGreaterEqual(recipe.embed.bits, 8)
        self.assertGreaterEqual(recipe.gate.bits, 8)


class ConvertTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        expert = np.linspace(-1.5, 1.5, 128 * 128, dtype=np.float32).reshape(128, 128)
        scale = np.array([[0.25]], dtype=np.float32)
        expert_q = _fp32_to_e4m3(expert / scale)
        attn = np.linspace(-0.5, 0.5, 64 * 64, dtype=np.float32).reshape(64, 64)
        attn_q = _fp32_to_e4m3(attn / 0.5)
        embed = conv.fp32_to_bf16(np.linspace(-0.2, 0.2, 8 * 64, dtype=np.float32).reshape(8, 64))
        gate = conv.fp32_to_bf16(np.ones((4, 64), dtype=np.float32))
        norm = conv.fp32_to_bf16(np.ones(64, dtype=np.float32))
        _write_safetensors(
            root / "model-00001-of-00002.safetensors",
            {
                "model.layers.1.block_sparse_moe.experts.0.w1.weight": ("F8_E4M3", expert_q),
                "model.layers.1.block_sparse_moe.experts.0.w1.weight_scale_inv": ("F32", scale),
                "model.layers.0.self_attn.o_proj.weight": ("F8_E4M3", attn_q),
                "model.layers.0.self_attn.o_proj.weight_scale_inv": (
                    "F32",
                    np.array([[0.5]], dtype=np.float32),
                ),
            },
        )
        _write_safetensors(
            root / "model-00002-of-00002.safetensors",
            {
                "model.embed_tokens.weight": ("BF16", embed),
                "model.layers.1.block_sparse_moe.gate.weight": ("BF16", gate),
                "model.layers.0.input_layernorm.weight": ("BF16", norm),
                "model.layers.0.self_attn.vha_postmix_U": (
                    "F8_E4M3",
                    _fp32_to_e4m3(np.linspace(-0.1, 0.1, 64, dtype=np.float32).reshape(1, 64) / 0.1),
                ),
                "model.layers.0.self_attn.vha_postmix_U_scale_inv": (
                    "F32",
                    np.array([[0.1]], dtype=np.float32),
                ),
            },
        )
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "ernie5_5_moe",
                    "architectures": ["Ernie5_5_MoeForCausalLM"],
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "vocab_size": 8,
                    "n_routed_experts": 1,
                    "quantization_config": {
                        "quant_method": "fp8",
                        "fmt": "e4m3",
                        "scale_fmt": "ue8m0",
                        "weight_block_size": [128, 128],
                    },
                }
            )
        )
        (root / "tokenizer_config.json").write_text('{"bos_token": "<s>"}')

    def test_dry_run_does_not_write_dest(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dest = Path(tmp) / "dest"
            src.mkdir()
            self._fixture(src)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "Scripts" / "ernie55_fp8_to_mlx.py"),
                    "--src",
                    str(src),
                    "--dest",
                    str(dest),
                    "--recipe",
                    "ernie55-128gb",
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertFalse(dest.exists())
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("ERNIE55-DRY-RUN")]
            self.assertEqual(len(line), 1)
            summary = json.loads(line[0].split(" ", 1)[1])
            self.assertEqual(summary["recipe"], "ernie55-128gb")
            self.assertGreater(summary["roles"]["expert"]["bytes"], 0)
            self.assertIn("embed", summary["roles"])

    def test_streaming_convert_matches_mlx_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            dest = Path(tmp) / "dest"
            src.mkdir()
            self._fixture(src)
            summary = conv.convert(
                src,
                dest,
                conv.RECIPES["ernie55-128gb"],
                dry_run=False,
                max_shard_bytes=1024,
            )
            self.assertGreater(summary["shards"], 1)
            config = json.loads((dest / "config.json").read_text())
            self.assertEqual(config["model_type"], "ernie5_5_moe")
            self.assertEqual(config["novamlx"]["endpoint"], "/v1/completions")
            self.assertEqual(config["novamlx"]["recipe"], "ernie55-128gb")
            self.assertEqual(config["quantization_config"]["bits"], 6)
            expert = config["quantization"][
                "model.layers.1.block_sparse_moe.experts.0.w1"
            ]
            self.assertEqual(expert["bits"], 2)
            self.assertEqual(expert["group_size"], 32)
            embed = config["quantization"]["model.embed_tokens"]
            self.assertEqual(embed["bits"], 8)
            attn = config["quantization"]["model.layers.0.self_attn.o_proj"]
            self.assertEqual(attn["bits"], 6)
            self.assertNotIn(
                "model.layers.1.block_sparse_moe.gate",
                config["quantization"],
            )
            index = json.loads((dest / "model.safetensors.index.json").read_text())
            names = set(index["weight_map"])
            self.assertIn("model.layers.1.block_sparse_moe.experts.0.w1.weight", names)
            self.assertIn("model.layers.1.block_sparse_moe.experts.0.w1.scales", names)
            self.assertIn("model.layers.1.block_sparse_moe.experts.0.w1.biases", names)
            self.assertNotIn(
                "model.layers.1.block_sparse_moe.experts.0.w1.weight_scale_inv",
                names,
            )
            self.assertIn("model.layers.1.block_sparse_moe.gate.weight", names)
            self.assertIn("model.layers.0.input_layernorm.weight", names)
            self.assertIn("model.layers.0.self_attn.vha_postmix_U", names)
            self.assertTrue((dest / "tokenizer_config.json").is_file())

            loaded = {}
            for shard in dest.glob("model-*.safetensors"):
                loaded.update(_read_safetensors(shard))
            weight_dtype, weight = loaded[
                "model.layers.1.block_sparse_moe.experts.0.w1.weight"
            ]
            self.assertEqual(weight_dtype, "U32")
            self.assertEqual(weight.shape, (128, 128 * 2 // 32))
            _, scales = loaded["model.layers.1.block_sparse_moe.experts.0.w1.scales"]
            _, biases = loaded["model.layers.1.block_sparse_moe.experts.0.w1.biases"]
            self.assertEqual(scales.shape[-1], 128 // 32)
            restored = conv.affine_dequantize(
                weight,
                conv.bf16_to_fp32(scales),
                conv.bf16_to_fp32(biases),
                32,
                2,
            )
            source = conv.SafetensorSource(src)
            planned = [p for p in conv.plan_tensors(source, conv.RECIPES["ernie55-128gb"], (128, 128)) if p.ref.name.endswith("experts.0.w1.weight")][0]
            original = conv.materialize_float(source, planned, (128, 128))
            # 2-bit reconstruction stays in the same direction as the FP8 values.
            corr = np.corrcoef(original.ravel(), restored.ravel())[0, 1]
            self.assertGreater(corr, 0.9)
            gate_dtype, gate = loaded["model.layers.1.block_sparse_moe.gate.weight"]
            self.assertEqual(gate_dtype, "BF16")
            self.assertEqual(gate.shape, (4, 64))

    def test_unknown_recipe_exits(self):
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "Scripts" / "ernie55_fp8_to_mlx.py"),
                "--src",
                "/tmp",
                "--dest",
                "/tmp/out",
                "--recipe",
                "nope",
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
