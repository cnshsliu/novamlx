#!/usr/bin/env python3
"""Patch mlx-swift-lm AttentionUtils.swift to try NovaMLX FusedQuantizedSDPA first."""

import os
import sys

CHECKOUT = ".build/checkouts/mlx-swift-lm/Libraries/MLXLMCommon/AttentionUtils.swift"
MARKER = "// NOVAMLX_FUSED_SDPA_PATCHED"


def main():
    if not os.path.exists(CHECKOUT):
        print(f"  AttentionUtils.swift not found at {CHECKOUT}")
        return

    with open(CHECKOUT, "r") as f:
        content = f.read()

    if MARKER in content:
        print("  AttentionUtils.swift already patched")
        return

    old_quantized_call = """        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )"""

    new_code = """        if let fused = _novamlxFusedSDPA?(
            queries,
            quantizedKeys,
            quantizedValues,
            scale,
            quantizedKVCache.groupSize,
            quantizedKVCache.bits
        ) {
            return fused
        }
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )"""

    if old_quantized_call not in content:
        print(
            "  WARNING: Could not find quantizedScaledDotProductAttention call in AttentionUtils.swift"
        )
        print("  The patch pattern may have changed. Skipping.")
        return

    content = content.replace(old_quantized_call, new_code)

    hook_decl = """import Foundation
import MLX

public nonisolated(unsafe) var _novamlxFusedSDPA: (
    (MLXArray, (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?), Float, Int, Int
) -> MLXArray?)?

"""

    content = content.replace("import Foundation\n", hook_decl, 1)
    content = content.replace("import MLX\n\nimport MLX\n", "import MLX\n\n", 1)

    with open(CHECKOUT, "w") as f:
        f.write(content)

    print("  Patched AttentionUtils.swift with FusedQuantizedSDPA hook")


if __name__ == "__main__":
    main()
