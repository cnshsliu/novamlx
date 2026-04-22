# NovaMLX Patches for MLX Dependencies

This document tracks all patches applied to external MLX libraries that NovaMLX depends on.

**Purpose:** When upgrading MLX dependencies, follow the workflow in `Scripts/upgrade-mlx-deps.sh`: compile with upstream code first, test with the models listed below. Only re-apply patches if upstream hasn't fixed the issue.

**Last updated:** 2026-04-21

---

## Summary

| Patch | Library | File | Status |
|---|---|---|---|
| Pre-quantized weight loading | mlx-swift-lm | `Libraries/MLXLMCommon/Load.swift` | Active |
| Pre-quantized SwitchLinear init | mlx-swift-lm | `Libraries/MLXLMCommon/SwitchLayers.swift` | Active |
| Pre-quantized QuantizedEmbedding init | mlx-swift | `Source/MLXNN/Quantized.swift` | Active |
| Local mlx-swift dependency | mlx-swift-lm | `Package.swift` | Active |

**Test model:** `mlx-community/Qwen3.6-35B-A3B-nvfp4`

---

## Patch 1: Pre-quantized weight loading (mlx-swift-lm)

**File:** `mlx-swift-lm/Libraries/MLXLMCommon/Load.swift`

**Problem:** The `quantize(model:filter:apply:)` function in `Load.swift` calls `MLX.quantized()` on every layer that has `.scales` in the weights dictionary. For pre-quantized models (weights already packed as uint32), this re-quantization crashes with SIGTRAP in the Metal backend (`mlx_quantize`).

**Models affected:**
- `mlx-community/Qwen3.6-35B-A3B-nvfp4` (nvfp4 4-bit + affine 8-bit gate layers)
- Any future model that ships with pre-quantized weights in safetensor files

**Fix:** Before the default `quantize()` call, detect pre-quantized paths:
1. Check if `weights["\(path).scales"]` exists AND `weights["\(path).weight"].dtype == .uint32`
2. For each pre-quantized module, create the quantized layer directly from the loaded arrays:
   - `Linear` → `QuantizedLinear(weight:bias:scales:biases:groupSize:bits:mode:)`
   - `Embedding` → `QuantizedEmbedding(weight:scales:biases:groupSize:bits:mode:)`
   - `SwitchLinear` → `QuantizedSwitchLinear(weight:bias:scales:biases:...)`
3. Call `model.update(modules:)` to replace the original layers
4. Skip these paths in the subsequent `quantize()` call

**Upstream status:** Not fixed as of mlx-swift-lm main (2026-04-21). The library's `quantize()` is designed for on-the-fly quantization, not for loading pre-quantized weights.

---

## Patch 2: Pre-quantized QuantizedSwitchLinear init (mlx-swift-lm)

**File:** `mlx-swift-lm/Libraries/MLXLMCommon/SwitchLayers.swift`

**Problem:** `QuantizedSwitchLinear` only has an init that takes a `SwitchLinear` and calls `MLX.quantized()` on its weight. For pre-quantized models, this crashes.

**Fix:** Added a new init that accepts pre-quantized `(weight, bias, scales, biases, inputDims, outputDims, numExperts, groupSize, bits, mode)` and calls `super.init()` directly without re-quantizing.

**Upstream status:** Not fixed as of main (2026-04-21).

---

## Patch 3: Pre-quantized QuantizedEmbedding init (mlx-swift)

**File:** `vendors/mlx-swift/Source/MLXNN/Quantized.swift`

**Problem:** `QuantizedEmbedding` only has an init that takes a float `weight` and calls `MLX.quantized()`. For pre-quantized embeddings (like in nvfp4 models), this crashes.

**Fix:** Added a new init:
```swift
public init(
    weight: MLXArray, scales: MLXArray, biases: MLXArray?,
    groupSize: Int, bits: Int, mode: QuantizationMode = .affine
)
```
This init stores the pre-quantized arrays directly without calling `MLX.quantized()`.

**Upstream status:** Not fixed as of mlx-swift v0.31.3 (2026-04-21).

---

## Patch 4: Local mlx-swift dependency path (mlx-swift-lm)

**File:** `mlx-swift-lm/Package.swift`

**Problem:** NovaMLX's `Package.swift` references mlx-swift via local path (`vendors/mlx-swift`), but mlx-swift-lm's `Package.swift` references it via URL (`https://github.com/ml-explore/mlx-swift`). SwiftPM sees the same package identity with two different origins and emits a conflict warning (future error).

**Fix:** Changed mlx-swift-lm's dependency from:
```swift
.package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
```
to:
```swift
.package(path: "../vendors/mlx-swift"),
```

This ensures both NovaMLX and mlx-swift-lm reference the same local copy.

**Upstream status:** Cannot be upstreamed - this is specific to NovaMLX's local fork setup.

---

## How to Test After Upgrade

### 1. Build
```bash
swift build -c release
```

### 2. Deploy
```bash
cp .build/release/NovaMLX dist/NovaMLX.app/Contents/MacOS/
pkill -f NovaMLX
open dist/NovaMLX.app
```

### 3. Load the test model
```bash
curl -s -X POST "http://127.0.0.1:6591/admin/models/load" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abcd1234" \
  -d '{"modelId": "mlx-community/Qwen3.6-35B-A3B-nvfp4"}'
```

**Expected (patched):** `{"loaded":true,"id":"mlx-community/Qwen3.6-35B-A3B-nvfp4",...}`

**Expected (unpatched / upstream):** `SIGTRAP` crash in `mlx_quantize` during model load. Check `~/Library/Logs/DiagnosticReports/` for `NovaMLX-*.ips`.

### 4. Send a chat completion
```bash
curl -s -X POST "http://127.0.0.1:6590/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer abcd1234" \
  -d '{
    "model": "mlx-community/Qwen3.6-35B-A3B-nvfp4",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "stream": false
  }'
```

**Expected:** Valid JSON response with assistant content.

---

## How to Upgrade MLX Dependencies

Run the automated script:
```bash
bash Scripts/upgrade-mlx-deps.sh
```

Or manually:
1. `cd vendors/mlx-swift && git fetch origin && git checkout main` (or latest tag)
2. `cd mlx-swift-lm && git fetch origin && git checkout main` (or latest tag)
3. `rm -rf .build/ && swift build -c release`
4. Deploy and test with Qwen3.6 (see Test section above)
5. **If tests pass:** Upstream fixed it. Remove the patches from this file and commit.
6. **If tests fail:** Re-apply the patches. Update this file with any changes.

---

## Notes

- These patches are **workarounds**, not proper upstream fixes. The correct fix would be for `mlx-swift-lm`'s `Load.swift` to natively detect pre-quantized weights and skip the `quantize()` step.
- If you submit a PR upstream, reference this file and remove the patch once it's merged.
- The `vendors/` directory contains complete git repos with `.git/`, so you can fetch upstream, diff, and cherry-pick normally.
