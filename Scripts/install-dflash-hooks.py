#!/usr/bin/env python3
"""Install DFlash target hooks into mlx-swift-lm.

NovaMLX compiles against upstream ml-explore/mlx-swift-lm (no DFlash types) and
against the local fork (DFlash already on the text model). This script:

- adds `protocol DFlashTarget` when upstream does not have it, so NovaMLX's
  typealias compiles;
- adds `novamlxDFlashVerify` / embedding / lm_head on Qwen3.5 text and VLM.

The text hook delegates when `dflashVerify` already exists. Otherwise it runs
the layer stack and concatenates the requested hidden taps. The VLM hook is
the same idea and does not conform to `DFlashTarget`, so DSpark's existing
cast is unchanged. Idempotent: files that already contain NOVAMLX_DFLASH_HOOK
are left alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "NOVAMLX_DFLASH_HOOK"

PROTOCOL = '''import Foundation
import MLX
import MLXNN

// NOVAMLX_DFLASH_HOOK — protocol only. Methods are installed on the models.
public protocol DFlashTarget: LanguageModel {
    var dflashEmbedding: Embedding { get }
    func dflashEmbed(_ tokens: MLXArray) -> MLXArray
    func dflashLmHead(_ hidden: MLXArray) -> MLXArray
    func dflashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray)
    func dflashRollback(cache: [KVCache], nRejected: Int, specWidth: Int)
}
'''

TEXT_DELEGATE = '''
// NOVAMLX_DFLASH_HOOK
extension Qwen35TextModel {
    public var novamlxDFlashEmbedding: Embedding { dflashEmbedding }

    public func novamlxDFlashLmHead(_ hidden: MLXArray) -> MLXArray {
        dflashLmHead(hidden)
    }

    public func novamlxDFlashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray) {
        dflashVerify(tokens, cache: cache, tapLayerIds: tapLayerIds)
    }
}

extension Qwen35Model {
    public var novamlxDFlashEmbedding: Embedding { dflashEmbedding }

    public func novamlxDFlashLmHead(_ hidden: MLXArray) -> MLXArray {
        dflashLmHead(hidden)
    }

    public func novamlxDFlashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray) {
        dflashVerify(tokens, cache: cache, tapLayerIds: tapLayerIds)
    }
}
'''

TEXT_UPSTREAM = '''
// NOVAMLX_DFLASH_HOOK
extension Qwen35TextModelInner {
    func novamlxDFlashForward(
        _ inputs: MLXArray, cache: [KVCache?]?, tapLayerIds: Set<Int>
    ) -> (normed: MLXArray, fused: MLXArray) {
        var hiddenStates = embedTokens(inputs)
        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }
        let faMask =
            faIdx.map {
                createAttentionMask(h: hiddenStates, cache: cacheArray?[$0])
            } ?? MLXFast.ScaledDotProductAttentionMaskMode.none
        let ssmMask = ssmIdx.flatMap {
            createSSMMask(h: hiddenStates, cache: cacheArray?[$0] as? MambaCache)
        }
        var captured: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask =
                layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask,
                cache: cacheArray?[i], checkpointAfter: nil)
            if tapLayerIds.contains(i) {
                captured.append(hiddenStates)
            }
        }
        let fused: MLXArray
        if captured.isEmpty {
            fused = hiddenStates
        } else if captured.count == 1 {
            fused = captured[0]
        } else {
            fused = concatenated(captured, axis: -1)
        }
        return (norm(hiddenStates), fused)
    }
}

extension Qwen35TextModel {
    public var novamlxDFlashEmbedding: Embedding { model.embedTokens }

    public func novamlxDFlashLmHead(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func novamlxDFlashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray) {
        let ids = tokens.ndim == 1 ? tokens.reshaped(1, -1) : tokens
        let optional = cache.map { $0.map { $0 as KVCache? } }
        let (normed, fused) = model.novamlxDFlashForward(
            ids, cache: optional, tapLayerIds: Set(tapLayerIds))
        return (novamlxDFlashLmHead(normed), fused)
    }
}

extension Qwen35Model {
    public var novamlxDFlashEmbedding: Embedding { languageModel.novamlxDFlashEmbedding }

    public func novamlxDFlashLmHead(_ hidden: MLXArray) -> MLXArray {
        languageModel.novamlxDFlashLmHead(hidden)
    }

    public func novamlxDFlashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray) {
        languageModel.novamlxDFlashVerify(tokens, cache: cache, tapLayerIds: tapLayerIds)
    }
}
'''

VLM_FORK = '''
// NOVAMLX_DFLASH_HOOK
extension Qwen35Language.Model {
    func novamlxDFlashForward(
        _ inputs: MLXArray, cache: [KVCache?]?, tapLayerIds: Set<Int>
    ) -> (normed: MLXArray, fused: MLXArray) {
        var hiddenStates = embedTokens(inputs)
        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }
        let faMaskMode = createAttentionMask(
            h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
        let faMask: MLXArray?
        if case .array(let arrayMask) = faMaskMode {
            faMask = arrayMask
        } else {
            faMask = nil
        }
        let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)
        var captured: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            let layerSSMMask = layer.isLinear ? ssmMask : nil
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: layerSSMMask,
                cache: cacheArray?[index],
                positionIds: nil,
                checkpointAfter: nil)
            if tapLayerIds.contains(index) {
                captured.append(hiddenStates)
            }
        }
        let fused: MLXArray
        if captured.isEmpty {
            fused = hiddenStates
        } else if captured.count == 1 {
            fused = captured[0]
        } else {
            fused = concatenated(captured, axis: -1)
        }
        return (norm(hiddenStates), fused)
    }
}

extension Qwen35 {
    public var novamlxDFlashEmbedding: Embedding { languageModel.model.embedTokens }

    public func novamlxDFlashLmHead(_ hidden: MLXArray) -> MLXArray {
        if let head = languageModel.lmHead {
            return head(hidden)
        }
        return languageModel.model.embedTokens.asLinear(hidden)
    }

    public func novamlxDFlashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray) {
        let ids = tokens.ndim == 1 ? tokens.reshaped(1, -1) : tokens
        let optional = cache.map { $0.map { $0 as KVCache? } }
        let (normed, fused) = languageModel.model.novamlxDFlashForward(
            ids, cache: optional, tapLayerIds: Set(tapLayerIds))
        return (novamlxDFlashLmHead(normed), fused)
    }
}
'''

VLM_UPSTREAM = '''
// NOVAMLX_DFLASH_HOOK
extension Qwen35Language.Model {
    func novamlxDFlashForward(
        _ inputs: MLXArray, cache: [KVCache?]?, tapLayerIds: Set<Int>
    ) -> (normed: MLXArray, fused: MLXArray) {
        var hiddenStates = embedTokens(inputs)
        var cacheArray = cache
        if cacheArray == nil {
            cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        }
        let faMask: MLXArray?
        if let faIdx {
            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
        } else {
            faMask = nil
        }
        let ssmMask = ssmIdx.flatMap {
            createSSMMask(h: hiddenStates, cache: cacheArray?[$0] as? MambaCache)
        }
        var captured: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            let layerSSMMask = layer.isLinear ? ssmMask : nil
            hiddenStates = layer(
                hiddenStates,
                attentionMask: faMask,
                ssmMask: layerSSMMask,
                cache: cacheArray?[index],
                positionIds: nil,
                checkpointAfter: nil)
            if tapLayerIds.contains(index) {
                captured.append(hiddenStates)
            }
        }
        let fused: MLXArray
        if captured.isEmpty {
            fused = hiddenStates
        } else if captured.count == 1 {
            fused = captured[0]
        } else {
            fused = concatenated(captured, axis: -1)
        }
        return (norm(hiddenStates), fused)
    }
}

extension Qwen35 {
    public var novamlxDFlashEmbedding: Embedding { languageModel.model.embedTokens }

    public func novamlxDFlashLmHead(_ hidden: MLXArray) -> MLXArray {
        if let head = languageModel.lmHead {
            return head(hidden)
        }
        return languageModel.model.embedTokens.asLinear(hidden)
    }

    public func novamlxDFlashVerify(
        _ tokens: MLXArray, cache: [KVCache]?, tapLayerIds: [Int]
    ) -> (logits: MLXArray, fused: MLXArray) {
        let ids = tokens.ndim == 1 ? tokens.reshaped(1, -1) : tokens
        let optional = cache.map { $0.map { $0 as KVCache? } }
        let (normed, fused) = languageModel.model.novamlxDFlashForward(
            ids, cache: optional, tapLayerIds: Set(tapLayerIds))
        return (novamlxDFlashLmHead(normed), fused)
    }
}
'''


def append_once(path: Path, body: str) -> None:
    text = path.read_text()
    if MARKER in text:
        print(f"→ {path} already has {MARKER}")
        return
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + body)
    print(f"→ installed DFlash hook in {path}")


def ensure_protocol(root: Path) -> None:
    common = root / "Libraries" / "MLXLMCommon"
    if not common.is_dir():
        return
    for path in common.glob("*.swift"):
        if "protocol DFlashTarget" in path.read_text():
            print(f"→ DFlashTarget already in {path.name}")
            return
    dest = common / "NovamlxDFlashTarget.swift"
    dest.write_text(PROTOCOL)
    print(f"→ wrote {dest}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "mlx-swift-lm")
    if not root.is_dir():
        print(f"mlx-swift-lm not found at {root}, skipping DFlash hooks")
        return 0
    ensure_protocol(root)
    text = root / "Libraries" / "MLXLLM" / "Models" / "Qwen35.swift"
    vision = root / "Libraries" / "MLXVLM" / "Models" / "Qwen35.swift"
    if text.is_file():
        source = text.read_text()
        if "func dflashVerify" in source:
            append_once(text, TEXT_DELEGATE)
        else:
            append_once(text, TEXT_UPSTREAM)
    if vision.is_file():
        source = vision.read_text()
        if "let faIdx: Int?" in source:
            append_once(vision, VLM_UPSTREAM)
        else:
            append_once(vision, VLM_FORK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
