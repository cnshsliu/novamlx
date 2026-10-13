import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// Calls installed on Qwen3.5 text and VLM backbones by `Scripts/install-dflash-hooks.py`.
/// Upstream mlx-swift-lm has no `DFlashTarget`. The drafter still needs per-layer taps.
struct DFlashHooks {
    let embedding: Embedding
    let lmHead: (MLXArray) -> MLXArray
    let verify: (MLXArray, [KVCache]?, [Int]) -> (logits: MLXArray, fused: MLXArray)

    static func make(model: any LanguageModel) -> DFlashHooks? {
        if let text = model as? Qwen35Model {
            return DFlashHooks(
                embedding: text.novamlxDFlashEmbedding,
                lmHead: { text.novamlxDFlashLmHead($0) },
                verify: { tokens, cache, taps in
                    text.novamlxDFlashVerify(tokens, cache: cache, tapLayerIds: taps)
                }
            )
        }
        if let text = model as? Qwen35TextModel {
            return DFlashHooks(
                embedding: text.novamlxDFlashEmbedding,
                lmHead: { text.novamlxDFlashLmHead($0) },
                verify: { tokens, cache, taps in
                    text.novamlxDFlashVerify(tokens, cache: cache, tapLayerIds: taps)
                }
            )
        }
        if let vision = model as? MLXVLM.Qwen35 {
            return DFlashHooks(
                embedding: vision.novamlxDFlashEmbedding,
                lmHead: { vision.novamlxDFlashLmHead($0) },
                verify: { tokens, cache, taps in
                    vision.novamlxDFlashVerify(tokens, cache: cache, tapLayerIds: taps)
                }
            )
        }
        if let target = model as? any DFlashTarget {
            return DFlashHooks(
                embedding: target.dflashEmbedding,
                lmHead: { target.dflashLmHead($0) },
                verify: { tokens, cache, taps in
                    target.dflashVerify(tokens, cache: cache, tapLayerIds: taps)
                }
            )
        }
        return nil
    }
}
