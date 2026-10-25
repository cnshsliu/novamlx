import Foundation
import MLX
import MLXFast
import MLXNN
import NovaMLXUtils

// MARK: - Flint decode megakernel (single compiled call per token)

/// Whole-model compiled single-token decode, mirroring Recursive's `rec`
/// megakernel approach: one `compile`d function covers all 49 layers, so the
/// per-call Swift wrapper overhead (~0.3 ms × 98 calls for the per-layer
/// segmentation, which measured SLOWER than eager) is paid exactly once.
///
/// KV state lives outside the KVCache objects as plain arrays threaded
/// through the compiled graph as arguments/outputs:
///
/// - Sliding layers (35): a right-aligned buffer of exactly `slidingWindow`
///   rows. A single unconditional `concat(buffer[..., 1:, :], k_new)` append
///   is correct in every regime — it drops a padding row while warming up
///   and the oldest real row once full. Validity comes from an additive mask
///   built from the token-count tensor, so every shape stays static.
/// - Full-attention layers (14): the same right-aligned padded scheme with a
///   fixed capacity (`fullCapacity`, default 8192) — bounded KV, the same
///   trade-off `GenerateParameters.maxKVSize` already offers. All shapes
///   static → the graph traces exactly once, no shapeless recompiles.
///
/// Prefill runs the same functional body token-by-token (eager) so the
/// buffers warm up with identical semantics; decode switches to the single
/// compiled call. `FlintSpeedTests` asserts greedy token-for-token parity
/// with the eager KVCache path.
final class FlintMegaDecode: @unchecked Sendable {
    private let model: FlintModelInner
    private let args: FlintConfiguration
    private let invFreq: MLXArray
    private let windowIdx: MLXArray
    /// Capacity of the padded full-attention buffers.
    let fullCapacity: Int
    private var fullIdx: MLXArray
    private var compiled: (([MLXArray]) -> [MLXArray])?

    init(model: FlintModelInner, args: FlintConfiguration, fullCapacity: Int = 8192) {
        self.model = model
        self.args = args
        self.fullCapacity = fullCapacity
        let dim = args.headDim
        let exponent = (0 ..< dim / 2).map { i -> Float in -Float(2 * i) / Float(dim) }
        self.invFreq = pow(MLXArray(Float(args.ropeTheta)), MLXArray(exponent))
        self.windowIdx = MLXArray.arange(0, args.slidingWindow)
        self.fullIdx = MLXArray.arange(0, fullCapacity)
    }

    // MARK: state

    struct State {
        /// Per layer: `[keys, values]`, each `[1, kvHeads, S, headDim]`.
        /// Sliding layers: S = slidingWindow (right-aligned, zero-padded).
        /// Full layers: S = tokens so far (exact, grows).
        var kv: [[MLXArray]]
        /// Absolute token count processed (prompt + decoded).
        var tokens: Int
    }

    func initialState() -> State {
        State(
            kv: model.layers.map { layer in
                let s = layer.useSliding ? args.slidingWindow : fullCapacity
                let shape: [Int] = [1, args.kvHeads, s, args.headDim]
                return [MLXArray.zeros(shape), MLXArray.zeros(shape)]
            }, tokens: 0)
    }

    // MARK: rope

    private func ropeAngles(_ pos: MLXArray) -> MLXArray {
        pos.asType(.float32) * invFreq
    }

    private func applyRoPE(_ x: MLXArray, angles: MLXArray) -> MLXArray {
        // x: [B, H, L, D]; non-interleaved half-split, matching flint.py.
        // d from config (not x.dim) — dim() breaks shapeless compile traces.
        let d = args.headDim
        let x1 = x[.ellipsis, 0 ..< d / 2]
        let x2 = x[.ellipsis, (d / 2)...]
        let cosA = MLX.cos(angles).reshaped(1, 1, 1, d / 2).asType(x.dtype)
        let sinA = MLX.sin(angles).reshaped(1, 1, 1, d / 2).asType(x.dtype)
        return concatenated([x1 * cosA - x2 * sinA, x2 * cosA + x1 * sinA], axis: -1)
    }

    /// Additive mask for a right-aligned padded buffer of `capacity` rows:
    /// row `j` is valid iff it holds one of the `min(tokens, capacity)` real
    /// entries. Covers both sliding (window) and bounded full-attention
    /// (fullCapacity) buffers.
    private func bufferMask(idx: MLXArray, capacity: Int, tokens: MLXArray) -> MLXArray {
        let valid = minimum(tokens.asType(.float32), MLXArray(Float(capacity)))
        return MLX.where(
            idx.reshaped(1, 1, 1, -1) .>= (MLXArray(Float(capacity)) - valid),
            MLXArray(0.0),
            MLXArray(-1.0e9))
    }

    private func slidingMask(tokens: MLXArray) -> MLXArray {
        bufferMask(idx: windowIdx, capacity: args.slidingWindow, tokens: tokens)
    }

    private func fullMask(tokens: MLXArray) -> MLXArray {
        bufferMask(idx: fullIdx, capacity: fullCapacity, tokens: tokens)
    }

    // MARK: layer body (traceable; L may be > 1 for eager prefill)

    private func layerBody(
        _ layer: FlintDecoderLayer, x: MLXArray, pos: MLXArray,
        keys: MLXArray, values: MLXArray, tokens: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let attn = layer.selfAttn
        // B and L are constants (single batch, single token — dim() is not
        // usable inside a shapeless compiled trace).
        let B = 1
        let L = 1
        let normed = layer.inputNorm(x)

        var q = attn.qNorm(attn.qProj(normed).reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var k = attn.kNorm(attn.kProj(normed).reshaped(B, L, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let v = attn.vProj(normed).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if layer.useSliding {
            let angles = ropeAngles(pos)
            q = applyRoPE(q, angles: angles)
            k = applyRoPE(k, angles: angles)
        }

        // Both buffer kinds use the unconditional left-shift append: drops a
        // padding row while warming up, the oldest real row once full.
        // Right-aligned always, every shape static.
        let newKeys = concatenated([keys[.ellipsis, 1..., 0...], k], axis: 2)
        let newValues = concatenated([values[.ellipsis, 1..., 0...], v], axis: 2)

        // L is always 1 here (prefill is token-by-token, decode is single
        // token), so the single-query mask above is complete.
        let mask = (layer.useSliding ? slidingMask(tokens: tokens) : fullMask(tokens: tokens))

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: newKeys, values: newValues,
            scale: pow(Float(args.headDim), -0.5),
            mask: ProcessInfo.processInfo.environment["MEGA_NOMASK"] == "1" ? nil : mask,
            sinks: ProcessInfo.processInfo.environment["MEGA_NOSINK"] == "1" ? nil : attn.softmaxOffset
        )
        let attnOut = attn.oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))

        let h = x + attnOut
        let mlpNormed = layer.postNorm(h)
        if let moe = layer.mlp as? FlintSparseMoeBlock {
            return (h + moe(mlpNormed), newKeys, newValues)
        }
        return (h + (layer.mlp as! FlintMLP)(mlpNormed), newKeys, newValues)
    }

    // MARK: prefill (eager, token-sequential to keep buffer semantics identical)

    /// Feed one token through the functional body without compiling.
    /// Returns the post-norm hidden (logits head applied by caller).
    func prefillToken(tokenId: Int, state: inout State) -> MLXArray {
        var x = model.embedTokens(MLXArray(Int32(tokenId)).reshaped(1, 1))
        let pos = MLXArray(Float(state.tokens))
        let tokens = MLXArray(Float(state.tokens + 1))
        var newKV: [[MLXArray]] = []
        for (i, layer) in model.layers.enumerated() {
            let (nx, k, v) = layerBody(
                layer, x: x, pos: pos,
                keys: state.kv[i][0], values: state.kv[i][1],
                tokens: tokens)
            x = nx
            newKV.append([k, v])
        }
        state.kv = newKV
        state.tokens += 1
        return model.norm(x)
    }

    // MARK: compiled decode

    /// Decode one token through the single compiled graph.
    /// Returns the post-norm hidden (logits head applied by caller).
    func decodeHidden(tokenId: Int, state: inout State) -> MLXArray {
        if compiled == nil { build() }

        var inputs: [MLXArray] = [MLXArray(Int32(tokenId)).reshaped(1, 1)]
        inputs.append(MLXArray(Float(state.tokens)))  // pos
        inputs.append(MLXArray(Float(state.tokens + 1)))  // tokens after this write
        for layer in 0 ..< model.layers.count {
            inputs.append(state.kv[layer][0])
            inputs.append(state.kv[layer][1])
        }

        let outputs = compiled!(inputs)
        var newKV: [[MLXArray]] = []
        for i in 0 ..< model.layers.count {
            newKV.append([outputs[1 + 2 * i], outputs[1 + 2 * i + 1]])
        }
        state.kv = newKV
        state.tokens += 1
        return outputs[0]
    }

    private func build() {
        let layerCount = model.layers.count
        compiled = compile { [self] (argsIn: [MLXArray]) -> [MLXArray] in
            let tokenIds = argsIn[0]
            let pos = argsIn[1]
            let tokens = argsIn[2]
            var x = model.embedTokens(tokenIds)
            var outputs: [MLXArray] = []
            for i in 0 ..< layerCount {
                let (nx, k, v) = layerBody(
                    model.layers[i], x: x, pos: pos,
                    keys: argsIn[3 + 2 * i], values: argsIn[3 + 2 * i + 1],
                    tokens: tokens)
                x = nx
                outputs.append(k)
                outputs.append(v)
            }
            return [model.norm(x)] + outputs
        }
    }
}
