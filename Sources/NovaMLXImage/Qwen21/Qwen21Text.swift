import Foundation
import MLX
import MLXNN
import NovaMLXCore
import Tokenizers

enum Qwen21Prompt {
    static let system = "Comprehend and analyze the provided prompt."
    static let prefix = "<|im_start|>system\n\(system)<|im_end|>\n"

    static func template(_ prompt: String) -> String {
        let body = prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? " " : prompt
        return prefix + "<|im_start|>user\n\(body)<|im_end|>\n<|im_start|>assistant\n"
    }
}

final class Qwen21TextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float) {
        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = headDim
        self._qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, rope: (MLXArray, MLXArray)) -> MLXArray {
        let batch = hidden.dim(0)
        let seq = hidden.dim(1)
        var query = qNorm(qProj(hidden).reshaped([batch, seq, numHeads, headDim]))
        var key = kNorm(kProj(hidden).reshaped([batch, seq, numKVHeads, headDim]))
        let value = vProj(hidden).reshaped([batch, seq, numKVHeads, headDim])
        query = query.transposed(0, 2, 1, 3)
        key = key.transposed(0, 2, 1, 3)
        let valueHeads = value.transposed(0, 2, 1, 3)
        let (cos, sin) = rope
        query = Self.applyRope(query, cos: cos, sin: sin)
        key = Self.applyRope(key, cos: cos, sin: sin)
        let groups = numHeads / numKVHeads
        let keyFull = Self.repeatKV(key, groups: groups)
        let valueFull = Self.repeatKV(valueHeads, groups: groups)
        let scale = 1 / sqrt(Float(headDim))
        let attended = MLXFast.scaledDotProductAttention(
            queries: query.asType(.float32),
            keys: keyFull.asType(.float32),
            values: valueFull.asType(.float32),
            scale: scale,
            mask: .causal
        ).asType(hidden.dtype)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, seq, numHeads * headDim])
        return oProj(merged)
    }

    private static func repeatKV(_ x: MLXArray, groups: Int) -> MLXArray {
        if groups == 1 { return x }
        let batch = x.dim(0)
        let heads = x.dim(1)
        let seq = x.dim(2)
        let dim = x.dim(3)
        let expanded = broadcast(
            x.expandedDimensions(axis: 2),
            to: [batch, heads, groups, seq, dim]
        )
        return expanded.reshaped([batch, heads * groups, seq, dim])
    }

    private static func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let cosB = cos.expandedDimensions(axis: 1)
        let sinB = sin.expandedDimensions(axis: 1)
        return x * cosB + rotateHalf(x) * sinB
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let parts = split(x, indices: [x.dim(-1) / 2], axis: -1)
        return concatenated([-parts[1], parts[0]], axis: -1)
    }
}

final class Qwen21TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hidden: Int, intermediate: Int) {
        self._gate.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._up.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._down.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class Qwen21TextLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: Qwen21TextAttention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen21TextMLP

    init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, intermediate: Int, eps: Float) {
        self._inputNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._attention.wrappedValue = Qwen21TextAttention(
            hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim, eps: eps
        )
        self._postNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._mlp.wrappedValue = Qwen21TextMLP(hidden: hidden, intermediate: intermediate)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray, rope: (MLXArray, MLXArray)) -> MLXArray {
        var x = hidden + attention(inputNorm(hidden), rope: rope)
        x = x + mlp(postNorm(x))
        return x
    }
}

final class Qwen21TextEncoder: Module {
    let headDim: Int
    let ropeTheta: Float
    let mropeSection: [Int]
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen21TextLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(
        vocab: Int = 151_936,
        hidden: Int = 4096,
        layers: Int = 36,
        heads: Int = 32,
        kvHeads: Int = 8,
        intermediate: Int = 12_288,
        headDim: Int = 128,
        ropeTheta: Float = 5_000_000,
        eps: Float = 1e-6,
        mropeSection: [Int] = [24, 20, 20]
    ) {
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.mropeSection = mropeSection
        self._embed.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        self._layers.wrappedValue = (0..<layers).map { _ in
            Qwen21TextLayer(
                hidden: hidden,
                heads: heads,
                kvHeads: kvHeads,
                headDim: headDim,
                intermediate: intermediate,
                eps: eps
            )
        }
        self._norm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        super.init()
    }

    func callAsFunction(_ tokenIDs: MLXArray) -> MLXArray {
        let batch = tokenIDs.dim(0)
        let seq = tokenIDs.dim(1)
        var hidden = embed(tokenIDs)
        let rope = rotary(hidden, batch: batch, seq: seq)
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, rope: rope)
            if index % 4 == 3 {
                eval(hidden)
            }
        }
        return norm(hidden)
    }

    private func rotary(_ hidden: MLXArray, batch: Int, seq: Int) -> (MLXArray, MLXArray) {
        let positions = MLXArray((0..<seq).map { Int32($0) }).reshaped([1, seq])
        let positionIDs = broadcast(positions, to: [3, batch, seq]).asType(.float32)
        let idx = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) / Float(headDim) })
        let inv = 1 / exp(log(MLXArray(ropeTheta)) * idx)
        let invExpanded = broadcast(
            inv.reshaped([1, 1, inv.dim(0), 1]),
            to: [3, batch, inv.dim(0), 1]
        )
        let posExpanded = positionIDs.expandedDimensions(axis: 2)
        var freqs = matmul(invExpanded, posExpanded).transposed(0, 1, 3, 2)
        freqs = interleave(freqs)
        let emb = concatenated([freqs, freqs], axis: -1)
        let dtype = hidden.dtype
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }

    /// Interleaved mRoPE. Text-only positions are shared, so this is standard RoPE,
    /// but the slot mix is still applied so a later edit path can pass distinct axes.
    private func interleave(_ freqs: MLXArray) -> MLXArray {
        var result = freqs[0]
        let width = result.dim(-1)
        for axis in 1..<3 {
            let length = mropeSection[axis] * 3
            var mask = Array(repeating: false, count: width)
            var cursor = axis
            while cursor < length && cursor < width {
                mask[cursor] = true
                cursor += 3
            }
            let selector = MLXArray(mask.map { $0 ? Float(1) : Float(0) })
            result = result * (1 - selector) + freqs[axis] * selector
        }
        return result
    }
}

final class Qwen21PromptEncoder {
    private let tokenizer: any Tokenizer
    private var cache: [String: MLXArray] = [:]
    private var prefixLength: Int?

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    func encode(_ prompt: String, model: Qwen21TextEncoder) throws -> MLXArray {
        let text = Qwen21Prompt.template(prompt)
        if let cached = cache[text] {
            return cached
        }
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        let drop = systemPrefixLength()
        guard ids.count > drop else {
            throw NovaMLXError.inferenceFailed("Qwen-Image 2.1 prompt was shorter than the system prefix")
        }
        let tokens = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
        let hidden = model(tokens)
        eval(hidden)
        let embeds = hidden[0..., drop..., 0...]
        eval(embeds)
        cache[text] = embeds
        return embeds
    }

    private func systemPrefixLength() -> Int {
        if let prefixLength { return prefixLength }
        let count = tokenizer.encode(text: Qwen21Prompt.prefix, addSpecialTokens: false).count
        prefixLength = count
        return count
    }
}
