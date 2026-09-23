import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

final class Qwen3TTSTalker: Module {
    let config: Qwen3TTSConfig.Qwen3TalkerConfig
    @ModuleInfo(key: "model") var model: TalkerBody
    @ModuleInfo(key: "text_projection") var textProjection: ResizeMLP
    @ModuleInfo(key: "codec_head") var codecHead: Linear
    @ModuleInfo(key: "code_predictor") var codePredictor: CodePredictor

    init(_ config: Qwen3TTSConfig.Qwen3TalkerConfig) {
        self.config = config
        self._model.wrappedValue = TalkerBody(config)
        self._textProjection.wrappedValue = ResizeMLP(config.textHidden, config.textHidden, config.hidden)
        self._codecHead.wrappedValue = Linear(config.hidden, config.vocab, bias: false)
        self._codePredictor.wrappedValue = CodePredictor(config)
    }

    func embedText(_ ids: MLXArray) -> MLXArray {
        textProjection(model.textEmbedding(ids))
    }

    func embedCodec(_ ids: MLXArray) -> MLXArray {
        model.codecEmbedding(ids)
    }

    func callAsFunction(_ embeds: MLXArray, cache: [KVCache]?) -> (MLXArray, MLXArray) {
        let hidden = model(embeds, cache: cache)
        return (codecHead(hidden), hidden)
    }
}

final class ResizeMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(_ input: Int, _ mid: Int, _ output: Int) {
        self._fc1.wrappedValue = Linear(input, mid, bias: true)
        self._fc2.wrappedValue = Linear(mid, output, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(silu(fc1(x)))
    }
}

final class TalkerBody: Module {
    let config: Qwen3TTSConfig.Qwen3TalkerConfig
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding
    @ModuleInfo(key: "text_embedding") var textEmbedding: Embedding
    @ModuleInfo(key: "layers") var layers: [TalkerLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Qwen3TTSConfig.Qwen3TalkerConfig) {
        self.config = config
        self._codecEmbedding.wrappedValue = Embedding(embeddingCount: config.vocab, dimensions: config.hidden)
        self._textEmbedding.wrappedValue = Embedding(embeddingCount: config.textVocab, dimensions: config.textHidden)
        self._layers.wrappedValue = (0..<config.layers).map { _ in TalkerLayer(config, mrope: true) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rms)
    }

    func callAsFunction(_ embeds: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = embeds.dim(0)
        let length = embeds.dim(1)
        let offset = cache?.first?.offset ?? 0
        let pos = MLXArray(Int32(offset)..<Int32(offset + length)).reshaped([1, length])
        let position = MLX.broadcast(pos, to: [batch, length])
        let (cos, sin) = mrope(position, dim: config.headDim, base: config.ropeTheta, sections: config.mrope)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = length > 1 ? .causal : .none
        var x = embeds
        for (i, layer) in layers.enumerated() {
            x = layer(x, cos: cos, sin: sin, mask: mask, cache: cache?[i])
        }
        return norm(x)
    }
}

final class TalkerLayer: Module {
    let mrope: Bool
    let headDim: Int
    let heads: Int
    let kvHeads: Int
    let scale: Float
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attn: TalkerAttention
    @ModuleInfo(key: "mlp") var mlp: TalkerMLP

    init(_ config: Qwen3TTSConfig.Qwen3TalkerConfig, mrope: Bool) {
        self.mrope = mrope
        headDim = config.headDim
        heads = config.heads
        kvHeads = config.kvHeads
        scale = 1 / Foundation.sqrt(Float(config.headDim))
        self._inputNorm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rms)
        self._postNorm.wrappedValue = RMSNorm(dimensions: config.hidden, eps: config.rms)
        self._attn.wrappedValue = TalkerAttention(
            hidden: config.hidden, heads: config.heads, kvHeads: config.kvHeads,
            headDim: config.headDim, eps: config.rms
        )
        self._mlp.wrappedValue = TalkerMLP(config.hidden, config.intermediate)
    }

    init(predictor: Qwen3TTSConfig.Predictor) {
        self.mrope = false
        headDim = predictor.headDim
        heads = predictor.heads
        kvHeads = predictor.kvHeads
        scale = 1 / Foundation.sqrt(Float(predictor.headDim))
        self._inputNorm.wrappedValue = RMSNorm(dimensions: predictor.hidden, eps: predictor.rms)
        self._postNorm.wrappedValue = RMSNorm(dimensions: predictor.hidden, eps: predictor.rms)
        self._attn.wrappedValue = TalkerAttention(
            hidden: predictor.hidden, heads: predictor.heads, kvHeads: predictor.kvHeads,
            headDim: predictor.headDim, eps: predictor.rms
        )
        self._mlp.wrappedValue = TalkerMLP(predictor.hidden, predictor.intermediate)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var y = x + attn(inputNorm(x), cos: cos, sin: sin, scale: scale, mask: mask, cache: cache)
        y = y + mlp(postNorm(y))
        return y
    }
}

final class TalkerAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float) {
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self._qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        var q = qNorm(qProj(x).reshaped(b, t, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, t, kvHeads, headDim)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = applyRoPE(q, cos: cos, sin: sin)
        k = applyRoPE(k, cos: cos, sin: sin)
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

final class TalkerMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ hidden: Int, _ mid: Int) {
        self._gate.wrappedValue = Linear(hidden, mid, bias: false)
        self._up.wrappedValue = Linear(hidden, mid, bias: false)
        self._down.wrappedValue = Linear(mid, hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class CodePredictor: Module {
    let config: Qwen3TTSConfig.Predictor
    let talkerHidden: Int
    @ModuleInfo(key: "small_to_mtp_projection") var projection: Linear
    @ModuleInfo(key: "model") var model: CodePredictorBody
    @ModuleInfo(key: "lm_head") var lmHead: [Linear]

    init(_ talker: Qwen3TTSConfig.Qwen3TalkerConfig) {
        let pred = talker.predictor
        config = pred
        talkerHidden = talker.hidden
        let predHidden = pred.hidden
        let predVocab = pred.vocab
        let headCount = pred.codeGroups - 1
        self._projection.wrappedValue = Linear(talker.hidden, predHidden, bias: true)
        self._model.wrappedValue = CodePredictorBody(talker)
        self._lmHead.wrappedValue = (0..<headCount).map { _ in
            Linear(predHidden, predVocab, bias: false)
        }
    }

    var codecEmbedding: [Embedding] { model.codecEmbedding }

    func callAsFunction(_ embeds: MLXArray, cache: [KVCache?], step: Int) -> MLXArray {
        let hidden = model(projection(embeds), cache: cache)
        return lmHead[step](hidden)
    }
}

final class CodePredictorBody: Module {
    let config: Qwen3TTSConfig.Predictor
    let ropeTheta: Float
    @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]
    @ModuleInfo(key: "layers") var layers: [TalkerLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ talker: Qwen3TTSConfig.Qwen3TalkerConfig) {
        let pred = talker.predictor
        config = pred
        ropeTheta = pred.ropeTheta
        let groups = pred.codeGroups - 1
        let vocab = pred.vocab
        let embedDim = talker.hidden
        self._codecEmbedding.wrappedValue = (0..<groups).map { _ in
            Embedding(embeddingCount: vocab, dimensions: embedDim)
        }
        self._layers.wrappedValue = (0..<pred.layers).map { _ in TalkerLayer(predictor: pred) }
        self._norm.wrappedValue = RMSNorm(dimensions: pred.hidden, eps: pred.rms)
    }

    func callAsFunction(_ embeds: MLXArray, cache: [KVCache?]) -> MLXArray {
        let batch = embeds.dim(0)
        let length = embeds.dim(1)
        let offset = cache.first??.offset ?? 0
        let pos = MLXArray(Int32(offset)..<Int32(offset + length)).reshaped([1, length])
        let position = MLX.broadcast(pos, to: [batch, length])
        let (cos, sin) = ropePair(position, dim: config.headDim, base: ropeTheta)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = length > 1 ? .causal : .none
        var x = embeds
        for (i, layer) in layers.enumerated() {
            x = layer(x, cos: cos, sin: sin, mask: mask, cache: cache[i])
        }
        return norm(x)
    }
}

func applyRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let c = cos.expandedDimensions(axis: 1)
    let s = sin.expandedDimensions(axis: 1)
    return x * c + rotateHalf(x) * s
}

func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., 0..<half]
    let x2 = x[0..., 0..., 0..., half...]
    return MLX.concatenated([-x2, x1], axis: -1)
}

func ropePair(_ position: MLXArray, dim: Int, base: Float) -> (MLXArray, MLXArray) {
    let half = dim / 2
    let idx = MLXArray(stride(from: Float(0), to: Float(dim), by: 2))
    let inv = 1 / MLX.pow(MLXArray(base), idx / Float(dim))
    let pos = position.asType(.float32).expandedDimensions(axis: -1)
    let freqs = pos * inv.reshaped([1, 1, half])
    let emb = MLX.concatenated([freqs, freqs], axis: -1)
    return (MLX.cos(emb), MLX.sin(emb))
}

func mrope(_ position: MLXArray, dim: Int, base: Float, sections: [Int]) -> (MLXArray, MLXArray) {
    // Audio uses the same index on all three MRoPE axes, so the interleaved
    // mix equals ordinary RoPE. `sections` is the checkpoint layout.
    precondition(sections.count == 3)
    return ropePair(position, dim: dim, base: base)
}
