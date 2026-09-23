import Foundation
import MLX
import MLXNN

final class Qwen21ZeroCenterRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xf = x.asType(.float32)
        let rrms = rsqrt(mean(xf * xf, axes: [-1], keepDims: true) + eps)
        let scale = weight.asType(.float32) + 1
        return (xf * rrms * scale).asType(dtype)
    }
}

/// 3-axis RoPE tables. Not a `Module`: the tables are constants, not checkpoint weights.
final class Qwen21RopeTables {
    private let cosTables: [MLXArray]
    private let sinTables: [MLXArray]
    private let axesDim: [Int]

    init(theta: Float = 10_000, axesDim: [Int] = [16, 56, 56]) {
        self.axesDim = axesDim
        let positive = Array(0..<8192)
        var negative = Array(0..<1024)
        negative.reverse()
        let negPositions = negative.map { -$0 - 1 }
        let index = positive + negPositions
        var cosTables = [MLXArray]()
        var sinTables = [MLXArray]()
        for dim in axesDim {
            let (cos, sin) = Self.table(index: index, dim: dim, theta: theta)
            cosTables.append(cos)
            sinTables.append(sin)
        }
        self.cosTables = cosTables
        self.sinTables = sinTables
    }

    func frequencies(textLen: Int, height: Int, width: Int) -> (MLXArray, MLXArray) {
        let axes = Qwen21RopeLayout.axes(textLen: textLen, height: height, width: width)
        let lists = [axes.frame, axes.height, axes.width]
        var cosParts = [MLXArray]()
        var sinParts = [MLXArray]()
        for i in 0..<3 {
            let ids = MLXArray(lists[i].map { Int32(Qwen21RopeLayout.tableIndex($0)) })
            cosParts.append(cosTables[i].take(ids, axis: 0))
            sinParts.append(sinTables[i].take(ids, axis: 0))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }

    private static func table(index: [Int], dim: Int, theta: Float) -> (MLXArray, MLXArray) {
        let positions = MLXArray(index.map { Float($0) })
        let scales = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
        let omega = 1 / pow(MLXArray(theta), scales)
        let freqs = outer(positions, omega)
        eval(freqs)
        return (cos(freqs), sin(freqs))
    }
}

final class Qwen21Attention: Module {
    let numHeads: Int
    let headDim: Int
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    /// Checkpoint key is `to_out.0.weight`: a one-element list, not a named child.
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm

    init(dim: Int = 4096, numHeads: Int = 32, headDim: Int = 128, eps: Float = 1e-6) {
        self.numHeads = numHeads
        self.headDim = headDim
        self._toQ.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._toK.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._toV.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
        self._toOut.wrappedValue = [Linear(numHeads * headDim, dim, bias: false)]
        self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        textLen: Int?
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let seq = hidden.dim(1)
        var query = toQ(hidden).reshaped([batch, seq, numHeads, headDim])
        var key = toK(hidden).reshaped([batch, seq, numHeads, headDim])
        let value = toV(hidden).reshaped([batch, seq, numHeads, headDim])
        query = normQ(query)
        key = normK(key)
        query = Self.applyRope(query, cos: ropeCos, sin: ropeSin)
        key = Self.applyRope(key, cos: ropeCos, sin: ropeSin)
        query = query.transposed(0, 2, 1, 3)
        key = key.transposed(0, 2, 1, 3)
        let valueHeads = value.transposed(0, 2, 1, 3)
        let scale = 1 / sqrt(Float(headDim))
        let attended: MLXArray
        if let textLen {
            let text = MLXFast.scaledDotProductAttention(
                queries: query[0..., 0..., ..<textLen, 0...],
                keys: key[0..., 0..., ..<textLen, 0...],
                values: valueHeads[0..., 0..., ..<textLen, 0...],
                scale: scale,
                mask: .causal
            )
            let target = MLXFast.scaledDotProductAttention(
                queries: query[0..., 0..., textLen..., 0...],
                keys: key,
                values: valueHeads,
                scale: scale,
                mask: .none
            )
            attended = concatenated([text, target], axis: 2)
        } else {
            attended = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: key,
                values: valueHeads,
                scale: scale,
                mask: .none
            )
        }
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, seq, numHeads * headDim])
        return toOut[0](merged)
    }

    private static func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let pairs = x.asType(.float32).reshaped([x.dim(0), x.dim(1), x.dim(2), -1, 2])
        let real = pairs[0..., 0..., 0..., 0..., 0]
        let imag = pairs[0..., 0..., 0..., 0..., 1]
        let freqCos = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let freqSin = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let outReal = real * freqCos - imag * freqSin
        let outImag = real * freqSin + imag * freqCos
        let stacked = stacked([outReal, outImag], axis: -1)
        return stacked.reshaped(x.shape).asType(dtype)
    }
}

final class Qwen21SwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "out") var outProj: Linear
    @ModuleInfo(key: "gate_layer") var gate: Linear

    init(hidden: Int, inner: Int) {
        self._proj.wrappedValue = Linear(hidden, inner, bias: false)
        self._outProj.wrappedValue = Linear(inner, hidden, bias: false)
        self._gate.wrappedValue = Linear(hidden, inner, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outProj(silu(gate(x)) * proj(x))
    }
}

final class Qwen21TransformerBlock: Module {
    let eps: Float
    @ModuleInfo(key: "attn") var attn: Qwen21Attention
    @ModuleInfo(key: "img_mlp") var mlp: Qwen21SwiGLU

    init(dim: Int = 4096, heads: Int = 32, headDim: Int = 128, mlpRatio: Int = 3, eps: Float = 1e-6) {
        self.eps = eps
        self._attn.wrappedValue = Qwen21Attention(dim: dim, numHeads: heads, headDim: headDim, eps: eps)
        self._mlp.wrappedValue = Qwen21SwiGLU(hidden: dim, inner: dim * mlpRatio)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        mod1: MLXArray,
        mod2: MLXArray,
        ropeCos: MLXArray,
        ropeSin: MLXArray,
        textLen: Int?
    ) -> MLXArray {
        let scaleGate1 = mod1.split(parts: 2, axis: -1)
        let scaleGate2 = mod2.split(parts: 2, axis: -1)
        let norm1 = MLXFast.layerNorm(hidden, weight: nil, bias: nil, eps: eps) * (1 + scaleGate1[0])
        var hidden = hidden + tanh(scaleGate1[1]) * attn(norm1, ropeCos: ropeCos, ropeSin: ropeSin, textLen: textLen)
        let norm2 = MLXFast.layerNorm(hidden, weight: nil, bias: nil, eps: eps) * (1 + scaleGate2[0])
        hidden = hidden + tanh(scaleGate2[1]) * mlp(norm2)
        return hidden
    }
}

final class Qwen21TextProjection: Module, UnaryLayer {
    @ModuleInfo(key: "text_norm") var textNorm: Qwen21ZeroCenterRMSNorm
    @ModuleInfo(key: "in_layer") var inLayer: Linear
    @ModuleInfo(key: "out_layer") var outLayer: Linear

    init(dim: Int = 4096, eps: Float = 1e-6) {
        self._textNorm.wrappedValue = Qwen21ZeroCenterRMSNorm(dimensions: dim, eps: eps)
        self._inLayer.wrappedValue = Linear(dim, dim, bias: false)
        self._outLayer.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outLayer(geluApproximate(inLayer(textNorm(x))))
    }
}

final class Qwen21NoParam: Module {
    override init() {
        super.init()
    }
}

/// Checkpoint key `modulation.layers.1.weight`. Slot 0 is SiLU and has no weights.
final class Qwen21Modulation: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(dim: Int) {
        self._layers.wrappedValue = [Qwen21NoParam(), Linear(dim, dim * 4, bias: false)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let linear = layers[1] as! any UnaryLayer
        return linear(silu(x))
    }
}

final class Qwen21TimestepEmbedder: Module, UnaryLayer {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(dim: Int) {
        self._linear1.wrappedValue = Linear(256, dim, bias: false)
        self._linear2.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ projected: MLXArray) -> MLXArray {
        linear2(silu(linear1(projected)))
    }
}

final class Qwen21TimeEmbed: Module, UnaryLayer {
    @ModuleInfo(key: "timestep_embedder") var embedder: Qwen21TimestepEmbedder

    init(dim: Int = 4096) {
        self._embedder.wrappedValue = Qwen21TimestepEmbedder(dim: dim)
        super.init()
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let half = 128
        let steps = MLXArray((0..<half).map { Float($0) })
        let frequencies = exp(-log(MLXArray(Float(10_000))) * steps / Float(half))
        let args = timestep.asType(.float32).expandedDimensions(axis: 1) * 1000 * frequencies
        let embedded = concatenated([cos(args), sin(args)], axis: -1).asType(.bfloat16)
        return embedder(embedded)
    }
}

final class Qwen21AdaNorm: Module {
    let eps: Float
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int = 4096, eps: Float = 1e-6) {
        self.eps = eps
        self._linear.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }

    func scale(for temb: MLXArray) -> MLXArray {
        linear(silu(temb))
    }

    func callAsFunction(_ hidden: MLXArray, scale: MLXArray) -> MLXArray {
        MLXFast.layerNorm(hidden, weight: nil, bias: nil, eps: eps) * (1 + scale)
    }
}

final class Qwen21Transformer: Module {
    let rope = Qwen21RopeTables()
    @ModuleInfo(key: "time_text_embed") var timeEmbed: Qwen21TimeEmbed
    @ModuleInfo(key: "txt_in") var textIn: Qwen21TextProjection
    @ModuleInfo(key: "img_in") var imageIn: Linear
    @ModuleInfo(key: "modulation") var modulation: Qwen21Modulation
    @ModuleInfo(key: "transformer_blocks") var blocks: [Qwen21TransformerBlock]
    @ModuleInfo(key: "norm_out") var normOut: Qwen21AdaNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear
    private var geometry: [GeometryKey: (MLXArray, MLXArray)] = [:]

    init(
        inChannels: Int = 64,
        outChannels: Int = 64,
        layers: Int = 32,
        heads: Int = 32,
        headDim: Int = 128,
        mlpRatio: Int = 3
    ) {
        let dim = heads * headDim
        self._timeEmbed.wrappedValue = Qwen21TimeEmbed(dim: dim)
        self._textIn.wrappedValue = Qwen21TextProjection(dim: dim)
        self._imageIn.wrappedValue = Linear(inChannels, dim, bias: false)
        self._modulation.wrappedValue = Qwen21Modulation(dim: dim)
        self._blocks.wrappedValue = (0..<layers).map { _ in
            Qwen21TransformerBlock(dim: dim, heads: heads, headDim: headDim, mlpRatio: mlpRatio)
        }
        self._normOut.wrappedValue = Qwen21AdaNorm(dim: dim)
        self._projOut.wrappedValue = Linear(dim, outChannels, bias: false)
        super.init()
    }

    func callAsFunction(
        latents: MLXArray,
        encoder: MLXArray,
        sigma: Float,
        latentHeight: Int,
        latentWidth: Int
    ) -> MLXArray {
        let textLen = encoder.dim(1)
        let imageTokens = latents.dim(1)
        let timestep = MLXArray([sigma, Float(0)]).asType(.float32)
        let (ropeCos, ropeSin) = geometry(
            textLen: textLen, height: latentHeight, width: latentWidth
        )
        let temb = timeEmbed(timestep)
        let mods = modulation(temb).split(parts: 2, axis: -1)
        let mod1 = Self.selectRows(mods[0], textLen: textLen, imageTokens: imageTokens)
        let mod2 = Self.selectRows(mods[1], textLen: textLen, imageTokens: imageTokens)
        var hidden = concatenated([textIn(encoder), imageIn(latents)], axis: 1)
        for block in blocks {
            hidden = block(hidden, mod1: mod1, mod2: mod2, ropeCos: ropeCos, ropeSin: ropeSin, textLen: textLen)
            eval(hidden)
        }
        let scale = Self.selectRows(normOut.scale(for: temb), textLen: textLen, imageTokens: imageTokens)
        hidden = projOut(normOut(hidden, scale: scale))
        return hidden[0..., textLen..., 0...]
    }

    private func geometry(textLen: Int, height: Int, width: Int) -> (MLXArray, MLXArray) {
        let key = GeometryKey(textLen: textLen, height: height, width: width)
        if let cached = geometry[key] {
            return cached
        }
        let made = rope.frequencies(textLen: textLen, height: height, width: width)
        geometry[key] = made
        return made
    }

    private static func selectRows(_ params: MLXArray, textLen: Int, imageTokens: Int) -> MLXArray {
        let dim = params.dim(-1)
        let text = broadcast(params[1].reshaped([1, 1, dim]), to: [1, textLen, dim])
        let image = broadcast(params[0].reshaped([1, 1, dim]), to: [1, imageTokens, dim])
        return concatenated([text, image], axis: 1)
    }
}

private struct GeometryKey: Hashable {
    let textLen: Int
    let height: Int
    let width: Int
}

private func outer(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    matmul(a.reshaped([-1, 1]), b.reshaped([1, -1]))
}
