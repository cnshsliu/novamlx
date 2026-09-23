import Foundation
import MLX
import MLXFast
import MLXNN

final class Qwen3SpeechDecoder: Module {
    let totalUpsample: Int
    @ModuleInfo(key: "pre_transformer") var transformer: DecoderTransformer
    @ModuleInfo(key: "quantizer") var quantizer: SplitRVQ
    @ModuleInfo(key: "pre_conv") var preConv: CausalConv
    @ModuleInfo(key: "upsample") var upsample: [[Module]]
    @ModuleInfo(key: "decoder") var stages: [Module]

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig) {
        let rates = config.upsampleRates + config.upsamplingRatios
        totalUpsample = rates.reduce(1, *)
        self._transformer.wrappedValue = DecoderTransformer(config)
        self._quantizer.wrappedValue = SplitRVQ(config)
        self._preConv.wrappedValue = CausalConv(config.codebookDim, config.latent, kernel: 3)
        let latent = config.latent
        self._upsample.wrappedValue = config.upsamplingRatios.map { factor in
            [
                CausalTranspose(latent, latent, kernel: factor, stride: factor),
                ConvNeXtBlock(latent),
            ]
        }
        let outDim = config.decoderDim / Int(pow(2.0, Double(config.upsampleRates.count)))
        self._stages.wrappedValue = [
            DecoderInitialConv(config.latent, config.decoderDim, kernel: 7),
            VocoderBlock(config, 0),
            VocoderBlock(config, 1),
            VocoderBlock(config, 2),
            VocoderBlock(config, 3),
            SnakeBeta(outDim),
            DecoderOutputConv(outDim, kernel: 7),
        ]
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var hidden = quantizer.decode(codes).transposed(0, 2, 1)
        hidden = preConv(hidden)
        hidden = transformer(hidden)
        for stage in upsample {
            hidden = (stage[0] as! CausalTranspose).callAsFunction(hidden)
            hidden = (stage[1] as! ConvNeXtBlock).callAsFunction(hidden)
        }
        var wav = hidden
        wav = (stages[0] as! DecoderInitialConv).callAsFunction(wav)
        wav = (stages[1] as! VocoderBlock).callAsFunction(wav)
        wav = (stages[2] as! VocoderBlock).callAsFunction(wav)
        wav = (stages[3] as! VocoderBlock).callAsFunction(wav)
        wav = (stages[4] as! VocoderBlock).callAsFunction(wav)
        wav = (stages[5] as! SnakeBeta).callAsFunction(wav)
        wav = (stages[6] as! DecoderOutputConv).callAsFunction(wav)
        wav = clip(wav.transposed(0, 2, 1), min: MLXArray(-1), max: MLXArray(1))
        return wav
    }

    func chunked(_ codes: MLXArray, chunk: Int = 300, context: Int = 25) -> MLXArray {
        var parts: [MLXArray] = []
        var start = 0
        let time = codes.dim(-1)
        while start < time {
            let end = min(start + chunk, time)
            let ctx = start - context > 0 ? context : start
            let slice = codes[0..., 0..., (start - ctx)..<end]
            let wav = decode(slice)
            parts.append(wav[0..., 0..., (ctx * totalUpsample)...])
            start = end
        }
        return MLX.concatenated(parts, axis: -1)
    }
}

final class UpsampleStages: Module {
    @ModuleInfo(key: "0") var s0: UpsamplePair
    @ModuleInfo(key: "1") var s1: UpsamplePair

    init(_ channels: Int, _ ratios: [Int]) {
        self._s0.wrappedValue = UpsamplePair(channels, ratios[0])
        self._s1.wrappedValue = UpsamplePair(channels, ratios[1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        s1.block(s1.conv(s0.block(s0.conv(x))))
    }
}

final class UpsamplePair: Module {
    @ModuleInfo(key: "0") var conv: CausalTranspose
    @ModuleInfo(key: "1") var block: ConvNeXtBlock

    init(_ channels: Int, _ factor: Int) {
        self._conv.wrappedValue = CausalTranspose(channels, channels, kernel: factor, stride: factor)
        self._block.wrappedValue = ConvNeXtBlock(channels)
    }
}

final class DecoderStages: Module {
    @ModuleInfo(key: "0") var initial: DecoderInitialConv
    @ModuleInfo(key: "1") var b0: VocoderBlock
    @ModuleInfo(key: "2") var b1: VocoderBlock
    @ModuleInfo(key: "3") var b2: VocoderBlock
    @ModuleInfo(key: "4") var b3: VocoderBlock
    @ModuleInfo(key: "5") var snake: SnakeBeta
    @ModuleInfo(key: "6") var out: DecoderOutputConv

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig) {
        self._initial.wrappedValue = DecoderInitialConv(config.latent, config.decoderDim, kernel: 7)
        self._b0.wrappedValue = VocoderBlock(config, 0)
        self._b1.wrappedValue = VocoderBlock(config, 1)
        self._b2.wrappedValue = VocoderBlock(config, 2)
        self._b3.wrappedValue = VocoderBlock(config, 3)
        let outDim = config.decoderDim / Int(pow(2.0, Double(config.upsampleRates.count)))
        self._snake.wrappedValue = SnakeBeta(outDim)
        self._out.wrappedValue = DecoderOutputConv(outDim, kernel: 7)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        out(snake(b3(b2(b1(b0(initial(x)))))))
    }
}

final class SnakeBeta: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(_ channels: Int) {
        self._alpha.wrappedValue = MLXArray.zeros([channels])
        self._beta.wrappedValue = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha)
        let b = exp(beta)
        return x + (1 / (b + 1e-9)) * sin(x * a).square()
    }
}

final class ConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dw: CausalConv
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pw1: Linear
    @ModuleInfo(key: "pwconv2") var pw2: Linear
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(_ dim: Int) {
        self._dw.wrappedValue = CausalConv(dim, dim, kernel: 7, groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pw1.wrappedValue = Linear(dim, 4 * dim, bias: true)
        self._pw2.wrappedValue = Linear(4 * dim, dim, bias: true)
        self._gamma.wrappedValue = MLXArray.ones([dim]) * 1e-6
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = pw2(gelu(pw1(norm(dw(x)))))
        y = gamma * y
        return x + y
    }
}

final class CausalConv: Module {
    let padding: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(_ inCh: Int, _ outCh: Int, kernel: Int, dilation: Int = 1, groups: Int = 1) {
        let effective = (kernel - 1) * dilation + 1
        padding = effective - 1
        self._conv.wrappedValue = Conv1d(
            inputChannels: inCh, outputChannels: outCh, kernelSize: kernel,
            dilation: dilation, groups: groups, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = padding > 0
            ? MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((padding, 0)), IntOrPair((0, 0))])
            : x
        return conv(padded)
    }
}

final class CausalTranspose: Module {
    let trim: Int
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(_ channels: Int, _ out: Int, kernel: Int, stride: Int) {
        trim = kernel - stride
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: channels, outputChannels: out, kernelSize: kernel, stride: stride, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        if trim > 0 { return y[0..., 0..<max(y.dim(1) - trim, 0), 0...] }
        return y
    }
}

final class DecoderInitialConv: Module {
    let kernel: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(_ inCh: Int, _ outCh: Int, kernel: Int) {
        self.kernel = kernel
        self._conv.wrappedValue = Conv1d(inputChannels: inCh, outputChannels: outCh, kernelSize: kernel, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((kernel - 1, 0)), IntOrPair((0, 0))])
        return conv(padded)
    }
}

final class DecoderOutputConv: Module {
    let kernel: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(_ channels: Int, kernel: Int) {
        self.kernel = kernel
        self._conv.wrappedValue = Conv1d(inputChannels: channels, outputChannels: 1, kernelSize: kernel, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [IntOrPair((0, 0)), IntOrPair((kernel - 1, 0)), IntOrPair((0, 0))])
        return conv(padded)
    }
}

final class VocoderBlock: Module {
    @ModuleInfo(key: "block") var block: [Module]

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig, _ index: Int) {
        let inDim = config.decoderDim / Int(pow(2.0, Double(index)))
        let outDim = config.decoderDim / Int(pow(2.0, Double(index + 1)))
        self._block.wrappedValue = [
            SnakeBeta(inDim),
            DecoderUpsample(inDim, outDim, config.upsampleRates[index]),
            ResidualUnit(outDim, dilation: 1),
            ResidualUnit(outDim, dilation: 3),
            ResidualUnit(outDim, dilation: 9),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = (block[0] as! SnakeBeta).callAsFunction(x)
        y = (block[1] as! DecoderUpsample).callAsFunction(y)
        y = (block[2] as! ResidualUnit).callAsFunction(y)
        y = (block[3] as! ResidualUnit).callAsFunction(y)
        y = (block[4] as! ResidualUnit).callAsFunction(y)
        return y
    }
}

final class VocoderSteps: Module {
    @ModuleInfo(key: "0") var snake: SnakeBeta
    @ModuleInfo(key: "1") var up: DecoderUpsample
    @ModuleInfo(key: "2") var r1: ResidualUnit
    @ModuleInfo(key: "3") var r3: ResidualUnit
    @ModuleInfo(key: "4") var r9: ResidualUnit

    init(_ inDim: Int, _ outDim: Int, _ rate: Int) {
        self._snake.wrappedValue = SnakeBeta(inDim)
        self._up.wrappedValue = DecoderUpsample(inDim, outDim, rate)
        self._r1.wrappedValue = ResidualUnit(outDim, dilation: 1)
        self._r3.wrappedValue = ResidualUnit(outDim, dilation: 3)
        self._r9.wrappedValue = ResidualUnit(outDim, dilation: 9)
    }
}

final class DecoderUpsample: Module {
    let trim: Int
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(_ inDim: Int, _ outDim: Int, _ rate: Int) {
        let kernel = 2 * rate
        trim = kernel - rate
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inDim, outputChannels: outDim, kernelSize: kernel, stride: rate, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        if trim > 0 { return y[0..., 0..<max(y.dim(1) - trim, 0), 0...] }
        return y
    }
}

final class ResidualUnit: Module {
    @ModuleInfo(key: "act1") var act1: SnakeBeta
    @ModuleInfo(key: "conv1") var conv1: CausalConv
    @ModuleInfo(key: "act2") var act2: SnakeBeta
    @ModuleInfo(key: "conv2") var conv2: CausalConv

    init(_ dim: Int, dilation: Int) {
        self._act1.wrappedValue = SnakeBeta(dim)
        self._conv1.wrappedValue = CausalConv(dim, dim, kernel: 7, dilation: dilation)
        self._act2.wrappedValue = SnakeBeta(dim)
        self._conv2.wrappedValue = CausalConv(dim, dim, kernel: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(act2(conv1(act1(x))))
    }
}

final class DecoderTransformer: Module {
    let headDim: Int
    let ropeTheta: Float
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "norm") var norm: DecoderRMS

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig) {
        headDim = config.headDim
        ropeTheta = config.ropeTheta
        self._inputProj.wrappedValue = Linear(config.latent, config.hidden, bias: true)
        self._outputProj.wrappedValue = Linear(config.hidden, config.latent, bias: true)
        self._layers.wrappedValue = (0..<config.layers).map { _ in DecoderLayer(config) }
        self._norm.wrappedValue = DecoderRMS(config.hidden, config.rms)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputProj(x)
        let length = h.dim(1)
        let pos = MLXArray(0..<Int32(length)).reshaped([1, length])
        let position = MLX.broadcast(pos, to: [h.dim(0), length])
        let (cos, sin) = ropePair(position, dim: headDim, base: ropeTheta)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = length > 1 ? .causal : .none
        for layer in layers {
            h = layer(h, cos: cos, sin: sin, mask: mask)
        }
        return outputProj(norm(h))
    }
}

final class DecoderLayer: Module {
    let scale: Float
    @ModuleInfo(key: "input_layernorm") var inNorm: DecoderRMS
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: DecoderRMS
    @ModuleInfo(key: "self_attn") var attn: DecoderAttention
    @ModuleInfo(key: "mlp") var mlp: TalkerMLP
    @ModuleInfo(key: "self_attn_layer_scale") var attnScale: LayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpScale: LayerScale

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig) {
        scale = 1 / Foundation.sqrt(Float(config.headDim))
        self._inNorm.wrappedValue = DecoderRMS(config.hidden, config.rms)
        self._postNorm.wrappedValue = DecoderRMS(config.hidden, config.rms)
        self._attn.wrappedValue = DecoderAttention(config)
        self._mlp.wrappedValue = TalkerMLP(config.hidden, config.intermediate)
        self._attnScale.wrappedValue = LayerScale(config.hidden, config.layerScale)
        self._mlpScale.wrappedValue = LayerScale(config.hidden, config.layerScale)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        var y = x + attnScale(attn(inNorm(x), cos: cos, sin: sin, scale: scale, mask: mask))
        y = y + mlpScale(mlp(postNorm(y)))
        return y
    }
}

final class DecoderAttention: Module {
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig) {
        heads = config.heads
        headDim = config.headDim
        self._qProj.wrappedValue = Linear(config.hidden, config.heads * config.headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hidden, config.kvHeads * config.headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hidden, config.kvHeads * config.headDim, bias: false)
        self._oProj.wrappedValue = Linear(config.heads * config.headDim, config.hidden, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let q = applyRoPE(qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3), cos: cos, sin: sin)
        let k = applyRoPE(kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3), cos: cos, sin: sin)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        )
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

final class DecoderRMS: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ size: Int, _ eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([size])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let normed = xf * rsqrt(xf.square().mean(axis: -1, keepDims: true) + eps)
        return (weight * normed).asType(x.dtype)
    }
}

final class LayerScale: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ channels: Int, _ initial: Float) {
        self._scale.wrappedValue = MLXArray.ones([channels]) * initial
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { scale * x }
}

final class SplitRVQ: Module {
    let semantic: Int
    @ModuleInfo(key: "rvq_first") var first: RVQ
    @ModuleInfo(key: "rvq_rest") var rest: RVQ

    init(_ config: Qwen3TTSConfig.Qwen3DecoderConfig) {
        semantic = config.semanticQuantizers
        let dim = config.codebookDim / 2
        self._first.wrappedValue = RVQ(dim: dim, bins: config.codebookSize, count: semantic, io: config.codebookDim)
        self._rest.wrappedValue = RVQ(
            dim: dim, bins: config.codebookSize,
            count: config.quantizers - semantic, io: config.codebookDim
        )
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var y = first.decode(codes[0..., 0..<semantic, 0...])
        if codes.dim(1) > semantic {
            y = y + rest.decode(codes[0..., semantic..., 0...])
        }
        return y
    }
}

final class RVQ: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Conv1d
    @ModuleInfo(key: "output_proj") var outputProj: Conv1d
    @ModuleInfo(key: "vq") var vq: RVQLayers

    init(dim: Int, bins: Int, count: Int, io: Int) {
        self._inputProj.wrappedValue = Conv1d(inputChannels: io, outputChannels: dim, kernelSize: 1, bias: false)
        self._outputProj.wrappedValue = Conv1d(inputChannels: dim, outputChannels: io, kernelSize: 1, bias: false)
        self._vq.wrappedValue = RVQLayers(dim: dim, bins: bins, count: count)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        var sum = MLXArray.zeros([codes.dim(0), vq.dim, codes.dim(2)])
        for i in 0..<vq.count {
            sum = sum + vq.item(i).decode(codes[0..., i, 0...])
        }
        let nlc = outputProj(sum.transposed(0, 2, 1))
        return nlc.transposed(0, 2, 1)
    }
}

final class RVQLayers: Module {
    let dim: Int
    let count: Int
    @ModuleInfo(key: "layers") var layers: [VQLayer]

    init(dim: Int, bins: Int, count: Int) {
        self.dim = dim
        self.count = count
        self._layers.wrappedValue = (0..<count).map { _ in VQLayer(dim: dim, bins: bins) }
    }

    func item(_ index: Int) -> VQLayer { layers[index] }
}

final class CodebookList: Module {
    private var stored: [VQLayer] = []
    @ModuleInfo(key: "0") var l0: VQLayer
    @ModuleInfo(key: "1") var l1: VQLayer?
    @ModuleInfo(key: "2") var l2: VQLayer?
    @ModuleInfo(key: "3") var l3: VQLayer?
    @ModuleInfo(key: "4") var l4: VQLayer?
    @ModuleInfo(key: "5") var l5: VQLayer?
    @ModuleInfo(key: "6") var l6: VQLayer?
    @ModuleInfo(key: "7") var l7: VQLayer?
    @ModuleInfo(key: "8") var l8: VQLayer?
    @ModuleInfo(key: "9") var l9: VQLayer?
    @ModuleInfo(key: "10") var l10: VQLayer?
    @ModuleInfo(key: "11") var l11: VQLayer?
    @ModuleInfo(key: "12") var l12: VQLayer?
    @ModuleInfo(key: "13") var l13: VQLayer?
    @ModuleInfo(key: "14") var l14: VQLayer?

    init(dim: Int, bins: Int, count: Int) {
        let made = (0..<count).map { _ in VQLayer(dim: dim, bins: bins) }
        self._l0.wrappedValue = made[0]
        if count > 1 { self._l1.wrappedValue = made[1] }
        if count > 2 { self._l2.wrappedValue = made[2] }
        if count > 3 { self._l3.wrappedValue = made[3] }
        if count > 4 { self._l4.wrappedValue = made[4] }
        if count > 5 { self._l5.wrappedValue = made[5] }
        if count > 6 { self._l6.wrappedValue = made[6] }
        if count > 7 { self._l7.wrappedValue = made[7] }
        if count > 8 { self._l8.wrappedValue = made[8] }
        if count > 9 { self._l9.wrappedValue = made[9] }
        if count > 10 { self._l10.wrappedValue = made[10] }
        if count > 11 { self._l11.wrappedValue = made[11] }
        if count > 12 { self._l12.wrappedValue = made[12] }
        if count > 13 { self._l13.wrappedValue = made[13] }
        if count > 14 { self._l14.wrappedValue = made[14] }
        stored = made
    }

    func item(_ index: Int) -> VQLayer { stored[index] }
}

final class VQLayer: Module {
    let dim: Int
    @ModuleInfo(key: "codebook") var codebook: Codebook

    init(dim: Int, bins: Int) {
        self.dim = dim
        self._codebook.wrappedValue = Codebook(dim: dim, bins: bins)
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        codebook.embed(codes).transposed(0, 2, 1)
    }
}

final class Codebook: Module {
    let dim: Int
    @ModuleInfo(key: "embed") var embed: Embedding

    init(dim: Int, bins: Int) {
        self.dim = dim
        self._embed.wrappedValue = Embedding(embeddingCount: bins, dimensions: dim)
    }
}
