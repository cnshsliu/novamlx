import Foundation
import MLX
import MLXNN

/// Wan-style channel RMSNorm. Checkpoint gamma is reshaped to `(channels,)`.
final class Qwen21ChannelNorm: Module, UnaryLayer {
    let weight: MLXArray
    let scale: Float

    init(channels: Int) {
        self.weight = MLXArray.ones([channels])
        self.scale = sqrt(Float(channels))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xf = x.asType(.float32)
        let norm = sqrt(sum(xf * xf, axes: [1], keepDims: true))
        let normalized = xf / maximum(norm, MLXArray(Float(1e-12)))
        let gamma = weight.asType(.float32).reshaped([1, weight.dim(0), 1, 1])
        return (normalized * scale * gamma).asType(dtype)
    }
}

final class Qwen21CausalConv: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(input: Int, output: Int, kernel: Int, padding: Int) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: input,
            outputChannels: output,
            kernelSize: IntOrPair(kernel),
            stride: 1,
            padding: IntOrPair(padding),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x.transposed(0, 2, 3, 1))
        return y.transposed(0, 3, 1, 2)
    }
}

final class Qwen21Resample: Module, UnaryLayer {
    let mode: String
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int, output: Int, mode: String) {
        self.mode = mode
        if mode == "downsample" {
            self._conv.wrappedValue = Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 3,
                stride: 2,
                padding: 0,
                bias: true
            )
        } else {
            self._conv.wrappedValue = Conv2d(
                inputChannels: channels,
                outputChannels: output,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                bias: true
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.transposed(0, 2, 3, 1)
        if mode == "upsample" {
            y = repeated(y, count: 2, axis: 1)
            y = repeated(y, count: 2, axis: 2)
        } else {
            y = padded(y, widths: [0, [0, 1], [0, 1], 0])
        }
        return conv(y).transposed(0, 3, 1, 2)
    }
}

final class Qwen21ResBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: Qwen21ChannelNorm
    @ModuleInfo(key: "conv1") var conv1: Qwen21CausalConv
    @ModuleInfo(key: "norm2") var norm2: Qwen21ChannelNorm
    @ModuleInfo(key: "conv2") var conv2: Qwen21CausalConv
    @ModuleInfo(key: "conv_shortcut") var shortcut: Qwen21CausalConv?

    init(input: Int, output: Int) {
        self._norm1.wrappedValue = Qwen21ChannelNorm(channels: input)
        self._conv1.wrappedValue = Qwen21CausalConv(input: input, output: output, kernel: 3, padding: 1)
        self._norm2.wrappedValue = Qwen21ChannelNorm(channels: output)
        self._conv2.wrappedValue = Qwen21CausalConv(input: output, output: output, kernel: 3, padding: 1)
        if input == output {
            self._shortcut.wrappedValue = nil
        } else {
            self._shortcut.wrappedValue = Qwen21CausalConv(input: input, output: output, kernel: 1, padding: 0)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = shortcut?(x) ?? x
        var y = conv1(silu(norm1(x)))
        y = conv2(silu(norm2(y)))
        return y + residual
    }
}

final class Qwen21SpatialAttention: Module, UnaryLayer {
    let channels: Int
    @ModuleInfo(key: "norm") var norm: Qwen21ChannelNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(channels: Int) {
        self.channels = channels
        self._norm.wrappedValue = Qwen21ChannelNorm(channels: channels)
        self._toQKV.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels * 3, kernelSize: 1, bias: true
        )
        self._proj.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 1, bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(2)
        let width = x.dim(3)
        let tokens = height * width
        let normed = norm(x).transposed(0, 2, 3, 1)
        let qkv = toQKV(normed).reshaped([batch, tokens, 3, channels])
        let q = qkv[0..., 0..., 0, 0...]
        let k = qkv[0..., 0..., 1, 0...]
        let v = qkv[0..., 0..., 2, 0...]
        let scale = 1 / sqrt(Float(channels))
        let scores = matmul(q, k.transposed(0, 2, 1)) * scale
        let hidden = matmul(softmax(scores, axis: -1), v)
        let projected = proj(hidden.reshaped([batch, height, width, channels]))
        return projected.transposed(0, 3, 1, 2) + x
    }
}

final class Qwen21MidBlock: Module, UnaryLayer {
    @ModuleInfo(key: "resnets") var resnets: [Qwen21ResBlock]
    @ModuleInfo(key: "attentions") var attentions: [Qwen21SpatialAttention]

    init(channels: Int) {
        self._resnets.wrappedValue = [
            Qwen21ResBlock(input: channels, output: channels),
            Qwen21ResBlock(input: channels, output: channels),
        ]
        self._attentions.wrappedValue = [Qwen21SpatialAttention(channels: channels)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = attentions[0](resnets[0](x))
        return resnets[1](y)
    }
}

enum Qwen21AvgDown {
    static func call(_ x: MLXArray, input: Int, output: Int, factorT: Int, factorS: Int) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(2)
        let width = x.dim(3)
        let factor = factorT * factorS * factorS
        let group = input * factor / output
        var y = x.expandedDimensions(axis: 2)
        if factorT == 2 {
            y = padded(y, widths: [0, 0, [1, 0], 0, 0])
        }
        y = y.reshaped([
            batch, input, 1, factorT, height / factorS, factorS, width / factorS, factorS,
        ])
        y = y.transposed(0, 1, 3, 5, 7, 2, 4, 6)
        y = y.reshaped([batch, input * factor, height / factorS, width / factorS])
        y = y.reshaped([batch, output, group, height / factorS, width / factorS])
        return y.mean(axis: 2)
    }
}

enum Qwen21DupUp {
    static func call(_ x: MLXArray, input: Int, output: Int, factorT: Int, factorS: Int = 2) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(2)
        let width = x.dim(3)
        let factor = factorT * factorS * factorS
        let repeats = output * factor / input
        var y = repeated(x, count: repeats, axis: 1)
        y = y.reshaped([batch, output, factorT, factorS, factorS, 1, height, width])
        y = y.transposed(0, 1, 5, 2, 6, 3, 7, 4)
        y = y.reshaped([batch, output, factorT, height * factorS, width * factorS])
        return y[0..., 0..., (factorT - 1)..., 0..., 0...].squeezed(axis: 2)
    }
}

final class Qwen21DownBlock: Module, UnaryLayer {
    let input: Int
    let output: Int
    let factorT: Int
    let factorS: Int
    @ModuleInfo(key: "resnets") var resnets: [Qwen21ResBlock]
    @ModuleInfo(key: "downsampler") var downsampler: Qwen21Resample?

    init(input: Int, output: Int, resnets: Int, temporal: Bool, downsample: Bool) {
        self.input = input
        self.output = output
        self.factorT = temporal ? 2 : 1
        self.factorS = downsample ? 2 : 1
        var blocks = [Qwen21ResBlock]()
        var current = input
        for _ in 0..<resnets {
            blocks.append(Qwen21ResBlock(input: current, output: output))
            current = output
        }
        self._resnets.wrappedValue = blocks
        if downsample {
            self._downsampler.wrappedValue = Qwen21Resample(channels: output, output: output, mode: "downsample")
        } else {
            self._downsampler.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for block in resnets {
            y = block(y)
        }
        if let downsampler {
            y = downsampler(y)
        }
        return y + Qwen21AvgDown.call(x, input: input, output: output, factorT: factorT, factorS: factorS)
    }
}

final class Qwen21UpBlock: Module, UnaryLayer {
    let input: Int
    let output: Int
    let factorT: Int
    let up: Bool
    @ModuleInfo(key: "resnets") var resnets: [Qwen21ResBlock]
    @ModuleInfo(key: "upsampler") var upsampler: Qwen21Resample?

    init(input: Int, output: Int, resnets: Int, temporal: Bool, upsample: Bool) {
        self.input = input
        self.output = output
        self.factorT = temporal ? 2 : 1
        self.up = upsample
        var blocks = [Qwen21ResBlock]()
        var current = input
        for _ in 0..<(resnets + 1) {
            blocks.append(Qwen21ResBlock(input: current, output: output))
            current = output
        }
        self._resnets.wrappedValue = blocks
        if upsample {
            self._upsampler.wrappedValue = Qwen21Resample(channels: output, output: output, mode: "upsample")
        } else {
            self._upsampler.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for block in resnets {
            y = block(y)
        }
        guard up, let upsampler else { return y }
        y = upsampler(y)
        return y + Qwen21DupUp.call(x, input: input, output: output, factorT: factorT)
    }
}

final class Qwen21Encoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") var convIn: Qwen21CausalConv
    @ModuleInfo(key: "down_blocks") var downBlocks: [Qwen21DownBlock]
    @ModuleInfo(key: "mid_block") var mid: Qwen21MidBlock
    @ModuleInfo(key: "norm_out") var normOut: Qwen21ChannelNorm
    @ModuleInfo(key: "conv_out") var convOut: Qwen21CausalConv

    override init() {
        let dims = [96, 96, 192, 384, 768, 768]
        let temporal = [false, true, true, true]
        self._convIn.wrappedValue = Qwen21CausalConv(input: 4, output: dims[0], kernel: 3, padding: 1)
        self._downBlocks.wrappedValue = (0..<5).map { index in
            Qwen21DownBlock(
                input: dims[index],
                output: dims[index + 1],
                resnets: 2,
                temporal: index < temporal.count ? temporal[index] : false,
                downsample: index < 4
            )
        }
        self._mid.wrappedValue = Qwen21MidBlock(channels: dims[dims.count - 1])
        self._normOut.wrappedValue = Qwen21ChannelNorm(channels: dims[dims.count - 1])
        self._convOut.wrappedValue = Qwen21CausalConv(
            input: dims[dims.count - 1], output: 128, kernel: 3, padding: 1
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convIn(x)
        for block in downBlocks {
            y = block(y)
            eval(y)
        }
        y = mid(y)
        return convOut(silu(normOut(y)))
    }
}

final class Qwen21Decoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") var convIn: Qwen21CausalConv
    @ModuleInfo(key: "mid_block") var mid: Qwen21MidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [Qwen21UpBlock]
    @ModuleInfo(key: "norm_out") var normOut: Qwen21ChannelNorm
    @ModuleInfo(key: "conv_out") var convOut: Qwen21CausalConv

    override init() {
        let dims = [1152, 1152, 1152, 576, 288, 144]
        let temporal = [true, true, true, false]
        self._convIn.wrappedValue = Qwen21CausalConv(input: 64, output: dims[0], kernel: 3, padding: 1)
        self._mid.wrappedValue = Qwen21MidBlock(channels: dims[0])
        self._upBlocks.wrappedValue = (0..<5).map { index in
            Qwen21UpBlock(
                input: dims[index],
                output: dims[index + 1],
                resnets: 2,
                temporal: index < temporal.count ? temporal[index] : false,
                upsample: index < 4
            )
        }
        self._normOut.wrappedValue = Qwen21ChannelNorm(channels: dims[dims.count - 1])
        self._convOut.wrappedValue = Qwen21CausalConv(
            input: dims[dims.count - 1], output: 4, kernel: 3, padding: 1
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = mid(convIn(x))
        for block in upBlocks {
            y = block(y)
            eval(y)
        }
        return convOut(silu(normOut(y)))
    }
}

final class Qwen21VAE: Module {
    static let latentChannels = 64
    @ModuleInfo(key: "encoder") var encoder: Qwen21Encoder
    @ModuleInfo(key: "quant_conv") var quantConv: Qwen21CausalConv
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Qwen21CausalConv
    @ModuleInfo(key: "decoder") var decoder: Qwen21Decoder

    override init() {
        self._encoder.wrappedValue = Qwen21Encoder()
        self._quantConv.wrappedValue = Qwen21CausalConv(input: 128, output: 128, kernel: 1, padding: 0)
        self._postQuantConv.wrappedValue = Qwen21CausalConv(input: 64, output: 64, kernel: 1, padding: 0)
        self._decoder.wrappedValue = Qwen21Decoder()
        super.init()
    }

    func encode(_ image: MLXArray, mean: MLXArray, std: MLXArray) -> MLXArray {
        var pixels = image
        if pixels.ndim == 5 {
            pixels = pixels.squeezed(axis: 2)
        }
        if pixels.dim(1) == 3 {
            let alpha = MLXArray.ones([pixels.dim(0), 1, pixels.dim(2), pixels.dim(3)], dtype: pixels.dtype)
            pixels = concatenated([pixels, alpha], axis: 1)
        }
        var latents = quantConv(encoder(pixels))
        latents = latents[0..., ..<Self.latentChannels, 0..., 0...]
        return (latents - mean) / std
    }

    func decode(_ latents: MLXArray, mean: MLXArray, std: MLXArray) -> MLXArray {
        var z = latents
        if z.ndim == 5, z.dim(2) == 1 {
            z = z.squeezed(axis: 2)
        }
        z = z * std + mean
        let decoded = decoder(postQuantConv(z))
        return decoded[0..., ..<3, 0..., 0...]
    }
}
