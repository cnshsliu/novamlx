import MLX
import MLXNN

/// ECAPA-TDNN speaker encoder. Mel input is [batch, time, 128].
final class Qwen3TTSSpeakerEncoder: Module {
    @ModuleInfo(key: "blocks") var blocks: [Module]
    @ModuleInfo(key: "mfa") var mfa: TDNNBlock
    @ModuleInfo(key: "asp") var asp: AttentivePool
    @ModuleInfo(key: "fc") var fc: Conv1d

    override init() {
        let channels = [512, 512, 512, 512, 1536]
        let kernels = [5, 3, 3, 3, 1]
        let dilations = [1, 2, 3, 4, 1]
        var items: [Module] = [TDNNBlock(128, channels[0], kernels[0], dilations[0])]
        items.append(contentsOf: (1..<channels.count - 1).map { i in
            SERes2Block(channels[i - 1], channels[i], kernel: kernels[i], dilation: dilations[i])
        })
        self._blocks.wrappedValue = items
        self._mfa.wrappedValue = TDNNBlock(channels[4], channels[4], kernels[4], dilations[4])
        self._asp.wrappedValue = AttentivePool(channels: channels[4])
        self._fc.wrappedValue = Conv1d(
            inputChannels: channels[4] * 2, outputChannels: 2048, kernelSize: 1, bias: true
        )
    }

    func embed(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed(0, 2, 1)
        var states: [MLXArray] = []
        x = (blocks[0] as! TDNNBlock).callAsFunction(x)
        states.append(x)
        for block in blocks.dropFirst() {
            x = (block as! SERes2Block).callAsFunction(x)
            states.append(x)
        }
        x = MLX.concatenated(Array(states.dropFirst()), axis: 1)
        x = mfa(x)
        x = asp(x)
        x = fc(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        return x.squeezed(axis: -1)
    }
}

final class TDNNBlock: Module {
    let pad: Int
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(_ inCh: Int, _ outCh: Int, _ kernel: Int, _ dilation: Int) {
        pad = (kernel - 1) * dilation / 2
        self._conv.wrappedValue = Conv1d(
            inputChannels: inCh, outputChannels: outCh, kernelSize: kernel,
            dilation: dilation, bias: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.transposed(0, 2, 1)
        y = reflectPadTime(y, pad)
        return relu(conv(y).transposed(0, 2, 1))
    }
}

final class SERes2Block: Module {
    @ModuleInfo(key: "tdnn1") var tdnn1: TDNNBlock
    @ModuleInfo(key: "res2net_block") var res2: Res2NetBlock
    @ModuleInfo(key: "tdnn2") var tdnn2: TDNNBlock
    @ModuleInfo(key: "se_block") var se: SEBlock

    init(_ inCh: Int, _ outCh: Int, kernel: Int, dilation: Int) {
        self._tdnn1.wrappedValue = TDNNBlock(inCh, outCh, 1, 1)
        self._res2.wrappedValue = Res2NetBlock(outCh, outCh, kernel: kernel, dilation: dilation)
        self._tdnn2.wrappedValue = TDNNBlock(outCh, outCh, 1, 1)
        self._se.wrappedValue = SEBlock(outCh, 128, outCh)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        se(tdnn2(res2(tdnn1(x)))) + x
    }
}

final class Res2NetBlock: Module {
    let scale = 8
    @ModuleInfo(key: "blocks") var blocks: [TDNNBlock]

    init(_ inCh: Int, _ outCh: Int, kernel: Int, dilation: Int) {
        let inPart = inCh / scale
        let hidden = outCh / scale
        self._blocks.wrappedValue = (0..<scale - 1).map { _ in TDNNBlock(inPart, hidden, kernel, dilation) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let chunks = x.split(parts: scale, axis: 1)
        var outputs: [MLXArray] = []
        var prev: MLXArray?
        for (i, chunk) in chunks.enumerated() {
            let part: MLXArray
            if i == 0 {
                part = chunk
            } else if i == 1 {
                part = blocks[i - 1](chunk)
            } else {
                part = blocks[i - 1](chunk + prev!)
            }
            prev = part
            outputs.append(part)
        }
        return MLX.concatenated(outputs, axis: 1)
    }
}

final class SEBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(_ inCh: Int, _ se: Int, _ outCh: Int) {
        self._conv1.wrappedValue = Conv1d(inputChannels: inCh, outputChannels: se, kernelSize: 1)
        self._conv2.wrappedValue = Conv1d(inputChannels: se, outputChannels: outCh, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: 2, keepDims: true).transposed(0, 2, 1)
        let scale = sigmoid(conv2(relu(conv1(mean)))).transposed(0, 2, 1)
        return x * scale
    }
}

final class AttentivePool: Module {
    @ModuleInfo(key: "tdnn") var tdnn: TDNNBlock
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int) {
        self._tdnn.wrappedValue = TDNNBlock(channels * 3, 128, 1, 1)
        self._conv.wrappedValue = Conv1d(inputChannels: 128, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axis: 2, keepDims: true)
        let std = sqrt(MLX.variance(x, axis: 2, keepDims: true) + 1e-12)
        let length = x.dim(2)
        let attentionIn = MLX.concatenated([
            x,
            MLX.broadcast(mean, to: [x.dim(0), x.dim(1), length]),
            MLX.broadcast(std, to: [x.dim(0), x.dim(1), length]),
        ], axis: 1)
        var attention = tanh(tdnn(attentionIn)).transposed(0, 2, 1)
        attention = softmax(conv(attention).transposed(0, 2, 1), axis: 2)
        let pooledMean = (attention * x).sum(axis: 2, keepDims: true)
        let varSum = (attention * (x - pooledMean).square()).sum(axis: 2, keepDims: true)
        let pooledStd = sqrt(maximum(varSum, MLXArray(1e-12)))
        return MLX.concatenated([pooledMean, pooledStd], axis: 1)
    }
}

func takeReversed(_ x: MLXArray, axis: Int) -> MLXArray {
    let index = MLXArray((0..<x.dim(axis)).reversed().map(Int32.init))
    return MLX.take(x, index, axis: axis)
}

func reflectPadTime(_ x: MLXArray, _ pad: Int) -> MLXArray {
    if pad <= 0 { return x }
    let left = takeReversed(x[0..., 1..<(pad + 1), 0...], axis: 1)
    let rightStart = x.dim(1) - pad - 1
    let right = takeReversed(x[0..., rightStart..<(x.dim(1) - 1), 0...], axis: 1)
    return MLX.concatenated([left, x, right], axis: 1)
}
