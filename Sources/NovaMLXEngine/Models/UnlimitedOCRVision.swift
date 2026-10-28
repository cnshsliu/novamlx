import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - SAM

final class UnlimitedOCRSAMAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

    init(dim: Int, heads: Int, input: Int) {
        self.heads = heads
        let headDim = dim / heads
        scale = pow(Float(headDim), -0.5)
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        _proj.wrappedValue = Linear(dim, dim, bias: true)
        _relPosH.wrappedValue = MLXArray.zeros([2 * input - 1, headDim])
        _relPosW.wrappedValue = MLXArray.zeros([2 * input - 1, headDim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2)
        let mixed = qkv(x).reshaped(b, h * w, 3, heads, -1).transposed(2, 0, 3, 1, 4)
        let q = mixed[0].reshaped(b * heads, h * w, -1)
        let k = mixed[1].reshaped(b, heads, h * w, -1)
        let v = mixed[2].reshaped(b, heads, h * w, -1)
        let bias = relativeBias(q, height: h, width: w, batch: b)
        let queries = q.reshaped(b, heads, h * w, -1)
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v, scale: scale, mask: .array(bias))
        let merged = out.transposed(0, 2, 1, 3).reshaped(b, h, w, -1)
        return proj(merged)
    }

    private func relativeBias(_ q: MLXArray, height: Int, width: Int, batch: Int) -> MLXArray {
        let relH = relative(qSize: height, kSize: height, table: relPosH)
        let relW = relative(qSize: width, kSize: width, table: relPosW)
        let dim = q.dim(-1)
        let rows = q.reshaped(q.dim(0), height, width, dim)
        var hBias = einsum("bhwc,hkc->bhwk", rows, relH).expandedDimensions(axis: -1)
        var wBias = einsum("bhwc,wkc->bhwk", rows, relW).expandedDimensions(axis: -2)
        hBias = hBias.reshaped(q.dim(0), height * width, height, 1)
        wBias = wBias.reshaped(q.dim(0), height * width, 1, width)
        let sum = (hBias + wBias).reshaped(batch, heads, height * width, height * width)
        return sum.asType(q.dtype)
    }

    private func relative(qSize: Int, kSize: Int, table: MLXArray) -> MLXArray {
        let maxDist = 2 * max(qSize, kSize) - 1
        var pos = table
        if pos.dim(0) != maxDist {
            let dtype = pos.dtype
            var grid = pos.asType(.float32).reshaped(1, pos.dim(0), -1).transposed(0, 2, 1)
            let scale = Float(grid.dim(2)) / Float(maxDist)
            let indices = MLXArray(Int32(0)..<Int32(maxDist)).asType(.float32) * scale
            let lo = indices.floor().asType(.int32)
            let hi = MLX.minimum(lo + 1, MLXArray(Int32(grid.dim(2) - 1)))
            let weight = indices - lo.asType(.float32)
            let left = grid.take(lo, axis: 2) * (1 - weight)
            let right = grid.take(hi, axis: 2) * weight
            grid = (left + right).asType(dtype)
            pos = grid.reshaped(-1, maxDist).transposed(1, 0)
        }
        let qScale = max(Float(kSize) / Float(qSize), 1)
        let kScale = max(Float(qSize) / Float(kSize), 1)
        let qCoords = MLXArray(Int32(0)..<Int32(qSize)).asType(.float32).expandedDimensions(axis: 1) * qScale
        let kCoords = MLXArray(Int32(0)..<Int32(kSize)).asType(.float32).expandedDimensions(axis: 0) * kScale
        let coords = (qCoords - kCoords) + Float(kSize - 1) * kScale
        return pos[coords.asType(.int32)]
    }
}

final class UnlimitedOCRSAMBlock: Module {
    let window: Int
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: UnlimitedOCRSAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: UnlimitedOCRGEGLU

    init(dim: Int, heads: Int, window: Int, input: Int) {
        self.window = window
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _attn.wrappedValue = UnlimitedOCRSAMAttention(
            dim: dim, heads: heads, input: window == 0 ? input : window)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        _mlp.wrappedValue = UnlimitedOCRGEGLU(input: dim, hidden: dim * 4, gelu: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = norm1(x)
        var pad = (0, 0)
        if window > 0 {
            let parts = UnlimitedOCRWindows.partition(hidden, window: window)
            hidden = parts.0
            pad = (parts.1, parts.2)
        }
        hidden = attn(hidden)
        if window > 0 {
            hidden = UnlimitedOCRWindows.unpartition(
                hidden, window: window, padded: pad, original: (x.dim(1), x.dim(2)))
        }
        let y = x + hidden
        return y + mlp(norm2(y))
    }
}

enum UnlimitedOCRWindows {
    static func partition(_ x: MLXArray, window: Int) -> (MLXArray, Int, Int) {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        let padH = (window - h % window) % window
        let padW = (window - w % window) % window
        var y = x
        if padH > 0 || padW > 0 {
            y = padded(y, widths: [0, [0, padH], [0, padW], 0])
        }
        let hp = h + padH
        let wp = w + padW
        y = y.reshaped(b, hp / window, window, wp / window, window, c)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, window, window, c)
        return (y, hp, wp)
    }

    static func unpartition(
        _ windows: MLXArray, window: Int, padded pad: (Int, Int), original: (Int, Int)
    ) -> MLXArray {
        let hp = pad.0, wp = pad.1
        let count = hp / window * (wp / window)
        let b = windows.dim(0) / count
        var y = windows.reshaped(b, hp / window, wp / window, window, window, -1)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(b, hp, wp, -1)
        if hp > original.0 || wp > original.1 {
            y = y[0..., ..<original.0, ..<original.1, 0...]
        }
        return y
    }
}

final class UnlimitedOCRSAM: Module, UnaryLayer {
    @ModuleInfo(key: "patch_embed") var patch: UnlimitedOCRConv
    @ParameterInfo(key: "pos_embed") var position: MLXArray
    let blocks: [UnlimitedOCRSAMBlock]
    /// HF / mlx-vlm store the neck as a list: `neck.0` … `neck.3`.
    /// `Sequential` keeps that order and is a unary `callAsFunction(MLXArray)`.
    /// Sanitize moves those keys to `neck.layers.*`.
    @ModuleInfo(key: "neck") var neck: Sequential
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    init(_ config: UnlimitedOCRConfiguration) {
        let side = UnlimitedOCRSupport.baseSide / config.samPatch
        _patch.wrappedValue = UnlimitedOCRConv(
            input: 3, output: config.samWidth, kernel: config.samPatch,
            stride: config.samPatch, bias: true, name: "proj")
        _position.wrappedValue = MLXArray.zeros([1, side, side, config.samWidth])
        blocks = (0..<config.samLayers).map { index in
            let global = config.samGlobalAttention.contains(index)
            return UnlimitedOCRSAMBlock(
                dim: config.samWidth, heads: config.samHeads,
                window: global ? 0 : config.samWindow, input: side)
        }
        let down = config.samDownsample
        let neckWidth = config.samNeck
        _neck.wrappedValue = Sequential(layers: [
            Conv2d(inputChannels: config.samWidth, outputChannels: neckWidth, kernelSize: 1, bias: false),
            LayerNorm(dimensions: neckWidth, eps: 1e-6),
            Conv2d(
                inputChannels: neckWidth, outputChannels: neckWidth,
                kernelSize: 3, padding: 1, bias: false),
            LayerNorm(dimensions: neckWidth, eps: 1e-6),
        ])
        _net2.wrappedValue = Conv2d(
            inputChannels: config.samNeck, outputChannels: down[0],
            kernelSize: 3, stride: 2, padding: 1, bias: false)
        _net3.wrappedValue = Conv2d(
            inputChannels: down[0], outputChannels: down.count > 1 ? down[1] : 1024,
            kernelSize: 3, stride: 2, padding: 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = patch(x)
        if position.dim(1) == hidden.dim(1) {
            hidden = hidden + position
        } else {
            let nchw = position.transposed(0, 3, 1, 2).asType(.float32)
            let resized = bicubicInterpolate(
                nchw, size: (hidden.dim(1), hidden.dim(2)), alignCorners: false, antialias: true)
            hidden = hidden + resized.asType(position.dtype).transposed(0, 2, 3, 1)
        }
        for block in blocks { hidden = block(hidden) }
        hidden = neck(hidden)
        hidden = net2(hidden)
        return net3(hidden)
    }
}

/// Conv stored at `proj` or used directly. SAM's patch stem is `patch_embed.proj`.
final class UnlimitedOCRConv: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    init(input: Int, output: Int, kernel: Int, stride: Int, bias: Bool, name: String) {
        _proj.wrappedValue = Conv2d(
            inputChannels: input, outputChannels: output,
            kernelSize: kernel, stride: stride, bias: bias)
        super.init()
        _ = name
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// SAM MLP. Checkpoint keys are `lin1` / `lin2`.
final class UnlimitedOCRGEGLU: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(input: Int, hidden: Int, gelu: Bool) {
        _lin1.wrappedValue = Linear(input, hidden, bias: true)
        _lin2.wrappedValue = Linear(hidden, input, bias: true)
        super.init()
        _ = gelu
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(MLXNN.gelu(lin1(x)))
    }
}

// MARK: - CLIP

final class UnlimitedOCRClipAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "qkv_proj") var qkv: Linear
    @ModuleInfo(key: "out_proj") var out: Linear

    init(dim: Int, heads: Int) {
        self.heads = heads
        scale = pow(Float(dim / heads), -0.5)
        _qkv.wrappedValue = Linear(dim, dim * 3, bias: true)
        _out.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mixed = qkv(x)
        let width = x.dim(-1)
        let q = mixed[.ellipsis, ..<width]
        let k = mixed[.ellipsis, width..<(width * 2)]
        let v = mixed[.ellipsis, (width * 2)...]
        let b = x.dim(0), n = x.dim(1)
        let queries = q.reshaped(b, n, heads, -1).transposed(0, 2, 1, 3)
        let keys = k.reshaped(b, n, heads, -1).transposed(0, 2, 1, 3)
        let values = v.reshaped(b, n, heads, -1).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none)
        return self.out(out.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}

final class UnlimitedOCRClipMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(hidden: Int, intermediate: Int) {
        _fc1.wrappedValue = Linear(hidden, intermediate, bias: true)
        _fc2.wrappedValue = Linear(intermediate, hidden, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(MLXNN.gelu(fc1(x))) }
}

final class UnlimitedOCRClipLayer: Module {
    @ModuleInfo(key: "self_attn") var attn: UnlimitedOCRClipAttention
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: UnlimitedOCRClipMLP
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm

    init(hidden: Int, intermediate: Int, heads: Int) {
        _attn.wrappedValue = UnlimitedOCRClipAttention(dim: hidden, heads: heads)
        _norm1.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-6)
        _mlp.wrappedValue = UnlimitedOCRClipMLP(hidden: hidden, intermediate: intermediate)
        _norm2.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = x + attn(norm1(x))
        return y + mlp(norm2(y))
    }
}

final class UnlimitedOCRClipStack: Module {
    let layers: [UnlimitedOCRClipLayer]
    init(_ config: UnlimitedOCRConfiguration) {
        layers = (0..<config.clipLayers).map { _ in
            UnlimitedOCRClipLayer(
                hidden: config.clipHidden, intermediate: config.clipIntermediate,
                heads: config.clipHeads)
        }
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in layers { y = layer(y) }
        return y
    }
}

final class UnlimitedOCRProjector: Module {
    @ModuleInfo(key: "layers") var layers: Linear
    init(_ config: UnlimitedOCRConfiguration) {
        _layers.wrappedValue = Linear(config.projectorInput, config.projectorHidden, bias: true)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { layers(x) }
}
