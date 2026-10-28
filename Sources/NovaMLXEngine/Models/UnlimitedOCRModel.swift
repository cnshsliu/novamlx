import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM
import PDFKit

// MARK: - Language

final class UnlimitedOCRDenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(input: Int, hidden: Int) {
        _gate.wrappedValue = Linear(input, hidden, bias: false)
        _up.wrappedValue = Linear(input, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, input, bias: false)
        super.init()
    }

    private var compiledToken: (([MLXArray]) -> [MLXArray])?

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // One static trace for a single token. Prefill stays eager.
        guard x.dim(0) == 1, x.dim(1) == 1 else { return eager(x) }
        if compiledToken == nil {
            compiledToken = compile { [self] (args: [MLXArray]) -> [MLXArray] in
                [self.eager(args[0])]
            }
        }
        return compiledToken!([x])[0]
    }

    private func eager(_ x: MLXArray) -> MLXArray {
        down(compiledSiluProduct(gate(x), up(x)))
    }
}

final class UnlimitedOCRMoE: Module {
    let topK: Int
    let expertsCount: Int
    let scale: Float
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var shared: UnlimitedOCRDenseMLP

    init(_ config: UnlimitedOCRConfiguration) {
        topK = config.expertsPerToken
        expertsCount = config.routedExperts
        scale = 1
        _gate.wrappedValue = Linear(config.hiddenSize, config.routedExperts, bias: false)
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.routedExperts)
        _shared.wrappedValue = UnlimitedOCRDenseMLP(
            input: config.hiddenSize,
            hidden: config.moeIntermediateSize * config.sharedExperts)
        super.init()
    }

    private var compiledToken: (([MLXArray]) -> [MLXArray])?

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Traced once at [1, 1, hidden] after 4-bit quantize. Not shapeless.
        // `SwitchGLU` then stays on `gatherQuantizedMM` of the top-k experts.
        guard x.dim(0) == 1, x.dim(1) == 1 else { return eager(x) }
        if compiledToken == nil {
            compiledToken = compile { [self] (args: [MLXArray]) -> [MLXArray] in
                [self.eager(args[0])]
            }
        }
        return compiledToken!([x])[0]
    }

    private func eager(_ x: MLXArray) -> MLXArray {
        let scores = softmax(gate(x).asType(.float32), axis: -1, precise: true)
        let singleRow = x.dim(0) == 1 && x.dim(1) == 1
        let (inds, chosen) = singleRow
            ? UnlimitedOCRExpertTopK.selectRow(scores, experts: expertsCount, k: topK)
            : UnlimitedOCRExpertTopK.chain(scores, k: topK)
        let mixed = weightedExpertSum(experts(x, inds), (chosen * scale).asType(x.dtype))
        return mixed + shared(x)
    }
}

/// Single-row top-k. One threadgroup, ascending, same experts as
/// `argPartition` + `takeAlong`. Prefill and any wider row keep that chain.
enum UnlimitedOCRExpertTopK {
    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [Int: MLXFast.MLXFastKernel] = [:]

    static func selectRow(_ scores: MLXArray, experts: Int, k: Int) -> (MLXArray, MLXArray) {
        guard experts <= 1024, k > 0, k <= experts, scores.dtype == .float32,
            let kernel = kernel(experts: experts, k: k)
        else {
            return chain(scores, k: k)
        }
        let shape = [1, 1, k]
        let out = kernel(
            [scores],
            grid: (experts, 1, 1),
            threadGroup: (experts, 1, 1),
            outputShapes: [shape, shape],
            outputDTypes: [.uint32, .float32]
        )
        return (out[0].asType(.int32), out[1])
    }

    static func chain(_ scores: MLXArray, k: Int) -> (MLXArray, MLXArray) {
        let kth = scores.dim(-1) - k
        let inds = argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...].asType(.int32)
        return (inds, takeAlong(scores, inds, axis: -1))
    }

    private static func kernel(experts: Int, k: Int) -> MLXFast.MLXFastKernel? {
        lock.lock()
        defer { lock.unlock() }
        let key = experts * 1024 + k
        if let cached = kernels[key] { return cached }
        let source = """
            uint t = thread_position_in_threadgroup.x;
            threadgroup ulong sk[\(experts)];
            threadgroup float top_v[\(k)];
            float v = selection[t];
            uint b = (v == 0.0f) ? 0u : as_type<uint>(v);
            uint mono = isnan(v) ? 0xFFFFFFFFu : (b ^ ((uint)(((int)b) >> 31) | 0x80000000u));
            ulong key = (((ulong)mono) << 32) | (ulong)t;
            sk[t] = key;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            int above = 0;
            for (uint j = 0; j < \(experts)u; ++j) {
                above += (sk[j] > key) ? 1 : 0;
            }
            if (above < \(k)) {
                uint slot = (uint)(\(k) - 1 - above);
                top_v[slot] = selection[t];
                inds[slot] = t;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0) {
                for (int q = 0; q < \(k); ++q) {
                    scores[q] = top_v[q];
                }
            }
            """
        let kernel = MLXFast.metalKernel(
            name: "uocr_router_topk_\(experts)_\(k)",
            inputNames: ["selection"],
            outputNames: ["inds", "scores"],
            source: source,
            header: "",
            ensureRowContiguous: true
        )
        kernels[key] = kernel
        return kernel
    }
}

final class UnlimitedOCRRingAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: UnlimitedOCRConfiguration) {
        heads = config.attentionHeads
        scale = pow(Float(config.headDim), -0.5)
        let dim = config.hiddenSize
        let width = config.headDim * heads
        _qProj.wrappedValue = Linear(dim, width, bias: false)
        _kProj.wrappedValue = Linear(dim, width, bias: false)
        _vProj.wrappedValue = Linear(dim, width, bias: false)
        _oProj.wrappedValue = Linear(width, dim, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: UnlimitedOCRRingCache, rope: UnlimitedOCRRope
    ) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), dim = x.dim(2) / heads
        let mask = cache.makeMask(n: n, windowSize: nil, returnArray: false)
        let positions = MLXArray(Int32(cache.offset)..<Int32(cache.offset + n))
        let angles = rope.linear(positions: positions, dtype: x.dtype)
        var q = qProj(x).reshaped(b, n, heads, dim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, n, heads, dim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, n, heads, dim).transposed(0, 2, 1, 3)
        q = UnlimitedOCRDashMath.applyRope(q, cos: angles.0, sin: angles.1)
        k = UnlimitedOCRDashMath.applyRope(k, cos: angles.0, sin: angles.1)
        let (keys, values) = cache.update(keys: k, values: v)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}

final class UnlimitedOCRDashBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: UnlimitedOCRDashAttention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ config: UnlimitedOCRConfiguration, layer: Int) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attention.wrappedValue = UnlimitedOCRDashAttention(config, layer: layer)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if layer < config.firstDenseLayers {
            _mlp.wrappedValue = UnlimitedOCRDenseMLP(
                input: config.hiddenSize, hidden: config.intermediateSize)
        } else {
            _mlp.wrappedValue = UnlimitedOCRMoE(config)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, _ ctx: UnlimitedOCRDashContext, _ cache: UnlimitedOCRDashCache
    ) -> MLXArray {
        let y = x + attention(inputNorm(x), ctx, cache)
        if let dense = mlp as? UnlimitedOCRDenseMLP {
            return y + dense(postNorm(y))
        }
        return y + (mlp as! UnlimitedOCRMoE)(postNorm(y))
    }
}

final class UnlimitedOCRRingBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: UnlimitedOCRRingAttention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ config: UnlimitedOCRConfiguration, layer: Int) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attention.wrappedValue = UnlimitedOCRRingAttention(config)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if layer < config.firstDenseLayers {
            _mlp.wrappedValue = UnlimitedOCRDenseMLP(
                input: config.hiddenSize, hidden: config.intermediateSize)
        } else {
            _mlp.wrappedValue = UnlimitedOCRMoE(config)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: UnlimitedOCRRingCache, rope: UnlimitedOCRRope) -> MLXArray {
        let y = x + attention(inputNorm(x), cache: cache, rope: rope)
        if let dense = mlp as? UnlimitedOCRDenseMLP {
            return y + dense(postNorm(y))
        }
        return y + (mlp as! UnlimitedOCRMoE)(postNorm(y))
    }
}

final class UnlimitedOCRText: Module {
    let dash: Bool
    let rope: UnlimitedOCRRope
    let runtime: UnlimitedOCRDashRuntime
    let ringWindow: Int
    var promptIDs: [Int] = []
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Module]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: UnlimitedOCRConfiguration) {
        dash = config.isDash
        rope = UnlimitedOCRRope(config)
        runtime = UnlimitedOCRDashRuntime(config)
        ringWindow = config.slidingWindow
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        if config.isDash {
            layers = (0..<config.hiddenLayers).map { UnlimitedOCRDashBlock(config, layer: $0) }
        } else {
            layers = (0..<config.hiddenLayers).map { UnlimitedOCRRingBlock(config, layer: $0) }
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func makeCache() -> [any KVCache] {
        if dash {
            return layers.map { _ in UnlimitedOCRDashCache(runtime: runtime) }
        }
        return layers.map { _ in UnlimitedOCRRingCache(window: ringWindow) }
    }

    func hidden(_ embeds: MLXArray, cache: [any KVCache], tokens: MLXArray?) -> MLXArray {
        if dash {
            if cache[0].offset == 0 { runtime.start(promptIDs) }
            let ctx = unlimitedOCRContext(
                runtime: runtime, rope: rope, tokens: tokens,
                offset: cache[0].offset, length: embeds.dim(1), dtype: embeds.dtype)
            var y = embeds
            for (layer, item) in zip(layers, cache) {
                y = (layer as! UnlimitedOCRDashBlock)(y, ctx, item as! UnlimitedOCRDashCache)
            }
            return norm(y)
        }
        var y = embeds
        for (layer, item) in zip(layers, cache) {
            y = (layer as! UnlimitedOCRRingBlock)(
                y, cache: item as! UnlimitedOCRRingCache, rope: rope)
        }
        return norm(y)
    }
}

final class UnlimitedOCRLanguage: Module {
    let config: UnlimitedOCRConfiguration
    @ModuleInfo(key: "model") var model: UnlimitedOCRText
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    private var quantized = false

    init(_ config: UnlimitedOCRConfiguration) {
        self.config = config
        _model.wrappedValue = UnlimitedOCRText(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    func makeCache() -> [any KVCache] { model.makeCache() }

    func hidden(_ embeds: MLXArray, cache: [any KVCache], tokens: MLXArray?) -> MLXArray {
        model.hidden(embeds, cache: cache, tokens: tokens)
    }

    func logits(_ hidden: MLXArray) -> MLXArray { lmHead(hidden) }

    /// Quantize the language linears after a BF16 load. Vision and the DASH
    /// router stay dense: the router picks blocks, and the vision tower is the
    /// OCR feature. Already-quantized packs are left alone.
    func quantizeLanguageIfNeeded() {
        guard !quantized, let bits = UnlimitedOCRSupport.languageBits, bits == 4 || bits == 8
        else { return }
        quantized = true
        quantize(model: self, groupSize: 64, bits: bits) { path, module in
            guard UnlimitedOCRSupport.shouldQuantizeLanguage(path: path) else { return false }
            return module is Linear || module is SwitchLinear
        }
    }
}

// MARK: - VLM

public final class UnlimitedOCR: Module, VLMModel, KVCacheDimensionProvider {
    public let config: UnlimitedOCRConfiguration
    @ModuleInfo(key: "sam_model") var sam: UnlimitedOCRSAM
    @ModuleInfo(key: "vision_model") var vision: UnlimitedOCRClipStackHolder
    @ModuleInfo(key: "projector") var projector: UnlimitedOCRProjector
    @ModuleInfo(key: "language_model") var language: UnlimitedOCRLanguage
    @ParameterInfo(key: "image_newline") var imageNewline: MLXArray
    @ParameterInfo(key: "view_separator") var viewSeparator: MLXArray

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: config.kvHeads, count: config.hiddenLayers) }
    public var loraLayers: [Module] { language.model.layers }

    public init(_ config: UnlimitedOCRConfiguration) {
        self.config = config
        _sam.wrappedValue = UnlimitedOCRSAM(config)
        _vision.wrappedValue = UnlimitedOCRClipStackHolder(config)
        _projector.wrappedValue = UnlimitedOCRProjector(config)
        _language.wrappedValue = UnlimitedOCRLanguage(config)
        let width = config.projectorHidden
        _imageNewline.wrappedValue = MLXArray.zeros([width])
        _viewSeparator.wrappedValue = MLXArray.zeros([width])
        super.init()
    }

    public func prepare() throws {
        language.quantizeLanguageIfNeeded()
    }

    public func newCache(parameters: GenerateParameters?) throws -> [any KVCache] {
        language.makeCache()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        UnlimitedOCRCheckpoint.sanitize(weights, config: config)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens.ndim == 1 ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        let ids = tokens.asArray(Int.self)
        language.model.promptIDs = ids
        let embeds = try mergedEmbeddings(tokens: tokens, input: input)
        let total = embeds.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = language.hidden(
                embeds[0..., range, 0...], cache: cache, tokens: tokens[0..., range])
            MLX.eval(cache.flatMap(\.state))
        }
        let hidden = language.hidden(
            embeds[0..., processed..., 0...], cache: cache, tokens: tokens[0..., processed...])
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: language.logits(hidden[0..., (-1)..., 0...])))
    }

    /// Decode step used by ContinuousBatcher and the VLM loops.
    /// The `LMInput.Text` overload is not what that path calls. The default
    /// `callAsFunction(_:cache:)` in `LanguageModel` traps.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokens = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        let caches = cache ?? language.makeCache()
        let embeds = language.model.embedTokens(tokens)
        let hidden = language.hidden(embeds, cache: caches, tokens: tokens)
        return language.logits(hidden)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: self(input.tokens, cache: cache))
    }

    private func mergedEmbeddings(tokens: MLXArray, input: LMInput) throws -> MLXArray {
        var embeds = language.model.embedTokens(tokens)
        guard let image = input.image else { return embeds }
        let globals = nhwc(image.pixels)
        let crops = input.video.map { nhwc($0.pixels) }
        let frames = image.frames ?? []
        let positions = imagePositions(tokens)
        var cursor = 0
        var patchCursor = 0
        for (index, frame) in frames.enumerated() {
            let features = pageFeatures(
                global: globals[index..<(index + 1)],
                crops: crops, width: frame.t, height: frame.h, patchCursor: &patchCursor)
            let end = cursor + features.dim(0)
            guard end <= positions.count else {
                throw NSError(
                    domain: "UnlimitedOCR", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Image features exceed <image> tokens."])
            }
            let rows = MLXArray(positions[cursor..<end].map(Int32.init))
            embeds[0, rows] = features
            cursor = end
        }
        return embeds
    }

    private func pageFeatures(
        global: MLXArray, crops: MLXArray?, width: Int, height: Int, patchCursor: inout Int
    ) -> MLXArray {
        let hasCrops = width > 1 || height > 1
        let count = width * height
        if hasCrops, let crops, count > 0 {
            let tiles = crops[patchCursor..<(patchCursor + count)]
            patchCursor += count
            let local = project(tiles)
            let globalFeatures = project(global)[0]
            return concatenated(
                [pack(local, rows: height, cols: width), packGlobal(globalFeatures), viewSeparator.reshaped(1, -1)],
                axis: 0)
        }
        let globalFeatures = project(global)[0]
        return concatenated([packGlobal(globalFeatures), viewSeparator.reshaped(1, -1)], axis: 0)
    }

    /// One tile at a time. A 900×1200 page is a global view plus a crop grid;
    /// fusing every SAM/CLIP tower into the language prefill holds the whole
    /// activation graph until the first token and can sit for minutes.
    private func project(_ images: MLXArray) -> MLXArray {
        let count = images.dim(0)
        if count <= 1 {
            let features = encodeTile(images)
            eval(features)
            return features
        }
        var parts: [MLXArray] = []
        parts.reserveCapacity(count)
        for index in 0..<count {
            let features = encodeTile(images[index..<(index + 1)])
            eval(features)
            parts.append(features)
            Memory.clearCache()
        }
        return concatenated(parts, axis: 0)
    }

    private func encodeTile(_ images: MLXArray) -> MLXArray {
        let sam = sam(images)
        let clip = vision.clip(images, patches: sam)
        let joined = concatenated([clip[0..., 1..., 0...], sam.reshaped(sam.dim(0), -1, sam.dim(-1))], axis: -1)
        return projector(joined)
    }

    private func pack(_ local: MLXArray, rows: Int, cols: Int) -> MLXArray {
        let hw = local.dim(1)
        let side = Int(Double(hw).squareRoot())
        let dim = local.dim(2)
        var grid = local.reshaped(rows, cols, side, side, dim).transposed(0, 2, 1, 3, 4)
        grid = grid.reshaped(rows * side, cols * side, dim)
        let newline = broadcast(imageNewline.reshaped(1, 1, dim), to: [rows * side, 1, dim])
        return concatenated([grid, newline], axis: 1).reshaped(-1, dim)
    }

    private func packGlobal(_ global: MLXArray) -> MLXArray {
        let side = Int(Double(global.dim(0)).squareRoot())
        let dim = global.dim(1)
        let grid = global.reshaped(side, side, dim)
        let newline = broadcast(imageNewline.reshaped(1, 1, dim), to: [side, 1, dim])
        return concatenated([grid, newline], axis: 1).reshaped(-1, dim)
    }

    private func imagePositions(_ tokens: MLXArray) -> [Int] {
        let ids = tokens.asArray(Int.self)
        return ids.indices.filter { ids[$0] == config.imageTokenID }
    }

    private func nhwc(_ x: MLXArray) -> MLXArray { x.transposed(0, 2, 3, 1) }
}

/// Holds the CLIP tower at `vision_model` with `embeddings` + `transformer`.
public final class UnlimitedOCRClipStackHolder: Module {
    @ModuleInfo(key: "embeddings") var embeddings: UnlimitedOCRClipEmbed
    @ModuleInfo(key: "pre_layrnorm") var preNorm: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: UnlimitedOCRClipStack
    let clipHidden: Int

    init(_ config: UnlimitedOCRConfiguration) {
        clipHidden = config.clipHidden
        _embeddings.wrappedValue = UnlimitedOCRClipEmbed(config)
        _preNorm.wrappedValue = LayerNorm(dimensions: config.clipHidden, eps: 1e-6)
        _transformer.wrappedValue = UnlimitedOCRClipStack(config)
        super.init()
    }

    func clip(_ x: MLXArray, patches: MLXArray?) -> MLXArray {
        transformer(preNorm(embeddings(x, patches: patches)))
    }
}

final class UnlimitedOCRClipEmbed: Module {
    let hidden: Int
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding") var patch: Conv2d
    @ModuleInfo(key: "position_embedding") var position: Embedding

    init(_ config: UnlimitedOCRConfiguration) {
        hidden = config.clipHidden
        let grid = 224 / config.clipPatch
        _classEmbedding.wrappedValue = MLXArray.zeros([hidden])
        _patch.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: hidden,
            kernelSize: IntOrPair(config.clipPatch), stride: IntOrPair(config.clipPatch),
            bias: false)
        _position.wrappedValue = Embedding(embeddingCount: grid * grid + 1, dimensions: hidden)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, patches: MLXArray?) -> MLXArray {
        let tokens: MLXArray
        if let patches {
            tokens = patches.reshaped(patches.dim(0), -1, patches.dim(-1))
        } else {
            let embedded = patch(x)
            tokens = embedded.reshaped(embedded.dim(0), -1, embedded.dim(-1))
        }
        let cls = broadcast(classEmbedding.reshaped(1, 1, hidden), to: [tokens.dim(0), 1, hidden])
            .asType(tokens.dtype)
        let hiddenStates = concatenated([cls, tokens], axis: 1)
        return hiddenStates + resized(count: hiddenStates.dim(1)).asType(hiddenStates.dtype)
    }

    private func resized(count: Int) -> MLXArray {
        let table = position.weight.expandedDimensions(axis: 0)
        if table.dim(1) == count { return table }
        let cls = table[0..., ..<1, 0...]
        let body = table[0..., 1..., 0...]
        let side = Int(Double(body.dim(1)).squareRoot())
        let target = Int(Double(count - 1).squareRoot())
        var spatial = body.reshaped(1, side, side, hidden).transposed(0, 3, 1, 2).asType(.float32)
        spatial = bicubicInterpolate(spatial, size: (target, target), alignCorners: false, antialias: false)
        let flat = spatial.asType(table.dtype).transposed(0, 2, 3, 1).reshaped(1, count - 1, hidden)
        return concatenated([cls, flat], axis: 1)
    }
}

// MARK: - Checkpoint

enum UnlimitedOCRCheckpoint {
    static func sanitize(
        _ weights: [String: MLXArray], config: UnlimitedOCRConfiguration
    ) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard var name = UnlimitedOCRSupport.renameWeight(key) else { continue }
            name = UnlimitedOCRSupport.remapSAMNeck(name)
            mapped[name] = transposeConv(name, value)
        }
        stackExperts(&mapped, layers: config.hiddenLayers, experts: config.routedExperts)
        return mapped
    }

    private static func transposeConv(_ key: String, _ value: MLXArray) -> MLXArray {
        let conv = key.hasSuffix("patch_embed.proj.weight")
            || key.hasSuffix("patch_embedding.weight")
            || key.hasSuffix("neck.layers.0.weight")
            || key.hasSuffix("neck.layers.2.weight")
            || key.hasSuffix("neck.0.weight")
            || key.hasSuffix("neck.2.weight")
            || key.hasSuffix("net_2.weight")
            || key.hasSuffix("net_3.weight")
        guard conv, value.ndim == 4 else { return value }
        let out = value.dim(0), kh = value.dim(1), kw = value.dim(2)
        if out >= kh && out >= kw && kh == kw { return value }
        return value.transposed(0, 2, 3, 1)
    }

    private static func stackExperts(
        _ weights: inout [String: MLXArray], layers: Int, experts: Int
    ) {
        for layer in 0..<layers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for name in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let probe = "\(prefix).experts.0.\(name).\(suffix)"
                    guard weights[probe] != nil else { continue }
                    var parts: [MLXArray] = []
                    for expert in 0..<experts {
                        let key = "\(prefix).experts.\(expert).\(name).\(suffix)"
                        guard let value = weights.removeValue(forKey: key) else { continue }
                        parts.append(value)
                    }
                    if parts.count == experts {
                        weights["\(prefix).switch_mlp.\(name).\(suffix)"] = stacked(parts, axis: 0)
                    }
                }
            }
        }
    }
}

// MARK: - Processor

public struct UnlimitedOCRProcessorConfiguration: Decodable, Sendable {
    public var pageLayout: Bool = true
    public var maxCrops: Int = 16
    public var imageTokenID: Int = UnlimitedOCRSupport.imageTokenID
    public var pageTokenID: Int = UnlimitedOCRSupport.pageTokenID
    enum CodingKeys: String, CodingKey {
        case pageLayout = "page_layout"
        case maxCrops = "max_crops"
        case imageTokenID = "image_token_id"
        case pageTokenID = "page_token_id"
    }
    public init() {}
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        pageLayout = try c.decodeIfPresent(Bool.self, forKey: .pageLayout) ?? true
        maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? (pageLayout ? 16 : 32)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? UnlimitedOCRSupport.imageTokenID
        pageTokenID = try c.decodeIfPresent(Int.self, forKey: .pageTokenID) ?? UnlimitedOCRSupport.pageTokenID
    }
}

public struct UnlimitedOCRProcessor: UserInputProcessor {
    let config: UnlimitedOCRProcessorConfiguration
    /// `MLXLMCommon.Tokenizer`. Unqualified `Tokenizer` is Nova's concrete struct.
    let tokenizer: any MLXLMCommon.Tokenizer

    public init(
        _ config: UnlimitedOCRProcessorConfiguration, tokenizer: any MLXLMCommon.Tokenizer
    ) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = try Self.collect(input.images)
        if images.isEmpty {
            let text = Self.promptText(input)
            let ids = [UnlimitedOCRSupport.bosTokenID] + encodeText(text)
            return LMInput(tokens: MLXArray(ids).expandedDimensions(axis: 0))
        }
        let prompt = Self.composePrompt(Self.promptText(input), images: images.count, pageLayout: config.pageLayout)
        let built = try build(prompt: prompt, images: images)
        return LMInput(
            text: .init(tokens: MLXArray(built.ids).expandedDimensions(axis: 0)),
            image: .init(pixels: built.globals, frames: built.frames),
            video: built.crops.map { .init(pixels: $0) })
    }

    private func build(prompt: String, images: [CIImage]) throws -> (
        ids: [Int], globals: MLXArray, crops: MLXArray?, frames: [THW]
    ) {
        var parts = prompt.components(separatedBy: "<image>")
        if parts.count == 2 && images.count > 1 {
            let mark = config.pageLayout ? "<PAGE><image>" : "<image>"
            let expanded = parts[0] + String(repeating: mark, count: images.count) + parts[1]
            parts = expanded.components(separatedBy: "<image>")
        }
        guard parts.count == images.count + 1 else {
            throw NSError(
                domain: "UnlimitedOCR", code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "Prompt has \(parts.count - 1) <image> markers for \(images.count) images."])
        }
        var ids = [UnlimitedOCRSupport.bosTokenID]
        var globals: [MLXArray] = []
        var crops: [MLXArray] = []
        var frames: [THW] = []
        let mean = CIColor(red: 0.5, green: 0.5, blue: 0.5)
        for (image, prefix) in zip(images, parts.dropLast()) {
            ids += encode(prefix)
            let grid = UnlimitedOCRSupport.cropGrid(
                width: Int(image.extent.width), height: Int(image.extent.height),
                maxCrops: config.maxCrops)
            let global = try render(
                Self.fit(image, side: UnlimitedOCRSupport.baseSide, color: mean), side: UnlimitedOCRSupport.baseSide)
            globals.append(global)
            var tiles: [CIImage] = []
            if grid.0 > 1 || grid.1 > 1 {
                tiles = Self.tiles(image, grid: grid, side: UnlimitedOCRSupport.tileSide)
            }
            for tile in tiles {
                crops.append(try render(tile, side: UnlimitedOCRSupport.tileSide))
            }
            frames.append(THW(grid.0, grid.1, 1))
            let local = UnlimitedOCRSupport.localTokenCount(widthCrops: grid.0, heightCrops: grid.1)
            let globalCount = UnlimitedOCRSupport.globalTokenCount()
            // Official model writes local tiles, then the global view, then the separator.
            ids += Array(repeating: config.imageTokenID, count: local + globalCount)
        }
        ids += encode(parts.last ?? "")
        let globalBatch = concatenated(globals, axis: 0)
        let cropBatch = crops.isEmpty ? nil : concatenated(crops, axis: 0)
        return (ids, globalBatch, cropBatch, frames)
    }

    private func encode(_ text: String) -> [Int] {
        guard !text.isEmpty else { return [] }
        var ids: [Int] = []
        let pieces = text.components(separatedBy: "<PAGE>")
        for (index, piece) in pieces.enumerated() {
            if index > 0 { ids.append(config.pageTokenID) }
            if !piece.isEmpty {
                ids += encodeText(piece)
            }
        }
        return ids
    }

    /// `encode(text:)` is the tokenizer call this checkout exposes. Drop a
    /// leading BOS when that call already inserted one.
    private func encodeText(_ text: String) -> [Int] {
        var ids = tokenizer.encode(text: text)
        if ids.first == UnlimitedOCRSupport.bosTokenID {
            ids.removeFirst()
        }
        return ids
    }

    private func render(_ image: CIImage, side: Int) throws -> MLXArray {
        let normalized = image.toSRGB().normalized(mean: (0.5, 0.5, 0.5), std: (0.5, 0.5, 0.5))
        return normalized.asMLXArray()
    }

    static func promptText(_ input: UserInput) -> String {
        switch input.prompt {
        case .text(let text): return text
        case .chat(let messages): return messages.map(\.content).joined(separator: "\n")
        case .messages(let messages):
            return messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
        }
    }

    static func composePrompt(_ text: String, images: Int, pageLayout: Bool) -> String {
        var body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.isEmpty || (images > 1 && body == "document parsing.") {
            body = images > 1 ? "Multi page parsing." : "document parsing."
        }
        guard !body.contains("<image>") else {
            if pageLayout && !body.contains("<PAGE>") {
                return body.replacingOccurrences(of: "<image>", with: "<PAGE><image>")
            }
            return body
        }
        let mark = pageLayout ? "<PAGE><image>" : "<image>"
        return String(repeating: mark, count: images) + body
    }

    static func fit(_ image: CIImage, side: Int, color: CIColor) -> CIImage {
        let extent = image.extent.integral
        let scale = min(CGFloat(side) / max(extent.width, 1), CGFloat(side) / max(extent.height, 1))
        let fitted = image.transformed(by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y))
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let placed = fitted.extent
        let dx = (CGFloat(side) - placed.width) / 2 - placed.origin.x
        let dy = (CGFloat(side) - placed.height) / 2 - placed.origin.y
        let background = CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        return fitted.transformed(by: CGAffineTransform(translationX: dx, y: dy)).composited(over: background)
            .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
    }

    static func tiles(_ image: CIImage, grid: (Int, Int), side: Int) -> [CIImage] {
        let width = grid.0 * side
        let height = grid.1 * side
        let extent = image.extent.integral
        let sx = CGFloat(width) / max(extent.width, 1)
        let sy = CGFloat(height) / max(extent.height, 1)
        let scaled = image.transformed(by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y))
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
        var pages: [CIImage] = []
        for index in 0..<(grid.0 * grid.1) {
            let col = index % grid.0
            let row = index / grid.0
            let rect = CGRect(x: col * side, y: row * side, width: side, height: side)
            pages.append(scaled.cropped(to: rect).transformed(by: CGAffineTransform(translationX: -rect.origin.x, y: -rect.origin.y)))
        }
        return pages
    }

    static func collect(_ images: [UserInput.Image]) throws -> [CIImage] {
        var result: [CIImage] = []
        for image in images {
            // `UserInput.Image` is an enum on this mlx-swift-lm. It has no
            // `.source` property.
            switch image {
            case .url(let url) where url.pathExtension.lowercased() == "pdf":
                result.append(contentsOf: try renderPDF(url))
            default:
                result.append(try image.asCIImage())
            }
        }
        return result
    }

    static func renderPDF(_ url: URL) throws -> [CIImage] {
        guard let document = PDFDocument(url: url), document.pageCount > 0 else {
            throw NSError(
                domain: "UnlimitedOCR", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Could not open PDF \(url.path)"])
        }
        let dpi = Double(ProcessInfo.processInfo.environment["NOVA_UOCR_PDF_DPI"] ?? "") ?? 150
        let scale = dpi / 72
        var images: [CIImage] = []
        for index in 0..<document.pageCount {
            guard let page = document.page(at: index) else { continue }
            let box = page.bounds(for: .mediaBox)
            let width = max(Int(box.width * scale), 1)
            let height = max(Int(box.height * scale), 1)
            guard let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { continue }
            context.setFillColor(CGColor(gray: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.scaleBy(x: scale, y: scale)
            page.draw(with: .mediaBox, to: context)
            if let cg = context.makeImage() {
                images.append(CIImage(cgImage: cg))
            }
        }
        return images
    }
}

// MARK: - Registration

enum UnlimitedOCRRegistration {
    static func register() async {
        await VLMTypeRegistry.shared.registerModelType("unlimited-ocr") { data in
            let config = try JSONDecoder.json5().decode(UnlimitedOCRConfiguration.self, from: data)
            return UnlimitedOCR(config)
        }
        await VLMTypeRegistry.shared.registerModelType("unlimited_ocr") { data in
            let config = try JSONDecoder.json5().decode(UnlimitedOCRConfiguration.self, from: data)
            return UnlimitedOCR(config)
        }
        await VLMProcessorTypeRegistry.shared.registerProcessorType(
            "UnlimitedOCRProcessor"
        ) { data, tokenizer in
            let config = (try? JSONDecoder.json5().decode(
                UnlimitedOCRProcessorConfiguration.self, from: data))
                ?? UnlimitedOCRProcessorConfiguration()
            return UnlimitedOCRProcessor(config, tokenizer: tokenizer)
        }
        VLMProcessorLoadingRegistry.shared.register(UnlimitedOCRProcessorResolver())
    }
}

struct UnlimitedOCRProcessorResolver: VLMProcessorLoadingResolver {
    func fallbackProcessorConfiguration(
        for context: VLMProcessorLoadingContext
    ) throws -> VLMProcessorConfiguration? {
        guard context.modelType == "unlimited-ocr" || context.modelType == "unlimited_ocr" else {
            return nil
        }
        let model = try? JSONDecoder.json5().decode(
            UnlimitedOCRConfiguration.self, from: context.configurationData)
        let page = model?.isDash ?? true
        let crops = model?.maxCrops ?? (page ? 16 : 32)
        let json = """
        {"processor_class":"UnlimitedOCRProcessor","page_layout":\(page),"max_crops":\(crops)}
        """
        return VLMProcessorConfiguration(
            data: Data(json.utf8), processorType: "UnlimitedOCRProcessor")
    }

    func processorType(
        for context: VLMProcessorLoadingContext, declaredProcessorType: String?
    ) throws -> String? {
        guard context.modelType == "unlimited-ocr" || context.modelType == "unlimited_ocr" else {
            return nil
        }
        return declaredProcessorType ?? "UnlimitedOCRProcessor"
    }
}
