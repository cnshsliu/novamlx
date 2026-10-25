import Foundation
import os
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import NovaMLXCore
import NovaMLXUtils

// Flint (RecursiveSI/flint) — ported from mlx-lm mlx_lm/models/flint.py.
//
// Architecture: 49-layer decoder, GQA (32 q / 4 kv heads, head_dim 128) with
// per-head QK RMSNorm and a learned per-head attention sink (`softmax_offset`,
// GPT-OSS style). `layer_types` mixes sliding-window-513 layers (RoPE,
// RotatingKVCache) with full-attention layers (no positional encoding,
// KVCacheSimple). MoE from layer 2: sigmoid routing to 6 of 256 experts with
// renormalized scores plus an ungated shared expert; layers 0-1 are dense.

public struct FlintConfiguration: Codable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var mlpOnlyLayers: [Int]
    var layerTypes: [String]
    var slidingWindow: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float = 10000.0
    var maxPositionEmbeddings: Int?
    var tieWordEmbeddings: Bool = false
    /// Transformers nests the theta in `rope_parameters` instead of the
    /// top-level `rope_theta`; the nested value wins when present.
    var ropeParameters: [String: Double]?
    /// Official RecursiveSI/flint-int4 export markers (bits=4, embedding 8,
    /// routed experts 4). Present ⇒ the rec megakernel path can engage.
    var mlxExport: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case mlpOnlyLayers = "mlp_only_layers"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case mlxExport = "mlx_export"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        // rope_parameters may carry string enums (e.g. rope_type: "default");
        // decode tolerantly and only read numeric rope_theta.
        var nested: [String: Double] = [:]
        if let raw = try c.decodeIfPresent([String: AnyCodable].self, forKey: .ropeParameters) {
            for (k, v) in raw {
                if let d = v.value as? Double { nested[k] = d }
                else if let i = v.value as? Int { nested[k] = Double(i) }
            }
        }
        ropeParameters = nested.isEmpty ? nil : nested
        if let theta = nested["rope_theta"] {
            ropeTheta = Float(theta)
        } else {
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        }
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        // mlx_export mixes numeric markers with a layer-index array and a
        // string scale rule. A strict [String: Double] decode throws and the
        // whole config fails; skipping the key leaves the INT4 gate dark.
        // Keep only the numeric fields the gate reads.
        var export: [String: Double] = [:]
        if let raw = try c.decodeIfPresent([String: AnyCodable].self, forKey: .mlxExport) {
            for (k, v) in raw {
                if let d = v.value as? Double { export[k] = d }
                else if let i = v.value as? Int { export[k] = Double(i) }
            }
        }
        mlxExport = export.isEmpty ? nil : export
    }
}

// MARK: - Attention

class FlintAttention: Module {
    let args: FlintConfiguration
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray

    /// RoPE only on sliding-window layers; full-attention layers have none.
    let rope: RoPELayer?

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.args = args
        self.headDim = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)
        let dim = args.hiddenSize

        _qProj.wrappedValue = Linear(dim, args.attentionHeads * args.headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, args.kvHeads * args.headDim, bias: false)
        _oProj.wrappedValue = Linear(args.attentionHeads * args.headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        _softmaxOffset.wrappedValue = zeros([args.attentionHeads])

        if !args.layerTypes.isEmpty, args.layerTypes[layerIdx] == "sliding_attention" {
            self.rope = initializeRope(
                dims: args.headDim,
                base: args.ropeTheta,
                traditional: false,
                scalingConfig: nil,
                maxPositionEmbeddings: args.maxPositionEmbeddings
            )
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        // Sinks are only supported by the full-precision SDPA path; a quantized
        // cache (TurboQuant) silently drops them, same trade-off as GPT-OSS.
        if let qcache = cache as? QuantizedKVCacheProtocol {
            let (qKeys, qValues) = qcache.updateQuantized(keys: keys, values: values)
            let out = quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: qKeys,
                quantizedValues: qValues,
                scale: scale,
                mask: mask,
                groupSize: qcache.groupSize,
                bits: qcache.bits,
                mode: qcache.mode
            )
            return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale,
            mask: mask,
            sinks: softmaxOffset
        )

        return oProj(out.swappedAxes(1, 2).reshaped(B, L, -1))
    }
}

// MARK: - MLP / MoE

class FlintMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class FlintSparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP

    init(_ args: FlintConfiguration) {
        self.topK = args.numExpertsPerTok
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )
        _sharedExpert.wrappedValue = FlintMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
    }

    /// Sigmoid routing: top-k by score, renormalized; plus an ungated shared
    /// expert (flint.py's fallback math — the Swift port has no
    /// `expert_routing` Metal kernel).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x.asType(.float32))
        let scores0 = sigmoid(logits)
        let k = topK
        let inds = argPartition(-scores0, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = takeAlong(scores0, inds, axis: -1)
        scores = scores / scores.sum(axis: -1, keepDims: true)

        let y = switchMLP(x, inds)
        let combined = weightedExpertSum(y, scores.asType(y.dtype))
        return combined + sharedExpert(x)
    }
}

// MARK: - Decoder

class FlintDecoderLayer: Module {
    let useSliding: Bool
    @ModuleInfo(key: "self_attn") var selfAttn: FlintAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ args: FlintConfiguration, layerIdx: Int) {
        self.useSliding =
            !args.layerTypes.isEmpty && args.layerTypes[layerIdx] == "sliding_attention"
        _selfAttn.wrappedValue = FlintAttention(args, layerIdx: layerIdx)
        if args.mlpOnlyLayers.contains(layerIdx) {
            self._mlp.wrappedValue = FlintMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        } else {
            self._mlp.wrappedValue = FlintSparseMoeBlock(args)
        }
        _inputNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputNorm(x), mask: mask, cache: cache)
        let normed = postNorm(h)
        if let moe = mlp as? FlintSparseMoeBlock {
            return h + moe(normed)
        }
        return h + (mlp as! FlintMLP)(normed)
    }
}

// MARK: - Compiled decode (single token)

/// Compiled single-token decode path. The Swift eager per-op overhead is
/// ~3x Python mlx-lm's (measured on M4 Max: layer 2.9ms vs 0.8ms with equal
/// single-op kernels), so at decode batch=1 the loop is CPU-dispatch bound.
/// Compiling the two static-shape halves of every layer removes it: segment
/// A fuses norms + projections + per-head QK norms + RoPE (position passed
/// as an input tensor so the trace is not frozen to one offset); the cache
/// update + SDPA stay outside (mutating state / varying S); segment B fuses
/// the output projection, residuals, and the whole MoE block.
///
/// RoPE here is the plain non-interleaved form (base = ropeTheta, half-split
/// rotation), matching `initializeRope(traditional: false)` — verified by
/// FlintModelTests against `applyRotaryPosition`.
final class FlintCompiledLayer {
    private let layer: FlintDecoderLayer
    private let attention: FlintAttention
    private let args: FlintConfiguration
    private var segA: ((MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray))?
    private var segB: ((MLXArray, MLXArray) -> MLXArray)?
    private let invFreq: MLXArray

    init(_ layer: FlintDecoderLayer, args: FlintConfiguration) {
        self.layer = layer
        self.attention = layer.selfAttn
        self.args = args
        // inv_freq = base^(-2i/D), i in 0..<D/2 — Flint uses plain RoPE.
        let dim = args.headDim
        let exponent = (0 ..< dim / 2).map { i -> Float in
            -Float(2 * i) / Float(dim)
        }
        self.invFreq = pow(MLXArray(Float(args.ropeTheta)), MLXArray(exponent))
    }

    /// angles = pos * invFreq for a scalar position tensor.
    private func ropeAngles(_ pos: MLXArray) -> MLXArray {
        pos.asType(.float32) * invFreq
    }

    private func applyRoPE(_ x: MLXArray, angles: MLXArray) -> MLXArray {
        // x: [B, H, L, D]; non-interleaved half-split rotation.
        let d = x.dim(-1)
        let x1 = x[.ellipsis, 0 ..< d / 2]
        let x2 = x[.ellipsis, (d / 2)...]
        let cosA = MLX.cos(angles).reshaped(1, 1, 1, d / 2).asType(x.dtype)
        let sinA = MLX.sin(angles).reshaped(1, 1, 1, d / 2).asType(x.dtype)
        let r1 = x1 * cosA - x2 * sinA
        let r2 = x2 * cosA + x1 * sinA
        return concatenated([r1, r2], axis: -1)
    }

    private func buildSegments() {
        let attn = attention
        let dim = args.headDim
        let sliding = layer.useSliding

        // A(x, pos) -> (q, k, v), all shapes static at L=1.
        segA = compile { [self] (x: MLXArray, pos: MLXArray) in
            let (B, L) = (x.dim(0), x.dim(1))
            var q = attn.qNorm(attn.qProj(x).reshaped(B, L, args.attentionHeads, -1))
                .transposed(0, 2, 1, 3)
            var k = attn.kNorm(attn.kProj(x).reshaped(B, L, args.kvHeads, -1))
                .transposed(0, 2, 1, 3)
            let v = attn.vProj(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
            if sliding {
                let angles = ropeAngles(pos)
                q = applyRoPE(q, angles: angles)
                k = applyRoPE(k, angles: angles)
            }
            _ = dim
            return (q, k, v)
        }

        // B(attnOut, residual) -> nextX.
        let layer = self.layer
        segB = compile { (attnOut: MLXArray, residual: MLXArray) in
            let h = residual + layer.selfAttn.oProj(attnOut.swappedAxes(1, 2).reshaped(residual.dim(0), residual.dim(1), -1))
            let normed = layer.postNorm(h)
            if let moe = layer.mlp as? FlintSparseMoeBlock {
                return h + moe(normed)
            }
            return h + (layer.mlp as! FlintMLP)(normed)
        }
    }

    /// One compiled decode step through this layer. `pos` is the absolute
    /// position of the token being decoded (scalar tensor).
    func callAsFunction(_ x: MLXArray, pos: MLXArray, cache: KVCache?) -> MLXArray {
        if segA == nil { buildSegments() }
        var (q, k, v) = segA!(x, pos)
        // No mask at L=1: every cached position is in the past, and the
        // rotating cache bounds sliding layers to the window.
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v,
            scale: pow(Float(args.headDim), -0.5),
            mask: .none,
            sinks: attention.softmaxOffset
        )
        return segB!(out, x)
    }
}



public class FlintModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layerTypes: [String]
    let layers: [FlintDecoderLayer]
    let windowSize: Int
    let slidingAttentionIndex: Int
    let fullAttentionIndex: Int

    /// Lazily-built compiled single-token decode path (nil until first use).
    nonisolated(unsafe) private(set) var compiledLayers: [FlintCompiledLayer]?
    let config: FlintConfiguration

    init(_ config: FlintConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.layerTypes = config.layerTypes
        self.layers = (0 ..< config.hiddenLayers).map { FlintDecoderLayer(config, layerIdx: $0) }
        self.windowSize = config.slidingWindow
        self.slidingAttentionIndex = layerTypes.firstIndex(of: "sliding_attention") ?? 0
        self.fullAttentionIndex = layerTypes.firstIndex(of: "full_attention") ?? 0
    }

    /// Single-token compiled decode: (tokenIds [1,1], position offset pair) -> logits-side hidden.
    /// Returns nil when the fast path is not applicable (multi-token prefill).
    func compiledDecode(
        _ inputs: MLXArray, cache: [KVCache], slidingPos: MLXArray, fullPos: MLXArray
    ) -> MLXArray? {
        guard inputs.dim(1) == 1, cache.count == layers.count else { return nil }
        if compiledLayers == nil {
            compiledLayers = layers.map { FlintCompiledLayer($0, args: config) }
        }
        var x = embedTokens(inputs)
        for (i, cl) in compiledLayers!.enumerated() {
            let pos = layerTypes[i] == "full_attention" ? fullPos : slidingPos
            x = cl(x, pos: pos, cache: cache[i])
        }
        return norm(x)
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        var x = inputEmbeddings ?? embedTokens(inputs)
        let caches: [KVCache?] = cache ?? [KVCache?](repeating: nil, count: layers.count)

        let seqLen = x.dim(1)
        var fullMask: MLXFast.ScaledDotProductAttentionMaskMode?
        var slidingMask: MLXFast.ScaledDotProductAttentionMaskMode?

        for (i, layer) in layers.enumerated() {
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
            if layerTypes[i] == "full_attention" {
                if fullMask == nil {
                    fullMask = makeAttentionMask(
                        n: seqLen, cache: caches[fullAttentionIndex], windowSize: nil)
                }
                maskMode = fullMask!
            } else {
                if slidingMask == nil {
                    slidingMask = makeAttentionMask(
                        n: seqLen, cache: caches[slidingAttentionIndex], windowSize: windowSize)
                }
                maskMode = slidingMask!
            }
            x = layer(x, mask: maskMode, cache: caches[i])
        }

        return norm(x)
    }
}

public class FlintModel: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    static let _compiledHitLog = OSAllocatedUnfairLock(initialState: false)

    /// Per-layer compiled segments measured SLOWER than eager (55 vs 51 ms per
    /// token): 98 small compiled calls pay ~0.3 ms Swift wrapper overhead each,
    /// eating the fusion win (a standalone compiled op-chain is 5.4x faster, so
    /// the mechanism works — the segmentation granularity is wrong). Disabled
    /// until the Qwen3Next-style whole-model segments (KV state as updatable
    /// inputs/outputs) land. Opt in with NOVA_FLINT_COMPILED_DECODE=1.
    static let compiledDecodeEnabled =
        ProcessInfo.processInfo.environment["NOVA_FLINT_COMPILED_DECODE"] == "1"

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: FlintModelInner
    let configuration: FlintConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] { model.layers }

    /// Flint emits tool calls in the GLM4 dialect:
    /// `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`.
    /// Declared as a class member (not an extension) so it wins over
    /// `LanguageModel`'s nil default when the factory reads the convention.
    public var toolCallFormat: ToolCallFormat? { .glm4 }

    public init(_ args: FlintConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = FlintModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Pack during prefill so the first decode token is not also the
        // first time the banks are built. Nil when this is not official INT4.
        if cache != nil {
            _ = recKernelIfAvailable()
        }
        if let rec = recDecodeLogits(inputs, cache: cache) {
            return rec
        }
        // Compiled single-token fast path: the eager loop is CPU-dispatch
        // bound in Swift (~3x Python mlx-lm). Only for plain caches — the
        // quantized-cache path needs its own SDPA and stays eager.
        // Gated off (NOVA_FLINT_COMPILED_DECODE). Rec decode above is the
        // official-INT4 path; this stays as the unused experiment.
        if Self.compiledDecodeEnabled, inputs.dim(1) == 1, let cache,
            cache.count == model.layers.count, !(cache[0] is QuantizedKVCacheProtocol)
        {
            if !FlintModel._compiledHitLog.withLock({ $0 }) {
                FlintModel._compiledHitLog.withLock { $0 = true }
                NovaMLXLog.info("[Flint] compiled decode path ACTIVE")
            }
            let pos = MLXArray(Int32(max(0, cache[model.slidingAttentionIndex].offset)))
            if let hidden = model.compiledDecode(
                inputs, cache: cache, slidingPos: pos, fullPos: pos)
            {
                if let lmHead {
                    return lmHead(hidden)
                }
                return model.embedTokens.asLinear(hidden)
            }
        }
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    // MARK: - rec megakernel gate (official INT4 only)

    nonisolated(unsafe) private(set) var recKernel: FlintRecKernel?
    nonisolated(unsafe) private var recKernelTried = false
    let recLock = NSLock()

    nonisolated(unsafe) private var recSession: FlintRecDecodeSession?
    nonisolated(unsafe) private var recDecodeDisabled = false
    nonisolated(unsafe) private(set) var recDecodeSteps = 0
    nonisolated(unsafe) private var recQuantFallbackLogged = false
    nonisolated(unsafe) private var recIngestFailedCache: ObjectIdentifier?
    nonisolated(unsafe) private var recActiveLogged = false
    /// Non-proxy caches only. Proxies keep this on themselves so two live
    /// sequences do not steal each other's handoff.
    nonisolated(unsafe) private var recSawSingleToken: ObjectIdentifier?

    /// True when the checkpoint carries the official RecursiveSI/flint-int4
    /// export markers (`bits` 4, `embedding_bits` 8, `routed_expert_bits` 4).
    /// Group size lives on the checkpoint's `quantization` block, not in
    /// `mlx_export` — requiring it here false-negatives the real config.
    public var isOfficialInt4: Bool {
        guard let e = configuration.mlxExport else { return false }
        return e["bits"] == 4 && e["embedding_bits"] == 8 && e["routed_expert_bits"] == 4
    }

    /// Build (once) the rec megakernel from the loaded weights. Returns nil
    /// for non-official layouts — the eager path stays authoritative.
    public func recKernelIfAvailable() -> FlintRecKernel? {
        recLock.lock()
        defer { recLock.unlock() }
        if recKernelTried { return recKernel }
        recKernelTried = true
        guard isOfficialInt4 else {
            NovaMLXLog.info("[Flint] rec megakernel off: not the official INT4 export")
            return nil
        }
        do {
            let packed = try FlintRecPacker.pack(self)
            let kernel = FlintRecKernel(packed: packed)
            recKernel = kernel
            NovaMLXLog.info("[Flint] rec megakernel armed (official INT4)")
            return kernel
        } catch {
            NovaMLXLog.error("[Flint] rec pack failed, staying eager: \(error)")
            return nil
        }
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let raw = try model.layerTypes.map { layerType in
            try makeHybridAttentionKVCache(
                parameters: parameters,
                slidingWindow: configuration.slidingWindow,
                usesSlidingWindow: layerType != "full_attention")
        }
        // Quantized KV cannot be copied into the megakernel's bf16 rings.
        // Leave those caches unwrapped so eager SDPA keeps its quant path.
        guard isOfficialInt4, parameters?.kvBits == nil else { return raw }
        guard !raw.contains(where: { $0 is QuantizedKVCacheProtocol }) else { return raw }
        return raw.enumerated().map { index, cache in
            FlintRecCacheProxy(inner: cache, layer: index, model: self)
        }
    }

    /// Prefill stays eager (rec is decode-only, same split as `QATSession`).
    /// After one eager single-token forward on a cache, further single-token
    /// decode — and any later multi-token verify on that same cache — runs
    /// `FlintRecKernel.forward`. The first one-token call stays eager because
    /// balanced prefill reserves the last prompt token as its own forward,
    /// and the fused scheduler's tail can be one token too. A failure before
    /// a token is committed falls back to eager; a failure after that
    /// disables rec for this model so the next step does not mix the two KV
    /// stores.
    private func recDecodeLogits(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray? {
        guard !recDecodeDisabled, let kernel = recKernel else { return nil }
        guard inputs.ndim == 2, inputs.dim(0) == 1 else { return nil }
        let seq = inputs.dim(1)
        guard seq >= 1, let cache, cache.count == model.layers.count else { return nil }
        if cache[0] is QuantizedKVCacheProtocol {
            if !recQuantFallbackLogged {
                recQuantFallbackLogged = true
                NovaMLXLog.info("[Flint] rec decode off: quantized KV cache")
            }
            return nil
        }
        guard let anchor = cache[0] as? AnyObject else { return nil }
        let cacheID = ObjectIdentifier(anchor)
        if recIngestFailedCache == cacheID { return nil }

        let continuing = recSessionMatches(cache)
        let offset = FlintRecCacheBridge.concrete(cache[0]).offset
        let sawSingleToken = (cache[0] as? FlintRecCacheProxy)?.sawSingleToken
            ?? (recSawSingleToken == cacheID)
        guard Self.recHandoffReady(
            seq: seq, cacheOffset: offset, continuing: continuing,
            sawSingleToken: sawSingleToken)
        else {
            if seq == 1 { noteSawSingleToken(cache, cacheID: cacheID) }
            return nil
        }

        let session: FlintRecDecodeSession
        if continuing, let existing = recSession {
            session = existing
        } else {
            do {
                session = try ingestRecSession(cache: cache, kernel: kernel, cacheID: cacheID)
            } catch {
                recIngestFailedCache = cacheID
                NovaMLXLog.error("[Flint] rec KV ingest failed, staying eager: \(error)")
                return nil
            }
        }

        let tokens = inputs.asType(.int32).asArray(Int32.self)
        guard tokens.count == seq else {
            NovaMLXLog.error("[Flint] rec decode off: token read \(tokens.count) != \(seq)")
            return nil
        }
        var rows: [MLXArray] = []
        rows.reserveCapacity(seq)
        var committed = 0
        do {
            for id in tokens {
                let flat = try kernel.forward(state: session.state, token: Int(id))
                guard flat.dim(0) == vocabularySize else {
                    session.state.position = max(0, session.state.position - 1)
                    throw NovaMLXError.inferenceFailed(
                        "rec logits \(flat.dim(0)) != vocab \(vocabularySize)")
                }
                noteRecStep(cache, session: session)
                committed += 1
                rows.append(flat)
                if !recActiveLogged {
                    recActiveLogged = true
                    NovaMLXLog.info("[Flint] rec megakernel decode ACTIVE")
                }
                recDecodeSteps += 1
            }
        } catch {
            undoRecSteps(cache, session: session, n: committed)
            recDecodeDisabled = true
            invalidateRecSession()
            NovaMLXLog.error("[Flint] rec decode failed, falling back to eager: \(error)")
            return nil
        }
        if rows.count == 1 {
            return rows[0].reshaped(1, 1, vocabularySize)
        }
        return stacked(rows, axis: 0).reshaped(1, rows.count, vocabularySize)
    }

    /// True when this forward should run on the megakernel.
    /// Wide chunks are prefill. The first one-token call on a cache stays
    /// eager: it may be the last prompt token. The following one-token call
    /// is decode and hands off, with that token already in the MLX cache.
    static func recHandoffReady(
        seq: Int, cacheOffset: Int, continuing: Bool, sawSingleToken: Bool
    ) -> Bool {
        if continuing { return true }
        guard seq == 1, cacheOffset > 0, sawSingleToken else { return false }
        return true
    }

    private func noteSawSingleToken(_ cache: [KVCache], cacheID: ObjectIdentifier) {
        var wrapped = false
        for case let proxy as FlintRecCacheProxy in cache {
            proxy.sawSingleToken = true
            wrapped = true
        }
        if !wrapped { recSawSingleToken = cacheID }
    }

    private func recSessionMatches(_ cache: [KVCache]) -> Bool {
        guard let session = recSession, let anchor = cache.first as? AnyObject else { return false }
        guard session.cacheID == ObjectIdentifier(anchor) else { return false }
        if let proxy = cache[0] as? FlintRecCacheProxy {
            return proxy.inner.offset == session.mlxOffset && proxy.recAhead == session.recAhead
        }
        return cache[0].offset == session.mlxOffset
    }

    private func ingestRecSession(
        cache: [KVCache], kernel: FlintRecKernel, cacheID: ObjectIdentifier
    ) throws -> FlintRecDecodeSession {
        let prompt = FlintRecCacheBridge.concrete(cache[0]).offset
        let window = configuration.slidingWindow
        for (index, layer) in cache.enumerated() {
            let concrete = FlintRecCacheBridge.concrete(layer)
            guard concrete.offset == prompt else {
                throw NovaMLXError.inferenceFailed(
                    "rec ingest layer \(index) offset \(concrete.offset) != \(prompt)")
            }
            if let rotating = concrete as? RotatingKVCache, rotating.maxSize != window {
                throw NovaMLXError.inferenceFailed(
                    "rec sliding window is \(window), cache is \(rotating.maxSize.map(String.init) ?? "nil")")
            }
        }
        let state = kernel.makeKVState(capacity: max(prompt + 1, 1))
        try kernel.ensureCapacity(state, tokens: max(prompt + 1, 1))
        if prompt > 0 {
            for (index, layer) in cache.enumerated() {
                guard let pair = FlintRecCacheBridge.orderedPair(from: layer) else {
                    throw NovaMLXError.inferenceFailed("rec ingest layer \(index) has no KV")
                }
                let global = model.layerTypes[index] == "full_attention"
                let seq = pair.0.ndim == 4 ? pair.0.dim(2) : pair.0.dim(-2)
                let expect = global ? prompt : min(prompt, window)
                guard seq == expect else {
                    throw NovaMLXError.inferenceFailed(
                        "rec ingest layer \(index) length \(seq) != \(expect)")
                }
                try kernel.writeKV(
                    state: state, layer: index, tokenOffset: 0, keys: pair.0, values: pair.1)
            }
        }
        state.position = prompt
        state.initial = prompt
        let proxies = cache.count == cache.compactMap({ $0 as? FlintRecCacheProxy }).count
            ? cache.compactMap { $0 as? FlintRecCacheProxy } : []
        let session = FlintRecDecodeSession(
            cacheID: cacheID, state: state, mlxOffset: prompt, proxies: proxies)
        recSession = session
        return session
    }

    private func noteRecStep(_ cache: [KVCache], session: FlintRecDecodeSession) {
        for layer in cache {
            (layer as? FlintRecCacheProxy)?.noteRecStep()
        }
        session.recAhead += 1
    }

    private func undoRecSteps(_ cache: [KVCache], session: FlintRecDecodeSession, n: Int) {
        guard n > 0 else { return }
        for layer in cache {
            (layer as? FlintRecCacheProxy)?.undoRecSteps(n)
        }
        session.state.position = max(session.mlxOffset, session.state.position - n)
        session.recAhead = max(0, session.recAhead - n)
    }

    /// Eager prefill wrote into the MLX caches. Drop the rec snapshot and
    /// put proxy offsets back on the inner cache.
    func invalidateRecSession() {
        recSession?.proxies.forEach { $0.resetRecAhead() }
        recSession = nil
    }

    /// The caller trimmed into the eager prefix. Proxies update themselves.
    func abandonRecSession() {
        recSession = nil
    }

    /// Speculative verify rejected a suffix. The kernel writes KV before the
    /// caller knows which tokens were kept, so rewind the position counter.
    func rewindRecSession(by n: Int) {
        guard n > 0, let session = recSession else { return }
        session.state.position = max(session.mlxOffset, session.state.position - n)
        session.recAhead = max(0, session.recAhead - n)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights
        if configuration.tieWordEmbeddings {
            w["lm_head.weight"] = nil
        }

        // RecursiveSI/flint-int4 stores packed experts as uint8; view them as
        // uint32 so the quantize(model:) pass pairs them with their scales.
        for (k, v) in w where v.dtype == .uint8 {
            let name = k.hasSuffix(".weight") ? String(k.dropLast(7)) : k
            if w["\(name).scales"] != nil {
                w[k] = v.view(dtype: .uint32)
            }
        }

        // Layer 0 is dense in Flint (mlp_only_layers), so probe per layer
        // instead of using a layer-0 guard like Qwen3Next does.

        for l in 0 ..< configuration.hiddenLayers {
            guard !configuration.mlpOnlyLayers.contains(l) else { continue }
            let prefix = "model.layers.\(l).mlp"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                // Stack only when the full expert set is present; a partial
                // set means the checkpoint is not per-expert sharded here.
                guard w["\(prefix).experts.0.\(n).weight"] != nil,
                    w["\(prefix).experts.\(configuration.numExperts - 1).\(n).weight"] != nil
                else { continue }
                let toJoin = (0 ..< configuration.numExperts).map { e in
                    w.removeValue(forKey: "\(prefix).experts.\(e).\(n).weight")!
                }
                w["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
            }
        }

        return w
    }
}
