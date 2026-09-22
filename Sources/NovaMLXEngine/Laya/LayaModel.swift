import Foundation
import MLX
import MLXNN
import NovaMLXCore

struct LayaEncoderConfig: Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var modelType: String
    var normEps: Float
    var normBias: Bool
    var attentionBias: Bool
    var mlpBias: Bool
    var localAttention: Int
    var layerTypes: [String]
    var globalRopeTheta: Float
    var localRopeTheta: Float
    var maxPositionEmbeddings: Int

    var headDim: Int { hiddenSize / numAttentionHeads }

    func ropeBase(_ kind: String) -> Float {
        kind == "full_attention" ? globalRopeTheta : localRopeTheta
    }

    static func parse(_ json: [String: Any]) throws -> LayaEncoderConfig {
        func int(_ key: String) throws -> Int {
            if let v = json[key] as? Int { return v }
            if let v = json[key] as? Double { return Int(v) }
            throw NovaMLXError.configurationError("Laya encoder missing \(key)")
        }
        func float(_ key: String, _ fallback: Float) -> Float {
            if let v = json[key] as? Double { return Float(v) }
            if let v = json[key] as? Int { return Float(v) }
            return fallback
        }
        func bool(_ key: String, _ fallback: Bool) -> Bool {
            json[key] as? Bool ?? fallback
        }
        let layers = try int("num_hidden_layers")
        let every = (json["global_attn_every_n_layers"] as? Int) ?? 3
        let types = (json["layer_types"] as? [String]) ?? (0..<layers).map { i in
            i % every == 0 ? "full_attention" : "sliding_attention"
        }
        let kind = (json["model_type"] as? String) ?? "modernbert"
        guard kind == "modernbert" || kind == "mmbert" else {
            throw NovaMLXError.unsupportedModel("Laya encoder \(kind) is not modernbert/mmbert")
        }
        return LayaEncoderConfig(
            vocabSize: try int("vocab_size"),
            hiddenSize: try int("hidden_size"),
            intermediateSize: try int("intermediate_size"),
            numHiddenLayers: layers,
            numAttentionHeads: try int("num_attention_heads"),
            modelType: kind,
            normEps: float("norm_eps", 1e-5),
            normBias: bool("norm_bias", false),
            attentionBias: bool("attention_bias", false),
            mlpBias: bool("mlp_bias", false),
            localAttention: (json["local_attention"] as? Int) ?? 128,
            layerTypes: types,
            globalRopeTheta: float("global_rope_theta", 160_000),
            localRopeTheta: float("local_rope_theta", 10_000),
            maxPositionEmbeddings: (json["max_position_embeddings"] as? Int) ?? 8192
        )
    }
}

final class LayaEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ cfg: LayaEncoderConfig) {
        self._tokEmbeddings.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize)
        self._norm.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.normEps, bias: cfg.normBias)
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        norm(tokEmbeddings(ids))
    }
}

final class LayaEncoderAttention: Module {
    let numHeads: Int
    let headDim: Int
    let ropeBase: Float
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ cfg: LayaEncoderConfig, kind: String) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.headDim
        ropeBase = cfg.ropeBase(kind)
        self._wqkv.wrappedValue = Linear(cfg.hiddenSize, 3 * cfg.hiddenSize, bias: cfg.attentionBias)
        self._wo.wrappedValue = Linear(cfg.hiddenSize, cfg.hiddenSize, bias: cfg.attentionBias)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let length = x.dim(1)
        let qkv = wqkv(x).reshaped([b, length, 3, numHeads, headDim])
        func head(_ index: Int) -> MLXArray {
            qkv[0..., 0..., index, 0..., 0...].transposed(0, 2, 1, 3)
        }
        var q = head(0).asType(.float32)
        var k = head(1).asType(.float32)
        let v = head(2).asType(.float32)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeBase, scale: 1, offset: 0)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeBase, scale: 1, offset: 0)
        let scale = 1 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask
        ).asType(x.dtype)
        return wo(out.transposed(0, 2, 1, 3).reshaped([b, length, numHeads * headDim]))
    }
}

final class LayaEncoderMLP: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ cfg: LayaEncoderConfig) {
        self._wi.wrappedValue = Linear(cfg.hiddenSize, 2 * cfg.intermediateSize, bias: cfg.mlpBias)
        self._wo.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: cfg.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = wi(x).split(parts: 2, axis: -1)
        return wo(gelu(parts[0]) * parts[1])
    }
}

final class LayaEncoderLayer: Module {
    let attentionType: String
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm?
    @ModuleInfo(key: "attn") var attn: LayaEncoderAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: LayaEncoderMLP

    init(_ cfg: LayaEncoderConfig, index: Int) {
        attentionType = cfg.layerTypes[index]
        self._attnNorm.wrappedValue = index == 0
            ? nil
            : LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.normEps, bias: cfg.normBias)
        self._attn.wrappedValue = LayaEncoderAttention(cfg, kind: attentionType)
        self._mlpNorm.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.normEps, bias: cfg.normBias)
        self._mlp.wrappedValue = LayaEncoderMLP(cfg)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let normed = attnNorm?(x) ?? x
        var y = x + attn(normed, mask: mask)
        y = y + mlp(mlpNorm(y))
        return y
    }
}

final class LayaModernBert: Module {
    let config: LayaEncoderConfig
    @ModuleInfo(key: "embeddings") var embeddings: LayaEmbeddings
    @ModuleInfo(key: "layers") var layers: [LayaEncoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ cfg: LayaEncoderConfig) {
        config = cfg
        self._embeddings.wrappedValue = LayaEmbeddings(cfg)
        self._layers.wrappedValue = (0..<cfg.numHiddenLayers).map { LayaEncoderLayer(cfg, index: $0) }
        self._finalNorm.wrappedValue = LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.normEps, bias: cfg.normBias)
    }

    func callAsFunction(_ ids: MLXArray, attentionMask: MLXArray) -> MLXArray {
        var x = embeddings(ids)
        let masks = LayaMasks.make(attentionMask, window: config.localAttention)
        for layer in layers {
            let mask = layer.attentionType == "full_attention" ? masks.full : masks.local
            x = layer(x, mask: mask)
        }
        return finalNorm(x)
    }
}

enum LayaMasks {
    static func make(_ attentionMask: MLXArray, window: Int) -> (full: MLXArray, local: MLXArray) {
        let valid = attentionMask.asType(.bool)
        let batch = valid.dim(0)
        let length = valid.dim(1)
        let flags = valid.asType(.int32).asArray(Int32.self)
        let radius = window / 2
        var full = [Int32](repeating: 0, count: batch * length)
        var local = [Int32](repeating: 0, count: batch * length * length)
        for b in 0..<batch {
            for q in 0..<length {
                let queryValid = flags[b * length + q] != 0
                let queryPad = !queryValid
                if queryValid {
                    full[b * length + q] = 1
                }
                for k in 0..<length where flags[b * length + k] != 0 && (abs(q - k) <= radius || queryPad) {
                    local[(b * length + q) * length + k] = 1
                }
            }
        }
        // Bool masks match the MLX port. An additive -1e9 mask trips
        // scaled_dot_product_attention and kills the process.
        return (
            MLXArray(full).reshaped([batch, 1, 1, length]).asType(.bool),
            MLXArray(local).reshaped([batch, 1, length, length]).asType(.bool)
        )
    }
}

final class LayaHeadAttention: Module {
    let numHeads: Int
    let headDim: Int
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dims: Int) {
        numHeads = max(1, dims / 64)
        headDim = dims / numHeads
        self._inProj.wrappedValue = Linear(dims, 3 * dims, bias: true)
        self._outProj.wrappedValue = Linear(dims, dims, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let length = x.dim(1)
        let qkv = inProj(x).reshaped([b, length, 3, numHeads, headDim])
        func head(_ index: Int) -> MLXArray {
            qkv[0..., 0..., index, 0..., 0...].transposed(0, 2, 1, 3)
        }
        let scale = 1 / sqrt(Float(headDim))
        let out = MLXFast.scaledDotProductAttention(
            queries: head(0).asType(.float32),
            keys: head(1).asType(.float32),
            values: head(2).asType(.float32),
            scale: scale,
            mask: mask
        ).asType(x.dtype)
        return outProj(out.transposed(0, 2, 1, 3).reshaped([b, length, numHeads * headDim]))
    }
}

final class LayaHeadLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: LayaHeadAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dims: Int) {
        self._selfAttn.wrappedValue = LayaHeadAttention(dims: dims)
        self._norm1.wrappedValue = LayerNorm(dimensions: dims)
        self._norm2.wrappedValue = LayerNorm(dimensions: dims)
        self._linear1.wrappedValue = Linear(dims, 4 * dims)
        self._linear2.wrappedValue = Linear(4 * dims, dims)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var y = x + selfAttn(norm1(x), mask: mask)
        y = y + linear2(relu(linear1(norm2(y))))
        return y
    }
}

final class LayaDecisionHead: Module {
    @ModuleInfo(key: "layers") var layers: [LayaHeadLayer]

    init(dims: Int, count: Int) {
        self._layers.wrappedValue = (0..<count).map { _ in LayaHeadLayer(dims: dims) }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var y = x
        for layer in layers {
            y = layer(y, mask: mask)
        }
        return y
    }
}

final class LayaDecisionModel: Module {
    let hidden: Int
    @ModuleInfo(key: "encoder") var encoder: LayaModernBert
    @ModuleInfo(key: "head") var head: LayaDecisionHead
    @ModuleInfo(key: "type_emb") var typeEmb: Embedding
    @ModuleInfo(key: "scorer") var scorer: Sequential
    @ModuleInfo(key: "act_head") var actHead: Sequential
    @ModuleInfo(key: "temperature") var temperature: MLXArray

    init(encoderConfig: LayaEncoderConfig, headLayers: Int, actClasses: Int) {
        let hidden = encoderConfig.hiddenSize
        self.hidden = hidden
        self._encoder.wrappedValue = LayaModernBert(encoderConfig)
        self._head.wrappedValue = LayaDecisionHead(dims: hidden, count: headLayers)
        self._typeEmb.wrappedValue = Embedding(embeddingCount: 3, dimensions: hidden)
        self._scorer.wrappedValue = Sequential(layers: [
            LayerNorm(dimensions: hidden),
            Linear(hidden, hidden),
            GELU(),
            Linear(hidden, 1),
        ])
        self._actHead.wrappedValue = Sequential(layers: [
            Linear(hidden + 4, 256),
            GELU(),
            Linear(256, actClasses),
        ])
        self._temperature.wrappedValue = MLXArray.ones([3])
    }

    func callAsFunction(
        inputIds: MLXArray,
        attentionMask: MLXArray,
        markerPos: MLXArray,
        markerMask: MLXArray,
        qtype: MLXArray
    ) -> (MLXArray, MLXArray) {
        var h = encoder(inputIds, attentionMask: attentionMask)
        let type = typeEmb(qtype)
        h = h + type.expandedDimensions(axis: 1)
        let headMask = LayaMasks.make(attentionMask, window: 1).full
        h = head(h, mask: headMask)
        let batch = h.dim(0)
        let markerCount = markerPos.dim(1)
        let positions = MLX.maximum(markerPos, MLXArray(0))
        var gathered: [MLXArray] = []
        for b in 0..<batch {
            var row: [MLXArray] = []
            let pos = positions[b].asArray(Int32.self)
            for m in 0..<markerCount {
                row.append(h[b, Int(pos[m])])
            }
            gathered.append(MLX.stacked(row, axis: 0))
        }
        let markers = MLX.stacked(gathered, axis: 0)
        var logits = scorer(markers).squeezed(axis: -1).asType(.float32)
        let keep = markerMask.asType(.bool)
        logits = MLX.which(keep, logits, MLXArray(-1e4))
        let probs = softmax(logits, axis: -1)
        let k = MLX.maximum(keep.sum(axis: -1), MLXArray(2)).asType(.float32)
        let entropy = -(probs * log(MLX.maximum(probs, MLXArray(1e-9)))).sum(axis: -1) / log(k)
        let sorted = MLX.sorted(probs, axis: -1)
        let last = sorted.dim(1) - 1
        let top1 = sorted[0..., last]
        let top0 = sorted[0..., last - 1]
        let features = MLX.stacked([top1, top1 - top0, entropy, k / 255], axis: -1)
        let pooled = MLX.concatenated([h[0..., 0].asType(.float32), features], axis: -1)
        let action = actHead(pooled).asType(.float32)
        return (logits, action)
    }
}
