import Foundation

// Port of mlx-dspark `config.py` — DSpark drafter config decoder.
// Handles the vLLM `speculators` packaging (translated into the DeepSpec schema)
// and DeepSpec-native flat configs.

public enum DSparkError: Error, LocalizedError {
    case invalidConfig(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let message): return "DSpark config: \(message)"
        }
    }
}

/// The nested `transformer_layer_config` of a vLLM-speculators DSpark head —
/// a standard Qwen3-style backbone config.
public struct DSparkTransformerLayerConfig: Sendable, Equatable {
    public var modelType: String
    public var hiddenSize: Int
    public var vocabSize: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var attentionBias: Bool
    public var ropeTheta: Float
    public var partialRotaryFactor: Float
    public var ropeType: String
    public var slidingWindow: Int?
    public var useSlidingWindow: Bool
    public var layerTypes: [String]

    public init(
        modelType: String,
        hiddenSize: Int,
        vocabSize: Int,
        numHiddenLayers: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        attentionBias: Bool,
        ropeTheta: Float,
        partialRotaryFactor: Float,
        ropeType: String,
        slidingWindow: Int?,
        useSlidingWindow: Bool,
        layerTypes: [String]
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.attentionBias = attentionBias
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeType = ropeType
        self.slidingWindow = slidingWindow
        self.useSlidingWindow = useSlidingWindow
        self.layerTypes = layerTypes
    }
}

public struct DSparkConfig: Sendable, Equatable {
    public enum Family: String, Sendable, Equatable {
        case gemma4
        case qwen3
    }

    public var family: Family
    public var hiddenSize: Int
    public var vocabSize: Int
    /// Reduced draft vocabulary (EAGLE-3 style, vLLM-speculators heads). nil = full vocab.
    public var draftVocabSize: Int?
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numGlobalKeyValueHeads: Int
    public var headDim: Int
    public var globalHeadDim: Int
    public var attentionKEqV: Bool
    public var attentionBias: Bool
    public var attentionSink: Bool
    public var ropeTheta: Float
    public var partialRotaryFactor: Float
    public var ropeType: String
    /// DFlash-lineage heads attend causally over a sliding window; DeepSpec heads
    /// are bidirectional full-attention blocks.
    public var causalBlock: Bool
    public var slidingWindow: Int?
    public var hasOwnLmHead: Bool
    public var hasOwnEmbed: Bool
    public var blockSize: Int
    public var maskTokenId: Int
    /// Which block slot holds the first prediction. DeepSpec heads sample the anchor
    /// slot (0); DFlash-derived speculators heads reserve slot 0 as a pure anchor (1).
    public var logitsStart: Int
    public var targetLayerIds: [Int]
    public var numTargetLayers: Int
    public var markovRank: Int
    public var markovHeadType: String
    public var enableConfidenceHead: Bool
    public var confidenceHeadWithMarkov: Bool
    public var finalLogitSoftcapping: Float?
    public var padTokenId: Int
    public var mlpActivation: String
    public var normStyle: String
    public var useVNorm: Bool
    /// The speculators backbone config this head was translated from
    /// (nil for DeepSpec-native configs).
    public var transformerLayerConfig: DSparkTransformerLayerConfig?

    /// Max draftable tokens per block — the slots carrying predictions.
    public var kdraft: Int { blockSize - logitsStart }
    /// Output width of lm_head / markov head — the draft vocabulary.
    public var outVocabSize: Int { draftVocabSize ?? vocabSize }

    public init(
        family: Family = .gemma4,
        hiddenSize: Int = 3840,
        vocabSize: Int = 262_144,
        draftVocabSize: Int? = nil,
        numHiddenLayers: Int = 5,
        intermediateSize: Int = 15_360,
        rmsNormEps: Float = 1e-6,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 8,
        numGlobalKeyValueHeads: Int = 1,
        headDim: Int = 256,
        globalHeadDim: Int = 512,
        attentionKEqV: Bool = true,
        attentionBias: Bool = false,
        attentionSink: Bool = false,
        ropeTheta: Float = 1_000_000,
        partialRotaryFactor: Float = 0.25,
        ropeType: String = "proportional",
        causalBlock: Bool = false,
        slidingWindow: Int? = nil,
        hasOwnLmHead: Bool = true,
        hasOwnEmbed: Bool = true,
        blockSize: Int = 7,
        maskTokenId: Int = 4,
        logitsStart: Int = 0,
        targetLayerIds: [Int] = [5, 17, 29, 41, 46],
        numTargetLayers: Int = 48,
        markovRank: Int = 256,
        markovHeadType: String = "vanilla",
        enableConfidenceHead: Bool = true,
        confidenceHeadWithMarkov: Bool = true,
        finalLogitSoftcapping: Float? = 30.0,
        padTokenId: Int = 0,
        mlpActivation: String = "gelu_tanh",
        normStyle: String = "gemma",
        useVNorm: Bool = true,
        transformerLayerConfig: DSparkTransformerLayerConfig? = nil
    ) {
        self.family = family
        self.hiddenSize = hiddenSize
        self.vocabSize = vocabSize
        self.draftVocabSize = draftVocabSize
        self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numGlobalKeyValueHeads = numGlobalKeyValueHeads
        self.headDim = headDim
        self.globalHeadDim = globalHeadDim
        self.attentionKEqV = attentionKEqV
        self.attentionBias = attentionBias
        self.attentionSink = attentionSink
        self.ropeTheta = ropeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeType = ropeType
        self.causalBlock = causalBlock
        self.slidingWindow = slidingWindow
        self.hasOwnLmHead = hasOwnLmHead
        self.hasOwnEmbed = hasOwnEmbed
        self.blockSize = blockSize
        self.maskTokenId = maskTokenId
        self.logitsStart = logitsStart
        self.targetLayerIds = targetLayerIds
        self.numTargetLayers = numTargetLayers
        self.markovRank = markovRank
        self.markovHeadType = markovHeadType
        self.enableConfidenceHead = enableConfidenceHead
        self.confidenceHeadWithMarkov = confidenceHeadWithMarkov
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.padTokenId = padTokenId
        self.mlpActivation = mlpActivation
        self.normStyle = normStyle
        self.useVNorm = useVNorm
        self.transformerLayerConfig = transformerLayerConfig
    }

    /// True when the config.json declares a DSpark draft head by any of the known
    /// signals (speculators model type / algorithm, model_type, architectures).
    public static func isDSparkDraftConfig(jsonObject json: [String: Any]) -> Bool {
        if let modelType = json["speculators_model_type"] as? String,
            modelType.lowercased() == "dspark"
        {
            return true
        }
        if let spec = json["speculators_config"] as? [String: Any],
            let algorithm = spec["algorithm"] as? String,
            algorithm.lowercased() == "dspark"
        {
            return true
        }
        if let modelType = json["model_type"] as? String,
            modelType.lowercased().contains("dspark")
        {
            return true
        }
        let architectures = (json["architectures"] as? [String]) ?? []
        return architectures.contains { $0.lowercased().contains("dspark") }
    }

    /// True when `modelDir`'s config.json is a DSpark drafter, not a chat backbone.
    public static func isDSparkDraftConfig(at modelDir: URL) -> Bool {
        let config = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: config),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        return isDSparkDraftConfig(jsonObject: json)
    }

    public static func fromJSON(_ data: Data) throws -> DSparkConfig {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DSparkError.invalidConfig("not a JSON object")
        }
        return try fromJSONObject(json)
    }

    public static func fromJSONObject(_ object: [String: Any]) throws -> DSparkConfig {
        var json = object
        var transformer: DSparkTransformerLayerConfig?

        // The vLLM-'speculators' packaging describes the same drafter with a different
        // schema, so translate it into the DeepSpec schema and fall through. This must
        // run before family detection: a speculators drafter carries its backbone
        // model_type nested under transformer_layer_config.
        if json["speculators_config"] != nil || json["speculators_model_type"] != nil {
            let translated = try translateSpeculators(json)
            transformer = translated.backbone
            json = translated.config
        }

        // A declared draft_vocab_size EQUAL to vocab_size is the full vocab: no
        // reduction and no d2t table. Keep nil so the full-vocab path stays untouched.
        if let draft = intValue(json["draft_vocab_size"]),
            let vocab = intValue(json["vocab_size"]), draft == vocab
        {
            json["draft_vocab_size"] = nil
        }

        // SpecForge packaging: nested dflash_config tagged projector_type "dspark" —
        // hoist the taps, and sample the anchor slot (logits_start 0).
        let dflashConfig = json["dflash_config"] as? [String: Any] ?? [:]
        let isSpecForge = (dflashConfig["projector_type"] as? String) == "dspark"
        if isSpecForge {
            for key in ["target_layer_ids", "mask_token_id"]
            where json[key] == nil && dflashConfig[key] != nil {
                json[key] = dflashConfig[key]
            }
            if json["logits_start"] == nil { json["logits_start"] = 0 }
        }

        // LiquidAI LFM2-style packaging: a DSpark architecture with the taps nested
        // inside dflash_config and no projector_type tag — hoist them the same way.
        let architectures = ((json["architectures"] as? [String]) ?? []).joined(separator: " ")
        if !isSpecForge, architectures.contains("DSpark"), json["target_layer_ids"] == nil,
            dflashConfig["target_layer_ids"] != nil
        {
            for key in ["target_layer_ids", "mask_token_id", "num_target_layers"]
            where json[key] == nil && dflashConfig[key] != nil {
                json[key] = dflashConfig[key]
            }
            if json["logits_start"] == nil { json["logits_start"] = 0 }
        }

        if json["block_size"] == nil, json.keys.contains(where: { $0.hasPrefix("dspark_") }) {
            throw DSparkError.invalidConfig(
                "full target model with an embedded DSpark drafter (dspark_* fields), "
                    + "not a standalone drafter checkpoint")
        }

        let modelType = (json["model_type"] as? String) ?? ""
        let family: Family
        if modelType.contains("qwen3") {
            family = .qwen3
        } else if modelType.contains("gemma4") {
            family = .gemma4
        } else {
            throw DSparkError.invalidConfig(
                "unsupported drafter family (model_type=\(modelType))")
        }

        let required = [
            "hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads", "block_size", "mask_token_id", "target_layer_ids",
        ]
        let missing = required.filter { json[$0] == nil }
        guard missing.isEmpty else {
            throw DSparkError.invalidConfig(
                "missing required DeepSpec drafter fields \(missing)")
        }

        let taps = ((json["target_layer_ids"] as? [Any]) ?? []).compactMap { intValue($0) }
        let hidden = try reqInt(json, "hidden_size")
        let heads = try reqInt(json, "num_attention_heads")

        if family == .qwen3 {
            let rope = json["rope_parameters"] as? [String: Any] ?? [:]
            let headDim = intValue(json["head_dim"]) ?? hidden / heads
            let causalBlock = (boolValue(json["dflash_query_causal"]) ?? false)
                || (boolValue(dflashConfig["causal"]) ?? false)
            let sink = boolValue(json["attention_sink_bias"]) ?? false
            var window: Int?
            if causalBlock || sink || !dflashConfig.isEmpty {
                if let value = intValue(json["sliding_window"]) {
                    window = value
                } else if boolValue(dflashConfig["use_swa"]) ?? false,
                    let value = intValue(dflashConfig["swa_window_size"])
                {
                    window = value
                }
            }
            let logitsStart: Int
            if let explicit = intValue(json["logits_start"]) {
                logitsStart = explicit
            } else if let anchor = boolValue(json["sample_from_anchor"]) {
                logitsStart = anchor ? 0 : 1
            } else {
                logitsStart = 0
            }
            return DSparkConfig(
                family: .qwen3,
                hiddenSize: hidden,
                vocabSize: try reqInt(json, "vocab_size"),
                draftVocabSize: intValue(json["draft_vocab_size"]),
                numHiddenLayers: try reqInt(json, "num_hidden_layers"),
                intermediateSize: try reqInt(json, "intermediate_size"),
                rmsNormEps: floatValue(json["rms_norm_eps"]) ?? 1e-6,
                numAttentionHeads: heads,
                numKeyValueHeads: intValue(json["num_key_value_heads"]) ?? 8,
                numGlobalKeyValueHeads: 1,
                headDim: headDim,
                globalHeadDim: 512,
                attentionKEqV: false,
                attentionBias: boolValue(json["attention_bias"]) ?? false,
                attentionSink: sink,
                ropeTheta: floatValue(rope["rope_theta"]) ?? floatValue(json["rope_theta"])
                    ?? 1_000_000,
                partialRotaryFactor: 1.0,
                ropeType: "default",
                causalBlock: causalBlock,
                slidingWindow: window,
                hasOwnLmHead: boolValue(json["has_lm_head"]) ?? true,
                hasOwnEmbed: true,
                blockSize: try reqInt(json, "block_size"),
                maskTokenId: try reqInt(json, "mask_token_id"),
                logitsStart: logitsStart,
                targetLayerIds: taps,
                numTargetLayers: intValue(json["num_target_layers"]) ?? 36,
                markovRank: intValue(json["markov_rank"]) ?? 256,
                markovHeadType: (json["markov_head_type"] as? String) ?? "vanilla",
                enableConfidenceHead: boolValue(json["enable_confidence_head"]) ?? true,
                confidenceHeadWithMarkov: boolValue(json["confidence_head_with_markov"]) ?? true,
                finalLogitSoftcapping: floatValue(json["final_logit_softcapping"]),
                padTokenId: intValue(json["pad_token_id"]) ?? 0,
                mlpActivation: "silu",
                normStyle: "qwen",
                useVNorm: false,
                transformerLayerConfig: transformer)
        }

        let rope =
            (json["rope_parameters"] as? [String: Any])?["full_attention"] as? [String: Any]
            ?? [:]
        return DSparkConfig(
            family: .gemma4,
            hiddenSize: hidden,
            vocabSize: try reqInt(json, "vocab_size"),
            draftVocabSize: intValue(json["draft_vocab_size"]),
            numHiddenLayers: try reqInt(json, "num_hidden_layers"),
            intermediateSize: try reqInt(json, "intermediate_size"),
            rmsNormEps: floatValue(json["rms_norm_eps"]) ?? 1e-6,
            numAttentionHeads: heads,
            numKeyValueHeads: intValue(json["num_key_value_heads"]) ?? 8,
            numGlobalKeyValueHeads: intValue(json["num_global_key_value_heads"]) ?? 1,
            headDim: intValue(json["head_dim"]) ?? 256,
            globalHeadDim: intValue(json["global_head_dim"]) ?? 512,
            attentionKEqV: boolValue(json["attention_k_eq_v"]) ?? true,
            attentionBias: boolValue(json["attention_bias"]) ?? false,
            attentionSink: boolValue(json["attention_sink_bias"]) ?? false,
            ropeTheta: floatValue(rope["rope_theta"]) ?? 1_000_000,
            partialRotaryFactor: floatValue(rope["partial_rotary_factor"]) ?? 0.25,
            ropeType: (rope["rope_type"] as? String) ?? "proportional",
            causalBlock: false,
            slidingWindow: nil,
            hasOwnLmHead: true,
            hasOwnEmbed: true,
            blockSize: try reqInt(json, "block_size"),
            maskTokenId: try reqInt(json, "mask_token_id"),
            logitsStart: intValue(json["logits_start"]) ?? 0,
            targetLayerIds: taps,
            numTargetLayers: intValue(json["num_target_layers"]) ?? 48,
            markovRank: intValue(json["markov_rank"]) ?? 256,
            markovHeadType: (json["markov_head_type"] as? String) ?? "vanilla",
            enableConfidenceHead: boolValue(json["enable_confidence_head"]) ?? true,
            confidenceHeadWithMarkov: boolValue(json["confidence_head_with_markov"]) ?? true,
            finalLogitSoftcapping: floatValue(json["final_logit_softcapping"]) ?? 30.0,
            padTokenId: intValue(json["pad_token_id"]) ?? 0,
            mlpActivation: "gelu_tanh",
            normStyle: "gemma",
            useVNorm: true,
            transformerLayerConfig: transformer)
    }

    /// Rewrite a vLLM-`speculators` DSpark config into the DeepSpec schema — a rename,
    /// not a port (the weight names already match DeepSpec 1:1).
    private static func translateSpeculators(_ json: [String: Any]) throws
        -> (config: [String: Any], backbone: DSparkTransformerLayerConfig)
    {
        let spec = json["speculators_config"] as? [String: Any] ?? [:]
        let algorithm = (json["speculators_model_type"] as? String) ?? (spec["algorithm"] as? String)
        guard algorithm == "dspark" else {
            throw DSparkError.invalidConfig(
                "vLLM 'speculators' checkpoint for algorithm \(algorithm ?? "nil") — "
                    + "only 'dspark' heads translate")
        }
        guard let layerConfig = json["transformer_layer_config"] as? [String: Any] else {
            throw DSparkError.invalidConfig(
                "speculators 'dspark' config is missing transformer_layer_config — "
                    + "without it the drafter backbone is undefined")
        }
        guard let tapsAny = json["aux_hidden_state_layer_ids"] as? [Any] else {
            throw DSparkError.invalidConfig(
                "speculators 'dspark' config is missing aux_hidden_state_layer_ids — "
                    + "without it the fusion taps are undefined")
        }
        let needed = [
            "hidden_size", "vocab_size", "num_hidden_layers", "intermediate_size",
            "num_attention_heads",
        ]
        let absent = needed.filter { layerConfig[$0] == nil }
        guard absent.isEmpty else {
            throw DSparkError.invalidConfig(
                "transformer_layer_config is missing \(absent) — "
                    + "the drafter backbone is underspecified")
        }

        let hidden = try reqInt(layerConfig, "hidden_size")
        let vocab = try reqInt(layerConfig, "vocab_size")
        let heads = try reqInt(layerConfig, "num_attention_heads")
        let layers = try reqInt(layerConfig, "num_hidden_layers")
        let intermediate = try reqInt(layerConfig, "intermediate_size")
        let kvHeads = intValue(layerConfig["num_key_value_heads"]) ?? heads
        let headDim = intValue(layerConfig["head_dim"]) ?? hidden / heads
        let eps = floatValue(layerConfig["rms_norm_eps"]) ?? 1e-6
        let attentionBias = boolValue(layerConfig["attention_bias"]) ?? false
        let rope = layerConfig["rope_parameters"] as? [String: Any] ?? [:]
        let ropeTheta = floatValue(rope["rope_theta"]) ?? 1_000_000
        let partialRotary = floatValue(rope["partial_rotary_factor"]) ?? 1.0
        let ropeType = (rope["rope_type"] as? String) ?? "default"
        let layerSlidingWindow = intValue(layerConfig["sliding_window"])
        let useSlidingWindow = boolValue(layerConfig["use_sliding_window"]) ?? false
        let layerTypes = (layerConfig["layer_types"] as? [String])
            ?? Array(repeating: "full_attention", count: layers)
        let taps = tapsAny.compactMap { intValue($0) }

        let backbone = DSparkTransformerLayerConfig(
            modelType: (layerConfig["model_type"] as? String) ?? "qwen3",
            hiddenSize: hidden,
            vocabSize: vocab,
            numHiddenLayers: layers,
            intermediateSize: intermediate,
            numAttentionHeads: heads,
            numKeyValueHeads: kvHeads,
            headDim: headDim,
            rmsNormEps: eps,
            attentionBias: attentionBias,
            ropeTheta: ropeTheta,
            partialRotaryFactor: partialRotary,
            ropeType: ropeType,
            slidingWindow: layerSlidingWindow,
            useSlidingWindow: useSlidingWindow,
            layerTypes: layerTypes)

        var out: [String: Any] = [
            "model_type": backbone.modelType,
            "hidden_size": hidden,
            "vocab_size": vocab,
            "num_hidden_layers": layers,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "head_dim": headDim,
            "rms_norm_eps": eps,
            "attention_bias": attentionBias,
            "rope_parameters": rope,
            "pad_token_id": intValue(layerConfig["pad_token_id"]) ?? 0,
            "target_layer_ids": taps,
            "num_target_layers": (taps.max() ?? 0) + 1,
        ]
        // Equal sizes mean no reduction; keep the key absent so the full-vocab path
        // stays untouched.
        if let draftVocab = intValue(json["draft_vocab_size"]), draftVocab != 0,
            draftVocab != vocab
        {
            out["draft_vocab_size"] = draftVocab
        }
        // DSpark's own knobs already sit at the top level in both schemas.
        for key in [
            "block_size", "mask_token_id", "markov_rank", "markov_head_type",
            "enable_confidence_head", "confidence_head_with_markov",
            "final_logit_softcapping",
        ] {
            if let value = json[key] { out[key] = value }
        }

        // DFlash-lineage causal sliding-window block attention: a head declaring
        // sliding_attention layer_types + use_sliding_window + sliding_window attends
        // causally over the window unless sliding_window_non_causal says bidirectional.
        if layerTypes.contains("sliding_attention"), useSlidingWindow,
            let window = layerSlidingWindow
        {
            out["sliding_window"] = window
            out["dflash_query_causal"] = !(boolValue(json["sliding_window_non_causal"]) ?? false)
        }
        if boolValue(json["attention_sink_bias"]) == true {
            out["attention_sink_bias"] = true
        }

        // How many of the block's slots carry predictions. block - speculative_tokens
        // == 0 is a legitimate answer (anchor-as-pos0 heads predict from slot 0); only
        // fall back to sample_from_anchor when the proposal width is missing or
        // malformed, and to 1 (the DFlash anchor-slot convention) when neither is said.
        let proposals = (spec["proposal_methods"] as? [[String: Any]]) ?? []
        let speculativeTokens = proposals.first.flatMap { intValue($0["speculative_tokens"]) }
        let block = intValue(json["block_size"])
        var start: Int?
        if let speculativeTokens, speculativeTokens != 0, let block, block != 0,
            (0 ..< block).contains(block - speculativeTokens)
        {
            start = block - speculativeTokens
        } else if let anchor = boolValue(json["sample_from_anchor"]) {
            start = anchor ? 0 : 1
        }
        out["logits_start"] = start ?? 1
        return (out, backbone)
    }
}

private func reqInt(_ json: [String: Any], _ key: String) throws -> Int {
    guard let value = intValue(json[key]) else {
        throw DSparkError.invalidConfig("missing \(key)")
    }
    return value
}

private func intValue(_ any: Any?) -> Int? {
    switch any {
    case let value as Int: return value
    case let value as Int64: return Int(value)
    case let value as NSNumber: return value.intValue
    case let value as Double: return Int(value)
    default: return nil
    }
}

private func floatValue(_ any: Any?) -> Float? {
    switch any {
    case let value as Float: return value
    case let value as Double: return Float(value)
    case let value as Int: return Float(value)
    case let value as NSNumber: return value.floatValue
    default: return nil
    }
}

private func boolValue(_ any: Any?) -> Bool? {
    switch any {
    case let value as Bool: return value
    case let value as NSNumber: return value.boolValue
    default: return nil
    }
}
