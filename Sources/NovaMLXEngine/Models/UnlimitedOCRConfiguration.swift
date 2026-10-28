import Foundation

/// `config.json` for `baidu/Unlimited-OCR-2` and the mlx-community gen-1 packs.
///
/// OCR-2 sets `sparse_attention_config.training_stage = dash`. Gen-1 leaves
/// that block out and keeps a 128-token R-SWA ring.
public struct UnlimitedOCRConfiguration: Decodable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var routedExperts: Int
    public var sharedExperts: Int
    public var expertsPerToken: Int
    public var firstDenseLayers: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var maxPositionEmbeddings: Int
    public var ropeScaling: UnlimitedOCRRopeScaling?
    public var slidingWindow: Int
    public var imageTokenID: Int
    public var pageTokenID: Int
    public var perPageImageTokens: Int
    public var isDash: Bool
    public var dash: UnlimitedOCRDashParams
    public var projectorInput: Int
    public var projectorHidden: Int
    public var clipHidden: Int
    public var clipIntermediate: Int
    public var clipLayers: Int
    public var clipHeads: Int
    public var clipPatch: Int
    public var samWidth: Int
    public var samLayers: Int
    public var samHeads: Int
    public var samPatch: Int
    public var samWindow: Int
    public var samGlobalAttention: [Int]
    public var samNeck: Int
    public var samDownsample: [Int]
    public var maxCrops: Int

    public var decodeAttendedKeys: Int { dash.attendedKeys }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case languageConfig = "language_config"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case projectorConfig = "projector_config"
        case sparseAttentionConfig = "sparse_attention_config"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vHeadDim = "v_head_dim"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case firstKDenseReplace = "first_k_dense_replace"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case slidingWindowSize = "sliding_window_size"
        case slidingWindow = "sliding_window"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try root.decodeIfPresent(String.self, forKey: .modelType) ?? "unlimited-ocr"
        let text = try root.decodeIfPresent(TextBox.self, forKey: .languageConfig)
            ?? root.decodeIfPresent(TextBox.self, forKey: .textConfig)
            ?? TextBox()
        let topSparse = try root.decodeIfPresent(SparseBox.self, forKey: .sparseAttentionConfig)
        let sparse = topSparse ?? text.sparseAttentionConfig
        vocabularySize = text.vocabSize ?? 129_280
        hiddenSize = text.hiddenSize ?? 1280
        intermediateSize = text.intermediateSize ?? 6848
        moeIntermediateSize = text.moeIntermediateSize ?? 896
        hiddenLayers = text.numHiddenLayers ?? 12
        attentionHeads = text.numAttentionHeads ?? 10
        kvHeads = text.numKeyValueHeads ?? attentionHeads
        let vDim = text.vHeadDim ?? 0
        headDim = vDim > 0 ? vDim : hiddenSize / max(attentionHeads, 1)
        routedExperts = text.nRoutedExperts ?? 64
        sharedExperts = text.nSharedExperts ?? 2
        expertsPerToken = text.numExpertsPerTok ?? 6
        firstDenseLayers = text.firstKDenseReplace ?? 1
        rmsNormEps = text.rmsNormEps ?? 1e-6
        ropeTheta = text.ropeTheta ?? 10_000
        maxPositionEmbeddings = text.maxPositionEmbeddings ?? 32_768
        ropeScaling = text.ropeScaling
        slidingWindow = text.slidingWindowSize ?? text.slidingWindow ?? 128
        let vision = try root.decodeIfPresent(VisionBox.self, forKey: .visionConfig)
        let projector = try root.decodeIfPresent(ProjectorBox.self, forKey: .projectorConfig)
        projectorInput = projector?.inputDim ?? 2048
        projectorHidden = projector?.nEmbed ?? hiddenSize
        clipHidden = vision?.clip?.width ?? 1024
        clipLayers = vision?.clip?.layers ?? 24
        clipHeads = vision?.clip?.heads ?? 16
        clipPatch = vision?.clip?.patchSize ?? 14
        // CLIP-L/14 is 4096. The checkpoint's mlp_ratio is not that width.
        clipIntermediate = vision?.intermediateSize ?? 4096
        samWidth = vision?.sam?.width ?? 768
        samLayers = vision?.sam?.layers ?? 12
        samHeads = vision?.sam?.heads ?? 12
        samPatch = 16
        samWindow = 14
        samGlobalAttention = vision?.sam?.globalAttnIndexes ?? [2, 5, 8, 11]
        samNeck = 256
        samDownsample = vision?.sam?.downsampleChannels ?? [512, 1024]
        imageTokenID = sparse?.imageTokenID ?? UnlimitedOCRSupport.imageTokenID
        pageTokenID = sparse?.pageTokenID ?? UnlimitedOCRSupport.pageTokenID
        perPageImageTokens = sparse?.perPageImageTokens ?? UnlimitedOCRSupport.globalTokenCount()
        isDash = sparse?.trainingStage == "dash" && (sparse?.useSparseAttention ?? true)
        dash = UnlimitedOCRDashParams(sparse: sparse, layers: hiddenLayers, headDim: headDim)
        maxCrops = isDash ? 16 : 32
    }

    private struct TextBox: Decodable {
        var vocabSize: Int? = nil
        var hiddenSize: Int? = nil
        var intermediateSize: Int? = nil
        var moeIntermediateSize: Int? = nil
        var numHiddenLayers: Int? = nil
        var numAttentionHeads: Int? = nil
        var numKeyValueHeads: Int? = nil
        var vHeadDim: Int? = nil
        var nRoutedExperts: Int? = nil
        var nSharedExperts: Int? = nil
        var numExpertsPerTok: Int? = nil
        var firstKDenseReplace: Int? = nil
        var rmsNormEps: Float? = nil
        var ropeTheta: Float? = nil
        var maxPositionEmbeddings: Int? = nil
        var ropeScaling: UnlimitedOCRRopeScaling? = nil
        var slidingWindowSize: Int? = nil
        var slidingWindow: Int? = nil
        var sparseAttentionConfig: SparseBox? = nil
        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case vHeadDim = "v_head_dim"
            case nRoutedExperts = "n_routed_experts"
            case nSharedExperts = "n_shared_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case firstKDenseReplace = "first_k_dense_replace"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeScaling = "rope_scaling"
            case slidingWindowSize = "sliding_window_size"
            case slidingWindow = "sliding_window"
            case sparseAttentionConfig = "sparse_attention_config"
        }
    }

    fileprivate struct SparseBox: Decodable {
        var trainingStage: String? = nil
        var useSparseAttention: Bool? = nil
        var blockSize: Int? = nil
        var localBlocks: Int? = nil
        var indexDim: Int? = nil
        var topK: Int? = nil
        var topM: Int? = nil
        var sigma: Float? = nil
        var imageOnly: Bool? = nil
        var imageTokenID: Int? = nil
        var pageTokenID: Int? = nil
        var perPageImageTokens: Int? = nil
        var ropeMode: String? = nil
        var ropeTokTheta: Float? = nil
        var ropePageTheta: Float? = nil
        var ropePagePairs: Int? = nil
        var ropeTokFreqMode: String? = nil
        var ropePageMod: Int? = nil
        var scale: [Float]? = nil
        var floor: [Float]? = nil
        var scalingFactor: Float? = nil
        var priorFloor: Float? = nil
        enum CodingKeys: String, CodingKey {
            case trainingStage = "training_stage"
            case useSparseAttention = "use_sparse_attention"
            case blockSize = "sparse_block_size"
            case localBlocks = "sparse_local_block"
            case indexDim = "sparse_index_dim"
            case topK = "dash_topk_blocks"
            case topM = "sparse_topk_blocks"
            case sigma = "dash_sigma"
            case imageOnly = "dash_image_only"
            case imageTokenID = "image_token_id"
            case pageTokenID = "page_token_id"
            case perPageImageTokens = "per_page_image_tokens"
            case ropeMode = "sparse_main_rope_mode"
            case ropeTokTheta = "sparse_rope_tok_theta"
            case ropePageTheta = "sparse_rope_page_theta"
            case ropePagePairs = "sparse_rope_page_pairs"
            case ropeTokFreqMode = "sparse_main_rope_tok_freq_mode"
            case ropePageMod = "sparse_rope_page_mod"
            case scale = "dash_scale_per_layer"
            case floor = "dash_floor_per_layer"
            case scalingFactor = "dash_scaling_factor"
            case priorFloor = "dash_prior_floor"
        }
    }

    private struct VisionBox: Decodable {
        var mlpRatio: Double? = nil
        var intermediateSize: Int? = nil
        var clip: Tower?
        var sam: SAM?
        struct Tower: Decodable {
            var width: Int? = nil
            var layers: Int? = nil
            var heads: Int? = nil
            var patchSize: Int? = nil
            enum CodingKeys: String, CodingKey {
                case width, layers, heads
                case patchSize = "patch_size"
            }
        }
        struct SAM: Decodable {
            var width: Int? = nil
            var layers: Int? = nil
            var heads: Int? = nil
            var globalAttnIndexes: [Int]? = nil
            var downsampleChannels: [Int]? = nil
            enum CodingKeys: String, CodingKey {
                case width, layers, heads
                case globalAttnIndexes = "global_attn_indexes"
                case downsampleChannels = "downsample_channels"
            }
        }
        enum CodingKeys: String, CodingKey {
            case mlpRatio = "mlp_ratio"
            case intermediateSize = "intermediate_size"
            case width
        }
        struct WidthBox: Decodable {
            var clip: Tower?
            var sam: SAM?
            enum CodingKeys: String, CodingKey {
                case clip = "clip-l-14-224"
                case sam = "sam_vit_b"
            }
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            mlpRatio = try c.decodeIfPresent(Double.self, forKey: .mlpRatio)
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
            clip = nil
            sam = nil
            if let box = try? c.decode(WidthBox.self, forKey: .width) {
                clip = box.clip
                sam = box.sam
            }
        }
    }

    private struct ProjectorBox: Decodable {
        var inputDim: Int? = nil
        var nEmbed: Int? = nil
        enum CodingKeys: String, CodingKey {
            case inputDim = "input_dim"
            case nEmbed = "n_embed"
        }
    }
}

public struct UnlimitedOCRRopeScaling: Codable, Sendable {
    public var factor: Float?
    public var originalMax: Int?
    public var betaFast: Float?
    public var betaSlow: Float?
    public var attentionFactor: Float?
    public var ropeType: String?
    enum CodingKeys: String, CodingKey {
        case factor
        case originalMax = "original_max_position_embeddings"
        case betaFast = "beta_fast"
        case betaSlow = "beta_slow"
        case attentionFactor = "attention_factor"
        case ropeType = "rope_type"
        case kind = "type"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        factor = try c.decodeIfPresent(Float.self, forKey: .factor)
        originalMax = try c.decodeIfPresent(Int.self, forKey: .originalMax)
        betaFast = try c.decodeIfPresent(Float.self, forKey: .betaFast)
        betaSlow = try c.decodeIfPresent(Float.self, forKey: .betaSlow)
        attentionFactor = try c.decodeIfPresent(Float.self, forKey: .attentionFactor)
        ropeType = try c.decodeIfPresent(String.self, forKey: .ropeType)
            ?? c.decodeIfPresent(String.self, forKey: .kind)
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(factor, forKey: .factor)
        try c.encodeIfPresent(originalMax, forKey: .originalMax)
        try c.encodeIfPresent(betaFast, forKey: .betaFast)
        try c.encodeIfPresent(betaSlow, forKey: .betaSlow)
        try c.encodeIfPresent(attentionFactor, forKey: .attentionFactor)
        try c.encodeIfPresent(ropeType, forKey: .ropeType)
    }
}

/// Inference knobs for one DASH layer stack. Decode always attends
/// `topK * blockSize + localBlocks * blockSize` keys.
public struct UnlimitedOCRDashParams: Sendable {
    public var blockSize: Int
    public var localBlocks: Int
    public var indexDim: Int
    public var topK: Int
    public var topM: Int
    public var sigma: Float
    public var imageOnly: Bool
    public var page2D: Bool
    public var ropeTokTheta: Float
    public var ropePageTheta: Float
    public var ropePagePairs: Int
    public var ropePageMod: Int
    public var resampleTokenFrequencies: Bool
    public var indexScale: [Float]
    public var floor: [Float]

    public var localWindow: Int { localBlocks * blockSize }
    public var attendedKeys: Int { topK * blockSize + localWindow }

    fileprivate init(sparse: UnlimitedOCRConfiguration.SparseBox?, layers: Int, headDim: Int) {
        blockSize = sparse?.blockSize ?? 32
        localBlocks = max(1, sparse?.localBlocks ?? 4)
        indexDim = sparse?.indexDim ?? headDim
        topK = sparse?.topK ?? 12
        topM = sparse?.topM ?? 16
        sigma = sparse?.sigma ?? 1
        imageOnly = sparse?.imageOnly ?? true
        page2D = (sparse?.ropeMode ?? "page2d") == "page2d"
        ropeTokTheta = sparse?.ropeTokTheta ?? 10_000
        ropePageTheta = sparse?.ropePageTheta ?? 256
        ropePagePairs = sparse?.ropePagePairs ?? 1
        ropePageMod = sparse?.ropePageMod ?? 2
        resampleTokenFrequencies = (sparse?.ropeTokFreqMode ?? "resample") != "truncate"
        let scaleDefault = sparse?.scalingFactor ?? 128
        let floorDefault = sparse?.priorFloor ?? 0.0001
        let denom = Float(indexDim).squareRoot()
        var scales: [Float] = []
        var floors: [Float] = []
        for layer in 0..<layers {
            let scale = sparse?.scale.flatMap { layer < $0.count ? $0[layer] : nil } ?? scaleDefault
            let floor = sparse?.floor.flatMap { layer < $0.count ? $0[layer] : nil } ?? floorDefault
            scales.append(scale / denom)
            floors.append(floor)
        }
        indexScale = scales
        floor = floors
    }

    public init() {
        self.init(sparse: nil, layers: 12, headDim: 128)
    }
}
