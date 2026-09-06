import Foundation

// NovaMLX-TIE: Tiered Inference Engine
//
// Manifest produced by `scripts/expert_shard_layout.py`. Describes the on-disk
// layout of a model that has been split into tiered shards:
//   - tier0.safetensors               shared weights (always resident)
//   - expert.L{NN}.E{NNN}.safetensors  per-expert weights (MoE strategy)
//   - layer.L{NN}.safetensors          per-layer weights (dense strategy)
//
// `strategy` selects how the runtime interprets the shards:
//   .expert — MoE model, per-expert shards, hooks fire from SwitchLinear
//   .layer  — Dense model, per-layer shards, hooks fire from Linear
//   .mixed  — Hybrid (e.g. Jamba): some layers MoE, some dense
//   .none   — Manifest present but no tiering applied (fallback)

public enum TierStrategy: String, Codable, Sendable {
    case expert    // MoE: per-expert files via SwitchLinear hook
    case layer     // Dense: per-layer files via Linear hook
    case mixed     // Hybrid (Jamba etc.): both
    case none      // Manifest present but no streaming
}

/// On-disk layout description for a tiered model.
public struct TierManifest: Codable, Sendable {
    public let version: Int
    public let converter: String
    public let sourceModel: String
    public let architecture: String
    public let layout: String          // "stacked" | "classic" | "layer" | "none" (legacy MoE detector)
    public var strategy: TierStrategy  // .expert | .layer | .mixed | .none
    public let tier0File: String
    public let tier0TensorCount: Int
    public let tier0Bytes: Int64
    public let expertCount: Int
    public let experts: [ExpertEntry]
    /// Per-layer shard entries (dense strategy). Empty for pure MoE.
    public let layers: [LayerEntry]?

    enum CodingKeys: String, CodingKey {
        case version, converter, layout, strategy, experts, layers
        case sourceModel = "source_model"
        case architecture
        case tier0File = "tier0_file"
        case tier0TensorCount = "tier0_tensor_count"
        case tier0Bytes = "tier0_bytes"
        case expertCount = "expert_count"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.version = try c.decode(Int.self, forKey: .version)
        self.converter = try c.decode(String.self, forKey: .converter)
        self.sourceModel = try c.decode(String.self, forKey: .sourceModel)
        self.architecture = try c.decode(String.self, forKey: .architecture)
        self.layout = try c.decode(String.self, forKey: .layout)
        // Backward-compat: old manifests have no `strategy` field. Infer from layout.
        self.strategy = (try c.decodeIfPresent(TierStrategy.self, forKey: .strategy))
            ?? .expert
        self.tier0File = try c.decode(String.self, forKey: .tier0File)
        self.tier0TensorCount = try c.decode(Int.self, forKey: .tier0TensorCount)
        self.tier0Bytes = try c.decode(Int64.self, forKey: .tier0Bytes)
        self.expertCount = try c.decode(Int.self, forKey: .expertCount)
        self.experts = try c.decode([ExpertEntry].self, forKey: .experts)
        self.layers = try c.decodeIfPresent([LayerEntry].self, forKey: .layers)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(version, forKey: .version)
        try c.encode(converter, forKey: .converter)
        try c.encode(sourceModel, forKey: .sourceModel)
        try c.encode(architecture, forKey: .architecture)
        try c.encode(layout, forKey: .layout)
        try c.encode(strategy, forKey: .strategy)
        try c.encode(tier0File, forKey: .tier0File)
        try c.encode(tier0TensorCount, forKey: .tier0TensorCount)
        try c.encode(tier0Bytes, forKey: .tier0Bytes)
        try c.encode(expertCount, forKey: .expertCount)
        try c.encode(experts, forKey: .experts)
        try c.encodeIfPresent(layers, forKey: .layers)
    }

    public struct ExpertEntry: Codable, Sendable {
        public let layer: Int
        public let expert: Int
        public let file: String
        public let bytes: Int64
        public let tensors: [String]
        public let stackedSource: Bool?

        enum CodingKeys: String, CodingKey {
            case layer, expert, file, bytes, tensors
            case stackedSource = "stacked_source"
        }
    }

    public struct LayerEntry: Codable, Sendable {
        public let layer: Int
        public let file: String
        public let bytes: Int64
        public let tensors: [String]
        public init(layer: Int, file: String, bytes: Int64, tensors: [String]) {
            self.layer = layer; self.file = file; self.bytes = bytes; self.tensors = tensors
        }
    }

    /// Memberwise initializer (for tests + programmatic construction).
    /// Decoder uses init(from:) for backward-compat with old manifests.
    public init(
        version: Int, converter: String, sourceModel: String, architecture: String,
        layout: String, strategy: TierStrategy, tier0File: String,
        tier0TensorCount: Int, tier0Bytes: Int64,
        expertCount: Int, experts: [ExpertEntry], layers: [LayerEntry]?
    ) {
        self.version = version
        self.converter = converter
        self.sourceModel = sourceModel
        self.architecture = architecture
        self.layout = layout
        self.strategy = strategy
        self.tier0File = tier0File
        self.tier0TensorCount = tier0TensorCount
        self.tier0Bytes = tier0Bytes
        self.expertCount = expertCount
        self.experts = experts
        self.layers = layers
    }

    /// Look up the file path for a given (layer, expert) tuple.
    public func expertFile(layer: Int, expert: Int) -> String? {
        experts.first { $0.layer == layer && $0.expert == expert }?.file
    }

    /// Look up the file path for a given dense layer.
    public func layerFile(layer: Int) -> String? {
        layers?.first { $0.layer == layer }?.file
    }

    /// Total bytes across all per-expert files (excludes tier0).
    public var totalExpertBytes: Int64 {
        experts.reduce(0) { $0 + $1.bytes }
    }

    /// Total bytes across all per-layer files (excludes tier0).
    public var totalLayerBytes: Int64 {
        (layers ?? []).reduce(0) { $0 + $1.bytes }
    }
}

public enum TierManifestError: Error {
    case notTiered(URL)
    case manifestMissing(URL)
    case manifestUnreadable(String)
    case unsupportedVersion(Int)
}

public enum TierManifestLoader {
    /// Returns a manifest if `modelDir` contains a `tier-manifest.json`, else nil.
    /// Use this to decide between TieredOffloadPolicy and eager load.
    public static func loadIfPresent(modelDir: URL) throws -> TierManifest? {
        let manifestURL = modelDir.appendingPathComponent("tier-manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return nil
        }
        do {
            let data = try Data(contentsOf: manifestURL)
            let manifest = try JSONDecoder().decode(TierManifest.self, from: data)
            guard manifest.version == 1 else {
                throw TierManifestError.unsupportedVersion(manifest.version)
            }
            return manifest
        } catch let error as TierManifestError {
            throw error
        } catch {
            throw TierManifestError.manifestUnreadable(error.localizedDescription)
        }
    }

    /// True iff the directory has a tier-manifest.json (i.e., was produced by
    /// `expert_shard_layout.py`).
    public static func isTiered(_ modelDir: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("tier-manifest.json").path
        )
    }
}
