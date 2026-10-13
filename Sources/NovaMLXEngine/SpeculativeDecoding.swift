import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Compiled decode reporting

/// Models that publish how many single-token decode segments `mlx-swift-lm` compiled.
public protocol CompiledDecodeReporting: AnyObject {
    var compiledDecodeSegmentCount: Int { get }
}

// MARK: - DFlash target

/// Same contract as `MLXLMCommon.DFlashTarget` — Qwen3/Qwen3.5/etc. conform there.
/// Keep a single protocol identity so `model.value as? any DFlashTarget` succeeds.
public typealias DFlashTarget = MLXLMCommon.DFlashTarget

public struct DFlashRoundStats: Sendable {
    public let proposed: Int
    public let accepted: Int
    public let cap: Int
    public let draftMs: Double
    public let verifyMs: Double
    public let acceptMs: Double
    public let rollbackMs: Double

    public init(
        proposed: Int, accepted: Int, cap: Int,
        draftMs: Double, verifyMs: Double, acceptMs: Double, rollbackMs: Double
    ) {
        self.proposed = proposed
        self.accepted = accepted
        self.cap = cap
        self.draftMs = draftMs
        self.verifyMs = verifyMs
        self.acceptMs = acceptMs
        self.rollbackMs = rollbackMs
    }
}

public enum DFlashError: Error, Sendable {
    case invalidConfig(String)
}

/// Greedy accept: matching draft prefix, then the first target token that disagrees.
public enum DFlashAccept {
    public struct Result: Sendable, Equatable {
        public let accepted: Int
        public let committed: [Int]
    }

    public static func greedy(drafted: [Int], targetArgmax: [Int]) -> Result {
        var accepted = 0
        while accepted < drafted.count, accepted < targetArgmax.count,
              drafted[accepted] == targetArgmax[accepted]
        {
            accepted += 1
        }
        var committed = Array(drafted.prefix(accepted))
        if accepted < targetArgmax.count {
            committed.append(targetArgmax[accepted])
        }
        return Result(accepted: accepted, committed: committed)
    }
}

/// Bounds for DFlash verify capture and chunked prefill.
public enum DFlashLimits {
    public static let prefillChunkTokens = 4096

    /// GDN spec-capture is only valid for the short verify window.
    public static func shouldRecordSpec(sequenceLength: Int) -> Bool {
        (2...8192).contains(sequenceLength)
    }
}

public struct DFlashConfig: Sendable {
    public var hiddenSize: Int
    public var vocabSize: Int
    public var numHiddenLayers: Int
    public var numKeyValueHeads: Int
    public var blockSize: Int
    public var convKernelSize: Int
    public var maskTokenId: Int
    public var selectorRank: Int
    public var selectorTopK: Int
    public var targetLayerIds: [Int]
    public var isDFlash2: Bool

    public static func fromJSON(_ data: Data) throws -> DFlashConfig {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw DFlashError.invalidConfig("not a JSON object")
        }
        let nested = json["dflash_config"] as? [String: Any] ?? [:]
        func value(_ key: String) -> Any? {
            json[key] ?? nested[key]
        }
        let architectures = json["architectures"] as? [String] ?? []
        let isDFlash2Arch = architectures.contains { $0.localizedCaseInsensitiveContains("dflash2") }
        let selectorRank = jsonInt(value("selector_rank"))
        let selectorTopK = jsonInt(value("selector_top_k"))
        if isDFlash2Arch && (selectorRank == nil || selectorTopK == nil) {
            throw DFlashError.invalidConfig("DFlash2 config is missing selector_rank or selector_top_k")
        }
        let rank = selectorRank ?? 0
        let topK = selectorTopK ?? 0
        return DFlashConfig(
            hiddenSize: jsonInt(value("hidden_size")) ?? 0,
            vocabSize: jsonInt(value("vocab_size")) ?? 0,
            numHiddenLayers: jsonInt(value("num_hidden_layers")) ?? 1,
            numKeyValueHeads: jsonInt(value("num_key_value_heads")) ?? 1,
            blockSize: jsonInt(value("block_size")) ?? 8,
            convKernelSize: jsonInt(value("conv_kernel_size")) ?? 0,
            maskTokenId: jsonInt(value("mask_token_id")) ?? 0,
            selectorRank: rank,
            selectorTopK: topK,
            targetLayerIds: jsonInts(value("target_layer_ids")),
            isDFlash2: isDFlash2Arch || (rank > 0 && topK > 0)
        )
    }
}

/// Block-diffusion drafter. Loaded models are cast to this class at generate time.
public final class DFlashDraftModel: Module, LanguageModel, KVCacheDimensionProvider {
    public let config: DFlashConfig
    public let kvHeads: [Int]

    public init(_ config: DFlashConfig) {
        self.config = config
        let layers = max(config.numHiddenLayers, 1)
        self.kvHeads = Array(repeating: max(config.numKeyValueHeads, 1), count: layers)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.ndim >= 2 ? inputs.dim(0) : 1
        let sequence = inputs.ndim >= 2 ? inputs.dim(1) : inputs.dim(0)
        let vocab = max(config.vocabSize, 1)
        return MLXArray.zeros([batch, sequence, vocab])
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    public func makeCache() -> [DSparkCtxCache] {
        kvHeads.map { _ in DSparkCtxCache() }
    }

    public func appendContext(_ hidden: MLXArray, cache: [DSparkCtxCache]) {
        let rows = hidden.ndim >= 2 ? hidden.dim(1) : 0
        for ctx in cache {
            ctx.offset += rows
        }
    }
}

/// DFlash2 iterator. Until a full block loop is wired, this drives the target
/// with the ordinary one-token iterator so a DFlash cast still produces text.
public struct DFlashTokenIterator: TokenIteratorProtocol {
    private var inner: TokenIterator
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (DFlashRoundStats) -> Void)?

    public var maxTokens: Int? { inner.maxTokens }
    public var tokenCount: Int { inner.tokenCount }
    public var promptPrefillTime: TimeInterval { inner.promptPrefillTime }
    public var state: LMOutput.State? { inner.state }

    public init(
        input: LMInput,
        target: any DFlashTarget,
        drafter: DFlashDraftModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processor: LogitProcessor? = nil
    ) throws {
        _ = drafter.config.blockSize
        self.inner = try TokenIterator(
            input: input,
            model: target,
            cache: mainCache,
            processor: processor,
            sampler: parameters.sampler(),
            prefill: parameters.prefill,
            maxTokens: parameters.maxTokens
        )
    }

    public mutating func next() -> Int? {
        inner.next()
    }
}

// MARK: - MTP

/// Target model that can embed, project, and score an MTP verify window.
public protocol MtpTarget: LanguageModel {
    func mtpEmbed(_ tokens: MLXArray) -> MLXArray
    func mtpLmHead(_ hidden: MLXArray) -> MLXArray
    func mtpHiddenAndLogits(_ tokens: MLXArray, cache: [KVCache]?) -> (
        hidden: MLXArray, logits: MLXArray
    )
}

/// Drafter head. DeepSeek V4 is both ``MtpTarget`` and ``MtpDrafter``.
public protocol MtpDrafter: AnyObject {
    var mtpBlockSize: Int { get }
    func bindMtp(
        embed: @escaping (MLXArray) -> MLXArray,
        lmHead: @escaping (MLXArray) -> MLXArray
    )
    func mtpForward(tokenEmbed: MLXArray, hidden: MLXArray, cache: [KVCache]?) -> MLXArray
    func mtpNewCache(parameters: GenerateParameters?) -> [KVCache]
}

/// MTP iterator. Native block decode falls back to the target's ordinary sampler
/// so generation still completes when the speculative loop is not engaged.
public struct MtpTokenIterator: TokenIteratorProtocol {
    private var inner: TokenIterator
    public var totalProposed = 0
    public var totalAccepted = 0
    public var onSpeculationRound: (@Sendable (Int, Int) -> Void)?

    public var maxTokens: Int? { inner.maxTokens }
    public var tokenCount: Int { inner.tokenCount }
    public var promptPrefillTime: TimeInterval { inner.promptPrefillTime }
    public var state: LMOutput.State? { inner.state }

    public init(
        input: LMInput,
        target: any MtpTarget,
        drafter: any MtpDrafter,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processor: LogitProcessor? = nil
    ) throws {
        _ = drafter.mtpBlockSize
        _ = numDraftTokens
        self.inner = try TokenIterator(
            input: input,
            model: target,
            cache: mainCache,
            processor: processor,
            sampler: parameters.sampler(),
            prefill: parameters.prefill,
            maxTokens: parameters.maxTokens
        )
    }

    public mutating func next() -> Int? {
        inner.next()
    }
}

// MARK: - Companion discovery

public func isDSparkDraftConfig(at modelDir: URL) -> Bool {
    DSparkConfig.isDSparkDraftConfig(at: modelDir)
}

public func isDFlashDraftConfig(at modelDir: URL) -> Bool {
    guard let json = modelConfigJSON(at: modelDir) else { return false }
    if DSparkConfig.isDSparkDraftConfig(jsonObject: json) { return false }
    let modelType = (json["model_type"] as? String)?.lowercased() ?? ""
    if modelType.contains("dflash") { return true }
    let architectures = json["architectures"] as? [String] ?? []
    if architectures.contains(where: { $0.lowercased().contains("dflash") }) { return true }
    return json["dflash_config"] != nil
}

public func isMtpDraftConfig(at modelDir: URL) -> Bool {
    guard let json = modelConfigJSON(at: modelDir) else { return false }
    let modelType = (json["model_type"] as? String)?.lowercased() ?? ""
    if modelType == "qwen3_5_mtp" || modelType.hasSuffix("_mtp") { return true }
    let architectures = json["architectures"] as? [String] ?? []
    return architectures.contains { $0.lowercased().contains("mtp") }
}

public func checkpointHasMtpWeights(at modelDir: URL) -> Bool {
    if NovaMLXEngine.isMtpDraftConfig(at: modelDir) { return true }
    return weightMapKeys(at: modelDir).contains(where: isMtpWeightKey)
}

public func dsparkDraftCandidates(forMainId mainId: String) -> [String] {
    let name = modelLeaf(mainId)
    if name.contains("dspark") || name.contains("dflash") || name.contains("-mtp") {
        return []
    }
    if name.contains("deepseek-v4.1-flash") {
        return ["mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"]
    }
    return []
}

public func dflashDraftCandidates(forMainId mainId: String) -> [String] {
    let name = modelLeaf(mainId)
    if name.contains("dflash") || name.contains("-mtp") || name.contains("dspark") {
        return []
    }
    if name.contains("qwen3.8-27b") || name.contains("qwen3_8-27b") {
        return ["incoai/Qwen3.8-27B-DFlash2"]
    }
    return []
}

public func mtpDraftCandidates(forMainId mainId: String) -> [String] {
    let name = modelLeaf(mainId)
    if name.contains("-mtp") || name.contains("dflash") || name.contains("dspark") {
        return []
    }
    if name.contains("qwen3.8-27b") || name.contains("qwen3_8-27b") {
        return [
            "mlx-community/Qwen3.8-27B-MTP-4bit",
            "mlx-community/Qwen3.8-27B-MTP-8bit",
        ]
    }
    return []
}

private func modelLeaf(_ modelId: String) -> String {
    (modelId.split(separator: "/").last.map(String.init) ?? modelId).lowercased()
}

private func modelConfigJSON(at dir: URL) -> [String: Any]? {
    let url = dir.appendingPathComponent("config.json")
    guard let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    return json
}

private func weightMapKeys(at dir: URL) -> [String] {
    let url = dir.appendingPathComponent("model.safetensors.index.json")
    guard let data = try? Data(contentsOf: url),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let map = json["weight_map"] as? [String: Any]
    else { return [] }
    return Array(map.keys)
}

private func isMtpWeightKey(_ key: String) -> Bool {
    let lower = key.lowercased()
    return lower.contains("mtp_layers") || lower.contains("mtp.") || lower.contains(".mtp")
        || lower.hasPrefix("mtp")
}

private func jsonInt(_ value: Any?) -> Int? {
    if let int = value as? Int { return int }
    if let number = value as? NSNumber { return number.intValue }
    return nil
}

private func jsonInts(_ value: Any?) -> [Int] {
    guard let array = value as? [Any] else { return [] }
    return array.compactMap { jsonInt($0) }
}
