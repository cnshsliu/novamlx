import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct FamilySamplingDefaults: Sendable {
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Float?
    public let frequencyPenalty: Float?
    public var repetitionPenalty: Float?
    public let maxTokens: Int?

    public init(
        temperature: Double? = nil, topP: Double? = nil, topK: Int? = nil,
        minP: Float? = nil, frequencyPenalty: Float? = nil,
        repetitionPenalty: Float? = nil, maxTokens: Int? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.frequencyPenalty = frequencyPenalty
        self.repetitionPenalty = repetitionPenalty
        self.maxTokens = maxTokens
    }
}

public struct ModelFamilyOptimization: Sendable {
    public let defaultKVBits: Int?
    public let defaultKVGroupSize: Int
    public let prefillStepSize: Int
    public let recommendedContextLength: Int
    public let repeatLastN: Int
    public let headDim: Int
    public let samplingDefaults: FamilySamplingDefaults?

    public init(
        defaultKVBits: Int? = nil,
        defaultKVGroupSize: Int = 64,
        prefillStepSize: Int = 512,
        recommendedContextLength: Int = 4096,
        repeatLastN: Int = 64,
        headDim: Int = 128,
        samplingDefaults: FamilySamplingDefaults? = nil
    ) {
        self.defaultKVBits = defaultKVBits
        self.defaultKVGroupSize = defaultKVGroupSize
        self.prefillStepSize = prefillStepSize
        self.recommendedContextLength = recommendedContextLength
        self.repeatLastN = repeatLastN
        self.headDim = headDim
        self.samplingDefaults = samplingDefaults
    }

    public func safeGroupSize(requestedGroupSize: Int? = nil) -> Int {
        let candidate = requestedGroupSize ?? defaultKVGroupSize
        if headDim % candidate == 0 { return candidate }
        let divisors = [32, 64, 48, 16, 128, 96]
        for d in divisors where headDim % d == 0 && d <= candidate { return d }
        for d in divisors.sorted(by: >) where headDim % d == 0 { return d }
        return 32
    }
}

public final class ModelFamilyRegistry: @unchecked Sendable {
    private let lock = NovaMLXLock()
    private var overrides: [String: ModelFamilyOptimization] = [:]

    public static let shared = ModelFamilyRegistry()

    public init() {}

    private static let familyDefaults: [ModelFamily: ModelFamilyOptimization] = [
        .llama: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 512,
            recommendedContextLength: 8192,
            repeatLastN: 64,
            samplingDefaults: FamilySamplingDefaults(temperature: 0.7, topP: 0.9)
        ),
        .mistral: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 512,
            recommendedContextLength: 8192,
            repeatLastN: 64,
            samplingDefaults: FamilySamplingDefaults(temperature: 0.7, topP: 0.9)
        ),
        .phi: ModelFamilyOptimization(
            defaultKVBits: nil,
            defaultKVGroupSize: 32,
            prefillStepSize: 256,
            recommendedContextLength: 4096,
            repeatLastN: 32,
            headDim: 96,
            samplingDefaults: FamilySamplingDefaults(
                temperature: 0.7,
                topP: 0.9,
                frequencyPenalty: 1.0,
                repetitionPenalty: 1.1
            )
        ),
        .qwen: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 512,
            recommendedContextLength: 8192,
            repeatLastN: 64,
            samplingDefaults: FamilySamplingDefaults(temperature: 1.0, topP: 0.95, topK: 20)
        ),
        .gemma: ModelFamilyOptimization(
            defaultKVBits: nil,
            defaultKVGroupSize: 32,
            prefillStepSize: 256,
            recommendedContextLength: 8192,
            repeatLastN: 32,
            headDim: 256,
            samplingDefaults: FamilySamplingDefaults(temperature: 1.0, topP: 0.95, topK: 64)
        ),
        .starcoder: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 1024,
            recommendedContextLength: 16384,
            repeatLastN: 16
        ),
    ]

    private static let architectureMapping: [String: ModelFamily] = [
        "LlamaForCausalLM": .llama,
        "MistralForCausalLM": .mistral,
        "Phi3ForCausalLM": .phi,
        "Qwen2ForCausalLM": .qwen,
        "Qwen2VLForConditionalGeneration": .qwen,
        "Gemma2ForCausalLM": .gemma,
        "Starcoder2ForCausalLM": .starcoder,
    ]

    public func optimization(for modelId: String, family: ModelFamily, architectures: [String]? = nil) -> ModelFamilyOptimization {
        if let override = lock.withLock({ overrides[modelId] }) {
            return override
        }

        if let archs = architectures {
            for arch in archs {
                if let mappedFamily = Self.architectureMapping[arch] {
                    if let opt = Self.familyDefaults[mappedFamily] {
                        return opt
                    }
                }
            }
        }

        return Self.familyDefaults[family] ?? ModelFamilyOptimization()
    }

    public func optimization(for modelId: String, family: ModelFamily) -> ModelFamilyOptimization {
        optimization(for: modelId, family: family, architectures: nil)
    }

    public func setOverride(_ opt: ModelFamilyOptimization, forModel modelId: String) {
        lock.withLock { overrides[modelId] = opt }
        NovaMLXLog.info("Model family optimization override set for \(modelId): kvBits=\(opt.defaultKVBits?.description ?? "nil"), prefill=\(opt.prefillStepSize), context=\(opt.recommendedContextLength)")
    }

    public func removeOverride(forModel modelId: String) {
        _ = lock.withLock { overrides.removeValue(forKey: modelId) }
    }

    public func allOverrides() -> [String: ModelFamilyOptimization] {
        lock.withLock { overrides }
    }
}
