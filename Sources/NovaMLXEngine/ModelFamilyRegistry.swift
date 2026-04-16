import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct ModelFamilyOptimization: Sendable {
    public let defaultKVBits: Int?
    public let defaultKVGroupSize: Int
    public let prefillStepSize: Int
    public let recommendedContextLength: Int
    public let repeatLastN: Int
    public let headDim: Int

    public init(
        defaultKVBits: Int? = nil,
        defaultKVGroupSize: Int = 64,
        prefillStepSize: Int = 512,
        recommendedContextLength: Int = 4096,
        repeatLastN: Int = 64,
        headDim: Int = 128
    ) {
        self.defaultKVBits = defaultKVBits
        self.defaultKVGroupSize = defaultKVGroupSize
        self.prefillStepSize = prefillStepSize
        self.recommendedContextLength = recommendedContextLength
        self.repeatLastN = repeatLastN
        self.headDim = headDim
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
            repeatLastN: 64
        ),
        .mistral: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 512,
            recommendedContextLength: 8192,
            repeatLastN: 64
        ),
        .phi: ModelFamilyOptimization(
            defaultKVBits: nil,
            defaultKVGroupSize: 32,
            prefillStepSize: 256,
            recommendedContextLength: 4096,
            repeatLastN: 32,
            headDim: 96
        ),
        .qwen: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 512,
            recommendedContextLength: 8192,
            repeatLastN: 64
        ),
        .gemma: ModelFamilyOptimization(
            defaultKVBits: 4,
            prefillStepSize: 256,
            recommendedContextLength: 8192,
            repeatLastN: 32
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
