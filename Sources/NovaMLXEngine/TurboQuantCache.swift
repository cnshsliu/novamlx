import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public final class TurboQuantService: @unchecked Sendable {
    public struct Config: Codable, Sendable {
        public let bits: Int
        public let groupSize: Int

        public init(bits: Int = 4, groupSize: Int = 64) {
            self.bits = bits
            self.groupSize = groupSize
        }

        public var estimatedCompressionRatio: Double {
            switch bits {
            case 2: return 8.0
            case 3: return 5.33
            case 4: return 4.0
            case 6: return 2.67
            case 8: return 2.0
            default: return 4.0
            }
        }

        public var bytesPerElement: Double {
            Double(bits) / 8.0
        }

        public func estimatedMemoryMB(
            headDim: Int,
            nKVHeads: Int,
            seqLen: Int,
            nLayers: Int,
            batchSize: Int
        ) -> Double {
            let totalElements = Double(batchSize * nKVHeads * seqLen * headDim * nLayers)
            let quantBytes = bytesPerElement * totalElements
            let scaleBytes = Double(batchSize * nKVHeads * seqLen * (headDim / groupSize) * nLayers) * 2.0
            return (quantBytes + scaleBytes) / (1024 * 1024)
        }

        public func savingsMB(
            headDim: Int,
            nKVHeads: Int,
            seqLen: Int,
            nLayers: Int,
            batchSize: Int
        ) -> Double {
            let fp16MB = 2.0 * Double(batchSize * nKVHeads * seqLen * headDim * nLayers * 2) / (1024 * 1024)
            return fp16MB - estimatedMemoryMB(headDim: headDim, nKVHeads: nKVHeads, seqLen: seqLen, nLayers: nLayers, batchSize: batchSize)
        }
    }

    private let lock = NovaMLXLock()
    private var activeConfigs: [String: Config] = [:]

    public init() {}

    public func setConfig(_ config: Config, forModel modelId: String) {
        lock.withLock { activeConfigs[modelId] = config }
    }

    public func getConfig(forModel modelId: String) -> Config? {
        lock.withLock { activeConfigs[modelId] }
    }

    public func removeConfig(forModel modelId: String) {
        _ = lock.withLock { activeConfigs.removeValue(forKey: modelId) }
    }

    public func allConfigs() -> [String: Config] {
        lock.withLock { activeConfigs }
    }

    public func applyToGenerateParameters(
        _ params: inout GenerateParameters,
        modelId: String,
        settings: ModelSettings
    ) {
        if let existingConfig = getConfig(forModel: modelId) {
            let safeGroupSize = validatedGroupSize(modelId: modelId, requested: existingConfig.groupSize)
            params.kvBits = existingConfig.bits
            params.kvGroupSize = safeGroupSize
            NovaMLXLog.info("TurboQuant applied for \(modelId): \(existingConfig.bits)-bit affine, group_size=\(safeGroupSize), ~\(String(format: "%.1f", existingConfig.estimatedCompressionRatio))x compression")
            return
        }

        let bits = settings.kvBits
        let groupSize = settings.kvGroupSize ?? 64

        guard let b = bits, b > 0 else { return }

        let safeGroupSize = validatedGroupSize(modelId: modelId, requested: groupSize)
        params.kvBits = b
        params.kvGroupSize = safeGroupSize

        let config = Config(bits: b, groupSize: safeGroupSize)
        setConfig(config, forModel: modelId)

        NovaMLXLog.info("TurboQuant affine KV cache enabled for \(modelId): \(b)-bit, group_size=\(safeGroupSize), ~\(String(format: "%.1f", config.estimatedCompressionRatio))x compression")
    }

    private func validatedGroupSize(modelId: String, requested: Int) -> Int {
        let familyOpt = ModelFamilyRegistry.shared.optimization(for: modelId, family: familyFromModelId(modelId))
        let headDim = familyOpt.headDim
        if headDim % requested == 0 { return requested }
        let corrected = familyOpt.safeGroupSize(requestedGroupSize: requested)
        NovaMLXLog.info("TurboQuant adjusted group_size from \(requested) to \(corrected) for head_dim=\(headDim)")
        return corrected
    }

    private func familyFromModelId(_ modelId: String) -> ModelFamily {
        let lower = modelId.lowercased()
        if lower.contains("phi") { return .phi }
        if lower.contains("llama") { return .llama }
        if lower.contains("mistral") { return .mistral }
        if lower.contains("qwen") { return .qwen }
        if lower.contains("gemma") { return .gemma }
        if lower.contains("starcoder") { return .starcoder }
        return .other
    }

    public static func recommendedBits(
        for modelSizeGB: Double,
        contextLength: Int,
        availableMemoryGB: Double
    ) -> Int {
        let memoryPressure = modelSizeGB / availableMemoryGB
        if memoryPressure > 0.7 { return 2 }
        if memoryPressure > 0.5 { return 3 }
        if contextLength > 8192 { return 4 }
        return 4
    }

    public func autoConfigure(
        modelId: String,
        modelSizeGB: Double,
        contextLength: Int,
        availableMemoryGB: Double
    ) -> Config {
        let bits = Self.recommendedBits(for: modelSizeGB, contextLength: contextLength, availableMemoryGB: availableMemoryGB)
        let config = Config(bits: bits, groupSize: 64)
        setConfig(config, forModel: modelId)
        NovaMLXLog.info("TurboQuant auto-configured for \(modelId): \(bits)-bit (model=\(String(format: "%.1f", modelSizeGB))GB, context=\(contextLength), available=\(String(format: "%.1f", availableMemoryGB))GB)")
        return config
    }

    public struct Stats: Codable, Sendable {
        public let modelId: String
        public let bits: Int
        public let groupSize: Int
        public let compressionRatio: Double
    }

    public func getStats(modelId: String) -> Stats? {
        guard let config = getConfig(forModel: modelId) else { return nil }
        return Stats(
            modelId: modelId,
            bits: config.bits,
            groupSize: config.groupSize,
            compressionRatio: config.estimatedCompressionRatio
        )
    }
}
