import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public final class EmbeddingContainer: @unchecked Sendable {
    public let identifier: ModelIdentifier
    public let config: ModelConfig
    public private(set) var embedderContainer: EmbedderModelContainer?
    public private(set) var isLoaded: Bool

    public init(identifier: ModelIdentifier, config: ModelConfig) {
        self.identifier = identifier
        self.config = config
        self.isLoaded = false
    }

    public func setLoaded(_ container: EmbedderModelContainer) {
        self.embedderContainer = container
        self.isLoaded = true
        NovaMLXLog.info("Embedding model loaded: \(identifier.displayName)")
    }

    public func unload() {
        embedderContainer = nil
        isLoaded = false
        MLX.Memory.clearCache()
        NovaMLXLog.info("Embedding model unloaded: \(identifier.displayName)")
    }
}

public struct EmbeddingResult: Sendable {
    public let model: String
    public let embeddings: [[Float]]
    public let promptTokens: Int
    public let totalTokens: Int

    public init(model: String, embeddings: [[Float]], promptTokens: Int, totalTokens: Int) {
        self.model = model
        self.embeddings = embeddings
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
    }
}

public final class EmbeddingService: @unchecked Sendable {
    private var containers: [String: EmbeddingContainer]
    private let lock = NovaMLXLock()

    public init() {
        self.containers = [:]
    }

    public func loadModel(from url: URL, config: ModelConfig, progress: (@Sendable (LoadPhase) -> Void)? = nil) async throws -> EmbeddingContainer {
        let container = EmbeddingContainer(
            identifier: config.identifier,
            config: config
        )
        NovaMLXLog.info("Loading embedding model from: \(url.path)")

        progress?(.loadingWeights)

        let embedderContainer = try await EmbedderModelFactory.shared.loadContainer(
            from: url,
            using: MLXEngine.tokenizerLoader
        )

        container.setLoaded(embedderContainer)

        progress?(.warmingUp)

        lock.withLock {
            containers[config.identifier.id] = container
        }

        progress?(.ready)
        return container
    }

    public func unloadModel(_ modelId: String) {
        lock.withLock {
            guard let container = containers.removeValue(forKey: modelId) else { return }
            container.unload()
        }
    }

    public func isLoaded(_ modelId: String) -> Bool {
        lock.withLock {
            containers[modelId]?.isLoaded ?? false
        }
    }

    public func loadedModelIds() -> [String] {
        lock.withLock { Array(containers.keys) }
    }

    public func embed(model: String, inputs: [String]) async throws -> EmbeddingResult {
        let container = lock.withLock { containers[model] }
        guard let container, container.isLoaded else {
            throw NovaMLXError.modelNotFound(model)
        }
        guard let embedderContainer = container.embedderContainer else {
            throw NovaMLXError.inferenceFailed("Embedding model not loaded")
        }

        let useBgeM3 = BgeM3EmbeddingProfile.matches(modelId: model)
        let result: [[Float]] = await embedderContainer.perform { context in
            var allTokenIds: [[Int]] = []
            var totalTokenCount = 0

            for text in inputs {
                var tokenIds = context.tokenizer.encode(text: text, addSpecialTokens: true)
                if useBgeM3 {
                    tokenIds = BgeM3EmbeddingProfile.truncate(tokenIds)
                }
                allTokenIds.append(tokenIds)
                totalTokenCount += tokenIds.count
            }

            let maxLength = allTokenIds.reduce(into: 16) { max, tokens in
                max = Swift.max(max, tokens.count)
            }

            // bge-m3's </s> is a real token (id 2). Padding with eos hides it
            // from the attention mask. The checkpoint pad id is <pad> (1).
            let padId = useBgeM3
                ? BgeM3EmbeddingProfile.padTokenId
                : (context.tokenizer.eosTokenId ?? 0)

            let padded = MLX.stacked(
                allTokenIds.map { tokens in
                    MLXArray(tokens + Array(repeating: padId, count: maxLength - tokens.count))
                }
            )
            let mask = (padded .!= MLXArray(padId))
            let tokenTypes = MLXArray.zeros(like: padded)

            let modelOutput = context.model(
                padded,
                positionIds: nil,
                tokenTypeIds: tokenTypes,
                attentionMask: mask
            )

            let pooled: MLXArray
            if useBgeM3, let hidden = modelOutput.hiddenStates {
                // Dedup's bge-m3 service mean-pools token states. This checkpoint
                // has no 1_Pooling file, so the shared pooler is the tanh CLS
                // head, which does not match that service. Other embedders keep
                // context.pooling.
                pooled = Self.meanPoolNormalized(hidden, mask: mask)
            } else {
                pooled = context.pooling(modelOutput, mask: mask, normalize: true)
            }
            pooled.eval()

            return pooled.map { $0.asArray(Float.self) }
        }

        MLX.Memory.clearCache()

        let promptTokens = result.isEmpty ? 0 : inputs.reduce(0) { sum, text in
            sum + text.count / 4
        }

        return EmbeddingResult(
            model: model,
            embeddings: result,
            promptTokens: promptTokens,
            totalTokens: promptTokens
        )
    }

    /// Masked mean of token states, then L2. Mask is 1 for real tokens and 0 for pad.
    private static func meanPoolNormalized(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        let typedMask = mask.asType(hidden.dtype)
        let weights = typedMask.expandedDimensions(axes: [-1])
        let pooled = sum(hidden * weights, axis: 1) / sum(typedMask, axis: -1, keepDims: true)
        let norm = sqrt(sum(pooled * pooled, axis: -1, keepDims: true))
        return pooled / maximum(norm, MLXArray(Float(1e-12)))
    }
}

enum BgeM3EmbeddingProfile {
    static let maxLength = 256

    static func matches(modelId: String) -> Bool {
        modelId.lowercased().contains("bge-m3")
    }

    /// Hugging Face single-sequence truncation: keep the final special token inside `maxLength`.
    static func truncate(_ tokenIds: [Int]) -> [Int] {
        guard tokenIds.count > maxLength else { return tokenIds }
        return Array(tokenIds.prefix(maxLength - 1)) + [tokenIds[tokenIds.count - 1]]
    }

    /// XLM-RoBERTa `<pad>`. `</s>` is 2 and must stay visible to the mask.
    static let padTokenId = 1
}
