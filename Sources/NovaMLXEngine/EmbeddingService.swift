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

    public func loadModel(from url: URL, config: ModelConfig) async throws -> EmbeddingContainer {
        let container = EmbeddingContainer(
            identifier: config.identifier,
            config: config
        )
        NovaMLXLog.info("Loading embedding model from: \(url.path)")

        let embedderContainer = try await EmbedderModelFactory.shared.loadContainer(
            from: url,
            using: MLXEngine.tokenizerLoader
        )

        container.setLoaded(embedderContainer)

        lock.withLock {
            containers[config.identifier.id] = container
        }

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

    public func embed(model: String, inputs: [String]) async throws -> EmbeddingResult {
        let container = lock.withLock { containers[model] }
        guard let container, container.isLoaded else {
            throw NovaMLXError.modelNotFound(model)
        }
        guard let embedderContainer = container.embedderContainer else {
            throw NovaMLXError.inferenceFailed("Embedding model not loaded")
        }

        let result: [[Float]] = await embedderContainer.perform { context in
            var allTokenIds: [[Int]] = []
            var totalTokenCount = 0

            for text in inputs {
                let tokenIds = context.tokenizer.encode(text: text, addSpecialTokens: true)
                allTokenIds.append(tokenIds)
                totalTokenCount += tokenIds.count
            }

            let maxLength = allTokenIds.reduce(into: 16) { max, tokens in
                max = Swift.max(max, tokens.count)
            }

            let padId = context.tokenizer.eosTokenId ?? 0

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

            let pooled = context.pooling(modelOutput, mask: mask, normalize: true)
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
}
