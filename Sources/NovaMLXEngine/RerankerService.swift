import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public final class RerankerContainer: @unchecked Sendable {
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
        NovaMLXLog.info("Reranker model loaded: \(identifier.displayName)")
    }

    public func unload() {
        embedderContainer = nil
        isLoaded = false
        MLX.Memory.clearCache()
        NovaMLXLog.info("Reranker model unloaded: \(identifier.displayName)")
    }
}

public struct RerankResultInternal: Sendable {
    public let index: Int
    public let score: Double
    public let totalTokens: Int

    public init(index: Int, score: Double, totalTokens: Int) {
        self.index = index
        self.score = score
        self.totalTokens = totalTokens
    }
}

public final class RerankerService: @unchecked Sendable {
    private var containers: [String: RerankerContainer]
    private let lock = NovaMLXLock()

    public init() {
        self.containers = [:]
    }

    public func loadModel(from url: URL, config: ModelConfig) async throws -> RerankerContainer {
        let container = RerankerContainer(
            identifier: config.identifier,
            config: config
        )
        NovaMLXLog.info("Loading reranker model from: \(url.path)")

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

    public func rerank(model: String, query: String, documents: [String], topN: Int?) async throws -> [RerankResultInternal] {
        let container = lock.withLock { containers[model] }
        guard let container, container.isLoaded else {
            throw NovaMLXError.modelNotFound(model)
        }
        guard let embedderContainer = container.embedderContainer else {
            throw NovaMLXError.inferenceFailed("Reranker model not loaded")
        }

        let results: [RerankResultInternal] = await embedderContainer.perform { context in
            var scores: [(index: Int, score: Double, tokens: Int)] = []
            var totalTokens = 0

            for (idx, doc) in documents.enumerated() {
                let pair = "\(query)</s></s>\(doc)"
                let tokenIds = context.tokenizer.encode(text: pair, addSpecialTokens: true)
                totalTokens += tokenIds.count

                let input = MLXArray(tokenIds)[.newAxis, 0...]
                let mask = MLXArray.ones(input.shape)
                let tokenTypes = MLXArray.zeros(input.shape)

                let output = context.model(
                    input,
                    positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask
                )

                let pooled = context.pooling(output, mask: mask, normalize: false)
                pooled.eval()

                let scoreValue = pooled.asArray(Float.self).first ?? 0.0
                scores.append((index: idx, score: Double(scoreValue), tokens: totalTokens))
            }

            let maxScore = scores.map(\.score).max() ?? 0.0
            let minScore = scores.map(\.score).min() ?? 0.0
            let range = maxScore - minScore
            let safeRange = range > 0 ? range : 1.0

            return scores.map { s in
                let normalized = (s.score - minScore) / safeRange
                return RerankResultInternal(index: s.index, score: normalized, totalTokens: s.tokens)
            }
        }

        MLX.Memory.clearCache()

        var sorted = results.sorted { $0.score > $1.score }
        if let topN, topN > 0, topN < sorted.count {
            sorted = Array(sorted.prefix(topN))
        }

        return sorted
    }
}
