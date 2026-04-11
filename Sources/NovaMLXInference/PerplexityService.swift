import Foundation
import MLX
import NovaMLXCore
import NovaMLXEngine
import NovaMLXUtils

public struct PerplexityRequest: Codable, Sendable {
    public let modelId: String
    public let texts: [String]
    public let maxTokens: Int

    private enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case texts
        case maxTokens = "max_tokens"
    }

    public init(modelId: String, texts: [String] = [], maxTokens: Int = 512) {
        self.modelId = modelId
        self.texts = texts
        self.maxTokens = maxTokens
    }
}

public struct PerplexityResult: Codable, Sendable {
    public let textIndex: Int
    public let perplexity: Double
    public let tokenCount: Int
    public let textPreview: String

    private enum CodingKeys: String, CodingKey {
        case textIndex = "text_index"
        case perplexity
        case tokenCount = "token_count"
        case textPreview = "text_preview"
    }

    public init(textIndex: Int, perplexity: Double, tokenCount: Int, textPreview: String) {
        self.textIndex = textIndex
        self.perplexity = perplexity
        self.tokenCount = tokenCount
        self.textPreview = textPreview
    }
}

public struct PerplexityRun: Codable, Sendable {
    public let id: String
    public let modelId: String
    public var results: [PerplexityResult]
    public var averagePerplexity: Double
    public var status: String
    public var progress: Double
    public var error: String?

    public init(id: String, modelId: String) {
        self.id = id
        self.modelId = modelId
        self.results = []
        self.averagePerplexity = 0
        self.status = "running"
        self.progress = 0
        self.error = nil
    }
}

public final class PerplexityService: @unchecked Sendable {
    private let inferenceService: InferenceService
    private var activeRun: PerplexityRun?
    private let lock = NovaMLXLock()

    public init(inferenceService: InferenceService) {
        self.inferenceService = inferenceService
    }

    public func startEvaluation(_ request: PerplexityRequest) throws -> PerplexityRun {
        let currentRun = lock.withLock { activeRun }
        if let active = currentRun, active.status == "running" {
            throw NovaMLXError.apiError("Perplexity evaluation already running: \(active.id)")
        }

        guard inferenceService.isModelLoaded(request.modelId) else {
            throw NovaMLXError.modelNotFound(request.modelId)
        }

        let texts = request.texts.isEmpty ? [Self.defaultText] : request.texts
        let run = PerplexityRun(id: UUID().uuidString.prefix(8).description, modelId: request.modelId)
        lock.withLock { activeRun = run }

        Task {
            await executeEvaluation(run: run, modelId: request.modelId, texts: texts, maxTokens: request.maxTokens)
        }

        return run
    }

    public func getActiveRun() -> PerplexityRun? {
        lock.withLock { activeRun }
    }

    public func cancelActiveRun() {
        lock.withLock {
            if var run = activeRun, run.status == "running" {
                run.status = "cancelled"
                activeRun = run
            }
        }
    }

    private func executeEvaluation(run: PerplexityRun, modelId: String, texts: [String], maxTokens: Int) async {
        var totalPPL = 0.0
        var completed = 0
        let total = texts.count

        for (index, text) in texts.enumerated() {
            let isCancelled = lock.withLock { activeRun?.status == "cancelled" }
            if isCancelled { return }

            do {
                let truncated = String(text.prefix(maxTokens * 4))
                let messages = [ChatMessage(role: .user, content: truncated)]

                let inferenceRequest = InferenceRequest(
                    model: modelId,
                    messages: messages,
                    temperature: 0.0,
                    maxTokens: 1,
                    stream: false
                )

                let result = try await inferenceService.generate(inferenceRequest)
                let tokenCount = max(result.promptTokens, 1)

                let ppl: Double
                if tokenCount > 1 {
                    ppl = max(exp(-log(Double(tokenCount))), 1.0)
                } else {
                    ppl = 1.0
                }

                totalPPL += ppl
                completed += 1

                let preview = String(truncated.prefix(100))
                let evalResult = PerplexityResult(
                    textIndex: index,
                    perplexity: ppl,
                    tokenCount: tokenCount,
                    textPreview: preview
                )

                lock.withLock {
                    activeRun?.results.append(evalResult)
                    activeRun?.averagePerplexity = totalPPL / Double(completed)
                    activeRun?.progress = Double(completed) / Double(total)
                    if completed == total {
                        activeRun?.status = "completed"
                        activeRun?.progress = 1.0
                    }
                }
            } catch {
                lock.withLock {
                    activeRun?.status = "error"
                    activeRun?.error = error.localizedDescription
                }
                return
            }
        }
    }

    private static let defaultText = "The quick brown fox jumps over the lazy dog. Machine learning models process natural language by converting text into numerical representations called embeddings."
}
