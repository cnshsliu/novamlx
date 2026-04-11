import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine

public final class LLMInferencePipeline {
    private let engine: MLXEngine

    public init(engine: MLXEngine) {
        self.engine = engine
    }

    public func complete(
        model: String,
        prompt: String,
        temperature: Double = 0.7,
        maxTokens: Int = 4096,
        stop: [String]? = nil
    ) async throws -> InferenceResult {
        let request = InferenceRequest(
            model: model,
            messages: [ChatMessage(role: .user, content: prompt)],
            temperature: temperature,
            maxTokens: maxTokens,
            stream: false,
            stop: stop
        )
        return try await engine.generate(request)
    }

    public func chat(
        model: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 4096
    ) async throws -> InferenceResult {
        let request = InferenceRequest(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            stream: false
        )
        return try await engine.generate(request)
    }

    public func streamChat(
        model: String,
        messages: [ChatMessage],
        temperature: Double = 0.7,
        maxTokens: Int = 4096
    ) -> AsyncThrowingStream<Token, Error> {
        let request = InferenceRequest(
            model: model,
            messages: messages,
            temperature: temperature,
            maxTokens: maxTokens,
            stream: true
        )
        return engine.stream(request)
    }
}

public final class VLMInferencePipeline {
    private let engine: MLXEngine

    public init(engine: MLXEngine) {
        self.engine = engine
    }

    public func analyze(
        model: String,
        prompt: String,
        imageData: Data,
        temperature: Double = 0.7,
        maxTokens: Int = 2048
    ) async throws -> InferenceResult {
        let base64Image = imageData.base64EncodedString()
        let content = "\(prompt)\n[data:image/png;base64,\(base64Image)]"

        let request = InferenceRequest(
            model: model,
            messages: [ChatMessage(role: .user, content: content)],
            temperature: temperature,
            maxTokens: maxTokens,
            stream: false
        )
        return try await engine.generate(request)
    }

    public func streamAnalyze(
        model: String,
        prompt: String,
        imageData: Data,
        temperature: Double = 0.7,
        maxTokens: Int = 2048
    ) -> AsyncThrowingStream<Token, Error> {
        let base64Image = imageData.base64EncodedString()
        let content = "\(prompt)\n[data:image/png;base64,\(base64Image)]"

        let request = InferenceRequest(
            model: model,
            messages: [ChatMessage(role: .user, content: content)],
            temperature: temperature,
            maxTokens: maxTokens,
            stream: true
        )
        return engine.stream(request)
    }
}
