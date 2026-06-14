import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils

// MARK: - LB Local Dispatch Helpers
// Used by the LBProxy closure to route `.local` members. Each helper decodes
// the (already model-rewritten) body into the appropriate request type and
// dispatches via the existing local handler chain.
//
// v1 note: these helpers reuse the extracted streaming handlers
// (handleStreamChat / handleStreamAnthropic / handleStreamResponses) and the
// non-streaming chat handler (handleChat). For non-streaming /v1/messages and
// /v1/responses there is no extracted non-stream handler, so those paths
// currently use inference.generate directly with a minimal response wrapper.
// Full fidelity (modelfile resolution, OCR auto-optimization, ensureModelReady
// load triggering) is intentionally skipped — LB members are expected to point
// at concrete model IDs that the caller has already loaded. If the model is
// not loaded, the request will fail at inference.generate and the LB will
// retry the next candidate.

extension NovaMLXAPIServer {

    /// Rewrite the "model" field in a JSON request body to `newModel`.
    /// If the body isn't valid JSON or has no "model" field, returns the
    /// original body unchanged.
    static func rewriteModel(in body: Data, to newModel: String) -> Data {
        guard var dict = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            return body
        }
        dict["model"] = newModel
        return (try? JSONSerialization.data(withJSONObject: dict)) ?? body
    }

    // MARK: - Chat Completions

    /// Dispatch a rewritten body through the local chat handler chain.
    /// `path` is one of "chat/completions".
    static func dispatchLocalChat(
        rawBody: Data, path: String,
        inference: InferenceService, cfg: ServerConfig,
        clientType: ClientType, coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let openAIReq = try JSONDecoder().decode(OpenAIRequest.self, from: rawBody)
        let messages = mapOpenAIMessages(openAIReq.messages)
        let sessionId = openAIReq.sessionId
        let responseFormat: ResponseFormat?
        var jsonSchemaDef: [String: Any]? = nil
        var regexPattern: String? = nil
        var gbnfGrammar: String? = nil
        if openAIReq.responseFormat?.type == "json_schema",
           let schemaField = openAIReq.responseFormat?.jsonSchema,
           let schemaDict = schemaField.schema {
            responseFormat = .jsonObject
            jsonSchemaDef = schemaDict.toDict()
        } else if openAIReq.responseFormat?.type == "json_object" {
            responseFormat = .jsonObject
        } else if openAIReq.responseFormat?.type == "regex",
                  let pattern = openAIReq.responseFormat?.regex {
            responseFormat = nil
            regexPattern = pattern
        } else if openAIReq.responseFormat?.type == "gbnf",
                  let grammar = openAIReq.responseFormat?.gbnf {
            responseFormat = nil
            gbnfGrammar = grammar
        } else {
            responseFormat = nil
        }

        if openAIReq.stream ?? false {
            return try await Self.handleStreamChat(
                openAIReq: openAIReq, messages: messages, inference: inference,
                sessionId: sessionId, responseFormat: responseFormat, jsonSchemaDef: jsonSchemaDef,
                regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
                cfg: cfg, clientType: clientType, coordinator: coordinator
            )
        } else {
            return try await Self.handleChat(
                openAIReq: openAIReq, messages: messages, inference: inference,
                sessionId: sessionId, responseFormat: responseFormat, jsonSchemaDef: jsonSchemaDef,
                regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
                cfg: cfg, clientType: clientType
            )
        }
    }

    // MARK: - Anthropic Messages

    /// Dispatch a rewritten body through the local messages handler chain.
    static func dispatchLocalMessages(
        rawBody: Data, path: String,
        inference: InferenceService, cfg: ServerConfig,
        clientType: ClientType, coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let anthropicReq = try JSONDecoder().decode(AnthropicRequest.self, from: rawBody)
        let messages = try mapAnthropicMessages(anthropicReq.messages, system: anthropicReq.system)

        if anthropicReq.stream ?? false {
            return try await Self.handleStreamAnthropic(
                anthropicReq: anthropicReq, messages: messages, inference: inference,
                cfg: cfg, clientType: clientType, coordinator: coordinator
            )
        }
        // Non-streaming Anthropic: no extracted handler, so build a minimal
        // response via inference.generate. This intentionally skips OCR and
        // modelfile resolution — LB members are concrete model IDs.
        let ocrSampling = OCROptimizer.samplingOverrides(
            modelName: anthropicReq.model,
            userTemperature: anthropicReq.temperature,
            userMaxTokens: anthropicReq.maxTokens,
            userRepetitionPenalty: nil
        )
        let ocrStop = OCROptimizer.applyStopSequences(anthropicReq.stopSequences, modelName: anthropicReq.model)
        let request = InferenceRequest(
            model: anthropicReq.model, messages: messages,
            temperature: ocrSampling.temperature,
            maxTokens: ocrSampling.maxTokens,
            topP: anthropicReq.topP, topK: anthropicReq.topK,
            stream: false, stop: ocrStop,
            thinkingBudget: anthropicReq.resolvedThinkingBudget,
            enableThinking: anthropicReq.resolvedEnableThinking,
            preserveThinking: anthropicReq.resolvedPreserveThinking
        )
        CurrentInferenceModel.shared.modelID = request.model
        defer { CurrentInferenceModel.shared.modelID = nil }
        let result = try await inference.generate(request)
        let response = AnthropicResponse(
            id: "msg_\(result.id.uuidString.prefix(12))",
            model: anthropicReq.model,
            content: [.init(type: "text", text: result.text)],
            stopReason: result.finishReason.rawValue,
            usage: .init(
                inputTokens: result.promptTokens,
                outputTokens: result.completionTokens
            )
        )
        return try Self.jsonResponse(response)
    }

    // MARK: - Responses API

    /// Dispatch a rewritten body through the local responses handler chain.
    static func dispatchLocalResponses(
        rawBody: Data, path: String,
        inference: InferenceService, cfg: ServerConfig,
        clientType: ClientType, coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let req = try JSONDecoder().decode(OpenAIResponseRequest.self, from: rawBody)
        if req.stream ?? false {
            return try await Self.handleStreamResponses(
                req: req, inference: inference, cfg: cfg,
                clientType: clientType, coordinator: coordinator
            )
        }
        return try await Self.handleResponsesRequest(
            req: req, inference: inference, cfg: cfg,
            clientType: clientType, coordinator: coordinator
        )
    }
}
