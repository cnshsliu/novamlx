import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

// MARK: - Responses API Handlers
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    // MARK: - Local Responses API Handlers

    static func handleResponsesRequest(
        req: OpenAIResponseRequest, inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        var messages = mapResponsesInput(req)

        if Self.isMovedToPeerGateway(req.model) {
            return try Self.peerGatewayMovedResponse()
        }

        // Resolve previous_response_id (or conversation.id alias): prepend stored messages
        if let prevId = req.resolvedPreviousResponseId {
            if let prevResp = ResponseStore.shared.get(prevId) {
                let prevMessages = Self.extractMessagesFromResponse(prevResp)
                messages = prevMessages + messages
            } else {
                throw NovaMLXError.apiError("previous_response_id '\(prevId)' not found or expired")
            }
        }

        // Convert text.format to response_format
        var responseFormat: ResponseFormat? = nil
        var jsonSchemaDef: [String: Any]? = nil
        if let fmt = req.text?.format, fmt.type == "json_schema" || fmt.type == "json_object" {
            responseFormat = .jsonObject
            if fmt.type == "json_schema", let schema = fmt.schema {
                jsonSchemaDef = unwrapAnyCodable(schema) as? [String: Any]
            }
        }

        // Convert tools to internal format
        let tools: [[String: Any]]? = req.tools?.compactMap { tool in
            guard tool.type == "function" else { return nil }
            var dict: [String: Any] = [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description ?? ""
                ] as [String: Any]
            ]
            if let params = tool.parameters {
                var fnDict = dict["function"] as! [String: Any]
                fnDict["parameters"] = unwrapAnyCodable(params)
                dict["function"] = fnDict
            }
            return dict
        }

        let request = InferenceRequest(
            model: req.model, messages: messages,
            tools: tools,
            temperature: req.temperature, maxTokens: req.maxOutputTokens,
            topP: req.topP, stream: false,
            responseFormat: responseFormat,
            jsonSchemaDef: jsonSchemaDef,
            enableThinking: req.reasoning != nil,
            httpRequestId: HTTPHelpers.requestID(from: nil)
        )

        CurrentInferenceModel.shared.modelID = request.model
        defer { CurrentInferenceModel.shared.modelID = nil }
        let result = try await inference.generate(request)
        let responseId = "resp_\(result.id.uuidString.prefix(24))"
        let outputItemId = "msg_\(result.id.uuidString.prefix(24))"

        var outputItems: [ResponseOutputItem] = []
        outputItems.append(.message(ResponseOutputMessage(
            id: outputItemId,
            content: [ResponseContentItem(text: result.text)]
        )))
        if let toolCalls = result.toolCalls {
            for tc in toolCalls {
                outputItems.append(.functionCall(ResponseOutputFunctionCall(
                    id: "fc_\(tc.id.prefix(24))",
                    callId: tc.id,
                    name: tc.functionName,
                    arguments: tc.arguments
                )))
            }
        }

        let ctxWin = inference.getContextWindow(for: req.model) ?? 0
        let scaledInput = clientType.shouldScaleContext
            ? cfg.scaleTokenCount(result.promptTokens, modelContextWindow: ctxWin)
            : result.promptTokens
        let scaledOutput = clientType.shouldScaleContext
            ? cfg.scaleTokenCount(result.completionTokens, modelContextWindow: ctxWin)
            : result.completionTokens

        let response = OpenAIResponseObject(
            id: responseId,
            model: result.model,
            output: outputItems,
            usage: ResponsesUsage(inputTokens: scaledInput, outputTokens: scaledOutput)
        ).withRequestEcho(from: req)
        // Store with user messages prepended for previous_response_id multi-turn
        var storeOutputItems: [ResponseOutputItem] = []
        for msg in messages {
            let itemId = "msg_user_\(UUID().uuidString.prefix(12))"
            storeOutputItems.append(.message(ResponseOutputMessage(id: itemId, role: msg.role.rawValue, content: [ResponseContentItem(text: msg.content ?? "")])))
        }
        storeOutputItems.append(contentsOf: outputItems)
        let storeResponse = OpenAIResponseObject(id: responseId, model: result.model, output: storeOutputItems, usage: ResponsesUsage(inputTokens: scaledInput, outputTokens: scaledOutput))
        ResponseStore.shared.put(storeResponse)
        if let convId = req.conversation?.id {
            ConversationStore.shared.record(conversationId: convId, responseId: responseId)
        }
        return try jsonResponse(response)
    }

    /// Extract ChatMessages from a stored response for conversation continuation
    static func extractMessagesFromResponse(_ response: OpenAIResponseObject) -> [ChatMessage] {
        var messages: [ChatMessage] = []
        for item in response.output {
            switch item {
            case .message(let msg):
                let role: ChatMessage.Role = msg.role == "assistant" ? .assistant : .user
                let text = msg.content.map { $0.text }.joined()
                messages.append(ChatMessage(role: role, content: text))
            case .functionCall(let fc):
                let toolCalls = [ToolCallResult(id: fc.callId, functionName: fc.name, arguments: fc.arguments)]
                messages.append(ChatMessage(role: .assistant, content: nil, toolCalls: toolCalls))
            case .reasoning:
                break
            }
        }
        return messages
    }

    // MARK: - Responses API Streaming

    static func handleStreamResponses(
        req: OpenAIResponseRequest, inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        var messages = mapResponsesInput(req)

        // Resolve previous_response_id
        if let prevId = req.previousResponseId {
            if let prevResp = ResponseStore.shared.get(prevId) {
                let prevMessages = Self.extractMessagesFromResponse(prevResp)
                messages = prevMessages + messages
            } else {
                throw NovaMLXError.apiError("previous_response_id '\(prevId)' not found or expired")
            }
        }

        let capturedMessages = messages
        let request = InferenceRequest(
            model: req.model, messages: messages,
            temperature: req.temperature, maxTokens: req.maxOutputTokens,
            topP: req.topP, stream: true,
            enableThinking: req.reasoning != nil,
            httpRequestId: HTTPHelpers.requestID(from: nil)
        )

        let reqTag = request.id.uuidString.prefix(8)
        NovaMLXLog.info("[SSE:\(reqTag)] Responses stream start — model=\(req.model) client=\(clientType) maxTokens=\(req.maxOutputTokens ?? -1) promptMsgs=\(capturedMessages.count) previousResponseId=\(req.previousResponseId ?? "nil")")
        let responseId = "resp_\(request.id.uuidString.prefix(24))"
        let msgId = "msg_\(request.id.uuidString.prefix(24))"
        let rsId = "rs_\(request.id.uuidString.prefix(24))"
        let modelId = req.model

        let keepAliveStream = Self.withSSEKeepAlive(
            Self.loadAwareStream(
                modelId: modelId, inference: inference,
                coordinator: coordinator, autoLoadCfg: cfg.autoLoad,
                inferenceStreamProducer: { inference.stream(request) }
            ),
            reqTag: String(reqTag)
        )

        let body: ResponseBody = .init { writer in
            CurrentInferenceModel.shared.modelID = modelId
            defer { CurrentInferenceModel.shared.modelID = nil }
            var fullText = ""
            var reasoningText = ""
            var tokenCount = 0
            let streamStart = Date()
            var firstTokenAt: Date? = nil
            var lastProgressAt = streamStart
            var lastProgressCount = 0
            var lastFinishReason: String? = nil
            let encoder = JSONEncoder()
            let isImplicitModel = ModelContainer.isImplicitThinkingModel(for: modelId)
            let thinkingParser = ThinkingParser(expectImplicitThinking: isImplicitModel)
            var outputItems: [ResponseOutputItem] = []
            var currentOutputIndex = 0
            var textMessageStarted = false
            var reasoningStarted = false
            var sequenceNumber = 0

            func sse(_ event: String, _ data: Encodable) async throws {
                sequenceNumber += 1
                var jsonData = try encoder.encode(data)
                if var obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                    obj["sequence_number"] = sequenceNumber
                    if let injected = try? JSONSerialization.data(withJSONObject: obj) {
                        jsonData = injected
                    }
                }
                try await writer.write(ByteBuffer(string: "event: \(event)\ndata: \(String(data: jsonData, encoding: .utf8) ?? "{}")\n\n"))
            }

            do {
                let emptyResp = ResponsesSSEResponse(id: responseId, status: "in_progress", model: modelId)
                try await sse("response.created", ResponsesSSECreated(response: emptyResp))
                try await sse("response.in_progress", ResponsesSSECreated(response: emptyResp))

                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        if !token.text.isEmpty {
                            let parsed = thinkingParser.feed(token.text)
                            if parsed.type == .thinking && !parsed.text.isEmpty {
                                if !reasoningStarted {
                                    reasoningStarted = true
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, status: "in_progress"))))
                                }
                                reasoningText += parsed.text
                                try await sse("response.reasoning.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: parsed.text))
                                try await sse("response.reasoning_text.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: parsed.text))
                            }
                            if parsed.type == .content && !parsed.text.isEmpty {
                                if !textMessageStarted {
                                    if reasoningStarted {
                                        let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                                        try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                        try await sse("response.reasoning_text.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                        try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, summary: summary))))
                                        outputItems.append(.reasoning(ResponseOutputReasoning(id: rsId, summary: summary)))
                                        currentOutputIndex += 1
                                        reasoningStarted = false
                                    }
                                    textMessageStarted = true
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .message(ResponseOutputMessage(id: msgId, status: "in_progress", content: []))))
                                    try await sse("response.content_part.added", ResponsesSSEContentPartAdded(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, part: ResponseContentItem(text: "")))
                                }
                                fullText += parsed.text
                                try await sse("response.output_text.delta", ResponsesSSETextDelta(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, delta: parsed.text))
                            }
                        }
                        tokenCount += 1
                        // TTFT + progress sampling
                        if firstTokenAt == nil {
                            firstTokenAt = Date()
                            let ttftMs = firstTokenAt!.timeIntervalSince(streamStart) * 1000
                            NovaMLXLog.info("[SSE:\(reqTag)] TTFT=\(String(format: "%.0f", ttftMs))ms — first content token")
                        }
                        let now = Date()
                        let sinceLast = now.timeIntervalSince(lastProgressAt)
                        if tokenCount - lastProgressCount >= 50 || sinceLast >= 5 {
                            let delta = tokenCount - lastProgressCount
                            let tps = sinceLast > 0 ? Double(delta) / sinceLast : 0
                            NovaMLXLog.info("[SSE:\(reqTag)] Progress: \(tokenCount) tokens, \(String(format: "%.1f", tps)) tok/s (last \(delta) in \(String(format: "%.1f", sinceLast))s)")
                            lastProgressCount = tokenCount
                            lastProgressAt = now
                        }
                        if token.finishReason != nil {
                            lastFinishReason = token.finishReason?.rawValue ?? "unknown"
                            let finalParsed = thinkingParser.finalize()
                            if !finalParsed.thinking.isEmpty {
                                if !reasoningStarted {
                                    reasoningStarted = true
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, status: "in_progress"))))
                                }
                                reasoningText += finalParsed.thinking
                                try await sse("response.reasoning.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: finalParsed.thinking))
                                try await sse("response.reasoning_text.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: finalParsed.thinking))
                            }
                            if reasoningStarted {
                                let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                                try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                try await sse("response.reasoning_text.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, summary: summary))))
                                outputItems.append(.reasoning(ResponseOutputReasoning(id: rsId, summary: summary)))
                                currentOutputIndex += 1
                                reasoningStarted = false
                            }
                            if !finalParsed.response.isEmpty {
                                fullText += finalParsed.response
                                if !textMessageStarted {
                                    textMessageStarted = true
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .message(ResponseOutputMessage(id: msgId, status: "in_progress", content: []))))
                                    try await sse("response.content_part.added", ResponsesSSEContentPartAdded(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, part: ResponseContentItem(text: "")))
                                }
                                try await sse("response.output_text.delta", ResponsesSSETextDelta(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, delta: finalParsed.response))
                            }
                            if textMessageStarted {
                                try await sse("response.output_text.done", ResponsesSSETextDone(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, text: fullText))
                                try await sse("response.content_part.done", ResponsesSSEContentPartDone(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, part: ResponseContentItem(text: fullText)))
                                try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .message(ResponseOutputMessage(id: msgId, content: [ResponseContentItem(text: fullText)]))))
                                outputItems.append(.message(ResponseOutputMessage(id: msgId, content: [ResponseContentItem(text: fullText)])))
                            }

                            let ctxWin = inference.getContextWindow(for: modelId) ?? 0
                            let pToks = clientType.shouldScaleContext
                                ? cfg.scaleTokenCount(token.promptTokens ?? 0, modelContextWindow: ctxWin)
                                : (token.promptTokens ?? 0)
                            let cToks = clientType.shouldScaleContext
                                ? cfg.scaleTokenCount(tokenCount, modelContextWindow: ctxWin)
                                : tokenCount
                            let completedResp = OpenAIResponseObject(
                                id: responseId, model: modelId,
                                output: outputItems,
                                usage: ResponsesUsage(inputTokens: pToks, outputTokens: cToks)
                            ).withRequestEcho(from: req)
                            try await sse("response.completed", ResponsesSSECompleted(response: completedResp))
                            var storeOutputItems: [ResponseOutputItem] = []
                            for msg in capturedMessages {
                                let itemId = "msg_user_\(UUID().uuidString.prefix(12))"
                                storeOutputItems.append(.message(ResponseOutputMessage(id: itemId, role: msg.role.rawValue, content: [ResponseContentItem(text: msg.content ?? "")])))
                            }
                            storeOutputItems.append(contentsOf: outputItems)
                            let storeResp = OpenAIResponseObject(id: responseId, model: modelId, output: storeOutputItems, usage: ResponsesUsage(inputTokens: pToks, outputTokens: cToks))
                            ResponseStore.shared.put(storeResp)
                            if let convId = req.conversation?.id {
                                ConversationStore.shared.record(conversationId: convId, responseId: responseId)
                            }
                        }
                    case .keepAlive:
                        try await writer.write(ByteBuffer(string: ": keep-alive\n\n"))
                    case .done:
                        break
                    }
                }
                try await writer.finish(nil)
                let totalSec = Date().timeIntervalSince(streamStart)
                let avgTps = totalSec > 0 ? Double(tokenCount) / totalSec : 0
                NovaMLXLog.info("[SSE:\(reqTag)] Responses stream complete — reason=\(lastFinishReason ?? "unknown") completionTokens=\(tokenCount) avgTPS=\(String(format: "%.1f", avgTps)) total=\(String(format: "%.2f", totalSec))s writer.finish returned cleanly")
            } catch {
                let elapsed = Date().timeIntervalSince(streamStart)
                NovaMLXLog.error("[SSE:\(reqTag)] Responses stream error after \(String(format: "%.2f", elapsed))s, \(tokenCount) tokens: \(error) — \(type(of: error))")
                sequenceNumber += 1
                let errEvent = ResponsesSSEError(code: "ERR_STREAM", message: error.localizedDescription)
                if let errData = try? JSONEncoder().encode(errEvent) {
                    var obj = (try? JSONSerialization.jsonObject(with: errData) as? [String: Any]) ?? [:]
                    obj["sequence_number"] = sequenceNumber
                    if let finalData = try? JSONSerialization.data(withJSONObject: obj) {
                        try? await writer.write(ByteBuffer(string: "event: error\ndata: \(String(data: finalData, encoding: .utf8) ?? "{}")\n\n"))
                    }
                }
                try? await writer.finish(nil)
                NovaMLXLog.info("[SSE:\(reqTag)] error-path writer.finish(nil) returned")
            }
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
    }

    // MARK: - Compact Endpoint

    static func handleCompactRequest(
        req: CompactRequest,
        inference: InferenceService
    ) async throws -> Response {
        // Flatten input items to text for summarization
        let textContent = Self.flattenInputToText(req.input)

        guard !textContent.isEmpty else {
            return try Self.jsonResponse(
                ["error": ["message": "No input provided for compaction", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        // Use local model to summarize
        let summaryPrompt = """
        Summarize the following conversation concisely, preserving all key facts, decisions, and context. \
        The summary will replace the original conversation to save context space.

        \(textContent)
        """
        let request = InferenceRequest(
            model: req.model,
            messages: [ChatMessage(role: .system, content: "You are a conversation compaction assistant. Produce dense, information-rich summaries."),
                       ChatMessage(role: .user, content: summaryPrompt)],
            maxTokens: 2048,
            stream: false,
            httpRequestId: HTTPHelpers.requestID(from: nil)
        )

        let result: InferenceResult
        do {
            result = try await inference.generate(request)
        } catch {
            NovaMLXLog.error("[Compact] Model generation failed: \(error)")
            return try Self.jsonResponse(
                ["error": ["message": "Compaction failed: \(error.localizedDescription)", "type": "server_error"]],
                httpStatus: .internalServerError
            )
        }

        let summary = result.text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Encode as base64 to mimic encrypted_content
        guard let summaryData = summary.data(using: .utf8) else {
            return try Self.jsonResponse(
                ["error": ["message": "Failed to encode summary", "type": "server_error"]],
                httpStatus: .internalServerError
            )
        }
        let encryptedContent = summaryData.base64EncodedString()

        let compactId = "cmp_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
        let msgId = "msg_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"

        let output: [CompactedOutputItem] = [
            CompactedOutputItem(id: msgId, encryptedContent: nil),
            CompactedOutputItem(id: compactId, encryptedContent: encryptedContent)
        ]

        let usage = ResponsesUsage(
            inputTokens: result.promptTokens,
            outputTokens: result.completionTokens
        )

        let response = CompactedResponse(
            id: "resp_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))",
            model: req.model,
            output: output,
            usage: usage
        )

        return try Self.jsonResponse(response)
    }

    // MARK: - Input Tokens Endpoint

    static func handleInputTokensRequest(
        req: InputTokensRequest,
        inference: InferenceService
    ) async throws -> Response {
        let textContent = Self.flattenInputToText(req.input)
        // Approximate token count: ~4 chars per token for English text
        // For accuracy, tokenize via the model's tokenizer if loaded
        var tokenCount: Int

        if inference.isModelLoaded(req.model),
           let container = inference.engine.pool.get(req.model),
           let tokenizer = container.tokenizer {
            // Use the real tokenizer
            tokenCount = tokenizer.encode(textContent).count
        } else {
            // Fallback: rough estimate
            tokenCount = max(1, textContent.count / 4)
        }

        let response = InputTokensResponse(inputTokens: tokenCount)
        return try Self.jsonResponse(response)
    }

    // MARK: - Helpers

    private static func flattenInputToText(_ input: ResponseInput?) -> String {
        guard let input else { return "" }
        switch input {
        case .text(let str):
            return str
        case .items(let items):
            return items.compactMap { item -> String? in
                switch item {
                case .message(let msg):
                    return "[\(msg.role)] \(msg.content.textValue)"
                case .functionCall(let fc):
                    return "[assistant/tool_call] \(fc.name)(\(fc.arguments))"
                case .functionCallOutput(let fco):
                    return "[tool] \(fco.output)"
                case .reasoning(let r):
                    return (r.summary ?? []).map { $0.text }.joined(separator: " ")
                case .skipped:
                    return nil
                }
            }.joined(separator: "\n")
        }
    }
}
