import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXUtils

// MARK: - Inference Handlers (Chat, Completion, Anthropic)
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    static func handleChat(
        openAIReq: OpenAIRequest, messages: [ChatMessage], inference: InferenceService,
        sessionId: String? = nil, responseFormat: ResponseFormat? = nil, jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil, gbnfGrammar: String? = nil,
        cfg: ServerConfig, clientType: ClientType,
        responseModelOverride: String? = nil,
        httpRequest: Request? = nil
    ) async throws -> Response {
        let ocrSampling = OCROptimizer.samplingOverrides(
            modelName: openAIReq.model,
            userTemperature: openAIReq.temperature,
            userMaxTokens: openAIReq.maxTokens,
            userRepetitionPenalty: openAIReq.repetitionPenalty.map { Float($0) }
        )
        let ocrStop = OCROptimizer.applyStopSequences(openAIReq.stop, modelName: openAIReq.model)

        let request = InferenceRequest(
            model: openAIReq.model, messages: messages,
            tools: openAIReq.tools?.map { tool in
                tool.mapValues { unwrapAnyCodable($0) }
            },
            temperature: ocrSampling.temperature,
            maxTokens: ocrSampling.maxTokens,
            topP: openAIReq.topP, topK: openAIReq.topK,
            minP: openAIReq.minP.map { Float($0) },
            frequencyPenalty: openAIReq.frequencyPenalty.map { Float($0) },
            presencePenalty: openAIReq.presencePenalty.map { Float($0) },
            repetitionPenalty: ocrSampling.repetitionPenalty,
            seed: openAIReq.seed,
            stream: false, stop: ocrStop,
            sessionId: sessionId, responseFormat: responseFormat,
            jsonSchemaDef: jsonSchemaDef,
            regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
            thinkingBudget: openAIReq.resolvedThinkingBudget,
            enableThinking: openAIReq.resolvedEnableThinking,
            preserveThinking: openAIReq.resolvedPreserveThinking,
            draftModel: openAIReq.draftModel,
            numDraftTokens: openAIReq.numDraftTokens,
            includeLogprobs: openAIReq.logprobs == true,
            topLogprobsCount: openAIReq.topLogprobs,
            httpRequestId: HTTPHelpers.requestID(from: httpRequest)
        )

        CurrentInferenceModel.shared.modelID = request.model
        defer { CurrentInferenceModel.shared.modelID = nil }
        let result = try await inference.generate(request)
        let finishReason: String
        let message: OpenAIChatMessage

        // Scrub control tokens from raw output
        var scrubbedText = result.text
        let shouldParseThinking = openAIReq.resolvedEnableThinking != false
        if scrubbedText.contains("<|") || (!shouldParseThinking && scrubbedText.contains("<think")) {
            if let regex = try? NSRegularExpression(pattern: shouldParseThinking ? "<\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)" : "<(?:\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)|/?think[^>]*)>") {
                let nsRange = NSRange(scrubbedText.startIndex..., in: scrubbedText)
                let matches = regex.matches(in: scrubbedText, range: nsRange)
                for match in matches.reversed() {
                    if let range = Range(match.range, in: scrubbedText) {
                        if shouldParseThinking {
                            let matched = String(scrubbedText[range])
                            if matched.contains("think") || matched.contains("thinking") { continue }
                        }
                        scrubbedText.removeSubrange(range)
                    }
                }
            }
        }
        let thinkingText: String?
        let responseText: String
        if shouldParseThinking {
            let isImplicitModel = ModelContainer.isImplicitThinkingModel(for: openAIReq.model)
            let thinkingParser = ThinkingParser(expectImplicitThinking: isImplicitModel)
            _ = thinkingParser.feed(scrubbedText)
            let finalResult = thinkingParser.finalize()
            // Truncate hallucinated role markers
            var cleanThinking = finalResult.thinking
            var cleanResponse = finalResult.response
            let hallucPatterns = ["\nuser\n", "\nmodel\n", "\nassistant\n", "user\n", "model\n"]
            for p in hallucPatterns {
                if let range = cleanThinking.range(of: p) { cleanThinking = String(cleanThinking[..<range.lowerBound]) }
                if let range = cleanResponse.range(of: p) { cleanResponse = String(cleanResponse[..<range.lowerBound]) }
            }
            thinkingText = cleanThinking.isEmpty ? nil : cleanThinking
            responseText = cleanResponse
        } else {
            responseText = scrubbedText
            thinkingText = nil
        }

        // Engine-produced tool calls take priority
        if let engineToolCalls = result.toolCalls, !engineToolCalls.isEmpty {
            finishReason = "tool_calls"
            message = OpenAIChatMessage(
                role: "assistant",
                content: responseText.isEmpty ? nil : responseText,
                reasoningContent: thinkingText,
                toolCalls: engineToolCalls.map { tc in
                    OpenAIToolCall(id: tc.id, function: OpenAIFunctionCall(name: tc.functionName, arguments: tc.arguments))
                }
            )
        } else {
            // Fallback: post-hoc text parsing
            let toolParsed = ToolCallParser.parse(responseText)
            if let toolCalls = toolParsed.toolCalls {
                finishReason = "tool_calls"
                message = OpenAIChatMessage(
                    role: "assistant",
                    content: toolParsed.content.isEmpty ? nil : toolParsed.content,
                    reasoningContent: thinkingText,
                    toolCalls: toolCalls.map { tc in
                        OpenAIToolCall(id: tc.id, function: OpenAIFunctionCall(name: tc.function.name, arguments: tc.function.arguments))
                    }
                )
            } else {
                finishReason = result.finishReason.rawValue
                message = OpenAIChatMessage(
                    role: "assistant",
                    content: responseText.isEmpty ? nil : responseText,
                    reasoningContent: thinkingText
                )
            }
        }
        let response = OpenAIResponse(
            id: "chatcmpl-\(result.id.uuidString.prefix(8))",
            model: responseModelOverride ?? result.model,
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: message,
                    finishReason: finishReason,
                    logprobs: result.tokenLogprobs.map { Self.buildLogprobs(from: $0) } ?? nil
                )
            ],
            usage: {
                let ctxWin = inference.getContextWindow(for: openAIReq.model) ?? 0
                let p = clientType.shouldScaleContext ? cfg.scaleTokenCount(result.promptTokens, modelContextWindow: ctxWin) : result.promptTokens
                let c = clientType.shouldScaleContext ? cfg.scaleTokenCount(result.completionTokens, modelContextWindow: ctxWin) : result.completionTokens
                return OpenAIUsage(promptTokens: p, completionTokens: c)
            }()
        )
        let ctxWin = inference.getContextWindow(for: openAIReq.model) ?? 0
        let scaledPrompt = clientType.shouldScaleContext
            ? cfg.scaleTokenCount(result.promptTokens, modelContextWindow: ctxWin) : result.promptTokens
        let scaledCompletion = clientType.shouldScaleContext
            ? cfg.scaleTokenCount(result.completionTokens, modelContextWindow: ctxWin) : result.completionTokens
        Self.recordTokenUsage(
            request: httpRequest,
            promptTokens: scaledPrompt,
            completionTokens: scaledCompletion,
            model: openAIReq.model,
            endpoint: "/v1/chat/completions"
        )
        return try jsonResponse(response)
    }

    static func handleStreamChat(
        openAIReq: OpenAIRequest, messages: [ChatMessage], inference: InferenceService,
        sessionId: String? = nil, responseFormat: ResponseFormat? = nil, jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil, gbnfGrammar: String? = nil,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator,
        responseModelOverride: String? = nil,
        httpRequest: Request? = nil
    ) async throws -> Response {
        let responseModel = responseModelOverride ?? openAIReq.model
        let ocrSampling = OCROptimizer.samplingOverrides(
            modelName: openAIReq.model,
            userTemperature: openAIReq.temperature,
            userMaxTokens: openAIReq.maxTokens,
            userRepetitionPenalty: openAIReq.repetitionPenalty.map { Float($0) }
        )
        let ocrStop = OCROptimizer.applyStopSequences(openAIReq.stop, modelName: openAIReq.model)

        let request = InferenceRequest(
            model: openAIReq.model, messages: messages,
            tools: openAIReq.tools?.map { tool in
                tool.mapValues { unwrapAnyCodable($0) }
            },
            temperature: ocrSampling.temperature,
            maxTokens: ocrSampling.maxTokens,
            topP: openAIReq.topP, topK: openAIReq.topK,
            minP: openAIReq.minP.map { Float($0) },
            frequencyPenalty: openAIReq.frequencyPenalty.map { Float($0) },
            presencePenalty: openAIReq.presencePenalty.map { Float($0) },
            repetitionPenalty: ocrSampling.repetitionPenalty,
            seed: openAIReq.seed,
            stream: true, stop: ocrStop,
            sessionId: sessionId, responseFormat: responseFormat,
            jsonSchemaDef: jsonSchemaDef,
            regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
            thinkingBudget: openAIReq.resolvedThinkingBudget,
            enableThinking: openAIReq.resolvedEnableThinking,
            preserveThinking: openAIReq.resolvedPreserveThinking,
            draftModel: openAIReq.draftModel,
            numDraftTokens: openAIReq.numDraftTokens,
            includeLogprobs: openAIReq.logprobs == true,
            topLogprobsCount: openAIReq.topLogprobs,
            httpRequestId: HTTPHelpers.requestID(from: httpRequest)
        )

        let modelId = openAIReq.model
        let autoLoadCfg = cfg.autoLoad
        let keepAliveStream = Self.withSSEKeepAlive(
            Self.loadAwareStream(
                modelId: modelId, inference: inference,
                coordinator: coordinator, autoLoadCfg: autoLoadCfg,
                inferenceStreamProducer: { inference.stream(request) }
            )
        )
        let chunkId = "chatcmpl-\(request.id.uuidString.prefix(8))"
        let includeUsage = openAIReq.streamOptions?.includeUsage == true
        let toolCallCounter = LockedCounter()
        let reqTag = String(request.id.uuidString.prefix(8))
        let toolCount = openAIReq.tools?.count ?? 0
        let maxTok = ocrSampling.maxTokens ?? openAIReq.maxTokens ?? -1
        NovaMLXLog.info("[SSE:\(reqTag)] OpenAI stream start — model=\(openAIReq.model) client=\(clientType) maxTokens=\(maxTok) tools=\(toolCount) promptMsgs=\(messages.count)")

        let body: ResponseBody = .init { writer in
            CurrentInferenceModel.shared.modelID = openAIReq.model
            defer { CurrentInferenceModel.shared.modelID = nil }
            let streamStart = ContinuousClock.now
            var firstTokenAt: ContinuousClock.Instant? = nil
            var lastProgressAt = ContinuousClock.now
            var lastProgressCount = 0
            var completionTokenCount = 0
            var promptTokenCount: Int? = nil
            var lastFinishReason: String? = nil
            do {
                let roleChunk = OpenAIStreamChunk(
                    id: chunkId,
                    model: responseModel,
                    choices: [
                        OpenAIStreamChoice(index: 0, delta: OpenAIDelta(role: "assistant", content: ""))
                    ]
                )
                let roleData = try JSONEncoder().encode(roleChunk)
                try await writer.write(ByteBuffer(string: "data: \(String(data: roleData, encoding: .utf8) ?? "")\n\n"))

                let shouldParseThinking = openAIReq.resolvedEnableThinking != false
                let isImplicitModel = ModelContainer.isImplicitThinkingModel(for: openAIReq.model)
                let thinkingParser = shouldParseThinking ? ThinkingParser(expectImplicitThinking: isImplicitModel) : nil
                var streamedResponse = ""
                // Local helper — invoked on every emitted token to track TTFT
                // and periodic progress. Throttled to every 50 tokens or 5 s.
                func logTokenProgress() {
                    if firstTokenAt == nil {
                        firstTokenAt = ContinuousClock.now
                        let ttft = firstTokenAt! - streamStart
                        let ttftMs = Double(ttft.components.seconds) * 1000 + Double(ttft.components.attoseconds) / 1e15
                        NovaMLXLog.info("[SSE:\(reqTag)] TTFT=\(String(format: "%.0f", ttftMs))ms — first content token")
                    }
                    let now = ContinuousClock.now
                    let sinceLast = now - lastProgressAt
                    if completionTokenCount - lastProgressCount >= 50 || sinceLast > .seconds(5) {
                        let elapsedSec = Double(sinceLast.components.seconds) + Double(sinceLast.components.attoseconds) / 1e18
                        let delta = completionTokenCount - lastProgressCount
                        let tps = elapsedSec > 0 ? Double(delta) / elapsedSec : 0
                        NovaMLXLog.info("[SSE:\(reqTag)] Progress: \(completionTokenCount) tokens, \(String(format: "%.1f", tps)) tok/s (last \(delta) in \(String(format: "%.1f", elapsedSec))s)")
                        lastProgressCount = completionTokenCount
                        lastProgressAt = now
                    }
                }
                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        var tokenLogprobs: OpenAILogprobs? = Self.tokenToLogprobEntry(token).map {
                            OpenAILogprobs(content: [$0])
                        }
                        if let tc = token.toolCall {
                            let idx = toolCallCounter.increment()
                            let tcDelta = OpenAIToolCallDelta(
                                index: idx,
                                id: tc.id,
                                type: "function",
                                function: OpenAIFunctionCallDelta(name: tc.functionName, arguments: tc.arguments)
                            )
                            let chunk = OpenAIStreamChunk(
                                id: chunkId,
                                model: responseModel,
                                choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(toolCalls: [tcDelta]))]
                            )
                            let data = try JSONEncoder().encode(chunk)
                            try await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                        } else if let finish = token.finishReason {
                            lastFinishReason = finish.rawValue
                            if let pt = token.promptTokens { promptTokenCount = pt }
                            if let tp = thinkingParser {
                                let finalParsed = tp.finalize()
                                var cleanResp = finalParsed.response
                                if !streamedResponse.isEmpty, cleanResp.hasPrefix(streamedResponse) {
                                    cleanResp = String(cleanResp.dropFirst(streamedResponse.count))
                                }
                                let hallucPatterns = ["\nuser\n", "\nmodel\n", "\nassistant\n", "user\n", "model\n"]
                                for p in hallucPatterns {
                                    if let range = cleanResp.range(of: p) { cleanResp = String(cleanResp[..<range.lowerBound]) }
                                }
                                if !cleanResp.isEmpty {
                                    completionTokenCount += 1
                                    logTokenProgress()
                                    let respDelta = OpenAIDelta(content: cleanResp)
                                    let respChunk = OpenAIStreamChunk(id: chunkId, model: responseModel, choices: [OpenAIStreamChoice(index: 0, delta: respDelta)], novaChannels: token.channels?.map { NovaHarmonyChannel(channel: $0.channel, text: $0.text) })
                                    let respData = try JSONEncoder().encode(respChunk)
                                    try await writer.write(ByteBuffer(string: "data: \(String(data: respData, encoding: .utf8) ?? "")\n\n"))
                                }
                            }
                            let finalChunk = OpenAIStreamChunk(
                                id: chunkId,
                                model: responseModel,
                                choices: [
                                    OpenAIStreamChoice(
                                        index: 0,
                                        delta: OpenAIDelta(),
                                        finishReason: finish.rawValue
                                    )
                                ],
                                novaChannels: token.channels?.map { NovaHarmonyChannel(channel: $0.channel, text: $0.text) }
                            )
                            let finalData = try JSONEncoder().encode(finalChunk)
                            try await writer.write(ByteBuffer(string: "data: \(String(data: finalData, encoding: .utf8) ?? "")\n\n"))
                        } else {
                            completionTokenCount += 1
                            logTokenProgress()
                            var cleanText = token.text
                            if cleanText.contains("<|") || (!shouldParseThinking && cleanText.contains("<think")) {
                                let pattern = shouldParseThinking
                                    ? "<\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)"
                                    : "<(?:\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)|/?think[^>]*)>"
                                if let regex = try? NSRegularExpression(pattern: pattern) {
                                    let nsRange = NSRange(cleanText.startIndex..., in: cleanText)
                                    let matches = regex.matches(in: cleanText, range: nsRange)
                                    for match in matches.reversed() {
                                        if let range = Range(match.range, in: cleanText) {
                                            if shouldParseThinking {
                                                let matched = String(cleanText[range])
                                                if matched.contains("think") || matched.contains("thinking") { continue }
                                            }
                                            cleanText.removeSubrange(range)
                                        }
                                    }
                                }
                            }
                            if let tp = thinkingParser {
                                var remaining = cleanText
                                while true {
                                    let parsed = tp.feed(remaining)
                                    remaining = ""
                                    if parsed.text.isEmpty { break }
                                    if parsed.type == .content {
                                        streamedResponse += parsed.text
                                    }
                                    let delta: OpenAIDelta = parsed.type == .thinking
                                        ? OpenAIDelta(reasoningContent: parsed.text)
                                        : OpenAIDelta(content: parsed.text)
                                    let chunk = OpenAIStreamChunk(
                                        id: chunkId,
                                        model: responseModel,
                                        choices: [OpenAIStreamChoice(index: 0, delta: delta, logprobs: tokenLogprobs)],
                                        novaChannels: token.channels?.map { NovaHarmonyChannel(channel: $0.channel, text: $0.text) }
                                    )
                                    tokenLogprobs = nil
                                    let data = try JSONEncoder().encode(chunk)
                                    try await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                                }
                            } else {
                                if cleanText.isEmpty { continue }
                                let delta = OpenAIDelta(content: cleanText)
                                let chunk = OpenAIStreamChunk(
                                    id: chunkId,
                                    model: responseModel,
                                    choices: [OpenAIStreamChoice(index: 0, delta: delta, logprobs: tokenLogprobs)],
                                    novaChannels: token.channels?.map { NovaHarmonyChannel(channel: $0.channel, text: $0.text) }
                                )
                                tokenLogprobs = nil
                                let data = try JSONEncoder().encode(chunk)
                                try await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                            }
                        }
                    case .keepAlive:
                        try await writer.write(ByteBuffer(string: ": keep-alive\n\n"))
                    case .done:
                        break
                    }
                }
                if includeUsage {
                    let resolvedPromptCount = promptTokenCount ?? inference.countTokens(model: openAIReq.model, messages: messages) ?? 0
                    let ctxWin = inference.getContextWindow(for: openAIReq.model) ?? 0
                    let scaledPrompt = clientType.shouldScaleContext
                        ? cfg.scaleTokenCount(resolvedPromptCount, modelContextWindow: ctxWin) : resolvedPromptCount
                    let scaledCompletion = clientType.shouldScaleContext
                        ? cfg.scaleTokenCount(completionTokenCount, modelContextWindow: ctxWin) : completionTokenCount
                    let usageChunk = OpenAIStreamChunk(
                        id: chunkId,
                        model: responseModel,
                        choices: [],
                        usage: OpenAIUsage(promptTokens: scaledPrompt, completionTokens: scaledCompletion)
                    )
                    let usageData = try JSONEncoder().encode(usageChunk)
                    try await writer.write(ByteBuffer(string: "data: \(String(data: usageData, encoding: .utf8) ?? "")\n\n"))
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                Self.applyKeepAlive(openAIReq.keepAlive, modelId: openAIReq.model, pool: inference.engine.pool)
                let totalElapsed = ContinuousClock.now - streamStart
                let totalSec = Double(totalElapsed.components.seconds) + Double(totalElapsed.components.attoseconds) / 1e18
                let avgTps = totalSec > 0 ? Double(completionTokenCount) / totalSec : 0
                let promptN = promptTokenCount ?? inference.countTokens(model: openAIReq.model, messages: messages) ?? 0
                let ctxWin = inference.getContextWindow(for: openAIReq.model) ?? 0
                let scaledPrompt = clientType.shouldScaleContext
                    ? cfg.scaleTokenCount(promptN, modelContextWindow: ctxWin) : promptN
                let scaledCompletion = clientType.shouldScaleContext
                    ? cfg.scaleTokenCount(completionTokenCount, modelContextWindow: ctxWin) : completionTokenCount
                Self.recordTokenUsage(
                    request: httpRequest,
                    promptTokens: scaledPrompt,
                    completionTokens: scaledCompletion,
                    model: openAIReq.model,
                    endpoint: "/v1/chat/completions"
                )
                NovaMLXLog.info("[SSE:\(reqTag)] Stream complete — reason=\(lastFinishReason ?? "unknown") completionTokens=\(completionTokenCount) promptTokens=\(promptN) avgTPS=\(String(format: "%.1f", avgTps)) total=\(String(format: "%.2f", totalSec))s [DONE]+finish sent")
                try await writer.finish(nil)
                NovaMLXLog.info("[SSE:\(reqTag)] writer.finish(nil) returned — response body closed cleanly")
            } catch {
                let totalElapsed = ContinuousClock.now - streamStart
                let totalSec = Double(totalElapsed.components.seconds) + Double(totalElapsed.components.attoseconds) / 1e18
                NovaMLXLog.error("[SSE:\(reqTag)] Stream error after \(completionTokenCount) tokens in \(String(format: "%.2f", totalSec))s: \(error) — \(type(of: error))")
                let (message, type, code) = Self.streamErrorFields(error)
                let errorDetail = OpenAIErrorDetail(message: message, type: type, code: code)
                let errorResp = OpenAIErrorResponse(error: errorDetail)
                if let data = try? JSONEncoder().encode(errorResp) {
                    let sseError = "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"
                    try? await writer.write(ByteBuffer(string: sseError))
                }
                try? await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                Self.applyKeepAlive(openAIReq.keepAlive, modelId: openAIReq.model, pool: inference.engine.pool)
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

    static func handleStreamAnthropic(
        anthropicReq: AnthropicRequest, messages: [ChatMessage], inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator,
        httpRequest: Request? = nil
    ) async throws -> Response {
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
            stream: true, stop: ocrStop,
            thinkingBudget: anthropicReq.thinkingBudget,
            enableThinking: anthropicReq.resolvedEnableThinking,
            preserveThinking: anthropicReq.resolvedPreserveThinking,
            httpRequestId: HTTPHelpers.requestID(from: httpRequest)
        )

        let reqTag = request.id.uuidString.prefix(8)
        let toolCount = anthropicReq.tools?.count ?? 0
        NovaMLXLog.info("[SSE:\(reqTag)] Anthropic stream start — model=\(anthropicReq.model) client=\(clientType) maxTokens=\(anthropicReq.maxTokens) tools=\(toolCount) promptMsgs=\(messages.count)")

        let modelId = anthropicReq.model
        let autoLoadCfg = cfg.autoLoad
        let keepAliveStream = Self.withSSEKeepAlive(
            Self.loadAwareStream(
                modelId: modelId, inference: inference,
                coordinator: coordinator, autoLoadCfg: autoLoadCfg,
                inferenceStreamProducer: { inference.stream(request) }
            ),
            reqTag: String(reqTag)
        )
        let msgId = "msg_\(request.id.uuidString.prefix(24))"

        let body: ResponseBody = .init { writer in
            CurrentInferenceModel.shared.modelID = anthropicReq.model
            defer { CurrentInferenceModel.shared.modelID = nil }
            var tokenCount = 0
            let streamStart = Date()
            var lastProgressAt = streamStart
            var lastProgressCount = 0
            do {
                try await writer.write(ByteBuffer(string: "event: ping\ndata: {\"type\":\"ping\"}\n\n"))
                NovaMLXLog.debug("[SSE:\(reqTag)] Sent initial headers + ping")

                let startEvent = AnthropicStreamEvent.messageStart(id: msgId, model: anthropicReq.model)
                let startData = try JSONEncoder().encode(startEvent)
                try await writer.write(ByteBuffer(string: "event: message_start\ndata: \(String(data: startData, encoding: .utf8) ?? "{}")\n\n"))

                NovaMLXLog.debug("[SSE:\(reqTag)] Waiting for first token from inference stream...")

                let shouldParseThinking = anthropicReq.resolvedEnableThinking != false
                let isAnthropicImplicitModel = ModelContainer.isImplicitThinkingModel(for: anthropicReq.model)
                let thinkingParser = shouldParseThinking ? ThinkingParser(expectImplicitThinking: isAnthropicImplicitModel) : nil
                var currentBlockIndex = 0
                var isInThinkingBlock = false
                var hasStartedTextBlock = false
                var streamedResponse = ""

                func startThinkingBlock() async throws {
                    let evt = AnthropicStreamEvent.contentBlockStart(index: currentBlockIndex, blockType: "thinking")
                    let data = try JSONEncoder().encode(evt)
                    try await writer.write(ByteBuffer(string: "event: content_block_start\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                    isInThinkingBlock = true
                }

                func endCurrentBlock() async throws {
                    try await writer.write(ByteBuffer(string: "event: content_block_stop\ndata: {}\n\n"))
                    currentBlockIndex += 1
                    isInThinkingBlock = false
                }

                func startTextBlock() async throws {
                    let evt = AnthropicStreamEvent.contentBlockStart(index: currentBlockIndex, blockType: "text")
                    let data = try JSONEncoder().encode(evt)
                    try await writer.write(ByteBuffer(string: "event: content_block_start\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                    isInThinkingBlock = false
                    hasStartedTextBlock = true
                }

                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        if token.finishReason != nil {
                            if let tp = thinkingParser {
                                let finalParsed = tp.finalize()
                                var cleanFinalResp = finalParsed.response
                                if !streamedResponse.isEmpty, cleanFinalResp.hasPrefix(streamedResponse) {
                                    cleanFinalResp = String(cleanFinalResp.dropFirst(streamedResponse.count))
                                }
                                let hallucPatterns = ["\nuser\n", "\nmodel\n", "\nassistant\n", "user\n", "model\n"]
                                for p in hallucPatterns {
                                    if let range = cleanFinalResp.range(of: p) { cleanFinalResp = String(cleanFinalResp[..<range.lowerBound]) }
                                }
                                if !cleanFinalResp.isEmpty {
                                if isInThinkingBlock {
                                    try await endCurrentBlock()
                                }
                                if !hasStartedTextBlock {
                                    try await startTextBlock()
                                }
                                tokenCount += 1
                                let deltaEvent = AnthropicStreamEvent.textDelta(cleanFinalResp)
                                let deltaData = try JSONEncoder().encode(deltaEvent)
                                try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: deltaData, encoding: .utf8) ?? "{}")\n\n"))
                                }
                            }
                            if isInThinkingBlock || hasStartedTextBlock {
                                try await writer.write(ByteBuffer(string: "event: content_block_stop\ndata: {}\n\n"))
                            }
                            let ctxWin = inference.getContextWindow(for: anthropicReq.model) ?? 0
                            let promptCount = token.promptTokens ?? inference.countTokens(model: anthropicReq.model, messages: messages) ?? 0
                            let scaledIn = clientType.shouldScaleContext ? cfg.scaleTokenCount(promptCount, modelContextWindow: ctxWin) : promptCount
                            let scaledOut = clientType.shouldScaleContext ? cfg.scaleTokenCount(tokenCount, modelContextWindow: ctxWin) : tokenCount
                            let stopReason = token.finishReason?.rawValue ?? "end_turn"
                            let deltaEv = AnthropicStreamEvent.messageDelta(stopReason: stopReason, usage: AnthropicUsage(inputTokens: scaledIn, outputTokens: scaledOut))
                            let deltaData = try JSONEncoder().encode(deltaEv)
                            try await writer.write(ByteBuffer(string: "event: message_delta\ndata: \(String(data: deltaData, encoding: .utf8) ?? "{}")\n\n"))
                            try await writer.write(ByteBuffer(string: "event: message_stop\ndata: {}\n\n"))
                            Self.recordTokenUsage(
                                request: httpRequest,
                                promptTokens: scaledIn,
                                completionTokens: scaledOut,
                                model: anthropicReq.model,
                                endpoint: "/v1/messages"
                            )
                            let elapsed = Date().timeIntervalSince(streamStart)
                            NovaMLXLog.info("[SSE:\(reqTag)] Stream complete — \(tokenCount) tokens in \(String(format: "%.1f", elapsed))s, usage: input=\(scaledIn) output=\(scaledOut)")
                        } else if !token.text.isEmpty {
                            if tokenCount == 0 {
                                let ttft = Date().timeIntervalSince(streamStart)
                                NovaMLXLog.info("[SSE:\(reqTag)] First token received (TTFT=\(String(format: "%.1f", ttft))s)")
                            }
                            tokenCount += 1
                            // Periodic progress sampling — every 50 tokens or 5 s.
                            let now = Date()
                            let sinceLast = now.timeIntervalSince(lastProgressAt)
                            if tokenCount - lastProgressCount >= 50 || sinceLast >= 5 {
                                let delta = tokenCount - lastProgressCount
                                let tps = sinceLast > 0 ? Double(delta) / sinceLast : 0
                                NovaMLXLog.info("[SSE:\(reqTag)] Progress: \(tokenCount) tokens, \(String(format: "%.1f", tps)) tok/s (last \(delta) in \(String(format: "%.1f", sinceLast))s)")
                                lastProgressCount = tokenCount
                                lastProgressAt = now
                            }

                            var cleanText = token.text
                            if cleanText.contains("<|") {
                                if let regex = try? NSRegularExpression(pattern: "<\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)") {
                                    let nsRange = NSRange(cleanText.startIndex..., in: cleanText)
                                    let matches = regex.matches(in: cleanText, range: nsRange)
                                    for match in matches.reversed() {
                                        if let range = Range(match.range, in: cleanText) {
                                            let matched = String(cleanText[range])
                                            if !matched.contains("think") && !matched.contains("thinking") {
                                                cleanText.removeSubrange(range)
                                            }
                                        }
                                    }
                                }
                            }

                            var remaining = cleanText
                            if let tp = thinkingParser {
                                while true {
                                    let parsed = tp.feed(remaining)
                                    remaining = ""
                                    if parsed.text.isEmpty { break }

                                    if parsed.type == .thinking {
                                        if !isInThinkingBlock {
                                            if hasStartedTextBlock {
                                                try await endCurrentBlock()
                                            }
                                            try await startThinkingBlock()
                                        }
                                        let deltaEvent = AnthropicStreamEvent.thinkingDelta(parsed.text)
                                        let deltaData = try JSONEncoder().encode(deltaEvent)
                                        try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: deltaData, encoding: .utf8) ?? "{}")\n\n"))
                                    } else {
                                        streamedResponse += parsed.text
                                        if isInThinkingBlock {
                                            try await endCurrentBlock()
                                        }
                                        if !hasStartedTextBlock {
                                            try await startTextBlock()
                                        }
                                        let deltaEvent = AnthropicStreamEvent.textDelta(parsed.text)
                                        let deltaData = try JSONEncoder().encode(deltaEvent)
                                        try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: deltaData, encoding: .utf8) ?? "{}")\n\n"))
                                    }
                                }
                            } else {
                                if !cleanText.isEmpty {
                                    if !hasStartedTextBlock {
                                        try await startTextBlock()
                                    }
                                    let deltaEvent = AnthropicStreamEvent.textDelta(cleanText)
                                    let deltaData = try JSONEncoder().encode(deltaEvent)
                                    try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: deltaData, encoding: .utf8) ?? "{}")\n\n"))
                                }
                            }
                        }
                    case .keepAlive:
                        let elapsed = Date().timeIntervalSince(streamStart)
                        NovaMLXLog.debug("[SSE:\(reqTag)] Keep-alive ping sent (\(String(format: "%.0f", elapsed))s elapsed, \(tokenCount) tokens so far)")
                    case .done:
                        NovaMLXLog.debug("[SSE:\(reqTag)] Stream done signal")
                    }
                }
                Self.applyKeepAlive(anthropicReq.keepAlive, modelId: anthropicReq.model, pool: inference.engine.pool)
            } catch {
                let elapsed = Date().timeIntervalSince(streamStart)
                NovaMLXLog.error("[SSE:\(reqTag)] Stream ERROR after \(String(format: "%.1f", elapsed))s, \(tokenCount) tokens sent: \(error) — \(type(of: error))")
                let (message, errorType) = Self.anthropicStreamErrorFields(error)
                let errorDetail = AnthropicErrorDetail(type: errorType, message: message)
                let errorResp = AnthropicErrorResponse(error: errorDetail)
                if let data = try? JSONEncoder().encode(errorResp) {
                    try? await writer.write(ByteBuffer(string: "event: error\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                }
                Self.applyKeepAlive(anthropicReq.keepAlive, modelId: anthropicReq.model, pool: inference.engine.pool)
                try? await writer.finish(nil)
                NovaMLXLog.info("[SSE:\(reqTag)] error-path writer.finish(nil) returned")
                return
            }
            try? await writer.finish(nil)
            NovaMLXLog.info("[SSE:\(reqTag)] writer.finish(nil) returned — response body closed cleanly")
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
    }

    static func handleCompletion(
        compReq: OpenAICompletionRequest, inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType
    ) async throws -> Response {
        let messages = [ChatMessage(role: .user, content: compReq.prompt)]
        let request = InferenceRequest(
            model: compReq.model, messages: messages,
            temperature: compReq.temperature, maxTokens: compReq.maxTokens,
            topP: compReq.topP, topK: compReq.topK,
            minP: compReq.minP.map { Float($0) },
            frequencyPenalty: compReq.frequencyPenalty.map { Float($0) },
            presencePenalty: compReq.presencePenalty.map { Float($0) },
            repetitionPenalty: compReq.repetitionPenalty.map { Float($0) },
            seed: compReq.seed,
            stream: false, stop: compReq.stop,
            httpRequestId: HTTPHelpers.requestID(from: nil)
        )

        CurrentInferenceModel.shared.modelID = request.model
        defer { CurrentInferenceModel.shared.modelID = nil }
        let result = try await inference.generate(request)
        let ctxWin = inference.getContextWindow(for: compReq.model) ?? 0
        let scaledP = clientType.shouldScaleContext ? cfg.scaleTokenCount(result.promptTokens, modelContextWindow: ctxWin) : result.promptTokens
        let scaledC = clientType.shouldScaleContext ? cfg.scaleTokenCount(result.completionTokens, modelContextWindow: ctxWin) : result.completionTokens
        let response = OpenAICompletionResponse(
            id: "cmpl-\(result.id.uuidString.prefix(8))",
            model: result.model,
            choices: [
                OpenAICompletionChoice(
                    index: 0,
                    text: result.text,
                    finishReason: result.finishReason.rawValue
                )
            ],
            usage: OpenAIUsage(promptTokens: scaledP, completionTokens: scaledC)
        )
        return try jsonResponse(response)
    }

    static func handleStreamCompletion(
        compReq: OpenAICompletionRequest, inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let messages = [ChatMessage(role: .user, content: compReq.prompt)]
        let request = InferenceRequest(
            model: compReq.model, messages: messages,
            temperature: compReq.temperature, maxTokens: compReq.maxTokens,
            topP: compReq.topP, topK: compReq.topK,
            minP: compReq.minP.map { Float($0) },
            frequencyPenalty: compReq.frequencyPenalty.map { Float($0) },
            presencePenalty: compReq.presencePenalty.map { Float($0) },
            repetitionPenalty: compReq.repetitionPenalty.map { Float($0) },
            seed: compReq.seed,
            stream: true, stop: compReq.stop,
            httpRequestId: HTTPHelpers.requestID(from: nil)
        )

        let modelId = compReq.model
        let autoLoadCfg = cfg.autoLoad
        let keepAliveStream = Self.withSSEKeepAlive(
            Self.loadAwareStream(
                modelId: modelId, inference: inference,
                coordinator: coordinator, autoLoadCfg: autoLoadCfg,
                inferenceStreamProducer: { inference.stream(request) }
            )
        )
        let chunkId = "cmpl-\(request.id.uuidString.prefix(8))"

        let body: ResponseBody = .init { writer in
            CurrentInferenceModel.shared.modelID = compReq.model
            defer { CurrentInferenceModel.shared.modelID = nil }
            var completionTokenCount = 0
            var completionPromptTokenCount: Int? = nil
            do {
                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        if let finish = token.finishReason {
                            if let pt = token.promptTokens { completionPromptTokenCount = pt }
                            let finalChunk = OpenAICompletionStreamChunk(
                                id: chunkId,
                                model: compReq.model,
                                choices: [
                                    OpenAICompletionStreamChoice(
                                        index: 0,
                                        text: "",
                                        finishReason: finish.rawValue
                                    )
                                ]
                            )
                            let finalData = try JSONEncoder().encode(finalChunk)
                            try await writer.write(ByteBuffer(string: "data: \(String(data: finalData, encoding: .utf8) ?? "")\n\n"))
                        } else {
                            completionTokenCount += 1
                            let chunk = OpenAICompletionStreamChunk(
                                id: chunkId,
                                model: compReq.model,
                                choices: [
                                    OpenAICompletionStreamChoice(index: 0, text: token.text)
                                ]
                            )
                            let data = try JSONEncoder().encode(chunk)
                            try await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                        }
                    case .keepAlive:
                        try await writer.write(ByteBuffer(string: ": keep-alive\n\n"))
                    case .done:
                        break
                    }
                }
                if clientType.shouldScaleContext, completionTokenCount > 0 {
                    let promptCount = completionPromptTokenCount ?? inference.countTokens(model: compReq.model, messages: messages) ?? 0
                    let ctxWin = inference.getContextWindow(for: compReq.model) ?? 0
                    let sp = clientType.shouldScaleContext ? cfg.scaleTokenCount(promptCount, modelContextWindow: ctxWin) : promptCount
                    let sc = clientType.shouldScaleContext ? cfg.scaleTokenCount(completionTokenCount, modelContextWindow: ctxWin) : completionTokenCount
                    let usageChunk = OpenAICompletionStreamChunk(
                        id: chunkId, model: compReq.model, choices: [],
                        usage: OpenAIUsage(promptTokens: sp, completionTokens: sc)
                    )
                    let usageData = try JSONEncoder().encode(usageChunk)
                    try await writer.write(ByteBuffer(string: "data: \(String(data: usageData, encoding: .utf8) ?? "")\n\n"))
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                Self.applyKeepAlive(compReq.keepAlive, modelId: compReq.model, pool: inference.engine.pool)
                try await writer.finish(nil)
            } catch {
                NovaMLXLog.error("Completion stream error: \(error)")
                let (message, type, code) = Self.streamErrorFields(error)
                let errorDetail = OpenAIErrorDetail(message: message, type: type, code: code)
                let errorResp = OpenAIErrorResponse(error: errorDetail)
                if let data = try? JSONEncoder().encode(errorResp) {
                    try? await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                }
                try? await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                Self.applyKeepAlive(compReq.keepAlive, modelId: compReq.model, pool: inference.engine.pool)
                try? await writer.finish(nil)
            }
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
    }
}
