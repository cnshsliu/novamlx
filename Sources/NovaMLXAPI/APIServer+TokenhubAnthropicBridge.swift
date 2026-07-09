import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils

// MARK: - Tokenhub Anthropic ↔ OpenAI Bridge
//
// Activates when a client sends /v1/messages (Anthropic format) but the
// resolved tokenhub provider speaks OpenAI only (no `anthropicEndpoint` set
// on TokenhubProvider). Without this bridge, `handleTokenhubPassthrough`
// would forward the raw Anthropic body to `<endpoint>/messages`, which 404s
// on DeepSeek/GLM/Qwen-compat upstreams and the client sees an empty reply.
//
// The bridge:
//   1. Decodes the body as AnthropicRequest.
//   2. Maps messages via the existing `mapAnthropicMessages` (same mapper
//      the local /v1/messages path uses).
//   3. Builds an OpenAI /chat/completions request body.
//   4. POSTs to `<provider.endpoint>/chat/completions`.
//   5. Translates the OpenAI response (or SSE stream) back to Anthropic shape.
//
// Providers that natively speak Anthropic set `anthropicEndpoint` and bypass
// this bridge — they keep using raw passthrough.

extension NovaMLXAPIServer {

    // MARK: - Discriminator

    /// True when the client sent /v1/messages but the provider doesn't expose
    /// a native Anthropic endpoint — i.e., we need to translate.
    static func needsAnthropicBridge(provider: TokenhubProvider, clientPath: String) -> Bool {
        guard clientPath == "messages" else { return false }
        if let ep = provider.anthropicEndpoint, !ep.isEmpty { return false }
        return true
    }

    // MARK: - Shared Helpers (extracted from local paths — no behavior change)

    /// Convert Anthropic tool definitions to OpenAI function-tool shape.
    /// Extracted from the inline closure at APIServer.swift (~line 785).
    static func anthropicToolsToOpenAITools(_ tools: [AnthropicTool]) -> [[String: Any]] {
        tools.map { tool in
            var dict: [String: Any] = ["name": tool.name]
            if let desc = tool.description { dict["description"] = desc }
            dict["type"] = "function"
            dict["function"] = [
                "name": tool.name,
                "description": tool.description ?? "",
                "parameters": unwrapAnyCodable(tool.inputSchema)
            ] as [String: Any]
            return dict
        }
    }

    /// Build an AnthropicResponse from an upstream OpenAIResponse.
    /// Handles text content, reasoning_content (→ thinking block),
    /// tool_calls (→ tool_use blocks), and finish_reason → stop_reason mapping.
    static func buildAnthropicResponse(from openAI: OpenAIResponse, model: String) -> AnthropicResponse {
        var blocks: [AnthropicContentBlock] = []
        var stopReason = "end_turn"

        if let choice = openAI.choices.first {
            let msg = choice.message
            // Thinking block first (matches local-path ordering).
            if let reasoning = msg.reasoningContent, !reasoning.isEmpty {
                blocks.append(.init(type: "thinking", thinking: reasoning))
            }
            // Tool calls → tool_use blocks.
            if let toolCalls = msg.toolCalls, !toolCalls.isEmpty {
                for tc in toolCalls {
                    let inputObj = parseJSONArgs(tc.function.arguments)
                    blocks.append(.init(
                        type: "tool_use",
                        id: tc.id,
                        name: tc.function.name,
                        input: inputObj
                    ))
                }
            }
            // Text content last (Anthropic clients typically expect text after thinking).
            if let content = msg.content, let text = content.textValue, !text.isEmpty {
                blocks.append(.init(type: "text", text: text))
            }
            if let fr = choice.finishReason {
                stopReason = mapFinishReasonToAnthropic(fr)
            }
        }

        let usage = openAI.usage ?? OpenAIUsage(promptTokens: 0, completionTokens: 0)
        return AnthropicResponse(
            id: "msg_\(openAI.id.prefix(24))",
            model: model,
            content: blocks,
            stopReason: stopReason,
            usage: .init(
                inputTokens: usage.promptTokens,
                outputTokens: usage.completionTokens
            )
        )
    }

    /// OpenAI `finish_reason` → Anthropic `stop_reason`.
    /// stop → end_turn, tool_calls → tool_use, length → max_tokens.
    /// Unknown values pass through unchanged.
    static func mapFinishReasonToAnthropic(_ finishReason: String) -> String {
        switch finishReason {
        case "stop": return "end_turn"
        case "tool_calls", "function_call": return "tool_use"
        case "length": return "max_tokens"
        case "stop_sequence": return "stop_sequence"
        default: return finishReason
        }
    }

    /// Parse a JSON-encoded arguments string into AnyCodable.
    /// Returns an empty object on parse failure (tool_use with no input).
    static func parseJSONArgs(_ args: String) -> AnyCodable {
        guard !args.isEmpty,
              let data = args.data(using: .utf8) else {
            return .dictionary([:])
        }
        // Re-decode through AnyCodable's own decoder to preserve type fidelity.
        return (try? JSONDecoder().decode(AnyCodable.self, from: data)) ?? .dictionary([:])
    }

    // MARK: - Non-Streaming Bridge

    /// Translate Anthropic→OpenAI, forward, translate response back.
    /// Used when `anthropicReq.stream` is nil/false.
    static func handleTokenhubAnthropicBridge(
        anthropicReq: AnthropicRequest,
        provider: TokenhubProvider,
        tag: String?,
        rawBody: Data? = nil
    ) async throws -> Response {
        // Detect cache_control in the client body — the OpenAI upstream cannot
        // honor Anthropic prompt-caching, so the client's intent (paying for a
        // cache write) is silently dropped. Log a WARN so it's grep-able.
        Self.warnIfCacheControlDropped(rawBody: rawBody, provider: provider)

        let chatBody = buildOpenAIChatCompletionsBody(from: anthropicReq, remoteModel: provider.remoteModel)
        let url = URL(string: provider.endpoint)!.appendingPathComponent("chat/completions")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.timeoutInterval = 120
        let apiKey = effectiveApiKey(provider)
        if !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: chatBody)

        NovaMLXLog.info("[TokenhubBridge] -> \(provider.name) (\(url.path)) model=\(provider.remoteModel) stream=false")

        let start = ContinuousClock.now
        let (data, urlResponse) = try await URLSession.shared.data(for: urlRequest)
        let elapsed = ContinuousClock.now - start
        let latencyMs = durationToMs(elapsed)

        guard let http = urlResponse as? HTTPURLResponse else {
            TokenhubManager.shared.recordMetric(providerId: provider.id, success: false, latencyMs: latencyMs)
            return Response(status: .internalServerError)
        }

        if http.statusCode >= 400 {
            let body = String(data: data, encoding: .utf8)?.prefix(300) ?? "nil"
            NovaMLXLog.warning("[TokenhubBridge] \(provider.name) error HTTP \(http.statusCode): \(body)")
            TokenhubManager.shared.recordMetric(providerId: provider.id, success: false, latencyMs: latencyMs)
            var headers: HTTPFields = [.contentType: "application/json"]
            headers[.init("X-Tokenhub-Provider")!] = provider.name
            return Response(status: .init(integerLiteral: http.statusCode), headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
        }

        guard let openAIResp = try? JSONDecoder().decode(OpenAIResponse.self, from: data) else {
            NovaMLXLog.error("[TokenhubBridge] failed to decode OpenAI response from \(provider.name)")
            TokenhubManager.shared.recordMetric(providerId: provider.id, success: false, latencyMs: latencyMs)
            return Response(status: .badGateway)
        }

        let anthropicResp = buildAnthropicResponse(from: openAIResp, model: anthropicReq.model)
        TokenhubManager.shared.recordMetric(providerId: provider.id, success: true, latencyMs: latencyMs)

        var response = try Self.jsonResponse(anthropicResp)
        response.headers[.init("X-Tokenhub-Provider")!] = provider.name
        return response
    }

    // MARK: - Streaming Bridge

    /// Streaming variant: translate OpenAI SSE chunks → Anthropic SSE events.
    static func handleTokenhubAnthropicBridgeStream(
        anthropicReq: AnthropicRequest,
        provider: TokenhubProvider,
        tag: String?,
        rawBody: Data? = nil
    ) async throws -> Response {
        Self.warnIfCacheControlDropped(rawBody: rawBody, provider: provider)

        var chatBody = buildOpenAIChatCompletionsBody(from: anthropicReq, remoteModel: provider.remoteModel)
        chatBody["stream"] = true
        // Request usage in the final chunk so we can populate message_delta.usage.
        if var streamOptions = (chatBody["stream_options"] as? [String: Any]) {
            streamOptions["include_usage"] = true
            chatBody["stream_options"] = streamOptions
        } else {
            chatBody["stream_options"] = ["include_usage": true]
        }

        let url = URL(string: provider.endpoint)!.appendingPathComponent("chat/completions")
        var mutableRequest = URLRequest(url: url)
        mutableRequest.httpMethod = "POST"
        mutableRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        mutableRequest.timeoutInterval = 120
        let apiKey = effectiveApiKey(provider)
        if !apiKey.isEmpty {
            mutableRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        mutableRequest.httpBody = try JSONSerialization.data(withJSONObject: chatBody)
        // Sendable closure requires immutable capture.
        let urlRequest = mutableRequest

        NovaMLXLog.info("[TokenhubBridge] -> \(provider.name) (\(url.path)) model=\(provider.remoteModel) stream=true")

        let msgId = "msg_\(UUID().uuidString.prefix(24))"
        let providerName = provider.name
        let providerId = provider.id
        let requestedModel = anthropicReq.model
        let streamStart = ContinuousClock.now

        let responseBody: ResponseBody = .init { writer in
            let (bytes, urlResponse): (URLSession.AsyncBytes, URLResponse)
            do {
                (bytes, urlResponse) = try await URLSession.shared.bytes(for: urlRequest)
            } catch {
                NovaMLXLog.error("[TokenhubBridge] stream open failed: \(error)")
                TokenhubManager.shared.recordMetric(providerId: providerId, success: false, latencyMs: 0)
                try? await writer.finish(nil)
                return
            }
            guard let http = urlResponse as? HTTPURLResponse, http.statusCode == 200 else {
                let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 502
                NovaMLXLog.warning("[TokenhubBridge] \(providerName) stream HTTP \(statusCode)")
                TokenhubManager.shared.recordMetric(providerId: providerId, success: false, latencyMs: 0)
                try? await writer.finish(nil)
                return
            }

            // State machine for translating OpenAI deltas → Anthropic events.
            var currentBlockIndex = -1
            var currentBlockType: String? = nil  // "text" | "thinking" | "tool_use"
            // Per-tool-call index → Anthropic content block index. OpenAI streams
            // tool_calls by array index; we map each to its own Anthropic block.
            var toolCallBlockIndices: [Int: Int] = [:]
            var tokenCount = 0
            var inputTokens = 0

            func startBlock(_ type: String) async throws {
                currentBlockIndex += 1
                currentBlockType = type
                let evt = AnthropicStreamEvent.contentBlockStart(index: currentBlockIndex, blockType: type)
                let data = try JSONEncoder().encode(evt)
                try await writer.write(ByteBuffer(string: "event: content_block_start\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
            }

            func endBlock() async throws {
                try await writer.write(ByteBuffer(string: "event: content_block_stop\ndata: {}\n\n"))
                currentBlockType = nil
            }

            do {
                // message_start — emitted once at the beginning.
                let startEvt = AnthropicStreamEvent.messageStart(id: msgId, model: requestedModel)
                let startData = try JSONEncoder().encode(startEvt)
                try await writer.write(ByteBuffer(string: "event: message_start\ndata: \(String(data: startData, encoding: .utf8) ?? "{}")\n\n"))

                for try await line in bytes.lines {
                    let trimmed = line.trimmingCharacters(in: .whitespaces)
                    // OpenAI SSE format: lines start with "data: ".
                    guard trimmed.hasPrefix("data:") else { continue }
                    let payload = trimmed.dropFirst(5).trimmingCharacters(in: .whitespaces)
                    if payload == "[DONE]" { break }
                    guard let payloadData = payload.data(using: .utf8),
                          let chunk = try? JSONDecoder().decode(OpenAIStreamChunk.self, from: payloadData) else {
                        continue
                    }
                    // Capture usage from any chunk that carries it (final usage-only chunk has no choices).
                    if let usage = chunk.usage {
                        inputTokens = usage.promptTokens
                        tokenCount = usage.completionTokens
                    }
                    guard let choice = chunk.choices.first else {
                        continue
                    }

                    let delta = choice.delta

                    // Capture usage from any chunk that carries it (final chunk convention).
                    if let usage = chunk.usage {
                        inputTokens = usage.promptTokens
                        tokenCount = usage.completionTokens
                    }

                    // thinking_delta (reasoning_content).
                    if let reasoning = delta.reasoningContent, !reasoning.isEmpty {
                        if currentBlockType != "thinking" {
                            if currentBlockType != nil { try await endBlock() }
                            try await startBlock("thinking")
                        }
                        let evt = AnthropicStreamEvent.thinkingDelta(reasoning)
                        let data = try JSONEncoder().encode(evt)
                        try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                    }

                    // text_delta (content).
                    if let text = delta.content, !text.isEmpty {
                        if currentBlockType != "text" {
                            if currentBlockType != nil { try await endBlock() }
                            try await startBlock("text")
                        }
                        tokenCount += 1
                        let evt = AnthropicStreamEvent.textDelta(text)
                        let data = try JSONEncoder().encode(evt)
                        try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                    }

                    // tool_calls deltas.
                    if let toolCalls = delta.toolCalls {
                        for tcDelta in toolCalls {
                            // First time we see this index → start a new tool_use block.
                            if toolCallBlockIndices[tcDelta.index] == nil {
                                if currentBlockType != nil { try await endBlock() }
                                try await startBlock("tool_use")
                                toolCallBlockIndices[tcDelta.index] = currentBlockIndex
                                // Emit content_block_start payload carries the tool name/id
                                // via a partial_json_delta right after, since Anthropic splits
                                // the tool_use block start (with id+name) from input deltas.
                                // We send a minimal start event above; emit input_delta for
                                // the initial arguments chunk below.
                            }
                            if let args = tcDelta.function?.arguments, !args.isEmpty {
                                let inputEvt: [String: Any] = [
                                    "type": "content_block_delta",
                                    "delta": [
                                        "type": "input_json_delta",
                                        "partial_json": args
                                    ] as [String: Any],
                                    "index": toolCallBlockIndices[tcDelta.index] ?? currentBlockIndex
                                ]
                                let evtData = try JSONSerialization.data(withJSONObject: inputEvt)
                                try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: evtData, encoding: .utf8) ?? "{}")\n\n"))
                            }
                        }
                    }

                    // finish_reason → close current block + message_delta + message_stop.
                    if let finishReason = choice.finishReason {
                        if currentBlockType != nil { try await endBlock() }
                        let stopReason = mapFinishReasonToAnthropic(finishReason)
                        let usage = AnthropicUsage(inputTokens: inputTokens, outputTokens: tokenCount)
                        let evt = AnthropicStreamEvent.messageDelta(stopReason: stopReason, usage: usage)
                        let data = try JSONEncoder().encode(evt)
                        try await writer.write(ByteBuffer(string: "event: message_delta\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                        try await writer.write(ByteBuffer(string: "event: message_stop\ndata: {}\n\n"))
                        let elapsed = ContinuousClock.now - streamStart
                        let elapsedSec = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
                        NovaMLXLog.info("[TokenhubBridge] stream complete — \(tokenCount) tokens in \(String(format: "%.1f", elapsedSec))s")
                        break
                    }
                }

                // If upstream ended without an explicit finish_reason, still close cleanly.
                if currentBlockType != nil {
                    try await endBlock()
                    let usage = AnthropicUsage(inputTokens: inputTokens, outputTokens: tokenCount)
                    let evt = AnthropicStreamEvent.messageDelta(stopReason: "end_turn", usage: usage)
                    let data = try JSONEncoder().encode(evt)
                    try await writer.write(ByteBuffer(string: "event: message_delta\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                    try await writer.write(ByteBuffer(string: "event: message_stop\ndata: {}\n\n"))
                }

                try await writer.finish(nil)
                TokenhubManager.shared.recordMetric(providerId: providerId, success: true, latencyMs: durationToMs(ContinuousClock.now - streamStart))
            } catch {
                NovaMLXLog.error("[TokenhubBridge] stream error: \(error)")
                TokenhubManager.shared.recordMetric(providerId: providerId, success: false, latencyMs: durationToMs(ContinuousClock.now - streamStart))
                try? await writer.finish(nil)
            }
        }

        var headers: HTTPFields = [
            .contentType: "text/event-stream",
            .cacheControl: "no-cache",
            .init("X-Tokenhub-Provider")!: provider.name
        ]
        return Response(status: .ok, headers: headers, body: responseBody)
    }

    // MARK: - Body Builder

    /// Translate an AnthropicRequest into an OpenAI /chat/completions body dict.
    /// Uses mapAnthropicMessages (same mapper as the local /v1/messages path)
    /// so message/tool_result fidelity is identical to local inference.
    static func buildOpenAIChatCompletionsBody(from req: AnthropicRequest, remoteModel: String) -> [String: Any] {
        let messages = (try? mapAnthropicMessages(req.messages, system: req.system)) ?? []
        var body: [String: Any] = [
            "model": remoteModel,
            "messages": messages.map { msg -> [String: Any] in
                var entry: [String: Any] = ["role": msg.role.rawValue]
                if let content = msg.content { entry["content"] = content }
                if let name = msg.name { entry["name"] = name }
                if let toolCallId = msg.toolCallId { entry["tool_call_id"] = toolCallId }
                if let toolCalls = msg.toolCalls {
                    entry["tool_calls"] = toolCalls.map { tc in
                        [
                            "id": tc.id,
                            "type": "function",
                            "function": ["name": tc.functionName, "arguments": tc.arguments]
                        ] as [String: Any]
                    }
                }
                return entry
            }
        ]
        body["max_tokens"] = req.maxTokens
        if let t = req.temperature { body["temperature"] = t }
        if let topP = req.topP { body["top_p"] = topP }
        if let stop = req.stopSequences, !stop.isEmpty { body["stop"] = stop }
        if let tools = req.tools, !tools.isEmpty {
            body["tools"] = anthropicToolsToOpenAITools(tools)
        }
        if let tc = req.toolChoice {
            // AnyCodable → already in OpenAI shape if it's a dict; pass through.
            body["tool_choice"] = unwrapAnyCodable(tc)
        }
        return body
    }

    // MARK: - Auth Helper (mirrors handleTokenhubPassthrough)

    private static func effectiveApiKey(_ p: TokenhubProvider) -> String {
        if p.tags.contains("managed") { return AuthCache.loadSession() ?? "" }
        return p.apiKey
    }

    // MARK: - Cache-Control Drop Detection

    /// Scan the raw Anthropic client body for any `cache_control` field. If found,
    /// emit a WARN log — the OpenAI-format upstream can't honor Anthropic prompt
    /// caching, so the client's request was silently downgraded. Also counts
    /// occurrences so users can grep the magnitude.
    static func warnIfCacheControlDropped(rawBody: Data?, provider: TokenhubProvider) {
        guard let data = rawBody,
              let s = String(data: data, encoding: .utf8) else { return }
        // Cheap substring scan — JSON structure varies (per-message, per-block,
        // per-tool) so a recursive walk is overkill. The literal `"cache_control"`
        // key only appears in Anthropic-format bodies, never as a value or in
        // OpenAI-format bodies routed through this bridge.
        let needle = "\"cache_control\""
        var count = 0
        var searchRange = s.startIndex..<s.endIndex
        while let r = s.range(of: needle, range: searchRange) {
            count += 1
            searchRange = r.upperBound..<s.endIndex
        }
        if count > 0 {
            NovaMLXLog.warning("[TokenhubBridge] Client sent \(count) cache_control block(s) but upstream \(provider.name) is OpenAI-format — prompt caching silently dropped (translated to /chat/completions)")
        }
    }
}
