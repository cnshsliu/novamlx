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

    // MARK: - TokenHub Responses API Passthrough
    // Converts Responses API request → Chat Completions → forward → convert response back

    static func handleTokenhubResponsesPassthrough(
        req: OpenAIResponseRequest,
        rawBody: Data,
        inference: InferenceService
    ) async throws -> Response {
        // DEBUG: dump raw request
        let debugRawURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("tokenhub_raw_request.json")
        try? rawBody.write(to: debugRawURL)
        NovaMLXLog.info("[Tokenhub/Responses] RAW REQUEST dumped to \(debugRawURL.path) prevId=\(req.previousResponseId ?? "none") input=\(req.input)")
        NovaMLXLog.info("[Tokenhub/Responses] TOOLS after Codable decode: \(req.tools?.count ?? -1) tools, stream=\(req.stream ?? false)")
        // Also dump raw JSON tools for comparison
        if let rawObj = try? JSONSerialization.jsonObject(with: rawBody) as? [String: Any],
           let rawTools = rawObj["tools"] as? [[String: Any]] {
            NovaMLXLog.info("[Tokenhub/Responses] RAW JSON tools count: \(rawTools.count)")
            for (i, t) in rawTools.enumerated() {
                let ttype = t["type"] as? String ?? "MISSING"
                let name = t["name"] as? String ?? "MISSING"
                NovaMLXLog.info("[Tokenhub/Responses]   raw tool[\(i)] type=\(ttype) name=\(name)")
            }
        } else {
            NovaMLXLog.info("[Tokenhub/Responses] RAW JSON: no tools field found in request body")
        }

        // Plain "tknet" (no provider) used to mean "load-balance across all
        // providers". That path is now superseded by named LBs via LBProxy
        // (use `model = "lb:<slug>"`). Reject bare tknet so callers migrate.
        // The LB dispatch for the Responses API happens upstream in
        // APIServer.swift (look for `Self.isLBModel(req.model)`).
        if req.model.lowercased() == "tknet" {
            return try Self.jsonResponse(
                ["error": ["message": "Bare 'tknet' load-balanced dispatch has been replaced by named LBs. Use 'lb:<slug>' (e.g. 'lb:coding-pool') or 'tknet:<provider-name>' for a specific provider.", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        guard let provider = TokenhubManager.shared.resolve(modelName: req.model, tag: nil) else {
            return try Self.jsonResponse(
                ["error": ["message": "Unknown tokenhub provider: \(req.model)", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        let isLB = false  // LB dispatch disabled until Task 7 (LBProxy)
        let maxRetries = isLB ? 2 : 0
        var triedProviders = Set<String>()
        var lastProvider = provider

        func effectiveApiKey(_ p: TokenhubProvider) -> String {
            if p.tags.contains("managed") { return AuthCache.loadSession() ?? "" }
            return p.apiKey
        }

        // Convert Responses API request → Chat Completions request
        func buildChatCompletionsBody(remoteModel: String) async throws -> Data {
            var messages: [[String: Any]] = []
            var messageImageBlocks: [Int: [String]] = [:]  // messageIdx → imageURLs for preprocessing
            if let instructions = req.instructions, !instructions.isEmpty {
                messages.append(["role": "system", "content": instructions])
            }

            // Resolve previous_response_id: prepend stored conversation history
            if let prevId = req.resolvedPreviousResponseId {
                if let prevResp = ResponseStore.shared.get(prevId) {
                    let prevMsgs = Self.extractMessagesFromResponse(prevResp)
                    for msg in prevMsgs {
                        var entry: [String: Any] = ["role": msg.role.rawValue]
                        if let content = msg.content { entry["content"] = content }
                        if let toolCalls = msg.toolCalls {
                            entry["tool_calls"] = toolCalls.map { tc in
                                [
                                    "id": tc.id,
                                    "type": "function",
                                    "function": ["name": tc.functionName, "arguments": tc.arguments]
                                ] as [String: Any]
                            }
                        }
                        messages.append(entry)
                    }
                } else {
                    NovaMLXLog.warning("[Tokenhub/Responses] previous_response_id '\(prevId)' not found in ResponseStore")
                }
            }

            switch req.input {
            case .text(let prompt):
                messages.append(["role": "user", "content": prompt])
            case .items(let items):
                for item in items {
                    switch item {
                    case .message(let msg):
                        let role = msg.role == "developer" ? "system" : msg.role
                        let (text, imageURLs) = ImagePreprocessor.extractContent(msg.content)
                        if imageURLs.isEmpty {
                            messages.append(["role": role, "content": text])
                        } else if lastProvider.supportsVision {
                            // Provider supports vision — forward images natively
                            NovaMLXLog.info("[Tokenhub/Responses] Provider \(lastProvider.name) supports vision, forwarding \(imageURLs.count) images natively")
                            var contentParts: [[String: Any]] = []
                            if !text.isEmpty {
                                contentParts.append(["type": "text", "text": text])
                            }
                            for url in imageURLs {
                                contentParts.append(["type": "image_url", "image_url": ["url": url]])
                            }
                            messages.append(["role": role, "content": contentParts])
                        } else {
                            // Text-only provider — collect images for OCR preprocessing
                            NovaMLXLog.info("[Tokenhub/Responses] Provider \(lastProvider.name) does NOT support vision, queuing \(imageURLs.count) images for OCR fallback")
                            messageImageBlocks[messages.count] = imageURLs
                            messages.append(["role": role, "content": text])
                        }
                    case .functionCall(let fc):
                        // Validate arguments — model may produce invalid JSON
                        var args = fc.arguments
                        if !isValidJSON(args) {
                            // Try to fix: quote unquoted values like {"key": some value}
                            args = fixJSONArguments(args)
                        }
                        messages.append([
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [[
                                "id": fc.callId,
                                "type": "function",
                                "function": ["name": fc.name, "arguments": args]
                            ] as [String: Any]]
                        ])
                    case .functionCallOutput(let fcOut):
                        messages.append(["role": "tool", "content": fcOut.output, "tool_call_id": fcOut.callId])
                    case .reasoning(let r):
                        // DeepSkip requires reasoning_content on assistant messages when using thinking mode
                        let summaryText = (r.summary ?? []).map { $0.text }.joined(separator: "\n")
                        if !summaryText.isEmpty {
                            for j in stride(from: messages.count - 1, through: 0, by: -1) {
                                if messages[j]["role"] as? String == "assistant" {
                                    messages[j]["reasoning_content"] = summaryText
                                    break
                                }
                            }
                        }
                    case .skipped:
                        break
                    }
                }
            case .none:
                break
            }

            // Merge consecutive same-role messages — most providers reject them
            var merged: [[String: Any]] = []
            for msg in messages {
                guard let last = merged.last else { merged.append(msg); continue }
                let lastRole = last["role"] as? String ?? ""
                let msgRole = msg["role"] as? String ?? ""
                if lastRole == msgRole {
                    if msgRole == "user" {
                        let prev = last["content"] as? String ?? ""
                        let cur = msg["content"] as? String ?? ""
                        merged[merged.count - 1]["content"] = prev + "\n" + cur
                    } else if msgRole == "assistant" {
                        var combined = last
                        if let tc = msg["tool_calls"] as? [[String: Any]] {
                            var existing = combined["tool_calls"] as? [[String: Any]] ?? []
                            existing.append(contentsOf: tc)
                            combined["tool_calls"] = existing
                            if let c = msg["content"] as? String, !c.isEmpty {
                                let prev = combined["content"] as? String ?? ""
                                combined["content"] = prev.isEmpty ? c : prev + "\n" + c
                            }
                        } else if let c = msg["content"] as? String, !c.isEmpty {
                            let prev = combined["content"] as? String ?? ""
                            combined["content"] = prev.isEmpty ? c : prev + "\n" + c
                        }
                        merged[merged.count - 1] = combined
                    } else {
                        merged.append(msg)
                    }
                } else {
                    merged.append(msg)
                }
            }
            messages = merged

            // Image preprocessing: convert images to text descriptions for text-only providers
            if !messageImageBlocks.isEmpty && !lastProvider.supportsVision {
                let backend = ImagePreprocessor.resolveBackend(provider: lastProvider, inference: inference)
                let result = await ImagePreprocessor.preprocess(
                    messages: messages,
                    imageBlocks: messageImageBlocks,
                    backend: backend,
                    inference: inference
                )
                messages = result.messages
                if result.imagesProcessed > 0 {
                    NovaMLXLog.info("[Tokenhub/Responses] Preprocessed \(result.imagesProcessed) images for text-only provider \(lastProvider.name)")
                }
            }

            // DeepSeek requires reasoning_content on ALL assistant messages when thinking mode is active
            let hasReasoning = messages.contains { ($0["role"] as? String) == "assistant" && $0["reasoning_content"] != nil }
            if hasReasoning {
                for i in messages.indices where (messages[i]["role"] as? String) == "assistant" {
                    if messages[i]["reasoning_content"] == nil {
                        messages[i]["reasoning_content"] = ""
                    }
                }
            }

            // Helper: check if string is valid JSON
            func isValidJSON(_ str: String) -> Bool {
                guard !str.isEmpty,
                      let data = str.data(using: .utf8),
                      let _ = try? JSONSerialization.jsonObject(with: data) else { return false }
                return true
            }

            // Helper: fix common malformed JSON from model output
            func fixJSONArguments(_ str: String) -> String {
                guard let regex = try? NSRegularExpression(pattern: ":\\s*([^\"\\[\\{\\dtnf][^\\}]*?)\\s*([\\},])") else { return "{}" }
                let range = NSRange(str.startIndex..., in: str)
                var fixed = str
                if let match = regex.firstMatch(in: str, range: range) {
                    if let valRange = Range(match.range(at: 1), in: str),
                       let endRange = Range(match.range(at: 2), in: str) {
                        let value = String(fixed[valRange])
                        let escaped = value.replacingOccurrences(of: "\"", with: "\\\"")
                        fixed = String(fixed[..<valRange.lowerBound]) + "\"" + escaped + "\"" + String(fixed[endRange.lowerBound...])
                    }
                }
                return fixed
            }

            var body: [String: Any] = [
                "model": remoteModel,
                "messages": messages
            ]
            if let temp = req.temperature { body["temperature"] = temp }
            if let topP = req.topP { body["top_p"] = topP }
            if let maxTokens = req.maxOutputTokens { body["max_tokens"] = maxTokens }
            if let stream = req.stream { body["stream"] = stream }
            if let serviceTier = req.serviceTier { body["service_tier"] = serviceTier }

            // Convert tools — only forward function-type tools with valid names
            if let rawTools = (try? JSONSerialization.jsonObject(with: rawBody)) as? [String: Any],
               let toolsArray = rawTools["tools"] as? [[String: Any]] {
                var functionTools: [[String: Any]] = []
                for tool in toolsArray {
                    let toolType = tool["type"] as? String ?? "function"
                    guard toolType == "function" || tool["function"] != nil else { continue }

                    if let name = tool["name"] as? String, !name.isEmpty {
                        var fnDict: [String: Any] = [
                            "name": name,
                            "description": tool["description"] as? String ?? ""
                        ]
                        if let params = tool["parameters"] as? [String: Any] {
                            fnDict["parameters"] = params
                        }
                        functionTools.append(["type": "function", "function": fnDict])
                    }
                    else if let fn = tool["function"] as? [String: Any],
                            let fnName = fn["name"] as? String, !fnName.isEmpty {
                        functionTools.append(tool)
                    }
                }
                if !functionTools.isEmpty {
                    body["tools"] = functionTools
                }
            } else if let tools = req.tools, !tools.isEmpty {
                let functionTools = tools.filter { $0.type == "function" && !$0.name.isEmpty }
                if !functionTools.isEmpty {
                    body["tools"] = functionTools.map { tool -> [String: Any] in
                        var fn: [String: Any] = [
                            "name": tool.name,
                            "description": tool.description ?? ""
                        ]
                        if let params = tool.parameters,
                           let paramData = try? JSONEncoder().encode(params),
                           let paramObj = try? JSONSerialization.jsonObject(with: paramData) {
                            fn["parameters"] = paramObj
                        }
                        return ["type": "function", "function": fn]
                    }
                }
            }

            // Forward tool_choice — normalize Responses format to Chat Completions format
            if let tc = req.toolChoice {
                switch tc {
                case .string(let s):
                    body["tool_choice"] = s
                case .dictionary(let dict):
                    if case .string("function") = dict["type"],
                       case .string(let name) = dict["name"] {
                        body["tool_choice"] = ["type": "function", "function": ["name": name]]
                    } else if let data = try? JSONEncoder().encode(dict),
                              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        body["tool_choice"] = obj
                    }
                default:
                    break
                }
            }

            // Convert text.format → response_format
            if let textFormat = req.text?.format {
                var rf: [String: Any] = ["type": textFormat.type]
                if let schema = textFormat.schema,
                   let schemaData = try? JSONEncoder().encode(schema),
                   let schemaObj = try? JSONSerialization.jsonObject(with: schemaData) {
                    rf["json_schema"] = schemaObj
                }
                body["response_format"] = rf
            }

            // Safety: strip any tools that somehow lack a valid function.name
            if var toolsArray = body["tools"] as? [[String: Any]] {
                toolsArray = toolsArray.filter { tool in
                    guard let fn = tool["function"] as? [String: Any],
                          let name = fn["name"] as? String, !name.isEmpty else {
                        return false
                    }
                    return true
                }
                body["tools"] = toolsArray.isEmpty ? nil : toolsArray
            }

            // DEBUG: dump final body after all conversions
            if let dumpData = try? JSONSerialization.data(withJSONObject: body, options: .prettyPrinted) {
                let debugURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("tokenhub_debug_messages.json")
                try? dumpData.write(to: debugURL)
                NovaMLXLog.info("[Tokenhub/Responses] DEBUG final body: \(messages.count) msgs, \(body["tools"] != nil ? "with tools" : "no tools")")
            }

            return try JSONSerialization.data(withJSONObject: body)
        }

        // Convert Chat Completions response → Responses API response
        func convertToResponsesResponse(_ chatData: Data, model: String) -> Data? {
            guard let json = try? JSONSerialization.jsonObject(with: chatData) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let first = choices.first,
                  let message = first["message"] as? [String: Any] else { return nil }

            let content = message["content"] as? String ?? ""
            let usage = json["usage"] as? [String: Any]
            let responseId = "resp_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"

            var toolCalls: [[String: Any]] = []
            if let tc = message["tool_calls"] as? [[String: Any]] {
                for (idx, call) in tc.enumerated() {
                    let fn = call["function"] as? [String: Any] ?? [:]
                    toolCalls.append([
                        "type": "function_call",
                        "id": "fc_\(responseId.suffix(12))_\(idx)",
                        "status": "completed",
                        "call_id": call["id"] ?? "",
                        "name": fn["name"] ?? "",
                        "arguments": fn["arguments"] ?? ""
                    ])
                }
            }

            var output: [[String: Any]] = []

            if !content.isEmpty {
                output.append([
                    "type": "message",
                    "id": "msg_\(responseId.suffix(12))",
                    "status": "completed",
                    "role": "assistant",
                    "content": [["type": "output_text", "text": content, "annotations": []]]
                ])
            }
            for tc in toolCalls {
                output.append(tc)
            }

            var response: [String: Any] = [
                "id": responseId,
                "object": "response",
                "created_at": Int(Date().timeIntervalSince1970),
                "completed_at": Int(Date().timeIntervalSince1970),
                "model": model,
                "status": "completed",
                "output": output
            ]
            // Echo back request params per OpenAI spec
            if let v = req.instructions { response["instructions"] = v }
            if let v = req.maxOutputTokens { response["max_output_tokens"] = v }
            if let v = req.temperature { response["temperature"] = v }
            if let v = req.topP { response["top_p"] = v }
            if let v = req.previousResponseId { response["previous_response_id"] = v }
            if let v = req.metadata { response["metadata"] = v }
            if let v = req.store { response["store"] = v }
            if let v = req.truncation { response["truncation"] = v }
            if let v = req.parallelToolCalls { response["parallel_tool_calls"] = v }
            if let v = req.toolChoice {
                if let data = try? JSONEncoder().encode(v),
                   let obj = try? JSONSerialization.jsonObject(with: data) {
                    response["tool_choice"] = obj
                }
            }
            if let v = req.tools, !v.isEmpty {
                if let data = try? JSONEncoder().encode(v),
                   let obj = try? JSONSerialization.jsonObject(with: data) {
                    response["tools"] = obj
                }
            }
            if let v = req.reasoning {
                if let data = try? JSONEncoder().encode(v),
                   let obj = try? JSONSerialization.jsonObject(with: data) {
                    response["reasoning"] = obj
                }
            }
            if let v = req.text {
                if let data = try? JSONEncoder().encode(v),
                   let obj = try? JSONSerialization.jsonObject(with: data) {
                    response["text"] = obj
                }
            }
            if let usage {
                response["usage"] = [
                    "input_tokens": usage["prompt_tokens"] ?? 0,
                    "output_tokens": usage["completion_tokens"] ?? 0,
                    "total_tokens": usage["total_tokens"] ?? 0
                ]
            }

            // Store for previous_response_id support — include user input for multi-turn
            var storeResponse = response
            var storeOutput = output
            var userText: String?
            switch req.input {
            case .text(let t): userText = t
            case .items(let items):
                userText = items.compactMap { item -> String? in
                    if case .message(let msg) = item { return msg.content.textValue }
                    return nil
                }.joined(separator: "\n")
            case .none: break
            }
            if let ut = userText, !ut.isEmpty {
                storeOutput.insert([
                    "type": "message",
                    "id": "msg_user_\(responseId.suffix(12))",
                    "status": "completed",
                    "role": "user",
                    "content": [["type": "output_text", "text": ut, "annotations": []]]
                ], at: 0)
            }
            storeResponse["output"] = storeOutput
            if let respObj = try? JSONDecoder().decode(OpenAIResponseObject.self, from: try JSONSerialization.data(withJSONObject: storeResponse)) {
                ResponseStore.shared.put(respObj)
                if let convId = req.conversation?.id {
                    ConversationStore.shared.record(conversationId: convId, responseId: responseId)
                }
            }

            return try? JSONSerialization.data(withJSONObject: response)
        }

        for attempt in 0...maxRetries {
            triedProviders.insert(lastProvider.name)
            let isStreaming = req.stream ?? false

            // Provider natively supports /v1/responses → raw passthrough, no conversion
            if lastProvider.supportsResponsesAPI {
                NovaMLXLog.info("[Tokenhub/Responses] -> \(lastProvider.name) RAW PASSTHROUGH streaming=\(isStreaming)\(attempt > 0 ? " retry#\(attempt)" : "")")

                var rawObj = (try? JSONSerialization.jsonObject(with: rawBody)) as? [String: Any] ?? [:]
                rawObj["model"] = lastProvider.remoteModel
                let forwardBody = try? JSONSerialization.data(withJSONObject: rawObj)

                let baseURL = URL(string: lastProvider.endpoint)!
                var urlRequest = URLRequest(url: baseURL.appendingPathComponent("responses"))
                urlRequest.httpMethod = "POST"
                urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if !effectiveApiKey(lastProvider).isEmpty {
                    urlRequest.setValue("Bearer \(effectiveApiKey(lastProvider))", forHTTPHeaderField: "Authorization")
                }
                urlRequest.timeoutInterval = 120
                urlRequest.httpBody = forwardBody

                if isStreaming {
                    let start = ContinuousClock.now
                    let (bytes, urlResponse) = try await URLSession.shared.bytes(for: urlRequest)
                    guard let http = urlResponse as? HTTPURLResponse, http.statusCode == 200 else {
                        let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 502
                        NovaMLXLog.warning("[Tokenhub/Responses] \(lastProvider.name) raw passthrough streaming failed HTTP \(statusCode)")
                        let elapsed = ContinuousClock.now - start
                        TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: false, latencyMs: durationToMs(elapsed))
                        if isLB, let next = pickRetryProvider(modelName: req.model, tag: nil, exclude: triedProviders) {
                            lastProvider = next
                            continue
                        }
                        return Response(status: .init(integerLiteral: statusCode))
                    }
                    let elapsed = ContinuousClock.now - start
                    TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: true, latencyMs: durationToMs(elapsed))

                    let providerName = lastProvider.name
                    let responseBody = ResponseBody { writer in
                        do {
                            for try await line in bytes.lines {
                                try await writer.write(ByteBuffer(string: "\(line)\n"))
                            }
                        } catch {
                            NovaMLXLog.warning("[Tokenhub/Responses] Raw passthrough stream error: \(error)")
                        }
                    }
                    var headers = HTTPFields()
                    headers[.init("X-Tokenhub-Provider")!] = providerName
                    return Response(status: .ok, headers: headers, body: responseBody)
                } else {
                    let start = ContinuousClock.now
                    let (data, urlResponse) = try await URLSession.shared.data(for: urlRequest)
                    let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 502
                    let elapsed = ContinuousClock.now - start
                    let success = (200...299).contains(statusCode)
                    TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: success, latencyMs: durationToMs(elapsed))

                    if !success {
                        NovaMLXLog.warning("[Tokenhub/Responses] \(lastProvider.name) raw passthrough failed HTTP \(statusCode)")
                        if isLB, let next = pickRetryProvider(modelName: req.model, tag: nil, exclude: triedProviders) {
                            lastProvider = next
                            continue
                        }
                        return Response(status: .init(integerLiteral: statusCode), body: ResponseBody(byteBuffer: ByteBuffer(data: data)))
                    }

                    var hdrs = HTTPFields()
                    hdrs[.init("X-Tokenhub-Provider")!] = lastProvider.name
                    return Response(status: .ok, headers: hdrs, body: ResponseBody(byteBuffer: ByteBuffer(data: data)))
                }
            }

            // Provider does NOT support /v1/responses → convert Responses→ChatCompletions
            NovaMLXLog.info("[Tokenhub/Responses] -> \(lastProvider.name) remoteModel=\(lastProvider.remoteModel) streaming=\(isStreaming)\(attempt > 0 ? " retry#\(attempt)" : "")")

            let bodyData = try await buildChatCompletionsBody(remoteModel: lastProvider.remoteModel)
            let baseURL = URL(string: lastProvider.endpoint)!
            var urlRequest = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
            urlRequest.httpMethod = "POST"
            urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if !effectiveApiKey(lastProvider).isEmpty {
                urlRequest.setValue("Bearer \(effectiveApiKey(lastProvider))", forHTTPHeaderField: "Authorization")
            }
            urlRequest.timeoutInterval = 120
            urlRequest.httpBody = bodyData

            if isStreaming {
                let start = ContinuousClock.now
                let (bytes, urlResponse) = try await URLSession.shared.bytes(for: urlRequest)
                guard let http = urlResponse as? HTTPURLResponse, http.statusCode == 200 else {
                    let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 502
                    NovaMLXLog.warning("[Tokenhub/Responses] \(lastProvider.name) streaming failed HTTP \(statusCode)")
                    let elapsed = ContinuousClock.now - start
                    TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: false, latencyMs: durationToMs(elapsed))
                    if isLB, let next = pickRetryProvider(modelName: req.model, tag: nil, exclude: triedProviders) {
                        lastProvider = next
                        continue
                    }
                    return Response(status: .init(integerLiteral: statusCode))
                }
                let elapsed = ContinuousClock.now - start
                TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: true, latencyMs: durationToMs(elapsed))

                let responseId = "resp_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
                let msgId = "msg_\(responseId.suffix(12))"
                let model = lastProvider.remoteModel
                let providerName = lastProvider.name

                let responseBody = ResponseBody { writer in
                    var sequenceNumber = 0
                    do {
                        func sse(_ event: String, _ data: Encodable) async throws {
                            sequenceNumber += 1
                            var jsonData = try JSONEncoder().encode(data)
                            if var obj = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
                                obj["sequence_number"] = sequenceNumber
                                if let injected = try? JSONSerialization.data(withJSONObject: obj) {
                                    jsonData = injected
                                }
                            }
                            try await writer.write(ByteBuffer(string: "event: \(event)\ndata: \(String(data: jsonData, encoding: .utf8) ?? "")\n\n"))
                        }

                        let emptyResp = ResponsesSSEResponse(id: responseId, status: "in_progress", model: model)
                        try await sse("response.created", ResponsesSSECreated(response: emptyResp))
                        try await sse("response.in_progress", ResponsesSSECreated(response: emptyResp))

                        var fullText = ""
                        var textMessageStarted = false
                        var outputItems: [ResponseOutputItem] = []
                        var currentOutputIndex = 0

                        struct ToolCallAccumulator {
                            var id: String
                            var callId: String
                            var name: String
                            var arguments: String
                        }
                        var toolCalls: [Int: ToolCallAccumulator] = [:]

                        var reasoningId: String? = nil
                        var reasoningText = ""
                        var reasoningStarted = false

                        func startTextMessage() async throws {
                            guard !textMessageStarted else { return }
                            textMessageStarted = true
                            try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .message(ResponseOutputMessage(id: msgId, status: "in_progress", content: []))))
                            try await sse("response.content_part.added", ResponsesSSEContentPartAdded(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, part: ResponseContentItem(text: "")))
                        }

                        func finishTextMessage() async throws {
                            guard textMessageStarted else { return }
                            textMessageStarted = false
                            try await sse("response.output_text.done", ResponsesSSETextDone(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, text: fullText))
                            try await sse("response.content_part.done", ResponsesSSEContentPartDone(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, part: ResponseContentItem(text: fullText)))
                            try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .message(ResponseOutputMessage(id: msgId, content: [ResponseContentItem(text: fullText)]))))
                            outputItems.append(.message(ResponseOutputMessage(id: msgId, content: [ResponseContentItem(text: fullText)])))
                            currentOutputIndex += 1
                        }

                        for try await line in bytes.lines {
                            guard line.hasPrefix("data: ") else { continue }
                            if line == "data: [DONE]" { break }

                            let jsonStr = String(line.dropFirst(6))
                            guard let jsonData = jsonStr.data(using: .utf8),
                                  let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                                  let choices = json["choices"] as? [[String: Any]],
                                  let first = choices.first else { continue }

                            let delta = first["delta"] as? [String: Any] ?? [:]

                            if let content = delta["content"] as? String, !content.isEmpty {
                                try await startTextMessage()
                                fullText += content
                                try await sse("response.output_text.delta", ResponsesSSETextDelta(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, delta: content))
                            }

                            if let reasoningContent = delta["reasoning_content"] as? String, !reasoningContent.isEmpty {
                                if !reasoningStarted {
                                    reasoningStarted = true
                                    let rsId = "rs_\(responseId.suffix(12))"
                                    reasoningId = rsId
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, status: "in_progress"))))
                                }
                                reasoningText += reasoningContent
                                if let rsId = reasoningId {
                                    try await sse("response.reasoning.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: reasoningContent))
                                    try await sse("response.reasoning_text.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: reasoningContent))
                                }
                            }

                            if let toolCallDeltas = delta["tool_calls"] as? [[String: Any]] {
                                try await finishTextMessage()

                                if reasoningStarted, let rsId = reasoningId {
                                    let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                                    try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                try await sse("response.reasoning_text.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                    try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, summary: summary))))
                                    outputItems.append(.reasoning(ResponseOutputReasoning(id: rsId, summary: summary)))
                                    currentOutputIndex += 1
                                    reasoningStarted = false
                                }

                                for tcDelta in toolCallDeltas {
                                    let tcIndex = tcDelta["index"] as? Int ?? 0
                                    if toolCalls[tcIndex] == nil {
                                        let tcId = "fc_\(responseId.suffix(12))_\(tcIndex)"
                                        let callId = tcDelta["id"] as? String ?? "call_\(tcId)"
                                        let fn = tcDelta["function"] as? [String: Any] ?? [:]
                                        let name = fn["name"] as? String ?? ""

                                        let outputIdx = currentOutputIndex + tcIndex
                                        toolCalls[tcIndex] = ToolCallAccumulator(id: tcId, callId: callId, name: name, arguments: "")

                                        try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: outputIdx, item: .functionCall(ResponseOutputFunctionCall(id: tcId, callId: callId, name: name, arguments: "", status: "in_progress"))))
                                    }

                                    if let fn = tcDelta["function"] as? [String: Any],
                                       let argsDelta = fn["arguments"] as? String, !argsDelta.isEmpty {
                                        toolCalls[tcIndex]?.arguments += argsDelta
                                        let outputIdx = currentOutputIndex + tcIndex
                                        let tc = toolCalls[tcIndex]!
                                        try await sse("response.function_call_arguments.delta", ResponsesSSEFunctionCallArgsDelta(itemId: tc.id, outputIndex: outputIdx, callId: tc.callId, delta: argsDelta))
                                    }
                                }
                            }
                        }

                        try await finishTextMessage()

                        if reasoningStarted, let rsId = reasoningId {
                            let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                            try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                            try await sse("response.reasoning_text.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                            try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, summary: summary))))
                            outputItems.append(.reasoning(ResponseOutputReasoning(id: rsId, summary: summary)))
                            currentOutputIndex += 1
                        }

                        for (tcIndex, tc) in toolCalls.sorted(by: { $0.key < $1.key }) {
                            let outputIdx = currentOutputIndex + tcIndex
                            try await sse("response.function_call_arguments.done", ResponsesSSEFunctionCallArgsDone(itemId: tc.id, outputIndex: outputIdx, callId: tc.callId, arguments: tc.arguments))
                            try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: outputIdx, item: .functionCall(ResponseOutputFunctionCall(id: tc.id, callId: tc.callId, name: tc.name, arguments: tc.arguments))))
                            outputItems.append(.functionCall(ResponseOutputFunctionCall(id: tc.id, callId: tc.callId, name: tc.name, arguments: tc.arguments)))
                        }
                        if !toolCalls.isEmpty {
                            currentOutputIndex += toolCalls.count
                        }

                        var allOutputItems: [ResponseOutputItem] = []
                        var userTextForStorage: String?
                        switch req.input {
                        case .text(let t): userTextForStorage = t
                        case .items(let items):
                            userTextForStorage = items.compactMap { item -> String? in
                                if case .message(let msg) = item { return msg.content.textValue }
                                return nil
                            }.joined(separator: "\n")
                        case .none: break
                        }
                        if let ut = userTextForStorage, !ut.isEmpty {
                            allOutputItems.append(.message(ResponseOutputMessage(
                                id: "msg_user_\(responseId.suffix(12))",
                                role: "user",
                                content: [ResponseContentItem(text: ut)]
                            )))
                        }
                        allOutputItems.append(contentsOf: outputItems)

                        let clientResp = OpenAIResponseObject(id: responseId, model: model, output: outputItems).withRequestEcho(from: req)
                        try await sse("response.completed", ResponsesSSECompleted(response: clientResp))
                        let storeResp = OpenAIResponseObject(id: responseId, model: model, output: allOutputItems)
                        ResponseStore.shared.put(storeResp)
                        if let convId = req.conversation?.id {
                            ConversationStore.shared.record(conversationId: convId, responseId: responseId)
                        }
                        try await writer.finish(nil)
                    } catch {
                        sequenceNumber += 1
                        let errEvent = ResponsesSSEError(code: "ERR_UPSTREAM", message: error.localizedDescription)
                        if let errData = try? JSONEncoder().encode(errEvent) {
                            var obj = (try? JSONSerialization.jsonObject(with: errData) as? [String: Any]) ?? [:]
                            obj["sequence_number"] = sequenceNumber
                            if let finalData = try? JSONSerialization.data(withJSONObject: obj) {
                                try? await writer.write(ByteBuffer(string: "event: error\ndata: \(String(data: finalData, encoding: .utf8) ?? "{}")\n\n"))
                            }
                        }
                        try? await writer.finish(nil)
                    }
                }

                var headers = HTTPFields()
                headers[.contentType] = "text/event-stream"
                headers[.cacheControl] = "no-cache"
                headers[.init("X-Tokenhub-Provider")!] = providerName
                return Response(status: .ok, headers: headers, body: responseBody)
            } else {
                // Non-streaming
                let start = ContinuousClock.now
                let (data, urlResponse) = try await URLSession.shared.data(for: urlRequest)
                let elapsed = ContinuousClock.now - start
                let latencyMs = durationToMs(elapsed)
                guard let http = urlResponse as? HTTPURLResponse else {
                    TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: false, latencyMs: latencyMs)
                    if isLB, let next = pickRetryProvider(modelName: req.model, tag: nil, exclude: triedProviders) {
                        lastProvider = next
                        continue
                    }
                    return Response(status: .internalServerError)
                }
                let success = http.statusCode < 400
                TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: success, latencyMs: latencyMs)

                if http.statusCode >= 400 {
                    let body = String(data: data, encoding: .utf8)?.prefix(300) ?? "nil"
                    NovaMLXLog.warning("[Tokenhub/Responses] \(lastProvider.name) error HTTP \(http.statusCode): \(body)")
                    if isLB, let next = pickRetryProvider(modelName: req.model, tag: nil, exclude: triedProviders) {
                        lastProvider = next
                        continue
                    }
                    var headers: HTTPFields = [.contentType: "application/json"]
                    headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                    return Response(status: .init(integerLiteral: http.statusCode), headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
                }

                if let convertedData = convertToResponsesResponse(data, model: req.model) {
                    var headers: HTTPFields = [.contentType: "application/json"]
                    headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                    return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: convertedData)))
                } else {
                    var headers: HTTPFields = [.contentType: "application/json"]
                    headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                    return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
                }
            }
        }

        return Response(status: .badGateway)
    }

    // MARK: - Local Responses API Handlers

    static func handleResponsesRequest(
        req: OpenAIResponseRequest, inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        var messages = mapResponsesInput(req)

        // Tokenhub routing
        if TokenhubManager.shared.isTokenhubModel(req.model) {
            let rawDict = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(req)) as? [String: Any]
            return try await Self.handleTokenhubPassthrough(
                modelName: req.model, rawBody: try JSONEncoder().encode(req),
                path: "responses", inference: inference,
                tag: rawDict?["tag"] as? String
            )
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
            enableThinking: req.reasoning != nil
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
            enableThinking: req.reasoning != nil
        )

        let reqTag = request.id.uuidString.prefix(8)
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
                        if token.finishReason != nil {
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
            } catch {
                NovaMLXLog.error("[SSE:\(reqTag)] Responses stream error: \(error)")
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
            stream: false
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
