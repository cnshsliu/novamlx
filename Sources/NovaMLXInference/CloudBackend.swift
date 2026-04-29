import Foundation
import NovaMLXCore
import NovaMLXUtils

// MARK: - Cloud Model Discovery & Inference Proxy

public actor CloudBackend {
    public static let shared = CloudBackend()

    // Fixed endpoint — hardcoded, no user configuration needed
    static let cloudBaseURL = URL(string: "https://chat.baystoneai.com/v1")!

    // Cached model list from remote
    private var cachedModels: [CloudModelInfo] = []
    private var lastFetchTime: Date = .distantPast

    // Refresh interval: 10 minutes
    private let refreshInterval: TimeInterval = 600

    // MARK: - Model Discovery

    /// Check if a model ID is a cloud model (ends with ":cloud")
    public static func isCloudModel(_ modelId: String) -> Bool {
        modelId.hasSuffix(":cloud")
    }

    /// Strip the ":cloud" suffix to get the remote model name
    public static func stripCloudSuffix(_ modelId: String) -> String {
        modelId.hasSuffix(":cloud")
            ? String(modelId.dropLast(6))
            : modelId
    }

    /// Fetch available models from remote endpoint. Returns models with ":cloud" suffix added.
    public func fetchModels() async -> [CloudModelInfo] {
        // Return cache if fresh
        if !cachedModels.isEmpty, Date().timeIntervalSince(lastFetchTime) < refreshInterval {
            return cachedModels
        }

        let url = Self.cloudBaseURL.appendingPathComponent("models")
        var request = URLRequest(url: url)
        request.timeoutInterval = 10

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                NovaMLXLog.error("Cloud model discovery failed: HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
                return cachedModels
            }

            struct ModelsResponse: Decodable {
                let data: [ModelEntry]
            }
            struct ModelEntry: Decodable {
                let id: String
            }

            let decoded = try JSONDecoder().decode(ModelsResponse.self, from: data)
            cachedModels = decoded.data.map { CloudModelInfo(remoteId: $0.id) }
            lastFetchTime = Date()
            NovaMLXLog.info("Cloud: discovered \(cachedModels.count) remote models")
            return cachedModels
        } catch {
            NovaMLXLog.error("Cloud model discovery error: \(error.localizedDescription)")
            return cachedModels
        }
    }

    /// Get cached model list (local names with ":cloud" suffix)
    public func getModels() -> [String] {
        cachedModels.map(\.localName)
    }

    // MARK: - Subscription Validation

    private func validateAccess() async throws {
        do {
            _ = try await CloudAuth.validate()
        } catch {
            NovaMLXLog.error("[Cloud] Auth failed: \(error.localizedDescription)")
            throw error
        }
    }

    // MARK: - OpenAI Proxy (Non-streaming)

    public func proxy(_ request: InferenceRequest) async throws -> InferenceResult {
        try await validateAccess()
        let remoteModel = Self.stripCloudSuffix(request.model)
        let startTime = Date()

        var urlRequest = URLRequest(url: Self.cloudBaseURL.appendingPathComponent("chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.timeoutInterval = 120

        let body = buildOpenAIBody(request: request, remoteModel: remoteModel, stream: false)
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        guard let http = response as? HTTPURLResponse else {
            throw CloudError.invalidResponse
        }
        guard http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CloudError.remoteError(http.statusCode, body)
        }

        return try parseOpenAIResponse(data: data, request: request, startTime: startTime)
    }

    // MARK: - OpenAI Proxy (Streaming)

    public func proxyStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let remoteModel = Self.stripCloudSuffix(request.model)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Validate auth before streaming
                    _ = try await CloudAuth.validate()

                    var urlRequest = URLRequest(url: Self.cloudBaseURL.appendingPathComponent("chat/completions"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.timeoutInterval = 120

                    let body = buildOpenAIBody(request: request, remoteModel: remoteModel, stream: true)
                    urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: urlRequest)
                    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                        throw CloudError.remoteError(statusCode, "Stream request failed")
                    }

                    var tokenIndex = 0
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let json = String(line.dropFirst(6))
                        if json == "[DONE]" { break }

                        if let tokens = parseOpenAISSEChunk(json, tokenIndex: &tokenIndex) {
                            for token in tokens {
                                continuation.yield(token)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Anthropic Proxy (Non-streaming)

    public func proxyAnthropic(_ request: InferenceRequest) async throws -> InferenceResult {
        try await validateAccess()
        let remoteModel = Self.stripCloudSuffix(request.model)
        let startTime = Date()

        var urlRequest = URLRequest(url: Self.cloudBaseURL.appendingPathComponent("messages"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        urlRequest.timeoutInterval = 120

        let body = buildAnthropicBody(request: request, remoteModel: remoteModel, stream: false)
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CloudError.remoteError(statusCode, body)
        }

        return try parseAnthropicResponse(data: data, request: request, startTime: startTime)
    }

    // MARK: - Anthropic Proxy (Streaming)

    public func proxyAnthropicStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let remoteModel = Self.stripCloudSuffix(request.model)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Validate auth before streaming
                    _ = try await CloudAuth.validate()

                    var urlRequest = URLRequest(url: Self.cloudBaseURL.appendingPathComponent("messages"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
                    urlRequest.timeoutInterval = 120

                    let body = buildAnthropicBody(request: request, remoteModel: remoteModel, stream: true)
                    urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: urlRequest)
                    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                        throw CloudError.remoteError(statusCode, "Stream request failed")
                    }

                    var tokenIndex = 0
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let json = String(line.dropFirst(6))

                        if let tokens = parseAnthropicSSEChunk(json, tokenIndex: &tokenIndex) {
                            for token in tokens {
                                continuation.yield(token)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Health Check

    public func healthCheck() async -> Bool {
        let url = Self.cloudBaseURL.appendingPathComponent("models")
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Static Stream Helper (for non-async contexts)

    /// Static proxyStream that can be called from synchronous contexts (like InferenceService.stream())
    public static func proxyStreamStatic(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let remoteModel = stripCloudSuffix(request.model)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Validate auth before streaming
                    _ = try await CloudAuth.validate()

                    var urlRequest = URLRequest(url: cloudBaseURL.appendingPathComponent("chat/completions"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.timeoutInterval = 120

                    var body: [String: Any] = ["model": remoteModel, "stream": true]
                    var messages: [[String: Any]] = []
                    for msg in request.messages {
                        var m: [String: Any] = ["role": msg.role.rawValue]
                        if let content = msg.content { m["content"] = content }
                        messages.append(m)
                    }
                    body["messages"] = messages
                    if let temp = request.temperature { body["temperature"] = temp }
                    if let maxTokens = request.maxTokens { body["max_tokens"] = maxTokens }
                    if let topP = request.topP { body["top_p"] = topP }
                    body["stream_options"] = ["include_usage": true]

                    urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (bytes, response) = try await URLSession.shared.bytes(for: urlRequest)
                    guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                        throw CloudError.remoteError(statusCode, "Stream request failed")
                    }

                    var tokenIndex = 0
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        let json = String(line.dropFirst(6))
                        if json == "[DONE]" { break }

                        if let tokens = parseSSEChunkStatic(json, tokenIndex: &tokenIndex) {
                            for token in tokens {
                                continuation.yield(token)
                            }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private static func parseSSEChunkStatic(_ json: String, tokenIndex: inout Int) -> [Token]? {
        guard let data = json.data(using: .utf8),
              let chunk = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = chunk["choices"] as? [[String: Any]],
              let first = choices.first
        else { return nil }

        let delta = first["delta"] as? [String: Any]
        let content = delta?["content"] as? String
        let reasoning = delta?["reasoning"] as? String
        let finishStr = first["finish_reason"] as? String

        var tokens: [Token] = []

        if let reasoning, !reasoning.isEmpty {
            tokens.append(Token(id: tokenIndex, text: reasoning))
            tokenIndex += 1
        }

        if let content, !content.isEmpty {
            tokens.append(Token(id: tokenIndex, text: content))
            tokenIndex += 1
        }

        if let finishStr, finishStr != "null" {
            let reason: FinishReason = finishStr == "length" ? .length : .stop
            tokens.append(Token(id: tokenIndex, text: "", finishReason: reason))
            tokenIndex += 1
        }

        return tokens.isEmpty ? nil : tokens
    }

    // MARK: - Private: Build Request Bodies

    private func buildOpenAIBody(request: InferenceRequest, remoteModel: String, stream: Bool) -> [String: Any] {
        var body: [String: Any] = [
            "model": remoteModel,
            "stream": stream,
        ]

        var messages: [[String: Any]] = []
        for msg in request.messages {
            var m: [String: Any] = ["role": msg.role.rawValue]
            if let content = msg.content { m["content"] = content }
            messages.append(m)
        }
        body["messages"] = messages

        if let temp = request.temperature { body["temperature"] = temp }
        if let maxTokens = request.maxTokens { body["max_tokens"] = maxTokens }
        if let topP = request.topP { body["top_p"] = topP }
        if let topK = request.topK { body["top_k"] = topK }
        if let freqPenalty = request.frequencyPenalty { body["frequency_penalty"] = freqPenalty }
        if let presPenalty = request.presencePenalty { body["presence_penalty"] = presPenalty }
        if let seed = request.seed { body["seed"] = seed }
        if let stop = request.stop, !stop.isEmpty { body["stop"] = stop }

        if stream {
            body["stream_options"] = ["include_usage": true]
        }

        return body
    }

    private func buildAnthropicBody(request: InferenceRequest, remoteModel: String, stream: Bool) -> [String: Any] {
        var body: [String: Any] = [
            "model": remoteModel,
            "max_tokens": request.maxTokens ?? 4096,
            "stream": stream,
        ]

        var messages: [[String: Any]] = []
        for msg in request.messages {
            if msg.role == .system {
                body["system"] = msg.content ?? ""
            } else {
                var m: [String: Any] = ["role": msg.role.rawValue]
                if let content = msg.content { m["content"] = content }
                messages.append(m)
            }
        }
        body["messages"] = messages

        if let temp = request.temperature { body["temperature"] = temp }
        if let topP = request.topP { body["top_p"] = topP }
        if let topK = request.topK { body["top_k"] = topK }
        if let stop = request.stop, !stop.isEmpty { body["stop_sequences"] = stop }

        return body
    }

    // MARK: - Private: Parse Responses

    private func parseOpenAIResponse(data: Data, request: InferenceRequest, startTime: Date) throws -> InferenceResult {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any]
        else {
            throw CloudError.parseError("Invalid OpenAI response format")
        }

        // Content may be null when model uses thinking (reasoning) and hits token limit
        let content = message["content"] as? String ?? ""
        let reasoning = message["reasoning"] as? String ?? ""
        let text = reasoning.isEmpty ? content : (content.isEmpty ? reasoning : reasoning + "\n\n" + content)

        let finishStr = first["finish_reason"] as? String ?? "stop"
        let finishReason: FinishReason = finishStr == "length" ? .length : .stop

        let usage = json["usage"] as? [String: Any]
        let promptTokens = usage?["prompt_tokens"] as? Int ?? 0
        let completionTokens = usage?["completion_tokens"] as? Int ?? 0
        let elapsed = Date().timeIntervalSince(startTime)
        let tps = elapsed > 0 && completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        return InferenceResult(
            id: request.id,
            model: request.model,
            text: text,
            tokensPerSecond: tps,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            finishReason: finishReason
        )
    }

    private func parseAnthropicResponse(data: Data, request: InferenceRequest, startTime: Date) throws -> InferenceResult {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = json["content"] as? [[String: Any]],
              let first = content.first,
              let text = first["text"] as? String
        else {
            throw CloudError.parseError("Invalid Anthropic response format")
        }

        let stopReason = json["stop_reason"] as? String ?? "end_turn"
        let finishReason: FinishReason = stopReason == "max_tokens" ? .length : .stop

        let usage = json["usage"] as? [String: Any]
        let promptTokens = usage?["input_tokens"] as? Int ?? 0
        let completionTokens = usage?["output_tokens"] as? Int ?? 0
        let elapsed = Date().timeIntervalSince(startTime)
        let tps = elapsed > 0 && completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        return InferenceResult(
            id: request.id,
            model: request.model,
            text: text,
            tokensPerSecond: tps,
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            finishReason: finishReason
        )
    }

    // MARK: - Private: Parse SSE Chunks

    private func parseOpenAISSEChunk(_ json: String, tokenIndex: inout Int) -> [Token]? {
        guard let data = json.data(using: .utf8),
              let chunk = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = chunk["choices"] as? [[String: Any]],
              let first = choices.first
        else { return nil }

        let delta = first["delta"] as? [String: Any]
        let content = delta?["content"] as? String
        let reasoning = delta?["reasoning"] as? String
        let finishStr = first["finish_reason"] as? String

        var tokens: [Token] = []

        if let reasoning, !reasoning.isEmpty {
            tokens.append(Token(id: tokenIndex, text: reasoning))
            tokenIndex += 1
        }

        if let content, !content.isEmpty {
            tokens.append(Token(id: tokenIndex, text: content))
            tokenIndex += 1
        }

        if let finishStr, finishStr != "null" {
            let reason: FinishReason = finishStr == "length" ? .length : .stop
            let token = Token(id: tokenIndex, text: "", logprob: nil, finishReason: reason)
            tokens.append(token)
            tokenIndex += 1
        }

        return tokens.isEmpty ? nil : tokens
    }

    private func parseAnthropicSSEChunk(_ json: String, tokenIndex: inout Int) -> [Token]? {
        guard let data = json.data(using: .utf8),
              let chunk = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = chunk["type"] as? String
        else { return nil }

        var tokens: [Token] = []

        switch type {
        case "content_block_delta":
            if let delta = chunk["delta"] as? [String: Any],
               let text = delta["text"] as? String, !text.isEmpty {
                tokens.append(Token(id: tokenIndex, text: text))
                tokenIndex += 1
            }

        case "message_delta":
            if let delta = chunk["delta"] as? [String: Any],
               let stopReason = delta["stop_reason"] as? String {
                let reason: FinishReason = stopReason == "max_tokens" ? .length : .stop
                tokens.append(Token(id: tokenIndex, text: "", finishReason: reason))
                tokenIndex += 1
            }

        default:
            break
        }

        return tokens.isEmpty ? nil : tokens
    }
}

// MARK: - Types

public struct CloudModelInfo: Sendable {
    /// Remote model ID (e.g. "Qwen3-235B-4bit")
    public let remoteId: String
    /// Local display name with ":cloud" suffix (e.g. "Qwen3-235B-4bit:cloud")
    public let localName: String

    public init(remoteId: String) {
        self.remoteId = remoteId
        self.localName = "\(remoteId):cloud"
    }
}

enum CloudError: LocalizedError {
    case invalidResponse
    case remoteError(Int, String)
    case parseError(String)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Cloud: invalid response from remote server"
        case .remoteError(let code, let body):
            return "Cloud: remote error \(code) — \(body.prefix(200))"
        case .parseError(let msg):
            return "Cloud: parse error — \(msg)"
        }
    }
}
