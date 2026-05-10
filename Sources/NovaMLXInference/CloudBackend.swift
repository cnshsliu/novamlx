import Foundation
import NovaMLXCore
import NovaMLXUtils

// MARK: - Cloud Model Discovery & Tokenhub Inference Proxy

public actor CloudBackend {
    public static let shared = CloudBackend()

    static let cloudBaseURL = URL(string: "https://chat.baystoneai.com/v1")!

    private var cachedModels: [CloudModelInfo] = []
    private var lastFetchTime: Date = .distantPast
    private let refreshInterval: TimeInterval = 600

    // MARK: - Model Discovery

    /// Fetch available models from remote endpoint for auto-provisioning.
    public func fetchModels() async -> [CloudModelInfo] {
        if !cachedModels.isEmpty, Date().timeIntervalSince(lastFetchTime) < refreshInterval {
            return cachedModels
        }

        let url = Self.cloudBaseURL.appendingPathComponent("models")
        var request = URLRequest(url: url)
        request.timeoutInterval = 10

        // Use session token for auth
        if let session = AuthCache.loadSession(), !session.isEmpty {
            request.setValue("Bearer \(session)", forHTTPHeaderField: "Authorization")
        }

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

    // MARK: - Health Check (Cloud)

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

    // MARK: - Tokenhub Provider Proxy (OpenAI, Non-streaming)

    public func proxy(_ request: InferenceRequest, provider: TokenhubProvider) async throws -> InferenceResult {
        let remoteModel = provider.remoteModel
        let startTime = Date()
        let baseURL = try Self.validatedURL(provider.endpoint)

        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !provider.apiKey.isEmpty {
            urlRequest.setValue("Bearer \(provider.apiKey)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.timeoutInterval = 120

        let body = buildOpenAIBody(request: request, remoteModel: remoteModel, stream: false)
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        guard let http = response as? HTTPURLResponse else { throw CloudError.invalidResponse }
        guard http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CloudError.remoteError(http.statusCode, body)
        }

        return try parseOpenAIResponse(data: data, request: request, startTime: startTime)
    }

    // MARK: - Tokenhub Provider Proxy (OpenAI, Streaming)

    public func proxyStream(_ request: InferenceRequest, provider: TokenhubProvider) -> AsyncThrowingStream<Token, Error> {
        let remoteModel = provider.remoteModel
        let endpoint = provider.endpoint
        let apiKey = provider.apiKey

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let baseURL = try Self.validatedURL(endpoint)
                    var urlRequest = URLRequest(url: baseURL.appendingPathComponent("chat/completions"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    if !apiKey.isEmpty {
                        urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                    }
                    urlRequest.timeoutInterval = 120

                    let body = Self.buildOpenAIBodyStatic(request: request, remoteModel: remoteModel, stream: true)
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
                            for token in tokens { continuation.yield(token) }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Tokenhub Provider Proxy (Anthropic, Non-streaming)

    public func proxyAnthropic(_ request: InferenceRequest, provider: TokenhubProvider) async throws -> InferenceResult {
        let remoteModel = provider.remoteModel
        let startTime = Date()
        let baseURL = try Self.validatedURL(provider.endpoint)

        var urlRequest = URLRequest(url: baseURL.appendingPathComponent("messages"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        if !provider.apiKey.isEmpty {
            urlRequest.setValue("Bearer \(provider.apiKey)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.timeoutInterval = 120

        let body = buildAnthropicBody(request: request, remoteModel: remoteModel, stream: false)
        urlRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        guard let http = response as? HTTPURLResponse else { throw CloudError.invalidResponse }
        guard http.statusCode == 200 else {
            let body = String(data: data, encoding: .utf8) ?? "unknown"
            throw CloudError.remoteError(http.statusCode, body)
        }

        return try parseAnthropicResponse(data: data, request: request, startTime: startTime)
    }

    // MARK: - Tokenhub Provider Proxy (Anthropic, Streaming)

    public func proxyAnthropicStream(_ request: InferenceRequest, provider: TokenhubProvider) -> AsyncThrowingStream<Token, Error> {
        let remoteModel = provider.remoteModel
        let endpoint = provider.endpoint
        let apiKey = provider.apiKey

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let baseURL = try Self.validatedURL(endpoint)
                    var urlRequest = URLRequest(url: baseURL.appendingPathComponent("messages"))
                    urlRequest.httpMethod = "POST"
                    urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
                    if !apiKey.isEmpty {
                        urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
                    }
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
                            for token in tokens { continuation.yield(token) }
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Health Check (Provider)

    public func healthCheck(provider: TokenhubProvider) async -> Bool {
        guard let baseURL = URL(string: provider.endpoint) else { return false }
        var request = URLRequest(url: baseURL.appendingPathComponent("models"))
        request.timeoutInterval = 10
        if !provider.apiKey.isEmpty {
            request.setValue("Bearer \(provider.apiKey)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Private Helpers

    private static func validatedURL(_ endpoint: String) throws -> URL {
        guard let url = URL(string: endpoint) else {
            throw CloudError.remoteError(-1, "Invalid provider endpoint: \(endpoint)")
        }
        return url
    }

    private static func buildOpenAIBodyStatic(request: InferenceRequest, remoteModel: String, stream: Bool) -> [String: Any] {
        var body: [String: Any] = ["model": remoteModel, "stream": stream]
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
        if stream { body["stream_options"] = ["include_usage": true] }
        return body
    }

    private func buildOpenAIBody(request: InferenceRequest, remoteModel: String, stream: Bool) -> [String: Any] {
        var body: [String: Any] = ["model": remoteModel, "stream": stream]
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
        if stream { body["stream_options"] = ["include_usage": true] }
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
            tokens.append(Token(id: tokenIndex, text: "", logprob: nil, finishReason: reason))
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
    public let remoteId: String

    public init(remoteId: String) {
        self.remoteId = remoteId
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
