import Foundation
import NovaMLXCore
import NovaMLXUtils

// MARK: - Cloud Model Discovery & Tokenhub Inference Proxy

public actor CloudBackend {
    public static let shared = CloudBackend()

    static let tknetBaseURL = URL(string: "https://api.tknet.ai/v1")!
    static let tknetManagementURL = URL(string: "https://tknet.ai/api/v1")!

    // MARK: - API Key Verification

    /// Verify tknet.ai API Key by fetching nova models.
    /// Returns true if key is valid and returns at least one nova model.
    public func verifySettingsApiKey(apiKey: String) async -> Bool {
        let url = Self.tknetManagementURL.appendingPathComponent("models")
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "tag", value: "nova")]

        var request = URLRequest(url: components.url!)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 10

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else { return false }
            if http.statusCode == 200 {
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let models = json["data"] as? [[String: Any]] {
                    return !models.isEmpty
                }
            }
            return false
        } catch {
            NovaMLXLog.error("tknet.ai API Key verification failed: \(error.localizedDescription)")
            return false
        }
    }

    /// Fetch nova-tagged models from tknet.ai using valid API Key.
    public func fetchTknetModels(apiKey: String) async -> [TknetModel] {
        let url = Self.tknetManagementURL.appendingPathComponent("models")
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "tag", value: "nova")]

        var request = URLRequest(url: components.url!)
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 10

        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            let decoded = try JSONDecoder().decode(TknetModelsResponse.self, from: data)
            NovaMLXLog.info("tknet.ai: discovered \(decoded.data.count) nova models")
            return decoded.data
        } catch {
            NovaMLXLog.error("tknet.ai model fetch error: \(error.localizedDescription)")
            return []
        }
    }

    // MARK: - Tokenhub Provider Proxy (OpenAI, Non-streaming)

    public func proxy(_ request: InferenceRequest, provider: TokenhubProvider) async throws -> InferenceResult {
        let remoteModel = provider.remoteModel
        let apiKey = TokenhubManager.shared.effectiveApiKey(for: provider)
        let startTime = Date()
        let url = try Self.requestURL(provider.endpoint, suffix: "chat/completions")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
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
        let apiKey = TokenhubManager.shared.effectiveApiKey(for: provider)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = try Self.requestURL(endpoint, suffix: "chat/completions")
                    var urlRequest = URLRequest(url: url)
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
        let apiKey = TokenhubManager.shared.effectiveApiKey(for: provider)
        let startTime = Date()
        let url = try Self.requestURL(provider.endpoint, suffix: "messages")

        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        if !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
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
        let apiKey = TokenhubManager.shared.effectiveApiKey(for: provider)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let url = try Self.requestURL(endpoint, suffix: "messages")
                    var urlRequest = URLRequest(url: url)
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
        guard let url = EndpointNormalizer.url(endpoint: provider.endpoint, suffix: "models") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        let apiKey = TokenhubManager.shared.effectiveApiKey(for: provider)
        if !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    // MARK: - Private Helpers

    /// Any-endpoint rule: the user's endpoint may be a base URL, a full
    /// request URL already ending in the suffix, query-stringed, or
    /// scheme-less. The normalizer handles every form.
    private static func requestURL(_ endpoint: String, suffix: String) throws -> URL {
        guard let url = EndpointNormalizer.url(endpoint: endpoint, suffix: suffix) else {
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

private struct TknetModelsResponse: Codable {
    let object: String
    let data: [TknetModel]
}

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
