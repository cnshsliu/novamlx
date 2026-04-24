import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdRouter
import Logging
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXMCP
import NovaMLXModelManager
import NovaMLXUtils

typealias AppContext = BasicRouterRequestContext

private final class LockedCounter: @unchecked Sendable {
    private var value = 0
    private let lock = NSLock()
    func increment() -> Int {
        lock.lock()
        defer { lock.unlock() }
        let v = value
        value += 1
        return v
    }
}

private func unwrapAnyCodable(_ ac: AnyCodable) -> Any {
    switch ac {
    case .string(let s): return s
    case .int(let i): return i
    case .double(let d): return d
    case .bool(let b): return b
    case .null: return NSNull()
    case .array(let a): return a.map { unwrapAnyCodable($0) }
    case .dictionary(let d): return d.mapValues { unwrapAnyCodable($0) }
    }
}

private func anyToAnyCodable(_ value: Any) -> AnyCodable {
    switch value {
    case let s as String: return .string(s)
    case let i as Int: return .int(i)
    case let d as Double: return .double(d)
    case let b as Bool: return .bool(b)
    case let a as [Any]: return .array(a.map { anyToAnyCodable($0) })
    case let d as [String: Any]: return .dictionary(d.mapValues { anyToAnyCodable($0) })
    default: return .string("\(value)")
    }
}

struct NovaMLXErrorMiddleware: RouterMiddleware {
    typealias Context = AppContext

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        do {
            return try await next(request, context)
        } catch let error as NovaMLXError {
            let detail = OpenAIErrorDetail(
                message: error.errorDescription ?? "Unknown error",
                type: error.apiErrorType,
                code: error.apiErrorCode
            )
            return Self.jsonError(status: error.httpStatus, detail: detail)
        } catch let error as DecodingError {
            let detail = OpenAIErrorDetail(
                message: Self.decodingErrorMessage(error),
                type: "invalid_request_error",
                code: "invalid_json"
            )
            return Self.jsonError(status: .badRequest, detail: detail)
        } catch {
            let detail = OpenAIErrorDetail(
                message: error.localizedDescription,
                type: "internal_error",
                code: "internal_error"
            )
            return Self.jsonError(status: .internalServerError, detail: detail)
        }
    }

    static func jsonError(
        status: HTTPResponse.Status,
        detail: OpenAIErrorDetail
    ) -> Response {
        let body = OpenAIErrorResponse(error: detail)
        guard let data = try? JSONEncoder().encode(body) else {
            return Response(status: status, headers: [.contentType: "application/json"])
        }
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    private static func decodingErrorMessage(_ error: DecodingError) -> String {
        switch error {
        case .typeMismatch(let type, let ctx):
            let path = ctx.codingPath.map(\.stringValue).joined(separator: ".")
            return "Type mismatch for \(type) at \(path): \(ctx.debugDescription)"
        case .valueNotFound(let type, let ctx):
            let path = ctx.codingPath.map(\.stringValue).joined(separator: ".")
            return "Missing value for \(type) at \(path): \(ctx.debugDescription)"
        case .keyNotFound(let key, let ctx):
            let path = ctx.codingPath.map(\.stringValue).joined(separator: ".")
            return "Missing required field '\(key.stringValue)'" + (path.isEmpty ? "" : " at \(path)")
        case .dataCorrupted(let ctx):
            let path = ctx.codingPath.map(\.stringValue).joined(separator: ".")
            return "Corrupted data" + (path.isEmpty ? "" : " at \(path)") + ": \(ctx.debugDescription)"
        @unknown default:
            return "Invalid request body"
        }
    }
}

private struct AdminAuthMiddleware: RouterMiddleware {
    typealias Context = AppContext

    let validKeys: [String]

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        if validKeys.isEmpty {
            let detail = OpenAIErrorDetail(
                message: "Admin API is disabled. Set apiKeys in ServerConfig to enable.",
                type: "forbidden_error",
                code: "admin_disabled"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .forbidden, detail: detail)
        }

        let authHeader = request.headers[.authorization]
        let token: String?
        if let authHeader, authHeader.hasPrefix("Bearer ") {
            token = String(authHeader.dropFirst(7))
        } else {
            token = request.headers[fields: HTTPField.Name("x-admin-key")!].first?.value
        }

        guard let token, validKeys.contains(token) else {
            let detail = OpenAIErrorDetail(
                message: "Invalid or missing admin API key.",
                type: "authentication_error",
                code: "invalid_api_key"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .unauthorized, detail: detail)
        }

        return try await next(request, context)
    }
}

private struct APIKeyAuthMiddleware: RouterMiddleware {
    typealias Context = AppContext

    let validKeys: [String]

    private static let publicPaths: Set<String> = ["/chat", "/health", "/v1/models"]
    private static let publicPrefixes: Set<String> = ["/v1/chat/history"]

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        if validKeys.isEmpty { return try await next(request, context) }

        let path = request.uri.path
        if Self.publicPaths.contains(path) || Self.publicPrefixes.contains(where: { path.hasPrefix($0) }) {
            return try await next(request, context)
        }

        // 1. Authorization: Bearer xxx (OpenAI / standard)
        let authHeader = request.headers[.authorization]
        let token: String?
        if let authHeader, authHeader.hasPrefix("Bearer ") {
            token = String(authHeader.dropFirst(7))
        } else if let xApiKey = request.headers[HTTPField.Name("x-api-key")!] {
            // 2. x-api-key header (Anthropic standard)
            token = xApiKey
        } else {
            token = nil
        }

        guard let token, validKeys.contains(token) else {
            let detail = OpenAIErrorDetail(
                message: "Invalid or missing API key.",
                type: "authentication_error",
                code: "invalid_api_key"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .unauthorized, detail: detail)
        }

        return try await next(request, context)
    }
}

private struct CORSMiddleware: RouterMiddleware {
    typealias Context = AppContext

    let allowedOrigins: String

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        if request.method == .options {
            var headers = HTTPFields()
            headers[.accessControlAllowOrigin] = allowedOrigins
            headers[HTTPField.Name("Access-Control-Allow-Methods")!] = "GET, POST, PUT, DELETE, OPTIONS"
            headers[HTTPField.Name("Access-Control-Allow-Headers")!] = "Content-Type, Authorization, X-Admin-Key, X-Api-Key, Anthropic-Version"
            headers[HTTPField.Name("Access-Control-Max-Age")!] = "86400"
            return Response(status: .noContent, headers: headers)
        }

        var response = try await next(request, context)
        response.headers[.accessControlAllowOrigin] = allowedOrigins
        return response
    }
}

private struct RequestIDMiddleware: RouterMiddleware {
    typealias Context = AppContext
    private static let requestIDHeader = HTTPField.Name("x-request-id")!

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let requestID: String
        if let hdr = request.headers[fields: Self.requestIDHeader].first?.value, !hdr.isEmpty {
            requestID = hdr
        } else {
            requestID = UUID().uuidString.prefix(8).description
        }
        var response = try await next(request, context)
        response.headers[Self.requestIDHeader] = requestID
        return response
    }
}

extension NovaMLXError {
    var httpStatus: HTTPResponse.Status {
        switch self {
        case .modelNotFound: .notFound
        case .modelLoadFailed: .serviceUnavailable
        case .inferenceFailed: .internalServerError
        case .configurationError: .internalServerError
        case .cacheError: .internalServerError
        case .apiError: .badRequest
        case .downloadFailed: .badGateway
        case .unsupportedModel: .badRequest
        case .contextWindowExceeded: .badRequest
        }
    }

    var apiErrorType: String {
        switch self {
        case .modelNotFound: "not_found_error"
        case .modelLoadFailed: "server_error"
        case .inferenceFailed: "server_error"
        case .configurationError: "server_error"
        case .cacheError: "server_error"
        case .apiError: "invalid_request_error"
        case .downloadFailed: "server_error"
        case .unsupportedModel: "invalid_request_error"
        case .contextWindowExceeded: "invalid_request_error"
        }
    }

    var apiErrorCode: String {
        switch self {
        case .modelNotFound: "model_not_found"
        case .modelLoadFailed: "model_load_failed"
        case .inferenceFailed: "inference_error"
        case .configurationError: "configuration_error"
        case .cacheError: "cache_error"
        case .apiError: "api_error"
        case .downloadFailed: "download_failed"
        case .unsupportedModel: "unsupported_model"
        case .contextWindowExceeded: "context_window_exceeded"
        }
    }
}

public final class NovaMLXAPIServer: @unchecked Sendable {
    private let inferenceService: InferenceService
    private let modelManager: ModelManager
    private let embeddingService: EmbeddingService
    private let rerankerService: RerankerService
    private let mcpManager: MCPManager
    private let benchmarkService: BenchmarkService
    private let perplexityService: PerplexityService
    private let updateChecker: UpdateChecker
    private let hfService: HuggingFaceService
    private let audioService: AudioService
    private let config: ServerConfig

    public init(
        inferenceService: InferenceService,
        modelManager: ModelManager,
        embeddingService: EmbeddingService = EmbeddingService(),
        rerankerService: RerankerService = RerankerService(),
        mcpManager: MCPManager = MCPManager(),
        config: ServerConfig = ServerConfig()
    ) {
        self.inferenceService = inferenceService
        self.modelManager = modelManager
        self.embeddingService = embeddingService
        self.rerankerService = rerankerService
        self.mcpManager = mcpManager
        self.benchmarkService = BenchmarkService(inferenceService: inferenceService)
        self.perplexityService = PerplexityService(inferenceService: inferenceService)
        self.updateChecker = UpdateChecker()
        self.hfService = HuggingFaceService(modelDirectory: modelManager.modelsDirectory)
        self.audioService = AudioService()
        self.config = config
        // When HF download completes, re-run discovery so model appears in registry
        self.hfService.onModelDownloaded = { [weak self] repoId in
            NovaMLXLog.info("[HF] Download completed for \(repoId), running model discovery...")
            self?.modelManager.discoverModels()
        }
    }

    public func start() async throws {
        let inference = self.inferenceService
        let models = self.modelManager
        let embeddings = self.embeddingService
        let reranker = self.rerankerService
        let mcp = self.mcpManager
        let benchmark = self.benchmarkService
        let perplexity = self.perplexityService
        let updater = self.updateChecker
        let cfg = self.config
        let hf = self.hfService
        let audio = self.audioService

        let rateLimiter = RateLimiter(config: RateLimitConfig())
        let securityHeaders = SecurityHeadersMiddleware()
        let requestSizeLimit = RequestSizeLimitMiddleware(maxMB: cfg.maxRequestSizeMB)
        let rateLimitMiddleware = RateLimitMiddleware.perAPIKey(limiter: rateLimiter)

        let mainRouter = RouterBuilder(context: AppContext.self) {
            CORSMiddleware(allowedOrigins: "*")
            RequestIDMiddleware()
            securityHeaders
            requestSizeLimit
            rateLimitMiddleware
            APIKeyAuthMiddleware(validKeys: cfg.apiKeys)
            NovaMLXErrorMiddleware()
            Get("/v1/models") { request, context in
                let modelList = models.downloadedModels()
                    .filter { inference.isModelLoaded($0.id) }
                    .map { OpenAIModel(id: $0.id) }

                // Append cloud models
                let cloudModelList = await inference.listCloudModels().map { OpenAIModel(id: $0) }
                let response = OpenAIModelsResponse(data: modelList + cloudModelList)
                return try Self.jsonResponse(response)
            }
            Post("/v1/chat/completions") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let openAIReq = try JSONDecoder().decode(OpenAIRequest.self, from: body)

                let messages = openAIReq.messages.map { msg in
                    let role: ChatMessage.Role = switch msg.role {
                    case "system": .system
                    case "assistant": .assistant
                    case "tool": .tool
                    default: .user
                    }
                    let imageURLs = msg.content?.imageParts.compactMap { part -> String? in
                        if case .imageUrl(let img) = part { return img.url } else { return nil }
                    }
                    return ChatMessage(role: role, content: msg.content?.textValue, images: (imageURLs?.isEmpty ?? true) ? nil : imageURLs)
                }

                if !inference.isModelLoaded(openAIReq.model) {
                    throw NovaMLXError.modelNotFound(openAIReq.model)
                }

                let sessionId = Self.extractSessionId(request: request, body: openAIReq.sessionId)
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
                        regexPattern: regexPattern, gbnfGrammar: gbnfGrammar
                    )
                } else {
                    return try await Self.handleChat(
                        openAIReq: openAIReq, messages: messages, inference: inference,
                        sessionId: sessionId, responseFormat: responseFormat, jsonSchemaDef: jsonSchemaDef,
                        regexPattern: regexPattern, gbnfGrammar: gbnfGrammar
                    )
                }
            }
            Post("/v1/messages") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let anthropicReq = try JSONDecoder().decode(AnthropicRequest.self, from: body)

                NovaMLXLog.info("[API] POST /v1/messages — model=\(anthropicReq.model), stream=\(anthropicReq.stream ?? false), maxTokens=\(anthropicReq.maxTokens), msgs=\(anthropicReq.messages.count)")

                var messages = anthropicReq.messages.map { msg in
                    let role: ChatMessage.Role = msg.role == "assistant" ? .assistant : .user
                    return ChatMessage(role: role, content: msg.textContent)
                }
                if let system = anthropicReq.system {
                    messages.insert(ChatMessage(role: .system, content: system.textContent), at: 0)
                }

                if !inference.isModelLoaded(anthropicReq.model) {
                    throw NovaMLXError.modelNotFound(anthropicReq.model)
                }

                if anthropicReq.stream ?? false {
                    return try await Self.handleStreamAnthropic(
                        anthropicReq: anthropicReq, messages: messages, inference: inference
                    )
                }

                let request = InferenceRequest(
                    model: anthropicReq.model, messages: messages,
                    tools: anthropicReq.tools?.map { tool in
                        var dict: [String: Any] = ["name": tool.name]
                        if let desc = tool.description { dict["description"] = desc }
                        dict["type"] = "function"
                        dict["function"] = [
                            "name": tool.name,
                            "description": tool.description ?? "",
                            "parameters": unwrapAnyCodable(tool.inputSchema)
                        ] as [String: Any]
                        return dict
                    },
                    temperature: anthropicReq.temperature, maxTokens: anthropicReq.maxTokens,
                    topP: anthropicReq.topP, stream: false, stop: anthropicReq.stopSequences
                )

                let result = try await inference.generate(request)
                let scaledInput = cfg.scaleTokenCount(result.promptTokens, modelContextWindow: inference.getContextWindow(for: anthropicReq.model) ?? 0)
                let scaledOutput = cfg.scaleTokenCount(result.completionTokens, modelContextWindow: inference.getContextWindow(for: anthropicReq.model) ?? 0)

                var content: [AnthropicContentBlock] = []
                if !result.text.isEmpty {
                    content.append(AnthropicContentBlock(text: result.text))
                }
                if let toolCalls = result.toolCalls {
                    for tc in toolCalls {
                        let inputCodable: AnyCodable
                        if let data = tc.arguments.data(using: .utf8),
                           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                            inputCodable = .dictionary(obj.mapValues { anyToAnyCodable($0) })
                        } else {
                            inputCodable = .dictionary([:])
                        }
                        content.append(AnthropicContentBlock(
                            type: "tool_use",
                            id: tc.id,
                            name: tc.functionName,
                            input: inputCodable
                        ))
                    }
                }
                if content.isEmpty { content.append(AnthropicContentBlock(text: "")) }

                let response = AnthropicResponse(
                    id: result.id.uuidString, model: result.model,
                    content: content,
                    stopReason: (result.toolCalls != nil && !result.toolCalls!.isEmpty) ? "tool_use" : result.finishReason.rawValue,
                    usage: AnthropicUsage(inputTokens: scaledInput, outputTokens: scaledOutput)
                )
                return try Self.jsonResponse(response)
            }
            Post("/v1/messages/count_tokens") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AnthropicTokenCountRequest.self, from: body)

                if !inference.isModelLoaded(req.model) {
                    throw NovaMLXError.modelNotFound(req.model)
                }

                var messages = req.messages.map { msg in
                    let role: ChatMessage.Role = msg.role == "assistant" ? .assistant : .user
                    return ChatMessage(role: role, content: msg.textContent)
                }
                if let system = req.system {
                    messages.insert(ChatMessage(role: .system, content: system.textContent), at: 0)
                }

                guard let tokenCount = inference.countTokens(model: req.model, messages: messages) else {
                    throw NovaMLXError.inferenceFailed("Failed to count tokens: model tokenizer not available")
                }

                let response = AnthropicTokenCountResponse(inputTokens: tokenCount)
                return try Self.jsonResponse(response)
            }
            Post("/v1/completions") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let compReq = try JSONDecoder().decode(OpenAICompletionRequest.self, from: body)

                if !inference.isModelLoaded(compReq.model) {
                    throw NovaMLXError.modelNotFound(compReq.model)
                }

                if compReq.stream ?? false {
                    return try await Self.handleStreamCompletion(
                        compReq: compReq, inference: inference
                    )
                } else {
                    return try await Self.handleCompletion(
                        compReq: compReq, inference: inference
                    )
                }
            }
            Post("/v1/embeddings") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let embReq = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: body)

                if !embeddings.isLoaded(embReq.model) {
                    throw NovaMLXError.modelNotFound(embReq.model)
                }

                let texts = embReq.input.texts
                let result = try await embeddings.embed(model: embReq.model, inputs: texts)

                let data = result.embeddings.enumerated().map { idx, vec in
                    OpenAIEmbeddingData(index: idx, embedding: vec)
                }
                let response = OpenAIEmbeddingResponse(
                    model: result.model,
                    data: data,
                    usage: OpenAIEmbeddingUsage(
                        promptTokens: result.promptTokens,
                        totalTokens: result.totalTokens
                    )
                )
                return try Self.jsonResponse(response)
            }
            Post("/v1/responses") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(OpenAIResponseRequest.self, from: body)

                if !inference.isModelLoaded(req.model) {
                    throw NovaMLXError.modelNotFound(req.model)
                }

                return try await Self.handleResponsesRequest(req: req, inference: inference)
            }
            Get("/v1/responses/{id}") { request, context in
                let responseId = try context.parameters.require("id")
                let store = ResponseStore.shared
                guard let response = store.get(responseId) else {
                    throw NovaMLXError.modelNotFound("Response \(responseId) not found")
                }
                return try Self.jsonResponse(response)
            }
            Delete("/v1/responses/{id}") { request, context in
                let responseId = try context.parameters.require("id")
                let store = ResponseStore.shared
                guard store.get(responseId) != nil else {
                    throw NovaMLXError.modelNotFound("Response \(responseId) not found")
                }
                store.delete(responseId)
                return Response(status: .ok, body: .init(byteBuffer: ByteBuffer(string: "{\"status\":\"deleted\"}")))
            }
            Post("/v1/audio/transcriptions") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                let data = Data(buffer: body)
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                let language = json?["language"] as? String
                let audioData: Data
                if let file = json?["file"] as? String, let fileData = Data(base64Encoded: file) {
                    audioData = fileData
                } else if let file = json?["audio"] as? String, let fileData = Data(base64Encoded: file) {
                    audioData = fileData
                } else {
                    audioData = data
                }
                let result = try await audio.transcribe(audioData: audioData, language: language)
                return try Self.jsonResponse(result)
            }
            Post("/v1/audio/speech") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] ?? [:]
                let text = json["input"] as? String ?? json["text"] as? String ?? ""
                let voice = json["voice"] as? String
                let speed = json["speed"] as? Double
                let language = json["language"] as? String
                let stream = json["stream"] as? Bool ?? false
                guard !text.isEmpty else {
                    return Response(status: .badRequest, body: .init(byteBuffer: ByteBuffer(string: "{\"error\":\"input text required\"}")))
                }
                let synthesisRequest = AudioService.SynthesisRequest(text: text, voice: voice, speed: speed, language: language)

                if stream {
                    let audioStream = audio.synthesizeStream(request: synthesisRequest)
                    let responseBody: ResponseBody = .init { writer in
                        do {
                            for try await chunk in audioStream {
                                try await writer.write(ByteBuffer(data: chunk.data))
                            }
                        } catch {
                            NovaMLXLog.error("Audio stream error: \(error)")
                        }
                    }
                    var headers = HTTPFields()
                    headers[.contentType] = "audio/wav"
                    return Response(status: .ok, headers: headers, body: responseBody)
                } else {
                    let wavData = try await audio.synthesize(request: synthesisRequest)
                    var headers = HTTPFields()
                    headers[.contentType] = "audio/wav"
                    return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: wavData)))
                }
            }
            Get("/v1/audio/voices") { _, _ in
                let voices = AudioService.supportedVoices()
                return try Self.jsonResponse(["voices": voices])
            }
            Get("/v1/audio/languages") { _, _ in
                let languages = AudioService.supportedLanguages()
                return try Self.jsonResponse(["languages": languages])
            }
            Get("/health") { _, _ in
                let stats = inference.stats
                let mcpStatuses = mcp.getServerStatuses()
                let body: [String: Any] = [
                    "status": "ok",
                    "loadedModels": stats.loadedModels,
                    "gpuMemoryUsed": stats.gpuMemoryUsed,
                    "mcp": [
                        "connectedServers": mcp.connectedServerCount,
                        "totalServers": mcp.totalServerCount,
                        "servers": mcpStatuses.map { ["name": $0.name, "state": $0.state.rawValue, "toolsCount": $0.toolsCount] },
                    ],
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            Get("/chat") { _, _ in
                let html = ChatHTML.render()
                return Response(
                    status: .ok,
                    headers: [.contentType: "text/html"],
                    body: .init(byteBuffer: ByteBuffer(string: html))
                )
            }
            Get("/v1/chat/history") { _, _ in
                let summaries = ChatHistoryStore.shared.list()
                return try Self.jsonResponse(summaries)
            }
            Get("/v1/chat/history/{id}") { _, context in
                let id = try context.parameters.require("id")
                guard let record = ChatHistoryStore.shared.get(id: id) else {
                    throw HTTPError(.notFound, message: "Chat not found")
                }
                return try Self.jsonResponse(record)
            }
            Post("/v1/chat/history") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                let record = try JSONDecoder().decode(ChatHistoryStore.ChatRecord.self, from: body)
                try ChatHistoryStore.shared.save(record)
                return Response(status: .ok)
            }
            Delete("/v1/chat/history/{id}") { _, context in
                let id = try context.parameters.require("id")
                try ChatHistoryStore.shared.delete(id: id)
                return Response(status: .ok)
            }
            Get("/v1/stats") { _, context in
                let stats = inference.stats
                let sessionMetrics = inference.engine.metrics
                let persistentMetrics = inference.engine.metricsStore.metrics
                let batchMetrics = inference.engine.batchScheduler.metrics
                let body: [String: Any] = [
                    "session": [
                        "loadedModels": stats.loadedModels,
                        "activeRequests": stats.activeRequests,
                        "gpuMemoryUsed": stats.gpuMemoryUsed,
                        "totalRequests": sessionMetrics.totalRequests,
                        "totalTokens": sessionMetrics.totalTokensGenerated,
                        "averageTokensPerSecond": sessionMetrics.averageTokensPerSecond,
                    ],
                    "allTime": [
                        "totalRequests": persistentMetrics.totalRequestsAllTime,
                        "totalTokens": persistentMetrics.totalTokensAllTime,
                        "totalInferenceTime": persistentMetrics.totalInferenceTimeAllTime,
                        "averageTokensPerSecond": persistentMetrics.averageTokensPerSecond,
                        "modelsLoaded": persistentMetrics.modelsLoaded,
                        "modelsUnloaded": persistentMetrics.modelsUnloaded,
                        "evictions": persistentMetrics.evictions,
                        "ttlEvictions": persistentMetrics.ttlEvictions,
                        "memoryPressureEvictions": persistentMetrics.memoryPressureEvictions,
                        "cacheHits": persistentMetrics.cacheHits,
                        "cacheMisses": persistentMetrics.cacheMisses,
                        "cacheHitRate": persistentMetrics.cacheHitRate,
                    ],
                    "byModel": persistentMetrics.totalRequestsByModel,
                    "batcher": [
                        "activeRequests": inference.batcherMetrics.activeRequests,
                        "queueDepth": inference.batcherMetrics.queueDepth,
                        "totalQueued": inference.batcherMetrics.totalQueued,
                        "totalCompleted": inference.batcherMetrics.totalCompleted,
                        "totalPreempted": inference.batcherMetrics.totalPreempted,
                        "peakActiveCount": inference.batcherMetrics.peakActiveCount,
                        "averageQueueWaitTime": inference.batcherMetrics.averageQueueWaitTime,
                        "maxBatchSize": inference.batcherMetrics.maxBatchSize,
                    ],
                    "fusedBatch": [
                        "pendingCount": batchMetrics.pendingCount,
                        "activeSequences": batchMetrics.activeSequences,
                        "totalFusedSteps": batchMetrics.totalFusedSteps,
                        "totalTokensViaFused": batchMetrics.totalTokensViaFused,
                        "peakBatchWidth": batchMetrics.peakBatchWidth,
                        "totalBatches": batchMetrics.totalBatches,
                    ],
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            RouteGroup("v1") {
                Post("/rerank") { request, context in
                    let body = try await request.body.collect(upTo: .max)
                    let rerankReq = try JSONDecoder().decode(RerankRequest.self, from: body)

                    if !reranker.isLoaded(rerankReq.model) {
                        throw NovaMLXError.modelNotFound(rerankReq.model)
                    }

                    let docs = rerankReq.documents.map { $0.text }
                    let results = try await reranker.rerank(
                        model: rerankReq.model,
                        query: rerankReq.query,
                        documents: docs,
                        topN: rerankReq.topN
                    )

                    let returnDocs = rerankReq.returnDocuments ?? true
                    let rerankResp = RerankResponse(
                        id: "rerank-\(UUID().uuidString.prefix(8))",
                        results: results.map { r in
                            let doc: RerankDocument? = returnDocs ? rerankReq.documents[r.index] : nil
                            return RerankResult(index: r.index, relevanceScore: r.score, document: doc)
                        },
                        model: rerankReq.model,
                        usage: RerankUsage(totalTokens: results.first?.totalTokens ?? 0)
                    )
                    return try Self.jsonResponse(rerankResp)
                }
                Post("/mcp/execute") { request, context in
                    let body = try await request.body.collect(upTo: .max)
                    guard let json = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                          let toolName = json["tool_name"] as? String else {
                        throw NovaMLXError.apiError("Invalid MCP execute request")
                    }
                    let mcpArgs = (json["arguments"] as? [String: Any]) ?? [:]
                    let mcpResult = try await mcp.executeTool(namespacedName: toolName, arguments: mcpArgs)

                    let respBody: [String: Any] = [
                        "tool_name": toolName,
                        "content": mcpResult.content,
                        "is_error": mcpResult.isError,
                        "error_message": mcpResult.errorMessage as Any
                    ]
                    let respData = try JSONSerialization.data(withJSONObject: respBody)
                    return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: respData)))
                }
                Post("/batch/completions") { request, _ in
                    struct BatchGenerateBody: Codable, Sendable {
                        let model: String
                        let messages: [ChatMessage]
                        let temperature: Double?
                        let maxTokens: Int?
                        let topP: Double?
                        let count: Int?
                    }
                    let body = try await request.body.collect(upTo: .max)
                    let req = try JSONDecoder().decode(BatchGenerateBody.self, from: body)
                    let batchSize = req.count ?? 1
                    let results = try await withThrowingTaskGroup(of: InferenceResult.self) { group in
                        for _ in 0..<batchSize {
                            let inferReq = InferenceRequest(
                                id: UUID(),
                                model: req.model,
                                messages: req.messages,
                                temperature: req.temperature,
                                maxTokens: req.maxTokens,
                                topP: req.topP
                            )
                            group.addTask {
                                try await inference.engine.batchScheduler.submit(inferReq)
                            }
                        }
                        var collected: [InferenceResult] = []
                        for try await result in group {
                            collected.append(result)
                        }
                        return collected
                    }
                    return try Self.jsonResponse(["results": results])
                }
            }
        }

        let adminRouter = RouterBuilder(context: AppContext.self) {
            AdminAuthMiddleware(validKeys: cfg.apiKeys)
            securityHeaders
            requestSizeLimit
            NovaMLXErrorMiddleware()
            Get("/admin/models") { request, context in
                let records = models.allRegisteredModels()
                let statuses = records.map { record -> AdminModelStatus in
                    AdminModelStatus(
                        id: record.id,
                        family: record.family.rawValue,
                        downloaded: models.isDownloaded(record.id),
                        loaded: inference.isModelLoaded(record.id) || embeddings.isLoaded(record.id),
                        sizeBytes: record.sizeBytes,
                        downloadedAt: record.downloadedAt
                    )
                }
                return try Self.jsonResponse(statuses)
            }
            Post("/admin/models/download") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdminDownloadRequest.self, from: body)

                guard models.getRecord(req.modelId) != nil else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                if let status = models.getDownloadStatus(req.modelId), status.state == .downloading {
                    return try Self.jsonResponse(status, httpStatus: .accepted)
                }

                if models.isDownloaded(req.modelId) {
                    guard let record = models.getRecord(req.modelId) else {
                        throw NovaMLXError.modelNotFound(req.modelId)
                    }
                    return try Self.jsonResponse(AdminModelStatus(
                        id: req.modelId, family: record.family.rawValue,
                        downloaded: true, loaded: inference.isModelLoaded(req.modelId),
                        sizeBytes: record.sizeBytes, downloadedAt: record.downloadedAt
                    ))
                }

                try models.startDownload(req.modelId)

                let status = models.getDownloadStatus(req.modelId) ?? DownloadStatus(modelId: req.modelId, state: .downloading)
                return try Self.jsonResponse(status, httpStatus: .accepted)
            }
            Post("/admin/models/status") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdminLoadRequest.self, from: body)

                if let status = models.getDownloadStatus(req.modelId) {
                    return try Self.jsonResponse(status)
                }
                let state: DownloadState = models.isDownloaded(req.modelId) ? .downloaded : .notDownloaded
                return try Self.jsonResponse(DownloadStatus(modelId: req.modelId, state: state))
            }
            Post("/admin/models/load") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdminLoadRequest.self, from: body)

                guard models.isDownloaded(req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                if inference.isModelLoaded(req.modelId) || embeddings.isLoaded(req.modelId) {
                    let record = models.getRecord(req.modelId)
                    return try Self.jsonResponse(AdminModelStatus(
                        id: req.modelId, family: record?.family.rawValue ?? "other",
                        downloaded: true, loaded: true,
                        sizeBytes: record?.sizeBytes ?? 0, downloadedAt: record?.downloadedAt
                    ))
                }

                guard let record = models.getRecord(req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                let config = ModelConfig(
                    identifier: ModelIdentifier(id: req.modelId, family: record.family),
                    modelType: record.modelType
                )

                if record.modelType == .embedding {
                    _ = try await embeddings.loadModel(from: record.localURL, config: config)
                } else {
                    try await inference.loadModel(at: record.localURL, config: config)
                }

                return try Self.jsonResponse(AdminModelStatus(
                    id: req.modelId, family: record.family.rawValue,
                    downloaded: true, loaded: true,
                    sizeBytes: record.sizeBytes, downloadedAt: record.downloadedAt
                ))
            }
            Post("/admin/models/unload") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdminLoadRequest.self, from: body)

                guard inference.isModelLoaded(req.modelId) || embeddings.isLoaded(req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                let record = models.getRecord(req.modelId)

                if inference.isModelLoaded(req.modelId) {
                    await inference.unloadModel(ModelIdentifier(id: req.modelId, family: record?.family ?? .other))
                }

                if embeddings.isLoaded(req.modelId) {
                    embeddings.unloadModel(req.modelId)
                }

                return try Self.jsonResponse(AdminModelStatus(
                    id: req.modelId, family: record?.family.rawValue ?? "other",
                    downloaded: models.isDownloaded(req.modelId), loaded: false,
                    sizeBytes: record?.sizeBytes ?? 0, downloadedAt: record?.downloadedAt
                ))
            }
            Delete("/admin/models/{id}") { request, context in
                let modelId = try context.parameters.require("id")

                if inference.isModelLoaded(modelId) {
                    await inference.unloadModel(ModelIdentifier(id: modelId, family: .other))
                }

                if embeddings.isLoaded(modelId) {
                    embeddings.unloadModel(modelId)
                }

                try models.deleteModel(modelId)

                return Response(status: .noContent)
            }
            Post("/admin/models/discover") { request, context in
                let discovered = models.discoverModels()
                let items = discovered.map { model -> AdminDiscoveredModel in
                    AdminDiscoveredModel(
                        id: model.modelId,
                        type: model.modelType.rawValue,
                        family: model.family.rawValue,
                        path: model.modelPath.path
                    )
                }
                return try Self.jsonResponse(AdminDiscoverResponse(discovered: items))
            }
            Get("/admin/models/{id}/settings") { request, context in
                let modelId = try context.parameters.require("id")
                let settings = inference.settingsManager.getSettings(modelId)
                return try Self.jsonResponse(AdminModelSettingsResponse(modelId: modelId, settings: settings))
            }
            Put("/admin/models/{id}/settings") { request, context in
                let modelId = try context.parameters.require("id")
                let body = try await request.body.collect(upTo: .max)
                var settings = inference.settingsManager.getSettings(modelId)

                if let update = try? JSONDecoder().decode(ModelSettingsUpdateRequest.self, from: body) {
                    if let v = update.maxContextWindow { settings.maxContextWindow = v }
                    if let v = update.maxTokens { settings.maxTokens = v }
                    if let v = update.temperature { settings.temperature = v }
                    if let v = update.topP { settings.topP = v }
                    if let v = update.topK { settings.topK = v }
                    if let v = update.minP { settings.minP = v }
                    if let v = update.repetitionPenalty { settings.repetitionPenalty = v }
                    if let v = update.presencePenalty { settings.presencePenalty = v }
                    if let v = update.frequencyPenalty { settings.frequencyPenalty = v }
                    if let v = update.ttlSeconds { settings.ttlSeconds = v }
                    if let v = update.modelAlias { settings.modelAlias = v }
                    if let v = update.isPinned { settings.isPinned = v }
                    if let v = update.isDefault { settings.isDefault = v }
                    if let v = update.displayName { settings.displayName = v }
                    if let v = update.description { settings.description = v }
                    if let v = update.thinkingBudget { settings.thinkingBudget = v }
                    if let v = update.kvBits { settings.kvBits = v }
                    if let v = update.kvGroupSize { settings.kvGroupSize = v }
                    if let v = update.kvMemoryBytesPerTokenOverride { settings.kvMemoryBytesPerTokenOverride = v }

                    inference.settingsManager.setSettings(modelId, settings)

                    if update.isPinned == true {
                        if inference.engine.getContainer(for: modelId) != nil {
                            inference.engine.pool.pin(modelId)
                        }
                    } else if update.isPinned == false {
                        inference.engine.pool.unpin(modelId)
                    }

                    if let container = inference.engine.getContainer(for: modelId) {
                        container.kvMemoryOverride = settings.kvMemoryBytesPerTokenOverride
                    }
                }

                return try Self.jsonResponse(AdminModelSettingsResponse(modelId: modelId, settings: settings))
            }
            Get("/admin/sessions") { request, context in
                let sessions = inference.engine.sessionManager.listSessions()
                return try Self.jsonResponse(sessions)
            }
            Delete("/admin/sessions/{id}") { request, context in
                let sessionId = try context.parameters.require("id")
                inference.engine.sessionManager.remove(sessionId)
                return Response(status: .noContent)
            }
            Delete("/admin/sessions") { request, context in
                let sessions = inference.engine.sessionManager.listSessions()
                for session in sessions {
                    inference.engine.sessionManager.remove(session.sessionId)
                }
                return Response(status: .noContent)
            }
            Post("/admin/sessions/{id}/save") { request, context in
                let sessionId = try context.parameters.require("id")
                do {
                    try await inference.engine.saveSession(sessionId)
                    return Response(status: .ok)
                } catch {
                    throw NovaMLXError.cacheError("Failed to save session: \(error.localizedDescription)")
                }
            }
            Post("/admin/sessions/fork") { request, context in
                let body = try await request.body.collect(upTo: 1024 * 1024)
                let req = try JSONDecoder().decode(AdminSessionForkRequest.self, from: body)
                do {
                    try await inference.forkSession(from: req.sourceId, into: req.targetId, modelId: req.modelId)
                    return Response(status: .ok)
                } catch {
                    throw NovaMLXError.cacheError("Failed to fork session: \(error.localizedDescription)")
                }
            }
            Get("/admin/cache/{modelId}/stats") { request, context in
                let modelId = try context.parameters.require("modelId")
                if let stats = inference.engine.getPrefixCacheStats(for: modelId) {
                    let obj: [String: String] = [
                        "hits": String(stats.hits),
                        "misses": String(stats.misses),
                        "tokensSaved": String(stats.tokensSaved),
                        "evictions": String(stats.evictions),
                        "totalBlocks": String(stats.totalBlocks),
                        "allocatedBlocks": String(stats.allocatedBlocks),
                        "freeBlocks": String(stats.freeBlocks),
                        "sharedBlocks": String(stats.sharedBlocks),
                        "ssdBlockCount": String(stats.ssdBlockCount),
                        "ssdTotalSize": String(stats.ssdTotalSize),
                    ]
                    let data = try JSONEncoder().encode(obj)
                    return Response(status: .ok, body: ResponseBody(byteBuffer: ByteBuffer(data: data)))
                }
                return Response(status: .notFound)
            }
            Delete("/admin/cache/{modelId}") { request, context in
                let modelId = try context.parameters.require("modelId")
                inference.engine.clearPrefixCache(for: modelId)
                return Response(status: .ok)
            }
            Get("/admin/api/device-info") { _, _ in
                let info = DeviceInfo.current()
                return try Self.jsonResponse(info)
            }
            Post("/admin/api/bench/start") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let benchReq = try JSONDecoder().decode(BenchmarkRequest.self, from: body)
                let run = try benchmark.startBenchmark(benchReq)
                return try Self.jsonResponse(run, httpStatus: .accepted)
            }
            Get("/admin/api/bench/status") { _, _ in
                if let run = benchmark.getActiveRun() {
                    return try Self.jsonResponse(run)
                }
                return try Self.jsonResponse(["status": "idle"])
            }
            Post("/admin/api/bench/cancel") { _, _ in
                benchmark.cancelActiveRun()
                return try Self.jsonResponse(["status": "cancelled"])
            }
            Post("/admin/api/stats/clear") { _, _ in
                inference.engine.resetSessionMetrics()
                return try Self.jsonResponse(["status": "cleared"])
            }
            Post("/admin/api/stats/clear-alltime") { _, _ in
                inference.engine.metricsStore.clearAllTime()
                return try Self.jsonResponse(["status": "cleared"])
            }
            Get("/admin/api/turboquant") { _, _ in
                let configs = inference.engine.turboQuantService.allConfigs()
                var result: [[String: Any]] = []
                for (modelId, config) in configs {
                    var entry: [String: Any] = [
                        "model_id": modelId,
                        "bits": config.bits,
                        "group_size": config.groupSize,
                        "scheme": "affine",
                        "compression_ratio": config.estimatedCompressionRatio
                    ]
                    if let stats = inference.engine.turboQuantService.getStats(modelId: modelId) {
                        entry["compression_ratio"] = stats.compressionRatio
                    }
                    result.append(entry)
                }
                let jsonData = try JSONSerialization.data(withJSONObject: ["configs": result])
                var headers = HTTPFields()
                headers[.contentType] = "application/json"
                return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: jsonData)))
            }
            Put("/admin/api/turboquant/{modelId}") { request, context in
                let modelId = try context.parameters.require("modelId")
                let body = try await request.body.collect(upTo: .max)
                struct TurboQuantRequest: Codable, Sendable {
                    let enabled: Bool
                    let bits: Int?
                    let groupSize: Int?
                    let modelSizeGB: Double?
                    let contextLength: Int?
                    let availableMemoryGB: Double?
                }
                let req = try JSONDecoder().decode(TurboQuantRequest.self, from: body)
                if req.enabled {
                    if let bits = req.bits {
                        let config = TurboQuantService.Config(bits: bits, groupSize: req.groupSize ?? 64)
                        inference.engine.turboQuantService.setConfig(config, forModel: modelId)
                    } else {
                        _ = inference.engine.turboQuantService.autoConfigure(
                            modelId: modelId,
                            modelSizeGB: req.modelSizeGB ?? 2.0,
                            contextLength: req.contextLength ?? 4096,
                            availableMemoryGB: req.availableMemoryGB ?? 16.0
                        )
                    }
                    return try Self.jsonResponse(["status": "enabled", "model_id": modelId])
                } else {
                    inference.engine.turboQuantService.removeConfig(forModel: modelId)
                    return try Self.jsonResponse(["status": "disabled", "model_id": modelId])
                }
            }
            Delete("/admin/api/turboquant/{modelId}") { request, context in
                let modelId = try context.parameters.require("modelId")
                inference.engine.turboQuantService.removeConfig(forModel: modelId)
                return try Self.jsonResponse(["status": "disabled", "model_id": modelId])
            }
            Put("/admin/api/model-family/{modelId}") { request, context in
                let modelId = try context.parameters.require("modelId")
                let body = try await request.body.collect(upTo: .max)
                struct FamilyOverrideRequest: Codable, Sendable {
                    let defaultKVBits: Int?
                    let defaultKVGroupSize: Int?
                    let prefillStepSize: Int?
                    let recommendedContextLength: Int?
                    let repeatLastN: Int?
                }
                let req = try JSONDecoder().decode(FamilyOverrideRequest.self, from: body)
                let opt = ModelFamilyOptimization(
                    defaultKVBits: req.defaultKVBits,
                    defaultKVGroupSize: req.defaultKVGroupSize ?? 64,
                    prefillStepSize: req.prefillStepSize ?? 512,
                    recommendedContextLength: req.recommendedContextLength ?? 4096,
                    repeatLastN: req.repeatLastN ?? 64
                )
                ModelFamilyRegistry.shared.setOverride(opt, forModel: modelId)
                return try Self.jsonResponse(["status": "ok", "model_id": modelId])
            }
            Delete("/admin/api/model-family/{modelId}") { request, context in
                let modelId = try context.parameters.require("modelId")
                ModelFamilyRegistry.shared.removeOverride(forModel: modelId)
                return try Self.jsonResponse(["status": "removed", "model_id": modelId])
            }
            Get("/admin/api/model-family") { _, _ in
                let overrides = ModelFamilyRegistry.shared.allOverrides()
                var result: [[String: Any]] = []
                for (modelId, opt) in overrides {
                    var entry: [String: Any] = [
                        "model_id": modelId,
                        "kv_group_size": opt.defaultKVGroupSize,
                        "prefill_step_size": opt.prefillStepSize,
                        "recommended_context_length": opt.recommendedContextLength,
                        "repeat_last_n": opt.repeatLastN
                    ]
                    if let bits = opt.defaultKVBits {
                        entry["kv_bits"] = bits
                    }
                    result.append(entry)
                }
                let jsonData = try JSONSerialization.data(withJSONObject: ["overrides": result])
                var headers = HTTPFields()
                headers[.contentType] = "application/json"
                return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: jsonData)))
            }
            Post("/admin/api/ppl/start") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let pplReq = try JSONDecoder().decode(PerplexityRequest.self, from: body)
                let run = try perplexity.startEvaluation(pplReq)
                return try Self.jsonResponse(run, httpStatus: .accepted)
            }
            Get("/admin/api/ppl/status") { _, _ in
                if let run = perplexity.getActiveRun() {
                    return try Self.jsonResponse(run)
                }
                return try Self.jsonResponse(["status": "idle"])
            }
            Post("/admin/api/ppl/cancel") { _, _ in
                perplexity.cancelActiveRun()
                return try Self.jsonResponse(["status": "cancelled"])
            }
            Get("/admin/adapters") { request, _ in
                let modelId = request.uri.query.flatMap { url in
                    URLComponents(string: "/" + url)?.queryItems?.first(where: { $0.name == "model_id" })?.value
                }
                let adapters: [AdapterInfo]
                if let modelId {
                    adapters = inference.engine.adapterService.listAdapters(for: modelId)
                } else {
                    adapters = inference.engine.adapterService.listAdapters()
                }
                return try Self.jsonResponse(["adapters": adapters])
            }
            Post("/admin/adapters/load") { request, _ in
                struct AdapterLoadRequest: Codable, Sendable {
                    let modelId: String
                    let path: String
                    let name: String?
                }
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdapterLoadRequest.self, from: body)

                guard let container = inference.engine.getContainer(for: req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                let adapterURL = URL(fileURLWithPath: req.path)
                let info = try await inference.engine.adapterService.loadAdapter(
                    from: adapterURL,
                    into: container,
                    name: req.name
                )
                return try Self.jsonResponse(info)
            }
            Post("/admin/adapters/unload") { request, _ in
                struct AdapterUnloadRequest: Codable, Sendable {
                    let modelId: String
                    let name: String
                }
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdapterUnloadRequest.self, from: body)

                guard let container = inference.engine.getContainer(for: req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                let info = try await inference.engine.adapterService.unloadAdapter(
                    name: req.name,
                    from: container
                )
                return try Self.jsonResponse(info)
            }
            Post("/admin/adapters/fuse") { request, _ in
                struct AdapterFuseRequest: Codable, Sendable {
                    let modelId: String
                    let name: String
                }
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdapterFuseRequest.self, from: body)

                guard let container = inference.engine.getContainer(for: req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                let info = try await inference.engine.adapterService.fuseAdapter(
                    name: req.name,
                    into: container
                )
                return try Self.jsonResponse(info)
            }
            Get("/admin/adapters/discover") { request, _ in
                let dir: String? = request.uri.query.flatMap { url in
                    URLComponents(string: "/" + url)?.queryItems?.first(where: { $0.name == "directory" })?.value
                }
                let searchDir = dir.map { URL(fileURLWithPath: $0) }
                    ?? NovaMLXPaths.modelsDir
                let adapters = inference.engine.adapterService.discoverAdapters(in: searchDir)
                return try Self.jsonResponse(["adapters": adapters])
            }
            Post("/admin/api/grammar/validate") { request, _ in
                struct GrammarValidateRequest: Codable, Sendable {
                    let type: String
                    let value: String
                }
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(GrammarValidateRequest.self, from: body)
                do {
                    switch req.type {
                    case "regex":
                        _ = try NSRegularExpression(pattern: req.value, options: [])
                        let jsonData = try JSONSerialization.data(withJSONObject: ["valid": true, "type": "regex"] as [String: Any])
                        var headers = HTTPFields()
                        headers[.contentType] = "application/json"
                        return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: jsonData)))
                    case "gbnf":
                        let rules = try GBNFParser.parse(req.value)
                        let result: [String: Any] = [
                            "valid": true,
                            "type": "gbnf",
                            "rules": rules.map { ["name": $0.name, "alternatives": $0.alternatives.count] as [String: Any] }
                        ]
                        let jsonData = try JSONSerialization.data(withJSONObject: result)
                        var headers = HTTPFields()
                        headers[.contentType] = "application/json"
                        return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: jsonData)))
                    default:
                        let jsonData = try JSONSerialization.data(withJSONObject: ["valid": false, "error": "Unknown grammar type: \(req.type). Use 'regex' or 'gbnf'."] as [String: Any])
                        var headers = HTTPFields()
                        headers[.contentType] = "application/json"
                        return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: jsonData)))
                    }
                } catch {
                    let jsonData = try JSONSerialization.data(withJSONObject: ["valid": false, "error": error.localizedDescription] as [String: Any])
                    var headers = HTTPFields()
                    headers[.contentType] = "application/json"
                    return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: jsonData)))
                }
            }
            Get("/admin/api/update-check") { _, _ in
                do {
                    let info = try await updater.checkForUpdates()
                    return try Self.jsonResponse(info)
                } catch {
                    return try Self.jsonResponse(["error": error.localizedDescription])
                }
            }
            Get("/admin/api/hf/search") { request, _ in
                do {
                    let query = request.uri.query ?? ""
                    let params = Self.parseQuery(query)
                    let q = params["q"] ?? ""
                    let limit = Int(params["limit"] ?? "50") ?? 50
                    let mlxOnly = params["mlx_only"] != "false"
                    let result = try await hf.searchModels(query: q, limit: limit, mlxOnly: mlxOnly)
                    return try Self.jsonResponse(result)
                } catch {
                    NovaMLXLog.error("HF search failed: \(error)")
                    return try Self.jsonResponse(["error": error.localizedDescription])
                }
            }
            Get("/admin/api/hf/model-info") { request, _ in
                do {
                    let query = request.uri.query ?? ""
                    let params = Self.parseQuery(query)
                    guard let repoId = params["repo_id"] else {
                        return try Self.jsonResponse(["error": "repo_id required"], httpStatus: .badRequest)
                    }
                    let detail = try await hf.getModelDetail(repoId: repoId)
                    return try Self.jsonResponse(detail)
                } catch {
                    return try Self.jsonResponse(["error": error.localizedDescription])
                }
            }
            Post("/admin/api/hf/download") { request, _ in
                do {
                    let body = try await request.body.collect(upTo: .max)
                    let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] ?? [:]
                    let repoId = json["repo_id"] as? String ?? ""
                    let hfToken = json["hf_token"] as? String
                    guard !repoId.isEmpty else {
                        return try Self.jsonResponse(["error": "repo_id required"], httpStatus: .badRequest)
                    }
                    let task = try await hf.startDownload(repoId: repoId, hfToken: hfToken)
                    return try Self.jsonResponse(["success": "true", "task_id": task.id] as [String: String])
                } catch {
                    return try Self.jsonResponse(["error": error.localizedDescription])
                }
            }
            Get("/admin/api/hf/tasks") { _, _ in
                let tasks = hf.getTasks()
                return try Self.jsonResponse(["tasks": tasks])
            }
            Post("/admin/api/hf/cancel") { request, _ in
                do {
                    let body = try await request.body.collect(upTo: .max)
                    let json = try JSONSerialization.jsonObject(with: body) as? [String: Any] ?? [:]
                    let taskId = json["task_id"] as? String ?? ""
                    let success = hf.cancelTask(id: taskId)
                    return try Self.jsonResponse(["success": success])
                } catch {
                    return try Self.jsonResponse(["error": error.localizedDescription])
                }
            }
            Get("/admin/api/rate-limits") { _, _ in
                let stats = rateLimiter.getStats()
                let data = try JSONSerialization.data(withJSONObject: stats)
                return Response(
                    status: .ok,
                    headers: [.contentType: "application/json"],
                    body: .init(byteBuffer: ByteBuffer(data: data))
                )
            }
            Get("/admin/dashboard") { _, _ in
                let html = Self.dashboardHTML()
                return Response(
                    status: .ok,
                    headers: [.contentType: "text/html"],
                    body: .init(byteBuffer: ByteBuffer(string: html))
                )
            }
        }

        var mainConfig = ApplicationConfiguration(address: .hostname(cfg.host, port: cfg.port))
        if let certPath = cfg.tlsCertPath {
            do {
                let identity = try TSTLSOptions.Identity.p12(filename: certPath, password: cfg.tlsKeyPassword ?? "")
                if let tlsOpts = TSTLSOptions.options(serverIdentity: identity) {
                    mainConfig = ApplicationConfiguration(
                        address: .hostname(cfg.host, port: cfg.port),
                        tlsOptions: tlsOpts
                    )
                    NovaMLXLog.info("TLS enabled for API server")
                }
            } catch {
                NovaMLXLog.error("Failed to load TLS certificate: \(error)")
            }
        }

        let mainApp = Application(
            router: mainRouter,
            configuration: mainConfig,
            logger: Logger(label: "NovaMLX.API")
        )

        let adminApp = Application(
            router: adminRouter,
            configuration: .init(address: .hostname(cfg.host, port: cfg.adminPort)),
            logger: Logger(label: "NovaMLX.Admin")
        )

        NovaMLXLog.info("NovaMLX API server starting on \(cfg.host):\(cfg.port)\(cfg.isTLSEnabled ? " (TLS)" : "")")
        NovaMLXLog.info("NovaMLX Admin API starting on \(cfg.host):\(cfg.adminPort)")

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await mainApp.run() }
            group.addTask { try await adminApp.run() }
            try await group.next()
            group.cancelAll()
        }
    }

    private static func handleChat(
        openAIReq: OpenAIRequest, messages: [ChatMessage], inference: InferenceService,
        sessionId: String? = nil, responseFormat: ResponseFormat? = nil, jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil, gbnfGrammar: String? = nil
    ) async throws -> Response {
        let request = InferenceRequest(
            model: openAIReq.model, messages: messages,
            tools: openAIReq.tools?.map { tool in
                tool.mapValues { unwrapAnyCodable($0) }
            },
            temperature: openAIReq.temperature, maxTokens: openAIReq.maxTokens,
            topP: openAIReq.topP, topK: openAIReq.topK,
            minP: openAIReq.minP.map { Float($0) },
            frequencyPenalty: openAIReq.frequencyPenalty.map { Float($0) },
            presencePenalty: openAIReq.presencePenalty.map { Float($0) },
            repetitionPenalty: openAIReq.repetitionPenalty.map { Float($0) },
            seed: openAIReq.seed,
            stream: false, stop: openAIReq.stop,
            sessionId: sessionId, responseFormat: responseFormat,
            jsonSchemaDef: jsonSchemaDef,
            regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
            thinkingBudget: openAIReq.thinkingBudget
        )

        let result = try await inference.generate(request)
        let finishReason: String
        let message: OpenAIChatMessage

        // Engine-produced tool calls take priority
        if let engineToolCalls = result.toolCalls, !engineToolCalls.isEmpty {
            finishReason = "tool_calls"
            message = OpenAIChatMessage(
                role: "assistant",
                content: result.text.isEmpty ? nil : result.text,
                toolCalls: engineToolCalls.map { tc in
                    OpenAIToolCall(id: tc.id, function: OpenAIFunctionCall(name: tc.functionName, arguments: tc.arguments))
                }
            )
        } else {
            // Fallback: post-hoc text parsing
            let parsed = ToolCallParser.parse(result.text)
            if let toolCalls = parsed.toolCalls {
                finishReason = "tool_calls"
                message = OpenAIChatMessage(
                    role: "assistant",
                    content: parsed.content.isEmpty ? nil : parsed.content,
                    toolCalls: toolCalls.map { tc in
                        OpenAIToolCall(id: tc.id, function: OpenAIFunctionCall(name: tc.function.name, arguments: tc.function.arguments))
                    }
                )
            } else {
                finishReason = result.finishReason.rawValue
                message = OpenAIChatMessage(role: "assistant", content: result.text)
            }
        }
        let response = OpenAIResponse(
            id: "chatcmpl-\(result.id.uuidString.prefix(8))",
            model: result.model,
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: message,
                    finishReason: finishReason
                )
            ],
            usage: OpenAIUsage(promptTokens: result.promptTokens, completionTokens: result.completionTokens)
        )
        return try jsonResponse(response)
    }

    private static func handleStreamChat(
        openAIReq: OpenAIRequest, messages: [ChatMessage], inference: InferenceService,
        sessionId: String? = nil, responseFormat: ResponseFormat? = nil, jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil, gbnfGrammar: String? = nil
    ) async throws -> Response {
        let request = InferenceRequest(
            model: openAIReq.model, messages: messages,
            tools: openAIReq.tools?.map { tool in
                tool.mapValues { unwrapAnyCodable($0) }
            },
            temperature: openAIReq.temperature, maxTokens: openAIReq.maxTokens,
            topP: openAIReq.topP, topK: openAIReq.topK,
            minP: openAIReq.minP.map { Float($0) },
            frequencyPenalty: openAIReq.frequencyPenalty.map { Float($0) },
            presencePenalty: openAIReq.presencePenalty.map { Float($0) },
            repetitionPenalty: openAIReq.repetitionPenalty.map { Float($0) },
            seed: openAIReq.seed,
            stream: true, stop: openAIReq.stop,
            sessionId: sessionId, responseFormat: responseFormat,
            jsonSchemaDef: jsonSchemaDef,
            regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
            thinkingBudget: openAIReq.thinkingBudget
        )

        let keepAliveStream = Self.withSSEKeepAlive(inference.stream(request))
        let chunkId = "chatcmpl-\(request.id.uuidString.prefix(8))"
        let includeUsage = openAIReq.streamOptions?.includeUsage == true
        let toolCallCounter = LockedCounter()

        let body: ResponseBody = .init { writer in
            do {
                let roleChunk = OpenAIStreamChunk(
                    id: chunkId,
                    model: openAIReq.model,
                    choices: [
                        OpenAIStreamChoice(index: 0, delta: OpenAIDelta(role: "assistant", content: ""))
                    ]
                )
                let roleData = try JSONEncoder().encode(roleChunk)
                try await writer.write(ByteBuffer(string: "data: \(String(data: roleData, encoding: .utf8) ?? "")\n\n"))

                var completionTokenCount = 0
                let thinkingParser = ThinkingParser()
                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
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
                                model: openAIReq.model,
                                choices: [OpenAIStreamChoice(index: 0, delta: OpenAIDelta(toolCalls: [tcDelta]))]
                            )
                            let data = try JSONEncoder().encode(chunk)
                            try await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                        } else if let finish = token.finishReason {
                            let finalChunk = OpenAIStreamChunk(
                                id: chunkId,
                                model: openAIReq.model,
                                choices: [
                                    OpenAIStreamChoice(
                                        index: 0,
                                        delta: OpenAIDelta(),
                                        finishReason: finish.rawValue
                                    )
                                ]
                            )
                            let finalData = try JSONEncoder().encode(finalChunk)
                            try await writer.write(ByteBuffer(string: "data: \(String(data: finalData, encoding: .utf8) ?? "")\n\n"))
                        } else {
                            completionTokenCount += 1
                            let parsed = thinkingParser.feed(token.text)
                            if parsed.text.isEmpty { continue }
                            let delta: OpenAIDelta = parsed.type == .thinking
                                ? OpenAIDelta(reasoningContent: parsed.text)
                                : OpenAIDelta(content: parsed.text)
                            let chunk = OpenAIStreamChunk(
                                id: chunkId,
                                model: openAIReq.model,
                                choices: [OpenAIStreamChoice(index: 0, delta: delta)]
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
                if includeUsage {
                    let promptTokenCount: Int
                    if let tc = inference.countTokens(model: openAIReq.model, messages: messages) {
                        promptTokenCount = tc
                    } else {
                        promptTokenCount = 0
                    }
                    let usageChunk = OpenAIStreamChunk(
                        id: chunkId,
                        model: openAIReq.model,
                        choices: [],
                        usage: OpenAIUsage(promptTokens: promptTokenCount, completionTokens: completionTokenCount)
                    )
                    let usageData = try JSONEncoder().encode(usageChunk)
                    try await writer.write(ByteBuffer(string: "data: \(String(data: usageData, encoding: .utf8) ?? "")\n\n"))
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                try await writer.finish(nil)
            } catch {
                NovaMLXLog.error("Stream error: \(error)")
                let (message, type, code) = Self.streamErrorFields(error)
                let errorDetail = OpenAIErrorDetail(message: message, type: type, code: code)
                let errorResp = OpenAIErrorResponse(error: errorDetail)
                if let data = try? JSONEncoder().encode(errorResp) {
                    let sseError = "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"
                    try? await writer.write(ByteBuffer(string: sseError))
                }
                try? await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
                try? await writer.finish(nil)
            }
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
    }

    private static func handleStreamAnthropic(
        anthropicReq: AnthropicRequest, messages: [ChatMessage], inference: InferenceService
    ) async throws -> Response {
        let request = InferenceRequest(
            model: anthropicReq.model, messages: messages,
            temperature: anthropicReq.temperature, maxTokens: anthropicReq.maxTokens,
            topP: anthropicReq.topP, stream: true, stop: anthropicReq.stopSequences
        )

        let reqTag = request.id.uuidString.prefix(8)
        NovaMLXLog.info("[SSE:\(reqTag)] Anthropic stream request started — model=\(anthropicReq.model), maxTokens=\(anthropicReq.maxTokens)")

        let keepAliveStream = Self.withSSEKeepAlive(inference.stream(request), reqTag: String(reqTag))
        let msgId = "msg_\(request.id.uuidString.prefix(24))"

        let body: ResponseBody = .init { writer in
            var tokenCount = 0
            let streamStart = Date()
            do {
                try await writer.write(ByteBuffer(string: "event: ping\ndata: {\"type\":\"ping\"}\n\n"))
                NovaMLXLog.info("[SSE:\(reqTag)] Sent initial headers + ping")

                let startEvent = AnthropicStreamEvent.messageStart(id: msgId, model: anthropicReq.model)
                let startData = try JSONEncoder().encode(startEvent)
                try await writer.write(ByteBuffer(string: "event: message_start\ndata: \(String(data: startData, encoding: .utf8) ?? "{}")\n\n"))

                let blockStart = AnthropicStreamEvent.contentBlockStart(index: 0)
                let blockStartData = try JSONEncoder().encode(blockStart)
                try await writer.write(ByteBuffer(string: "event: content_block_start\ndata: \(String(data: blockStartData, encoding: .utf8) ?? "{}")\n\n"))

                NovaMLXLog.info("[SSE:\(reqTag)] Waiting for first token from inference stream...")

                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        if token.finishReason != nil {
                            try await writer.write(ByteBuffer(string: "event: content_block_stop\ndata: {}\n\n"))
                            try await writer.write(ByteBuffer(string: "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"))
                            try await writer.write(ByteBuffer(string: "event: message_stop\ndata: {}\n\n"))
                            let elapsed = Date().timeIntervalSince(streamStart)
                            NovaMLXLog.info("[SSE:\(reqTag)] Stream complete — \(tokenCount) tokens in \(String(format: "%.1f", elapsed))s")
                        } else if !token.text.isEmpty {
                            if tokenCount == 0 {
                                let ttft = Date().timeIntervalSince(streamStart)
                                NovaMLXLog.info("[SSE:\(reqTag)] First token received (TTFT=\(String(format: "%.1f", ttft))s)")
                            }
                            tokenCount += 1
                            let deltaEvent = AnthropicStreamEvent.textDelta(token.text)
                            let deltaData = try JSONEncoder().encode(deltaEvent)
                            try await writer.write(ByteBuffer(string: "event: content_block_delta\ndata: \(String(data: deltaData, encoding: .utf8) ?? "{}")\n\n"))
                        }
                    case .keepAlive:
                        let elapsed = Date().timeIntervalSince(streamStart)
                        NovaMLXLog.info("[SSE:\(reqTag)] Keep-alive ping sent (\(String(format: "%.0f", elapsed))s elapsed, \(tokenCount) tokens so far)")
                    case .done:
                        NovaMLXLog.info("[SSE:\(reqTag)] Stream done signal")
                    }
                }
            } catch {
                let elapsed = Date().timeIntervalSince(streamStart)
                NovaMLXLog.error("[SSE:\(reqTag)] Stream ERROR after \(String(format: "%.1f", elapsed))s, \(tokenCount) tokens sent: \(error)")
                // Use Anthropic error format for Anthropic API endpoint
                let (message, errorType) = Self.anthropicStreamErrorFields(error)
                let errorDetail = AnthropicErrorDetail(type: errorType, message: message)
                let errorResp = AnthropicErrorResponse(error: errorDetail)
                if let data = try? JSONEncoder().encode(errorResp) {
                    try? await writer.write(ByteBuffer(string: "event: error\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                }
                try? await writer.finish(nil)
            }
            try? await writer.finish(nil)
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
    }

    private static func handleCompletion(
        compReq: OpenAICompletionRequest, inference: InferenceService
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
            stream: false, stop: compReq.stop
        )

        let result = try await inference.generate(request)
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
            usage: OpenAIUsage(promptTokens: result.promptTokens, completionTokens: result.completionTokens)
        )
        return try jsonResponse(response)
    }

    private static func handleStreamCompletion(
        compReq: OpenAICompletionRequest, inference: InferenceService
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
            stream: true, stop: compReq.stop
        )

        let keepAliveStream = Self.withSSEKeepAlive(inference.stream(request))
        let chunkId = "cmpl-\(request.id.uuidString.prefix(8))"

        let body: ResponseBody = .init { writer in
            do {
                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        if let finish = token.finishReason {
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
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
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
                try? await writer.finish(nil)
            }
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
    }

    private static func handleResponsesRequest(
        req: OpenAIResponseRequest, inference: InferenceService
    ) async throws -> Response {
        var messages: [ChatMessage] = []
        if let instructions = req.instructions {
            messages.append(ChatMessage(role: .system, content: instructions))
        }
        switch req.input {
        case .text(let prompt):
            messages.append(ChatMessage(role: .user, content: prompt))
        case .messages(let inputMessages):
            for msg in inputMessages {
                let role: ChatMessage.Role = msg.role == "assistant" ? .assistant : .user
                messages.append(ChatMessage(role: role, content: msg.content))
            }
        case .none:
            break
        }

        let request = InferenceRequest(
            model: req.model, messages: messages,
            temperature: req.temperature, maxTokens: req.maxOutputTokens,
            topP: req.topP, stream: false
        )

        let result = try await inference.generate(request)
        let responseId = "resp_\(result.id.uuidString.prefix(24))"
        let outputItemId = "msg_\(result.id.uuidString.prefix(24))"
        let response = OpenAIResponseObject(
            id: responseId,
            model: result.model,
            output: [
                ResponseOutputItem(
                    id: outputItemId,
                    content: [ResponseContentItem(text: result.text)]
                )
            ],
            usage: OpenAIUsage(promptTokens: result.promptTokens, completionTokens: result.completionTokens)
        )
        ResponseStore.shared.put(response)
        return try jsonResponse(response)
    }

    private enum SSEKeepAliveEvent: Sendable {
        case token(Token)
        case keepAlive
        case done
    }

    private static func withSSEKeepAlive(
        _ stream: AsyncThrowingStream<Token, Error>,
        interval: Duration = .seconds(10),
        reqTag: String = "unknown"
    ) -> AsyncThrowingStream<SSEKeepAliveEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    continuation.yield(.keepAlive)
                    for try await token in stream {
                        if Task.isCancelled {
                            NovaMLXLog.warning("[SSE:\(reqTag)] Inference stream consumer cancelled")
                            break
                        }
                        continuation.yield(.token(token))
                    }
                    NovaMLXLog.info("[SSE:\(reqTag)] Inference stream finished normally")
                    continuation.finish()
                } catch {
                    NovaMLXLog.error("[SSE:\(reqTag)] Inference stream error: \(error)")
                    continuation.finish(throwing: error)
                }
            }
            let heartbeat = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: interval)
                    guard !Task.isCancelled else { break }
                    continuation.yield(.keepAlive)
                }
            }
            continuation.onTermination = { reason in
                NovaMLXLog.warning("[SSE:\(reqTag)] SSE connection terminated: \(reason)")
                task.cancel()
                heartbeat.cancel()
            }
        }
    }

    private static func streamErrorFields(_ error: Error) -> (message: String, type: String, code: String) {
        if let error = error as? NovaMLXError {
            return (error.errorDescription ?? "Unknown error", error.apiErrorType, error.apiErrorCode)
        }
        return (error.localizedDescription, "internal_error", "internal_error")
    }

    /// Anthropic-format error fields for SSE error events.
    /// Returns (message, type) matching the Anthropic API error schema.
    private static func anthropicStreamErrorFields(_ error: Error) -> (message: String, type: String) {
        if let error = error as? NovaMLXError {
            return (error.errorDescription ?? "Unknown error", error.apiErrorType)
        }
        return (error.localizedDescription, "api_error")
    }

    private static let sessionIDHeader = HTTPField.Name("x-session-id")!

    private static func parseQuery(_ query: String) -> [String: String] {
        var result: [String: String] = [:]
        for pair in query.split(separator: "&") {
            let parts = pair.split(separator: "=", maxSplits: 1)
            if parts.count == 2 {
                result[String(parts[0])] = String(parts[1]).removingPercentEncoding ?? String(parts[1])
            } else if parts.count == 1 {
                result[String(parts[0])] = ""
            }
        }
        return result
    }

    private static func extractSessionId(request: Request, body: String?) -> String? {
        if let header = request.headers[fields: sessionIDHeader].first?.value, !header.isEmpty {
            return header
        }
        return body
    }

    private static func jsonResponse<T: Encodable>(_ value: T) throws -> Response {
        try jsonResponse(value, httpStatus: .ok)
    }

    private static func jsonResponse<T: Encodable>(_ value: T, httpStatus: HTTPResponse.Status) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(
            status: httpStatus,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    private static func dashboardHTML() -> String {
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NovaMLX Dashboard</title>
        <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0a0a0a;color:#e5e5e5;padding:24px}
        .container{max-width:1200px;margin:0 auto}
        h1{font-size:28px;font-weight:700;margin-bottom:8px;color:#fff}
        h1 span{color:#8b5cf6}
        .subtitle{color:#737373;margin-bottom:32px;font-size:14px}
        .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}
        .card{background:#171717;border:1px solid #262626;border-radius:12px;padding:20px}
        .card h2{font-size:13px;text-transform:uppercase;letter-spacing:0.05em;color:#a3a3a3;margin-bottom:12px}
        .card .value{font-size:32px;font-weight:700;color:#fff}
        .card .sub{font-size:13px;color:#737373;margin-top:4px}
        .status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
        .status-dot.ok{background:#22c55e}
        .status-dot.warn{background:#eab308}
        .status-dot.err{background:#ef4444}
        table{width:100%;border-collapse:collapse;font-size:13px}
        th{text-align:left;color:#737373;padding:8px 12px;border-bottom:1px solid #262626;font-weight:500}
        td{padding:8px 12px;border-bottom:1px solid #1a1a1a}
        .btn{display:inline-block;padding:6px 14px;border-radius:6px;border:1px solid #333;background:#1a1a1a;color:#e5e5e5;font-size:12px;cursor:pointer;margin:2px}
        .btn:hover{background:#262626}
        .btn.danger{border-color:#7f1d1d;color:#fca5a5}
        .btn.danger:hover{background:#7f1d1d}
        .bench-results{margin-top:12px}
        .bench-results table{margin-top:8px}
        #refresh-btn{position:fixed;top:24px;right:24px}
        .nav-links{display:flex;gap:8px;margin-bottom:16px}
        .nav-links a{color:var(--accent);text-decoration:none;font-size:13px;padding:4px 10px;border:1px solid #333;border-radius:6px}
        .nav-links a:hover{background:#1a1a1a}
        @media(max-width:768px){
        body{padding:16px}
        .container{max-width:100%}
        .grid{grid-template-columns:1fr}
        table{font-size:11px}
        th,td{padding:6px 8px}
        #refresh-btn{position:static;margin-bottom:16px}
        }
        </style>
        </head>
        <body>
        <div class="container">
        <h1>Nova<span>MLX</span> Dashboard</h1>
        <p class="subtitle" id="uptime">Loading...</p>
        <div class="nav-links">
        <button class="btn" id="refresh-btn" onclick="loadAll()">Refresh</button>
        <a href="/chat">Chat</a>
        </div>
        <div class="grid" id="cards"></div>
        <div class="card" style="margin-bottom:16px">
        <h2>Device Info</h2>
        <div id="device-info">Loading...</div>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>MCP Servers</h2>
        <div id="mcp-info">Loading...</div>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>Benchmark</h2>
        <div id="bench-info">
        <button class="btn" onclick="runBench()">Run Benchmark</button>
        <button class="btn" onclick="cancelBench()">Cancel</button>
        <div class="bench-results" id="bench-results"></div>
        </div>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>Actions</h2>
        <button class="btn danger" onclick="clearSessionStats()">Clear Session Stats</button>
        <button class="btn danger" onclick="clearAllTimeStats()">Clear All-Time Stats</button>
        </div>
        <div class="card" style="margin-bottom:16px">
        <h2>HuggingFace Model Browser</h2>
        <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap">
        <input id="hf-search" type="text" placeholder="Search models (e.g. llama mlx)..." style="flex:1;min-width:200px;padding:8px 12px;background:#1a1a1a;border:1px solid #333;border-radius:6px;color:#e5e5e5;font-size:13px">
        <label style="display:flex;align-items:center;gap:4px;font-size:12px;color:#a3a3a3"><input type="checkbox" id="hf-mlx-only" checked> MLX only</label>
        <button class="btn" onclick="hfSearch()">Search</button>
        </div>
        <div id="hf-results" style="max-height:500px;overflow-y:auto"></div>
        <div id="hf-tasks" style="margin-top:12px"></div>
        </div>
        </div>
        <script>
        const API=location.port==='8081'?'':':8081';
        const BASE='http://'+location.hostname+API;
        const ADMIN_BASE=BASE.replace(/:\\d+/,'');
        let adminToken='';
        function authHeaders(){return adminToken?{'Authorization':'Bearer '+adminToken}:{}}
        async function loadHealth(){
        const r=await fetch(BASE+'/health');
        const d=await r.json();
        document.getElementById('cards').innerHTML=`
        <div class="card"><h2>Status</h2><div class="value"><span class="status-dot ok"></span>${d.status||'ok'}</div><div class="sub">Loaded models: ${d.loadedModels||0}</div></div>
        <div class="card"><h2>GPU Memory</h2><div class="value">${((d.gpuMemoryUsed||0)/1024/1024/1024).toFixed(2)} GB</div><div class="sub">Active GPU allocation</div></div>
        `;
        if(d.mcp){
        const mcpHtml=d.mcp.servers&&d.mcp.servers.length?`<table><tr><th>Server</th><th>State</th><th>Tools</th></tr>${d.mcp.servers.map(s=>`<tr><td>${s.name}</td><td><span class="status-dot ${s.state==='connected'?'ok':'err'}"></span>${s.state}</td><td>${s.toolsCount}</td></tr>`).join('')}</table>`:'<div class="sub">No MCP servers configured</div>';
        document.getElementById('mcp-info').innerHTML=mcpHtml;
        }
        }
        async function loadStats(){
        const r=await fetch(BASE+'/v1/stats');
        const d=await r.json();
        const s=d.session||{};
        const a=d.allTime||{};
        document.getElementById('uptime').textContent=`Session: ${s.totalRequests||0} requests | ${(s.totalTokens||0).toLocaleString()} tokens | ${s.averageTokensPerSecond?.toFixed(1)||0} tok/s | All-time: ${a.totalRequests||0} requests`;
        }
        async function loadDeviceInfo(){
        try{
        const r=await fetch(ADMIN_BASE+':8081/admin/api/device-info',{headers:authHeaders()});
        const d=await r.json();
        document.getElementById('device-info').innerHTML=`<table><tr><th>Chip</th><td>${d.chipName||'N/A'}</td></tr><tr><th>Variant</th><td>${d.chipVariant||'N/A'}</td></tr><tr><th>Memory</th><td>${d.memoryGB||0} GB</td></tr><tr><th>GPU Cores</th><td>${d.gpuCores||0}</td></tr><tr><th>CPU Cores</th><td>${d.cpuCores||0}</td></tr><tr><th>OS</th><td>${d.osVersion||'N/A'}</td></tr><tr><th>NovaMLX</th><td>${d.novaMLXVersion||'N/A'}</td></tr></table>`;
        }catch(e){document.getElementById('device-info').textContent='Admin auth required'}
        }
        async function loadBenchStatus(){
        try{
        const r=await fetch(ADMIN_BASE+':8081/admin/api/bench/status',{headers:authHeaders()});
        const d=await r.json();
        if(d.status==='idle'){document.getElementById('bench-results').innerHTML='<div class="sub">No benchmark running</div>';return}
        let html=`<div class="sub">${d.status} - ${((d.progress||0)*100).toFixed(0)}%</div>`;
        if(d.results&&d.results.length){
        html+='<table><tr><th>Prompt Len</th><th>TTFT (ms)</th><th>Gen tok/s</th><th>Prefill tok/s</th><th>Peak Mem GB</th><th>Latency (s)</th></tr>';
        d.results.forEach(r=>{html+=`<tr><td>${r.prompt_length}</td><td>${r.ttft_ms.toFixed(0)}</td><td>${r.generation_tps.toFixed(1)}</td><td>${r.processing_tps.toFixed(1)}</td><td>${r.peak_memory_gb.toFixed(2)}</td><td>${r.e2e_latency_s.toFixed(2)}</td></tr>`});
        html+='</table>';
        }
        if(d.error)html+=`<div class="sub" style="color:#fca5a5">${d.error}</div>`;
        document.getElementById('bench-results').innerHTML=html;
        }catch(e){document.getElementById('bench-results').innerHTML='<div class="sub">Admin auth required</div>'}
        }
        async function runBench(){
        const model=prompt('Enter model ID to benchmark:');
        if(!model)return;
        try{
        await fetch(ADMIN_BASE+':8081/admin/api/bench/start',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({model_id:model,prompt_lengths:[512,2048,4096],generation_length:128})});
        setTimeout(loadBenchStatus,1000);
        }catch(e){alert('Failed: '+e)}
        }
        async function cancelBench(){
        await fetch(ADMIN_BASE+':8081/admin/api/bench/cancel',{method:'POST',headers:authHeaders()});
        setTimeout(loadBenchStatus,500);
        }
        async function clearSessionStats(){
        await fetch(ADMIN_BASE+':8081/admin/api/stats/clear',{method:'POST',headers:authHeaders()});
        loadStats();
        }
        async function clearAllTimeStats(){
        if(!confirm('Clear all-time stats? This cannot be undone.'))return;
        await fetch(ADMIN_BASE+':8081/admin/api/stats/clear-alltime',{method:'POST',headers:authHeaders()});
        loadStats();
        }
        function loadAll(){loadHealth();loadStats();loadDeviceInfo();loadBenchStatus();loadHFTasks()}
        loadAll();setInterval(function(){loadAll()},5000);
        async function hfSearch(){
        const q=document.getElementById('hf-search').value.trim();
        if(!q)return;
        const mlxOnly=document.getElementById('hf-mlx-only').checked;
        const p=page||1;
        try{
        const url=ADMIN_BASE+':8081/admin/api/hf/search?q='+encodeURIComponent(q)+(mlxOnly?'&mlx_only=true':'')+'&limit=10';
        const r=await fetch(url,{headers:authHeaders()});
        const d=await r.json();
        if(!d.models||!d.models.length){document.getElementById('hf-results').innerHTML='<div class="sub">No models found</div>';return}
        let html='<table><tr><th>Model</th><th>Downloads</th><th>Likes</th><th>Action</th></tr>';
        d.models.forEach(function(m){
        const dl=m.downloads?(m.downloads>1000?(m.downloads/1000).toFixed(1)+'k':m.downloads):'0';
        html+='<tr><td style="max-width:300px;word-break:break-all"><a href="https://huggingface.co/'+m.id+'" target="_blank" style="color:#8b5cf6;text-decoration:none">'+m.id+'</a></td><td>'+dl+'</td><td>'+(m.likes||0)+'</td><td><button class="btn" onclick="hfDownload(\''+m.id+'\')">Download</button></td></tr>';
        });
        html+='</table>';
        document.getElementById('hf-results').innerHTML=html;
        }catch(e){document.getElementById('hf-results').innerHTML='<div class="sub" style="color:#fca5a5">Admin auth required</div>'}
        }
        async function hfDownload(modelId){
        try{
        await fetch(ADMIN_BASE+':8081/admin/api/hf/download',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({repo_id:modelId})});
        loadHFTasks();
        }catch(e){alert('Download failed: '+e)}
        }
        async function loadHFTasks(){
        try{
        const r=await fetch(ADMIN_BASE+':8081/admin/api/hf/tasks',{headers:authHeaders()});
        const d=await r.json();
        if(!d.tasks||!d.tasks.length){document.getElementById('hf-tasks').innerHTML='';return}
        let html='<h3 style="font-size:13px;color:#a3a3a3;margin-bottom:8px">Downloads</h3><table><tr><th>Model</th><th>Progress</th><th>Status</th><th>Action</th></tr>';
        d.tasks.forEach(function(t){
        const pct=t.progress?t.progress.toFixed(0):'0';
        const mb=(t.downloadedBytes/1024/1024).toFixed(0)+'/'+(t.totalBytes/1024/1024).toFixed(0)+'MB';
        html+='<tr><td>'+t.repoId+'</td><td>'+pct+'% ('+mb+')</td><td>'+t.status+'</td><td>'+(t.status==='downloading'||t.status==='pending'?'<button class="btn danger" onclick="hfCancel(\''+t.id+'\')">Cancel</button>':'')+'</td></tr>';
        });
        html+='</table>';
        document.getElementById('hf-tasks').innerHTML=html;
        }catch(e){document.getElementById('hf-tasks').innerHTML=''}
        }
        async function hfCancel(taskId){
        await fetch(ADMIN_BASE+':8081/admin/api/hf/cancel',{method:'POST',headers:{'Content-Type':'application/json',...authHeaders()},body:JSON.stringify({task_id:taskId})});
        loadHFTasks();
        }
        document.getElementById('hf-search').addEventListener('keydown',function(e){if(e.key==='Enter')hfSearch()});
        </script>
        </body>
        </html>
        """
    }
}
