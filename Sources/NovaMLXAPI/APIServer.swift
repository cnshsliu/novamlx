import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdRouter
import ImageIO
import Logging
import NovaMLXCore
import NovaMLXDB
import NovaMLXDistributed
import NovaMLXEngine
import NovaMLXInference
import NovaMLXMCP
import NovaMLXModelManager
import NovaMLXUtils

typealias AppContext = BasicRouterRequestContext

final class LockedCounter: @unchecked Sendable {
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

func unwrapAnyCodable(_ ac: AnyCodable) -> Any {
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
            var response = Self.jsonError(status: error.httpStatus, detail: detail)
            if let retryAfter = error.retryAfter {
                response.headers[.init("Retry-After")!] = "\(retryAfter)"
            }
            return response
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

    let config: NovaMLXConfiguration

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let keys: [APIKey]
        do {
            keys = try NovaDB.shared.apiKeyStore.listAsAPIKey()
        } catch {
            // Fail closed: never fall through to open mode on a DB error.
            let detail = OpenAIErrorDetail(
                message: "Key store unavailable.",
                type: "server_error",
                code: "key_store_unavailable"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .serviceUnavailable, detail: detail)
        }

        // No keys configured → open mode, no auth required
        if keys.isEmpty {
            return try await next(request, context)
        }

        let token = Self.extractToken(from: request)
        guard let token else {
            let detail = OpenAIErrorDetail(
                message: "Invalid or missing admin API key.",
                type: "authentication_error",
                code: "invalid_api_key"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .unauthorized, detail: detail)
        }

        // Structured key lookup via SQLite store
        if let key = (try? NovaDB.shared.apiKeyStore.findAPIKeyByRawToken(token)) ?? nil {
            guard key.isActive else {
                let detail = OpenAIErrorDetail(
                    message: "API key is disabled or expired.",
                    type: "authentication_error",
                    code: "key_inactive"
                )
                return NovaMLXErrorMiddleware.jsonError(status: .unauthorized, detail: detail)
            }
            return try await next(request, context)
        }

        let detail = OpenAIErrorDetail(
            message: "Invalid or missing admin API key.",
            type: "authentication_error",
            code: "invalid_api_key"
        )
        return NovaMLXErrorMiddleware.jsonError(status: .unauthorized, detail: detail)
    }

    private static func extractToken(from request: Request) -> String? {
        let authHeader = request.headers[.authorization]
        if let authHeader, authHeader.hasPrefix("Bearer ") {
            return String(authHeader.dropFirst(7))
        }
        return request.headers[fields: HTTPField.Name("x-admin-key")!].first?.value
    }
}

private struct APIKeyAuthMiddleware: RouterMiddleware {
    typealias Context = AppContext

    let config: NovaMLXConfiguration
    let globalRateLimiter: RateLimiter

    private static let publicPaths: Set<String> = ["/", "/chat", "/health", "/v1/models", "/v1/stats", "/favicon.ico"]
    private static let publicPrefixes: Set<String> = ["/v1/chat/history", "/admin/"]

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        let keys: [APIKey]
        do {
            keys = try NovaDB.shared.apiKeyStore.listAsAPIKey()
        } catch {
            // Fail closed: never fall through to open mode on a DB error.
            let detail = OpenAIErrorDetail(
                message: "Key store unavailable.",
                type: "server_error",
                code: "key_store_unavailable"
            )
            return NovaMLXErrorMiddleware.jsonError(status: .serviceUnavailable, detail: detail)
        }

        // No keys at all — open mode
        if keys.isEmpty {
            return try await next(request, context)
        }

        let path = request.uri.path
        if Self.publicPaths.contains(path) || Self.publicPrefixes.contains(where: { path.hasPrefix($0) }) {
            return try await next(request, context)
        }

        let token = Self.extractToken(from: request)
        guard let token else {
            return Self.unauthorized("Invalid or missing API key.")
        }

        // Structured key lookup via SQLite store
        if let key = (try? NovaDB.shared.apiKeyStore.findAPIKeyByRawToken(token)) ?? nil {
            // Check active
            guard key.isActive else {
                return Self.unauthorized("API key is disabled or expired.")
            }

            // Check daily limits
            let withinLimits = NovaDB.shared.apiKeyStore.isWithinLimits(keyId: key.id)
            guard withinLimits else {
                let detail = OpenAIErrorDetail(
                    message: "Daily usage limit exceeded for this API key.",
                    type: "rate_limit_error",
                    code: "daily_limit_exceeded"
                )
                return NovaMLXErrorMiddleware.jsonError(status: .tooManyRequests, detail: detail)
            }

            // Check endpoint access
            if let allowedEndpoints = key.allowedEndpoints {
                let allowed = allowedEndpoints.contains { path.hasPrefix($0) }
                guard allowed else {
                    return Self.unauthorized("This API key does not have access to endpoint: \(path)")
                }
            }

            // Check rate limit (per-key or global)
            let rateLimiter = key.rateLimitPerSecond.map { rps in
                RateLimiter(config: RateLimitConfig(
                    requestsPerSecond: rps,
                    burstSize: key.rateLimitBurst ?? 20
                ))
            } ?? globalRateLimiter

            let rateKey = "key:\(key.id)"
            guard rateLimiter.allow(key: rateKey) else {
                let detail = OpenAIErrorDetail(
                    message: "Rate limit exceeded.",
                    type: "rate_limit_error",
                    code: "rate_limit_exceeded"
                )
                return NovaMLXErrorMiddleware.jsonError(status: .tooManyRequests, detail: detail)
            }

            return try await next(request, context)
        }

        return Self.unauthorized("Invalid or missing API key.")
    }

    private static func extractToken(from request: Request) -> String? {
        let authHeader = request.headers[.authorization]
        if let authHeader, authHeader.hasPrefix("Bearer ") {
            return String(authHeader.dropFirst(7))
        }
        if let xApiKey = request.headers[HTTPField.Name("x-api-key")!] {
            return xApiKey
        }
        return nil
    }

    private static func unauthorized(_ message: String) -> Response {
        let detail = OpenAIErrorDetail(
            message: message,
            type: "authentication_error",
            code: "invalid_api_key"
        )
        return NovaMLXErrorMiddleware.jsonError(status: .unauthorized, detail: detail)
    }

    private static func extractBearerToken(from request: Request) -> String? {
        let authHeader = request.headers[.authorization]
        if let authHeader, authHeader.hasPrefix("Bearer ") {
            return String(authHeader.dropFirst(7))
        }
        return request.headers[HTTPField.Name("x-api-key")!]
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
        case .insufficientMemory: .serviceUnavailable
        case .modelNotLoaded: .notFound
        case .modelLoadInProgress: .serviceUnavailable
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
        case .insufficientMemory: "server_error"
        case .modelNotLoaded: "not_found_error"
        case .modelLoadInProgress: "server_error"
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
        case .insufficientMemory: "insufficient_memory"
        case .modelNotLoaded: "model_not_loaded"
        case .modelLoadInProgress: "model_load_in_progress"
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
    private let config: ServerConfig
    private let startTime = Date()
    private var coordinator: AutoLoadCoordinator?
    private let capabilitiesDetector = ModelCapabilitiesDetector()
    private let modelfileManager = ModelfileManager()

    public init(
        inferenceService: InferenceService,
        modelManager: ModelManager,
        embeddingService: EmbeddingService = EmbeddingService(),
        rerankerService: RerankerService = RerankerService(),
        mcpManager: MCPManager = MCPManager(),
        config: ServerConfig = ServerConfig(),
        huggingfaceEndpoint: String? = nil
    ) {
        self.inferenceService = inferenceService
        self.modelManager = modelManager
        self.embeddingService = embeddingService
        self.rerankerService = rerankerService
        self.mcpManager = mcpManager
        self.benchmarkService = BenchmarkService(inferenceService: inferenceService)
        self.perplexityService = PerplexityService(inferenceService: inferenceService)
        self.updateChecker = UpdateChecker()
        self.hfService = HuggingFaceService(modelDirectory: modelManager.modelsDirectory, endpoint: huggingfaceEndpoint)
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
        let modelfileMgr = self.modelfileManager

        // Auto-load coordinator — lazy init
        let coordinator = AutoLoadCoordinator(
            inference: inference,
            embeddings: embeddings,
            models: models,
            settings: inference.settingsManager,
            defaultTTLSeconds: cfg.autoLoad.defaultTTLSecondsAfterAutoLoad.map { Int($0) }
        )
        self.coordinator = coordinator

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
            APIKeyAuthMiddleware(config: NovaMLXConfiguration.shared, globalRateLimiter: rateLimiter)
            NovaMLXErrorMiddleware()
            Get("/v1/models") { request, context in
                let detector = self.capabilitiesDetector
                let modelList = models.downloadedModels()
                    .filter { inference.isModelLoaded($0.id) || embeddings.isLoaded($0.id) || inference.transcriptionService.isLoaded($0.id) || inference.ttsService.listLoadedModels().contains($0.id) || inference.imageGenerationService.isLoaded($0.id) }
                    .map { record -> OpenAIModel in
                        let caps = detector.capabilities(
                            for: record.id,
                            modelType: record.modelType,
                            localURL: record.localURL
                        )
                        return OpenAIModel(
                            id: record.id,
                            nova: OpenAIModelNova(capabilities: caps)
                        )
                    }

                let response = OpenAIModelsResponse(data: modelList)
                return try Self.jsonResponse(response)
            }
            Get("/v1/models/{id}") { request, context in
                let modelId = try context.parameters.require("id")
                let models = self.modelManager.downloadedModels()
                guard let record = models.first(where: { $0.id == modelId }) else {
                    return try Self.jsonResponse(["error": ["message": "Model not found: \(modelId)", "type": "invalid_request_error"]], httpStatus: .notFound)
                }
                let detector = self.capabilitiesDetector
                let caps = detector.capabilities(
                    for: record.id,
                    modelType: record.modelType,
                    localURL: record.localURL
                )
                let model = OpenAIModel(id: record.id, nova: OpenAIModelNova(capabilities: caps))
                return try Self.jsonResponse(model)
            }
            Post("/v1/chat/completions") { request, context in
                let body = try await request.body.collect(upTo: .max)
                var openAIReq = try JSONDecoder().decode(OpenAIRequest.self, from: body)

                // Modelfile resolution: if model name matches a modelfile, swap to base model
                var modelfileName: String? = nil
                var modelfileSystemPrompt: String? = nil
                if let resolved = modelfileMgr.resolve(openAIReq.model) {
                    modelfileName = resolved.modelfileName
                    modelfileSystemPrompt = resolved.systemPrompt
                    let mfParams = resolved.parameters
                    // Merge tools: modelfile tools first, then request tools
                    var mergedTools = resolved.tools?.map { tool in
                        tool.mapValues { anyToAnyCodable($0.toAny()) }
                    } ?? []
                    if let reqTools = openAIReq.tools { mergedTools.append(contentsOf: reqTools) }
                    openAIReq = OpenAIRequest(
                        model: resolved.baseModel,
                        messages: openAIReq.messages,
                        tools: mergedTools.isEmpty ? nil : mergedTools,
                        toolChoice: openAIReq.toolChoice,
                        temperature: openAIReq.temperature ?? mfParams?.temperature,
                        topP: openAIReq.topP ?? mfParams?.topP,
                        topK: openAIReq.topK ?? mfParams?.topK,
                        minP: openAIReq.minP ?? mfParams?.minP,
                        maxTokens: openAIReq.maxTokens ?? mfParams?.maxTokens,
                        stream: openAIReq.stream,
                        streamOptions: openAIReq.streamOptions,
                        stop: openAIReq.stop ?? mfParams?.stop,
                        n: openAIReq.n,
                        frequencyPenalty: openAIReq.frequencyPenalty ?? mfParams?.frequencyPenalty,
                        presencePenalty: openAIReq.presencePenalty ?? mfParams?.presencePenalty,
                        repetitionPenalty: openAIReq.repetitionPenalty ?? mfParams?.repetitionPenalty,
                        seed: openAIReq.seed ?? mfParams?.seed,
                        sessionId: openAIReq.sessionId,
                        responseFormat: openAIReq.responseFormat,
                        thinkingBudget: openAIReq.thinkingBudget,
                        enableThinking: openAIReq.enableThinking,
                        preserveThinking: openAIReq.preserveThinking,
                        chatTemplateKwargs: openAIReq.chatTemplateKwargs,
                        reasoningEffort: openAIReq.reasoningEffort,
                        logprobs: openAIReq.logprobs,
                        topLogprobs: openAIReq.topLogprobs,
                        keepAlive: openAIReq.keepAlive
                    )
                    NovaMLXLog.info("[API] Modelfile resolved: \(resolved.modelfileName) -> base=\(resolved.baseModel)")
                }

                var messages = mapOpenAIMessages(openAIReq.messages)

                // Modelfile system prompt injection
                if let sp = modelfileSystemPrompt {
                    messages.insert(ChatMessage(role: .system, content: sp), at: 0)
                }

                // OCR auto-optimization
                if OCROptimizer.isOCRModel(openAIReq.model) {
                    messages = OCROptimizer.applyPrompt(messages: messages, modelName: openAIReq.model)
                }

                // Tokenhub routing: model name is "tknet" or "tknet:<provider>"
                if TokenhubManager.shared.isTokenhubModel(openAIReq.model) {
                    return try await Self.handleTokenhubPassthrough(
                        modelName: openAIReq.model, rawBody: Data(buffer: body),
                        path: "chat/completions", inference: inference,
                        tag: openAIReq.tag
                    )
                }

                let loadOutcome = try await Self.ensureModelReady(
                    modelId: openAIReq.model, isStreaming: openAIReq.stream ?? false,
                    cfg: cfg, inference: inference, embeddings: embeddings,
                    coordinator: coordinator, request: request
                )

                // Also ensure draft model is loaded if specified (speculative decoding)
                if let draftModelId = openAIReq.draftModel, !draftModelId.isEmpty {
                    _ = try await Self.ensureModelReady(
                        modelId: draftModelId, isStreaming: openAIReq.stream ?? false,
                        cfg: cfg, inference: inference, embeddings: embeddings,
                        coordinator: coordinator, request: request
                    )
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

                let clientType = ClientDetector.detect(request: request)
                var response: Response
                if openAIReq.stream ?? false {
                    response = try await Self.handleStreamChat(
                        openAIReq: openAIReq, messages: messages, inference: inference,
                        sessionId: sessionId, responseFormat: responseFormat, jsonSchemaDef: jsonSchemaDef,
                        regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
                        cfg: cfg, clientType: clientType,
                        coordinator: coordinator,
                        responseModelOverride: modelfileName
                    )
                } else {
                    response = try await Self.handleChat(
                        openAIReq: openAIReq, messages: messages, inference: inference,
                        sessionId: sessionId, responseFormat: responseFormat, jsonSchemaDef: jsonSchemaDef,
                        regexPattern: regexPattern, gbnfGrammar: gbnfGrammar,
                        cfg: cfg, clientType: clientType,
                        responseModelOverride: modelfileName
                    )
                    Self.applyKeepAlive(openAIReq.keepAlive, modelId: openAIReq.model, pool: inference.engine.pool)
                }
                if case .justLoaded(let ms) = loadOutcome {
                    response.headers[.init("X-Model-Cold-Load")!] = "true"
                    response.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                }
                if openAIReq.stream != true {
                    // Non-streaming usage tracked via handleChat result
                }
                return response
            }
            Post("/v1/messages") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let httpRequest = request  // capture before shadowing
                let anthropicReq = try JSONDecoder().decode(AnthropicRequest.self, from: body)

                NovaMLXLog.info("[API] POST /v1/messages — model=\(anthropicReq.model), stream=\(anthropicReq.stream ?? false), maxTokens=\(anthropicReq.maxTokens), msgs=\(anthropicReq.messages.count)")

                var messages: [ChatMessage]
                do {
                    messages = try mapAnthropicMessages(anthropicReq.messages, system: anthropicReq.system)
                } catch let err as AnthropicMappingError {
                    throw NovaMLXError.apiError(String(describing: err))
                }

                // Tokenhub routing for Anthropic requests
                if TokenhubManager.shared.isTokenhubModel(anthropicReq.model) {
                    let rawDict = try JSONSerialization.jsonObject(with: Data(buffer: body)) as? [String: Any]
                    let tag = rawDict?["tag"] as? String
                    return try await Self.handleTokenhubPassthrough(
                        modelName: anthropicReq.model, rawBody: Data(buffer: body),
                        path: "messages", inference: inference,
                        tag: tag
                    )
                }

                let anthropicLoadOutcome = try await Self.ensureModelReady(
                    modelId: anthropicReq.model, isStreaming: anthropicReq.stream ?? false,
                    cfg: cfg, inference: inference, embeddings: embeddings,
                    coordinator: coordinator, request: request
                )

                let anthropicClientType = ClientDetector.detect(request: request)
                if anthropicReq.stream ?? false {
                    var streamResp = try await Self.handleStreamAnthropic(
                        anthropicReq: anthropicReq, messages: messages, inference: inference,
                        cfg: cfg, clientType: anthropicClientType,
                        coordinator: coordinator
                    )
                    if case .justLoaded(let ms) = anthropicLoadOutcome {
                        streamResp.headers[.init("X-Model-Cold-Load")!] = "true"
                        streamResp.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                    }
                    return streamResp
                }

                let ocrSampling = OCROptimizer.samplingOverrides(
                    modelName: anthropicReq.model,
                    userTemperature: anthropicReq.temperature,
                    userMaxTokens: anthropicReq.maxTokens,
                    userRepetitionPenalty: nil
                )
                let ocrStop = OCROptimizer.applyStopSequences(anthropicReq.stopSequences, modelName: anthropicReq.model)

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
                let ctxWin = inference.getContextWindow(for: anthropicReq.model) ?? 0
                let scaledInput = anthropicClientType.shouldScaleContext
                    ? cfg.scaleTokenCount(result.promptTokens, modelContextWindow: ctxWin)
                    : result.promptTokens
                let scaledOutput = anthropicClientType.shouldScaleContext
                    ? cfg.scaleTokenCount(result.completionTokens, modelContextWindow: ctxWin)
                    : result.completionTokens

                var content: [AnthropicContentBlock] = []
                // Scrub control tokens and parse thinking
                let shouldParseThinking = anthropicReq.resolvedEnableThinking != false
                var scrubbedText = result.text
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
                if shouldParseThinking {
                    let isAnthropicImplicit = ModelContainer.isImplicitThinkingModel(for: anthropicReq.model)
                    let thinkingParser = ThinkingParser(expectImplicitThinking: isAnthropicImplicit)
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
                    if !cleanThinking.isEmpty {
                        content.append(AnthropicContentBlock(type: "thinking", thinking: cleanThinking))
                    }
                    if !cleanResponse.isEmpty {
                        content.append(AnthropicContentBlock(text: cleanResponse))
                    }
                } else {
                    // enable_thinking=false — all output is content, no thinking
                    if !scrubbedText.isEmpty {
                        content.append(AnthropicContentBlock(text: scrubbedText))
                    }
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
                var httpResponse = try Self.jsonResponse(response)
                if case .justLoaded(let ms) = anthropicLoadOutcome {
                    httpResponse.headers[.init("X-Model-Cold-Load")!] = "true"
                    httpResponse.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                }
                Self.applyKeepAlive(anthropicReq.keepAlive, modelId: anthropicReq.model, pool: inference.engine.pool)
                Self.recordTokenUsage(request: httpRequest, promptTokens: result.promptTokens, completionTokens: result.completionTokens, model: anthropicReq.model)
                return httpResponse
            }
            Post("/v1/messages/count_tokens") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AnthropicTokenCountRequest.self, from: body)

                if !inference.isModelLoaded(req.model) {
                    throw NovaMLXError.modelNotFound(req.model)
                }

                let messages: [ChatMessage]
                do {
                    messages = try mapAnthropicMessages(req.messages, system: req.system)
                } catch let err as AnthropicMappingError {
                    throw NovaMLXError.apiError(String(describing: err))
                }

                guard let tokenCount = inference.countTokens(model: req.model, messages: messages) else {
                    throw NovaMLXError.inferenceFailed("Failed to count tokens: model tokenizer not available")
                }

                let ctClientType = ClientDetector.detect(request: request)
                let ctCtxWin = inference.getContextWindow(for: req.model) ?? 0
                let scaledCount = ctClientType.shouldScaleContext
                    ? cfg.scaleTokenCount(tokenCount, modelContextWindow: ctCtxWin) : tokenCount

                let response = AnthropicTokenCountResponse(inputTokens: scaledCount)
                return try Self.jsonResponse(response)
            }
            Post("/v1/completions") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let compReq = try JSONDecoder().decode(OpenAICompletionRequest.self, from: body)

                let compLoadOutcome = try await Self.ensureModelReady(
                    modelId: compReq.model, isStreaming: compReq.stream ?? false,
                    cfg: cfg, inference: inference, embeddings: embeddings,
                    coordinator: coordinator, request: request
                )

                let compClientType = ClientDetector.detect(request: request)
                var compResponse: Response
                if compReq.stream ?? false {
                    compResponse = try await Self.handleStreamCompletion(
                        compReq: compReq, inference: inference,
                        cfg: cfg, clientType: compClientType,
                        coordinator: coordinator
                    )
                } else {
                    compResponse = try await Self.handleCompletion(
                        compReq: compReq, inference: inference,
                        cfg: cfg, clientType: compClientType
                    )
                    Self.applyKeepAlive(compReq.keepAlive, modelId: compReq.model, pool: inference.engine.pool)
                }
                if case .justLoaded(let ms) = compLoadOutcome {
                    compResponse.headers[.init("X-Model-Cold-Load")!] = "true"
                    compResponse.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                }
                return compResponse
            }
            Post("/v1/embeddings") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let embReq = try JSONDecoder().decode(OpenAIEmbeddingRequest.self, from: body)

                let embLoadOutcome = try await Self.ensureModelReady(
                    modelId: embReq.model, isStreaming: false,
                    cfg: cfg, inference: inference, embeddings: embeddings,
                    coordinator: coordinator, request: request
                )

                let texts = embReq.input.texts
                let result = try await embeddings.embed(model: embReq.model, inputs: texts)

                let clientType = ClientDetector.detect(request: request)
                let ctxWin = inference.getContextWindow(for: embReq.model) ?? 0
                let scaledPrompt = clientType.shouldScaleContext
                    ? cfg.scaleTokenCount(result.promptTokens, modelContextWindow: ctxWin) : result.promptTokens
                let scaledTotal = clientType.shouldScaleContext
                    ? cfg.scaleTokenCount(result.totalTokens, modelContextWindow: ctxWin) : result.totalTokens

                let data = result.embeddings.enumerated().map { idx, vec in
                    OpenAIEmbeddingData(index: idx, embedding: vec)
                }
                let response = OpenAIEmbeddingResponse(
                    model: result.model,
                    data: data,
                    usage: OpenAIEmbeddingUsage(
                        promptTokens: scaledPrompt,
                        totalTokens: scaledTotal
                    )
                )
                var embHttpResponse = try Self.jsonResponse(response)
                if case .justLoaded(let ms) = embLoadOutcome {
                    embHttpResponse.headers[.init("X-Model-Cold-Load")!] = "true"
                    embHttpResponse.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                }
                Self.applyKeepAlive(embReq.keepAlive, modelId: embReq.model, pool: inference.engine.pool)
                return embHttpResponse
            }
            Post("/v1/audio/transcriptions") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let contentType = request.headers[fields: HTTPField.Name("content-type")!].first?.value ?? ""

                let model: String
                let audioData: Data
                let language: String?
                let responseFormat: String
                let stream: Bool

                if contentType.lowercased().contains("multipart/form-data") {
                    let parts = try MultipartParser.parse(body: Data(body.readableBytesView), contentType: contentType)
                    guard let filePart = parts["file"] else {
                        throw NovaMLXError.apiError("Missing 'file' part in multipart upload")
                    }
                    audioData = filePart.body
                    model = parts["model"].flatMap { String(data: $0.body, encoding: .utf8) } ?? ""
                    language = parts["language"].flatMap { String(data: $0.body, encoding: .utf8) }
                    responseFormat = parts["response_format"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "json"
                    stream = parts["stream"].flatMap { String(data: $0.body, encoding: .utf8) } == "true"
                } else {
                    let req = try JSONDecoder().decode(TranscriptionRequest.self, from: body)
                    guard let data = Data(base64Encoded: req.file) else {
                        throw NovaMLXError.apiError("Invalid base64 audio data in 'file' field")
                    }
                    audioData = data
                    model = req.model
                    language = req.language
                    responseFormat = req.resolvedResponseFormat
                    stream = req.stream ?? false
                }

                if stream {
                    let tokenStream = inference.transcriptionService.transcribeStream(
                        modelId: model,
                        audioData: audioData,
                        language: language
                    )
                    return Response(
                        status: .ok,
                        headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive"],
                        body: AudioSSEStream.body(from: tokenStream)
                    )
                }

                let result = try await inference.transcriptionService.transcribe(
                    modelId: model,
                    audioData: audioData,
                    language: language,
                    responseFormat: responseFormat
                )

                switch responseFormat {
                case "text":
                    let textData = result.text.data(using: .utf8) ?? Data()
                    return Response(status: .ok, headers: [.contentType: "text/plain"],
                                    body: .init(byteBuffer: ByteBuffer(data: textData)))
                default:
                    let response = TranscriptionResponse(
                        text: result.text,
                        language: result.language,
                        duration: result.duration
                    )
                    return try Self.jsonResponse(response)
                }
            }

            // MARK: - Text-to-Speech
            Post("/v1/audio/speech") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let contentType = request.headers[fields: HTTPField.Name("content-type")!].first?.value ?? ""

                let input: String
                let model: String
                let voice: String
                let responseFormat: String
                let speed: Float
                var voiceProfile: VoiceProfile? = nil

                if contentType.lowercased().contains("multipart/form-data") {
                    let parts = try MultipartParser.parse(body: Data(body.readableBytesView), contentType: contentType)
                    guard let inputPart = parts["input"] else {
                        throw NovaMLXError.apiError("Missing 'input' part in multipart upload")
                    }
                    input = String(data: inputPart.body, encoding: .utf8) ?? ""
                    model = parts["model"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "tts"
                    voice = parts["voice"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "Tingting"
                    responseFormat = parts["response_format"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "wav"
                    speed = parts["speed"].flatMap { Float(String(data: $0.body, encoding: .utf8) ?? "") } ?? 1.0
                } else {
                    let req = try JSONDecoder().decode(TTSRequest.self, from: body)
                    input = req.input
                    model = req.model
                    voice = req.voice ?? "Tingting"
                    responseFormat = req.resolvedResponseFormat
                    speed = req.speed ?? 1.0
                }

                guard !input.isEmpty else {
                    throw NovaMLXError.apiError("'input' is required and must be non-empty")
                }

                // Resolve voice profile by name
                let profiles = VoiceProfileManager.shared.listProfiles()
                voiceProfile = profiles.first { $0.name.lowercased() == voice.lowercased() }

                let rate = Int(175.0 * Double(speed))

                let audioData = try await inference.ttsService.synthesize(
                    text: input,
                    voice: voice,
                    rate: rate,
                    engine: nil,
                    voiceProfile: voiceProfile
                )

                let mimeType = self.mimeType(forFormat: responseFormat)
                var headers: HTTPFields = [.contentType: mimeType]

                return Response(
                    status: .ok,
                    headers: headers,
                    body: .init(byteBuffer: ByteBuffer(data: audioData))
                )
            }

            Post("/v1/images/generations") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(ImageGenerationRequest.self, from: body)

                guard !req.prompt.isEmpty else {
                    throw NovaMLXError.apiError("'prompt' is required and must be non-empty")
                }

                let model = req.model
                let n = req.resolvedN
                let (width, height) = req.resolvedSize
                let format = req.resolvedResponseFormat

                // Auto-load check
                if !inference.imageGenerationService.isLoaded(model) {
                    guard let record = models.downloadedModels().first(where: { $0.id == model }) else {
                        throw NovaMLXError.modelNotFound(model)
                    }
                    guard record.modelType == .image else {
                        throw NovaMLXError.unsupportedModel("Model '\(model)' is not an image generation model (type: \(record.modelType))")
                    }
                    let config = ModelConfig(identifier: ModelIdentifier(id: model, family: record.family), modelType: .image)
                    _ = try await inference.imageGenerationService.loadModel(
                        from: record.localURL, config: config)
                }

                let result = try await inference.imageGenerationService.generate(
                    modelId: model,
                    prompt: req.prompt,
                    negativePrompt: req.negativePrompt ?? "",
                    n: n,
                    width: width,
                    height: height,
                    seed: req.seed.map { UInt64($0) },
                    steps: req.steps
                )

                let imageData: [ImageData]
                switch format {
                case "url":
                    // Write to temp file and return URL
                    imageData = try result.images.enumerated().map { (i, b64) in
                        guard let data = Data(base64Encoded: b64) else {
                            throw NovaMLXError.apiError("Failed to decode generated image")
                        }
                        let tempDir = FileManager.default.temporaryDirectory
                        let url = tempDir.appendingPathComponent("novamlx_img_\(Int(Date().timeIntervalSince1970))_\(i).png")
                        try data.write(to: url)
                        return ImageData(b64Json: nil, url: url.absoluteString, revisedPrompt: req.prompt)
                    }
                default:
                    imageData = result.images.map { b64 in
                        ImageData(b64Json: b64, url: nil, revisedPrompt: req.prompt)
                    }
                }

                let response = ImageGenerationResponse(
                    created: Int(Date().timeIntervalSince1970),
                    data: imageData,
                    model: model
                )
                return try Self.jsonResponse(response)
            }

            // MARK: - Image Edit

            Post("/v1/images/edits") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let contentType = request.headers[fields: HTTPField.Name("content-type")!].first?.value ?? ""

                guard contentType.lowercased().contains("multipart/form-data") else {
                    throw NovaMLXError.apiError("Content-Type must be multipart/form-data for image edits")
                }

                let parts = try MultipartParser.parse(body: Data(body.readableBytesView), contentType: contentType)

                guard let imagePart = parts["image"] else {
                    throw NovaMLXError.apiError("Missing 'image' part in multipart upload")
                }
                guard let promptPart = parts["prompt"] else {
                    throw NovaMLXError.apiError("Missing 'prompt' part in multipart upload")
                }

                let imageData = imagePart.body
                let maskData = parts["mask"]?.body
                let prompt = String(data: promptPart.body, encoding: .utf8) ?? ""
                let model = parts["model"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "sdxl-turbo"
                let n = Int(parts["n"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "1") ?? 1
                let size = parts["size"].flatMap { String(data: $0.body, encoding: .utf8) }
                let responseFormat = parts["response_format"].flatMap { String(data: $0.body, encoding: .utf8) }

                guard !prompt.isEmpty else {
                    throw NovaMLXError.apiError("'prompt' is required and must be non-empty")
                }

                let (width, height) = ImageEditRequest(
                    image: imageData, mask: maskData, prompt: prompt, model: model,
                    n: n, size: size, responseFormat: responseFormat
                ).resolvedSize
                let format = responseFormat ?? "b64_json"
                let resolvedN = min(max(n, 1), 4)

                // Convert uploaded image to CGImage
                guard let inputCGImage = Self.dataToCGImage(imageData) else {
                    throw NovaMLXError.apiError("Failed to decode input image")
                }
                let maskCGImage = maskData.flatMap { Self.dataToCGImage($0) }

                // Auto-load check
                if !inference.imageGenerationService.isLoaded(model) {
                    guard let record = models.downloadedModels().first(where: { $0.id == model }) else {
                        throw NovaMLXError.modelNotFound(model)
                    }
                    guard record.modelType == .image else {
                        throw NovaMLXError.unsupportedModel("Model '\(model)' is not an image generation model")
                    }
                    let config = ModelConfig(identifier: ModelIdentifier(id: model, family: record.family), modelType: .image)
                    _ = try await inference.imageGenerationService.loadModel(from: record.localURL, config: config)
                }

                let result = try await inference.imageGenerationService.edit(
                    modelId: model,
                    image: inputCGImage,
                    mask: maskCGImage,
                    prompt: prompt,
                    n: resolvedN,
                    width: width,
                    height: height
                )

                let imageDataResp: [ImageData]
                switch format {
                case "url":
                    imageDataResp = try result.images.enumerated().map { (i, b64) in
                        guard let data = Data(base64Encoded: b64) else {
                            throw NovaMLXError.apiError("Failed to decode generated image")
                        }
                        let url = FileManager.default.temporaryDirectory
                            .appendingPathComponent("novamlx_edit_\(Int(Date().timeIntervalSince1970))_\(i).png")
                        try data.write(to: url)
                        return ImageData(b64Json: nil, url: url.absoluteString, revisedPrompt: prompt)
                    }
                default:
                    imageDataResp = result.images.map { b64 in
                        ImageData(b64Json: b64, url: nil, revisedPrompt: prompt)
                    }
                }

                let response = ImageGenerationResponse(
                    created: Int(Date().timeIntervalSince1970),
                    data: imageDataResp,
                    model: model
                )
                return try Self.jsonResponse(response)
            }

            // MARK: - Image Variation

            Post("/v1/images/variations") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let contentType = request.headers[fields: HTTPField.Name("content-type")!].first?.value ?? ""

                guard contentType.lowercased().contains("multipart/form-data") else {
                    throw NovaMLXError.apiError("Content-Type must be multipart/form-data for image variations")
                }

                let parts = try MultipartParser.parse(body: Data(body.readableBytesView), contentType: contentType)

                guard let imagePart = parts["image"] else {
                    throw NovaMLXError.apiError("Missing 'image' part in multipart upload")
                }

                let imageData = imagePart.body
                let model = parts["model"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "sdxl-turbo"
                let n = Int(parts["n"].flatMap { String(data: $0.body, encoding: .utf8) } ?? "1") ?? 1
                let size = parts["size"].flatMap { String(data: $0.body, encoding: .utf8) }
                let responseFormat = parts["response_format"].flatMap { String(data: $0.body, encoding: .utf8) }

                let (width, height) = ImageVariationRequest(
                    image: imageData, model: model, n: n, size: size, responseFormat: responseFormat
                ).resolvedSize
                let format = responseFormat ?? "b64_json"
                let resolvedN = min(max(n, 1), 4)

                // Convert uploaded image to CGImage
                guard let inputCGImage = Self.dataToCGImage(imageData) else {
                    throw NovaMLXError.apiError("Failed to decode input image")
                }

                // Auto-load check
                if !inference.imageGenerationService.isLoaded(model) {
                    guard let record = models.downloadedModels().first(where: { $0.id == model }) else {
                        throw NovaMLXError.modelNotFound(model)
                    }
                    guard record.modelType == .image else {
                        throw NovaMLXError.unsupportedModel("Model '\(model)' is not an image generation model")
                    }
                    let config = ModelConfig(identifier: ModelIdentifier(id: model, family: record.family), modelType: .image)
                    _ = try await inference.imageGenerationService.loadModel(from: record.localURL, config: config)
                }

                let result = try await inference.imageGenerationService.variation(
                    modelId: model,
                    image: inputCGImage,
                    n: resolvedN,
                    width: width,
                    height: height
                )

                let imageDataResp: [ImageData]
                switch format {
                case "url":
                    imageDataResp = try result.images.enumerated().map { (i, b64) in
                        guard let data = Data(base64Encoded: b64) else {
                            throw NovaMLXError.apiError("Failed to decode generated image")
                        }
                        let url = FileManager.default.temporaryDirectory
                            .appendingPathComponent("novamlx_var_\(Int(Date().timeIntervalSince1970))_\(i).png")
                        try data.write(to: url)
                        return ImageData(b64Json: nil, url: url.absoluteString, revisedPrompt: nil)
                    }
                default:
                    imageDataResp = result.images.map { b64 in
                        ImageData(b64Json: b64, url: nil, revisedPrompt: nil)
                    }
                }

                let response = ImageGenerationResponse(
                    created: Int(Date().timeIntervalSince1970),
                    data: imageDataResp,
                    model: model
                )
                return try Self.jsonResponse(response)
            }

            Post("/v1/responses") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let rawBody = Data(buffer: body)
                // DEBUG: dump raw body BEFORE decode to catch decode failures
                let debugPreURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("tokenhub_pre_decode.json")
                try? rawBody.write(to: debugPreURL)
                NovaMLXLog.info("[Tokenhub/Responses] PRE-DECODE dump to \(debugPreURL.path)")

                let req: OpenAIResponseRequest
                do {
                    req = try JSONDecoder().decode(OpenAIResponseRequest.self, from: body)
                } catch {
                    NovaMLXLog.error("[Tokenhub/Responses] DECODE FAILED: \(error)")
                    throw error
                }

                // Tokenhub passthrough: convert Responses API → Chat Completions, forward, convert back
                if TokenhubManager.shared.isTokenhubModel(req.model) {
                    return try await Self.handleTokenhubResponsesPassthrough(
                        req: req, rawBody: rawBody, inference: inference
                    )
                }

                let isStreaming = req.stream ?? false
                let respLoadOutcome = try await Self.ensureModelReady(
                    modelId: req.model, isStreaming: isStreaming,
                    cfg: cfg, inference: inference, embeddings: embeddings,
                    coordinator: coordinator, request: request
                )

                let respClientType = ClientDetector.detect(request: request)
                if isStreaming {
                    var respResponse = try await Self.handleStreamResponses(
                        req: req, inference: inference, cfg: cfg,
                        clientType: respClientType, coordinator: coordinator
                    )
                    if case .justLoaded(let ms) = respLoadOutcome {
                        respResponse.headers[.init("X-Model-Cold-Load")!] = "true"
                        respResponse.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                    }
                    return respResponse
                } else {
                    var respResponse = try await Self.handleResponsesRequest(
                        req: req, inference: inference, cfg: cfg,
                        clientType: respClientType, coordinator: coordinator
                    )
                    Self.applyKeepAlive(req.keepAlive, modelId: req.model, pool: inference.engine.pool)
                    if case .justLoaded(let ms) = respLoadOutcome {
                        respResponse.headers[.init("X-Model-Cold-Load")!] = "true"
                        respResponse.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
                    }
                    return respResponse
                }
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
            Post("/v1/responses/{id}/cancel") { request, context in
                let responseId = try context.parameters.require("id")
                guard ResponseStore.shared.get(responseId) != nil else {
                    throw NovaMLXError.modelNotFound("Response \(responseId) not found")
                }
                // Local inference doesn't have a cancel API yet; return success for spec compat
                let data = try JSONSerialization.data(withJSONObject: ["id": responseId, "status": "cancelled"])
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            Get("/v1/responses/{id}/input_items") { request, context in
                let responseId = try context.parameters.require("id")
                guard let response = ResponseStore.shared.get(responseId) else {
                    throw NovaMLXError.modelNotFound("Response \(responseId) not found")
                }
                // Extract user input items from stored response
                var items: [[String: Any]] = []
                for item in response.output {
                    if case .message(let msg) = item, msg.role == "user" {
                        for content in msg.content {
                            items.append(["type": "message", "role": "user", "content": content.text])
                        }
                    }
                }
                let body: [String: Any] = ["object": "list", "data": items]
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            Post("/v1/responses/compact") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(CompactRequest.self, from: body)
                return try await Self.handleCompactRequest(req: req, inference: inference)
            }
            Post("/v1/responses/input_tokens") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(InputTokensRequest.self, from: body)
                return try await Self.handleInputTokensRequest(req: req, inference: inference)
            }
            Get("/health") { _, _ in
                let stats = inference.stats
                let mcpStatuses = mcp.getServerStatuses()
                let uptime = Date().timeIntervalSince(self.startTime)
                let diskFree = (try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())[.systemFreeSize] as? Int64) ?? 0
                let diskTotal = (try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())[.systemSize] as? Int64) ?? 0
                let body: [String: Any] = [
                    "status": "ok",
                    "loadedModels": stats.loadedModels,
                    "gpuMemoryUsed": stats.gpuMemoryUsed,
                    "uptime": Int(uptime),
                    "diskUsage": diskTotal - diskFree,
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

                var memoryEnforcerInfo: [String: Any] = ["enabled": false]
                if let enforcer = inference.engine.memoryEnforcer {
                    let ms = await enforcer.status
                    memoryEnforcerInfo = [
                        "enabled": ms.enabled,
                        "softLimitBytes": ms.softLimitBytes,
                        "hardLimitBytes": ms.hardLimitBytes,
                        "currentBytes": ms.currentBytes,
                        "utilization": ms.utilization,
                        "totalEvictions": ms.totalEvictions,
                    ]
                }

                let sysStats = SystemMonitor.shared.currentStats(
                    activeRequests: stats.activeRequests,
                    tokensPerSecond: sessionMetrics.averageTokensPerSecond,
                    gpuMemoryUsed: stats.gpuMemoryUsed
                )
                let body: [String: Any] = [
                    "session": [
                        "loadedModels": stats.loadedModels,
                        "activeRequests": stats.activeRequests,
                        "gpuMemoryUsed": stats.gpuMemoryUsed,
                        "totalRequests": sessionMetrics.totalRequests,
                        "totalTokens": sessionMetrics.totalTokensGenerated,
                        "averageTokensPerSecond": sessionMetrics.averageTokensPerSecond,
                        "recentTokensPerSecond": inference.engine.metricsStore.recentTokensPerSecond,
                        "cpuUsage": sysStats.cpuUsage,
                        "memoryUsed": sysStats.memoryUsed,
                        "memoryTotal": sysStats.memoryTotal,
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
                    "memoryEnforcer": memoryEnforcerInfo,
                ]
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            // SPA root — serves the full web UI
            Get("/") { _, _ in
                let html = WebUIBuilder.render()
                return Response(
                    status: .ok,
                    headers: [.contentType: "text/html"],
                    body: .init(byteBuffer: ByteBuffer(string: html))
                )
            }

            Get("/favicon.ico") { _, _ in
                let svg = "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%238b5cf6'/><text x='16' y='23' text-anchor='middle' font-size='20' fill='white' font-family='sans-serif' font-weight='bold'>N</text></svg>"
                return Response(
                    status: .ok,
                    headers: [.contentType: "image/svg+xml"],
                    body: .init(byteBuffer: ByteBuffer(string: svg))
                )
            }

            // Admin API proxy — forwards /admin/* to the admin server on localhost:{adminPort}
            // This lets the web UI (on port 6590) reach admin endpoints without CORS issues
            Get("/admin/**") { request, context in
                let path = "/admin/" + context.parameters.getCatchAll().joined(separator: "/")
                let query = request.uri.query ?? ""
                let fullPath = query.isEmpty ? path : path + "?" + query
                return try await Self.proxyAdminRequest(path: fullPath, method: "GET", body: nil, cfg: cfg)
            }
            Post("/admin/**") { request, context in
                let path = "/admin/" + context.parameters.getCatchAll().joined(separator: "/")
                let body = try await request.body.collect(upTo: .max)
                return try await Self.proxyAdminRequest(path: path, method: "POST", body: body, cfg: cfg)
            }
            Put("/admin/**") { request, context in
                let path = "/admin/" + context.parameters.getCatchAll().joined(separator: "/")
                let body = try await request.body.collect(upTo: .max)
                return try await Self.proxyAdminRequest(path: path, method: "PUT", body: body, cfg: cfg)
            }
            Delete("/admin/**") { request, context in
                let path = "/admin/" + context.parameters.getCatchAll().joined(separator: "/")
                let query = request.uri.query ?? ""
                let fullPath = query.isEmpty ? path : path + "?" + query
                return try await Self.proxyAdminRequest(path: fullPath, method: "DELETE", body: nil, cfg: cfg)
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

                    let rrClientType = ClientDetector.detect(request: request)
                    let rrCtxWin = inference.getContextWindow(for: rerankReq.model) ?? 0
                    let rawTotal = results.first?.totalTokens ?? 0
                    let scaledTotal = rrClientType.shouldScaleContext
                        ? cfg.scaleTokenCount(rawTotal, modelContextWindow: rrCtxWin) : rawTotal
                    let returnDocs = rerankReq.returnDocuments ?? true
                    let rerankResp = RerankResponse(
                        id: "rerank-\(UUID().uuidString.prefix(8))",
                        results: results.map { r in
                            let doc: RerankDocument? = returnDocs ? rerankReq.documents[r.index] : nil
                            return RerankResult(index: r.index, relevanceScore: r.score, document: doc)
                        },
                        model: rerankReq.model,
                        usage: RerankUsage(totalTokens: scaledTotal)
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
            AdminAuthMiddleware(config: NovaMLXConfiguration.shared)
            securityHeaders
            requestSizeLimit
            NovaMLXErrorMiddleware()
            Get("/admin/models") { request, context in
                let records = models.allRegisteredModels()
                var statuses: [AdminModelStatus] = []
                statuses.reserveCapacity(records.count)
                for record in records {
                    let isDownloaded = models.isDownloaded(record.id)
                    let isLoaded = inference.isModelLoaded(record.id) || embeddings.isLoaded(record.id) || inference.transcriptionService.isLoaded(record.id) || inference.imageGenerationService.isLoaded(record.id)
                    // Only check feasibility for downloaded, non-loaded, non-embedding models
                    var feasibility: MemoryFeasibility? = nil
                    if isDownloaded && !isLoaded && record.modelType != .embedding {
                        feasibility = await inference.checkMemoryFeasibility(
                            modelId: record.id, sizeBytes: record.sizeBytes, localURL: record.localURL
                        )
                    }

                    // SpecBoost status
                    var specBoost: SpecBoostInfo? = nil
                    if record.modelType == .llm {
                        let isHybrid: Bool
                        if isLoaded {
                            isHybrid = inference.isHybridModel(record.id)
                        } else {
                            isHybrid = false
                        }
                        let status = DraftModelRegistry.shared.boostStatus(
                            family: record.family,
                            isHybrid: isHybrid,
                            modelType: record.modelType,
                            draftModelLoaded: { id in inference.isModelLoaded(id) },
                            draftModelOnDisk: { id in models.isDownloaded(id) }
                        )
                        switch status {
                        case .ineligible(let reason):
                            specBoost = SpecBoostInfo(status: "ineligible", reason: reason)
                        case .eligible(let candidate):
                            specBoost = SpecBoostInfo(
                                status: "eligible",
                                draftModelId: candidate.draftModelId,
                                draftDisplayName: candidate.displayName,
                                draftDownloaded: models.isDownloaded(candidate.draftModelId),
                                draftLoaded: inference.engine.getContainer(for: candidate.draftModelId)?.isLoaded == true
                            )
                        case .active(let draftModelId):
                            specBoost = SpecBoostInfo(
                                status: "active",
                                draftModelId: draftModelId,
                                draftLoaded: true
                            )
                        }
                    }

                    statuses.append(AdminModelStatus(
                        id: record.id,
                        family: record.family.rawValue,
                        downloaded: isDownloaded,
                        loaded: isLoaded,
                        sizeBytes: record.sizeBytes,
                        downloadedAt: record.downloadedAt,
                        memoryFeasibility: feasibility,
                        specBoost: specBoost
                    ))
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
                        downloaded: true, loaded: inference.isModelLoaded(req.modelId) || embeddings.isLoaded(req.modelId) || inference.transcriptionService.isLoaded(req.modelId) || inference.imageGenerationService.isLoaded(req.modelId),
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

                if inference.isModelLoaded(req.modelId) || embeddings.isLoaded(req.modelId) || inference.transcriptionService.isLoaded(req.modelId) || inference.imageGenerationService.isLoaded(req.modelId) {
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
                } else if record.modelType == .audio {
                    if record.family == .dotsTts || record.family == .qwen3Tts {
                        try await inference.ttsService.loadModel(from: record.localURL)
                    } else {
                        _ = try await inference.transcriptionService.loadModel(from: record.localURL, config: config)
                    }
                    inference.saveLoadedModelsList()
                } else if record.modelType == .image {
                    _ = try await inference.imageGenerationService.loadModel(from: record.localURL, config: config)
                    inference.saveLoadedModelsList()
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

                guard inference.isModelLoaded(req.modelId) || embeddings.isLoaded(req.modelId) || inference.transcriptionService.isLoaded(req.modelId) || inference.imageGenerationService.isLoaded(req.modelId) else {
                    throw NovaMLXError.modelNotFound(req.modelId)
                }

                let record = models.getRecord(req.modelId)

                if inference.isModelLoaded(req.modelId) {
                    await inference.unloadModel(ModelIdentifier(id: req.modelId, family: record?.family ?? .other))
                }

                if embeddings.isLoaded(req.modelId) {
                    embeddings.unloadModel(req.modelId)
                }

                if inference.transcriptionService.isLoaded(req.modelId) {
                    inference.transcriptionService.unload(modelId: req.modelId)
                    inference.saveLoadedModelsList()
                }

                if inference.imageGenerationService.isLoaded(req.modelId) {
                    inference.imageGenerationService.unload(modelId: req.modelId)
                    inference.saveLoadedModelsList()
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
            // MARK: - Speed Boost (Draft Model)
            Post("/admin/models/boost/download") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdminLoadRequest.self, from: body)
                let modelId = req.modelId
                guard let record = models.getRecord(modelId) else {
                    throw NovaMLXError.modelNotFound(modelId)
                }
                let isHybrid = inference.isModelLoaded(modelId) && inference.isHybridModel(modelId)
                guard let candidate = DraftModelRegistry.shared.recommendation(
                    family: record.family, isHybrid: isHybrid
                ) else {
                    throw NovaMLXError.apiError("No compatible draft model for \(modelId)")
                }
                if models.getRecord(candidate.draftModelId) == nil {
                    models.register(
                        id: candidate.draftModelId,
                        family: candidate.family,
                        modelType: .llm,
                        remoteURL: "https://huggingface.co/\(candidate.downloadRepo)",
                        sizeBytes: UInt64(candidate.estimatedSizeMB) * 1_000_000
                    )
                }
                if models.isDownloaded(candidate.draftModelId) {
                    return try Self.jsonResponse(SpecBoostInfo(
                        status: "eligible",
                        draftModelId: candidate.draftModelId,
                        draftDisplayName: candidate.displayName,
                        draftDownloaded: true,
                        draftLoaded: inference.engine.getContainer(for: candidate.draftModelId)?.isLoaded == true
                    ))
                }
                if let status = models.getDownloadStatus(candidate.draftModelId), status.state == .downloading {
                    return try Self.jsonResponse(status, httpStatus: .accepted)
                }
                try models.startDownload(candidate.draftModelId)
                let status = models.getDownloadStatus(candidate.draftModelId)
                    ?? DownloadStatus(modelId: candidate.draftModelId, state: .downloading)
                return try Self.jsonResponse(status, httpStatus: .accepted)
            }
            Post("/admin/models/boost/load") { request, context in
                let body = try await request.body.collect(upTo: .max)
                let req = try JSONDecoder().decode(AdminLoadRequest.self, from: body)
                let modelId = req.modelId
                guard let record = models.getRecord(modelId) else {
                    throw NovaMLXError.modelNotFound(modelId)
                }
                let isHybrid = inference.isModelLoaded(modelId) && inference.isHybridModel(modelId)
                guard let candidate = DraftModelRegistry.shared.recommendation(
                    family: record.family, isHybrid: isHybrid
                ) else {
                    throw NovaMLXError.apiError("No compatible draft model for \(modelId)")
                }
                guard models.isDownloaded(candidate.draftModelId) else {
                    throw NovaMLXError.apiError("Draft model \(candidate.draftModelId) not downloaded yet")
                }
                if inference.engine.getContainer(for: candidate.draftModelId)?.isLoaded == true {
                    return try Self.jsonResponse(SpecBoostInfo(
                        status: "active",
                        draftModelId: candidate.draftModelId,
                        draftLoaded: true
                    ))
                }
                if models.getRecord(candidate.draftModelId) == nil {
                    models.register(
                        id: candidate.draftModelId,
                        family: candidate.family,
                        modelType: .llm,
                        remoteURL: "https://huggingface.co/\(candidate.downloadRepo)",
                        sizeBytes: UInt64(candidate.estimatedSizeMB) * 1_000_000
                    )
                }
                guard let draftRecord = models.getRecord(candidate.draftModelId) else {
                    throw NovaMLXError.modelNotFound(candidate.draftModelId)
                }
                let config = ModelConfig(
                    identifier: ModelIdentifier(id: candidate.draftModelId, family: candidate.family),
                    modelType: .llm
                )
                try await inference.loadModel(at: draftRecord.localURL, config: config)
                return try Self.jsonResponse(SpecBoostInfo(
                    status: "active",
                    draftModelId: candidate.draftModelId,
                    draftDisplayName: candidate.displayName,
                    draftLoaded: true
                ))
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
            Get("/admin/api/chat-template/diagnose/{modelId}") { [modelManager] _, context in
                // The {modelId} parameter is URL-encoded by the client (slashes
                // become %2F); Hummingbird's parameter capture decodes that.
                let modelId = try context.parameters.require("modelId")
                // Look up the registered family for this model.
                let family: NovaMLXCore.ModelFamily = modelManager.getRecord(modelId)?.family ?? .other
                let report = ChatTemplateDiagnostics.diagnose(
                    modelId: modelId,
                    modelsDir: NovaMLXPaths.modelsDir,
                    family: family
                )
                return try Self.jsonResponse(report)
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
            Get("/admin/api/memory") { _, _ in
                guard let enforcer = inference.engine.memoryEnforcer else {
                    let body: [String: Any] = [
                        "enabled": false,
                        "currentBytes": inference.engine.gpuActiveMemory
                    ]
                    let data = try JSONSerialization.data(withJSONObject: body)
                    return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
                }
                let status = await enforcer.status
                var body: [String: Any] = [
                    "enabled": status.enabled,
                    "softLimitBytes": status.softLimitBytes,
                    "hardLimitBytes": status.hardLimitBytes,
                    "currentBytes": status.currentBytes,
                    "utilization": status.utilization,
                    "totalEvictions": status.totalEvictions,
                    "physicalRAM": status.physicalRAM,
                ]
                if let model = status.lastEvictedModel { body["lastEvictedModel"] = model }
                if let time = status.lastEvictionTime { body["lastEvictionTime"] = time.ISO8601Format() }
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
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

            // MARK: - Modelfiles
            Get("/admin/modelfiles") { _, _ in
                let files = modelfileMgr.list()
                return try Self.jsonResponse(files)
            }
            Get("/admin/modelfiles/{name}") { request, context in
                let name = try context.parameters.require("name")
                guard let mf = modelfileMgr.get(name) else {
                    throw NovaMLXError.modelNotFound("Modelfile '\(name)' not found")
                }
                return try Self.jsonResponse(mf)
            }
            Post("/admin/modelfiles") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                let mf = try JSONDecoder().decode(Modelfile.self, from: body)
                try modelfileMgr.create(mf)
                return try Self.jsonResponse(mf, httpStatus: .created)
            }
            Put("/admin/modelfiles/{name}") { request, context in
                let name = try context.parameters.require("name")
                let body = try await request.body.collect(upTo: .max)
                var mf = try JSONDecoder().decode(Modelfile.self, from: body)
                // Force name to match URL parameter to prevent mismatches
                mf = Modelfile(
                    name: name, baseModel: mf.baseModel,
                    systemPrompt: mf.systemPrompt, parameters: mf.parameters,
                    tools: mf.tools, description: mf.description
                )
                try modelfileMgr.update(mf)
                return try Self.jsonResponse(mf)
            }
            Delete("/admin/modelfiles/{name}") { request, context in
                let name = try context.parameters.require("name")
                try modelfileMgr.delete(name)
                return Response(status: .ok, body: .init(byteBuffer: ByteBuffer(string: "{\"status\":\"deleted\"}")))
            }

            // MARK: - Tokenhub Provider CRUD
            let tokenhubMgr = TokenhubManager.shared
            Get("/admin/tokenhub/providers") { _, _ in
                let providers = tokenhubMgr.list()
                return try Self.jsonResponse(providers)
            }
            Post("/admin/tokenhub/providers") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                var provider = try JSONDecoder().decode(TokenhubProvider.self, from: Data(buffer: body))
                // Ensure id is derived from name
                provider = TokenhubProvider(
                    name: provider.name,
                    endpoint: provider.endpoint,
                    apiKey: provider.apiKey,
                    remoteModel: provider.remoteModel,
                    isEnabled: provider.isEnabled,
                    includeInLoadBalance: provider.includeInLoadBalance,
                    tags: provider.tags,
                    isLocal: provider.isLocal,
                    isFree: provider.isFree,
                    isManaged: provider.isManaged
                )
                try tokenhubMgr.create(provider)
                return try Self.jsonResponse(provider, httpStatus: .created)
            }
            Put("/admin/tokenhub/providers/{name}") { request, context in
                let name = try context.parameters.require("name")
                let body = try await request.body.collect(upTo: .max)
                var provider = try JSONDecoder().decode(TokenhubProvider.self, from: Data(buffer: body))
                provider = TokenhubProvider(
                    name: name,
                    endpoint: provider.endpoint,
                    apiKey: provider.apiKey,
                    remoteModel: provider.remoteModel,
                    isEnabled: provider.isEnabled,
                    includeInLoadBalance: provider.includeInLoadBalance,
                    tags: provider.tags,
                    isLocal: provider.isLocal,
                    isFree: provider.isFree,
                    isManaged: provider.isManaged
                )
                try tokenhubMgr.update(provider)
                return try Self.jsonResponse(provider)
            }
            Delete("/admin/tokenhub/providers/{name}") { request, context in
                let name = try context.parameters.require("name")
                try tokenhubMgr.delete(name)
                return Response(status: .ok, body: .init(byteBuffer: ByteBuffer(string: "{\"status\":\"deleted\"}")))
            }
            Post("/admin/tokenhub/test") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                var provider = try JSONDecoder().decode(TokenhubProvider.self, from: Data(buffer: body))
                let backend = CloudBackend.shared
                let ok = await backend.healthCheck(provider: provider)
                if ok {
                    provider.lastTestedAt = Date()
                    provider.lastStatus = "ok"
                    if let existing = tokenhubMgr.get(provider.name) {
                        var updated = provider
                        updated = TokenhubProvider(
                            name: existing.name,
                            endpoint: updated.endpoint,
                            apiKey: updated.apiKey,
                            remoteModel: updated.remoteModel,
                            isEnabled: updated.isEnabled,
                            includeInLoadBalance: updated.includeInLoadBalance
                        )
                        // Preserve test result timestamps
                        _ = try? tokenhubMgr.update(updated)
                    }
                }
                struct TestResult: Encodable {
                    let success: Bool
                    let provider: String
                    let endpoint: String
                }
                return try Self.jsonResponse(TestResult(
                    success: ok, provider: provider.name, endpoint: provider.endpoint
                ))
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
                    let mlxOnly = params["mlx_only"] == "true"
                    let endpoint = params["endpoint"] ?? params["mirror"]
                    NovaMLXLog.info("[HF][Search] admin request q=\(q) endpoint=\(endpoint ?? "official") mlxOnly=\(mlxOnly) limit=\(limit)")

                    let searchService: HuggingFaceService = {
                        if let ep = endpoint, !ep.isEmpty {
                            return HuggingFaceService(modelDirectory: self.modelManager.modelsDirectory, endpoint: ep)
                        }
                        return hf
                    }()

                    let result = try await searchService.searchModels(query: q, limit: limit, mlxOnly: mlxOnly)
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
                    let endpoint = params["endpoint"] ?? params["mirror"]

                    let infoService: HuggingFaceService = {
                        if let ep = endpoint, !ep.isEmpty {
                            return HuggingFaceService(modelDirectory: self.modelManager.modelsDirectory, endpoint: ep)
                        }
                        return hf
                    }()

                    let detail = try await infoService.getModelDetail(repoId: repoId)
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
                    let endpoint = json["endpoint"] as? String
                    guard !repoId.isEmpty else {
                        return try Self.jsonResponse(["error": "repo_id required"], httpStatus: .badRequest)
                    }

                    if let ep = endpoint, !ep.isEmpty {
                        NovaMLXLog.info("[HF] Download request for \(repoId) using custom endpoint: \(ep)")
                    }

                    // Always use the global hf instance so that the created HFDownloadTask
                    // ends up in the activeTasks that GET /admin/api/hf/tasks returns.
                    // The live mirrorEndpoint (if any) is passed down so ModelScope / custom
                    // mirrors still get the correct listing + resolve logic.
                    let task = try await hf.startDownload(
                        repoId: repoId,
                        hfToken: hfToken,
                        mirrorEndpoint: endpoint
                    )
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
            // Agent binary detection
            Get("/admin/api/agents/check") { _, _ in
                let agentsToCheck = [
                    ("openclaw", "OpenClaw", "https://github.com/openclaw/openclaw"),
                    ("hermes", "Hermes Agent", "https://github.com/hermes-agent/hermes"),
                    ("opencode", "OpenCode", "https://github.com/opencode-ai/opencode"),
                ]
                let searchPaths = ["/usr/local/bin", "/opt/homebrew/bin", NSString(string: "~/").expandingTildeInPath + "/.local/bin"]
                var results: [[String: Any]] = []
                for (binary, name, installUrl) in agentsToCheck {
                    var found = false
                    var foundPath: String? = nil
                    // Try which
                    let task = Process()
                    task.executableURL = URL(fileURLWithPath: "/usr/bin/which")
                    task.arguments = [binary]
                    let pipe = Pipe()
                    task.standardOutput = pipe
                    task.standardError = FileHandle.nullDevice
                    try? task.run()
                    task.waitUntilExit()
                    if task.terminationStatus == 0 {
                        let data = pipe.fileHandleForReading.readDataToEndOfFile()
                        if let path = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty {
                            found = true
                            foundPath = path
                        }
                    }
                    if !found {
                        for dir in searchPaths {
                            let p = dir + "/" + binary
                            if FileManager.default.isExecutableFile(atPath: p) {
                                found = true
                                foundPath = p
                                break
                            }
                        }
                    }
                    results.append([
                        "id": binary,
                        "name": name,
                        "installed": found,
                        "path": foundPath as Any,
                        "installUrl": installUrl,
                    ])
                }
                let data = try JSONSerialization.data(withJSONObject: results)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }

            // MARK: - API Key Management

            Get("/admin/keys") { _, _ in
                let keys = (try? NovaDB.shared.apiKeyStore.listAsAPIKey()) ?? []
                let masked: [[String: Any]] = keys.map { key in
                    var usage: [String: Any] = [
                        "totalTokensUsed": key.usage.totalTokensUsed,
                        "totalRequests": key.usage.totalRequests,
                        "dailyTokens": key.usage.periodTokens,
                        "dailyRequests": key.usage.periodRequests,
                    ]
                    if let lastUsed = key.usage.lastUsedAt {
                        usage["lastUsedAt"] = ISO8601DateFormatter().string(from: lastUsed)
                    }
                    var dict: [String: Any] = [
                        "id": key.id,
                        "name": key.name,
                        "keyPrefix": key.keyPrefix,
                        "createdAt": ISO8601DateFormatter().string(from: key.createdAt),
                        "isEnabled": key.isEnabled,
                        "isExpired": key.isExpired,
                        "isActive": key.isActive,
                        "usage": usage,
                    ]
                    if let exp = key.expiresAt { dict["expiresAt"] = ISO8601DateFormatter().string(from: exp) }
                    if let rps = key.rateLimitPerSecond { dict["rateLimitPerSecond"] = rps }
                    if let burst = key.rateLimitBurst { dict["rateLimitBurst"] = burst }
                    if let models = key.allowedModels { dict["allowedModels"] = models }
                    if let endpoints = key.allowedEndpoints { dict["allowedEndpoints"] = endpoints }
                    if let maxTokens = key.maxTokensPerPeriod { dict["maxTokensPerPeriod"] = maxTokens }
                    if let maxRequests = key.maxRequestsPerPeriod { dict["maxRequestsPerPeriod"] = maxRequests }
                    return dict
                }
                let data = try JSONSerialization.data(withJSONObject: masked)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }

            Post("/admin/keys") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                struct CreateKeyRequest: Decodable {
                    let name: String?
                    let rateLimitPerSecond: Double?
                    let rateLimitBurst: Int?
                    let allowedModels: [String]?
                    let allowedEndpoints: [String]?
                    let maxTokensPerPeriod: Int64?
                    let maxRequestsPerPeriod: Int64?
                    let usageResetPeriod: String?
                }
                let req = try JSONDecoder().decode(CreateKeyRequest.self, from: body)
                let name = req.name ?? "API Key"
                let period: UsageResetPeriod = req.usageResetPeriod.flatMap { UsageResetPeriod(rawValue: $0) } ?? .daily
                let (record, rawKey) = try NovaDB.shared.apiKeyStore.create(
                    name: name,
                    rateLimitPerSecond: req.rateLimitPerSecond,
                    rateLimitBurst: req.rateLimitBurst,
                    allowedModels: req.allowedModels,
                    allowedEndpoints: req.allowedEndpoints,
                    maxTokensPerPeriod: req.maxTokensPerPeriod,
                    maxRequestsPerPeriod: req.maxRequestsPerPeriod,
                    usageResetPeriod: period.rawValue
                )
                let resp: [String: String] = [
                    "id": record.id,
                    "name": record.name,
                    "key": rawKey,
                    "keyPrefix": record.keyPrefix,
                    "createdAt": ISO8601DateFormatter().string(from: record.createdAt),
                ]
                return try Self.jsonResponse(resp)
            }

            Get("/admin/keys/{id}") { request, context in
                guard let id = context.parameters.get("id", as: String.self) else {
                    throw NovaMLXError.apiError("Missing key ID")
                }
                guard let key = (try? NovaDB.shared.apiKeyStore.getAsAPIKey(id: id)) ?? nil else {
                    return try Self.jsonResponse(["error": "Key not found"], httpStatus: .notFound)
                }
                var usage: [String: Any] = [
                    "totalTokensUsed": key.usage.totalTokensUsed,
                    "totalRequests": key.usage.totalRequests,
                    "periodTokens": key.usage.periodTokens,
                    "periodRequests": key.usage.periodRequests,
                ]
                if let lastUsed = key.usage.lastUsedAt {
                    usage["lastUsedAt"] = ISO8601DateFormatter().string(from: lastUsed)
                }
                var dict: [String: Any] = [
                    "id": key.id,
                    "name": key.name,
                    "keyPrefix": key.keyPrefix,
                    "createdAt": ISO8601DateFormatter().string(from: key.createdAt),
                    "isEnabled": key.isEnabled,
                    "isExpired": key.isExpired,
                    "isActive": key.isActive,
                    "usage": usage,
                ]
                if let exp = key.expiresAt { dict["expiresAt"] = ISO8601DateFormatter().string(from: exp) }
                if let rps = key.rateLimitPerSecond { dict["rateLimitPerSecond"] = rps }
                if let burst = key.rateLimitBurst { dict["rateLimitBurst"] = burst }
                if let models = key.allowedModels { dict["allowedModels"] = models }
                if let endpoints = key.allowedEndpoints { dict["allowedEndpoints"] = endpoints }
                if let maxTokens = key.maxTokensPerPeriod { dict["maxTokensPerPeriod"] = maxTokens }
                if let maxRequests = key.maxRequestsPerPeriod { dict["maxRequestsPerPeriod"] = maxRequests }
                let data = try JSONSerialization.data(withJSONObject: dict)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }

            Put("/admin/keys/{id}") { request, context in
                guard let id = context.parameters.get("id", as: String.self) else {
                    throw NovaMLXError.apiError("Missing key ID")
                }
                let body = try await request.body.collect(upTo: .max)
                struct UpdateKeyRequest: Decodable {
                    let name: String?
                    let isEnabled: Bool?
                    let expiresAt: String?
                    let rateLimitPerSecond: Double?
                    let rateLimitBurst: Int?
                    let allowedModels: [String]?
                    let allowedEndpoints: [String]?
                    let maxTokensPerPeriod: Int64?
                    let maxRequestsPerPeriod: Int64?
                    let usageResetPeriod: String?
                }
                let req = try JSONDecoder().decode(UpdateKeyRequest.self, from: body)
                try NovaDB.shared.apiKeyStore.update(id: id) { rec in
                    if let name = req.name { rec.name = name }
                    if let isEnabled = req.isEnabled { rec.isEnabled = isEnabled }
                    if let expiresAtStr = req.expiresAt {
                        rec.expiresAt = (expiresAtStr == "never") ? nil : ISO8601DateFormatter().date(from: expiresAtStr)
                    }
                    if let rps = req.rateLimitPerSecond { rec.rateLimitPerSecond = rps }
                    if let burst = req.rateLimitBurst { rec.rateLimitBurst = burst }
                    if let models = req.allowedModels {
                        rec.allowedModels = Self.encodeJSONField(models.isEmpty ? nil : models)
                    }
                    if let endpoints = req.allowedEndpoints {
                        rec.allowedEndpoints = Self.encodeJSONField(endpoints.isEmpty ? nil : endpoints)
                    }
                    if let maxTokens = req.maxTokensPerPeriod { rec.maxTokensPerPeriod = maxTokens }
                    if let maxRequests = req.maxRequestsPerPeriod { rec.maxRequestsPerPeriod = maxRequests }
                    if let periodStr = req.usageResetPeriod {
                        rec.usageResetPeriod = periodStr
                    }
                }
                return try Self.jsonResponse(["status": "ok"])
            }

            Delete("/admin/keys/{id}") { request, context in
                guard let id = context.parameters.get("id", as: String.self) else {
                    throw NovaMLXError.apiError("Missing key ID")
                }
                try NovaDB.shared.apiKeyStore.delete(id: id)
                return try Self.jsonResponse(["status": "ok"])
            }

            Post("/admin/keys/{id}/rotate") { request, context in
                guard let id = context.parameters.get("id", as: String.self) else {
                    throw NovaMLXError.apiError("Missing key ID")
                }
                let (record, rawKey) = try NovaDB.shared.apiKeyStore.rotate(id: id)
                let resp: [String: String] = [
                    "id": record.id,
                    "key": rawKey,
                    "keyPrefix": record.keyPrefix,
                ]
                return try Self.jsonResponse(resp)
            }

            Get("/admin/keys/{id}/usage") { request, context in
                guard let id = context.parameters.get("id", as: String.self) else {
                    throw NovaMLXError.apiError("Missing key ID")
                }
                guard let key = (try? NovaDB.shared.apiKeyStore.getAsAPIKey(id: id)) ?? nil else {
                    return try Self.jsonResponse(["error": "Key not found"], httpStatus: .notFound)
                }
                var usage: [String: Any] = [
                    "totalTokensUsed": key.usage.totalTokensUsed,
                    "totalRequests": key.usage.totalRequests,
                    "periodTokens": key.usage.periodTokens,
                    "periodRequests": key.usage.periodRequests,
                ]
                if let lastUsed = key.usage.lastUsedAt {
                    usage["lastUsedAt"] = ISO8601DateFormatter().string(from: lastUsed)
                }
                let data = try JSONSerialization.data(withJSONObject: usage)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }

            // Config store read/write (backed by SQLite; JSON shape preserved for API compat)
            Get("/admin/api/config") { _, _ in
                let data: Data = await {
                    do { return try await NovaMLXConfiguration.shared.serializedConfigJSON() }
                    catch { return Data("{}".utf8) }
                }()
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            Put("/admin/api/config") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                do {
                    try await NovaMLXConfiguration.shared.applySerializedConfigJSON(Data(body.readableBytesView))
                } catch {
                    throw NovaMLXError.apiError("Invalid config JSON: \(error)")
                }
                return try Self.jsonResponse(["status": "ok", "message": "Config saved. Restart required."])
            }

            Get("/admin/api/log-level") { _, _ in
                try Self.jsonResponse(["level": "\(NovaMLXLog.fileLogLevel.rawValue)"] as [String: String])
            }
            Put("/admin/api/log-level") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                guard let json = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                      let level = json["level"] as? Int,
                      let logLevel = NovaMLXLog.LogLevel(rawValue: level) else {
                    return try Self.jsonResponse(["error": "Invalid level. Use 0=debug, 1=info, 2=warning, 3=error"], httpStatus: .badRequest)
                }
                NovaMLXLog.fileLogLevel = logLevel
                NovaMLXLog.info("[Admin] Log level changed to \(logLevel)")
                return try Self.jsonResponse(["level": logLevel.rawValue])
            }

            // MARK: - Cluster Admin
            Get("/admin/api/cluster/status") { _, _ in
                let body = ClusterAdminRoutes.shared.clusterStatus()
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            Get("/admin/api/cluster/discovery-debug") { _, _ in
                let body = ClusterAdminRoutes.shared.discoveryDebug()
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }
            Get("/admin/api/models/{id}/cluster/sync-status") { _, context in
                let modelId = try context.parameters.require("id")
                let body = ClusterAdminRoutes.shared.modelSyncStatus(modelId: modelId)
                let data = try JSONSerialization.data(withJSONObject: body)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }

            // Shard plan: GET /admin/api/cluster/shard-plan?model=<modelId>
            Get("/admin/api/cluster/shard-plan") { request, context in
                let modelId = request.uri.query?.split(separator: "&")
                    .compactMap { param -> String? in
                        let parts = param.split(separator: "=", maxSplits: 1)
                        guard parts.count == 2, parts[0] == "model" else { return nil }
                        return String(parts[1])
                    }.first ?? ""
                guard !modelId.isEmpty else {
                    let err = ["error": "missing ?model= parameter"]
                    let data = try JSONSerialization.data(withJSONObject: err)
                    return Response(status: .badRequest, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
                }
                if let plan = ClusterAdminRoutes.shared.currentShardPlan(modelId: modelId) {
                    let data = try JSONSerialization.data(withJSONObject: plan)
                    return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
                } else {
                    let data = try JSONSerialization.data(withJSONObject: ["status": "no_plan"])
                    return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
                }
            }

            // Worker registration: POST /admin/api/cluster/workers/register
            Post("/admin/api/cluster/workers/register") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                let spec = try JSONDecoder().decode(NodeSpec.self, from: body)
                let info = ClusterManager.shared.registerWorker(spec: spec)
                let data = try JSONEncoder().encode(["nodeId": info.nodeId, "status": info.status.rawValue])
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: data)))
            }

            // Worker heartbeat: POST /admin/api/cluster/workers/heartbeat
            Post("/admin/api/cluster/workers/heartbeat") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                struct HeartbeatPayload: Codable { let nodeId: String }
                let payload = try JSONDecoder().decode(HeartbeatPayload.self, from: body)
                ClusterManager.shared.updateHeartbeat(nodeId: payload.nodeId)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(string: "{\"ok\":true}")))
            }

            // Model download for workers: GET /admin/api/cluster/models/{id}/download
            Get("/admin/api/cluster/models/{id}/download") { _, context in
                let modelId = try context.parameters.require("id")
                guard let record = models.getRecord(modelId) else {
                    return Response(status: .notFound)
                }
                let modelDir = record.localURL.path
                let fm = FileManager.default
                guard fm.fileExists(atPath: modelDir) else {
                    return Response(status: .notFound)
                }
                // Stream the entire model directory as a tar.gz
                let process = Process()
                let pipe = Pipe()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
                process.arguments = ["-czf", "-", "-C", modelDir, "."]
                process.standardOutput = pipe
                try process.run()
                process.waitUntilExit()
                guard let data = try pipe.fileHandleForReading.readToEnd() else {
                    return Response(status: .internalServerError)
                }
                return Response(
                    status: .ok,
                    headers: [
                        .contentType: "application/gzip",
                        .contentDisposition: "attachment; filename=\"\(modelId.replacingOccurrences(of:"/",with:"-")).tar.gz\"",
                    ],
                    body: .init(byteBuffer: ByteBuffer(data: data))
                )
            }

            // Cluster model activation: POST /admin/api/cluster/activate-model
            Post("/admin/api/cluster/activate-model") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                guard let json = try? JSONSerialization.jsonObject(with: Data(buffer: body)) as? [String: String],
                      let modelId = json["modelId"] else {
                    return Response(status: .badRequest, headers: [.contentType: "application/json"],
                                    body: .init(byteBuffer: ByteBuffer(string: "{\"error\":true,\"message\":\"modelId required\"}")))
                }
                let result = await ClusterAdminRoutes.shared.activateModel(modelId: modelId)
                let data = try? JSONSerialization.data(withJSONObject: result)
                let jsonStr = data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                return Response(status: .ok, headers: [.contentType: "application/json"],
                                body: .init(byteBuffer: ByteBuffer(string: jsonStr)))
            }

            // Cluster model deactivation: POST /admin/api/cluster/deactivate-model
            Post("/admin/api/cluster/deactivate-model") { _, _ in
                let result = await ClusterAdminRoutes.shared.deactivateModel()
                let data = try? JSONSerialization.data(withJSONObject: result)
                let jsonStr = data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                return Response(status: .ok, headers: [.contentType: "application/json"],
                                body: .init(byteBuffer: ByteBuffer(string: jsonStr)))
            }

            // Cluster model status: GET /admin/api/cluster/model-status
            Get("/admin/api/cluster/model-status") { _, _ in
                let result = ClusterAdminRoutes.shared.modelStatus()
                let data = try? JSONSerialization.data(withJSONObject: result)
                let jsonStr = data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                return Response(status: .ok, headers: [.contentType: "application/json"],
                                body: .init(byteBuffer: ByteBuffer(string: jsonStr)))
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

        // Workers bind admin API to all interfaces so Coordinator can reach them
        let adminHost = (cfg.cluster?.role == "worker") ? "0.0.0.0" : cfg.host
        let adminApp = Application(
            router: adminRouter,
            configuration: .init(address: .hostname(adminHost, port: cfg.adminPort)),
            logger: Logger(label: "NovaMLX.Admin")
        )

        NovaMLXLog.info("NovaMLX API server starting on \(cfg.host):\(cfg.port)\(cfg.isTLSEnabled ? " (TLS)" : "")")
        NovaMLXLog.info("NovaMLX Admin API starting on \(adminHost):\(cfg.adminPort)")

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await mainApp.run() }
            group.addTask { try await adminApp.run() }
            try await group.next()
            group.cancelAll()
        }
    }


    // MARK: - Per-Request keep_alive

    static func applyKeepAlive(_ keepAlive: KeepAliveValue?, modelId: String, pool: EnginePool) {
        guard let ka = keepAlive else { return }
        pool.applyKeepAlive(modelId: modelId, deadline: ka.deadline())
    }

    // MARK: - API Key Usage Tracking

    /// JSON-encode an optional Encodable value into a String for the store's
    /// JSON-string columns (`allowed_models`, `allowed_endpoints`). Returns nil
    /// for nil input or encode failures. Used by the admin key update route.
    private nonisolated static func encodeJSONField<T: Encodable>(_ value: T?) -> String? {
        guard let value else { return nil }
        guard let data = try? JSONEncoder().encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func recordTokenUsage(request: Request, promptTokens: Int, completionTokens: Int, model: String? = nil) {
        let total = Int64(promptTokens) + Int64(completionTokens)
        guard total > 0 else { return }
        let token = extractRequestToken(request)
        guard let token else { return }
        Task {
            if let key = (try? NovaDB.shared.apiKeyStore.findAPIKeyByRawToken(token)) ?? nil {
                try? NovaDB.shared.apiKeyStore.recordUsage(keyId: key.id, tokens: total, model: model)
            }
        }
    }

    private static func extractRequestToken(_ request: Request) -> String? {
        let authHeader = request.headers[.authorization]
        if let authHeader, authHeader.hasPrefix("Bearer ") {
            return String(authHeader.dropFirst(7))
        }
        return request.headers[HTTPField.Name("x-api-key")!]
    }



    static func jsonResponse<T: Encodable>(_ value: T) throws -> Response {
        try jsonResponse(value, httpStatus: .ok)
    }

    /// Convert a stream Token's logprob data to OpenAI response format.
    /// Populates `bytes` with UTF-8 byte values per OpenAI spec.
    static func tokenToLogprobEntry(_ token: Token) -> OpenAILogprobEntry? {
        guard let logprob = token.logprob else { return nil }
        let topEntries: [OpenAITopLogprob] = (token.topLogprobs ?? []).map { tp in
            OpenAITopLogprob(
                token: tp.tokenText,
                logprob: tp.logprob,
                bytes: tp.tokenText.utf8.map(Int.init)
            )
        }
        return OpenAILogprobEntry(
            token: token.text,
            logprob: logprob,
            bytes: token.text.utf8.map(Int.init),
            topLogprobs: topEntries
        )
    }

    /// Build `OpenAILogprobs` from a collection of tokens with logprob data.
    static func buildLogprobs(from tokens: [Token]) -> OpenAILogprobs? {
        let entries = tokens.compactMap { tokenToLogprobEntry($0) }
        guard !entries.isEmpty else { return nil }
        return OpenAILogprobs(content: entries)
    }

    static func jsonResponse<T: Encodable>(_ value: T, httpStatus: HTTPResponse.Status) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(
            status: httpStatus,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

}
