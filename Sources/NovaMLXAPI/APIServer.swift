import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdRouter
import ImageIO
import Logging
import NovaMLXCore
import NovaMLXDistributed
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

    private static let publicPaths: Set<String> = ["/", "/chat", "/health", "/v1/models", "/v1/stats", "/favicon.ico"]
    private static let publicPrefixes: Set<String> = ["/v1/chat/history", "/admin/"]

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
            APIKeyAuthMiddleware(validKeys: cfg.apiKeys)
            NovaMLXErrorMiddleware()
            Get("/v1/models") { request, context in
                let detector = self.capabilitiesDetector
                let modelList = models.downloadedModels()
                    .filter { inference.isModelLoaded($0.id) || embeddings.isLoaded($0.id) || inference.transcriptionService.isLoaded($0.id) || inference.imageGenerationService.isLoaded($0.id) }
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
                return response
            }
            Post("/v1/messages") { request, context in
                let body = try await request.body.collect(upTo: .max)
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
                    seed: req.seed.map { UInt64($0) }
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
            AdminAuthMiddleware(validKeys: cfg.apiKeys)
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
                    _ = try await inference.transcriptionService.loadModel(from: record.localURL, config: config)
                } else if record.modelType == .image {
                    _ = try await inference.imageGenerationService.loadModel(from: record.localURL, config: config)
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
                }

                if inference.imageGenerationService.isLoaded(req.modelId) {
                    inference.imageGenerationService.unload(modelId: req.modelId)
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

            // Config file read/write
            Get("/admin/api/config") { _, _ in
                let configURL = await NovaMLXConfiguration.shared.configFileURL
                guard let data = try? Data(contentsOf: configURL),
                      let json = try? JSONSerialization.jsonObject(with: data) else {
                    return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(string: "{}")))
                }
                let responseData = try JSONSerialization.data(withJSONObject: json)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: ByteBuffer(data: responseData)))
            }
            Put("/admin/api/config") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                guard let json = try? JSONSerialization.jsonObject(with: body) else {
                    throw NovaMLXError.apiError("Invalid JSON")
                }
                let configURL = await NovaMLXConfiguration.shared.configFileURL
                let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: configURL, options: .atomic)
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

    // MARK: - Tokenhub Proxy

    /// Raw passthrough proxy for tokenhub. Forwards the original request body to the provider
    /// with the model name swapped to provider.remoteModel. No post-processing.
    private static func handleTokenhubPassthrough(
        modelName: String,
        rawBody: Data,
        path: String,
        inference: InferenceService,
        tag: String? = nil
    ) async throws -> Response {
        guard let provider = TokenhubManager.shared.resolve(modelName: modelName, tag: tag) else {
            return try Self.jsonResponse(
                ["error": ["message": "Unknown tokenhub provider: \(modelName)", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        // Swap model name in the raw JSON body
        var bodyDict = try JSONSerialization.jsonObject(with: rawBody) as? [String: Any] ?? [:]
        bodyDict["model"] = provider.remoteModel
        _ = try JSONSerialization.data(withJSONObject: bodyDict)
        let isStreaming = (bodyDict["stream"] as? Bool) ?? false
        let isLB = modelName.lowercased() == "tknet"
        let maxRetries = isLB ? 2 : 0

        var triedProviders = Set<String>()
        var lastProvider = provider

        // Resolve effective API key: managed providers use session token
        func effectiveApiKey(_ p: TokenhubProvider) -> String {
            if p.isManaged { return AuthCache.loadSession() ?? "" }
            return p.apiKey
        }

        for attempt in 0...maxRetries {
            triedProviders.insert(lastProvider.name)
            NovaMLXLog.info("[Tokenhub] -> \(lastProvider.name) (\(lastProvider.endpoint)/\(path)) remoteModel=\(lastProvider.remoteModel) managed=\(lastProvider.isManaged)\(attempt > 0 ? " retry#\(attempt)" : "")")

            // Build request for current provider
            var bodyForThis = bodyDict
            bodyForThis["model"] = lastProvider.remoteModel
            let bodyData = try JSONSerialization.data(withJSONObject: bodyForThis)

            let baseURL = URL(string: lastProvider.endpoint)!
            var urlRequest = URLRequest(url: baseURL.appendingPathComponent(path))
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
                    NovaMLXLog.warning("[Tokenhub] \(lastProvider.name) streaming failed HTTP \(statusCode)")
                    let elapsed = ContinuousClock.now - start
                    TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: false, latencyMs: durationToMs(elapsed))
                    // Retry with different provider if possible
                    if isLB, let next = pickRetryProvider(modelName: modelName, tag: tag, exclude: triedProviders) {
                        lastProvider = next
                        continue
                    }
                    return Response(status: .init(integerLiteral: statusCode))
                }
                let elapsed = ContinuousClock.now - start
                let latencyMs = durationToMs(elapsed)
                TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: true, latencyMs: latencyMs)

                let responseBody = ResponseBody { writer in
                    do {
                        for try await line in bytes.lines {
                            try await writer.write(ByteBuffer(string: line + "\n\n"))
                        }
                        try await writer.finish(nil)
                    } catch {
                        try? await writer.finish(nil)
                    }
                }

                var headers = HTTPFields()
                headers[.contentType] = "text/event-stream"
                headers[.cacheControl] = "no-cache"
                headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                return Response(status: .ok, headers: headers, body: responseBody)
            } else {
                let start = ContinuousClock.now
                let (data, urlResponse) = try await URLSession.shared.data(for: urlRequest)
                let elapsed = ContinuousClock.now - start
                let latencyMs = durationToMs(elapsed)
                guard let http = urlResponse as? HTTPURLResponse else {
                    TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: false, latencyMs: latencyMs)
                    if isLB, let next = pickRetryProvider(modelName: modelName, tag: tag, exclude: triedProviders) {
                        lastProvider = next
                        continue
                    }
                    return Response(status: .internalServerError)
                }
                let success = http.statusCode < 400
                TokenhubManager.shared.recordMetric(providerId: lastProvider.id, success: success, latencyMs: latencyMs)
                if http.statusCode >= 400 {
                    let body = String(data: data, encoding: .utf8)?.prefix(300) ?? "nil"
                    NovaMLXLog.warning("[Tokenhub] \(lastProvider.name) error HTTP \(http.statusCode): \(body)")
                    // Retry with different provider if possible
                    if isLB, let next = pickRetryProvider(modelName: modelName, tag: tag, exclude: triedProviders) {
                        lastProvider = next
                        continue
                    }
                }
                var headers: HTTPFields = [.contentType: "application/json"]
                headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                return Response(status: .init(integerLiteral: http.statusCode), headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
            }
        }

        // Should not reach here, but just in case
        return Response(status: .badGateway)
    }

    // MARK: - TokenHub Responses API Passthrough
    // Converts Responses API request → Chat Completions → forward → convert response back

    private static func handleTokenhubResponsesPassthrough(
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

        guard let provider = TokenhubManager.shared.resolve(modelName: req.model, tag: nil) else {
            return try Self.jsonResponse(
                ["error": ["message": "Unknown tokenhub provider: \(req.model)", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        let isLB = req.model.lowercased() == "tknet"
        let maxRetries = isLB ? 2 : 0
        var triedProviders = Set<String>()
        var lastProvider = provider

        func effectiveApiKey(_ p: TokenhubProvider) -> String {
            if p.isManaged { return AuthCache.loadSession() ?? "" }
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
            if let prevId = req.previousResponseId {
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
                            // Provider supports vision — build multimodal content parts
                            var contentParts: [[String: Any]] = []
                            if !text.isEmpty {
                                contentParts.append(["type": "text", "text": text])
                            }
                            for url in imageURLs {
                                contentParts.append(["type": "image_url", "image_url": ["url": url]])
                            }
                            messages.append(["role": role, "content": contentParts])
                        } else {
                            // Text-only provider — collect images for preprocessing
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
            // user+user → single user with concatenated content
            // assistant(text)+assistant(tool_calls) → single assistant with both
            var merged: [[String: Any]] = []
            for msg in messages {
                guard let last = merged.last else { merged.append(msg); continue }
                let lastRole = last["role"] as? String ?? ""
                let msgRole = msg["role"] as? String ?? ""
                if lastRole == msgRole {
                    if msgRole == "user" {
                        // Merge user messages
                        let prev = last["content"] as? String ?? ""
                        let cur = msg["content"] as? String ?? ""
                        merged[merged.count - 1]["content"] = prev + "\n" + cur
                    } else if msgRole == "assistant" {
                        // Merge: prev has text, new has tool_calls (or vice versa)
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
            // If any assistant msg has reasoning_content, backfill empty string on the rest
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

            // Convert tools — only forward function-type tools with valid names
            // Codex sends Responses API format (name at top level) AND non-function tools (web_search, etc)
            // Must convert Responses format → Chat Completions format for upstream
            if let rawTools = (try? JSONSerialization.jsonObject(with: rawBody)) as? [String: Any],
               let toolsArray = rawTools["tools"] as? [[String: Any]] {
                var functionTools: [[String: Any]] = []
                for tool in toolsArray {
                    let toolType = tool["type"] as? String ?? "function"
                    // Skip non-function tools (web_search, code_interpreter, etc.)
                    guard toolType == "function" || tool["function"] != nil else { continue }

                    // Try Responses API format: name at top level
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
                    // Try Chat Completions format: name is nested under "function"
                    else if let fn = tool["function"] as? [String: Any],
                            let fnName = fn["name"] as? String, !fnName.isEmpty {
                        functionTools.append(tool)
                    }
                    // else: tool has no valid name — skip it entirely
                }
                if !functionTools.isEmpty {
                    body["tools"] = functionTools
                }
            } else if let tools = req.tools, !tools.isEmpty {
                // Fallback to Codable-decoded tools if rawBody parsing fails
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
                "model": model,
                "status": "completed",
                "output": output
            ]
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
            }

            return try? JSONSerialization.data(withJSONObject: response)
        }

        for attempt in 0...maxRetries {
            triedProviders.insert(lastProvider.name)
            let isStreaming = req.stream ?? false

            // Provider natively supports /v1/responses → raw passthrough, no conversion
            if lastProvider.supportsResponsesAPI {
                NovaMLXLog.info("[Tokenhub/Responses] -> \(lastProvider.name) RAW PASSTHROUGH streaming=\(isStreaming)\(attempt > 0 ? " retry#\(attempt)" : "")")

                // Swap model name in raw body, forward as-is
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

                    // Pass through SSE events as-is, inject X-Tokenhub-Provider header
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
                    // Non-streaming: forward and return as-is
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
                    do {
                        func sse(_ event: String, _ data: Encodable) async throws {
                            let jsonData = try JSONEncoder().encode(data)
                            try await writer.write(ByteBuffer(string: "event: \(event)\ndata: \(String(data: jsonData, encoding: .utf8) ?? "")\n\n"))
                        }

                        let emptyResp = ResponsesSSEResponse(id: responseId, status: "in_progress", model: model)
                        try await sse("response.created", ResponsesSSECreated(response: emptyResp))
                        try await sse("response.in_progress", ResponsesSSECreated(response: emptyResp))

                        // Track output items: text message + potential tool calls + reasoning
                        var fullText = ""
                        var textMessageStarted = false
                        var outputItems: [ResponseOutputItem] = []
                        var currentOutputIndex = 0

                        // Tool call tracking (parallel calls supported)
                        struct ToolCallAccumulator {
                            var id: String
                            var callId: String
                            var name: String
                            var arguments: String
                        }
                        var toolCalls: [Int: ToolCallAccumulator] = [:]  // index → accumulator

                        // Reasoning tracking
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

                            // Handle text content
                            if let content = delta["content"] as? String, !content.isEmpty {
                                try await startTextMessage()
                                fullText += content
                                try await sse("response.output_text.delta", ResponsesSSETextDelta(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, delta: content))
                            }

                            // Handle reasoning/thinking content
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
                                }
                            }

                            // Handle tool calls (parallel calls at different indices)
                            if let toolCallDeltas = delta["tool_calls"] as? [[String: Any]] {
                                // Finish any open text message before tool calls
                                try await finishTextMessage()

                                // Finish reasoning if active
                                if reasoningStarted, let rsId = reasoningId {
                                    let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                                    try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                                    try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, summary: summary))))
                                    outputItems.append(.reasoning(ResponseOutputReasoning(id: rsId, summary: summary)))
                                    currentOutputIndex += 1
                                    reasoningStarted = false
                                }

                                for tcDelta in toolCallDeltas {
                                    let tcIndex = tcDelta["index"] as? Int ?? 0
                                    if toolCalls[tcIndex] == nil {
                                        // New tool call — extract id, name
                                        let tcId = "fc_\(responseId.suffix(12))_\(tcIndex)"
                                        let callId = tcDelta["id"] as? String ?? "call_\(tcId)"
                                        let fn = tcDelta["function"] as? [String: Any] ?? [:]
                                        let name = fn["name"] as? String ?? ""

                                        let outputIdx = currentOutputIndex + tcIndex
                                        toolCalls[tcIndex] = ToolCallAccumulator(id: tcId, callId: callId, name: name, arguments: "")

                                        // Emit output_item.added for function_call
                                        try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: outputIdx, item: .functionCall(ResponseOutputFunctionCall(id: tcId, callId: callId, name: name, arguments: "", status: "in_progress"))))
                                    }

                                    // Accumulate argument deltas
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

                        // Finish any open text message
                        try await finishTextMessage()

                        // Finish reasoning if still active
                        if reasoningStarted, let rsId = reasoningId {
                            let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                            try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
                            try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, summary: summary))))
                            outputItems.append(.reasoning(ResponseOutputReasoning(id: rsId, summary: summary)))
                            currentOutputIndex += 1
                        }

                        // Finish tool calls
                        for (tcIndex, tc) in toolCalls.sorted(by: { $0.key < $1.key }) {
                            let outputIdx = currentOutputIndex + tcIndex
                            try await sse("response.function_call_arguments.done", ResponsesSSEFunctionCallArgsDone(itemId: tc.id, outputIndex: outputIdx, callId: tc.callId, arguments: tc.arguments))
                            try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: outputIdx, item: .functionCall(ResponseOutputFunctionCall(id: tc.id, callId: tc.callId, name: tc.name, arguments: tc.arguments))))
                            outputItems.append(.functionCall(ResponseOutputFunctionCall(id: tc.id, callId: tc.callId, name: tc.name, arguments: tc.arguments)))
                        }
                        if !toolCalls.isEmpty {
                            currentOutputIndex += toolCalls.count
                        }

                        // Final completed event — include user input for previous_response_id multi-turn
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

                        // Client sees only assistant output; store includes user input for multi-turn
                        let clientResp = OpenAIResponseObject(id: responseId, model: model, output: outputItems)
                        try await sse("response.completed", ResponsesSSECompleted(response: clientResp))
                        let storeResp = OpenAIResponseObject(id: responseId, model: model, output: allOutputItems)
                        ResponseStore.shared.put(storeResp)
                        try await writer.finish(nil)
                    } catch {
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

                // Convert Chat Completions response → Responses API response
                if let convertedData = convertToResponsesResponse(data, model: req.model) {
                    var headers: HTTPFields = [.contentType: "application/json"]
                    headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                    return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: convertedData)))
                } else {
                    // Fallback: return raw Chat Completions response
                    var headers: HTTPFields = [.contentType: "application/json"]
                    headers[.init("X-Tokenhub-Provider")!] = lastProvider.name
                    return Response(status: .ok, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
                }
            }
        }

        return Response(status: .badGateway)
    }

    private static func durationToMs(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
    }

    private static func pickRetryProvider(modelName: String, tag: String?, exclude: Set<String>) -> TokenhubProvider? {
        let pool = TokenhubManager.shared.list().filter { $0.isEnabled && $0.includeInLoadBalance && !exclude.contains($0.name) }
        var filtered = pool
        if let tag, !tag.isEmpty {
            filtered = filtered.filter { $0.tags.contains(tag) }
        }
        return filtered.randomElement()
    }

    // MARK: - Per-Request keep_alive

    private static func applyKeepAlive(_ keepAlive: KeepAliveValue?, modelId: String, pool: EnginePool) {
        guard let ka = keepAlive else { return }
        pool.applyKeepAlive(modelId: modelId, deadline: ka.deadline())
    }

    // MARK: - Auto-Load Helpers

    enum LoadOutcome: Sendable {
        case alreadyLoaded
        case justLoaded(coldLoadMs: Int)
        /// Streaming: load is required but deferred to inside the response body
        /// so that withSSEKeepAlive's heartbeat covers the load window.
        case deferred
    }

    private static func ensureModelReady(
        modelId: String,
        isStreaming: Bool,
        cfg: ServerConfig,
        inference: InferenceService,
        embeddings: EmbeddingService,
        coordinator: AutoLoadCoordinator,
        request: Request
    ) async throws -> LoadOutcome {
        // Fast path: already loaded
        if inference.isModelLoaded(modelId) || embeddings.isLoaded(modelId) {
            return .alreadyLoaded
        }

        // Auto-load disabled — throw original error
        if !cfg.autoLoad.enabled {
            throw NovaMLXError.modelNotLoaded(modelId)
        }

        // X-Wait-Cold-Load: false → fire-and-forget + immediate 503
        let waitForColdLoad = parseWaitColdLoadHeader(request)
        if waitForColdLoad == false {
            Task.detached {
                try? await coordinator.ensureLoaded(
                    modelId,
                    options: AutoLoadCoordinator.Options(
                        evictOnConflict: cfg.autoLoad.evictOnConflict,
                        allowDownload: cfg.autoLoad.allowDownload
                    )
                )
            }
            throw NovaMLXError.modelLoadInProgress(modelId: modelId, etaSeconds: 60)
        }

        // For streaming requests: defer the actual load to inside the response
        // body. The streaming handler wraps inference.stream(...) with
        // loadAwareStream, and withSSEKeepAlive's heartbeat fires while the
        // load runs — so the client never sees dead air during a 30-90s cold
        // load. Without this, the client would block waiting for the response
        // headers/body to start while the eager load completes here.
        if isStreaming {
            return .deferred
        }

        // Non-streaming: do the load now (the connection blocks anyway).
        let deadline = computeColdLoadDeadline(request: request, cfg: cfg)
        let started = Date()

        let options = AutoLoadCoordinator.Options(
            evictOnConflict: cfg.autoLoad.evictOnConflict,
            allowDownload: cfg.autoLoad.allowDownload,
            coldLoadDeadline: deadline
        )

        try await withColdLoadTimeout(deadline: deadline, modelId: modelId) {
            try await coordinator.ensureLoaded(modelId, options: options)
        }

        let coldLoadMs = Int(Date().timeIntervalSince(started) * 1000)
        return .justLoaded(coldLoadMs: coldLoadMs)
    }

    /// Wraps an inference token stream with an optional pre-load step. When the
    /// model isn't loaded and auto-load is enabled, the load runs *before* the
    /// inference stream begins yielding tokens. Combined with withSSEKeepAlive,
    /// this produces SSE `:keep-alive\n\n` heartbeat traffic during the load
    /// window so the connection stays open through cold-load delays.
    private static func loadAwareStream(
        modelId: String,
        inference: InferenceService,
        coordinator: AutoLoadCoordinator,
        autoLoadCfg: AutoLoadConfig,
        inferenceStreamProducer: @Sendable @escaping () -> AsyncThrowingStream<Token, Error>
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    if !inference.isModelLoaded(modelId), autoLoadCfg.enabled {
                        try await coordinator.ensureLoaded(
                            modelId,
                            options: AutoLoadCoordinator.Options(
                                evictOnConflict: autoLoadCfg.evictOnConflict,
                                allowDownload: autoLoadCfg.allowDownload
                            )
                        )
                    }
                    for try await token in inferenceStreamProducer() {
                        if Task.isCancelled { break }
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func computeColdLoadDeadline(request: Request, cfg: ServerConfig) -> Date {
        if let headerVal = request.headers[.init("X-Request-Timeout")!]?.first,
           let secs = Double(String(headerVal)) {
            let capped = min(secs, cfg.autoLoad.coldLoadTimeoutMaxSeconds)
            return Date().addingTimeInterval(capped)
        }

        let base = cfg.requestTimeout
        let multiplied = base * cfg.autoLoad.coldLoadTimeoutMultiplier
        let withFloor = max(multiplied, cfg.autoLoad.coldLoadTimeoutSeconds)
        let capped = min(withFloor, cfg.autoLoad.coldLoadTimeoutMaxSeconds)
        return Date().addingTimeInterval(capped)
    }

    private static func withColdLoadTimeout<T: Sendable>(
        deadline: Date,
        modelId: String,
        operation: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                let interval = deadline.timeIntervalSinceNow
                if interval > 0 {
                    try await Task.sleep(for: .seconds(interval))
                }
                throw NovaMLXError.modelLoadInProgress(
                    modelId: modelId,
                    etaSeconds: 60
                )
            }
            defer { group.cancelAll() }
            guard let result = try await group.next() else {
                throw NovaMLXError.inferenceFailed("cold-load timeout race lost")
            }
            return result
        }
    }

    private static func parseWaitColdLoadHeader(_ request: Request) -> Bool? {
        guard let v = request.headers[.init("X-Wait-Cold-Load")!]?.first?.lowercased() else {
            return nil
        }
        return v == "false" || v == "0" || v == "no" ? false : (v == "true" || v == "1" || v == "yes" ? true : nil)
    }

    // MARK: - Inference Handlers

    private static func handleChat(
        openAIReq: OpenAIRequest, messages: [ChatMessage], inference: InferenceService,
        sessionId: String? = nil, responseFormat: ResponseFormat? = nil, jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil, gbnfGrammar: String? = nil,
        cfg: ServerConfig, clientType: ClientType,
        responseModelOverride: String? = nil
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
            topLogprobsCount: openAIReq.topLogprobs
        )

        CurrentInferenceModel.shared.modelID = request.model
        defer { CurrentInferenceModel.shared.modelID = nil }
        let result = try await inference.generate(request)
        let finishReason: String
        let message: OpenAIChatMessage

        // Scrub control tokens from raw output
        var scrubbedText = result.text
        let shouldParseThinking = openAIReq.resolvedEnableThinking != false
        // When enable_thinking=false, scrub ALL control tokens including think tags
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
            // enable_thinking=false — all output is content, no thinking
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
        return try jsonResponse(response)
    }

    private static func handleStreamChat(
        openAIReq: OpenAIRequest, messages: [ChatMessage], inference: InferenceService,
        sessionId: String? = nil, responseFormat: ResponseFormat? = nil, jsonSchemaDef: [String: Any]? = nil,
        regexPattern: String? = nil, gbnfGrammar: String? = nil,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator,
        responseModelOverride: String? = nil
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
            topLogprobsCount: openAIReq.topLogprobs
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

        let body: ResponseBody = .init { writer in
            CurrentInferenceModel.shared.modelID = openAIReq.model
            defer { CurrentInferenceModel.shared.modelID = nil }
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

                var completionTokenCount = 0
                var promptTokenCount: Int? = nil
                let shouldParseThinking = openAIReq.resolvedEnableThinking != false
                let isImplicitModel = ModelContainer.isImplicitThinkingModel(for: openAIReq.model)
                let thinkingParser = shouldParseThinking ? ThinkingParser(expectImplicitThinking: isImplicitModel) : nil
                // Track content already streamed via feed() so finalize() only
                // emits the unflushed leftover (preCloseContent + remaining buffer).
                // Without this, content that feed() yielded incrementally is also
                // included in finalize().response and would be emitted twice.
                var streamedResponse = ""
                for try await event in keepAliveStream {
                    switch event {
                    case .token(let token):
                        // Compute logprob data once per token. A single decode token
                        // can split into multiple SSE chunks (e.g., ThinkingParser
                        // emits thinking + content separately). Attach logprobs to
                        // only the FIRST emitted chunk to avoid double-counting.
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
                            // Capture prompt token count from engine's final token
                            if let pt = token.promptTokens { promptTokenCount = pt }
                            // Flush ThinkingParser before emitting stop chunk
                            if let tp = thinkingParser {
                                let finalParsed = tp.finalize()
                                // Strip the prefix that was already emitted via feed() so
                                // we only emit the unflushed leftover.
                                var cleanResp = finalParsed.response
                                if !streamedResponse.isEmpty, cleanResp.hasPrefix(streamedResponse) {
                                    cleanResp = String(cleanResp.dropFirst(streamedResponse.count))
                                }
                                // Truncate hallucinated role markers in finalize
                                let hallucPatterns = ["\nuser\n", "\nmodel\n", "\nassistant\n", "user\n", "model\n"]
                                for p in hallucPatterns {
                                    if let range = cleanResp.range(of: p) { cleanResp = String(cleanResp[..<range.lowerBound]) }
                                }
                                if !cleanResp.isEmpty {
                                    completionTokenCount += 1
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
                            var cleanText = token.text
                            if cleanText.contains("<|") || (!shouldParseThinking && cleanText.contains("<think")) {
                                // When enable_thinking=false, scrub ALL control tokens including think tags
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
                                // Drain repeatedly: a single token.text may contain
                                // both a thinking block and content (e.g. when an
                                // upstream scrubber normalizes a non-standard
                                // thinking marker into <think>...</think> in one
                                // shot). feed() emits one type at a time, breaking
                                // on type transitions, so we loop until empty.
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
                                    tokenLogprobs = nil  // consume — only first chunk carries logprobs
                                    let data = try JSONEncoder().encode(chunk)
                                    try await writer.write(ByteBuffer(string: "data: \(String(data: data, encoding: .utf8) ?? "")\n\n"))
                                }
                            } else {
                                // enable_thinking=false — all output is content
                                if cleanText.isEmpty { continue }
                                let delta = OpenAIDelta(content: cleanText)
                                let chunk = OpenAIStreamChunk(
                                    id: chunkId,
                                    model: responseModel,
                                    choices: [OpenAIStreamChoice(index: 0, delta: delta, logprobs: tokenLogprobs)],
                                    novaChannels: token.channels?.map { NovaHarmonyChannel(channel: $0.channel, text: $0.text) }
                                )
                                tokenLogprobs = nil  // consume — only first chunk carries logprobs
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
                Self.applyKeepAlive(openAIReq.keepAlive, modelId: openAIReq.model, pool: inference.engine.pool)
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
        anthropicReq: AnthropicRequest, messages: [ChatMessage], inference: InferenceService,
        cfg: ServerConfig, clientType: ClientType,
        coordinator: AutoLoadCoordinator
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
            preserveThinking: anthropicReq.resolvedPreserveThinking
        )

        let reqTag = request.id.uuidString.prefix(8)
        NovaMLXLog.info("[SSE:\(reqTag)] Anthropic stream request started — model=\(anthropicReq.model), maxTokens=\(anthropicReq.maxTokens)")

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
                // Track content already streamed via feed() so finalize() only
                // emits leftover. Mirrors the OpenAI streaming path. See comment
                // in handleStreamChat for rationale.
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
                            // Flush ThinkingParser before closing blocks
                            if let tp = thinkingParser {
                                let finalParsed = tp.finalize()
                                // Strip prefix already emitted via feed()
                                var cleanFinalResp = finalParsed.response
                                if !streamedResponse.isEmpty, cleanFinalResp.hasPrefix(streamedResponse) {
                                    cleanFinalResp = String(cleanFinalResp.dropFirst(streamedResponse.count))
                                }
                                // Truncate hallucinated role markers
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
                            // Close current block if open
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
                            let elapsed = Date().timeIntervalSince(streamStart)
                            NovaMLXLog.info("[SSE:\(reqTag)] Stream complete — \(tokenCount) tokens in \(String(format: "%.1f", elapsed))s, usage: input=\(scaledIn) output=\(scaledOut)")
                        } else if !token.text.isEmpty {
                            if tokenCount == 0 {
                                let ttft = Date().timeIntervalSince(streamStart)
                                NovaMLXLog.info("[SSE:\(reqTag)] First token received (TTFT=\(String(format: "%.1f", ttft))s)")
                            }
                            tokenCount += 1

                            // Scrub control tokens from stream chunk
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

                            // Drain repeatedly: see handleStreamChat for rationale.
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
                                // No thinking parser — emit text directly
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
                NovaMLXLog.error("[SSE:\(reqTag)] Stream ERROR after \(String(format: "%.1f", elapsed))s, \(tokenCount) tokens sent: \(error)")
                // Use Anthropic error format for Anthropic API endpoint
                let (message, errorType) = Self.anthropicStreamErrorFields(error)
                let errorDetail = AnthropicErrorDetail(type: errorType, message: message)
                let errorResp = AnthropicErrorResponse(error: errorDetail)
                if let data = try? JSONEncoder().encode(errorResp) {
                    try? await writer.write(ByteBuffer(string: "event: error\ndata: \(String(data: data, encoding: .utf8) ?? "{}")\n\n"))
                }
                Self.applyKeepAlive(anthropicReq.keepAlive, modelId: anthropicReq.model, pool: inference.engine.pool)
                try? await writer.finish(nil)
                return
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
            stream: false, stop: compReq.stop
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

    private static func handleStreamCompletion(
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
            stream: true, stop: compReq.stop
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

    private static func handleResponsesRequest(
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

        // Resolve previous_response_id: prepend stored messages
        if let prevId = req.previousResponseId {
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
        // Main text output
        outputItems.append(.message(ResponseOutputMessage(
            id: outputItemId,
            content: [ResponseContentItem(text: result.text)]
        )))
        // Tool calls
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
        )
        // Store with user messages prepended for previous_response_id multi-turn
        var storeOutputItems: [ResponseOutputItem] = []
        for msg in messages {
            let itemId = "msg_user_\(UUID().uuidString.prefix(12))"
            storeOutputItems.append(.message(ResponseOutputMessage(id: itemId, role: msg.role.rawValue, content: [ResponseContentItem(text: msg.content ?? "")])))
        }
        storeOutputItems.append(contentsOf: outputItems)
        let storeResponse = OpenAIResponseObject(id: responseId, model: result.model, output: storeOutputItems, usage: ResponsesUsage(inputTokens: scaledInput, outputTokens: scaledOutput))
        ResponseStore.shared.put(storeResponse)
        return try jsonResponse(response)
    }

    /// Extract ChatMessages from a stored response for conversation continuation
    private static func extractMessagesFromResponse(_ response: OpenAIResponseObject) -> [ChatMessage] {
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
                break  // Reasoning items are not converted to chat messages
            }
        }
        return messages
    }

    // MARK: - Responses API Streaming

    private static func handleStreamResponses(
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

            func sse(_ event: String, _ data: Encodable) async throws {
                let jsonData = try encoder.encode(data)
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
                            // Feed through ThinkingParser to separate reasoning from text
                            let parsed = thinkingParser.feed(token.text)
                            if parsed.type == .thinking && !parsed.text.isEmpty {
                                if !reasoningStarted {
                                    reasoningStarted = true
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, status: "in_progress"))))
                                }
                                reasoningText += parsed.text
                                try await sse("response.reasoning.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: parsed.text))
                            }
                            if parsed.type == .content && !parsed.text.isEmpty {
                                if !textMessageStarted {
                                    // Finish reasoning if active
                                    if reasoningStarted {
                                        let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                                        try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
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
                            // Finalize any remaining thinking content
                            let finalParsed = thinkingParser.finalize()
                            if !finalParsed.thinking.isEmpty {
                                if !reasoningStarted {
                                    reasoningStarted = true
                                    try await sse("response.output_item.added", ResponsesSSEOutputItemAdded(outputIndex: currentOutputIndex, item: .reasoning(ResponseOutputReasoning(id: rsId, status: "in_progress"))))
                                }
                                reasoningText += finalParsed.thinking
                                try await sse("response.reasoning.delta", ResponsesSSEReasoningDelta(itemId: rsId, outputIndex: currentOutputIndex, delta: finalParsed.thinking))
                            }
                            if reasoningStarted {
                                let summary = reasoningText.isEmpty ? nil : [ResponsesReasoningSummary(text: String(reasoningText.prefix(500)))]
                                try await sse("response.reasoning.done", ResponsesSSEReasoningDone(itemId: rsId, outputIndex: currentOutputIndex, summary: summary))
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
                            // Finish text message
                            if textMessageStarted {
                                try await sse("response.output_text.done", ResponsesSSETextDone(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, text: fullText))
                                try await sse("response.content_part.done", ResponsesSSEContentPartDone(itemId: msgId, outputIndex: currentOutputIndex, contentIndex: 0, part: ResponseContentItem(text: fullText)))
                                try await sse("response.output_item.done", ResponsesSSEOutputItemDone(outputIndex: currentOutputIndex, item: .message(ResponseOutputMessage(id: msgId, content: [ResponseContentItem(text: fullText)]))))
                                outputItems.append(.message(ResponseOutputMessage(id: msgId, content: [ResponseContentItem(text: fullText)])))
                            }

                            // response.completed
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
                            )
                            try await sse("response.completed", ResponsesSSECompleted(response: completedResp))
                            // Store with user messages for previous_response_id multi-turn
                            var storeOutputItems: [ResponseOutputItem] = []
                            for msg in capturedMessages {
                                let itemId = "msg_user_\(UUID().uuidString.prefix(12))"
                                storeOutputItems.append(.message(ResponseOutputMessage(id: itemId, role: msg.role.rawValue, content: [ResponseContentItem(text: msg.content ?? "")])))
                            }
                            storeOutputItems.append(contentsOf: outputItems)
                            let storeResp = OpenAIResponseObject(id: responseId, model: modelId, output: storeOutputItems, usage: ResponsesUsage(inputTokens: pToks, outputTokens: cToks))
                            ResponseStore.shared.put(storeResp)
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
                try? await writer.finish(nil)
            }
        }

        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream", .cacheControl: "no-cache", .connection: "keep-alive", .init("X-Accel-Buffering")!: "no"],
            body: body
        )
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
            // Shared guard prevents double-yield/finish when onTermination
            // races with the inference consumer or heartbeat tasks.
            let guard_ = FinishGuard()

            let task = Task {
                do {
                    guard !guard_.isDone else { return }
                    continuation.yield(.keepAlive)
                    for try await token in stream {
                        if Task.isCancelled {
                            NovaMLXLog.debug("[SSE:\(reqTag)] Inference stream consumer cancelled")
                            break
                        }
                        guard !guard_.isDone else { return }
                        continuation.yield(.token(token))
                    }
                    NovaMLXLog.debug("[SSE:\(reqTag)] Inference stream finished normally")
                    if guard_.tryMarkFinished() {
                        continuation.finish()
                    }
                } catch {
                    NovaMLXLog.error("[SSE:\(reqTag)] Inference stream error: \(error)")
                    if guard_.tryMarkFinished() {
                        continuation.finish(throwing: error)
                    }
                }
            }
            let heartbeat = Task {
                while !Task.isCancelled {
                    try? await Task.sleep(for: interval)
                    guard !Task.isCancelled else { break }
                    guard !guard_.isDone else { return }
                    continuation.yield(.keepAlive)
                }
            }
            continuation.onTermination = { reason in
                // Command-009: finished(nil) is the normal AsyncThrowingStream completion
                // (no error thrown). Only WARN on real failures; normal close is DEBUG.
                if case .finished(let error?) = reason {
                    NovaMLXLog.warning("[SSE:\(reqTag)] SSE connection terminated with error: \(error)")
                } else {
                    NovaMLXLog.debug("[SSE:\(reqTag)] SSE connection terminated: \(reason)")
                }
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

    // MARK: - Admin API Proxy

    private static func proxyAdminRequest(path: String, method: String, body: ByteBuffer?, cfg: ServerConfig) async throws -> Response {
        let targetURL = "http://127.0.0.1:\(cfg.adminPort)\(path)"
        guard let url = URL(string: targetURL) else {
            throw NovaMLXError.apiError("Invalid proxy target: \(targetURL)")
        }
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = method
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey = cfg.apiKeys.first {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            urlRequest.httpBody = Data(buffer: body)
        }
        let (data, resp) = try await URLSession.shared.data(for: urlRequest)
        guard let httpResp = resp as? HTTPURLResponse else {
            throw NovaMLXError.apiError("Invalid response from admin server")
        }
        let status = HTTPResponse.Status(code: httpResp.statusCode)
        var headers: HTTPFields = [.contentType: httpResp.value(forHTTPHeaderField: "Content-Type") ?? "application/json"]
        if let cacheControl = httpResp.value(forHTTPHeaderField: "Cache-Control") {
            headers[.cacheControl] = cacheControl
        }
        return Response(status: status, headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
    }

    private static func jsonResponse<T: Encodable>(_ value: T) throws -> Response {
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

    private static func jsonResponse<T: Encodable>(_ value: T, httpStatus: HTTPResponse.Status) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(
            status: httpStatus,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    /// Convert raw PNG/JPEG data to a CGImage.
    private static func dataToCGImage(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
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
