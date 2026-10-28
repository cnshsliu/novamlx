import Foundation
import HTTPTypes
import Hummingbird
import NIOCore
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils

// MARK: - Vendor OCR routes
//
// Baidu, Aliyun, and Tencent general-OCR shapes. Each one calls the same
// Unlimited-OCR-2 chat path (`document parsing.` plus the image) and maps
// the markup into that vendor's JSON. `/v1/chat/completions` is unchanged.
//
// Auth is the Nova Bearer token or `x-api-key`. Baidu `access_token`,
// Tencent `SecretId`, and ACS / TC3 `Authorization` signatures are not
// credentials. A vendor signature in `Authorization` does not authenticate
// the request; send the Nova key as `Bearer` or `x-api-key`.

struct VendorOCRMiddleware: RouterMiddleware {
    typealias Context = AppContext

    let inference: InferenceService
    let cfg: ServerConfig
    let embeddings: EmbeddingService
    let coordinator: AutoLoadCoordinator

    func handle(
        _ request: Request,
        context: Context,
        next: (Request, Context) async throws -> Response
    ) async throws -> Response {
        guard request.method == .post else {
            return try await next(request, context)
        }
        let path = VendorOCRCatalog.normalize(request.uri.path)
        guard VendorOCRCatalog.isVendorOCRFamily(path) else {
            return try await next(request, context)
        }
        if path == VendorOCRCatalog.actionPath {
            return try await NovaMLXAPIServer.handleVendorOCRAction(
                request: request,
                inference: inference,
                cfg: cfg,
                embeddings: embeddings,
                coordinator: coordinator
            )
        }
        if let route = VendorOCRCatalog.route(path: path, action: nil) {
            return try await NovaMLXAPIServer.handleVendorOCR(
                request: request,
                route: route,
                inference: inference,
                cfg: cfg,
                embeddings: embeddings,
                coordinator: coordinator
            )
        }
        let query = NovaMLXAPIServer.vendorOCRQuery(request)
        let action = request.headers[HTTPField.Name("X-TC-Action")!] ?? query["Action"]
        let vendor = VendorOCRCatalog.vendorForUnknown(path: path, action: action)
        return NovaMLXAPIServer.vendorOCRFailureResponse(
            .unsupported(VendorOCRCatalog.unsupportedMessage),
            vendor: vendor,
            request: request
        )
    }
}

extension NovaMLXAPIServer {
    static func handleVendorOCRAction(
        request: Request,
        inference: InferenceService,
        cfg: ServerConfig,
        embeddings: EmbeddingService,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let body = try await vendorOCRBody(request)
        let action = vendorOCRAction(request: request, body: body)
        if let route = VendorOCRCatalog.route(path: VendorOCRCatalog.actionPath, action: action) {
            return try await handleVendorOCR(
                request: request,
                route: route,
                body: body,
                inference: inference,
                cfg: cfg,
                embeddings: embeddings,
                coordinator: coordinator
            )
        }
        let vendor = VendorOCRCatalog.vendorForUnknown(path: VendorOCRCatalog.actionPath, action: action)
        let message = action == nil
            ? "Action is required on POST /ocr. Use /ocr/RecognizeGeneral, /ocr/GeneralBasicOCR, /ocr/GeneralAccurateOCR, or a Baidu /rest/2.0/ocr/v1 path."
            : VendorOCRCatalog.unsupportedMessage
        return vendorOCRFailureResponse(.unsupported(message), vendor: vendor, request: request)
    }

    static func handleVendorOCR(
        request: Request,
        route: VendorOCRRoute,
        body suppliedBody: Data? = nil,
        inference: InferenceService,
        cfg: ServerConfig,
        embeddings: EmbeddingService,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let logID = VendorOCRResponses.makeLogID()
        let requestID = HTTPHelpers.requestID(from: request) ?? UUID().uuidString
        do {
            return try await performVendorOCR(
                request: request,
                route: route,
                body: suppliedBody,
                logID: logID,
                requestID: requestID,
                inference: inference,
                cfg: cfg,
                embeddings: embeddings,
                coordinator: coordinator
            )
        } catch {
            let payload = VendorOCRResponses.failure(
                VendorOCRFailure.from(error),
                vendor: route.vendor,
                logID: logID,
                requestID: requestID
            )
            return vendorOCRResponse(payload)
        }
    }

    static func vendorOCRFailureResponse(
        _ failure: VendorOCRFailure,
        vendor: VendorOCRVendor,
        request: Request
    ) -> Response {
        let payload = VendorOCRResponses.failure(
            failure,
            vendor: vendor,
            logID: VendorOCRResponses.makeLogID(),
            requestID: HTTPHelpers.requestID(from: request) ?? UUID().uuidString
        )
        return vendorOCRResponse(payload)
    }

    static func vendorOCRQuery(_ request: Request) -> [String: String] {
        guard let raw = request.uri.query, !raw.isEmpty else { return [:] }
        return VendorOCRRequestParser.formFields(Data(raw.utf8))
    }

    private static func performVendorOCR(
        request: Request,
        route: VendorOCRRoute,
        body suppliedBody: Data?,
        logID: Int,
        requestID: String,
        inference: InferenceService,
        cfg: ServerConfig,
        embeddings: EmbeddingService,
        coordinator: AutoLoadCoordinator
    ) async throws -> Response {
        let body: Data
        if let suppliedBody {
            body = suppliedBody
        } else {
            body = try await vendorOCRBody(request)
        }
        let parsed = try VendorOCRRequestParser.parse(
            body: body,
            contentType: request.headers[.contentType] ?? "",
            query: vendorOCRQuery(request),
            headerModel: request.headers[HTTPField.Name("X-Nova-Model")!]
        )
        if Self.isMovedToPeerGateway(parsed.modelID) {
            throw VendorOCRRouteError.unavailable(Self.peerGatewayMovedMessage)
        }

        NovaMLXLog.info("[API] POST \(request.uri.path) vendor=\(route.vendor.rawValue) product=\(route.product.rawValue) model=\(parsed.modelID)")

        let outcome = try await Self.ensureModelReady(
            modelId: parsed.modelID,
            isStreaming: false,
            cfg: cfg,
            inference: inference,
            embeddings: embeddings,
            coordinator: coordinator,
            request: request
        )

        let prepared = try prepareEngineImage(parsed.image)
        defer {
            if let temporary = prepared.temporaryFile {
                try? FileManager.default.removeItem(at: temporary)
            }
        }

        let sampling = VendorOCREngine.sampling(modelID: parsed.modelID)
        let inferenceRequest = InferenceRequest(
            model: parsed.modelID,
            messages: [VendorOCREngine.userMessage(image: prepared.reference)],
            temperature: sampling.temperature,
            maxTokens: sampling.maxTokens,
            repetitionPenalty: sampling.repetitionPenalty,
            stream: false,
            stop: OCROptimizer.applyStopSequences(nil, modelName: parsed.modelID),
            enableThinking: false,
            httpRequestId: requestID
        )

        CurrentInferenceModel.shared.modelID = parsed.modelID
        defer { CurrentInferenceModel.shared.modelID = nil }
        let result = try await inference.generate(inferenceRequest)
        var pixelSize = prepared.pixelSize
        if pixelSize == nil, let path = prepared.localPath,
           let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
           data.count <= 32 * 1024 * 1024
        {
            pixelSize = VendorOCRImages.pixelSize(of: data)
        }
        let lines = UnlimitedOCRMarkup.parse(result.text, imagePixelSize: pixelSize)
        var response = vendorOCRResponse(VendorOCRResponses.success(
            vendor: route.vendor,
            lines: lines,
            logID: logID,
            requestID: requestID,
            imageSize: pixelSize
        ))
        if case .justLoaded(let ms) = outcome {
            response.headers[.init("X-Model-Cold-Load")!] = "true"
            response.headers[.init("X-Model-Load-Time-Ms")!] = "\(ms)"
        }
        Self.recordTokenUsage(
            request: request,
            promptTokens: result.promptTokens,
            completionTokens: result.completionTokens,
            model: parsed.modelID,
            endpoint: request.uri.path
        )
        return response
    }

    private static func vendorOCRBody(_ request: Request) async throws -> Data {
        let buffer = try await request.body.collect(upTo: .max)
        return Data(buffer: buffer)
    }

    /// `X-TC-Action`, then the `Action` query, then a JSON `Action` field.
    private static func vendorOCRAction(request: Request, body: Data) -> String? {
        if let header = nonempty(request.headers[HTTPField.Name("X-TC-Action")!]) {
            return header
        }
        if let query = nonempty(vendorOCRQuery(request)["Action"]) {
            return query
        }
        let type = request.headers[.contentType]?.lowercased() ?? ""
        let looksJSON = type.contains("json") || body.first == UInt8(ascii: "{")
        guard looksJSON,
              let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
              let action = nonempty(object["Action"] as? String)
        else { return nil }
        return action
    }

    private static func nonempty(_ value: String?) -> String? {
        guard let value else { return nil }
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private struct PreparedEngineImage {
        var reference: String
        var pixelSize: OCRPixelSize?
        var localPath: String?
        var temporaryFile: URL?
    }

    private static func prepareEngineImage(_ image: VendorOCRImageSource) throws -> PreparedEngineImage {
        switch image {
        case .pdf(let data):
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("novamlx-ocr-\(UUID().uuidString).pdf")
            do {
                try data.write(to: url, options: .atomic)
            } catch {
                throw VendorOCRParseError.imageFormat("could not store the PDF upload")
            }
            return PreparedEngineImage(reference: url.path, pixelSize: nil, localPath: nil, temporaryFile: url)
        case .dataURL(let value, let size):
            return PreparedEngineImage(reference: value, pixelSize: size, localPath: nil, temporaryFile: nil)
        case .remoteURL(let value):
            return PreparedEngineImage(reference: value, pixelSize: nil, localPath: nil, temporaryFile: nil)
        case .filePath(let value):
            return PreparedEngineImage(reference: value, pixelSize: nil, localPath: value, temporaryFile: nil)
        }
    }

    private static func vendorOCRResponse(_ payload: VendorOCRPayload) -> Response {
        let data = VendorOCRResponses.encode(payload)
        return Response(
            status: vendorOCRHTTPStatus(payload.status),
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: ByteBuffer(data: data))
        )
    }

    private static func vendorOCRHTTPStatus(_ code: Int) -> HTTPResponse.Status {
        switch code {
        case 400: .badRequest
        case 500: .internalServerError
        case 503: .serviceUnavailable
        default: .ok
        }
    }
}
