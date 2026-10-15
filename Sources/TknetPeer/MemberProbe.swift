import AsyncHTTPClient
import Foundation
import NIOCore

/// Outcome of one member test. `ok` means the member is callable: a bare
/// invocation through its interface template succeeded (or, for quota-heavy
/// kinds, the endpoint answered an authed discovery request).
public struct MemberProbeResult: Equatable, Sendable {
    public let ok: Bool
    public let httpStatus: Int
    /// Model ids found by discovery (`discoverModels`). Empty for tests.
    public let models: [String]
    /// `nova.type` per model id from discovery, for interface mapping.
    public let modelTypes: [String: String]
    /// Human-facing summary, safe to show inline in the member list.
    public let message: String

    public init(ok: Bool, httpStatus: Int, models: [String],
                modelTypes: [String: String] = [:], message: String) {
        self.ok = ok
        self.httpStatus = httpStatus
        self.models = models
        self.modelTypes = modelTypes
        self.message = message
    }
}

/// Member testing (2026-10-15): a member is a bare-callable unit — full
/// interface URL + model name + calling template. `test` invokes it for real
/// with a minimal payload; kinds that would burn quota or take seconds
/// (image generation, speech synthesis, …) fall back to an authed discovery
/// request unless forced deep.
public enum MemberProbe {
    /// Shared client; lives for the process lifetime so it is never dropped
    /// un-shutdown (AsyncHTTPClient traps on that in debug builds).
    private static let client = HTTPClient(eventLoopGroupProvider: .singleton)

    // MARK: - Discovery

    /// `GET {base-with-/v1}/models` — the model list a factory advertises.
    public static func discoverModels(base: URL, apiKey: String?) async -> MemberProbeResult {
        let url = UpstreamBodyCodec.upstreamURL(endpoint: modelsBase(base), path: "models")
        var request = HTTPClientRequest(url: url)
        request.method = .GET
        if let apiKey, !apiKey.isEmpty {
            request.headers.add(name: "authorization", value: "Bearer \(apiKey)")
        }
        do {
            let response = try await client.execute(request, timeout: .seconds(15))
            let status = Int(response.status.code)
            let body = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
            guard (200..<300).contains(status) else {
                let detail = UpstreamErrorText.message(from: body, status: status)
                return MemberProbeResult(
                    ok: false, httpStatus: status, models: [],
                    message: "HTTP \(status) — \(detail)")
            }
            let models = parseModels(body)
            let types = parseModelTypes(body)
            let summary = models.isEmpty
                ? "HTTP \(status)"
                : "HTTP \(status) · \(models.count) models"
            return MemberProbeResult(
                ok: true, httpStatus: status, models: models, modelTypes: types, message: summary)
        } catch {
            return MemberProbeResult(
                ok: false, httpStatus: 0, models: [], message: errorSummary(error))
        }
    }

    /// `{scheme://host[:port]}/v1` for discovery off a factory base.
    public static func modelsBase(_ base: URL) -> URL {
        var parts = URLComponents(url: base, resolvingAgainstBaseURL: false) ?? URLComponents()
        var path = parts.path
        while path.hasSuffix("/") { path.removeLast() }
        if !path.hasSuffix("/v1") { path += "/v1" }
        parts.path = path
        parts.query = nil
        return parts.url ?? base
    }

    /// OpenAI-style `{"data":[{"id":…}]}` (Anthropic uses the same shape).
    /// Unknown or malformed bodies yield `[]`, never a throw.
    public static func parseModels(_ data: Data) -> [String] {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = object["data"] as? [[String: Any]] else { return [] }
        var seen = Set<String>()
        var ids: [String] = []
        for item in list {
            guard let id = item["id"] as? String, !id.isEmpty, seen.insert(id).inserted else { continue }
            ids.append(id)
        }
        return ids.sorted()
    }

    /// `nova.type` per model id from the same discovery body. NovaMLX nests
    /// it under each entry; other servers simply have no entry.
    public static func parseModelTypes(_ data: Data) -> [String: String] {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let list = object["data"] as? [[String: Any]] else { return [:] }
        var types: [String: String] = [:]
        for item in list {
            guard let id = item["id"] as? String else { continue }
            if let nova = item["nova"] as? [String: Any],
               let type = nova["type"] as? String, !type.isEmpty {
                types[id] = type
            }
        }
        return types
    }

    /// True for NovaMLX DSpark draft/companion blocks (speculators, `_block7`
    /// companions, `…_novamlx` draft variants). They ride along with a main
    /// model and are never standalone members, so imports skip them.
    public static func isDraftModel(_ modelId: String) -> Bool {
        modelId.lowercased().contains("dspark")
    }

    /// NovaMLX `/v1/models` advertises each model's `nova.type`. Map it (plus
    /// the model id for audio disambiguation) onto the interface a member of
    /// that model should use. Nil when the type is not importable.
    public static func interfaceKind(modelType: String, modelId: String) -> InterfaceKind? {
        let type = modelType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let name = modelId.split(separator: "/").last.map { $0.lowercased() } ?? modelId.lowercased()
        switch type {
        case "llm", "vlm", "":
            return .openaiChat
        case "embedding":
            return .openaiEmbeddings
        case "image", "image_generation":
            return .openaiImages
        case "decision":
            return .novaDecision
        case "audio":
            if name.contains("asr") || name.contains("whisper") { return .openaiTranscription }
            return .openaiSpeech
        default:
            return nil
        }
    }

    // MARK: - Bare-call test

    /// Test one member the honest way: a real invocation with the smallest
    /// payload that proves the member produces output. No shortcuts — a
    /// passing test means this member actually answered (Lucas 2026-10-15:
    /// discovery-request "light tests" masked wrong keys and had to go).
    /// `timeoutSeconds` defaults generously: image generation takes a while.
    public static func test(
        route: MemberRoute, model: String, apiKey: String?,
        timeoutSeconds: Double = 180
    ) async -> MemberProbeResult {
        var request = HTTPClientRequest(url: route.url.absoluteString)
        request.method = .POST
        if let apiKey, !apiKey.isEmpty {
            if route.kind.usesXAPIKey {
                request.headers.add(name: "x-api-key", value: apiKey)
                request.headers.add(name: "anthropic-version", value: "2023-06-01")
            } else {
                request.headers.add(name: "authorization", value: "Bearer \(apiKey)")
            }
        }
        if let json = barePayload(kind: route.kind, model: model) {
            request.headers.add(name: "content-type", value: "application/json")
            request.body = .bytes(Data(json.utf8))
        } else if let (body, contentType) = bareMultipartBody(kind: route.kind, model: model) {
            request.headers.add(name: "content-type", value: contentType)
            request.body = .bytes(body)
        } else {
            return MemberProbeResult(
                ok: false, httpStatus: 0, models: [],
                message: "no test template for \(route.kind.rawValue)")
        }
        do {
            let response = try await client.execute(
                request, timeout: .seconds(max(1, Int64(timeoutSeconds.rounded()))))
            let status = Int(response.status.code)
            let body = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
            guard (200..<300).contains(status) else {
                let detail = UpstreamErrorText.message(from: body, status: status)
                return MemberProbeResult(
                    ok: false, httpStatus: status, models: [],
                    message: "HTTP \(status) — \(detail)")
            }
            guard responseLooksComplete(kind: route.kind, body: body) else {
                return MemberProbeResult(
                    ok: false, httpStatus: status, models: [],
                    message: "HTTP \(status) but the response has no content — not callable")
            }
            return MemberProbeResult(
                ok: true, httpStatus: status, models: [],
                message: "HTTP \(status) · bare call ok")
        } catch {
            return MemberProbeResult(
                ok: false, httpStatus: 0, models: [], message: errorSummary(error))
        }
    }

    /// Factory base behind one route: the full interface URL minus the
    /// canonical kind path.
    public static func factoryBase(of route: MemberRoute) -> URL {
        var text = route.url.absoluteString
        let path = route.kind.path
        if text.hasSuffix(path) {
            text.removeLast(path.count)
        }
        while text.hasSuffix("/") { text.removeLast() }
        return URL(string: text) ?? route.url
    }

    /// Minimal real-invocation body per kind. `max_tokens`/`max_output_tokens`
    /// of 1 keeps the cost at a single token.
    static func barePayload(kind: InterfaceKind, model: String) -> String? {
        let encoded = model.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        switch kind {
        case .openaiChat:
            return #"{"model":"\#(encoded)","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"stream":false}"#
        case .novaDecision:
            // Jev/Laya standard body: {model, state, questions}. Every
            // question needs type + instructions; criteria is optional for
            // noul. The noul example mirrors the TypeSafe API docs.
            return #"{"model":"\#(encoded)","state":"ping","questions":{"q1":{"type":"noul","instructions":"Does this convey urgency?"}}}"#
        case .anthropicMessages:
            return #"{"model":"\#(encoded)","messages":[{"role":"user","content":"ping"}],"max_tokens":1}"#
        case .openaiResponses:
            return #"{"model":"\#(encoded)","input":"ping","max_output_tokens":16}"#
        case .openaiEmbeddings:
            return #"{"model":"\#(encoded)","input":"ping"}"#
        case .openaiSpeech:
            return #"{"model":"\#(encoded)","input":"hi","voice":"alloy"}"#
        case .openaiImages:
            return #"{"model":"\#(encoded)","prompt":"a single black dot on a white background","n":1}"#
        default:
            return nil
        }
    }

    /// Multipart templates for the file-bearing kinds: a 1x1 PNG image and a
    /// 100ms silent WAV, both tiny and valid.
    static func bareMultipartBody(
        kind: InterfaceKind, model: String
    ) -> (body: Data, contentType: String)? {
        switch kind {
        case .openaiImageEdits:
            return multipart(fields: [
                ("model", nil, Data(model.utf8)),
                ("prompt", nil, Data("keep it".utf8)),
                ("image", "probe.png", tinyPNG),
            ])
        case .openaiTranscription:
            return multipart(fields: [
                ("model", nil, Data(model.utf8)),
                ("file", "probe.wav", silentWAV),
            ])
        case .novaBackgroundRemoval:
            return multipart(fields: [
                ("model", nil, Data(model.utf8)),
                ("image", "probe.png", tinyPNG),
            ])
        default:
            return nil
        }
    }

    /// 1x1 transparent PNG.
    static let tinyPNG = Data(base64Encoded:
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")!

    /// ~100ms of 8kHz 16-bit mono silence — a valid, parseable WAV.
    static let silentWAV: Data = {
        let sampleRate = 8000
        let frames = sampleRate / 10
        let dataSize = frames * 2
        var data = Data()
        func append(_ value: UInt32) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        func append16(_ value: UInt16) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(Data("RIFF".utf8))
        append(UInt32(36 + dataSize))
        data.append(Data("WAVE".utf8))
        data.append(Data("fmt ".utf8))
        append(UInt32(16)); append16(1); append16(1)
        append(UInt32(sampleRate)); append(UInt32(sampleRate * 2))
        append16(2); append16(16)
        data.append(Data("data".utf8))
        append(UInt32(dataSize))
        data.append(Data(count: dataSize))
        return data
    }()

    static func multipart(
        fields: [(name: String, filename: String?, content: Data)]
    ) -> (body: Data, contentType: String) {
        let boundary = "tknet-probe-\(UUID().uuidString)"
        var body = Data()
        func line(_ text: String) { body.append(Data(text.utf8)) }
        for field in fields {
            line("--\(boundary)\r\n")
            if let filename = field.filename {
                line("Content-Disposition: form-data; name=\"\(field.name)\"; filename=\"\(filename)\"\r\n")
                line("Content-Type: application/octet-stream\r\n\r\n")
            } else {
                line("Content-Disposition: form-data; name=\"\(field.name)\"\r\n\r\n")
            }
            body.append(field.content)
            line("\r\n")
        }
        line("--\(boundary)--\r\n")
        return (body, "multipart/form-data; boundary=\(boundary)")
    }

    /// 2xx is not enough (peer-bot's standard): the body must carry real
    /// content for the kind — a choice, a vector, or output text.
    static func responseLooksComplete(kind: InterfaceKind, body: Data) -> Bool {
        // Audio replies are raw bytes, not JSON.
        if kind == .openaiSpeech { return !body.isEmpty }
        guard let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            return false
        }
        switch kind {
        case .openaiChat, .anthropicMessages:
            if let choices = object["choices"] as? [[String: Any]], !choices.isEmpty { return true }
            if let content = object["content"] as? [[String: Any]], !content.isEmpty { return true }
            return false
        case .openaiResponses:
            return (object["output"] as? [[String: Any]])?.isEmpty == false
        case .openaiEmbeddings:
            return (object["data"] as? [[String: Any]])?.isEmpty == false
        case .novaDecision:
            return (object["answers"] as? [String: Any])?.isEmpty == false
        case .openaiImages, .openaiImageEdits, .novaBackgroundRemoval:
            return (object["data"] as? [[String: Any]])?.isEmpty == false
        case .openaiTranscription:
            return object["text"] != nil
        default:
            return true
        }
    }

    private static func errorSummary(_ error: Error) -> String {
        let text = String(describing: error)
        return text.split(separator: "(").first.map(String.init) ?? text
    }
}
