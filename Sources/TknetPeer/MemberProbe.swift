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

    public enum Depth: Equatable, Sendable {
        /// Real invocation with a minimal payload — the token-factory grade.
        case full
        /// Authed discovery request: proves reachability + auth, not tokens.
        case light
    }

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

    /// Test one member by invoking it. Depth defaults from the kind:
    /// quota-heavy kinds get `.light` unless `deep` is true.
    public static func test(
        route: MemberRoute, model: String, apiKey: String?,
        deep: Bool = false, timeoutSeconds: Double = 30
    ) async -> MemberProbeResult {
        let depth: Depth = (deep || !route.kind.defaultsToLightProbe) ? .full : .light
        if depth == .light {
            var result = await discoverModels(base: route.url, apiKey: apiKey)
            result = MemberProbeResult(
                ok: result.ok, httpStatus: result.httpStatus, models: [],
                message: result.ok
                    ? "auth ok (light test — \(route.kind.displayName))"
                    : result.message)
            return result
        }
        guard let payload = barePayload(kind: route.kind, model: model) else {
            return MemberProbeResult(
                ok: false, httpStatus: 0, models: [],
                message: "no test template for \(route.kind.rawValue)")
        }
        var request = HTTPClientRequest(url: route.url.absoluteString)
        request.method = .POST
        request.headers.add(name: "content-type", value: "application/json")
        if let apiKey, !apiKey.isEmpty {
            if route.kind.usesXAPIKey {
                request.headers.add(name: "x-api-key", value: apiKey)
                request.headers.add(name: "anthropic-version", value: "2023-06-01")
            } else {
                request.headers.add(name: "authorization", value: "Bearer \(apiKey)")
            }
        }
        request.body = .bytes(Data(payload.utf8))
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

    /// Minimal real-invocation body per kind. `max_tokens`/`max_output_tokens`
    /// of 1 keeps the cost at a single token.
    static func barePayload(kind: InterfaceKind, model: String) -> String? {
        let encoded = model.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
        switch kind {
        case .openaiChat, .novaDecision:
            return #"{"model":"\#(encoded)","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"stream":false}"#
        case .anthropicMessages:
            return #"{"model":"\#(encoded)","messages":[{"role":"user","content":"ping"}],"max_tokens":1}"#
        case .openaiResponses:
            return #"{"model":"\#(encoded)","input":"ping","max_output_tokens":16}"#
        case .openaiEmbeddings:
            return #"{"model":"\#(encoded)","input":"ping"}"#
        default:
            return nil
        }
    }

    /// 2xx is not enough (peer-bot's standard): the body must carry real
    /// content for the kind — a choice, a vector, or output text.
    static func responseLooksComplete(kind: InterfaceKind, body: Data) -> Bool {
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
        default:
            return true
        }
    }

    private static func errorSummary(_ error: Error) -> String {
        let text = String(describing: error)
        return text.split(separator: "(").first.map(String.init) ?? text
    }
}
