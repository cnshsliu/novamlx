import AsyncHTTPClient
import Foundation
import NIOCore

/// Outcome of one member connectivity test. `ok` means the endpoint answered
/// 2xx on `GET {endpoint}/models` — reachable and authorized, no tokens spent.
public struct MemberProbeResult: Equatable, Sendable {
    public let ok: Bool
    public let httpStatus: Int
    /// Model ids advertised by the upstream, deduplicated and sorted. Empty
    /// when the body was not an OpenAI-style `{"data":[{"id":…}]}` list.
    public let models: [String]
    /// Human-facing summary, safe to show inline in the member list.
    public let message: String

    public init(ok: Bool, httpStatus: Int, models: [String], message: String) {
        self.ok = ok
        self.httpStatus = httpStatus
        self.models = models
        self.message = message
    }
}

/// One-shot health check for a member endpoint. Mirrors `PoolForwarder`'s
/// request shape (Bearer key, `/models` under the member's base) so a passing
/// test predicts a working forward path.
public enum MemberProbe {
    /// Shared client; lives for the process lifetime so it is never dropped
    /// un-shutdown (AsyncHTTPClient traps on that in debug builds).
    private static let client = HTTPClient(eventLoopGroupProvider: .singleton)

    /// `{endpoint}/models`, tolerating trailing slashes on the stored base.
    public static func modelsURL(endpoint: URL) -> String {
        UpstreamBodyCodec.upstreamURL(endpoint: endpoint, path: "models")
    }

    /// True for NovaMLX DSpark draft/companion blocks (speculators, `_block7`
    /// companions, `…_novamlx` draft variants). They ride along with a main
    /// model and are never standalone members, so imports skip them.
    public static func isDraftModel(_ modelId: String) -> Bool {
        modelId.lowercased().contains("dspark")
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

    /// GET `{endpoint}/models`. 2xx passes; anything else fails with the
    /// upstream's own error text when it sent one.
    public static func probe(
        endpoint: URL, apiKey: String?, timeoutSeconds: Double = 8
    ) async -> MemberProbeResult {
        var request = HTTPClientRequest(url: modelsURL(endpoint: endpoint))
        request.method = .GET
        if let apiKey, !apiKey.isEmpty {
            request.headers.add(name: "authorization", value: "Bearer \(apiKey)")
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
            let models = parseModels(body)
            let summary = models.isEmpty
                ? "HTTP \(status)"
                : "HTTP \(status) · \(models.count) models"
            return MemberProbeResult(
                ok: true, httpStatus: status, models: models, message: summary)
        } catch {
            return MemberProbeResult(
                ok: false, httpStatus: 0, models: [],
                message: errorSummary(error))
        }
    }

    private static func errorSummary(_ error: Error) -> String {
        let text = String(describing: error)
        return text.split(separator: "(").first.map(String.init) ?? text
    }
}
