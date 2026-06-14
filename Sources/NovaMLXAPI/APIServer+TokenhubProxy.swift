import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils

// MARK: - Tokenhub Chat Completions Proxy
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    // MARK: - Tokenhub Shared Helpers

    static func durationToMs(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1000 + Double(duration.components.attoseconds) / 1e15
    }

    // MARK: - Chat Completions Passthrough

    /// Raw passthrough proxy for tokenhub. Forwards the original request body to the provider
    /// with the model name swapped to provider.remoteModel. No post-processing.
    ///
    /// Single-attempt: no failover. LB-style multi-provider dispatch lives in `LBProxy`
    /// and is invoked via the `lb:<slug>` prefix; this handler is for `tknet:<provider-name>`.
    static func handleTokenhubPassthrough(
        modelName: String,
        rawBody: Data,
        path: String,
        inference: InferenceService,
        tag: String? = nil
    ) async throws -> Response {
        // Plain "tknet" (no provider) used to mean "load-balance across all
        // providers". That path is now superseded by named LBs via LBProxy
        // (use `model = "lb:<slug>"`). Reject bare tknet so callers migrate.
        if modelName.lowercased() == "tknet" {
            return try Self.jsonResponse(
                ["error": ["message": "Bare 'tknet' load-balanced dispatch has been replaced by named LBs. Use 'lb:<slug>' (e.g. 'lb:coding-pool') or 'tknet:<provider-name>' for a specific provider.", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        guard let provider = TokenhubManager.shared.resolve(modelName: modelName, tag: tag) else {
            return try Self.jsonResponse(
                ["error": ["message": "Unknown tokenhub provider: \(modelName)", "type": "invalid_request_error"]],
                httpStatus: .badRequest
            )
        }

        // Swap model name in the raw JSON body
        var bodyDict = try JSONSerialization.jsonObject(with: rawBody) as? [String: Any] ?? [:]
        bodyDict["model"] = provider.remoteModel
        let isStreaming = (bodyDict["stream"] as? Bool) ?? false

        NovaMLXLog.info("[Tokenhub] -> \(provider.name) (\(provider.endpoint)/\(path)) remoteModel=\(provider.remoteModel) managed=\(provider.tags.contains("managed"))")

        // Resolve effective API key: cloud-managed providers (tagged "managed")
        // inherit the user's session token; all others use their own API key.
        func effectiveApiKey(_ p: TokenhubProvider) -> String {
            if p.tags.contains("managed") { return AuthCache.loadSession() ?? "" }
            return p.apiKey
        }

        let bodyData = try JSONSerialization.data(withJSONObject: bodyDict)
        let baseURL = URL(string: provider.endpoint)!
        var urlRequest = URLRequest(url: baseURL.appendingPathComponent(path))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let apiKey = effectiveApiKey(provider)
        if !apiKey.isEmpty {
            urlRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        urlRequest.timeoutInterval = 120
        urlRequest.httpBody = bodyData

        if isStreaming {
            let start = ContinuousClock.now
            let (bytes, urlResponse) = try await URLSession.shared.bytes(for: urlRequest)
            guard let http = urlResponse as? HTTPURLResponse, http.statusCode == 200 else {
                let statusCode = (urlResponse as? HTTPURLResponse)?.statusCode ?? 502
                NovaMLXLog.warning("[Tokenhub] \(provider.name) streaming failed HTTP \(statusCode)")
                let elapsed = ContinuousClock.now - start
                TokenhubManager.shared.recordMetric(providerId: provider.id, success: false, latencyMs: durationToMs(elapsed))
                return Response(status: .init(integerLiteral: statusCode))
            }
            let elapsed = ContinuousClock.now - start
            let latencyMs = durationToMs(elapsed)
            TokenhubManager.shared.recordMetric(providerId: provider.id, success: true, latencyMs: latencyMs)

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
            headers[.init("X-Tokenhub-Provider")!] = provider.name
            return Response(status: .ok, headers: headers, body: responseBody)
        } else {
            let start = ContinuousClock.now
            let (data, urlResponse) = try await URLSession.shared.data(for: urlRequest)
            let elapsed = ContinuousClock.now - start
            let latencyMs = durationToMs(elapsed)
            guard let http = urlResponse as? HTTPURLResponse else {
                TokenhubManager.shared.recordMetric(providerId: provider.id, success: false, latencyMs: latencyMs)
                return Response(status: .internalServerError)
            }
            let success = http.statusCode < 400
            TokenhubManager.shared.recordMetric(providerId: provider.id, success: success, latencyMs: latencyMs)
            if http.statusCode >= 400 {
                let body = String(data: data, encoding: .utf8)?.prefix(300) ?? "nil"
                NovaMLXLog.warning("[Tokenhub] \(provider.name) error HTTP \(http.statusCode): \(body)")
            }
            var headers: HTTPFields = [.contentType: "application/json"]
            headers[.init("X-Tokenhub-Provider")!] = provider.name
            return Response(status: .init(integerLiteral: http.statusCode), headers: headers, body: .init(byteBuffer: ByteBuffer(data: data)))
        }
    }
}
