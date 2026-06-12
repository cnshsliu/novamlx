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

    static func pickRetryProvider(modelName: String, tag: String?, exclude: Set<String>) -> TokenhubProvider? {
        let pool = TokenhubManager.shared.list().filter { $0.isEnabled && $0.includeInLoadBalance && !exclude.contains($0.name) }
        var filtered = pool
        if let tag, !tag.isEmpty {
            filtered = filtered.filter { $0.tags.contains(tag) }
        }
        return filtered.randomElement()
    }

    // MARK: - Chat Completions Passthrough

    /// Raw passthrough proxy for tokenhub. Forwards the original request body to the provider
    /// with the model name swapped to provider.remoteModel. No post-processing.
    static func handleTokenhubPassthrough(
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
}
