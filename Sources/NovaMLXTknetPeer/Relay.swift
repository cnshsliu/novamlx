import AsyncHTTPClient
import Foundation
import NIOCore

/// Forwards tunnel requests to the configured HTTP source and streams the
/// reply back as frames. One `handle` call = one request lifecycle = exactly
/// one terminal `responseEnd`. Source API keys are sent only to the source
/// itself; they are never logged and never appear in any frame.
public final class Relay: Sendable {
    /// Holds the live config so capability edits take effect without restarts.
    public final class ConfigHolder: @unchecked Sendable {
        private let lock = NSLock()
        private var config: PeerConfig

        public init(_ config: PeerConfig) { self.config = config }

        public var value: PeerConfig {
            lock.lock(); defer { lock.unlock() }
            return config
        }

        public func update(_ config: PeerConfig) {
            lock.lock(); defer { lock.unlock() }
            self.config = config
        }
    }

    private let configHolder: ConfigHolder
    private let secrets: any SecretStore
    private let client: HTTPClient

    public init(config: PeerConfig, secrets: any SecretStore) {
        self.configHolder = ConfigHolder(config)
        self.secrets = secrets
        // `.singleton` shares one NIO event loop group across all relays in
        // the process; each HTTPClient still owns its lifecycle and is shut
        // down in `shutdown()` below.
        self.client = HTTPClient(eventLoopGroupProvider: .singleton)
    }

    public func updateConfig(_ config: PeerConfig) { configHolder.update(config) }

    /// Stops the upstream HTTP client. AsyncHTTPClient asserts in debug builds
    /// if it is deallocated without shutdown, so owners must call this before
    /// dropping the relay.
    public func shutdown() async throws { try await client.shutdown() }

    /// Streams the full reply for one tunnel request: zero or more
    /// `responseChunk` frames followed by exactly one terminal `responseEnd`.
    /// The stream also finishes if the consumer stops iterating.
    public func handle(_ request: RequestFrame) -> AsyncStream<Frame> {
        AsyncStream { continuation in
            let task = Task {
                await self.run(request: request) { frame in continuation.yield(frame) }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func run(request: RequestFrame, emit: @escaping @Sendable (Frame) -> Void) async {
        let start = ContinuousClock.now
        func elapsedMs() -> Double {
            let elapsed = ContinuousClock.now - start
            return Double(elapsed.components.seconds) * 1_000
                + Double(elapsed.components.attoseconds) / 1e15
        }
        func fail(_ message: String, upstreamStatus: Int = 0) {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: 0, totalMs: elapsedMs(), promptTokens: 0,
                completionTokens: 0, upstreamStatus: upstreamStatus, errorMessage: message)))
        }

        let config = configHolder.value
        guard let capability = config.capabilities.first(where: { $0.model == request.model }) else {
            fail("no capability for model \(request.model)")
            return
        }
        guard let source = config.sources.first(where: { $0.id == capability.sourceId }) else {
            fail("no source \(capability.sourceId) for model \(request.model)")
            return
        }
        // Anthropic-source forwarding needs a different wire path (/v1/messages,
        // x-api-key scheme); it is not implemented in Phase 1. Reject outright
        // rather than silently mis-forwarding an OpenAI-shaped request.
        guard source.type != .anthropic else {
            fail("anthropic source forwarding not yet supported")
            return
        }
        // Remaining source types (openaiCompatible, localNovaMLX) are
        // OpenAI-format only.
        guard request.apiFormat == .openai else {
            fail("api format \(request.apiFormat.rawValue) unsupported by source \(source.id)")
            return
        }

        // Rewrite the model to the upstream name; the demand-facing name
        // must never leak upstream. Inject only the source key.
        guard var json = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any] else {
            fail("request body is not a JSON object")
            return
        }
        json["model"] = source.upstreamModel
        guard let rewritten = try? JSONSerialization.data(withJSONObject: json) else {
            fail("model rewrite failed")
            return
        }

        var headRequest = HTTPClientRequest(
            url: source.endpoint.appendingPathComponent("chat/completions").absoluteString)
        headRequest.method = .POST
        headRequest.headers.add(name: "content-type", value: "application/json")
        let key = (try? secrets.load(source.apiKeyRef)) ?? nil
        if let key, !key.isEmpty {
            headRequest.headers.add(name: "authorization", value: "Bearer \(key)")
        }
        headRequest.body = .bytes(rewritten)

        var upstreamStatus = 0
        var ttftMs = 0.0
        var sawFirstByte = false
        var usage: (prompt: Int, completion: Int)?
        do {
            let timeoutSeconds = max(1, Int64(config.requestTimeoutSeconds.rounded()))
            let response = try await client.execute(headRequest, timeout: .seconds(timeoutSeconds))
            upstreamStatus = Int(response.status.code)

            guard (200...299).contains(upstreamStatus) else {
                var message = "upstream status \(upstreamStatus)"
                if let buffer = try? await response.body.collect(upTo: 1 << 20) {
                    let text = String(buffer: buffer)
                    if let data = text.data(using: .utf8),
                       let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                       let error = object["error"] as? String {
                        message = error
                    }
                }
                fail(message, upstreamStatus: upstreamStatus)
                return
            }

            for try await buffer in response.body {
                if !sawFirstByte {
                    sawFirstByte = true
                    ttftMs = elapsedMs()
                }
                let data = Data(buffer.readableBytesView)
                usage = Relay.parseUsage(from: data, previous: usage)
                emit(.responseChunk(reqId: request.reqId, payload: data))
            }
            let tokens = usage ?? (prompt: 0, completion: 0)
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .completed, ttftMs: ttftMs, totalMs: elapsedMs(),
                promptTokens: tokens.prompt, completionTokens: tokens.completion,
                upstreamStatus: upstreamStatus, errorMessage: nil)))
        } catch is CancellationError {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .cancelled, ttftMs: ttftMs, totalMs: elapsedMs(),
                promptTokens: 0, completionTokens: 0,
                upstreamStatus: upstreamStatus, errorMessage: nil)))
        } catch let error as HTTPClientError where error == .deadlineExceeded {
            // Whole-request deadline expiry, including mid-body; telemetry
            // callers expect a distinct timeout status, not a generic failure.
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .timeout, ttftMs: ttftMs, totalMs: elapsedMs(),
                promptTokens: 0, completionTokens: 0,
                upstreamStatus: upstreamStatus, errorMessage: nil)))
        } catch {
            // Never include the key in error output; AsyncHTTPClient errors
            // surface the URL at most, and keys travel only in headers.
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: ttftMs, totalMs: elapsedMs(),
                promptTokens: 0, completionTokens: 0,
                upstreamStatus: upstreamStatus, errorMessage: String(describing: error))))
        }
    }

    /// Extracts OpenAI-style usage from SSE `data:` lines or a plain JSON body,
    /// keeping the last-seen values. Non-JSON bodies (SSE comments, [DONE])
    /// leave the previous usage untouched.
    static func parseUsage(
        from data: Data, previous: (prompt: Int, completion: Int)?
    ) -> (prompt: Int, completion: Int)? {
        guard let text = String(data: data, encoding: .utf8) else { return previous }
        var result = previous ?? (0, 0)
        var found = false
        for line in text.split(separator: "\n") {
            let payload = line.hasPrefix("data: ") ? String(line.dropFirst(6)) : String(line)
            guard payload != "[DONE]",
                  let chunk = payload.data(using: .utf8),
                  let json = (try? JSONSerialization.jsonObject(with: chunk)) as? [String: Any],
                  let usage = json["usage"] as? [String: Any] else { continue }
            result.prompt = usage["prompt_tokens"] as? Int ?? result.prompt
            result.completion = usage["completion_tokens"] as? Int ?? result.completion
            found = true
        }
        return found ? result : previous
    }
}
