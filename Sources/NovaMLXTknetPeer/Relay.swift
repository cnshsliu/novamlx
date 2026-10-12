import Foundation

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
            var incoming = config
            incoming.migrateLegacyPoolsIfNeeded()
            let previous = Dictionary(uniqueKeysWithValues: self.config.pools.map { ($0.id, $0.requestCount) })
            for index in incoming.pools.indices {
                let id = incoming.pools[index].id
                if let kept = previous[id], kept > incoming.pools[index].requestCount {
                    incoming.pools[index].requestCount = kept
                }
            }
            self.config = incoming
        }

        /// One bump per accepted request, shared by every strategy's rotation.
        public func bumpRequestCount(poolId: String) {
            lock.lock(); defer { lock.unlock() }
            guard let index = config.pools.firstIndex(where: { $0.id == poolId }) else { return }
            config.pools[index].requestCount += 1
        }
    }

    private let forwarder: PoolForwarder
    private let ownsForwarder: Bool

    public init(config: PeerConfig, secrets: any SecretStore) {
        self.forwarder = PoolForwarder(config: config, secrets: secrets)
        self.ownsForwarder = true
    }

    /// Gateway and tunnel share one forwarder. The owner shuts it down.
    public init(forwarder: PoolForwarder) {
        self.forwarder = forwarder
        self.ownsForwarder = false
    }

    public func updateConfig(_ config: PeerConfig) { forwarder.updateConfig(config) }

    /// Stops the upstream HTTP client when this relay created it.
    /// AsyncHTTPClient asserts in debug builds if it is deallocated without
    /// shutdown, so the owner must call this (or `PoolForwarder.shutdown`)
    /// before dropping the client.
    public func shutdown() async throws {
        if ownsForwarder { try await forwarder.shutdown() }
    }

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

        let config = forwarder.currentConfig
        guard let capability = config.capabilities.first(where: { $0.model == request.model }) else {
            fail("no capability for model \(request.model)")
            return
        }
        guard let pool = config.pool(id: capability.sourceId) else {
            fail("no pool \(capability.sourceId) for model \(request.model)")
            return
        }
        // Anthropic forwarding needs a different wire path. Reject when every
        // enabled member is anthropic so an OpenAI-shaped request is not
        // mis-forwarded. Mixed pools skip anthropic members inside the forwarder.
        let enabled = config.members(in: pool).filter(\.isEnabled)
        if !enabled.isEmpty, enabled.allSatisfy({ $0.type == .anthropic }) {
            fail("anthropic source forwarding not yet supported")
            return
        }
        guard request.apiFormat == .openai else {
            fail("api format \(request.apiFormat.rawValue) unsupported by source \(pool.id)")
            return
        }

        let outcome = await forwarder.forward(
            poolId: pool.id, body: request.body, timeoutSeconds: config.requestTimeoutSeconds)
        switch outcome {
        case .cancelled:
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .cancelled, ttftMs: 0, totalMs: elapsedMs(), promptTokens: 0,
                completionTokens: 0, upstreamStatus: 0, errorMessage: nil)))
        case .unknownPool:
            fail("no pool \(capability.sourceId) for model \(request.model)")
        case .poolDisabled, .noHealthyMembers:
            fail("no healthy members for model \(request.model)")
        case .allMembersFailed(let message, let status, let timedOut):
            if timedOut {
                emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                    status: .timeout, ttftMs: 0, totalMs: elapsedMs(), promptTokens: 0,
                    completionTokens: 0, upstreamStatus: status, errorMessage: nil)))
            } else {
                fail(message, upstreamStatus: status)
            }
        case .success(let upstream):
            await emitHTTP(upstream, request: request, started: start, emit: emit)
        }
    }

    private func emitHTTP(
        _ upstream: UpstreamBody, request: RequestFrame,
        started: ContinuousClock.Instant, emit: @escaping @Sendable (Frame) -> Void
    ) async {
        func elapsedMs() -> Double {
            let elapsed = ContinuousClock.now - started
            return Double(elapsed.components.seconds) * 1_000
                + Double(elapsed.components.attoseconds) / 1e15
        }
        func fail(_ message: String, upstreamStatus: Int) {
            emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                status: .failed, ttftMs: 0, totalMs: elapsedMs(), promptTokens: 0,
                completionTokens: 0, upstreamStatus: upstreamStatus, errorMessage: message)))
        }
        if !(200...299).contains(upstream.status) {
            var collected = Data()
            for await chunk in upstream.body { collected.append(chunk) }
            fail(UpstreamErrorText.message(from: collected, status: upstream.status), upstreamStatus: upstream.status)
            return
        }

        var ttftMs = 0.0
        var sawFirstByte = false
        var usage: Relay.UsageAgg?
        var usageRemainder = ""
        for await data in upstream.body {
            if !sawFirstByte {
                sawFirstByte = true
                ttftMs = elapsedMs()
            }
            let consumed = Relay.consumeUsage(incoming: data, remainder: usageRemainder, previous: usage)
            usage = consumed.usage
            usageRemainder = consumed.remainder
            emit(.responseChunk(reqId: request.reqId, payload: data))
        }
        if let terminal = upstream.terminal.current() {
            switch terminal {
            case .cancelled:
                emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                    status: .cancelled, ttftMs: ttftMs, totalMs: elapsedMs(),
                    promptTokens: 0, completionTokens: 0,
                    upstreamStatus: upstream.status, errorMessage: nil)))
            case .timedOut:
                emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                    status: .timeout, ttftMs: ttftMs, totalMs: elapsedMs(),
                    promptTokens: 0, completionTokens: 0,
                    upstreamStatus: upstream.status, errorMessage: nil)))
            case .failed(let message):
                emit(.responseEnd(reqId: request.reqId, result: RequestResult(
                    status: .failed, ttftMs: ttftMs, totalMs: elapsedMs(),
                    promptTokens: 0, completionTokens: 0,
                    upstreamStatus: upstream.status, errorMessage: message)))
            }
            return
        }
        if !usageRemainder.isEmpty {
            let tail = Relay.consumeUsage(incoming: Data(), remainder: usageRemainder + "\n", previous: usage)
            usage = tail.usage
        }
        let tokens = usage ?? Relay.UsageAgg()
        emit(.responseEnd(reqId: request.reqId, result: RequestResult(
            status: .completed, ttftMs: ttftMs, totalMs: elapsedMs(),
            promptTokens: tokens.prompt, completionTokens: tokens.completion,
            upstreamStatus: upstream.status, errorMessage: nil,
            cachedTokens: Relay.cachedTokens(from: usage))))
    }

    /// Extracts OpenAI-style usage from SSE `data:` lines or a plain JSON body,
    /// keeping the last-seen values. Non-JSON bodies (SSE comments, [DONE])
    /// leave the previous usage untouched.
    static func parseUsage(
        from data: Data, previous: Relay.UsageAgg?
    ) -> Relay.UsageAgg? {
        Relay.consumeUsage(incoming: data, remainder: "", previous: previous).usage
    }

    /// Line-buffered usage extraction (P1.4): a `usage` object split across two
    /// body reads used to stay at zero tokens. Only complete lines are parsed;
    /// the trailing partial line is carried into the next call. The last
    /// complete usage object wins, matching the previous single-buffer behavior.
    static func consumeUsage(
        incoming: Data,
        remainder: String,
        previous: UsageAgg?
    ) -> (usage: UsageAgg?, remainder: String) {
        guard let incomingText = String(data: incoming, encoding: .utf8) else {
            return (previous, remainder)
        }
        let text = remainder + incomingText
        var result = previous ?? UsageAgg()
        var found = false
        var trailing = ""
        var lines: [Substring] = text.split(separator: "\n", omittingEmptySubsequences: true)
        // text.split drops a trailing separator; detect an incomplete last line
        // (no newline after it) and carry it instead of parsing garbage.
        if !text.isEmpty && !text.hasSuffix("\n") {
            if let last = lines.last {
                trailing = String(last)
                lines = Array(lines.dropLast())
            }
        }
        for line in lines {
            let payload = line.hasPrefix("data: ") ? String(line.dropFirst(6)) : String(line)
            guard payload != "[DONE]",
                  let chunk = payload.data(using: .utf8),
                  let json = (try? JSONSerialization.jsonObject(with: chunk)) as? [String: Any],
                  let usage = json["usage"] as? [String: Any] else { continue }
            result.prompt = usage["prompt_tokens"] as? Int ?? result.prompt
            result.completion = usage["completion_tokens"] as? Int ?? result.completion
            if let details = usage["prompt_tokens_details"] as? [String: Any],
               let cached = details["cached_tokens"] as? Int {
                result.cached = cached
            }
            found = true
        }
        return (found ? result : previous, trailing)
    }

    /// Streamed-usage aggregate including the LOCAL source's prefix-cache hit
    /// count (prompt_tokens_details.cached_tokens — worker-measured, the only
    /// trusted source; absent for third-party sources → nil → bills as 0).
    struct UsageAgg {
        var prompt: Int = 0
        var completion: Int = 0
        var cached: Int?
    }

    static func cachedTokens(from agg: UsageAgg?) -> Int? {
        guard let agg, let cached = agg.cached, cached > 0 else { return nil }
        return min(cached, agg.prompt) // can never exceed the prompt itself
    }
}
