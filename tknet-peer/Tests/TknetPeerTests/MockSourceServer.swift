import Foundation
import Hummingbird
@testable import TknetPeer

/// Real local HTTP server pretending to be an OpenAI-compatible source.
/// Serves streaming SSE replies (shared by Tasks 5, 7, 12) so the relay does
/// real socket I/O through AsyncHTTPClient. Test hooks record the last
/// authorization header and body model, and can fail one request with a
/// chosen status. The server never logs anything; source keys stay private.
final class MockSourceServer: @unchecked Sendable {
    private let lock = NSLock()
    private var _failNextWithStatus = 0
    private var _delayNextResponse: TimeInterval = 0
    private var _lastAuthorizationHeader: String?
    private var _lastBodyModel: String?
    private var _lastPath: String?
    private var _lastXAPIKey: String?
    private var _lastAnthropicVersion: String?
    private var _lastAnthropicBeta: String?
    private var _lastProbeAuthorization: String?
    private var _lastProbePath: String?
    private var serverTask: Task<Void, Never>?
    private var _port = 0

    /// Test hook: when non-zero, the next request gets this status once.
    var failNextWithStatus: Int {
        get { lock.lock(); defer { lock.unlock() }; return _failNextWithStatus }
        set { lock.lock(); defer { lock.unlock() }; _failNextWithStatus = newValue }
    }
    private var _failNextBody = #"{"error":"boom"}"#
    /// Test hook: body of the failure response (defaults to the flat
    /// {"error":"boom"} shape). Set to NovaMLX's nested
    /// {"error":{"message":...}} shape to exercise extraction.
    var failNextBody: String {
        get { lock.lock(); defer { lock.unlock() }; return _failNextBody }
        set { lock.lock(); defer { lock.unlock() }; _failNextBody = newValue }
    }
    private var _lastNoAutoloadHeader: String?
    private var _hitCount = 0
    /// Last X-No-Autoload header value the server saw (nil when absent).
    var lastNoAutoloadHeader: String? {
        lock.lock(); defer { lock.unlock() }; return _lastNoAutoloadHeader
    }
    /// How many chat-completion requests this server has accepted.
    var hitCount: Int {
        lock.lock(); defer { lock.unlock() }; return _hitCount
    }
    /// Test hook: when non-zero, the next request sleeps this many seconds in
    /// its handler before producing any response (drives client deadline
    /// testing; a slow upstream head is what triggers deadlineExceeded).
    var delayNextResponse: TimeInterval {
        get { lock.lock(); defer { lock.unlock() }; return _delayNextResponse }
        set { lock.lock(); defer { lock.unlock() }; _delayNextResponse = newValue }
    }
    /// Last `authorization` header the server saw (nil when none was sent).
    var lastAuthorizationHeader: String? {
        lock.lock(); defer { lock.unlock() }; return _lastAuthorizationHeader
    }
    /// Last `model` field in the request body the server saw.
    var lastBodyModel: String? {
        lock.lock(); defer { lock.unlock() }; return _lastBodyModel
    }
    /// Last request path, such as `/v1/embeddings`.
    var lastPath: String? {
        lock.lock(); defer { lock.unlock() }; return _lastPath
    }
    var lastXAPIKey: String? {
        lock.lock(); defer { lock.unlock() }; return _lastXAPIKey
    }
    var lastAnthropicVersion: String? {
        lock.lock(); defer { lock.unlock() }; return _lastAnthropicVersion
    }
    var lastAnthropicBeta: String? {
        lock.lock(); defer { lock.unlock() }; return _lastAnthropicBeta
    }
    /// Last `authorization` header seen on a GET (models probe).
    var lastProbeAuthorization: String? {
        lock.lock(); defer { lock.unlock() }; return _lastProbeAuthorization
    }
    /// Last path seen on a GET (models probe).
    var lastProbePath: String? {
        lock.lock(); defer { lock.unlock() }; return _lastProbePath
    }
    /// Actual bound port; valid after `start()` returns.
    var port: Int {
        lock.lock(); defer { lock.unlock() }; return _port
    }

    static let chunk1 = #"{"id":"1","choices":[{"delta":{"content":"Hello"}}]}"#
    static let chunk2 = #"{"id":"1","choices":[{"delta":{"content":" world"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#

    /// Starts on an OS-assigned ephemeral port (bind 0) and returns once the
    /// listener is up, so `port` is the real bound port — no guessing.
    func start() async throws {
        let router = Router()
        let paths = [
            "/v1/chat/completions", "/v1/messages", "/v1/embeddings",
            "/v1/images/generations", "/v1/images/edits",
            "/v1/images/background-removals",
            "/v1/audio/speech", "/v1/audio/transcriptions",
            "/v1/decisions",
        ]
        for path in paths {
            router.post(RouterPath(path)) { [weak self] request, _ -> Response in
                guard let self else {
                    return Response(status: .internalServerError, body: Self.textBody("server gone"))
                }
                let body = try await request.body.collect(upTo: .max)
                if self.delayNextResponse > 0 {
                    let delay = self.delayNextResponse
                    self.delayNextResponse = 0
                    try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                }
                let raw = Data(body.readableBytesView)
                let contentType = request.headers[.contentType]
                let model = UpstreamBodyCodec.poolName(body: raw, contentType: contentType)
                self.record(
                    authorization: request.headers[.authorization],
                    model: model,
                    noAutoload: request.headers[values: .init("X-No-Autoload")!].first,
                    path: request.uri.path,
                    xAPIKey: request.headers[values: .init("x-api-key")!].first,
                    anthropicVersion: request.headers[values: .init("anthropic-version")!].first,
                    anthropicBeta: request.headers[values: .init("anthropic-beta")!].first)
                return self.respond()
            }
        }

        // Model listing for MemberProbe tests. Records separately from the
        // chat counters so probe tests never disturb hit-count assertions.
        router.get(RouterPath("/v1/models")) { [weak self] request, _ -> Response in
            guard let self else {
                return Response(status: .internalServerError, body: Self.textBody("server gone"))
            }
            self.recordProbe(
                authorization: request.headers[.authorization],
                path: request.uri.path)
            var buffer = ByteBuffer()
            buffer.writeString(#"{"object":"list","data":[{"id":"llama-4-8b"},{"id":"qwen3.5-14b"},{"id":"qwen3.5-14b"}]}"#)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: buffer))
        }

        let port = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Int, Error>) in
            let once = Once()
            let app = Application(
                router: router,
                configuration: .init(address: .hostname("127.0.0.1", port: 0)),
                onServerRunning: { channel in
                    once.run { cont.resume(returning: channel.localAddress?.port ?? 0) }
                }
            )
            let task = Task<Void, Never> {
                do {
                    try await app.run()
                    once.run { cont.resume(throwing: MockSourceServerError.serverStoppedEarly) }
                } catch {
                    once.run { cont.resume(throwing: error) }
                }
            }
            self.serverTask = task
        }
        setPort(port)
    }

    /// Cancels the server task and waits for the listener to go down.
    func stop() async {
        guard let task = serverTask else { return }
        task.cancel()
        _ = await task.value
        serverTask = nil
    }

    // MARK: - Internals

    private enum MockSourceServerError: Error { case serverStoppedEarly }

    /// Runs the closure exactly once so a CheckedContinuation is never
    /// resumed twice (server-running and run()-failure race).
    private final class Once: @unchecked Sendable {
        private let lock = NSLock()
        private var done = false
        func run(_ body: () -> Void) {
            lock.lock(); defer { lock.unlock() }
            guard !done else { return }
            done = true
            body()
        }
    }

    private func record(
        authorization: String?, model: String?, noAutoload: String?, path: String,
        xAPIKey: String?, anthropicVersion: String?, anthropicBeta: String?
    ) {
        lock.lock(); defer { lock.unlock() }
        _lastAuthorizationHeader = authorization
        _lastBodyModel = model
        _lastNoAutoloadHeader = noAutoload
        _lastPath = path
        _lastXAPIKey = xAPIKey
        _lastAnthropicVersion = anthropicVersion
        _lastAnthropicBeta = anthropicBeta
        _hitCount += 1
    }

    private func recordProbe(authorization: String?, path: String) {
        lock.lock(); defer { lock.unlock() }
        _lastProbeAuthorization = authorization
        _lastProbePath = path
    }

    private func setPort(_ port: Int) {
        lock.lock(); defer { lock.unlock() }
        _port = port
    }

    private func respond() -> Response {
        if failNextWithStatus > 0 {
            let code = failNextWithStatus
            failNextWithStatus = 0
            let body = failNextBody
            return Response(status: .init(code: code), body: Self.textBody(body))
        }
        let lines = ["data: \(Self.chunk1)", "data: \(Self.chunk2)", "data: [DONE]"]
        return Response(
            status: .ok,
            headers: [.contentType: "text/event-stream"],
            // contentLength nil -> chunked transfer encoding; one write per SSE
            // event so the relay sees (at least) one chunk frame per event.
            // finish(nil) is mandatory: without it the HTTP response never
            // terminates and the client's body iteration hangs.
            body: .init(contentLength: nil) { writer in
                for line in lines {
                    var buffer = ByteBuffer()
                    buffer.writeString(line + "\n\n")
                    try await writer.write(buffer)
                    try await Task.sleep(nanoseconds: 10_000_000)
                }
                try await writer.finish(nil)
            }
        )
    }

    private static func textBody(_ text: String) -> ResponseBody {
        var buffer = ByteBuffer()
        buffer.writeString(text)
        return .init(byteBuffer: buffer)
    }
}
