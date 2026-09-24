import Foundation
import Hummingbird

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
    private var serverTask: Task<Void, Never>?
    private var _port = 0

    /// Test hook: when non-zero, the next request gets this status once.
    var failNextWithStatus: Int {
        get { lock.lock(); defer { lock.unlock() }; return _failNextWithStatus }
        set { lock.lock(); defer { lock.unlock() }; _failNextWithStatus = newValue }
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
        router.post("/v1/chat/completions") { [weak self] request, _ -> Response in
            guard let self else {
                return Response(status: .internalServerError, body: Self.textBody("server gone"))
            }
            let body = try await request.body.collect(upTo: .max)
            if self.delayNextResponse > 0 {
                let delay = self.delayNextResponse
                self.delayNextResponse = 0
                try await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            }
            let json = (try? JSONSerialization.jsonObject(with: Data(body.readableBytesView))) as? [String: Any]
            self.record(authorization: request.headers[.authorization], model: json?["model"] as? String)
            return self.respond()
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

    private func record(authorization: String?, model: String?) {
        lock.lock(); defer { lock.unlock() }
        _lastAuthorizationHeader = authorization
        _lastBodyModel = model
    }

    private func setPort(_ port: Int) {
        lock.lock(); defer { lock.unlock() }
        _port = port
    }

    private func respond() -> Response {
        if failNextWithStatus > 0 {
            let code = failNextWithStatus
            failNextWithStatus = 0
            return Response(status: .init(code: code), body: Self.textBody(#"{"error":"boom"}"#))
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
