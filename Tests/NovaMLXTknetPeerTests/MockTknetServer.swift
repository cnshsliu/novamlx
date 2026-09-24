import Foundation
import Hummingbird

/// Minimal tknet.ai stand-in: POST /api/node/register, GET /api/node/demand.
/// Shares MockSourceServer's bind machinery (port 0 + onServerRunning
/// continuation + Once guard) so `port` is the real bound port. The fixed
/// peer token "tok-42" lives only in test bodies; the server never logs.
final class MockTknetServer: @unchecked Sendable {
    private let lock = NSLock()
    private var _port = 0
    private var serverTask: Task<Void, Never>?

    /// Actual bound port; valid after `start()` returns.
    var port: Int {
        lock.lock(); defer { lock.unlock() }; return _port
    }

    static let registerBody = #"{"peerId":"peer-42","token":"tok-42"}"#
    static let demandBody =
        #"{"entries":[{"demandId":"d1","model":"m","modality":"language","note":null}]}"#
    static let validToken = "tok-42"

    /// Starts on an OS-assigned ephemeral port (bind 0) and returns once the
    /// listener is up.
    func start() async throws {
        let router = Router()
        router.post("/api/node/register") { request, _ -> Response in
            _ = try await request.body.collect(upTo: .max)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: Self.textBody(Self.registerBody))
        }
        router.get("/api/node/demand") { request, _ -> Response in
            guard request.headers[.authorization] == "Bearer \(Self.validToken)" else {
                return Response(status: .unauthorized, body: Self.textBody("{}"))
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: Self.textBody(Self.demandBody))
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
                    once.run { cont.resume(throwing: MockTknetServerError.serverStoppedEarly) }
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

    private enum MockTknetServerError: Error { case serverStoppedEarly }

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

    private func setPort(_ port: Int) {
        lock.lock(); defer { lock.unlock() }
        _port = port
    }

    private static func textBody(_ text: String) -> ResponseBody {
        var buffer = ByteBuffer()
        buffer.writeString(text)
        return .init(byteBuffer: buffer)
    }
}
