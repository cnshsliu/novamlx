import Foundation
import Hummingbird

/// Minimal tknet.ai stand-in: POST /api/peer/register, GET /api/peer/demand.
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
    static let earningsBody =
        #"{"availability":{"total_earned":"3.50","available":"2.00","on_hold":"1.50","paid_out":"0.00"},"by_model":[{"model":"coding","requests":4,"tokens":120,"earned":"3.50"}],"recent":[{"request_id":"r1","model":"coding","prompt_tokens":20,"completion_tokens":100,"gross_amount":"3.50","payout_id":null,"created_at":"2026-10-01T00:00:00Z"}]}"#
    static let validToken = "tok-42"
    static let session = "sess-42"
    static let userBody =
        #"{"user":{"userId":"user-1","email":"lucas@example.com","name":"Lucas","tier":"standard","status":"active","role":"user"}}"#

    /// Starts on an OS-assigned ephemeral port (bind 0) and returns once the
    /// listener is up.
    func start() async throws {
        let router = Router()
        router.post("/api/peer/register") { request, _ -> Response in
            _ = try await request.body.collect(upTo: .max)
            // Real tknet.ai answers 201 Created (src/routes/api/peer/register).
            // The mock must match: an earlier `.ok`-only mock hid the client's
            // `== .ok` status check, which rejected every real registration.
            return Response(
                status: .created,
                headers: [.contentType: "application/json"],
                body: Self.textBody(Self.registerBody))
        }
        router.get("/api/peer/earnings") { request, _ -> Response in
            guard request.headers[.authorization] == "Bearer \(Self.validToken)" else {
                return Response(status: .unauthorized, body: Self.textBody("{}"))
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: Self.textBody(Self.earningsBody))
        }
        router.get("/api/peer/demand") { request, _ -> Response in
            guard request.headers[.authorization] == "Bearer \(Self.validToken)" else {
                return Response(status: .unauthorized, body: Self.textBody("{}"))
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: Self.textBody(Self.demandBody))
        }
        router.post("/api/user/auth") { request, _ -> Response in
            struct LoginBody: Decodable { let email: String; let password: String }
            let body = try await request.body.collect(upTo: .max)
            let login = try JSONDecoder().decode(LoginBody.self, from: Data(body.readableBytesView))
            guard login.email == "lucas@example.com", login.password == "secret" else {
                return Response(status: .unauthorized,
                                headers: [.contentType: "application/json"],
                                body: Self.textBody(#"{"error":"Invalid credentials"}"#))
            }
            return Response(
                status: .ok,
                headers: [
                    .contentType: "application/json",
                    .setCookie: "session=\(Self.session); Path=/; HttpOnly; SameSite=Lax"
                ],
                body: Self.textBody(Self.userBody))
        }
        router.get("/api/user/auth") { request, _ -> Response in
            guard request.headers[.cookie]?.contains("session=\(Self.session)") == true else {
                return Response(status: .unauthorized,
                                headers: [.contentType: "application/json"],
                                body: Self.textBody(#"{"error":"Session expired"}"#))
            }
            return Response(status: .ok,
                            headers: [.contentType: "application/json"],
                            body: Self.textBody(Self.userBody))
        }
        router.delete("/api/user/auth") { _, _ -> Response in
            Response(status: .ok, headers: [.contentType: "application/json"],
                     body: Self.textBody(#"{"ok":true}"#))
        }
        router.post("/api/user/peers/claim") { request, _ -> Response in
            struct ClaimBody: Decodable { let peerId: String; let token: String }
            guard request.headers[.cookie]?.contains("session=\(Self.session)") == true else {
                return Response(status: .unauthorized,
                                headers: [.contentType: "application/json"],
                                body: Self.textBody(#"{"error":"Not authenticated"}"#))
            }
            let body = try await request.body.collect(upTo: .max)
            let claim = try JSONDecoder().decode(ClaimBody.self, from: Data(body.readableBytesView))
            guard claim.peerId == "peer-42", claim.token == Self.validToken else {
                return Response(status: .forbidden,
                                headers: [.contentType: "application/json"],
                                body: Self.textBody(#"{"error":"Token does not match this peer"}"#))
            }
            return Response(status: .ok,
                            headers: [.contentType: "application/json"],
                            body: Self.textBody(#"{"ok":true,"peerName":"lucas-mac"}"#))
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
