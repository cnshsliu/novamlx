import AsyncHTTPClient
import Foundation
import NIOCore

/// Registration + demand-list REST calls against tknet.ai. The peer token is
/// a secret: it is returned to the caller and only ever travels in the
/// `authorization` header — it is never logged and never appears in errors.
public struct TknetREST: Sendable {
    /// `.singleton` shares one NIO event loop group across all clients in the
    /// process; this client still owns its lifecycle and must be shut down
    /// via `shutdown()` before being dropped (AsyncHTTPClient asserts in
    /// debug builds otherwise).
    private let client = HTTPClient(eventLoopGroupProvider: .singleton)

    public init() {}

    /// Stops the underlying HTTP client. Owners must call this before
    /// dropping the instance.
    public func shutdown() async throws { try await client.shutdown() }

    /// Registers this peer with tknet.ai and returns the assigned peer id
    /// plus the secret token used for all later authenticated calls.
    public func register(server: URL, peerName: String,
                         previousPeerId: String? = nil, previousToken: String? = nil) async throws -> (peerId: String, token: String) {
        struct RegisterRequest: Encodable {
            let name: String
            var previousPeerId: String? { nil }
            var previousToken: String? { nil }
        }
        struct Wire: Encodable {
            let name: String
            let previousPeerId: String?
            let previousToken: String?
        }
        struct Payload: Decodable { let peerId: String; let token: String }

        var request = HTTPClientRequest(
            url: server.appendingPathComponent("api/peer/register").absoluteString)
        request.method = .POST
        request.headers.add(name: "content-type", value: "application/json")
        request.body = .bytes(try JSONEncoder().encode(
            Wire(name: peerName, previousPeerId: previousPeerId, previousToken: previousToken)))

        let response = try await client.execute(request, timeout: .seconds(30))
        // The server answers 201 Created on success — accept any 2xx, not just 200.
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        return (peerId: payload.peerId, token: payload.token)
    }

    /// Best-effort extraction of {"error": "..."} from a failed response.
    private static func errorMessage(_ response: HTTPClientResponse) async -> String? {
        guard let body = try? await response.body.collect(upTo: 1 << 16) else { return nil }
        let data = Data(body.readableBytesView)
        guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let msg = json["error"] as? String ?? json["message"] as? String else { return nil }
        return msg
    }

    /// Fetches the current demand list; requires the registration token.
    public func fetchDemand(server: URL, token: String) async throws -> [DemandEntry] {
        struct Payload: Decodable { let entries: [DemandEntry] }

        var request = HTTPClientRequest(
            url: server.appendingPathComponent("api/peer/demand").absoluteString)
        request.headers.add(name: "authorization", value: "Bearer \(token)")

        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        let data = Data(try await response.body.collect(upTo: 16 << 20).readableBytesView)
        return try JSONDecoder().decode(Payload.self, from: data).entries
    }

    /// Supplier money view: totals, payout availability (7-day hold),
    /// per-model breakdown, recent ledger entries.
    public func fetchEarnings(server: URL, token: String) async throws -> PeerEarningsSummary {
        var request = HTTPClientRequest(
            url: server.appendingPathComponent("api/peer/earnings").absoluteString)
        request.headers.add(name: "authorization", value: "Bearer \(token)")
        request.headers.add(name: "x-peer-protocol", value: "1")

        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        return try JSONDecoder().decode(PeerEarningsSummary.self, from: data)
    }

    /// Sign in to the tknet account. The session token is the `session`
    /// cookie. Callers store it; this method does not.
    public func login(server: URL, email: String, password: String) async throws -> TknetLogin {
        struct Body: Encodable { let email: String; let password: String }
        struct UserWire: Decodable { let userId: String; let email: String; let name: String }
        struct Payload: Decodable { let user: UserWire }

        var request = HTTPClientRequest(url: endpoint(server, "api/user/auth"))
        request.method = .POST
        request.headers.add(name: "content-type", value: "application/json")
        request.body = .bytes(try JSONEncoder().encode(Body(email: email, password: password)))

        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        guard let session = Self.sessionToken(setCookie: response.headers["set-cookie"]) else {
            throw RESTError.badStatus(Int(response.status.code), message: "Server did not return a session")
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        let user = try JSONDecoder().decode(Payload.self, from: data).user
        return TknetLogin(
            session: session,
            account: TknetAccount(userId: user.userId, email: user.email, name: user.name))
    }

    /// The account for a stored session. 401 means the session is no longer valid.
    public func currentUser(server: URL, session: String) async throws -> TknetAccount {
        struct UserWire: Decodable { let userId: String; let email: String; let name: String }
        struct Payload: Decodable { let user: UserWire }

        var request = HTTPClientRequest(url: endpoint(server, "api/user/auth"))
        request.method = .GET
        request.headers.add(name: "cookie", value: "session=\(session)")
        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        let user = try JSONDecoder().decode(Payload.self, from: data).user
        return TknetAccount(userId: user.userId, email: user.email, name: user.name)
    }

    public func logout(server: URL, session: String) async throws {
        var request = HTTPClientRequest(url: endpoint(server, "api/user/auth"))
        request.method = .DELETE
        request.headers.add(name: "cookie", value: "session=\(session)")
        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        _ = try await response.body.collect(upTo: 1 << 16)
    }

    /// Link this Mac's peer to the signed-in account. The connection key
    /// proves possession. It travels in the request body and is not logged.
    public func claimPeer(server: URL, session: String, peerId: String, connectionKey: String) async throws -> String {
        struct Body: Encodable { let peerId: String; let token: String }
        struct Payload: Decodable { let peerName: String? }

        var request = HTTPClientRequest(url: endpoint(server, "api/user/peers/claim"))
        request.method = .POST
        request.headers.add(name: "content-type", value: "application/json")
        request.headers.add(name: "cookie", value: "session=\(session)")
        request.body = .bytes(try JSONEncoder().encode(Body(peerId: peerId, token: connectionKey)))

        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code), message: await Self.errorMessage(response))
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        return (try? JSONDecoder().decode(Payload.self, from: data).peerName) ?? ""
    }

    /// `session=<token>` from one or more Set-Cookie headers.
    public static func sessionToken(setCookie headers: [String]) -> String? {
        for header in headers {
            for part in header.split(separator: ";") {
                let pair = part.split(separator: "=", maxSplits: 1)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                guard pair.count == 2, pair[0] == "session", !pair[1].isEmpty else { continue }
                return pair[1]
            }
        }
        return nil
    }

    private func endpoint(_ server: URL, _ path: String) -> String {
        server.appendingPathComponent(path).absoluteString
    }
}

/// Signed-in tknet account. No session token here.
public struct TknetAccount: Equatable, Sendable {
    public let userId: String
    public let email: String
    public let name: String
}

/// Login result. `session` is a secret and must go to the secret store.
public struct TknetLogin: Equatable, Sendable {
    public let session: String
    public let account: TknetAccount
}

/// GET /api/peer/earnings payload (snake_case keys match the server).
public struct PeerEarningsSummary: Decodable, Sendable {
    public struct Availability: Decodable, Sendable {
        public let totalEarned: String
        public let available: String
        public let onHold: String
        public let paidOut: String
    }

    public struct ModelRow: Decodable, Sendable {
        public let model: String
        public let requests: Int
        public let tokens: Int
        public let earned: String
    }

    public struct Entry: Decodable, Sendable {
        public let requestId: String
        public let model: String
        public let promptTokens: Int
        public let completionTokens: Int
        public let grossAmount: String
        public let payoutId: Int?
        public let createdAt: String
    }

    public let availability: Availability
    public let byModel: [ModelRow]
    public let recent: [Entry]
}

public enum RESTError: Error, Equatable {
    /// Server answered with a non-2xx status; carries the numeric code and
    /// the server's human message when the body is our JSON error shape.
    case badStatus(Int, message: String? = nil)
}

extension RESTError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .badStatus(let code, let message):
            if let message, !message.isEmpty { return message }
            switch code {
            case 409: return "名字已被占用（409）"
            case 429: return "注册太频繁（429）—— 每台地址每小时限 10 次，稍后再试"
            default: return "服务器返回 HTTP \(code)"
            }
        }
    }
}
