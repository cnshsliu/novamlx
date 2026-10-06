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
            throw RESTError.badStatus(Int(response.status.code))
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        return (peerId: payload.peerId, token: payload.token)
    }

    /// Fetches the current demand list; requires the registration token.
    public func fetchDemand(server: URL, token: String) async throws -> [DemandEntry] {
        struct Payload: Decodable { let entries: [DemandEntry] }

        var request = HTTPClientRequest(
            url: server.appendingPathComponent("api/peer/demand").absoluteString)
        request.headers.add(name: "authorization", value: "Bearer \(token)")

        let response = try await client.execute(request, timeout: .seconds(30))
        guard (200..<300).contains(response.status.code) else {
            throw RESTError.badStatus(Int(response.status.code))
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
            throw RESTError.badStatus(Int(response.status.code))
        }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        return try JSONDecoder().decode(PeerEarningsSummary.self, from: data)
    }
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
    /// Server answered with a non-2xx status; carries the numeric code.
    case badStatus(Int)
}
