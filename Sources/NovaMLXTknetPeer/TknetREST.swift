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
    public func register(server: URL, peerName: String) async throws -> (peerId: String, token: String) {
        struct RegisterRequest: Encodable { let name: String }
        struct Payload: Decodable { let peerId: String; let token: String }

        var request = HTTPClientRequest(
            url: server.appendingPathComponent("api/node/register").absoluteString)
        request.method = .POST
        request.headers.add(name: "content-type", value: "application/json")
        request.body = .bytes(try JSONEncoder().encode(RegisterRequest(name: peerName)))

        let response = try await client.execute(request, timeout: .seconds(30))
        guard response.status == .ok else { throw RESTError.badStatus(Int(response.status.code)) }
        let data = Data(try await response.body.collect(upTo: 1 << 20).readableBytesView)
        let payload = try JSONDecoder().decode(Payload.self, from: data)
        return (peerId: payload.peerId, token: payload.token)
    }

    /// Fetches the current demand list; requires the registration token.
    public func fetchDemand(server: URL, token: String) async throws -> [DemandEntry] {
        struct Payload: Decodable { let entries: [DemandEntry] }

        var request = HTTPClientRequest(
            url: server.appendingPathComponent("api/node/demand").absoluteString)
        request.headers.add(name: "authorization", value: "Bearer \(token)")

        let response = try await client.execute(request, timeout: .seconds(30))
        guard response.status == .ok else { throw RESTError.badStatus(Int(response.status.code)) }
        let data = Data(try await response.body.collect(upTo: 16 << 20).readableBytesView)
        return try JSONDecoder().decode(Payload.self, from: data).entries
    }
}

public enum RESTError: Error, Equatable {
    /// Server answered with a non-2xx status; carries the numeric code.
    case badStatus(Int)
}
