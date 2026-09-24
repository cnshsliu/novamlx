import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Tknet REST")
struct TknetRESTTests {
    @Test("register returns node id and token")
    func register() async throws {
        let server = MockTknetServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let rest = TknetREST()
        defer { Task { try? await rest.shutdown() } }
        let (nodeId, token) = try await rest.register(
            server: URL(string: "http://127.0.0.1:\(server.port)")!, nodeName: "lucas-mac")
        #expect(nodeId == "node-42")
        #expect(token == "tok-42")
    }

    @Test("demand requires the token")
    func demand() async throws {
        let server = MockTknetServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let rest = TknetREST()
        defer { Task { try? await rest.shutdown() } }
        let entries = try await rest.fetchDemand(
            server: URL(string: "http://127.0.0.1:\(server.port)")!, token: "tok-42")
        #expect(entries == [DemandEntry(demandId: "d1", model: "m", modality: "language", note: nil)])
        await #expect(throws: Error.self) {
            _ = try await rest.fetchDemand(
                server: URL(string: "http://127.0.0.1:\(server.port)")!, token: "wrong")
        }
    }
}
