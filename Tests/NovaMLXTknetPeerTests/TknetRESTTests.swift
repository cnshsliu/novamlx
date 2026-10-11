import Foundation
import Testing
@testable import NovaMLXTknetPeer

@Suite("Tknet REST")
struct TknetRESTTests {
    @Test("register returns peer id and token")
    func register() async throws {
        let server = MockTknetServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let rest = TknetREST()
        defer { Task { try? await rest.shutdown() } }
        let (peerId, token) = try await rest.register(
            server: URL(string: "http://127.0.0.1:\(server.port)")!, peerName: "lucas-mac")
        #expect(peerId == "peer-42")
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

    @Test("session cookie is the token after the session name")
    func sessionCookie() {
        let header = "session=sess-42; Path=/; HttpOnly; SameSite=Lax"
        #expect(TknetREST.sessionToken(setCookie: [header]) == "sess-42")
        #expect(TknetREST.sessionToken(setCookie: ["other=1"]) == nil)
    }

    @Test("login keeps the session and claim uses it")
    func loginAndClaim() async throws {
        let server = MockTknetServer()
        try await server.start()
        defer { Task { await server.stop() } }
        let rest = TknetREST()
        defer { Task { try? await rest.shutdown() } }
        let base = URL(string: "http://127.0.0.1:\(server.port)")!
        let login = try await rest.login(server: base, email: "lucas@example.com", password: "secret")
        #expect(login.session == "sess-42")
        #expect(login.account.email == "lucas@example.com")
        let user = try await rest.currentUser(server: base, session: login.session)
        #expect(user.name == "Lucas")
        let name = try await rest.claimPeer(
            server: base, session: login.session, peerId: "peer-42", connectionKey: "tok-42")
        #expect(name == "lucas-mac")
        try await rest.logout(server: base, session: login.session)
    }
}
