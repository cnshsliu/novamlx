import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing

/// Live-server checks. The API test bundle already requires NovaMLX on :6590.
@Suite("Peer gateway cutover")
struct GatewayCutoverTests {
    private var base: String {
        ProcessInfo.processInfo.environment["NOVA_API_URL"] ?? "http://127.0.0.1:6590"
    }

    private func postChat(model: String) async throws -> (Int, String) {
        var request = URLRequest(url: URL(string: "\(base)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (status, String(data: data, encoding: .utf8) ?? "")
    }

    @Test("lb: and tknet: models return 404 pointing at Peer")
    func movedModelsAre404() async throws {
        let moved = "Load-balanced and Tokenhub models moved to Tknet Peer"
        let lb = try await postChat(model: "lb:coding")
        #expect(lb.0 == 404)
        #expect(lb.1.contains(moved))
        #expect(lb.1.contains("6600"))

        let named = try await postChat(model: "tknet:foo")
        #expect(named.0 == 404)
        #expect(named.1.contains(moved))

        let bare = try await postChat(model: "tknet")
        #expect(bare.0 == 404)
        #expect(bare.1.contains(moved))
    }

    @Test("a normal model id is not the Peer migration 404")
    func localModelIsNotTheMigrationError() async throws {
        let (status, body) = try await postChat(model: "definitely-not-a-pool")
        #expect(!body.contains("moved to Tknet Peer"))
        #expect(status != 404 || !body.contains("tknet-peer migrate-from-novamlx"))
    }
}
