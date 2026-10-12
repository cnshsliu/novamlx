import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import Testing
@testable import TknetPeer

@Suite("Segment 2 migration acceptance")
struct NovaMigrationAcceptanceTests {
    @Test("migrated local member fails over to the remote member through the gateway")
    func failoverThroughMigratedPool() async throws {
        let nova = MockSourceServer()
        let remote = MockSourceServer()
        try await nova.start()
        try await remote.start()
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-mig-\(UUID().uuidString)"))
        let input = NovaMigrationInput(
            providers: [NovaProviderSnapshot(
                id: "remote", name: "Remote",
                endpoint: "http://127.0.0.1:\(remote.port)/v1",
                apiKey: "sk-remote", remoteModel: "remote-model")],
            loadBalancers: [NovaLBSnapshot(
                id: "lb-1", name: "Coding", slug: "coding", strategy: "round_robin", maxRetries: 3)],
            members: [
                NovaLBMemberSnapshot(lbSlug: "coding", kind: "local", ref: "Qwen3-8B"),
                NovaLBMemberSnapshot(lbSlug: "coding", kind: "remote", ref: "Remote"),
            ])
        let migrated = NovaConfigMigrator.migrate(
            input, existing: PeerConfig.defaultConfig(),
            options: NovaMigrationOptions(localNovaBaseURL: URL(string: "http://127.0.0.1:\(nova.port)/v1")!))
        for (ref, value) in migrated.secrets { secrets.save(value, for: ref) }
        var config = migrated.config
        config.gatewayPort = 0
        let forwarder = PoolForwarder(config: config, secrets: secrets)
        let gateway = LocalGateway()
        do {
            try await gateway.start(host: "127.0.0.1", port: 0, forwarder: forwarder)
            nova.failNextWithStatus = 500
            var request = URLRequest(url: URL(string: "http://127.0.0.1:\(gateway.port)/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.httpBody = Data(#"{"model":"coding","messages":[]}"#.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            #expect(!data.isEmpty)
            #expect(nova.hitCount == 1)
            #expect(remote.hitCount == 1)
            #expect(remote.lastBodyModel == "remote-model")
        } catch {
            await gateway.stop()
            try? await forwarder.shutdown()
            await nova.stop(); await remote.stop()
            throw error
        }
        await gateway.stop()
        try? await forwarder.shutdown()
        await nova.stop(); await remote.stop()
    }
}
