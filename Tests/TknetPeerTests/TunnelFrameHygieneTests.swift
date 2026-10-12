import Foundation
import Testing
@testable import TknetPeer

@Suite("Tunnel frame hygiene")
struct TunnelFrameHygieneTests {
    private let endpoint = "http://10.0.0.9:8080/v1"
    private let secret = "sk-super-secret-peer-key"
    private let poolId = "pool-opaque"

    private func assertClean(_ payload: String, sourceLocation: SourceLocation = #_sourceLocation) {
        #expect(!payload.contains("10.0.0.9"), sourceLocation: sourceLocation)
        #expect(!payload.contains("8080"), sourceLocation: sourceLocation)
        #expect(!payload.contains(secret), sourceLocation: sourceLocation)
        #expect(!payload.contains(endpoint), sourceLocation: sourceLocation)
        #expect(!payload.contains("apiKey"), sourceLocation: sourceLocation)
        #expect(!payload.contains("endpoint"), sourceLocation: sourceLocation)
        #expect(payload.contains(poolId), sourceLocation: sourceLocation)
    }

    @Test("encoded hello and capabilitiesUpdate omit member endpoints and secrets")
    func encodedFramesOmitMemberMaterial() {
        let capability = Capability(
            demandId: "demand-1", model: "coding", sourceId: poolId,
            sourceType: .openaiCompatible, priceIn: 1, priceOut: 2)
        let hello = FrameCodec.encode(.hello(
            peerId: "peer-1", capabilities: [capability],
            protocolVersion: TknetPeer.protocolVersion, appVersion: TknetPeer.version))
        let update = FrameCodec.encode(.capabilitiesUpdate([capability]))
        assertClean(hello)
        assertClean(update)
    }

    @Test("a live hello does not carry the member endpoint or secret")
    func liveHelloOmitsMemberMaterial() async throws {
        let pair = InMemoryTransportPair()
        var config = PeerConfig.defaultConfig()
        config.peerId = "peer-1"
        config.gatewayPort = 0
        config.members = [MemberConfig(
            id: "member-1", name: "hidden", type: .openaiCompatible,
            endpoint: URL(string: endpoint)!, apiKeyRef: "src/member-1",
            upstreamModel: "upstream-hidden")]
        config.pools = [PoolConfig(
            id: poolId, name: "coding", memberIds: ["member-1"])]
        config.capabilities = [Capability(
            demandId: "demand-1", model: "coding", sourceId: poolId,
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-hygiene-\(UUID().uuidString)"))
        secrets.save(secret, for: "src/member-1")
        let service = PeerService(
            config: config, secrets: secrets,
            transportFactory: { pair.peerSide },
            delay: { _ in try await Task.sleep(nanoseconds: 10_000_000) })
        try await service.start(tunnel: true)

        guard case .hello(_, let caps, _, _)? = await nextSignificant(from: pair.serverSide.inbound) else {
            Issue.record("expected hello")
            await service.stop()
            return
        }
        #expect(caps.map(\.sourceId) == [poolId])
        assertClean(FrameCodec.encode(.hello(
            peerId: "peer-1", capabilities: caps,
            protocolVersion: TknetPeer.protocolVersion, appVersion: TknetPeer.version)))

        var edited = service.currentConfig
        edited.capabilities[0].priceOut = 3
        await service.applyConfig(edited)
        var update: Frame?
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, update == nil {
            guard let frame = await pair.serverSide.inbound.next(timeout: 1) else { continue }
            if case .capabilitiesUpdate = frame { update = frame }
        }
        guard case .capabilitiesUpdate(let updated)? = update else {
            Issue.record("expected capabilitiesUpdate")
            await service.stop()
            return
        }
        #expect(updated.map(\.sourceId) == [poolId])
        assertClean(FrameCodec.encode(.capabilitiesUpdate(updated)))
        await service.stop()
    }

    private func nextSignificant(from stream: AsyncStream<Frame>, timeout: TimeInterval = 10) async -> Frame? {
        while true {
            guard let frame = await stream.next(timeout: timeout) else { return nil }
            if case .heartbeat = frame { continue }
            return frame
        }
    }
}
