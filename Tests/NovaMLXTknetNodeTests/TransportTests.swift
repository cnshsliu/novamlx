import Foundation
import Testing
@testable import NovaMLXTknetNode

@Suite("Transport")
struct TransportTests {
    @Test("in-memory pair delivers frames both directions")
    func pairDelivery() async throws {
        let pair = InMemoryTransportPair()
        let server = pair.serverSide
        let node = pair.nodeSide

        try await node.send(.heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))
        let got = await server.inbound.next()
        #expect(got == .heartbeat(Heartbeat(activeReq: 0, queueDepth: 0)))

        try await server.send(.requestCancel(reqId: "r1"))
        let cancel = await node.inbound.next()
        #expect(cancel == .requestCancel(reqId: "r1"))
    }
}
