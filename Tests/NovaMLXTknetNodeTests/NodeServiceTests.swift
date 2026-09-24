import Foundation
import os
import Testing
@testable import NovaMLXTknetNode

@Suite("Node service")
struct NodeServiceTests {
    /// Fixed 10 ms sleep regardless of the requested duration: backoffs are
    /// fast and heartbeat loops don't spin hot (the requested values are the
    /// client's concern, not ours).
    private static let quickDelay: @Sendable (Double) async throws -> Void = { _ in
        try await Task.sleep(nanoseconds: 10_000_000)
    }

    /// Thread-safe append-only log of published statuses.
    private final class StatusRecorder: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: [NodeStatus]())
        private var task: Task<Void, Never>?

        func record(_ stream: AsyncStream<NodeStatus>) {
            task = Task {
                for await status in stream {
                    self.state.withLock { $0.append(status) }
                }
            }
        }

        var all: [NodeStatus] { state.withLock { $0 } }

        /// Polls until some published status satisfies the predicate.
        @discardableResult
        func wait(until predicate: (NodeStatus) -> Bool,
                  timeout: TimeInterval = 10) async -> Bool {
            let deadline = Date().addingTimeInterval(timeout)
            while Date() < deadline {
                if all.contains(where: predicate) { return true }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            return all.contains(where: predicate)
        }
    }

    /// Thread-safe box for the last config the persist hook received.
    private final class ConfigBox: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: nil as NodeConfig?)
        func store(_ config: NodeConfig) { state.withLock { $0 = config } }
        var last: NodeConfig? { state.withLock { $0 } }
    }

    private func makeService(
        pair: InMemoryTransportPair,
        sourceEndpoint: URL,
        nodeId: String? = "node-1",
        onConfigChange: (@Sendable (NodeConfig) -> Void)? = nil
    ) -> NodeService {
        var config = NodeConfig.defaultConfig()
        config.nodeId = nodeId
        config.sources = [SourceConfig(
            id: "s1", name: "mock", type: .openaiCompatible,
            endpoint: sourceEndpoint, apiKeyRef: "s1", upstreamModel: "u")]
        config.capabilities = [Capability(
            demandId: "d1", model: "m", sourceId: "s1",
            sourceType: .openaiCompatible, priceIn: 0, priceOut: 0)]
        let secrets = FileSecretStore(directory: FileManager.default.temporaryDirectory
            .appendingPathComponent("tknet-ns-\(UUID().uuidString)"))
        secrets.save("k", for: "s1")
        return NodeService(
            config: config, secrets: secrets,
            transportFactory: { pair.nodeSide },
            delay: Self.quickDelay,
            onConfigChange: onConfigChange)
    }

    /// Next frame that is not a heartbeat (heartbeats interleave freely with
    /// test traffic; callers care about protocol frames).
    private func nextSignificant(
        from stream: AsyncStream<Frame>, timeout: TimeInterval = 10
    ) async -> Frame? {
        while true {
            guard let frame = await stream.next(timeout: timeout) else { return nil }
            if case .heartbeat = frame { continue }
            return frame
        }
    }

    @Test("end-to-end over in-memory tunnel: request in, chunks + end out")
    func e2eInMemory() async throws {
        let source = MockSourceServer()
        try await source.start()
        defer { Task { await source.stop() } }
        let pair = InMemoryTransportPair()
        let recorder = StatusRecorder()
        let service = makeService(
            pair: pair,
            sourceEndpoint: URL(string: "http://127.0.0.1:\(source.port)/v1")!)
        recorder.record(service.statusStream)
        try await service.start()
        let server = pair.serverSide

        guard case .hello(let nodeId, let caps)? = await nextSignificant(from: server.inbound)
        else {
            Issue.record("expected hello")
            return
        }
        #expect(nodeId == "node-1")
        #expect(caps.map(\.demandId) == ["d1"])

        try await server.send(.request(RequestFrame(
            reqId: "r1", model: "m", apiFormat: .openai,
            body: Data(#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}"#.utf8))))

        // Bounded read of frames until responseEnd(r1).
        var received: [Frame] = []
        while let frame = await nextSignificant(from: server.inbound, timeout: 15) {
            received.append(frame)
            if case .responseEnd = frame { break }
        }
        #expect(received.contains { $0.isChunk })
        guard case .responseEnd("r1", let result)? = received.last else {
            Issue.record("expected terminal responseEnd for r1, got \(received.count) frames")
            return
        }
        #expect(result.status == .completed)
        #expect(result.completionTokens == 2)

        // Counters land on the status stream: request in-flight, then done.
        #expect(await recorder.wait(until: { $0.activeRequests == 1 }))
        #expect(await recorder.wait(until: {
            $0.totalRequests == 1 && $0.totalCompletionTokens == 2
        }))
        #expect(await recorder.wait(until: { $0.activeRequests == 0 }))
        #expect(await recorder.wait(until: { $0.connection == .connected }))

        await service.stop()
        #expect(await recorder.wait(until: { $0.connection == .idle }))
        await service.stop()  // idempotent
    }

    @Test("start() refuses an unregistered node")
    func startRequiresRegistration() async throws {
        let pair = InMemoryTransportPair()
        let service = makeService(
            pair: pair,
            sourceEndpoint: URL(string: "http://127.0.0.1:1/v1")!,
            nodeId: nil)
        await #expect(throws: TunnelError.notRegistered) {
            try await service.start()
        }
        await service.stop()  // teardown with nothing running must not trap
    }

    @Test("demand lifecycle: absent demand retires capability, returning demand revives it")
    func demandLifecycle() async throws {
        let pair = InMemoryTransportPair()
        let recorder = StatusRecorder()
        let persisted = ConfigBox()
        let service = makeService(
            pair: pair,
            sourceEndpoint: URL(string: "http://127.0.0.1:1/v1")!,
            onConfigChange: { persisted.store($0) })
        recorder.record(service.statusStream)
        try await service.start()
        let server = pair.serverSide
        guard case .hello? = await nextSignificant(from: server.inbound) else {
            Issue.record("expected hello")
            return
        }

        // d1 disappears from the live demand list → capability retires.
        try await server.send(.demandUpdate([
            DemandEntry(demandId: "d2", model: "other", modality: "language", note: nil)]))
        #expect(await recorder.wait(until: { $0.capabilities.isEmpty && !$0.demand.isEmpty }))
        #expect(service.currentConfig.capabilities.count == 1)  // declaration kept
        #expect(persisted.last?.activeCapabilities.isEmpty == true)  // retirement persisted

        // d1 comes back → capability is active again.
        try await server.send(.demandUpdate([
            DemandEntry(demandId: "d1", model: "m", modality: "language", note: nil)]))
        // The demand guard distinguishes this from the pre-update status,
        // which also advertises d1 as active.
        #expect(await recorder.wait(until: {
            $0.demand.map(\.demandId) == ["d1"] && $0.capabilities.map(\.demandId) == ["d1"]
        }))
        #expect(service.currentDemand.map(\.demandId) == ["d1"])

        await service.stop()
    }

    @Test("applyConfig pushes capability edits without reconnect")
    func applyConfigHotSwap() async throws {
        let pair = InMemoryTransportPair()
        let persisted = ConfigBox()
        let service = makeService(
            pair: pair,
            sourceEndpoint: URL(string: "http://127.0.0.1:1/v1")!,
            onConfigChange: { persisted.store($0) })
        try await service.start()
        let server = pair.serverSide
        guard case .hello? = await nextSignificant(from: server.inbound) else {
            Issue.record("expected hello")
            return
        }

        var edited = service.currentConfig
        edited.capabilities = [
            Capability(demandId: "d1", model: "m", sourceId: "s1",
                       sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
            Capability(demandId: "d9", model: "m9", sourceId: "s1",
                       sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
        ]
        await service.applyConfig(edited)
        guard case .capabilitiesUpdate(let caps)? = await nextSignificant(from: server.inbound)
        else {
            Issue.record("expected capabilitiesUpdate after applyConfig")
            return
        }
        #expect(caps.map(\.demandId) == ["d1", "d9"])
        #expect(service.currentConfig == edited)
        #expect(persisted.last == edited)

        await service.stop()
    }

    @Test("reconcile retires absent demands and revives returning ones")
    func reconcileLifecycle() {
        var config = NodeConfig.defaultConfig()
        config.capabilities = [
            Capability(demandId: "d1", model: "a", sourceId: "s1",
                       sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
            Capability(demandId: "d2", model: "b", sourceId: "s1",
                       sourceType: .openaiCompatible, priceIn: 0, priceOut: 0),
        ]
        #expect(config.reconcile(demandIds: ["d1", "d2"]).isEmpty)
        #expect(config.activeCapabilities.map(\.demandId).sorted() == ["d1", "d2"])
        #expect(config.reconcile(demandIds: ["d2"]) == ["d1"])  // d1 newly retired
        #expect(config.activeCapabilities.map(\.demandId) == ["d2"])
        #expect(config.capabilities.count == 2)                  // nothing deleted
        #expect(config.reconcile(demandIds: ["d2"]).isEmpty)     // stays retired
        #expect(config.reconcile(demandIds: ["d1", "d2"]).isEmpty)  // d1 revived
        #expect(config.activeCapabilities.map(\.demandId).sorted() == ["d1", "d2"])
    }
}
