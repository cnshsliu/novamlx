import Foundation
import Logging
import os

/// Aggregated snapshot the CLI and the Mac page render. `capabilities` is the
/// live serving set (`NodeConfig.activeCapabilities` — declarations whose
/// demand retired are excluded; the full declaration list lives in
/// `currentConfig`). `totalRequests` counts requests that reached a terminal
/// `responseEnd` (matches the relay's one-end-per-request contract).
public struct NodeStatus: Equatable, Sendable {
    public var connection: TunnelStatus
    public var activeRequests: Int
    public var totalRequests: Int
    public var totalCompletionTokens: Int
    public var lastError: String?
    public var demand: [DemandEntry]
    public var capabilities: [Capability]

    public init(connection: TunnelStatus, activeRequests: Int = 0, totalRequests: Int = 0,
                totalCompletionTokens: Int = 0, lastError: String? = nil,
                demand: [DemandEntry] = [], capabilities: [Capability] = []) {
        self.connection = connection; self.activeRequests = activeRequests
        self.totalRequests = totalRequests; self.totalCompletionTokens = totalCompletionTokens
        self.lastError = lastError; self.demand = demand; self.capabilities = capabilities
    }
}

/// The single façade both the CLI and the Mac page talk to. Owns the relay
/// and the tunnel client, aggregates status, and applies operator config
/// edits live. Call `start()` once and `stop()` before dropping it — the
/// relay's AsyncHTTPClient traps in debug builds when dropped un-shutdown.
public final class NodeService: @unchecked Sendable {
    public let statusStream: AsyncStream<NodeStatus>
    private let statusContinuation: AsyncStream<NodeStatus>.Continuation

    /// Everything mutable lives under one lock (same pattern as
    /// TunnelClient); no lock scope ever awaits. `connection` and `demand`
    /// are mirrors of the client's streams so a publish never has to hop
    /// onto another task.
    private struct State {
        var config: NodeConfig
        var client: TunnelClient?
        var relay: Relay?
        var runTask: Task<Void, Never>?
        var statusTask: Task<Void, Never>?
        var demandTask: Task<Void, Never>?
        var aggregates = Aggregates()
    }

    /// Everything the status snapshot is built from.
    private struct Aggregates {
        var connection: TunnelStatus = .idle
        var activeRequests = 0
        var totalRequests = 0
        var totalCompletionTokens = 0
        var lastError: String?
        var demand: [DemandEntry] = []
    }

    private let state: OSAllocatedUnfairLock<State>

    private let secrets: any SecretStore
    private let transportFactory: TransportFactory
    private let delay: @Sendable (Double) async throws -> Void
    /// Persist hook (atomic 0600 write is the caller's job): fired whenever
    /// the service itself changes the config — demand reconciliation and
    /// `applyConfig` today. Nil (default) means the process holds the config
    /// in memory only.
    private let onConfigChange: (@Sendable (NodeConfig) -> Void)?
    private let logger = Logger(label: "TknetNode.NodeService")

    public init(config: NodeConfig, secrets: any SecretStore,
                transportFactory: @escaping TransportFactory,
                delay: @escaping @Sendable (Double) async throws -> Void = {
                    try await Task.sleep(nanoseconds: UInt64($0 * 1_000_000_000))
                },
                onConfigChange: (@Sendable (NodeConfig) -> Void)? = nil) {
        self.secrets = secrets
        self.transportFactory = transportFactory
        self.delay = delay
        self.onConfigChange = onConfigChange
        self.state = OSAllocatedUnfairLock(initialState: State(config: config))
        (statusStream, statusContinuation) = AsyncStream.makeStream()
    }

    /// Connects the tunnel and keeps it alive until `stop()`. Throws
    /// `TunnelError.notRegistered` when the node has no id yet (register
    /// first, then start).
    public func start() async throws {
        guard let initialConfig = takeStartConfig() else {
            throw TunnelError.notRegistered
        }

        let relay = Relay(config: initialConfig, secrets: secrets)
        let client = TunnelClient(
            config: initialConfig, relay: relay,
            transportFactory: observingFactory(),
            delay: delay
        )
        client.onResult = { [weak self] result in
            guard let self else { return }
            self.state.withLock { state in
                state.aggregates.totalRequests += 1
                state.aggregates.totalCompletionTokens += result.completionTokens
                if case .failed = result.status, let message = result.errorMessage {
                    state.aggregates.lastError = message
                }
            }
            self.publish()
        }

        guard install(client: client, relay: relay) else {
            try? await relay.shutdown()  // double start(): keep the first one
            logger.warning("start() called twice; ignoring the second start")
            return
        }

        // Connection-state mirror: republish on every transition.
        let statusTask = Task { [weak self] in
            for await status in client.statusStream {
                self?.updateConnection(status)
            }
        }
        // Demand lifecycle: reconcile declared capabilities against the live
        // list (retire absent ids, revive returning ones), persist, publish.
        let demandTask = Task { [weak self] in
            for await entries in client.demandStream {
                self?.updateDemand(entries)
            }
        }
        let runTask = Task { await client.start() }

        // Register the tasks only if a concurrent stop() hasn't already torn
        // the client out; TunnelClient never finishes its streams, so an
        // orphaned aggregation task would otherwise loop forever.
        let orphaned = state.withLock { state -> Bool in
            if state.client === client {
                state.statusTask = statusTask
                state.demandTask = demandTask
                state.runTask = runTask
                return false
            }
            return true
        }
        if orphaned {
            statusTask.cancel()
            demandTask.cancel()
            runTask.cancel()
            client.stop()
            _ = await runTask.value
            try? await relay.shutdown()
        }
    }

    /// Idempotent teardown. Order matters: cancel the start task (ends the
    /// session loop), stop the client (marks stopping, cancels in-flight
    /// dispatches), then shut the relay down — AsyncHTTPClient has a debug
    /// deinit precondition that traps otherwise. Publishes a final `.idle`.
    public func stop() async {
        let (client, relay, run, statusTask, demandTask) = detach()
        run?.cancel()
        statusTask?.cancel()
        demandTask?.cancel()
        client?.stop()
        if let run {
            _ = await run.value
        }
        try? await relay?.shutdown()
        state.withLock { state in
            state.aggregates.connection = .idle
            state.aggregates.activeRequests = 0
        }
        publish()
    }

    /// Operator edited config (sources, capabilities, limits, prices).
    /// Propagates to the live relay without a reconnect; capability edits
    /// additionally go out as a `capabilitiesUpdate` frame when connected
    /// and are advertised by every subsequent hello (including after
    /// reconnects). `concurrencyLimit` and the hello identity stay bound to
    /// the config `start()` saw — restart to change those.
    public func applyConfig(_ newConfig: NodeConfig) async {
        let client = state.withLock { state -> TunnelClient? in
            state.config = newConfig
            return state.client
        }
        client?.updateRelay(newConfig)
        try? await client?.updateCapabilities(newConfig.capabilities)
        onConfigChange?(newConfig)
        publish()
    }

    public var currentConfig: NodeConfig {
        state.withLock { $0.config }
    }

    /// The last demand list the server pushed (empty before the first
    /// `demandUpdate` frame).
    public var currentDemand: [DemandEntry] {
        state.withLock { $0.aggregates.demand }
    }

    // MARK: - Aggregation

    private func updateConnection(_ status: TunnelStatus) {
        let changed = state.withLock { state -> Bool in
            guard state.aggregates.connection != status else { return false }
            state.aggregates.connection = status
            return true
        }
        if changed { publish() }
    }

    private func updateDemand(_ entries: [DemandEntry]) {
        let (newlyRetired, snapshot): ([String], NodeConfig) = state.withLock { state in
            let newlyRetired = state.config.reconcile(demandIds: Set(entries.map(\.demandId)))
            state.aggregates.demand = entries
            return (newlyRetired, state.config)
        }
        if !newlyRetired.isEmpty {
            // Demand ids only — never source keys or secrets.
            logger.info("demands retired by server: \(newlyRetired)")
        }
        onConfigChange?(snapshot)
        publish()
    }

    private func noteRequestStarted() {
        state.withLock { $0.aggregates.activeRequests += 1 }
        publish()
    }

    private func noteRequestEnded() {
        state.withLock { $0.aggregates.activeRequests = max(0, $0.aggregates.activeRequests - 1) }
        publish()
    }

    private func publish() {
        let status = state.withLock { state in
            NodeStatus(
                connection: state.aggregates.connection,
                activeRequests: state.aggregates.activeRequests,
                totalRequests: state.aggregates.totalRequests,
                totalCompletionTokens: state.aggregates.totalCompletionTokens,
                lastError: state.aggregates.lastError,
                demand: state.aggregates.demand,
                capabilities: state.config.activeCapabilities)
        }
        statusContinuation.yield(status)
    }

    // MARK: - State helpers (non-async so the unfair lock is legal there)

    /// Config snapshot if the node is registered and not yet started.
    private func takeStartConfig() -> NodeConfig? {
        state.withLock { state in
            guard state.config.nodeId != nil else { return nil }
            return state.config
        }
    }

    /// Installs the freshly built client/relay; false when already started.
    private func install(client: TunnelClient, relay: Relay) -> Bool {
        state.withLock { state in
            guard state.client == nil else { return false }
            state.client = client
            state.relay = relay
            return true
        }
    }

    /// Pulls every task handle for teardown and clears the started state.
    private func detach() -> (TunnelClient?, Relay?,
                              Task<Void, Never>?, Task<Void, Never>?, Task<Void, Never>?) {
        state.withLock { state in
            let out = (state.client, state.relay, state.runTask,
                       state.statusTask, state.demandTask)
            state.client = nil
            state.relay = nil
            state.runTask = nil
            state.statusTask = nil
            state.demandTask = nil
            return out
        }
    }

    /// Wraps the caller's transports so the service can observe protocol
    /// traffic without the client exposing its slot map: requests in, terminal
    /// frames (responseEnd out, requestCancel in) end.
    private func observingFactory() -> TransportFactory {
        { [weak self, transportFactory] () async throws -> any TunnelTransport in
            guard let self else { throw TunnelError.connectionClosed }
            return ObservingTransport(
                underlying: try await transportFactory(),
                onRequest: { [weak self] in self?.noteRequestStarted() },
                onTerminal: { [weak self] in self?.noteRequestEnded() })
        }
    }
}

/// Transport decorator feeding NodeService's in-flight counter. A request
/// counts as active from the inbound `.request` frame until its terminal
/// frame: an outbound `responseEnd` (normal, refused, timed out, cancelled
/// mid-flight) or an inbound `requestCancel`. A cancellation race can produce
/// both, so the decrement clamps at zero — worst case the counter reads 0 a
/// heartbeat early, never negative or stuck.
private final class ObservingTransport: TunnelTransport, @unchecked Sendable {
    let inbound: AsyncStream<Frame>
    private let inboundContinuation: AsyncStream<Frame>.Continuation
    private let underlying: any TunnelTransport
    private let onTerminal: @Sendable () -> Void
    private let closed = OSAllocatedUnfairLock(initialState: false)

    init(underlying: any TunnelTransport,
         onRequest: @escaping @Sendable () -> Void,
         onTerminal: @escaping @Sendable () -> Void) {
        self.underlying = underlying
        self.onTerminal = onTerminal
        (inbound, inboundContinuation) = AsyncStream.makeStream()
        let task = Task {
            for await frame in underlying.inbound {
                switch frame {
                case .request:
                    onRequest()
                case .requestCancel:
                    onTerminal()
                default:
                    break
                }
                inboundContinuation.yield(frame)
            }
            inboundContinuation.finish()
        }
        inboundContinuation.onTermination = { _ in task.cancel() }
    }

    func send(_ frame: Frame) async throws {
        if case .responseEnd = frame { onTerminal() }
        try await underlying.send(frame)
    }

    func close() async {
        let alreadyClosed = closed.withLock { state -> Bool in
            if state { return true }
            state = true
            return false
        }
        guard !alreadyClosed else { return }
        await underlying.close()
    }
}
