import Foundation
import Logging

/// Aggregated snapshot the CLI and the Mac page render. `capabilities` is the
/// live serving set (`PeerConfig.activeCapabilities` — declarations whose
/// demand retired are excluded; the full declaration list lives in
/// `currentConfig`). `totalRequests` counts requests that reached a terminal
/// `responseEnd` (matches the relay's one-end-per-request contract).
public struct PeerStatus: Equatable, Sendable {
    public var connection: TunnelStatus
    public var activeRequests: Int
    public var totalRequests: Int
    public var totalCompletionTokens: Int
    public var lastError: String?
    public var demand: [DemandEntry]
    public var capabilities: [Capability]
    public var gatewayListening: Bool
    public var gatewayPort: Int?

    public init(connection: TunnelStatus, activeRequests: Int = 0, totalRequests: Int = 0,
                totalCompletionTokens: Int = 0, lastError: String? = nil,
                demand: [DemandEntry] = [], capabilities: [Capability] = [],
                gatewayListening: Bool = false, gatewayPort: Int? = nil) {
        self.connection = connection; self.activeRequests = activeRequests
        self.totalRequests = totalRequests; self.totalCompletionTokens = totalCompletionTokens
        self.lastError = lastError; self.demand = demand; self.capabilities = capabilities
        self.gatewayListening = gatewayListening; self.gatewayPort = gatewayPort
    }
}

public enum PeerServiceError: Error, Equatable, Sendable, CustomStringConvertible, LocalizedError {
    /// `startTunnel` before `start`, or after `stop` tore the listener down.
    case gatewayNotRunning
    /// No `peerId` — register before dialing.
    case notRegistered

    public var description: String {
        switch self {
        case .gatewayNotRunning:
            return "Local gateway is not running."
        case .notRegistered:
            return "No peer id — register before dialing the tunnel."
        }
    }

    public var errorDescription: String? { description }
}

/// The single façade both the CLI and the Peer app talk to. Owns the local
/// gateway and, separately, the tunnel client. `stopTunnel()` leaves the
/// gateway listening; `stop()` tears both down. Call `stop()` before
/// dropping the service — the relay's AsyncHTTPClient traps in debug builds
/// when dropped un-shutdown.
public final class PeerService: @unchecked Sendable {
    public let statusStream: AsyncStream<PeerStatus>
    private let statusContinuation: AsyncStream<PeerStatus>.Continuation

    /// Everything mutable lives under one lock (same pattern as
    /// TunnelClient); no lock scope ever awaits. `connection` and `demand`
    /// are mirrors of the client's streams so a publish never has to hop
    /// onto another task.
    private struct State {
        var config: PeerConfig
        var client: TunnelClient?
        var relay: Relay?
        var forwarder: PoolForwarder?
        var gateway: LocalGateway?
        var started = false
        /// True while `startTunnel` is between claim and install, so a second
        /// caller is a no-op instead of dialing twice.
        var tunnelStarting = false
        /// Bumped by `stop` / `stopTunnel` so an in-flight dial cannot install
        /// after the operator already tore the tunnel down.
        var tunnelGeneration: UInt64 = 0
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
        var gatewayListening = false
        var gatewayPort: Int?
    }

    private let state: PeerLock<State>

    private let secrets: any SecretStore
    private let transportFactory: TransportFactory
    private let delay: @Sendable (Double) async throws -> Void
    /// Persist hook (atomic 0600 write is the caller's job): fired whenever
    /// the service itself changes the config — demand reconciliation and
    /// `applyConfig` today. Nil (default) means the process holds the config
    /// in memory only.
    private let onConfigChange: (@Sendable (PeerConfig) -> Void)?
    private let logger = Logger(label: "TknetPeer.PeerService")

    public init(config: PeerConfig, secrets: any SecretStore,
                transportFactory: @escaping TransportFactory,
                delay: @escaping @Sendable (Double) async throws -> Void = {
                    try await Task.sleep(nanoseconds: UInt64($0 * 1_000_000_000))
                },
                onConfigChange: (@Sendable (PeerConfig) -> Void)? = nil) {
        self.secrets = secrets
        self.transportFactory = transportFactory
        self.delay = delay
        self.onConfigChange = onConfigChange
        self.state = PeerLock(initialState: State(config: config))
        (statusStream, statusContinuation) = AsyncStream.makeStream()
    }

    /// Starts the local gateway, then dials the tunnel when `tunnel` is true
    /// and this peer already has an id. Missing registration skips the tunnel
    /// and still serves the gateway — a token factory does not need tknet to
    /// balance local upstreams.
    public func start(tunnel: Bool = true) async throws {
        guard let initialConfig = claimStart() else {
            logger.warning("start() called twice; ignoring the second start")
            return
        }

        let forwarder = PoolForwarder(config: initialConfig, secrets: secrets)
        let gateway = LocalGateway()
        do {
            try await gateway.start(
                host: initialConfig.gatewayHost,
                port: initialConfig.gatewayPort,
                forwarder: forwarder)
        } catch {
            state.withLock { $0.started = false }
            try? await forwarder.shutdown()
            throw error
        }
        guard noteGateway(up: true, port: gateway.port, forwarder: forwarder, gateway: gateway) else {
            await gateway.stop()
            try? await forwarder.shutdown()
            return
        }

        let dial = tunnel && initialConfig.peerId != nil
        if !dial {
            if tunnel {
                logger.info("no peer id; serving the local gateway without a tunnel")
            }
            return
        }
        do {
            try await startTunnel()
        } catch PeerServiceError.gatewayNotRunning, PeerServiceError.notRegistered {
            // stop() won the race, or the id disappeared. The listener is
            // already down in the first case; leave it up in the second.
            return
        }
    }

    /// Dial the tunnel while the gateway is already listening. Idempotent
    /// when a tunnel is up or a dial is in progress. Throws when the gateway
    /// is not running or this peer has no id.
    public func startTunnel() async throws {
        guard let claimed = try claimedTunnel() else { return }
        await runTunnel(config: claimed.config, forwarder: claimed.forwarder, generation: claimed.generation)
    }

    /// Tear down the tunnel only. The local gateway and forwarder keep
    /// serving. Publishes `connection: .idle` with `gatewayListening` unchanged.
    public func stopTunnel() async {
        let detached = detach(includeGateway: false)
        await finish(detached, gatewayDown: false)
    }

    /// Idempotent full teardown (gateway and tunnel). Used on app Quit and
    /// CLI exit. Order matters: cancel the start task, stop the client, then
    /// shut the forwarder down — AsyncHTTPClient has a debug deinit
    /// precondition that traps otherwise. Publishes a final `.idle`.
    public func stop() async {
        let detached = detach(includeGateway: true)
        await finish(detached, gatewayDown: true)
    }

    /// Operator edited config (sources, capabilities, limits, prices).
    /// Propagates to the live relay without a reconnect; capability edits
    /// additionally go out as a `capabilitiesUpdate` frame when connected
    /// and are advertised by every subsequent hello (including after
    /// reconnects). `concurrencyLimit` and the hello identity stay bound to
    /// the config `start()` saw — restart to change those.
    public func applyConfig(_ newConfig: PeerConfig) async {
        var migrated = newConfig
        migrated.migrateLegacyPoolsIfNeeded()
        let newConfig = migrated
        let client = state.withLock { state -> TunnelClient? in
            state.config = newConfig
            state.forwarder?.updateConfig(newConfig)
            return state.client
        }
        client?.updateRelay(newConfig)
        try? await client?.updateCapabilities(newConfig.capabilities)
        onConfigChange?(newConfig)
        publish()
    }

    public var currentConfig: PeerConfig {
        state.withLock { $0.config }
    }

    /// Bound gateway port after `start`, if the listener came up.
    public var gatewayListenPort: Int? {
        state.withLock { $0.aggregates.gatewayPort }
    }

    /// The last demand list the server pushed (empty before the first
    /// `demandUpdate` frame).
    public var currentDemand: [DemandEntry] {
        state.withLock { $0.aggregates.demand }
    }

    // MARK: - Aggregation

    private func updateConnection(_ status: TunnelStatus, from client: TunnelClient) {
        let changed = state.withLock { state -> Bool in
            // Ignore a tunnel that stop/stopTunnel already replaced. A late
            // `.stopping` from the old client must not clobber the new one.
            guard state.client === client else { return false }
            guard state.aggregates.connection != status else { return false }
            state.aggregates.connection = status
            return true
        }
        if changed { publish() }
    }

    private func updateDemand(_ entries: [DemandEntry], from client: TunnelClient) {
        let reconciled: ([String], PeerConfig)? = state.withLock { state in
            guard state.client === client else { return nil }
            let newlyRetired = state.config.reconcile(demandIds: Set(entries.map(\.demandId)))
            state.aggregates.demand = entries
            return (newlyRetired, state.config)
        }
        guard let (newlyRetired, snapshot) = reconciled else { return }
        if !newlyRetired.isEmpty {
            // Demand ids only — never source keys or secrets.
            logger.info("demands retired by server: \(newlyRetired)")
        }
        onConfigChange?(snapshot)
        publish()
    }

    private func noteRequestStarted() {
        let changed = state.withLock { state -> Bool in
            guard state.client != nil else { return false }
            state.aggregates.activeRequests += 1
            return true
        }
        if changed { publish() }
    }

    private func noteRequestEnded() {
        let changed = state.withLock { state -> Bool in
            guard state.client != nil else { return false }
            state.aggregates.activeRequests = max(0, state.aggregates.activeRequests - 1)
            return true
        }
        if changed { publish() }
    }

    private func publish() {
        let status = state.withLock { state in
            PeerStatus(
                connection: state.aggregates.connection,
                activeRequests: state.aggregates.activeRequests,
                totalRequests: state.aggregates.totalRequests,
                totalCompletionTokens: state.aggregates.totalCompletionTokens,
                lastError: state.aggregates.lastError,
                demand: state.aggregates.demand,
                capabilities: state.config.activeCapabilities,
                gatewayListening: state.aggregates.gatewayListening,
                gatewayPort: state.aggregates.gatewayPort)
        }
        statusContinuation.yield(status)
    }

    // MARK: - State helpers (non-async so the unfair lock is legal there)

    /// Marks the service started and returns the migrated config. Nil when
    /// `start` is already in progress or running.
    private func claimStart() -> PeerConfig? {
        state.withLock { state in
            guard !state.started else { return nil }
            state.started = true
            var config = state.config
            config.migrateLegacyPoolsIfNeeded()
            state.config = config
            return config
        }
    }

    /// Records the live gateway. False when `stop` already won the race.
    private func noteGateway(up: Bool, port: Int, forwarder: PoolForwarder, gateway: LocalGateway) -> Bool {
        let accepted = state.withLock { state -> Bool in
            guard state.started else { return false }
            state.forwarder = forwarder
            state.gateway = gateway
            state.aggregates.gatewayListening = up
            state.aggregates.gatewayPort = port
            return true
        }
        if accepted { publish() }
        return accepted
    }

    private struct ClaimedTunnel: Sendable {
        var config: PeerConfig
        var forwarder: PoolForwarder
        var generation: UInt64
    }

    private enum TunnelClaim {
        case ready(ClaimedTunnel)
        case alreadyUp
        case notRunning
        case notRegistered
    }

    /// Nil when a tunnel is already up or a dial is already in progress.
    private func claimedTunnel() throws -> ClaimedTunnel? {
        switch claimTunnel() {
        case .alreadyUp:
            return nil
        case .notRunning:
            throw PeerServiceError.gatewayNotRunning
        case .notRegistered:
            throw PeerServiceError.notRegistered
        case .ready(let claimed):
            return claimed
        }
    }

    /// Marks a dial in progress. Does not increment the generation — only
    /// stop paths do, so a dial that loses the race fails install.
    private func claimTunnel() -> TunnelClaim {
        state.withLock { state in
            guard state.started, let forwarder = state.forwarder else { return .notRunning }
            if state.client != nil || state.tunnelStarting { return .alreadyUp }
            guard state.config.peerId != nil else { return .notRegistered }
            state.tunnelStarting = true
            return .ready(ClaimedTunnel(
                config: state.config, forwarder: forwarder, generation: state.tunnelGeneration))
        }
    }

    private func runTunnel(config: PeerConfig, forwarder: PoolForwarder, generation: UInt64) async {
        let relay = Relay(forwarder: forwarder)
        let client = TunnelClient(
            config: config, relay: relay,
            transportFactory: observingFactory(),
            delay: delay
        )
        client.onResult = { [weak self, weak client] result in
            guard let self, let client else { return }
            let changed = self.state.withLock { state -> Bool in
                guard state.client === client else { return false }
                state.aggregates.totalRequests += 1
                state.aggregates.totalCompletionTokens += result.completionTokens
                if case .failed = result.status, let message = result.errorMessage {
                    state.aggregates.lastError = message
                }
                return true
            }
            if changed { self.publish() }
        }

        guard install(client: client, relay: relay, generation: generation) else {
            client.stop()
            state.withLock { state in
                if state.tunnelGeneration == generation, state.client == nil {
                    state.tunnelStarting = false
                }
            }
            logger.warning("startTunnel() lost the race; not installing")
            return
        }

        let statusTask = Task { [weak self] in
            for await status in client.statusStream {
                self?.updateConnection(status, from: client)
            }
        }
        let demandTask = Task { [weak self] in
            for await entries in client.demandStream {
                self?.updateDemand(entries, from: client)
            }
        }
        let runTask = Task { await client.start() }

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
        }
    }

    /// Installs the freshly built client/relay. False when stop already ran
    /// or the generation moved.
    private func install(client: TunnelClient, relay: Relay, generation: UInt64) -> Bool {
        state.withLock { state in
            guard state.started, state.client == nil, state.tunnelGeneration == generation else {
                return false
            }
            state.client = client
            state.relay = relay
            state.tunnelStarting = false
            return true
        }
    }

    /// Pulls tunnel task handles. When `includeGateway` is true, also takes
    /// the listener and clears `started` (full `stop`).
    private func detach(includeGateway: Bool) -> (TunnelClient?, LocalGateway?, PoolForwarder?,
                                                  Task<Void, Never>?, Task<Void, Never>?, Task<Void, Never>?) {
        state.withLock { state in
            let gateway = includeGateway ? state.gateway : nil
            let forwarder = includeGateway ? state.forwarder : nil
            let out = (state.client, gateway, forwarder, state.runTask, state.statusTask, state.demandTask)
            state.client = nil
            state.relay = nil
            state.runTask = nil
            state.statusTask = nil
            state.demandTask = nil
            state.tunnelStarting = false
            state.tunnelGeneration &+= 1
            if includeGateway {
                state.gateway = nil
                state.forwarder = nil
                state.started = false
            }
            return out
        }
    }

    private func finish(
        _ detached: (TunnelClient?, LocalGateway?, PoolForwarder?,
                     Task<Void, Never>?, Task<Void, Never>?, Task<Void, Never>?),
        gatewayDown: Bool
    ) async {
        let (client, gateway, forwarder, run, statusTask, demandTask) = detached
        run?.cancel()
        statusTask?.cancel()
        demandTask?.cancel()
        client?.stop()
        if let run {
            _ = await run.value
        }
        if gatewayDown {
            await gateway?.stop()
            try? await forwarder?.shutdown()
        }
        state.withLock { state in
            state.aggregates.connection = .idle
            state.aggregates.activeRequests = 0
            if gatewayDown {
                state.aggregates.gatewayListening = false
                state.aggregates.gatewayPort = nil
                state.started = false
            }
        }
        publish()
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

/// Transport decorator feeding PeerService's in-flight counter. A request
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
    private let closed = PeerLock(initialState: false)

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
