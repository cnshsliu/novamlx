import Foundation
import os

public enum TunnelStatus: Equatable, Sendable {
    case idle, connecting, connected, backingOff(seconds: Double), stopping
}

public enum TunnelError: Error { case notRegistered, connectionClosed }

/// Owns one node↔server tunnel: connect, hello, heartbeat, reconnect with
/// jittered exponential backoff, request dispatch to the Relay, cancel and
/// demand lifecycle propagation. Transport-agnostic (WS now, slow-poll later).
public final class TunnelClient: @unchecked Sendable {
    /// Everything mutable lives under one lock; status transitions are also
    /// mirrored to `statusStream` (yielded outside the lock).
    private struct State {
        var status: TunnelStatus = .idle
        var running = false
        var activeRequests: [String: RequestSlot] = [:]
        var pendingCapabilities: [Capability]?
        var currentTransport: (any TunnelTransport)?
        var demand: [DemandEntry] = []
        /// Latest operator config, if `updateConfig` ever ran. Every hello
        /// advertises these capabilities; falls back to the init config.
        var config: NodeConfig?
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    private let statusContinuation: AsyncStream<TunnelStatus>.Continuation
    public let statusStream: AsyncStream<TunnelStatus>
    private let demandContinuation: AsyncStream<[DemandEntry]>.Continuation
    public let demandStream: AsyncStream<[DemandEntry]>

    private let config: NodeConfig
    /// Internal (not private): Task 7's NodeService rebinds this relay when
    /// the operator edits config, without rebuilding the client.
    let relay: Relay
    private let transportFactory: TransportFactory
    private let delay: @Sendable (Double) async throws -> Void
    private let heartbeatInterval: TimeInterval
    private let maxBackoffSeconds: Double

    /// One in-flight tunnel request. The slot is registered in `handle`
    /// BEFORE the dispatch task exists, so a relay stream that terminates
    /// instantly can never race the concurrency accounting; the identity
    /// check on removal keeps a resurrected slot impossible.
    private final class RequestSlot: @unchecked Sendable {
        private let slotLock = OSAllocatedUnfairLock(initialState: nil as Task<Void, Never>?)
        var task: Task<Void, Never>? { slotLock.withLock { $0 } }
        func setTask(_ task: Task<Void, Never>) { slotLock.withLock { $0 = task } }
    }

    public var status: TunnelStatus { state.withLock { $0.status } }
    public var demand: [DemandEntry] { state.withLock { $0.demand } }

    /// Fired once per terminal `responseEnd` dispatched through the relay, on
    /// the dispatch task. Set before `start()`; owners (NodeService) aggregate
    /// request/token counters from it. Never carries secret material.
    public var onResult: (@Sendable (RequestResult) -> Void)?

    public init(config: NodeConfig, relay: Relay,
                transportFactory: @escaping TransportFactory,
                delay: @escaping @Sendable (Double) async throws -> Void = { seconds in
                    try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                },
                heartbeatInterval: TimeInterval = 30, maxBackoffSeconds: Double = 60) {
        self.config = config
        self.relay = relay
        self.transportFactory = transportFactory
        self.delay = delay
        self.heartbeatInterval = heartbeatInterval
        self.maxBackoffSeconds = maxBackoffSeconds
        (statusStream, statusContinuation) = AsyncStream.makeStream()
        (demandStream, demandContinuation) = AsyncStream.makeStream()
    }

    /// Connects and keeps the tunnel alive until `stop()`. Every ended
    /// session (server close, dial failure, fatal error) retries after a
    /// jittered exponential backoff.
    public func start() async {
        state.withLock { $0.running = true }
        var attempt = 0
        while state.withLock({ $0.running }) && !Task.isCancelled {
            setStatus(.connecting)
            do {
                let transport = try await transportFactory()
                setStatus(.connected)
                attempt = 0
                try await runSession(transport: transport)
                // Session ended cleanly (server closed) → reconnect below.
            } catch {
                // Dial failure or fatal session error → reconnect below.
                if Task.isCancelled { break }
            }
            guard status != .stopping else { break }
            attempt += 1
            let base = min(pow(2, Double(attempt - 1)), maxBackoffSeconds)
            let jittered = base * Double.random(in: 0.5...1.5)
            setStatus(.backingOff(seconds: jittered))
            try? await delay(jittered)
        }
        setStatus(.idle)
    }

    /// Idempotent: marks the client stopping, ends the reconnect loop, and
    /// cancels every in-flight relay dispatch.
    public func stop() {
        setStatus(.stopping)
        state.withLock { $0.running = false }
        cancelAllRequests()
    }

    /// Operator edited capabilities → push without reconnect. When not
    /// connected the new list is flushed inside the next session's hello.
    public func updateCapabilities(_ capabilities: [Capability]) async throws {
        guard case .connected = status else {
            state.withLock { $0.pendingCapabilities = capabilities }
            return
        }
        try await send(.capabilitiesUpdate(capabilities))
    }

    /// Record the latest operator config so every future hello advertises
    /// the current capabilities, never the set frozen at init. Called from
    /// `updateRelay` (NodeService.applyConfig's path).
    public func updateConfig(_ config: NodeConfig) {
        state.withLock { $0.config = config }
    }

    /// Push a config edit into the live relay (sources/prices) without a
    /// reconnect. Capability edits additionally go out as a
    /// `capabilitiesUpdate` frame via `updateCapabilities(_:)` and ride the
    /// next hello after any reconnect (`updateConfig`).
    public func updateRelay(_ config: NodeConfig) {
        updateConfig(config)
        relay.updateConfig(config)
    }

    // MARK: - Internals

    private func setStatus(_ new: TunnelStatus) {
        state.withLock { $0.status = new }
        statusContinuation.yield(new)
    }

    private func runSession(transport: any TunnelTransport) async throws {
        guard let nodeId = config.nodeId else {
            throw TunnelError.notRegistered
        }
        state.withLock { $0.currentTransport = transport }
        defer { state.withLock { $0.currentTransport = nil } }
        defer { cancelAllRequests() }

        // Advertise the LATEST capabilities (post-applyConfig), filtered to
        // the server's live demand list when we have one: retired demands
        // must not be re-advertised on reconnect. With no demand list yet
        // (fresh connect) advertise everything declared, as before.
        let helloCapabilities: [Capability] = state.withLock { state in
            let declared = state.config?.capabilities ?? config.capabilities
            guard !state.demand.isEmpty else { return declared }
            let liveIds = Set(state.demand.map(\.demandId))
            return declared.filter { liveIds.contains($0.demandId) }
        }
        try await sendOn(transport, .hello(nodeId: nodeId, capabilities: helloCapabilities))
        let pending = state.withLock { state -> [Capability]? in
            let pending = state.pendingCapabilities
            state.pendingCapabilities = nil
            return pending
        }
        if let pending {
            try await sendOn(transport, .capabilitiesUpdate(pending))
        }

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask { [heartbeatInterval, weak self] in
                    // Honour cancellation: the `try?` swallows the delay's
                    // CancellationError, so without the isCancelled checks a
                    // cancelled heartbeat task would spin forever and the
                    // group (hence the reconnect loop) could never exit.
                    while let self, self.status == .connected, !Task.isCancelled {
                        try? await self.delay(heartbeatInterval * Double.random(in: 0.9...1.1))
                        guard self.status == .connected, !Task.isCancelled else { break }
                        try await self.send(.heartbeat(Heartbeat(
                            activeReq: self.activeCount(), queueDepth: 0)))
                    }
                }
                group.addTask { [weak self] in
                    guard let self else { return }
                    for await frame in transport.inbound {
                        self.handle(frame: frame, transport: transport)
                    }
                    throw TunnelError.connectionClosed
                }
                try await group.next()
                group.cancelAll()
            }
        } catch {
            await transport.close()
            throw error
        }
        await transport.close()
    }

    private func activeCount() -> Int { state.withLock { $0.activeRequests.count } }

    private func cancelAllRequests() {
        let slots = state.withLock { state -> [RequestSlot] in
            let slots = Array(state.activeRequests.values)
            state.activeRequests = [:]
            return slots
        }
        slots.forEach { $0.task?.cancel() }
    }

    private func handle(frame: Frame, transport: any TunnelTransport) {
        switch frame {
        case .request(let request):
            // Concurrency cap (operator-declared, default 1). Over-dispatch is
            // refused with an immediate failed end frame — declining work is
            // the scheduler's problem; failing it would be ours.
            if state.withLock({ $0.activeRequests.count }) >= config.concurrencyLimit {
                let refuse = Frame.responseEnd(reqId: request.reqId, result: RequestResult(
                    status: .failed, ttftMs: 0, totalMs: 0, promptTokens: 0, completionTokens: 0,
                    upstreamStatus: 0,
                    errorMessage: "node busy (concurrency \(config.concurrencyLimit))"))
                Task { try? await sendOn(transport, refuse) }
                return
            }
            let slot = RequestSlot()
            state.withLock { $0.activeRequests[request.reqId] = slot }
            let task = Task { [relay, weak self, slot] in
                for await reply in relay.handle(request) {
                    guard let self else { return }
                    if case .responseEnd(_, let result) = reply { self.onResult?(result) }
                    try? await self.sendOn(transport, reply)
                }
                // The relay guarantees exactly one terminal `responseEnd`
                // before its stream finishes: free the slot now, only if it
                // is still ours (a same-reqId re-dispatch must not be robbed).
                self?.releaseSlot(slot, for: request.reqId)
            }
            slot.setTask(task)
        case .requestCancel(let reqId):
            let task = state.withLock { state -> Task<Void, Never>? in
                let slot = state.activeRequests[reqId]
                state.activeRequests[reqId] = nil
                return slot?.task
            }
            task?.cancel()
        case .demandUpdate(let entries):
            state.withLock { $0.demand = entries }
            demandContinuation.yield(entries)
        case .hello, .capabilitiesUpdate, .heartbeat, .responseChunk, .responseEnd, .error:
            break  // server→node protocol violation; ignore silently in v1
        }
    }

    private func releaseSlot(_ slot: RequestSlot, for reqId: String) {
        state.withLock { state in
            if state.activeRequests[reqId] === slot {
                state.activeRequests[reqId] = nil
            }
        }
    }

    /// Sends on the live session transport (heartbeat/capabilities path).
    private func send(_ frame: Frame) async throws {
        guard let transport = state.withLock({ $0.currentTransport }) else {
            throw TunnelError.connectionClosed
        }
        try await transport.send(frame)
    }

    private func sendOn(_ transport: any TunnelTransport, _ frame: Frame) async throws {
        try await transport.send(frame)
    }
}

/// Counting semaphore for async code. Task 7's NodeService uses it to cap
/// tunnel-level work; it lives here so the client module owns the primitive
/// (no dependencies, transport-agnostic).
public final class AsyncSemaphore: @unchecked Sendable {
    private struct State {
        var permits: Int
        var waiters: [CheckedContinuation<Void, Never>] = []
    }
    private let state: OSAllocatedUnfairLock<State>

    public init(permits: Int) {
        precondition(permits >= 0, "AsyncSemaphore permits must be non-negative")
        self.state = OSAllocatedUnfairLock(initialState: State(permits: permits))
    }

    /// Suspends until a permit is available.
    public func wait() async {
        await withCheckedContinuation { continuation in
            let resumeNow = state.withLock { state -> Bool in
                if state.permits > 0 {
                    state.permits -= 1
                    return true
                }
                state.waiters.append(continuation)
                return false
            }
            if resumeNow { continuation.resume() }
        }
    }

    /// Releases a permit, resuming the longest-waiting caller if any
    /// (permits are not handed to waiters beyond the first).
    public func signal() {
        let next: CheckedContinuation<Void, Never>? = state.withLock { state in
            guard let first = state.waiters.first else {
                state.permits += 1
                return nil
            }
            state.waiters.removeFirst()
            return first
        }
        next?.resume()
    }
}
