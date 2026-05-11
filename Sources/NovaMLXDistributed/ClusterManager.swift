import Foundation
import Logging

// MARK: - ClusterError

/// Errors raised by ``ClusterManager`` operations.
public enum ClusterError: Error, LocalizedError {
    case noWorkersRegistered
    case workerNotFound(nodeId: String)
    case alreadyInitialized
    case notCoordinator

    public var errorDescription: String? {
        switch self {
        case .noWorkersRegistered:
            "No workers registered in the cluster"
        case .workerNotFound(let nodeId):
            "Worker not found: \(nodeId)"
        case .alreadyInitialized:
            "ClusterManager is already running"
        case .notCoordinator:
            "This node is not configured as coordinator"
        }
    }
}

// MARK: - WorkerStatus

/// Lifecycle status of a worker node.
public enum WorkerStatus: String, Codable, Sendable, Equatable {
    /// Worker has announced itself but handshake is not yet complete.
    case registering
    /// Worker is idle and available for shard assignment.
    case ready
    /// Worker is currently loading a model shard.
    case loading
    /// Worker is actively serving inference requests.
    case active
    /// Worker is syncing model weights.
    case syncing
    /// Worker missed heartbeat threshold — considered offline.
    case disconnected
    /// Worker reported an unrecoverable error.
    case failed
}

// MARK: - WorkerInfo

/// Tracks registration and health state for a single worker node.
public struct WorkerInfo: Codable, Sendable, Equatable {
    public let nodeId: String
    public let spec: NodeSpec
    public var status: WorkerStatus
    public let registeredAt: Date
    public var lastHeartbeat: Date

    public init(
        nodeId: String,
        spec: NodeSpec,
        status: WorkerStatus = .registering,
        registeredAt: Date = Date(),
        lastHeartbeat: Date = Date()
    ) {
        self.nodeId = nodeId
        self.spec = spec
        self.status = status
        self.registeredAt = registeredAt
        self.lastHeartbeat = lastHeartbeat
    }
}

// MARK: - ClusterManager

/// Coordinator-side manager that discovers workers via Bonjour, tracks registration
/// and heartbeats, and exposes the active/spare worker pools for the shard engine.
///
/// Thread safety: all mutable state is guarded by ``queue``. The class is marked
/// `@unchecked Sendable` because the serial queue serialises access.
public final class ClusterManager: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = ClusterManager()

    // MARK: - Public properties

    /// The cluster configuration supplied at startup. `nil` before ``startAsCoordinator(config:)``.
    public private(set) var config: ClusterConfig?

    /// Whether the coordinator loop is active.
    public private(set) var isRunning: Bool = false

    /// Callback fired when a worker transitions to ``WorkerStatus/disconnected``.
    public var onWorkerDisconnected: ((String) -> Void)?

    /// Snapshot of all registered workers (thread-safe copy).
    public var workers: [String: WorkerInfo] {
        queue.sync { _workers }
    }

    /// Workers in ``WorkerStatus/active`` status.
    public var activeWorkers: [WorkerInfo] {
        queue.sync {
            _workers.values.filter { $0.status == .active }
        }
    }

    /// Workers in ``WorkerStatus/ready`` status — available for shard assignment.
    public var spareWorkers: [WorkerInfo] {
        queue.sync {
            _workers.values.filter { $0.status == .ready }
        }
    }

    // MARK: - Private state

    /// All mutable state is accessed exclusively on this serial queue.
    private let queue = DispatchQueue(label: "com.novamlx.cluster-manager", qos: .userInitiated)

    /// Underlying worker storage — access only via ``queue``.
    private var _workers: [String: WorkerInfo] = [:]

    /// Bonjour service published by the coordinator.
    private var netService: NetService?

    /// Repeating timer that checks worker health.
    private var heartbeatTimer: Timer?

    /// Logger instance.
    private let logger = Logger(label: "NovaMLXDistributed.ClusterManager")

    // MARK: - Lifecycle

    private init() {}

    /// Start the coordinator: stores config, advertises via Bonjour, begins heartbeat monitoring.
    ///
    /// - Parameter config: Cluster configuration (must have ``ClusterRole/coordinator``).
    public func startAsCoordinator(config: ClusterConfig) throws {
        try queue.sync {
            guard !isRunning else {
                throw ClusterError.alreadyInitialized
            }
            guard config.role == .coordinator else {
                throw ClusterError.notCoordinator
            }

            self.config = config
            self.isRunning = true
            self._workers = [:]
        }

        logger.info("ClusterManager starting as coordinator on \(config.coordinatorHost):\(config.coordinatorPort)")

        advertiseBonjour(port: config.coordinatorPort)
        startHeartbeatMonitoring()
    }

    /// Shut down the coordinator: stops Bonjour, cancels timers, clears workers.
    public func stop() {
        var removedIds: [String] = []
        var netServiceRef: NetService?
        var timerRef: Timer?

        queue.sync {
            guard isRunning else { return }
            isRunning = false

            removedIds = Array(_workers.keys)
            _workers.removeAll()

            netServiceRef = netService
            timerRef = heartbeatTimer
            netService = nil
            heartbeatTimer = nil
            config = nil
        }

        // Fire disconnect callbacks outside queue.sync to avoid deadlock.
        // (Callbacks may call back into ClusterManager.)
        for nodeId in removedIds {
            onWorkerDisconnected?(nodeId)
        }

        // Bonjour / Timer cleanup on main run-loop.
        // NetService.stop() and Timer.invalidate() must run on the main run-loop.
        if let svc = netServiceRef {
            nonisolated(unsafe) let s = svc
            DispatchQueue.main.async { s.stop() }
        }
        if let t = timerRef {
            nonisolated(unsafe) let timer = t
            DispatchQueue.main.async { timer.invalidate() }
        }

        logger.info("ClusterManager stopped")
    }

    // MARK: - Worker registration

    /// Register a new worker node.
    ///
    /// - Parameter spec: Hardware specification of the joining worker.
    /// - Returns: The ``WorkerInfo`` record created for the worker.
    @discardableResult
    public func registerWorker(spec: NodeSpec) -> WorkerInfo {
        let info = WorkerInfo(
            nodeId: spec.nodeId,
            spec: spec,
            status: .registering
        )

        queue.sync {
            _workers[spec.nodeId] = info
        }

        logger.info("Worker registered: \(spec.nodeId) (\(spec.hostname):\(spec.port), memory=\(spec.totalMemoryBytes))")
        return info
    }

    /// Refresh the heartbeat timestamp for a known worker.
    ///
    /// If the worker was ``WorkerStatus/disconnected``, it is promoted back to
    /// ``WorkerStatus/registering`` so the coordinator can re-evaluate it.
    public func updateHeartbeat(nodeId: String) {
        queue.sync {
            guard var worker = _workers[nodeId] else { return }
            worker.lastHeartbeat = Date()
            if worker.status == .disconnected {
                worker.status = .registering
                logger.info("Worker \(nodeId) reconnected after disconnect")
            }
            _workers[nodeId] = worker
        }
    }

    /// Remove a worker from the cluster.
    public func removeWorker(nodeId: String) {
        let existed: Bool
        existed = queue.sync {
            let removed = _workers.removeValue(forKey: nodeId) != nil
            return removed
        }

        if existed {
            logger.info("Worker removed: \(nodeId)")
            onWorkerDisconnected?(nodeId)
        }
    }

    /// Update the status of a specific worker.
    public func setWorkerStatus(nodeId: String, status: WorkerStatus) throws {
        try queue.sync {
            guard var worker = _workers[nodeId] else {
                throw ClusterError.workerNotFound(nodeId: nodeId)
            }
            let old = worker.status
            worker.status = status
            _workers[nodeId] = worker
            logger.debug("Worker \(nodeId) status: \(old) -> \(status)")
        }
    }

    // MARK: - Bonjour discovery

    /// Advertise the coordinator via Bonjour on `_novamlx._tcp.`.
    private func advertiseBonjour(port: Int) {
        DispatchQueue.main.async { [self] in
            let service = NetService(
                domain: "",
                type: "_novamlx._tcp.",
                name: "NovaMLX-Coordinator",
                port: Int32(port)
            )
            service.publish(options: [.listenForConnections])
            self.netService = service
            self.logger.info("Bonjour service published on port \(port)")
        }
    }

    // MARK: - Heartbeat monitoring

    /// Start a repeating timer (5-second interval) that checks worker health.
    ///
    /// Workers that have not sent a heartbeat within the timeout (30 seconds)
    /// are marked ``WorkerStatus/disconnected`` and the ``onWorkerDisconnected``
    /// callback is fired.
    private func startHeartbeatMonitoring() {
        DispatchQueue.main.async { [self] in
            let timer = Timer.scheduledTimer(
                withTimeInterval: 5.0,
                repeats: true
            ) { [weak self] _ in
                self?.checkWorkerHealth()
            }
            self.heartbeatTimer = timer
            RunLoop.main.add(timer, forMode: .common)
        }
    }

    /// Check all workers and mark stale ones as disconnected.
    private func checkWorkerHealth() {
        let timeout: TimeInterval = 30.0
        let now = Date()
        var disconnectedIds: [String] = []

        queue.sync {
            for (nodeId, worker) in _workers where worker.status != .disconnected && worker.status != .failed {
                if now.timeIntervalSince(worker.lastHeartbeat) > timeout {
                    _workers[nodeId]?.status = .disconnected
                    disconnectedIds.append(nodeId)
                }
            }
        }

        for nodeId in disconnectedIds {
            logger.warning("Worker \(nodeId) heartbeat timeout — marking disconnected")
            onWorkerDisconnected?(nodeId)
        }
    }

    // MARK: - Debug introspection

    /// Returns a dictionary summarising cluster state for the admin API debug endpoint.
    public func discoveryDebugInfo() -> [String: Any] {
        queue.sync {
            var workersArray: [[String: Any]] = []
            for (_, worker) in _workers.sorted(by: { $0.key < $1.key }) {
                workersArray.append([
                    "nodeId": worker.nodeId,
                    "hostname": worker.spec.hostname,
                    "port": worker.spec.port,
                    "status": worker.status.rawValue,
                    "totalMemoryBytes": worker.spec.totalMemoryBytes,
                    "computeCapability": worker.spec.computeCapability,
                    "registeredAt": ISO8601DateFormatter().string(from: worker.registeredAt),
                    "lastHeartbeat": ISO8601DateFormatter().string(from: worker.lastHeartbeat),
                ])
            }

            return [
                "isRunning": isRunning,
                "role": config?.role.rawValue ?? "none",
                "coordinatorPort": config?.coordinatorPort ?? 0,
                "strategy": config?.strategy.rawValue ?? "none",
                "totalWorkers": _workers.count,
                "activeWorkers": _workers.values.filter { $0.status == .active }.count,
                "spareWorkers": _workers.values.filter { $0.status == .ready }.count,
                "disconnectedWorkers": _workers.values.filter { $0.status == .disconnected }.count,
                "workers": workersArray,
            ]
        }
    }
}
