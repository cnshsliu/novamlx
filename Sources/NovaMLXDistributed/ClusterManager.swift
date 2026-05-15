import Foundation
import Logging
import NovaMLXCore

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
    /// Coordinator couldn't reach worker — considered offline.
    case disconnected
    /// Worker reported an unrecoverable error.
    case failed
}

// MARK: - WorkerInfo

/// Tracks registration and health state for a single worker.
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

/// Coordinator-side manager that polls Workers' admin APIs, tracks their state,
/// and exposes the active/spare worker pools for the shard engine.
///
/// **Polling model**: The Coordinator actively polls each known Worker's admin API
/// every 5 seconds. No SSH tunnels or worker→coordinator registration needed.
/// Workers simply expose their admin API on `0.0.0.0`.
///
/// Thread safety: all mutable state is guarded by ``queue``.
public final class ClusterManager: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = ClusterManager()

    // MARK: - Public properties

    public private(set) var config: ClusterConfig?
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

    private let queue = DispatchQueue(label: "com.novamlx.cluster-manager", qos: .userInitiated)
    private var _workers: [String: WorkerInfo] = [:]
    private var netService: NetService?
    private var pollTimer: Timer?
    private let logger = Logger(label: "NovaMLXDistributed.ClusterManager")

    /// Consecutive poll failures per worker nodeId — triggers disconnect after 3.
    private var pollFailCount: [String: Int] = [:]

    /// API key for authenticating with Worker admin APIs.
    private var apiKey: String?

    // MARK: - Lifecycle

    private init() {}

    /// Start the coordinator: stores config, begins polling known workers.
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
            self.pollFailCount = [:]
            self.apiKey = Self.readAPIKey()
        }

        logger.info("ClusterManager starting as coordinator — polling mode")

        // Load persisted worker deployments so we know which IPs to poll
        WorkerDeployer.shared.loadDeployments()

        advertiseBonjour(port: config.coordinatorPort)
        startPolling()
        pollKnownWorkers()
    }

    /// Shut down the coordinator.
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
            timerRef = pollTimer
            netService = nil
            pollTimer = nil
            config = nil
        }

        for nodeId in removedIds {
            onWorkerDisconnected?(nodeId)
        }

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

    // MARK: - Worker management

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

        logger.info("Worker registered: \(spec.nodeId) (\(spec.hostname), memory=\(spec.totalMemoryBytes))")
        return info
    }

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

    public func removeWorker(nodeId: String) {
        let existed = queue.sync {
            let removed = _workers.removeValue(forKey: nodeId) != nil
            pollFailCount.removeValue(forKey: nodeId)
            return removed
        }

        if existed {
            logger.info("Worker removed: \(nodeId)")
            onWorkerDisconnected?(nodeId)
        }
    }

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

    // MARK: - Worker polling

    /// Start a 5-second repeating poll of all known worker endpoints.
    private func startPolling() {
        DispatchQueue.main.async { [self] in
            let timer = Timer.scheduledTimer(
                withTimeInterval: 5.0,
                repeats: true
            ) { [weak self] _ in
                self?.pollKnownWorkers()
            }
            self.pollTimer = timer
            RunLoop.main.add(timer, forMode: .common)
        }
    }

    /// Poll every known Worker's admin API — discovered via deployments or manual registration.
    func pollKnownWorkers() {
        let endpoints = collectWorkerEndpoints()
        let key = queue.sync { apiKey }

        for (host, port) in endpoints {
            pollWorker(host: host, port: port, apiKey: key)
        }
    }

    /// Gather (host, port) pairs from WorkerDeployer deployments + existing workers.
    private func collectWorkerEndpoints() -> [(host: String, port: Int)] {
        var endpoints: [(String, Int)] = []

        // From deployments — WorkerDeployer records
        for (_, deployment) in WorkerDeployer.shared.deployments {
            guard deployment.phase == .running || deployment.phase == .idle else { continue }
            endpoints.append((deployment.host, 6591))
        }

        // From existing workers — use their stored hostname/IP
        let workerSnapshot = queue.sync { _workers }
        for (_, worker) in workerSnapshot {
            let host = worker.spec.hostname
            let port = worker.spec.port
            // Avoid duplicates
            if !endpoints.contains(where: { $0.0 == host && $0.1 == port }) {
                endpoints.append((host, port))
            }
        }

        return endpoints
    }

    /// Poll a single Worker's admin API for cluster status.
    private func pollWorker(host: String, port: Int, apiKey: String?) {
        guard let url = URL(string: "http://\(host):\(port)/admin/api/cluster/status") else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 5.0
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let task = URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            guard let self else { return }

            if let error {
                self.handlePollFailure(host: host, port: port, error: error)
                return
            }

            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode),
                  let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else {
                self.handlePollFailure(host: host, port: port, error: nil)
                return
            }

            self.handlePollSuccess(host: host, port: port, json: json)
        }
        task.resume()
    }

    /// Process a successful poll — update or register the worker.
    private func handlePollSuccess(host: String, port: Int, json: [String: Any]) {
        guard let localSpec = json["localSpec"] as? [String: Any] else {
            logger.warning("Worker \(host):\(port) responded but missing localSpec")
            return
        }

        let nodeId = localSpec["nodeId"] as? String ?? "\(host)-\(port)"
        let hostname = localSpec["hostname"] as? String ?? host
        let memory = localSpec["memory"] as? UInt64 ?? 0
        let cpuModel = localSpec["cpuModel"] as? String ?? ""

        let spec = NodeSpec(
            nodeId: nodeId,
            totalMemoryBytes: memory,
            computeCapability: 1.0,
            hostname: hostname,
            port: port,
            cpuModel: cpuModel,
            networkHost: host  // Store the actual IP for TCP connections
        )

        queue.sync {
            pollFailCount[nodeId] = 0
            if var existing = _workers[nodeId] {
                // Update heartbeat, promote to ready on successful poll
                existing.lastHeartbeat = Date()
                // spec updated on next full discovery
                if existing.status == .disconnected || existing.status == .registering {
                    existing.status = .ready
                    logger.info("Worker \(nodeId) \(existing.status == .disconnected ? "back online" : "registered → ready")")
                }
                _workers[nodeId] = existing
            } else {
                // New worker discovered via poll
                let info = WorkerInfo(
                    nodeId: nodeId,
                    spec: spec,
                    status: .ready,
                    registeredAt: Date(),
                    lastHeartbeat: Date()
                )
                _workers[nodeId] = info
                logger.info("Worker discovered via poll: \(nodeId) (\(hostname), \(cpuModel), \(memory) bytes)")
            }
        }
    }

    /// Handle poll failure — increment fail count, disconnect after 3 misses.
    private func handlePollFailure(host: String, port: Int, error: Error?) {
        let nodeId = queue.sync {
            // Find worker by hostname or host
            for (_, worker) in _workers {
                if worker.spec.hostname == host || worker.spec.hostname.contains(host) {
                    return worker.nodeId
                }
            }
            return "\(host)-\(port)"
        }

        let failCount = queue.sync {
            pollFailCount[nodeId, default: 0] + 1
        }

        if failCount >= 3 {
            var wasActive = false
            queue.sync {
                pollFailCount[nodeId] = failCount
                if var worker = _workers[nodeId] {
                    wasActive = worker.status != .disconnected
                    worker.status = .disconnected
                    worker.lastHeartbeat = Date()
                    _workers[nodeId] = worker
                }
            }

            if wasActive {
                if let error {
                    logger.warning("Worker \(nodeId) unreachable after \(failCount) polls: \(error.localizedDescription)")
                } else {
                    logger.warning("Worker \(nodeId) unreachable after \(failCount) polls")
                }
                onWorkerDisconnected?(nodeId)
            }
        } else {
            queue.sync { pollFailCount[nodeId] = failCount }
            if let error {
                logger.debug("Worker \(host):\(port) poll failed (\(failCount)/3): \(error.localizedDescription)")
            }
        }
    }

    // MARK: - Debug introspection

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
                    "cpuModel": worker.spec.cpuModel,
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

    // MARK: - Helpers

    /// Read first apiKey from config.json for authenticating with Worker admin APIs.
    private static func readAPIKey() -> String? {
        let configPath = NovaMLXPaths.configFile
        guard let data = try? Data(contentsOf: configPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let server = json["server"] as? [String: Any],
              let keys = server["apiKeys"] as? [String],
              let first = keys.first else { return nil }
        return first
    }
}
