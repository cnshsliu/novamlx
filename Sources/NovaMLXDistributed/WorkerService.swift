import Foundation
import Logging
import Network

// MARK: - DiscoveredService

/// Captures the name, type, and domain of a Bonjour-discovered service.
///
/// Needed because `NWEndpoint.service` is an enum case, not a standalone type.
public struct DiscoveredService: Sendable, Equatable {
    public let name: String
    public let type: String
    public let domain: String

    public init(name: String, type: String, domain: String) {
        self.name = name
        self.type = type
        self.domain = domain
    }
}

// MARK: - WorkerService

/// Worker-side service that discovers the coordinator via Bonjour, registers this node,
/// and sends periodic heartbeats to maintain cluster membership.
///
/// Thread safety: all mutable state is guarded by ``queue``. The class is marked
/// `@unchecked Sendable` because the serial queue serialises access.
public final class WorkerService: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = WorkerService()

    // MARK: - Public properties

    /// The coordinator host resolved via config or Bonjour discovery. `nil` until resolved.
    public private(set) var coordinatorHost: String?

    /// The coordinator port resolved via config or Bonjour discovery. `nil` until resolved.
    public private(set) var coordinatorPort: Int?

    /// Whether this worker has successfully registered with the coordinator.
    public private(set) var isRegistered: Bool = false

    /// Whether the worker service loop is active.
    public private(set) var isRunning: Bool = false

    // MARK: - Private state

    /// All mutable state is accessed exclusively on this serial queue.
    private let queue = DispatchQueue(
        label: "com.novamlx.worker-service",
        qos: .userInitiated
    )

    /// Bonjour browser for coordinator discovery.
    private var browser: NWBrowser?

    /// Repeating timer that sends heartbeats to the coordinator.
    private var heartbeatTimer: Timer?

    /// Stored cluster configuration supplied at startup.
    private var config: ClusterConfig?

    /// Logger instance.
    private let logger = Logger(label: "NovaMLXDistributed.WorkerService")

    // MARK: - Lifecycle

    private init() {}

    /// Start the worker service.
    ///
    /// If `config.coordinatorHost` is a non-empty string, registration proceeds
    /// immediately. Otherwise the worker discovers the coordinator via Bonjour
    /// (`_novamlx._tcp.`) and registers automatically once found.
    ///
    /// - Parameter config: Cluster configuration (must have ``ClusterRole/worker``).
    public func start(config: ClusterConfig) {
        queue.sync {
            guard !isRunning else { return }
            self.config = config
            self.isRunning = true
            self.isRegistered = false
        }

        logger.info("WorkerService starting (coordinatorHost=\(config.coordinatorHost))")

        if !config.coordinatorHost.isEmpty {
            // Direct coordinator host provided — register immediately.
            registerWithCoordinator(host: config.coordinatorHost, port: config.coordinatorPort)
        } else {
            // Discover coordinator via Bonjour.
            discoverCoordinator()
        }
    }

    /// Shut down the worker service: stops Bonjour browser, cancels heartbeat timer.
    public func stop() {
        var browserToStop: NWBrowser?
        var timerToInvalidate: Timer?

        queue.sync {
            guard isRunning else { return }
            isRunning = false
            isRegistered = false
            coordinatorHost = nil
            coordinatorPort = nil

            browserToStop = browser
            timerToInvalidate = heartbeatTimer
            browser = nil
            heartbeatTimer = nil
            config = nil
        }

        // Clean up outside queue to avoid potential deadlock with NWBrowser callbacks.
        if let browserToStop {
            browserToStop.cancel()
        }
        if let timerToInvalidate {
            DispatchQueue.main.async { timerToInvalidate.invalidate() }
        }

        logger.info("WorkerService stopped")
    }

    // MARK: - Registration

    /// Register this worker node with the coordinator.
    ///
    /// Sends a POST request to `/admin/api/cluster/workers/register` with a ``NodeSpec``
    /// JSON body. On success, sets ``isRegistered`` to `true` and starts the heartbeat loop.
    ///
    /// - Parameters:
    ///   - host: Coordinator hostname or IP address.
    ///   - port: Coordinator admin API port.
    public func registerWithCoordinator(host: String, port: Int) {
        queue.sync {
            self.coordinatorHost = host
            self.coordinatorPort = port
        }

        logger.info("Registering with coordinator at \(host):\(port)")

        let spec = collectLocalSpec()
        guard let url = URL(string: "http://\(host):\(port)/admin/api/cluster/workers/register") else {
            logger.error("Invalid coordinator URL: \(host):\(port)")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            let encoder = JSONEncoder()
            request.httpBody = try encoder.encode(spec)
        } catch {
            logger.error("Failed to encode NodeSpec: \(error)")
            return
        }

        let task = URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            guard let self else { return }

            if let error {
                self.logger.error("Registration failed: \(error)")
                return
            }

            if let httpResponse = response as? HTTPURLResponse,
               (200...299).contains(httpResponse.statusCode)
            {
                self.queue.sync {
                    self.isRegistered = true
                }
                self.logger.info("Successfully registered with coordinator (nodeId=\(spec.nodeId))")
                self.startHeartbeat()
            } else {
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                self.logger.error("Registration rejected (HTTP \(statusCode))")
            }
        }
        task.resume()
    }

    // MARK: - Bonjour discovery

    /// Browse for the coordinator via Bonjour (`_novamlx._tcp.`).
    ///
    /// When a service is discovered, ``handleDiscoveredService(_:)`` is called to
    /// resolve its host/port and register.
    public func discoverCoordinator() {
        logger.info("Starting Bonjour discovery for _novamlx._tcp.")

        let parameters = NWParameters.tcp
        let descriptor = NWBrowser.Descriptor.bonjour(type: "_novamlx._tcp.", domain: "local.")
        let browser = NWBrowser(for: descriptor, using: parameters)

        browser.stateUpdateHandler = { [weak self] (state: NWBrowser.State) in
            switch state {
            case .ready:
                self?.logger.info("Bonjour browser ready")
            case .failed(let error):
                self?.logger.error("Bonjour browser failed: \(error)")
            case .waiting(let error):
                self?.logger.debug("Bonjour browser waiting: \(error)")
            default:
                break
            }
        }

        browser.browseResultsChangedHandler = { [weak self]
            (results: Set<NWBrowser.Result>, changes: Set<NWBrowser.Result.Change>) in
            for result in results {
                switch result.endpoint {
                case .service(let name, let type, let domain, _):
                    let service = DiscoveredService(name: name, type: type, domain: domain)
                    self?.handleDiscoveredService(service)
                default:
                    break
                }
            }
        }

        browser.start(queue: DispatchQueue.main)

        queue.sync {
            self.browser = browser
        }
    }

    /// Resolve a discovered Bonjour service and register with the coordinator.
    ///
    /// Uses `NWConnection` to resolve the endpoint's host and port, then calls
    /// ``registerWithCoordinator(host:port:)``.
    ///
    /// - Parameter service: The discovered Bonjour service details.
    public func handleDiscoveredService(_ service: DiscoveredService) {
        logger.info("Discovered coordinator service: \(service.name)")

        // Resolve the service using a connection attempt to extract host/port.
        let endpoint = NWEndpoint.service(
            name: service.name,
            type: service.type,
            domain: service.domain,
            interface: nil
        )
        let connection = NWConnection(to: endpoint, using: .tcp)

        connection.stateUpdateHandler = { [weak self] (state: NWConnection.State) in
            guard let self else { return }

            switch state {
            case .ready:
                if let remoteEndpoint = connection.currentPath?.remoteEndpoint,
                   case .hostPort(let host, let port) = remoteEndpoint
                {
                    let hostString = "\(host)"
                    let portInt = Int(port.rawValue)
                    self.logger.info("Resolved coordinator: \(hostString):\(portInt)")
                    connection.cancel()
                    self.registerWithCoordinator(host: hostString, port: portInt)
                } else {
                    self.logger.warning("Could not resolve coordinator endpoint")
                    connection.cancel()
                }

            case .failed(let error):
                self.logger.error("Coordinator resolution failed: \(error)")
                connection.cancel()

            default:
                break
            }
        }

        connection.start(queue: DispatchQueue.global(qos: .userInitiated))
    }

    // MARK: - Heartbeat

    /// Start a repeating 5-second heartbeat timer.
    ///
    /// Each tick POSTs the worker's `nodeId` to the coordinator's heartbeat endpoint.
    private func startHeartbeat() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }

            let timer = Timer.scheduledTimer(
                withTimeInterval: 5.0,
                repeats: true
            ) { [weak self] _ in
                self?.sendHeartbeat()
            }
            RunLoop.main.add(timer, forMode: .common)

            self.queue.sync {
                self.heartbeatTimer = timer
            }
        }
    }

    /// Send a single heartbeat to the coordinator.
    private func sendHeartbeat() {
        // Capture values outside queue.sync to avoid nested sync on collectLocalSpec.
        let spec = collectLocalSpec()
        let nodeId = spec.nodeId

        var host: String?
        var port: Int?
        queue.sync {
            host = coordinatorHost
            port = coordinatorPort
        }

        guard let host, let port,
              let url = URL(string: "http://\(host):\(port)/admin/api/cluster/workers/heartbeat")
        else {
            logger.error("Cannot send heartbeat — coordinator not resolved")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            let body = ["nodeId": nodeId]
            request.httpBody = try JSONEncoder().encode(body)
        } catch {
            logger.error("Failed to encode heartbeat payload: \(error)")
            return
        }

        let task = URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            if let error {
                self?.logger.error("Heartbeat failed: \(error)")
                return
            }
            if let httpResponse = response as? HTTPURLResponse,
               !(200...299).contains(httpResponse.statusCode)
            {
                self?.logger.warning("Heartbeat rejected (HTTP \(httpResponse.statusCode))")
            }
        }
        task.resume()
    }

    // MARK: - Local spec collection

    /// Collect the hardware specification of this node.
    ///
    /// - Returns: A ``NodeSpec`` populated from `ProcessInfo`:
    ///   - `nodeId` is `"\(hostname)-\(totalMemory)"`.
    ///   - `totalMemoryBytes` from `ProcessInfo.processInfo.physicalMemory`.
    ///   - `hostname` from `ProcessInfo.processInfo.hostName`.
    ///   - `port` is the admin port from the stored config (default `6591`).
    public func collectLocalSpec() -> NodeSpec {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let hostname = ProcessInfo.processInfo.hostName
        let nodeId = "\(hostname)-\(physicalMemory)"
        let port = queue.sync { config?.coordinatorPort ?? 6591 }

        return NodeSpec(
            nodeId: nodeId,
            totalMemoryBytes: physicalMemory,
            computeCapability: 1.0,
            hostname: hostname,
            port: port
        )
    }
}
