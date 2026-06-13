import Foundation
import CryptoKit
import Logging
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils

// MARK: - DiscoveredService

/// Captures the name, type, and domain of a Bonjour-discovered service.
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

/// Worker-side service in passive mode.
///
/// The Coordinator polls this worker's admin API — no outbound registration,
/// heartbeats, or SSH tunnels needed. The admin API is bound to `0.0.0.0` by
/// ``NovaMLXAPI/APIServer`` when `cluster.role == "worker"`.
///
/// Thread safety: all mutable state is guarded by ``queue``.
public final class WorkerService: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = WorkerService()

    // MARK: - Public properties

    /// Whether the worker service is active.
    public private(set) var isRunning: Bool = false

    // MARK: - Private state

    private let queue = DispatchQueue(
        label: "com.novamlx.worker-service",
        qos: .userInitiated
    )

    /// Stored cluster configuration supplied at startup.
    private var config: ClusterConfig?

    /// API key read from config.json for authenticating admin API requests.
    private var apiKey: String?

    private let logger = Logger(label: "NovaMLXDistributed.WorkerService")

    // MARK: - Lifecycle

    private init() {}

    /// Start the worker service.
    ///
    /// Simply records the config and marks the service as running.
    /// The Coordinator will discover and poll this worker via admin API.
    ///
    /// - Parameter config: Cluster configuration (must have ``ClusterRole/worker``).
    public func start(config: ClusterConfig) {
        queue.sync {
            guard !isRunning else { return }
            self.config = config
            self.isRunning = true
            self.apiKey = Self.readAPIKey()
        }
        logger.info("WorkerService started in passive mode — Coordinator polls via admin API")

        // Start WorkerShardService for distributed inference (listens on port 7010/7011)
        Task {
            do {
                try await WorkerShardService.shared.start()
                logger.info("[WorkerService] WorkerShardService listening for Coordinator shard commands")
                try await WorkerShardService.shared.run()
            } catch {
                logger.error("WorkerShardService error: \(error)")
            }
        }
    }

    /// Shut down the worker service.
    public func stop() {
        queue.sync {
            guard isRunning else { return }
            isRunning = false
            config = nil
        }
        logger.info("WorkerService stopped")
    }

    // MARK: - Local spec collection

    /// Collect the hardware specification of this node.
    public func collectLocalSpec() -> NodeSpec {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let hostname = ProcessInfo.processInfo.hostName
        let nodeId = "\(hostname)-\(physicalMemory)"
        let port = queue.sync { config?.coordinatorPort ?? 6591 }

        let fingerprint = Self.computeBinaryFingerprint()
        let cfgHash = Self.computeConfigHash()

        return NodeSpec(
            nodeId: nodeId,
            totalMemoryBytes: physicalMemory,
            computeCapability: 1.0,
            hostname: hostname,
            port: port,
            cpuModel: Self.readCPUModel(),
            binaryFingerprint: fingerprint,
            configHash: cfgHash
        )
    }

    /// Returns a simple but stable fingerprint of the running binary.
    static func computeBinaryFingerprint() -> String {
        // Use the public version + current executable modification time for now.
        let version = NovaMLXCore.version
        let execURL = Bundle.main.executableURL ?? Bundle.main.bundleURL
        let modDate = (try? FileManager.default.attributesOfItem(atPath: execURL.path)[.modificationDate] as? Date) ?? Date()
        let time = Int(modDate.timeIntervalSince1970)
        return "\(version)-\(time)"
    }

    /// Computes a hash of the authoritative cluster policy (from clusterPolicyStore).
    static func computeConfigHash() -> String? {
        guard let json = try? NovaDB.shared.clusterPolicyStore.get(),
              let data = json.data(using: .utf8) else { return nil }
        let hash = SHA256.hash(data: data)
        return hash.compactMap { String(format: "%02x", $0) }.joined()
    }

    /// Read CPU model string via sysctl (e.g. "Apple M2 Pro", "Apple M4 Max").
    static func readCPUModel() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0)
        return CString( buf).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Read first apiKey from config.json for admin API auth.
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
