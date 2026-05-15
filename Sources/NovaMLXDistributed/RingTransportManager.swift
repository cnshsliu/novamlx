import Foundation
import MLX
import Cmlx
import NovaMLXCore
import NovaMLXUtils

/// Manages the MLX Ring distributed group lifecycle.
///
/// The Ring backend uses TCP sockets for inter-process array transfer.
/// It requires a JSON hostfile and MLX_RANK env var before initialization.
///
/// Hostfile format:
/// ```json
/// [
///   ["coordinator_ip:port1", "coordinator_ip:port2"],
///   ["worker_ip:port1", "worker_ip:port2"]
/// ]
/// ```
///
/// Multiple addresses per rank enable parallel TCP connections for throughput.
public final class RingTransportManager: @unchecked Sendable {

    public static let shared = RingTransportManager()

    private let lock = NSLock()
    private var _group: DistributedGroup = .uninitialized
    private var _hostfilePath: String?

    /// The active Ring distributed group, or `.uninitialized` if not set up.
    public var group: DistributedGroup {
        lock.withLock { _group }
    }

    /// Whether the Ring group is initialized and ready for send/recv.
    public var isReady: Bool {
        lock.withLock { _group.isValid }
    }

    private init() {}

    // MARK: - Setup

    /// Initialize the Ring distributed group.
    ///
    /// - Parameters:
    ///   - hosts: Array of (ip, port) pairs for each rank. Index 0 = coordinator.
    ///   - rank: This process's rank (0 = coordinator, 1 = worker, etc.)
    ///   - connectionsPerRank: Number of parallel TCP connections per peer (default: 2).
    /// - Returns: The initialized distributed group.
    @discardableResult
    public func initialize(
        hosts: [(ip: String, port: UInt16)],
        rank: Int,
        connectionsPerRank: Int = 2
    ) -> DistributedGroup {
        // Tear down existing group
        tearDown()

        // Build hostfile JSON
        var hostfileEntries: [[String]] = []
        for host in hosts {
            var addresses: [String] = []
            for i in 0..<connectionsPerRank {
                addresses.append("\(host.ip):\(host.port + UInt16(i))")
            }
            hostfileEntries.append(addresses)
        }

        guard let hostfileJSON = try? JSONSerialization.data(
            withJSONObject: hostfileEntries,
            options: [.prettyPrinted, .sortedKeys]
        ) else {
            NovaMLXLog.error("[RingTransport] Failed to serialize hostfile JSON")
            return .uninitialized
        }

        // Write hostfile to temp location
        let hostfilePath = NSTemporaryDirectory() + "mlx_ring_hostfile_\(rank).json"
        guard FileManager.default.createFile(atPath: hostfilePath, contents: hostfileJSON) else {
            NovaMLXLog.error("[RingTransport] Failed to write hostfile to \(hostfilePath)")
            return .uninitialized
        }

        NovaMLXLog.info("[RingTransport] Hostfile written to \(hostfilePath)")
        NovaMLXLog.info("[RingTransport] Hostfile content: \(String(data: hostfileJSON, encoding: .utf8) ?? "?")")

        lock.withLock { _hostfilePath = hostfilePath }

        // Set env vars for Ring backend
        setenv("MLX_HOSTFILE", hostfilePath, 1)
        setenv("MLX_RANK", "\(rank)", 1)
        setenv("MLX_RING_VERBOSE", "1", 1)  // Enable ring logging for debugging

        NovaMLXLog.info("[RingTransport] Initializing Ring backend (rank=\(rank), hosts=\(hosts.count))...")

        let group = MLXDistributedWrapper.initialize(strict: true, backend: "ring")

        if group.isValid {
            NovaMLXLog.info("[RingTransport] Ring group initialized: rank=\(group.rank), size=\(group.size)")
            lock.withLock { _group = group }
        } else {
            NovaMLXLog.error("[RingTransport] Ring group initialization failed")
        }

        return group
    }

    /// Initialize using a pre-built hostfile JSON string (sent from coordinator to worker).
    @discardableResult
    public func initializeFromHostfileJSON(
        _ json: String,
        rank: Int
    ) -> DistributedGroup {
        guard let data = json.data(using: .utf8) else {
            NovaMLXLog.error("[RingTransport] Invalid hostfile JSON string")
            return .uninitialized
        }

        let hostfilePath = NSTemporaryDirectory() + "mlx_ring_hostfile_\(rank).json"
        guard FileManager.default.createFile(atPath: hostfilePath, contents: data) else {
            NovaMLXLog.error("[RingTransport] Failed to write hostfile")
            return .uninitialized
        }

        lock.withLock { _hostfilePath = hostfilePath }

        setenv("MLX_HOSTFILE", hostfilePath, 1)
        setenv("MLX_RANK", "\(rank)", 1)
        setenv("MLX_RING_VERBOSE", "1", 1)

        NovaMLXLog.info("[RingTransport] Initializing Ring backend from hostfile JSON (rank=\(rank))...")

        let group = MLXDistributedWrapper.initialize(strict: true, backend: "ring")

        if group.isValid {
            NovaMLXLog.info("[RingTransport] Ring group initialized: rank=\(group.rank), size=\(group.size)")
            lock.withLock { _group = group }
        } else {
            NovaMLXLog.error("[RingTransport] Ring group initialization failed")
        }

        return group
    }

    // MARK: - Transport

    /// Send an array to the specified rank via Ring transport.
    public func send(_ array: MLXArray, to dst: Int) -> MLXArray {
        let g = lock.withLock { _group }
        guard g.isValid else {
            NovaMLXLog.error("[RingTransport] Cannot send: group not initialized")
            return array
        }
        return MLXDistributedWrapper.send(array, to: dst, group: g)
    }

    /// Receive an array from the specified rank with known shape.
    public func recv(shape: [Int], dtype: DType = .float32, from src: Int) -> MLXArray {
        let g = lock.withLock { _group }
        guard g.isValid else {
            NovaMLXLog.error("[RingTransport] Cannot recv: group not initialized")
            return MLXArray.zeros(shape)
        }
        return MLXDistributedWrapper.recv(shape: shape, dtype: dtype, from: src, group: g)
    }

    /// Receive an array matching the reference's shape and dtype.
    public func recvLike(_ reference: MLXArray, from src: Int) -> MLXArray {
        let g = lock.withLock { _group }
        guard g.isValid else {
            NovaMLXLog.error("[RingTransport] Cannot recvLike: group not initialized")
            return MLXArray.zeros(reference.shape)
        }
        return MLXDistributedWrapper.recvLike(reference, from: src, group: g)
    }

    // MARK: - Teardown

    /// Tear down the Ring group and clean up resources.
    public func tearDown() {
        lock.withLock {
            _group = .uninitialized
            if let path = _hostfilePath {
                try? FileManager.default.removeItem(atPath: path)
                _hostfilePath = nil
            }
        }
        // Clear env vars
        unsetenv("MLX_HOSTFILE")
        unsetenv("MLX_RANK")
        unsetenv("MLX_RING_VERBOSE")
        NovaMLXLog.info("[RingTransport] Torn down")
    }

    /// Build the hostfile JSON for the given hosts.
    public static func buildHostfileJSON(
        coordinatorIP: String,
        coordinatorPort: UInt16,
        workerIP: String,
        workerPort: UInt16,
        connectionsPerRank: Int = 2
    ) -> String {
        var coordAddrs: [String] = []
        var workerAddrs: [String] = []
        for i in 0..<connectionsPerRank {
            coordAddrs.append("\(coordinatorIP):\(coordinatorPort + UInt16(i))")
            workerAddrs.append("\(workerIP):\(workerPort + UInt16(i))")
        }
        let entries = [coordAddrs, workerAddrs] as [[String]]
        guard let data = try? JSONSerialization.data(withJSONObject: entries, options: .prettyPrinted) else {
            return "[]"
        }
        return String(data: data, encoding: .utf8) ?? "[]"
    }
}
