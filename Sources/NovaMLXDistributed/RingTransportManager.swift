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
        NovaMLXLog.info("[RingTransport] Hostfile path: \(hostfilePath)")
        if let content = try? String(contentsOfFile: hostfilePath) {
            NovaMLXLog.info("[RingTransport] Hostfile content:\n\(content)")
        }

        logToRingDebug("Starting Ring init - hostfile=\(hostfilePath), rank=\(rank)")
        logToRingDebug("Hostfile content:\n\(try? String(contentsOfFile: hostfilePath) ?? "unreadable")")

        // Dump current IPv4 interfaces for diagnostics (very useful when debugging link-local vs static IP issues)
        dumpIPv4Interfaces()

        // === AGGRESSIVE DIAGNOSTICS ===
        dumpMLXEnvironment()
        preflightSocketTest(hostfilePath: hostfilePath, rank: rank)
        detectLinkLocalAndWarn(hostfilePath: hostfilePath, rank: rank)

        if rank == 0 {
            testRingConnectivity(hostfilePath: hostfilePath, myRank: rank)
        }

        logToRingDebug("Diagnostics complete. Starting actual MLX Ring init...")

        let start = CFAbsoluteTimeGetCurrent()

        // Run the potentially hanging init in a background queue with timeout
        // so we never hard-hang the main service thread and can report progress.
        let group = initializeRingWithTimeout(hostfilePath: hostfilePath, rank: rank, timeoutSeconds: 20)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        NovaMLXLog.info("[RingTransport] mlx_distributed_init (ring, rank=\(rank)) finished after \(String(format: "%.3f", elapsed))s")

        if group.isValid {
            NovaMLXLog.info("[RingTransport] Ring group initialized successfully: rank=\(group.rank), size=\(group.size)")
            lock.withLock { _group = group }
        } else {
            NovaMLXLog.error("[RingTransport] Ring initialization FAILED or TIMED OUT (rank=\(rank))")
            NovaMLXLog.error("[RingTransport] Common causes:")
            NovaMLXLog.error("  - Both machines still on link-local 169.254.x.x (run Scripts/setup-thunderbolt-ring.sh)")
            NovaMLXLog.error("  - Firewall / network extension blocking the ports")
            NovaMLXLog.error("  - MLX Ring C++ bug with certain interface names or IPv6 preference")
            NovaMLXLog.error("  - Worker process did not have MLX_HOSTFILE / MLX_RANK visible at launch time")
        }

        return group
    }

    private func dumpIPv4Interfaces() {
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let list = interfaces else {
            NovaMLXLog.warning("[RingTransport] Could not enumerate interfaces")
            return
        }
        defer { freeifaddrs(list) }

        var ptr = list
        NovaMLXLog.info("[RingTransport] Current IPv4 interfaces:")
        while let current = ptr.pointee.ifa_next {
            ptr = current
            guard let addr = ptr.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET),
                  let namePtr = ptr.pointee.ifa_name else { continue }

            let ifname = String(cString: namePtr)
            var ipBuffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
            var addrCopy = addr.pointee
            inet_ntop(AF_INET, &addrCopy.sa_data.2, &ipBuffer, socklen_t(INET_ADDRSTRLEN))
            let ip = String(decoding: ipBuffer.prefix(while: { $0 != 0 }).map { UInt8($0) }, as: UTF8.self)

            if ip != "127.0.0.1" {
                NovaMLXLog.info("  \(ifname): \(ip)")
            }
        }
    }

    private func dumpMLXEnvironment() {
        NovaMLXLog.info("[RingTransport] MLX environment at Ring init time:")
        let mlxKeys = ["MLX_HOSTFILE", "MLX_RANK", "MLX_RING_VERBOSE", "MLX_LOG_LEVEL", "MLX_DEBUG"]
        for key in mlxKeys {
            if let val = getenv(key) {
                NovaMLXLog.info("  \(key) = \(String(cString: val))")
            } else {
                NovaMLXLog.info("  \(key) = (not set)")
            }
        }
    }

    /// Writes very detailed information to ~/.nova/ring-debug.log
    /// This file is extremely useful when debugging Ring hangs.
    private func logToRingDebug(_ message: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logDir = home.appendingPathComponent(".nova")
        let logFile = logDir.appendingPathComponent("ring-debug.log")

        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)

        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "[\(timestamp)] [rank=\(getenv("MLX_RANK").map { String(cString: $0) } ?? "?")] \(message)\n"

        if let data = line.data(using: .utf8) {
            if FileManager.default.fileExists(atPath: logFile.path) {
                if let handle = try? FileHandle(forWritingTo: logFile) {
                    handle.seekToEndOfFile()
                    handle.write(data)
                    try? handle.close()
                }
            } else {
                try? data.write(to: logFile)
            }
        }
    }

    /// Attempts to bind/listen on the ports listed in the hostfile before asking MLX to do it.
    /// Sets SO_REUSEADDR + SO_REUSEPORT for robustness on Thunderbolt.
    private func preflightSocketTest(hostfilePath: String, rank: Int) {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: hostfilePath)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [[String]] else {
            NovaMLXLog.warning("[RingTransport] Could not parse hostfile for preflight test")
            return
        }

        guard rank < json.count else { return }

        let myAddrs = json[rank]
        NovaMLXLog.info("[RingTransport] Preflight socket test (with SO_REUSE*) for rank \(rank): \(myAddrs)")

        for addrStr in myAddrs {
            let parts = addrStr.split(separator: ":")
            guard parts.count == 2,
                  let port = UInt16(parts[1]) else { continue }

            let sock = socket(AF_INET, SOCK_STREAM, 0)
            if sock < 0 {
                NovaMLXLog.error("[RingTransport] socket() failed: errno=\(errno) (\(String(cString: strerror(errno)))")
                continue
            }

            // Be nice to Thunderbolt networking restarts
            var opt: Int32 = 1
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))
            setsockopt(sock, SOL_SOCKET, SO_REUSEPORT, &opt, socklen_t(MemoryLayout<Int32>.size))

            var sin = sockaddr_in()
            sin.sin_family = sa_family_t(AF_INET)
            sin.sin_port = in_port_t(port.bigEndian)
            sin.sin_addr.s_addr = INADDR_ANY

            let bindResult = withUnsafePointer(to: &sin) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }

            if bindResult != 0 {
                NovaMLXLog.error("[RingTransport] bind(\(port)) failed: errno=\(errno) (\(String(cString: strerror(errno))))")
            } else {
                let listenResult = listen(sock, 1)
                if listenResult != 0 {
                    NovaMLXLog.error("[RingTransport] listen(\(port)) failed: errno=\(errno)")
                } else {
                    NovaMLXLog.info("[RingTransport] Preflight ✓ bound+listening on port \(port) (SO_REUSEADDR/PORT set)")
                }
            }
            close(sock)
        }
    }

    /// Public version used by ClusterModelManager before starting the handshake.
    public func testConnectivity(hostfileJSON: String, myRank: Int) {
        let tempPath = NSTemporaryDirectory() + "ring_connectivity_test_\(myRank).json"
        try? hostfileJSON.data(using: .utf8)?.write(to: URL(fileURLWithPath: tempPath))
        testRingConnectivity(hostfilePath: tempPath, myRank: myRank)
        try? FileManager.default.removeItem(atPath: tempPath)
    }

    /// Tries to TCP connect to the *other* rank's addresses.
    /// This is the best way to know whether the Thunderbolt link is actually usable before both sides block in Ring init.
    private func testRingConnectivity(hostfilePath: String, myRank: Int) {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: hostfilePath)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [[String]] else {
            return
        }

        let otherRank = 1 - myRank
        guard otherRank < json.count else { return }

        let peerAddrs = json[otherRank]
        NovaMLXLog.info("[RingTransport] Testing TCP reachability to peer (rank \(otherRank)) addresses: \(peerAddrs)")

        for addrStr in peerAddrs {
            let parts = addrStr.split(separator: ":")
            guard parts.count == 2,
                  let port = UInt16(parts[1]),
                  let ip = parts.first else { continue }

            let sock = socket(AF_INET, SOCK_STREAM, 0)
            guard sock >= 0 else { continue }

            var sin = sockaddr_in()
            sin.sin_family = sa_family_t(AF_INET)
            sin.sin_port = in_port_t(port.bigEndian)
            inet_pton(AF_INET, String(ip), &sin.sin_addr)

            let connectResult = withUnsafePointer(to: &sin) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    connect(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }

            if connectResult == 0 {
                NovaMLXLog.info("[RingTransport] ✓ TCP connect to \(addrStr) succeeded — network path looks good")
                close(sock)
                return
            } else {
                NovaMLXLog.warning("[RingTransport] TCP connect to \(addrStr) failed: errno=\(errno) (\(String(cString: strerror(errno))))")
            }
            close(sock)
        }
        NovaMLXLog.error("[RingTransport] Could not TCP-connect to any of the peer's Ring addresses. The link is probably not ready or firewalled.")
    }

    private func detectLinkLocalAndWarn(hostfilePath: String, rank: Int) {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: hostfilePath)),
              let json = try? JSONSerialization.jsonObject(with: data) as? [[String]] else {
            return
        }

        guard rank < json.count else { return }

        let myAddrs = json[rank]
        let hasLinkLocal = myAddrs.contains { $0.hasPrefix("169.254.") }

        if hasLinkLocal {
            NovaMLXLog.error("═══════════════════════════════════════════════════════════════")
            NovaMLXLog.error("[RingTransport] *** WARNING: You are using link-local addresses (169.254.x.x) ***")
            NovaMLXLog.error("[RingTransport] MLX Ring frequently hangs or fails to connect on these addresses over Thunderbolt.")
            NovaMLXLog.error("[RingTransport] STRONGLY RECOMMENDED: Run Scripts/setup-thunderbolt-ring.sh and assign stable private IPs (10.42.0.1 / 10.42.0.2).")
            NovaMLXLog.error("═══════════════════════════════════════════════════════════════")
        }
    }

    /// Runs mlx_distributed_init in a background queue with a hard timeout.
    /// Prevents the entire worker process from hanging forever on a bad Ring init.
    private func initializeRingWithTimeout(hostfilePath: String, rank: Int, timeoutSeconds: TimeInterval) -> DistributedGroup {
        let semaphore = DispatchSemaphore(value: 0)
        var resultGroup: DistributedGroup = .uninitialized

        DispatchQueue.global(qos: .userInitiated).async {
            // We re-set the env vars inside the background task to be extra safe
            setenv("MLX_HOSTFILE", hostfilePath, 1)
            setenv("MLX_RANK", "\(rank)", 1)
            setenv("MLX_RING_VERBOSE", "1", 1)

            self.logToRingDebug("Background thread: calling mlx_distributed_init (strict=false, backend=ring)")

            NovaMLXLog.info("[RingTransport] (background) Calling MLXDistributedWrapper.initialize for ring (rank=\(rank))...")
            let g = MLXDistributedWrapper.initialize(strict: false, backend: "ring")
            resultGroup = g
            semaphore.signal()
        }

        let waitResult = semaphore.wait(timeout: .now() + timeoutSeconds)

        if waitResult == .timedOut {
            logToRingDebug("TIMEOUT after \(timeoutSeconds)s - Ring init is stuck")
            NovaMLXLog.error("[RingTransport] *** TIMEOUT: mlx_distributed_init did not return after \(timeoutSeconds)s ***")
            NovaMLXLog.error("[RingTransport] This almost always means the C++ RingGroup constructor is stuck in accept() or connect().")
            NovaMLXLog.error("[RingTransport] Check that both machines can reach each other's IPs on the Thunderbolt link.")
            return .uninitialized
        }

        logToRingDebug("Ring init completed. valid=\(resultGroup.isValid), size=\(resultGroup.size)")
        return resultGroup
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

    /// Initialize using JACCL (RDMA over Thunderbolt) backend.
    /// Tries JACCL first (auto-discovery via IBV), falls back to Ring (hostfile-based TCP).
    @discardableResult
    public func initializeJACCL(rank: Int) -> DistributedGroup {
        tearDown()

        setenv("MLX_RANK", "\(rank)", 1)

        // Try JACCL first (RDMA, zero-copy)
        if MLXDistributedWrapper.isBackendAvailable("jaccl") {
            NovaMLXLog.info("[RingTransport] Trying JACCL backend (rank=\(rank))...")
            let group = MLXDistributedWrapper.initialize(strict: false, backend: "jaccl")
            if group.isValid && group.size > 1 {
                NovaMLXLog.info("[RingTransport] JACCL group initialized: rank=\(group.rank), size=\(group.size)")
                lock.withLock { _group = group }
                return group
            }
            NovaMLXLog.warning("[RingTransport] JACCL init returned size=\(group.size) — RDMA may not be enabled. Run 'sudo rdma_ctl enable' on all nodes, then restart.")
        } else {
            NovaMLXLog.info("[RingTransport] JACCL backend not available — install mlx-rdma-gpu or enable RDMA. Falling back to Ring (TCP).")
        }

        // Fallback: Ring backend with hostfile (TCP, properly discovers peers)
        NovaMLXLog.info("[RingTransport] Falling back to Ring backend — requires hostfile from coordinator")
        // Ring init is handled by the hostfile-based initialize() method
        // JACCL auto-discovery didn't work; caller should use hostfile approach instead
        return .uninitialized
    }

    /// Initialize Ring transport with hostfile for proper 2-node discovery.
    /// Both sides must use the same hostfile and call init simultaneously.
    @discardableResult
    public func initializeRingWithHostfile(
        coordinatorIP: String,
        coordinatorPort: UInt16,
        workerIP: String,
        workerPort: UInt16,
        rank: Int
    ) -> DistributedGroup {
        let json = RingTransportManager.buildHostfileJSON(
            coordinatorIP: coordinatorIP,
            coordinatorPort: coordinatorPort,
            workerIP: workerIP,
            workerPort: workerPort
        )
        return initializeFromHostfileJSON(json, rank: rank)
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
