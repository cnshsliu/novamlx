import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXUtils

// MARK: - Wire Format

/// Binary wire format for tensor transport over TCP.
///
/// Layout per message:
/// ```
/// [4B]  magic:     0x4E4F5641 ("NOVA")
/// [4B]  msgType:   1=tensor, 2=ack, 3=error, 4=shutdown
/// [4B]  ndim:      number of dimensions
/// [4B]  dtype:     mlx_dtype raw value (UInt32)
/// [8B]  nbytes:    total tensor bytes
/// [4B]  sequence:  sequence number
/// [4B]  reserved:  padding
/// [ndim × 8B] shape: each dimension as Int64
/// [nbytes]     data: raw tensor bytes
/// ```
enum WireFormat {
    static let magic: UInt32 = 0x4E4F5641
    static let headerSize = 32

    enum MessageType: UInt32 {
        case tensor = 1
        case ack = 2
        case error = 3
        case shutdown = 4
    }

    struct TensorHeader {
        let magic: UInt32
        let msgType: UInt32
        let ndim: UInt32
        let dtypeRaw: UInt32
        let nbytes: UInt64
        let sequence: UInt32
        let reserved: UInt32
    }

    /// Encode a tensor header + shape + data into a single Data buffer.
    static func encode(array: MLXArray) -> Data {
        let arrayData = array.asData(access: .copy)
        let shape = arrayData.shape
        let dtype = arrayData.dType
        let ndim = shape.count
        let nbytes = arrayData.data.count

        // Header: 32 bytes
        var buf = Data(capacity: headerSize + ndim * 8 + nbytes)
        buf.append(contentsOf: withUnsafeBytes(of: magic.bigEndian) { Array($0) })
        buf.append(contentsOf: withUnsafeBytes(of: MessageType.tensor.rawValue.bigEndian) { Array($0) })
        buf.append(contentsOf: withUnsafeBytes(of: UInt32(ndim).bigEndian) { Array($0) })
        buf.append(contentsOf: withUnsafeBytes(of: dtype.cmlxDtype.rawValue.bigEndian) { Array($0) })
        buf.append(contentsOf: withUnsafeBytes(of: nbytes.bigEndian) { Array($0) })
        buf.append(contentsOf: withUnsafeBytes(of: UInt32(0).bigEndian) { Array($0) }) // sequence
        buf.append(contentsOf: withUnsafeBytes(of: UInt32(0).bigEndian) { Array($0) }) // reserved

        // Shape: ndim × 8 bytes
        for dim in shape {
            var dim64 = Int64(dim).bigEndian
            let dimData = withUnsafeBytes(of: &dim64) { Data($0) }
            buf.append(dimData)
        }

        // Tensor data
        buf.append(arrayData.data)
        return buf
    }

    /// Decode a tensor from raw bytes (header + shape + data).
    static func decode(_ data: Data) throws -> MLXArray {
        guard data.count >= headerSize else {
            throw TransportError.invalidHeader("too short: \(data.count) bytes")
        }

        // Parse header
        let m = data.readUInt32(at: 0)
        guard m == magic else {
            throw TransportError.invalidMagic
        }
        let msgType = data.readUInt32(at: 4)
        guard msgType == MessageType.tensor.rawValue else {
            throw TransportError.invalidHeader("unexpected msgType: \(msgType)")
        }
        let ndim = Int(data.readUInt32(at: 8))
        let dtypeRaw = data.readUInt32(at: 12)
        let nbytes = Int(data.readUInt64(at: 16))

        guard let dtype = DTypeFromRaw(dtypeRaw) else {
            throw TransportError.invalidDType(dtypeRaw)
        }

        let shapeStart = headerSize
        let shapeEnd = shapeStart + ndim * 8
        let dataEnd = shapeEnd + nbytes

        guard data.count >= dataEnd else {
            throw TransportError.invalidHeader(
                "incomplete: expected \(dataEnd) bytes, got \(data.count)")
        }

        // Parse shape
        var shape = [Int]()
        shape.reserveCapacity(ndim)
        for i in 0..<ndim {
            let offset = shapeStart + i * 8
            var dim64: Int64 = 0
            withUnsafeMutableBytes(of: &dim64) { ptr in
                data.copyBytes(to: ptr, from: offset..<(offset + 8))
            }
            shape.append(Int(Int64(bigEndian: dim64)))
        }

        // Extract tensor data
        let tensorData = data.subdata(in: shapeEnd..<dataEnd)
        return MLXArray(tensorData, shape, dtype: dtype)
    }

    /// Compute total wire size for an array (without encoding).
    static func wireSize(for array: MLXArray) -> Int {
        let d = array.asData(access: .copy)
        return headerSize + d.shape.count * 8 + d.data.count
    }
}

// MARK: - Transport Errors

public enum TransportError: Error, CustomStringConvertible {
    case connectionFailed(String)
    case sendFailed(String)
    case recvFailed(String)
    case invalidMagic
    case invalidDType(UInt32)
    case invalidHeader(String)
    case notConnected(nodeId: String)
    case timeout(nodeId: String)
    case cancelled

    public var description: String {
        switch self {
        case .connectionFailed(let msg): "TransportError.connectionFailed: \(msg)"
        case .sendFailed(let msg): "TransportError.sendFailed: \(msg)"
        case .recvFailed(let msg): "TransportError.recvFailed: \(msg)"
        case .invalidMagic: "TransportError.invalidMagic"
        case .invalidDType(let raw): "TransportError.invalidDType(\(raw))"
        case .invalidHeader(let msg): "TransportError.invalidHeader: \(msg)"
        case .notConnected(let id): "TransportError.notConnected(\(id))"
        case .timeout(let id): "TransportError.timeout(\(id))"
        case .cancelled: "TransportError.cancelled"
        }
    }
}

// MARK: - Node Endpoint

/// Network address of a cluster node.
public struct NodeEndpoint: Sendable, Equatable, Codable {
    public let nodeId: String
    public let host: String
    public let port: UInt16

    public init(nodeId: String, host: String, port: UInt16) {
        self.nodeId = nodeId
        self.host = host
        self.port = port
    }
}

// MARK: - TensorTransport Protocol

/// Abstraction for sending/receiving tensors between cluster nodes.
///
/// Design: data plane / control plane separation.
/// TensorTransport handles the data plane (raw binary tensor transfer).
/// Control plane (discovery, health, shard assignment) is handled separately.
public protocol TensorTransport: Sendable {
    /// Send a tensor to a remote node.
    func send(_ array: MLXArray, to nodeId: String) async throws

    /// Receive a tensor from a remote node. Blocks until data arrives.
    func recv(from nodeId: String) async throws -> MLXArray

    /// Establish a TCP connection to a remote node.
    func connect(to endpoint: NodeEndpoint) async throws

    /// Start listening for incoming connections on the given port.
    func startListening(port: UInt16) async throws

    /// Graceful shutdown — closes all connections and stops listening.
    func shutdown()
}

// MARK: - TCPTensorTransport

/// Concrete ``TensorTransport`` using raw TCP sockets with binary framing.
///
/// Performance characteristics:
/// - Zero-copy Data extraction via MLXArray.asData(access: .copy)
/// - TCP_NODELAY for minimum latency
/// - 4MB socket buffers for large prefill activations
/// - Persistent connections — one per node pair, kept alive
public final class TCPTensorTransport: TensorTransport, @unchecked Sendable {

    private var connections: [String: TCPConnection] = [:]
    private var listener: TCPListener?
    private let lock = NSLock()
    private var isShutdown = false

    public init() {}

    public func send(_ array: MLXArray, to nodeId: String) async throws {
        guard let conn = lock.withLock({ connections[nodeId] }) else {
            throw TransportError.notConnected(nodeId: nodeId)
        }
        try conn.send(array)
    }

    public func recv(from nodeId: String) async throws -> MLXArray {
        guard let conn = lock.withLock({ connections[nodeId] }) else {
            throw TransportError.notConnected(nodeId: nodeId)
        }
        return try conn.recv()
    }

    public func connect(to endpoint: NodeEndpoint) async throws {
        let conn = try TCPConnection(to: endpoint)
        lock.withLock { connections[endpoint.nodeId] = conn }
        NovaMLXLog.debug("[TensorTransport] Connected to \(endpoint.nodeId) at \(endpoint.host):\(endpoint.port)")
    }

    public func startListening(port: UInt16) async throws {
        let listener = try TCPListener(port: port) { [weak self] nodeId, conn in
            self?.lock.withLock { self?.connections[nodeId] = conn }
            NovaMLXLog.debug("[TensorTransport] Accepted connection from \(nodeId)")
        }
        lock.withLock { self.listener = listener }
        NovaMLXLog.info("[TensorTransport] Listening on port \(port)")
    }

    public func shutdown() {
        lock.withLock {
            isShutdown = true
            connections.values.forEach { $0.close() }
            connections.removeAll()
            listener?.close()
            listener = nil
        }
    }
}

// MARK: - TCP Connection

/// A persistent TCP connection to a single remote node.
/// Uses POSIX sockets with blocking I/O on a serial queue.
final class TCPConnection: @unchecked Sendable {
    fileprivate let socket: Int32
    fileprivate let queue = DispatchQueue(label: "com.novamlx.tensor-conn", qos: .userInteractive)
    let nodeId: String

    init(to endpoint: NodeEndpoint) throws {
        self.nodeId = endpoint.nodeId
        self.socket = try Self.connectSocket(host: endpoint.host, port: endpoint.port)
    }

    /// Called by TCPListener when accepting an incoming connection.
    init(acceptedSocket: Int32, nodeId: String) {
        self.nodeId = nodeId
        self.socket = acceptedSocket
    }

    func send(_ array: MLXArray) throws {
        let wireData = WireFormat.encode(array: array)
        try queue.sync {
            try Self.sendAll(socket: self.socket, data: wireData)
        }
    }

    func recv() throws -> MLXArray {
        try queue.sync {
            // Read header
            let headerData = try Self.recvExact(socket: self.socket, count: WireFormat.headerSize)
            let m = headerData.readUInt32(at: 0)
            guard m == WireFormat.magic else { throw TransportError.invalidMagic }

            let ndim = Int(headerData.readUInt32(at: 8))
            let _ = headerData.readUInt32(at: 12) // dtypeRaw — read again with full data
            let nbytes = Int(headerData.readUInt64(at: 16))

            // Read shape
            let shapeSize = ndim * 8
            let shapeData = shapeSize > 0 ? try Self.recvExact(socket: self.socket, count: shapeSize) : Data()

            // Read tensor payload
            let payloadData = try Self.recvExact(socket: self.socket, count: nbytes)

            // Reassemble full message for decode
            var full = Data()
            full.append(headerData)
            full.append(shapeData)
            full.append(payloadData)
            return try WireFormat.decode(full)
        }
    }

    func close() {
        Darwin.close(socket)
    }

    // MARK: - Shard Service Helpers (used by WorkerShardService + RemoteShardPolicy)

    /// Send raw Data bytes over this connection.
    func sendData(_ data: Data) throws {
        try queue.sync {
            try Self.sendAll(socket: self.socket, data: data)
        }
    }

    /// Receive exactly `count` raw bytes.
    func recvData(count: Int) throws -> Data {
        try queue.sync {
            try Self.recvExact(socket: self.socket, count: count)
        }
    }

    /// Send an MLXArray as a wire-format tensor.
    /// Optimized: write header+shape first, then stream tensor bytes directly.
    func sendTensor(_ array: MLXArray) throws {
        try queue.sync {
            let arrayData = array.asData(access: .copy)
            let shape = arrayData.shape
            let dtype = arrayData.dType
            let ndim = shape.count
            let nbytes = arrayData.data.count

            // Build header + shape (small, ~48 bytes for 3D tensor)
            var header = Data(capacity: WireFormat.headerSize + ndim * 8)
            header.append(contentsOf: withUnsafeBytes(of: WireFormat.magic.bigEndian) { Array($0) })
            header.append(contentsOf: withUnsafeBytes(of: WireFormat.MessageType.tensor.rawValue.bigEndian) { Array($0) })
            header.append(contentsOf: withUnsafeBytes(of: UInt32(ndim).bigEndian) { Array($0) })
            header.append(contentsOf: withUnsafeBytes(of: dtype.cmlxDtype.rawValue.bigEndian) { Array($0) })
            header.append(contentsOf: withUnsafeBytes(of: UInt64(nbytes).bigEndian) { Array($0) })
            header.append(contentsOf: withUnsafeBytes(of: UInt32(0).bigEndian) { Array($0) })
            header.append(contentsOf: withUnsafeBytes(of: UInt32(0).bigEndian) { Array($0) })
            for dim in shape {
                var dim64 = Int64(dim).bigEndian
                header.append(contentsOf: withUnsafeBytes(of: &dim64) { Array($0) })
            }

            // Send header + shape
            try Self.sendAll(socket: self.socket, data: header)
            // Send tensor bytes directly (avoids copying into larger buffer)
            try Self.sendAll(socket: self.socket, data: arrayData.data)
        }
    }

    /// Receive an MLXArray tensor from the wire format.
    /// Optimized: single allocation, recv directly into buffer, zero-copy shape parse.
    func recvTensor() throws -> MLXArray {
        try queue.sync {
            // Read header (32 bytes) directly into pre-allocated buffer
            var headerBuf = Data(count: WireFormat.headerSize)
            try headerBuf.withUnsafeMutableBytes { ptr in
                var remaining = WireFormat.headerSize
                var offset = 0
                while remaining > 0 {
                    let n = Darwin.recv(self.socket, ptr.baseAddress! + offset, remaining, 0)
                    if n <= 0 { throw TransportError.recvFailed("recv header failed") }
                    remaining -= n
                    offset += n
                }
            }

            let m = headerBuf.readUInt32(at: 0)
            guard m == WireFormat.magic else { throw TransportError.invalidMagic }
            let msgType = headerBuf.readUInt32(at: 4)
            guard msgType == WireFormat.MessageType.tensor.rawValue else {
                throw TransportError.invalidHeader("unexpected msgType: \(msgType)")
            }
            let ndim = Int(headerBuf.readUInt32(at: 8))
            let dtypeRaw = headerBuf.readUInt32(at: 12)
            let nbytes = Int(headerBuf.readUInt64(at: 16))
            guard let dtype = DTypeFromRaw(dtypeRaw) else {
                throw TransportError.invalidDType(dtypeRaw)
            }

            // Read shape + payload in one allocation
            let payloadSize = ndim * 8 + nbytes
            var payload = Data(count: payloadSize)
            try payload.withUnsafeMutableBytes { ptr in
                var remaining = payloadSize
                var offset = 0
                while remaining > 0 {
                    let chunkSize = min(remaining, 1 << 20)
                    let n = Darwin.recv(self.socket, ptr.baseAddress! + offset, chunkSize, 0)
                    if n <= 0 { throw TransportError.recvFailed("recv payload failed") }
                    remaining -= n
                    offset += n
                }
            }

            // Parse shape directly from payload (zero-copy via bounds-checked access)
            var shape = [Int]()
            shape.reserveCapacity(ndim)
            for i in 0..<ndim {
                let off = i * 8
                var dim64: Int64 = 0
                withUnsafeMutableBytes(of: &dim64) { ptr in
                    payload.copyBytes(to: ptr, from: off..<(off + 8))
                }
                shape.append(Int(Int64(bigEndian: dim64)))
            }

            // Extract tensor data — use subdataWithRange to avoid full copy
            let tensorStart = ndim * 8
            let tensorData = payload.subdata(in: tensorStart..<(tensorStart + nbytes))
            return MLXArray(tensorData, shape, dtype: dtype)
        }
    }

    // MARK: - Socket Helpers

    /// Resolve hostname to IPv4 address. Handles mDNS .local names that getaddrinfo may time out on.
    private static func resolveHost(_ host: String) -> String? {
        // Fast path: already an IP address
        if host.split(separator: ".").compactMap({ UInt8($0) }).count == 4 {
            return host
        }
        // Try getaddrinfo first (works for regular DNS)
        var hints = addrinfo()
        hints.ai_family = AF_INET
        var resolved: UnsafeMutablePointer<addrinfo>?
        if getaddrinfo(host, nil, &hints, &resolved) == 0, let ai = resolved {
            defer { freeaddrinfo(ai) }
            var hostBuf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(ai.pointee.ai_addr, ai.pointee.ai_addrlen, &hostBuf, socklen_t(hostBuf.count), nil, 0, NI_NUMERICHOST) == 0 {
                return CString( hostBuf)
            }
        }
        // Fallback: use system hostname command for mDNS .local resolution
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["bash", "-c", "getent hosts \(host) 2>/dev/null | head -1 | cut -d' ' -f1"]
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = FileHandle.nullDevice
        guard (try? proc.run()) != nil else { return nil }
        proc.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let ip = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return ip.contains(".") ? ip : nil
    }

    private static func connectSocket(host: String, port: UInt16) throws -> Int32 {
        // Resolve hostname to IP before connecting (handles .local mDNS names)
        let resolvedHost = resolveHost(host) ?? host
        let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else {
            throw TransportError.connectionFailed("socket() failed: \(CString( strerror(errno)))")
        }

        // TCP_NODELAY — disable Nagle's algorithm for minimum latency
        var flag: Int32 = 1
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, socklen_t(MemoryLayout.size(ofValue: flag)))

        // Large socket buffers for prefill activations (up to 16MB+)
        var bufSize: Int32 = 4 * 1024 * 1024
        setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &bufSize, socklen_t(MemoryLayout.size(ofValue: bufSize)))
        setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout.size(ofValue: bufSize)))

        // Use resolved IP (or original host if resolution failed)
        let targetHost = resolvedHost
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = inet_addr(targetHost)

        guard addr.sin_addr.s_addr != INADDR_NONE else {
            Darwin.close(sock)
            throw TransportError.connectionFailed("inet_addr failed for \(targetHost) (original: \(host))")
        }

        let connectResult = withUnsafePointer(to: addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(sock, sockaddrPtr, socklen_t(MemoryLayout.size(ofValue: addr)))
            }
        }

        guard connectResult == 0 else {
            let err = CString( strerror(errno))
            Darwin.close(sock)
            throw TransportError.connectionFailed("connect() to \(targetHost):\(port) failed: \(err)")
        }

        return sock
    }

    /// Send all bytes — handles partial writes.
    static func sendAll(socket: Int32, data: Data) throws {
        try data.withUnsafeBytes { ptr in
            var sent = 0
            let total = data.count
            while sent < total {
                let n = Darwin.write(socket, ptr.baseAddress! + sent, total - sent)
                if n < 0 {
                    throw TransportError.sendFailed("write() error: \(CString( strerror(errno)))")
                }
                if n == 0 {
                    throw TransportError.sendFailed("connection closed")
                }
                sent += n
            }
        }
    }

    /// Receive exactly `count` bytes — handles partial reads.
    /// Optimized: recv directly into pre-allocated Data, no [UInt8] intermediate.
    static func recvExact(socket: Int32, count: Int) throws -> Data {
        var buf = Data(count: count)
        let bytesRead = try buf.withUnsafeMutableBytes { ptr -> Int in
            var totalRead = 0
            while totalRead < count {
                let chunkSize = min(count - totalRead, 1 << 20)
                let n = Darwin.recv(socket, ptr.baseAddress! + totalRead, chunkSize, 0)
                if n < 0 {
                    throw TransportError.recvFailed("recv() error: \(CString( strerror(errno)))")
                }
                if n == 0 {
                    throw TransportError.recvFailed("connection closed (expected \(count - totalRead) more bytes)")
                }
                totalRead += n
            }
            return totalRead
        }
        // Trim if we over-allocated (shouldn't happen but safety)
        if bytesRead < count {
            buf = buf.subdata(in: 0..<bytesRead)
        }
        return buf
    }
}

// MARK: - TCP Listener

/// Listens for incoming TCP connections and hands them to a callback.
final class TCPListener: @unchecked Sendable {
    private let socket: Int32
    private let onAccept: (String, TCPConnection) -> Void
    private var acceptTask: Task<Void, Never>?

    init(port: UInt16, onAccept: @escaping (String, TCPConnection) -> Void) throws {
        self.onAccept = onAccept

        let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else {
            throw TransportError.connectionFailed("socket() failed: \(CString( strerror(errno)))")
        }

        // Allow address reuse (restart without TIME_WAIT issues)
        var reuse: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &reuse, socklen_t(MemoryLayout.size(ofValue: reuse)))

        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(port).bigEndian
        addr.sin_addr.s_addr = INADDR_ANY

        let bindResult = withUnsafePointer(to: addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.bind(sock, sockaddrPtr, socklen_t(MemoryLayout.size(ofValue: addr)))
            }
        }
        guard bindResult == 0 else {
            let err = CString( strerror(errno))
            Darwin.close(sock)
            throw TransportError.connectionFailed("bind() failed: \(err)")
        }

        guard Darwin.listen(sock, 16) == 0 else {
            let err = CString( strerror(errno))
            Darwin.close(sock)
            throw TransportError.connectionFailed("listen() failed: \(err)")
        }

        self.socket = sock

        // Accept loop on background thread
        acceptTask = Task.detached { [weak self] in
            self?.acceptLoop()
        }
    }

    private func acceptLoop() {
        while !Task.isCancelled {
            var clientAddr = sockaddr_in()
            var clientAddrLen = socklen_t(MemoryLayout.size(ofValue: clientAddr))
            let clientSock = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                    Darwin.accept(socket, sockaddrPtr, &clientAddrLen)
                }
            }

            guard clientSock >= 0 else { continue }

            // Set TCP_NODELAY + buffers on accepted socket
            var flag: Int32 = 1
            setsockopt(clientSock, IPPROTO_TCP, TCP_NODELAY, &flag, socklen_t(MemoryLayout.size(ofValue: flag)))
            var bufSize: Int32 = 4 * 1024 * 1024
            setsockopt(clientSock, SOL_SOCKET, SO_SNDBUF, &bufSize, socklen_t(MemoryLayout.size(ofValue: bufSize)))
            setsockopt(clientSock, SOL_SOCKET, SO_RCVBUF, &bufSize, socklen_t(MemoryLayout.size(ofValue: bufSize)))

            // Convert client address to string for nodeId
            var addrBuf = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
            inet_ntop(AF_INET, &clientAddr.sin_addr, &addrBuf, socklen_t(INET_ADDRSTRLEN))
            let clientHost = CString( addrBuf)
            let nodeId = "\(clientHost):\(Int(clientAddr.sin_port.byteSwapped))"

            let conn = TCPConnection(acceptedSocket: clientSock, nodeId: nodeId)
            onAccept(nodeId, conn)
        }
    }

    func close() {
        acceptTask?.cancel()
        Darwin.close(socket)
    }
}

// MARK: - Data Helpers

extension Data {
    func readUInt32(at offset: Int) -> UInt32 {
        guard offset + 4 <= count else { return 0 }
        return subdata(in: offset..<(offset + 4)).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
    }

    func readUInt64(at offset: Int) -> UInt64 {
        guard offset + 8 <= count else { return 0 }
        return subdata(in: offset..<(offset + 8)).withUnsafeBytes { $0.load(as: UInt64.self).bigEndian }
    }
}

// MARK: - DType Wire Encoding

/// Encode DType as UInt32 for wire format.
func DTypeToRaw(_ dtype: DType) -> UInt32 {
    dtype.cmlxDtype.rawValue
}

/// Decode DType from UInt32 wire value.
func DTypeFromRaw(_ raw: UInt32) -> DType? {
    let allCases: [DType] = [.bool, .uint8, .uint16, .uint32, .uint64,
                             .int8, .int16, .int32, .int64,
                             .float16, .float32, .bfloat16, .complex64, .float64]
    return allCases.first { $0.cmlxDtype.rawValue == raw }
}
