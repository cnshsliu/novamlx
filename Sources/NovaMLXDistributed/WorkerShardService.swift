import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine

// MARK: - Shard Service Protocol

/// Messages in the shard service protocol.
enum ShardServiceMessage: UInt32 {
    case assignShard = 1     // Coordinator → Worker: here's your shard assignment
    case bindWeights = 2     // Coordinator → Worker: load weights and create KV caches
    case compute = 3         // Coordinator → Worker: run forward pass on this input
    case computeResult = 4   // Worker → Coordinator: here's the output tensor
    case releaseWeights = 5  // Coordinator → Worker: release weights
    case shutdown = 6        // Coordinator → Worker: stop service
    case error = 7           // Worker → Coordinator: something went wrong
    case computeAndSample = 8 // Coordinator → Worker: forward + argmax, return 4-byte token ID
    case sampledToken = 9    // Worker → Coordinator: here's the sampled token ID (4 bytes)
    case speculativeVerify = 10 // Coord → Worker: batch forward + per-position argmax, return K+1 token IDs
    case verifiedTokens = 11    // Worker → Coord: array of verified token IDs (K+1 × 4 bytes)
    case rollbackCache = 12     // Coord → Worker: trim KV cache to position N
    case cacheRolledBack = 13   // Worker → Coord: ack
    case initTransport = 14     // Coord → Worker: initialize JACCL/Ring transport (rank, backend)
    case transportReady = 15    // Worker → Coord: transport initialized and ready
}

// MARK: - Shard Service Wire Format

/// Wire format for shard service messages (control plane, not tensor data).
///
/// Tensors are sent separately via TensorTransport after the control message.
enum ShardWireFormat {
    static let headerSize = 16

    // Header layout:
    // [0-3]   magic: UInt32 (0x4E4F5641)
    // [4-7]   msgType: UInt32 (ShardServiceMessage raw value)
    // [8-11]  payloadSize: UInt32 (JSON payload bytes, 0 if no payload)
    // [12-15] flags: UInt32 (0x01 = tensor follows, 0x00 = no tensor)

    static func encode(msgType: ShardServiceMessage, payload: Data = Data(), hasTensor: Bool = false) -> Data {
        var buf = Data(capacity: headerSize + payload.count)
        buf.append(contentsOf: withUnsafeBytes(of: UInt32(0x4E4F5641).bigEndian) { Data($0) })
        buf.append(contentsOf: withUnsafeBytes(of: msgType.rawValue.bigEndian) { Data($0) })
        buf.append(contentsOf: withUnsafeBytes(of: UInt32(payload.count).bigEndian) { Data($0) })
        buf.append(contentsOf: withUnsafeBytes(of: UInt32(hasTensor ? 1 : 0).bigEndian) { Data($0) })
        buf.append(payload)
        return buf
    }

    static func decodeHeader(_ data: Data) -> (msgType: ShardServiceMessage, payloadSize: Int, hasTensor: Bool)? {
        guard data.count >= headerSize else { return nil }
        let magic = data.readUInt32(at: 0)
        guard magic == 0x4E4F5641 else { return nil }
        guard let msgType = ShardServiceMessage(rawValue: data.readUInt32(at: 4)) else { return nil }
        let payloadSize = Int(data.readUInt32(at: 8))
        let flags = data.readUInt32(at: 12)
        return (msgType, payloadSize, (flags & 1) != 0)
    }
}

// MARK: - Shard Assignment Payload

/// Extended shard assignment with model ID so the Worker knows which model to load.
struct ShardAssignmentPayload: Codable {
    let assignment: ShardAssignment
    let modelId: String
    let modelPath: String?
    let isFirst: Bool
    let isLast: Bool
}

// MARK: - Shard Service Errors

public enum ShardServiceError: Error, CustomStringConvertible {
    case notStarted
    case alreadyRunning
    case connectionFailed(String)
    case invalidMessage(String)
    case computeFailed(String)
    case coordinatorNotConnected
    case modelNotAvailable(String)

    public var description: String {
        switch self {
        case .notStarted: "ShardServiceError.notStarted"
        case .alreadyRunning: "ShardServiceError.alreadyRunning"
        case .connectionFailed(let msg): "ShardServiceError.connectionFailed: \(msg)"
        case .invalidMessage(let msg): "ShardServiceError.invalidMessage: \(msg)"
        case .computeFailed(let msg): "ShardServiceError.computeFailed: \(msg)"
        case .coordinatorNotConnected: "ShardServiceError.coordinatorNotConnected"
        case .modelNotAvailable(let msg): "ShardServiceError.modelNotAvailable: \(msg)"
        }
    }
}

// MARK: - WorkerShardService

/// Runs on worker nodes. Listens for shard assignments from the coordinator,
/// receives input activations via TensorTransport, runs forward computation,
/// and sends output activations back.
///
/// Architecture:
/// - Control channel: TCP connection for shard assignment + commands
/// - Data channel: TensorTransport for sending/receiving MLXArrays
///
/// Lifecycle:
/// 1. Coordinator connects to worker's ShardService
/// 2. Coordinator sends assignShard with ShardAssignment + modelId
/// 3. Coordinator sends bindWeights — worker loads model, creates SlicedForwardPolicy
/// 4. For each step, coordinator sends compute + input tensor
/// 5. Worker runs SlicedForwardPolicy.compute(), returns output tensor
/// 6. On completion, coordinator sends releaseWeights or shutdown
public final class WorkerShardService: @unchecked Sendable {

    /// Shared singleton.
    public static let shared = WorkerShardService()

    private let port: UInt16
    private let transport: TCPTensorTransport
    private var listener: TCPListener?
    private var coordinatorConn: TCPConnection?
    private let lock = NSLock()
    private var isRunning = false

    /// Current compute policy for the assigned shard.
    private var policy: ComputePolicy?

    /// The model ID assigned by the coordinator.
    private var assignedModelId: String?

    /// The shard assignment received from the coordinator.
    private var currentAssignment: ShardAssignment?

    /// Whether this shard is the first/last in the pipeline.
    private var isShardFirst: Bool = false
    private var isShardLast: Bool = false

    /// The model path on local disk (sent by Coordinator).
    private var currentModelPath: String?

    /// Reference to MLXEngine for loading models.
    private weak var engine: MLXEngine?

    private init(port: UInt16 = 7010) {
        self.port = port
        self.transport = TCPTensorTransport()
    }

    /// Set the engine reference (called during app startup on Worker nodes).
    public func setEngine(_ engine: MLXEngine) {
        lock.withLock { self.engine = engine }
    }

    /// Start listening for coordinator connections.
    public func start(transportPort: UInt16 = 7011) async throws {
        lock.withLock {
            guard !isRunning else { return }
            isRunning = true
        }

        try await transport.startListening(port: transportPort)

        listener = try TCPListener(port: port) { [weak self] nodeId, conn in
            self?.lock.withLock {
                self?.coordinatorConn = conn
            }
            NovaMLXLog.info("[WorkerShardService] Coordinator connected: \(nodeId)")
        }

        NovaMLXLog.info("[WorkerShardService] Listening on control port \(port), transport port \(transportPort)")
    }

    /// Main event loop — waits for coordinator connection, then processes messages.
    /// Loops to accept new coordinator connections after the previous one disconnects.
    /// Returns only on shutdown or cancellation.
    public func run() async throws {
        while !Task.isCancelled {
            // Wait for coordinator to connect (polling with backoff)
            lock.withLock { coordinatorConn = nil }
            var conn: TCPConnection? = nil
            while !Task.isCancelled {
                conn = lock.withLock { coordinatorConn }
                if conn != nil { break }
                try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            }
            guard let conn = conn else { return }

            NovaMLXLog.info("[WorkerShardService] Starting message loop for coordinator")

            do {
                while !Task.isCancelled {
                    let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
                    guard let header = ShardWireFormat.decodeHeader(headerData) else {
                        throw ShardServiceError.invalidMessage("bad header")
                    }

                    var payload = Data()
                    if header.payloadSize > 0 {
                        payload = try conn.recvData(count: header.payloadSize)
                    }

                    switch header.msgType {
                    case .assignShard:
                        try handleAssignShard(payload: payload)

                    case .bindWeights:
                        do {
                            try await handleBindWeights()
                            let ack = ShardWireFormat.encode(msgType: .computeResult)
                            try conn.sendData(ack)
                            NovaMLXLog.info("[WorkerShardService] Sent bindWeights ack to coordinator")
                        } catch {
                            NovaMLXLog.error("[WorkerShardService] bindWeights failed: \(error)")
                            let errorPayload = error.localizedDescription.data(using: .utf8) ?? Data()
                            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
                            try? conn.sendData(msg)
                        }

                    case .compute:
                        try await handleCompute(conn: conn, hasTensor: header.hasTensor, payload: payload)

                    case .computeAndSample:
                        try await handleComputeAndSample(conn: conn, hasTensor: header.hasTensor)

                    case .releaseWeights:
                        handleReleaseWeights()

                    case .shutdown:
                        NovaMLXLog.info("[WorkerShardService] Shutdown received")
                        return

                    case .speculativeVerify:
                        try await handleSpeculativeVerify(conn: conn, hasTensor: header.hasTensor)

                    case .rollbackCache:
                        try await handleRollbackCache(conn: conn, payload: payload)

                    case .initTransport:
                        try await handleInitTransport(conn: conn, payload: payload)

                    case .computeResult, .error, .sampledToken, .verifiedTokens, .cacheRolledBack, .transportReady:
                        NovaMLXLog.warning("[WorkerShardService] Unexpected message from coordinator: \(header.msgType)")
                    }
                }
            } catch {
                // Connection closed or error — loop back to wait for new coordinator
                NovaMLXLog.info("[WorkerShardService] Connection ended: \(error), waiting for new coordinator...")
            }
        }
    }

    public func stop() {
        lock.withLock {
            isRunning = false
            coordinatorConn?.close()
            coordinatorConn = nil
            listener?.close()
            listener = nil
        }
        transport.shutdown()
    }

    // MARK: - Message Handlers

    private func handleAssignShard(payload: Data) throws {
        guard !payload.isEmpty else {
            throw ShardServiceError.invalidMessage("assignShard requires payload")
        }
        let decoded = try JSONDecoder().decode(ShardAssignmentPayload.self, from: payload)
        currentAssignment = decoded.assignment
        assignedModelId = decoded.modelId
        currentModelPath = decoded.modelPath
        isShardFirst = decoded.isFirst
        isShardLast = decoded.isLast
        NovaMLXLog.info("[WorkerShardService] Assigned shard: layers \(decoded.assignment.startLayer)..<\(decoded.assignment.endLayer), model=\(decoded.modelId)")
    }

    private func handleBindWeights() async throws {
        guard let assignment = currentAssignment,
              let modelId = assignedModelId else {
            NovaMLXLog.warning("[WorkerShardService] bindWeights without prior assignShard")
            return
        }

        let engineRef = lock.withLock { engine }
        guard let engine = engineRef else {
            throw ShardServiceError.modelNotAvailable("No engine reference")
        }

        // Load model directly into main engine using Worker's local model path
        if engine.getContainer(for: modelId) == nil {
            NovaMLXLog.info("[WorkerShardService] Loading model \(modelId) into engine...")
            // Always resolve from Worker's local models dir — Coordinator's path is remote
            let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
            let config = ModelConfig(identifier: ModelIdentifier(id: modelId, family: .qwen))
            _ = try await engine.loadModel(from: modelDir, config: config, skipMemoryGate: true)
            NovaMLXLog.info("[WorkerShardService] Model \(modelId) loaded into engine")
        }

        let isFirst = assignment.startLayer == 0

        let slicedPolicy = SlicedForwardPolicy(
            assignment: assignment,
            engine: engine,
            modelId: modelId,
            isFirst: isFirst,
            isLast: isShardLast
        )

        try await slicedPolicy.bindWeights()
        lock.withLock { self.policy = slicedPolicy }
        NovaMLXLog.info("[WorkerShardService] Weights bound for layers \(assignment.startLayer)..<\(assignment.endLayer)")
    }

    private func handleCompute(conn: TCPConnection, hasTensor: Bool, payload: Data) async throws {
        guard let policy = lock.withLock({ policy }) else {
            let errorPayload = "No policy assigned".data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
            return
        }

        let inputTensor: MLXArray
        let useRing = RingTransportManager.shared.isReady

        // Detect prefill (seq_len > 1) and reset caches for new conversation.
        // Without this, the worker's caches carry over from previous requests,
        // causing the model to continue the old conversation.
        let needsCacheReset: Bool

        if useRing && !hasTensor {
            // Ring/JACCL transport: coordinator sent shape in TCP payload, tensor via Ring
            guard payload.count >= 8 else {
                throw ShardServiceError.invalidMessage("Ring compute requires shape payload")
            }
            let ndim = (payload.count - 4) / 4  // Last 4 bytes = dtype raw value
            var shape = [Int]()
            for i in 0..<ndim {
                let dim = payload[i*4..<(i*4+4)].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
                shape.append(Int(dim))
            }
            // dtype encoded as last 4 bytes (DType raw value)
            let dtypeRaw = payload[payload.count-4..<payload.count].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
            let dtype = DTypeFromRaw(dtypeRaw) ?? .float32
            inputTensor = RingTransportManager.shared.recv(shape: shape, dtype: dtype, from: 0)
            needsCacheReset = shape.count >= 2 && shape[1] > 1
        } else {
            guard hasTensor else {
                throw ShardServiceError.invalidMessage("compute message requires tensor")
            }
            inputTensor = try conn.recvTensor()
            needsCacheReset = inputTensor.ndim >= 2 && inputTensor.dim(1) > 1
        }

        if needsCacheReset {
            if let slicedPolicy = policy as? SlicedForwardPolicy {
                try? await slicedPolicy.resetCaches()
            }
        }

        do {
            let output = try await policy.compute(input: inputTensor)

            if useRing {
                // Send output shape + dtype via TCP payload (so coord can recv with correct params)
                let shape = output.shape
                var shapePayload = Data(capacity: shape.count * 4 + 4)
                for dim in shape {
                    shapePayload.append(contentsOf: withUnsafeBytes(of: UInt32(dim).bigEndian) { Data($0) })
                }
                shapePayload.append(contentsOf: withUnsafeBytes(of: DTypeToRaw(output.dtype).bigEndian) { Data($0) })
                let resultHeader = ShardWireFormat.encode(msgType: .computeResult, payload: shapePayload)
                try conn.sendData(resultHeader)
                // Send tensor via Ring/JACCL transport
                eval(output)
                _ = RingTransportManager.shared.send(output, to: 0)
            } else {
                let resultHeader = ShardWireFormat.encode(msgType: .computeResult, hasTensor: true)
                try conn.sendData(resultHeader)
                let outputToSend = output.dtype != .bfloat16 ? output.asType(.bfloat16) : output
                try conn.sendTensor(outputToSend)
            }
        } catch {
            let errorPayload = error.localizedDescription.data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
        }
    }

    private func handleComputeAndSample(conn: TCPConnection, hasTensor: Bool) async throws {
        guard let policy = lock.withLock({ policy }) else {
            let errorPayload = "No policy assigned".data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
            return
        }

        guard hasTensor else {
            throw ShardServiceError.invalidMessage("computeAndSample requires input tensor")
        }

        let inputTensor = try conn.recvTensor()

        do {
            let logits = try await policy.compute(input: inputTensor)
            let tokenId = argmaxToken(logits)
            let resultHeader = ShardWireFormat.encode(msgType: .sampledToken, payload: withUnsafeBytes(of: Int32(tokenId).bigEndian) { Data($0) })
            try conn.sendData(resultHeader)
        } catch {
            let errorPayload = error.localizedDescription.data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
        }
    }

    private func handleReleaseWeights() {
        lock.withLock {
            policy?.releaseWeights()
            policy = nil
        }
        NovaMLXLog.info("[WorkerShardService] Weights released")
    }

    /// Handle speculativeVerify: receive [1, K+1, hidden] tensor, run forward + per-position argmax,
    /// return K+1 token IDs as verifiedTokens payload.
    private func handleSpeculativeVerify(conn: TCPConnection, hasTensor: Bool) async throws {
        guard let policy = lock.withLock({ policy }) else {
            let errorPayload = "No policy assigned".data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
            return
        }

        guard hasTensor else {
            throw ShardServiceError.invalidMessage("speculativeVerify requires input tensor")
        }

        let inputTensor = try conn.recvTensor()

        do {
            let logits = try await policy.compute(input: inputTensor)
            // logits shape: [seq_len, vocab_size] (squeezed from [1, K+1, vocab])
            // or [K+1, vocab_size]
            let seqLen = logits.ndim >= 2 ? logits.dim(logits.ndim - 2) : 1

            // Argmax each position → K+1 token IDs
            var tokenIds: [Int32] = []
            tokenIds.reserveCapacity(seqLen)
            for pos in 0..<seqLen {
                let posLogits: MLXArray
                if logits.ndim == 3 {
                    posLogits = logits[0..., pos, 0...]
                } else if logits.ndim == 2 {
                    posLogits = logits[pos, 0...]
                } else {
                    posLogits = logits
                }
                let id = MLX.argMax(posLogits.flattened()).item(Int.self)
                tokenIds.append(Int32(id))
            }

            // Send K+1 × 4 bytes as verifiedTokens payload
            var payload = Data(capacity: tokenIds.count * 4)
            for id in tokenIds {
                payload.append(contentsOf: withUnsafeBytes(of: id.bigEndian) { Data($0) })
            }
            let resultHeader = ShardWireFormat.encode(msgType: .verifiedTokens, payload: payload)
            try conn.sendData(resultHeader)
        } catch {
            let errorPayload = error.localizedDescription.data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
        }
    }

    /// Handle rollbackCache: trim KV cache to a specific position.
    private func handleRollbackCache(conn: TCPConnection, payload: Data) async throws {
        guard let policy = lock.withLock({ self.policy }) else {
            let errorPayload = "No policy assigned".data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
            return
        }

        // Parse position from payload
        guard !payload.isEmpty,
              let json = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
              let position = json["position"] as? Int else {
            let errorPayload = "Invalid rollbackCache payload".data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
            return
        }

        do {
            try await policy.rollbackCache(position: position)
            let ack = ShardWireFormat.encode(msgType: .cacheRolledBack)
            try conn.sendData(ack)
        } catch {
            let errorPayload = error.localizedDescription.data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
        }
    }

    /// Handle initTransport: initialize Ring/JACCL distributed transport on worker.
    private func handleInitTransport(conn: TCPConnection, payload: Data) async throws {
        guard !payload.isEmpty,
              let json = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
              let rank = json["rank"] as? Int else {
            let errorPayload = "Invalid initTransport payload".data(using: .utf8) ?? Data()
            let msg = ShardWireFormat.encode(msgType: .error, payload: errorPayload)
            try conn.sendData(msg)
            return
        }

        let backend = json["backend"] as? String ?? "ring"
        let hostfileJSON = json["hostfileJSON"] as? String
        NovaMLXLog.info("[WorkerShardService] Initializing transport: backend=\(backend), rank=\(rank)")

        // Send ack BEFORE transport init (which blocks until coord also inits).
        // Coordinator reads this ack, then inits on its side — both sides meet in init.
        let ack = ShardWireFormat.encode(msgType: .transportReady)
        try conn.sendData(ack)

        // Now init transport (blocks until coord also inits)
        let group: DistributedGroup
        if let hostfile = hostfileJSON {
            group = RingTransportManager.shared.initializeFromHostfileJSON(hostfile, rank: rank)
        } else {
            group = RingTransportManager.shared.initializeJACCL(rank: rank)
        }

        if group.isValid {
            NovaMLXLog.info("[WorkerShardService] Transport ready: rank=\(group.rank), size=\(group.size)")
        } else {
            NovaMLXLog.warning("[WorkerShardService] Transport init failed (backend=\(backend)), using TCP fallback")
        }
    }
}
