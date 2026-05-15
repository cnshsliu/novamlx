import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils

// MARK: - RemoteShardPolicy

/// A ``ComputePolicy`` that delegates computation to a remote worker node.
///
/// Instead of running forward passes locally, this policy:
/// 1. Sends input activations to the worker via TCP
/// 2. Worker runs the forward pass through its assigned layers
/// 3. Receives output activations back via TCP
///
/// Used by the coordinator for shards assigned to remote workers.
public final class RemoteShardPolicy: ComputePolicy, @unchecked Sendable {

    public let assignment: ShardAssignment
    public private(set) var isReady: Bool = false

    private let workerEndpoint: NodeEndpoint
    private let modelId: String
    private let modelPath: String?
    private let isFirst: Bool
    private let isLast: Bool
    private var connection: TCPConnection?
    private let lock = NSLock()

    /// Whether to use MLX Ring/JACCL transport for tensor data (falls back to TCP).
    private var useRingTransport: Bool

    /// Worker rank for Ring transport (always 1 for 2-node setup).
    private let workerRank: Int = 1

    public init(assignment: ShardAssignment, workerEndpoint: NodeEndpoint, modelId: String, modelPath: String? = nil, isFirst: Bool = false, isLast: Bool = false, useRingTransport: Bool = false) {
        self.assignment = assignment
        self.isFirst = isFirst
        self.isLast = isLast
        self.workerEndpoint = workerEndpoint
        self.modelId = modelId
        self.modelPath = modelPath
        self.useRingTransport = useRingTransport && RingTransportManager.shared.isReady
    }

    /// Enable Ring/JACCL transport after the distributed group is initialized.
    /// Called by ClusterModelManager after both sides have initialized transport.
    func enableRingTransport() {
        guard RingTransportManager.shared.isReady else {
            NovaMLXLog.warning("[RemoteShardPolicy] Cannot enable Ring transport: group not ready")
            return
        }
        useRingTransport = true
        NovaMLXLog.info("[RemoteShardPolicy] Ring/JACCL transport ENABLED for data plane")
    }

    /// Quantize activation to bfloat16 for transport (~2x compression).
    private func quantizeForTransport(_ array: MLXArray) -> MLXArray {
        if array.dtype == .bfloat16 { return array }
        return array.asType(.bfloat16)
    }

    /// Dequantize bfloat16 back to float32 for head computation.
    private func dequantizeFromTransport(_ array: MLXArray) -> MLXArray {
        if array.dtype == .float32 { return array }
        return array.asType(.float32)
    }

    public func bindWeights() async throws {
        let conn = try TCPConnection(to: workerEndpoint)
        lock.withLock {
            self.connection = conn
        }

        // Send assignShard message with modelId and modelPath
        let payload = try JSONEncoder().encode(ShardAssignmentPayload(
            assignment: assignment,
            modelId: modelId,
            modelPath: modelPath,
            isFirst: isFirst,
            isLast: isLast
        ))
        let msg = ShardWireFormat.encode(msgType: .assignShard, payload: payload)
        try conn.sendData(msg)

        // Send bindWeights message — Worker will load model and create SlicedForwardPolicy
        let bindMsg = ShardWireFormat.encode(msgType: .bindWeights)
        try conn.sendData(bindMsg)

        // Wait for Worker ack/nack
        let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
        guard let header = ShardWireFormat.decodeHeader(headerData) else {
            throw ShardServiceError.invalidMessage("bad bindWeights response header")
        }

        if header.msgType == .error {
            var errorMsg = "bindWeights failed on worker"
            if header.payloadSize > 0 {
                let errorData = try conn.recvData(count: header.payloadSize)
                errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
            }
            throw ShardServiceError.computeFailed(errorMsg)
        }

        isReady = true
        NovaMLXLog.info("[RemoteShardPolicy] Connected to worker \(workerEndpoint.host):\(workerEndpoint.port), shard layers \(assignment.startLayer)..<\(assignment.endLayer)")
    }

    public func compute(input: MLXArray) async throws -> MLXArray {
        guard isReady, let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }

        NovaMLXLog.debug("[RemoteShardPolicy] compute: sending to worker (input shape: \(input.shape), useRing: \(useRingTransport))")

        if useRingTransport {
            // Send shape + dtype in payload so worker can Ring recv with correct params
            var shapePayload = Data(capacity: input.shape.count * 4 + 4)
            for dim in input.shape {
                shapePayload.append(contentsOf: withUnsafeBytes(of: UInt32(dim).bigEndian) { Data($0) })
            }
            shapePayload.append(contentsOf: withUnsafeBytes(of: DTypeToRaw(input.dtype).bigEndian) { Data($0) })
            let computeMsg = ShardWireFormat.encode(msgType: .compute, payload: shapePayload, hasTensor: false)
            try conn.sendData(computeMsg)

            // Send input tensor via Ring transport (data plane)
            eval(input)
            _ = RingTransportManager.shared.send(input, to: workerRank)

            // Receive result tensor via Ring transport
            let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
            guard let header = ShardWireFormat.decodeHeader(headerData) else {
                throw ShardServiceError.invalidMessage("bad response header")
            }
            if header.msgType == .error {
                var errorMsg = "unknown error"
                if header.payloadSize > 0 {
                    let errorData = try conn.recvData(count: header.payloadSize)
                    errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
                }
                throw ShardServiceError.computeFailed(errorMsg)
            }

            // Worker sends result shape + dtype via TCP payload, actual tensor via Ring
            if header.payloadSize > 0 {
                let shapeData = try conn.recvData(count: header.payloadSize)
                let ndim = (shapeData.count - 4) / 4
                var shape = [Int]()
                for i in 0..<ndim {
                    let dim = shapeData[i*4..<(i*4+4)].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
                    shape.append(Int(dim))
                }
                let dtypeRaw = shapeData[shapeData.count-4..<shapeData.count].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
                let dtype = DTypeFromRaw(dtypeRaw) ?? .float32
                let result = RingTransportManager.shared.recv(shape: shape, dtype: dtype, from: workerRank)
                return result
            }
            // Defensive fallback: worker used TCP instead of Ring (payloadSize=0, hasTensor=true)
            if header.hasTensor {
                NovaMLXLog.warning("[RemoteShardPolicy] Worker sent TCP response in Ring mode, falling back to TCP recv")
                useRingTransport = false
                return try conn.recvTensor()
            }
            throw ShardServiceError.invalidMessage("Ring compute: no shape info in response")
        } else {
            // Original TCP transport path (bfloat16 quantized for ~2x compression)
            let computeMsg = ShardWireFormat.encode(msgType: .compute, hasTensor: true)
            try conn.sendData(computeMsg)
            try conn.sendTensor(quantizeForTransport(input))

            let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
            guard let header = ShardWireFormat.decodeHeader(headerData) else {
                throw ShardServiceError.invalidMessage("bad response header")
            }

            switch header.msgType {
            case .computeResult:
                guard header.hasTensor else {
                    throw ShardServiceError.invalidMessage("computeResult missing tensor")
                }
                let raw = try conn.recvTensor()
                return dequantizeFromTransport(raw)

            case .error:
                var errorMsg = "unknown error"
                if header.payloadSize > 0 {
                    let errorData = try conn.recvData(count: header.payloadSize)
                    errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
                }
                throw ShardServiceError.computeFailed(errorMsg)

            default:
                throw ShardServiceError.invalidMessage("unexpected response: \(header.msgType)")
            }
        }
    }

    // MARK: - Remote Sampling (compute + argmax on worker, return token ID)

    /// Send compute + sample request to worker. Worker runs forward pass + argmax
    /// and returns a 4-byte Int32 token ID instead of the full logits tensor.
    ///
    /// This is the #1 TPS optimization: 4 bytes vs 970KB per decode step.
    public func computeAndSample(input: MLXArray) async throws -> Int {
        guard isReady, let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }

        // Send computeAndSample command + input tensor (TCP path only for now)
        let msg = ShardWireFormat.encode(msgType: .computeAndSample, hasTensor: true)
        try conn.sendData(msg)
        try conn.sendTensor(input)

        // Receive 4-byte token ID back
        let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
        guard let header = ShardWireFormat.decodeHeader(headerData) else {
            throw ShardServiceError.invalidMessage("bad computeAndSample response header")
        }

        switch header.msgType {
        case .sampledToken:
            guard header.payloadSize >= 4 else {
                throw ShardServiceError.invalidMessage("sampledToken payload too small")
            }
            let tokenData = try conn.recvData(count: 4)
            let tokenId = tokenData.withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
            return Int(tokenId)

        case .error:
            var errorMsg = "unknown error"
            if header.payloadSize > 0 {
                let errorData = try conn.recvData(count: header.payloadSize)
                errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
            }
            throw ShardServiceError.computeFailed(errorMsg)

        default:
            throw ShardServiceError.invalidMessage("unexpected computeAndSample response: \(header.msgType)")
        }
    }

    // MARK: - Pipeline Support (split send/receive for overlap)

    /// Send compute request without waiting for result.
    /// Used by pipelined prefill to overlap Coordinator compute with Worker compute.
    func sendCompute(input: MLXArray) throws {
        guard isReady, let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }
        if useRingTransport {
            // Send shape + dtype in payload so worker can Ring recv with correct params
            var shapePayload = Data(capacity: input.shape.count * 4 + 4)
            for dim in input.shape {
                shapePayload.append(contentsOf: withUnsafeBytes(of: UInt32(dim).bigEndian) { Data($0) })
            }
            shapePayload.append(contentsOf: withUnsafeBytes(of: DTypeToRaw(input.dtype).bigEndian) { Data($0) })
            let computeMsg = ShardWireFormat.encode(msgType: .compute, payload: shapePayload, hasTensor: false)
            try conn.sendData(computeMsg)
            eval(input)
            _ = RingTransportManager.shared.send(input, to: workerRank)
        } else {
            let computeMsg = ShardWireFormat.encode(msgType: .compute, hasTensor: true)
            try conn.sendData(computeMsg)
            try conn.sendTensor(quantizeForTransport(input))
        }
    }

    /// Receive result from a previous sendCompute call.
    func recvResult() throws -> MLXArray {
        guard let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }
        if useRingTransport {
            // Read control header (worker sends ack/error via TCP, may include shape info)
            let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
            guard let header = ShardWireFormat.decodeHeader(headerData) else {
                throw ShardServiceError.invalidMessage("bad response header")
            }
            if header.msgType == .error {
                var errorMsg = "unknown error"
                if header.payloadSize > 0 {
                    let errorData = try conn.recvData(count: header.payloadSize)
                    errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
                }
                throw ShardServiceError.computeFailed(errorMsg)
            }
            // Worker sends output shape + dtype via TCP payload
            // Then actual tensor via JACCL/Ring
            if header.payloadSize > 0 {
                let shapeData = try conn.recvData(count: header.payloadSize)
                let ndim = (shapeData.count - 4) / 4  // Last 4 bytes = dtype
                var shape = [Int]()
                for i in 0..<ndim {
                    let dim = shapeData[i*4..<(i*4+4)].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
                    shape.append(Int(dim))
                }
                let dtypeRaw = shapeData[shapeData.count-4..<shapeData.count].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
                let dtype = DTypeFromRaw(dtypeRaw) ?? .float32
                let result = RingTransportManager.shared.recv(shape: shape, dtype: dtype, from: workerRank)
                return result
            }
            // Defensive fallback: worker used TCP instead of Ring
            if header.hasTensor {
                NovaMLXLog.warning("[RemoteShardPolicy] Worker sent TCP recvResult in Ring mode, falling back")
                useRingTransport = false
                return try conn.recvTensor()
            }
            throw ShardServiceError.invalidMessage("Ring recvResult: no shape info in header")
        } else {
            let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
            guard let header = ShardWireFormat.decodeHeader(headerData) else {
                throw ShardServiceError.invalidMessage("bad response header")
            }
            switch header.msgType {
            case .computeResult:
                guard header.hasTensor else {
                    throw ShardServiceError.invalidMessage("computeResult missing tensor")
                }
                return dequantizeFromTransport(try conn.recvTensor())
            case .error:
                var errorMsg = "unknown error"
                if header.payloadSize > 0 {
                    let errorData = try conn.recvData(count: header.payloadSize)
                    errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
                }
                throw ShardServiceError.computeFailed(errorMsg)
            default:
                throw ShardServiceError.invalidMessage("unexpected response: \(header.msgType)")
            }
        }
    }

    public func releaseWeights() {
        if let conn = lock.withLock({ connection }) {
            let msg = ShardWireFormat.encode(msgType: .releaseWeights)
            try? conn.sendData(msg)
            conn.close()
        }
        lock.withLock { connection = nil }
        isReady = false
    }

    // MARK: - Transport Init

    /// Send initTransport to worker so it initializes Ring/JACCL distributed transport.
    /// Called after bindWeights. Worker responds with transportReady or error.
    /// Returns true if worker acknowledged (not necessarily if transport succeeded —
    /// the actual JACCL init happens asynchronously while coord also inits).
    @discardableResult
    func sendInitTransport(backend: String = "ring", rank: Int = 1, hostfileJSON: String? = nil) throws -> Bool {
        guard let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }
        var payload: [String: Any] = [
            "backend": backend,
            "rank": rank
        ]
        if let json = hostfileJSON {
            payload["hostfileJSON"] = json
        }
        let data = try JSONSerialization.data(withJSONObject: payload)
        let msg = ShardWireFormat.encode(msgType: .initTransport, payload: data)
        try conn.sendData(msg)

        // Wait for transportReady ack (worker sends this BEFORE blocking on JACCL init)
        let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
        guard let header = ShardWireFormat.decodeHeader(headerData) else {
            NovaMLXLog.warning("[RemoteShardPolicy] Bad initTransport response, using TCP fallback")
            return false
        }
        if header.msgType == .transportReady {
            NovaMLXLog.info("[RemoteShardPolicy] Worker acknowledged initTransport (backend=\(backend))")
            return true
        } else if header.msgType == .error {
            var errorMsg = "transport init failed"
            if header.payloadSize > 0 {
                let errorData = try conn.recvData(count: header.payloadSize)
                errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
            }
            NovaMLXLog.warning("[RemoteShardPolicy] Worker transport init failed: \(errorMsg), using TCP fallback")
            return false
        } else {
            NovaMLXLog.warning("[RemoteShardPolicy] Unexpected initTransport response: \(header.msgType)")
            return false
        }
    }

    // MARK: - Speculative Verification

    /// Send batch forward + per-position argmax request to worker.
    /// Input: [1, K+1, hidden_size] → Returns K+1 verified token IDs.
    public func speculativeVerify(input: MLXArray) async throws -> [Int] {
        guard isReady, let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }

        let msg = ShardWireFormat.encode(msgType: .speculativeVerify, hasTensor: true)
        try conn.sendData(msg)
        try conn.sendTensor(input)

        let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
        guard let header = ShardWireFormat.decodeHeader(headerData) else {
            throw ShardServiceError.invalidMessage("bad speculativeVerify response header")
        }

        switch header.msgType {
        case .verifiedTokens:
            let count = header.payloadSize / 4
            guard count > 0, header.payloadSize >= 4 else {
                throw ShardServiceError.invalidMessage("verifiedTokens payload too small")
            }
            let tokenData = try conn.recvData(count: header.payloadSize)
            var ids: [Int] = []
            ids.reserveCapacity(count)
            for i in 0..<count {
                let offset = i * 4
                let id = tokenData[offset..<(offset + 4)].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
                ids.append(Int(id))
            }
            return ids

        case .error:
            var errorMsg = "unknown error"
            if header.payloadSize > 0 {
                let errorData = try conn.recvData(count: header.payloadSize)
                errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
            }
            throw ShardServiceError.computeFailed(errorMsg)

        default:
            throw ShardServiceError.invalidMessage("unexpected speculativeVerify response: \(header.msgType)")
        }
    }

    /// Tell the worker to trim its KV cache to a specific position.
    public func rollbackCache(position: Int) async throws {
        guard isReady, let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }

        let payload = try JSONSerialization.data(withJSONObject: ["position": position])
        let msg = ShardWireFormat.encode(msgType: .rollbackCache, payload: payload)
        try conn.sendData(msg)

        let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
        guard let header = ShardWireFormat.decodeHeader(headerData) else {
            throw ShardServiceError.invalidMessage("bad rollbackCache response header")
        }

        switch header.msgType {
        case .cacheRolledBack:
            return // success
        case .error:
            var errorMsg = "unknown error"
            if header.payloadSize > 0 {
                let errorData = try conn.recvData(count: header.payloadSize)
                errorMsg = String(data: errorData, encoding: .utf8) ?? errorMsg
            }
            throw ShardServiceError.computeFailed(errorMsg)
        default:
            throw ShardServiceError.invalidMessage("unexpected rollbackCache response: \(header.msgType)")
        }
    }
}
