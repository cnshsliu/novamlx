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

    /// Whether to use MLX Ring transport for tensor data (falls back to TCP).
    private let useRingTransport: Bool

    /// Worker rank for Ring transport (always 1 for 2-node setup).
    private let workerRank: Int = 1

    public init(assignment: ShardAssignment, workerEndpoint: NodeEndpoint, modelId: String, modelPath: String? = nil, isFirst: Bool = false, isLast: Bool = false, useRingTransport: Bool = true) {
        self.assignment = assignment
        self.isFirst = isFirst
        self.isLast = isLast
        self.workerEndpoint = workerEndpoint
        self.modelId = modelId
        self.modelPath = modelPath
        self.useRingTransport = useRingTransport && RingTransportManager.shared.isReady
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
            // Send compute command via TCP (control plane)
            let computeMsg = ShardWireFormat.encode(msgType: .compute, hasTensor: false)
            try conn.sendData(computeMsg)

            // Send input tensor via Ring transport (data plane)
            eval(input)
            _ = RingTransportManager.shared.send(input, to: workerRank)

            // Receive result tensor via Ring transport
            // We know the output shape matches input's batch/seq dims but with vocab size
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

            // Worker sends result shape info via TCP, actual tensor via Ring
            let result = RingTransportManager.shared.recvLike(input, from: workerRank)
            return result
        } else {
            // Original TCP transport path
            let computeMsg = ShardWireFormat.encode(msgType: .compute, hasTensor: true)
            try conn.sendData(computeMsg)
            try conn.sendTensor(input)

            let headerData = try conn.recvData(count: ShardWireFormat.headerSize)
            guard let header = ShardWireFormat.decodeHeader(headerData) else {
                throw ShardServiceError.invalidMessage("bad response header")
            }

            switch header.msgType {
            case .computeResult:
                guard header.hasTensor else {
                    throw ShardServiceError.invalidMessage("computeResult missing tensor")
                }
                return try conn.recvTensor()

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
            let computeMsg = ShardWireFormat.encode(msgType: .compute, hasTensor: false)
            try conn.sendData(computeMsg)
            eval(input)
            _ = RingTransportManager.shared.send(input, to: workerRank)
        } else {
            let computeMsg = ShardWireFormat.encode(msgType: .compute, hasTensor: true)
            try conn.sendData(computeMsg)
            try conn.sendTensor(input)
        }
    }

    /// Receive result from a previous sendCompute call.
    func recvResult() throws -> MLXArray {
        guard let conn = lock.withLock({ connection }) else {
            throw ShardEngineError.notReady
        }
        if useRingTransport {
            // Read control header (worker sends ack/error via TCP)
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
            // Receive actual tensor via Ring — we need shape info from somewhere
            // For now, create a dummy template and use recvLike
            // TODO: pass actual shape info through TCP control header
            fatalError("Ring transport recvResult requires shape info — use compute() instead for now")
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
                return try conn.recvTensor()
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
