import Foundation
import MLX

// MARK: - ShardEngineError

/// Errors thrown by ShardEngine operations.
public enum ShardEngineError: Error, Sendable {
    /// The compute policy is not ready (weights not bound).
    case notReady
}

// MARK: - ComputePolicy

/// Protocol for a policy that manages weight loading and forward computation
/// for a shard of model layers.
public protocol ComputePolicy: Sendable {
    /// Whether the weights are loaded and computation can proceed.
    var isReady: Bool { get }

    /// Load and bind weights into GPU memory.
    func bindWeights() async throws

    /// Run forward computation through the assigned layers.
    ///
    /// - Parameters:
    ///   - input: Input activation tensor.
    ///   - cache: In-out KV cache (or other per-layer state).
    /// - Returns: Output activation tensor.
    func compute(input: MLXArray, cache: inout [Any]) throws -> MLXArray

    /// Release weights from GPU memory.
    func releaseWeights()
}

// MARK: - FitInMemoryPolicy

/// A ``ComputePolicy`` that assumes all assigned layers fit in local memory.
///
/// This is the baseline policy: weights are loaded on ``bindWeights()`` and
/// ``compute(input:cache:)`` is a placeholder that passes input through unchanged
/// (real forward pass wiring comes in a later task).
public final class FitInMemoryPolicy: ComputePolicy, @unchecked Sendable {

    /// The shard assignment this policy manages.
    public let assignment: ShardAssignment

    /// Whether weights are currently bound and ready for computation.
    public private(set) var isReady: Bool = false

    public init(assignment: ShardAssignment) {
        self.assignment = assignment
    }

    public func bindWeights() async throws {
        // Placeholder: in production, this loads safetensors shard into GPU memory.
        isReady = true
    }

    public func compute(input: MLXArray, cache: inout [Any]) throws -> MLXArray {
        guard isReady else {
            throw ShardEngineError.notReady
        }
        // Placeholder: real forward pass through assigned layers comes later.
        return input
    }

    public func releaseWeights() {
        isReady = false
    }
}

// MARK: - ShardEngine

/// Orchestrates pipeline-parallel inference across a shard of model layers.
///
/// Each ``ShardEngine`` owns one ``ComputePolicy`` and is responsible for:
/// 1. Receiving activations from the previous shard (if not the first).
/// 2. Running forward computation through its assigned layers.
/// 3. Sending activations to the next shard (if not the last).
/// 4. Returning final logits if it is the last shard.
public final class ShardEngine: @unchecked Sendable {

    /// The distributed communication group.
    public let group: DistributedGroup

    /// Which layers this shard is responsible for.
    public let assignment: ShardAssignment

    /// The compute policy driving forward computation.
    public let policy: ComputePolicy

    /// Whether this shard is the last in the pipeline (rank == size - 1).
    /// `false` when group size is 0 (uninitialized).
    public let isLastShard: Bool

    /// Whether this shard is the first in the pipeline (rank == 0).
    /// `false` when group size is 0 (uninitialized).
    public let isFirstShard: Bool

    public init(
        group: DistributedGroup,
        assignment: ShardAssignment,
        policy: ComputePolicy
    ) {
        self.group = group
        self.assignment = assignment
        self.policy = policy
        let size = group.size
        self.isLastShard = size > 0 && group.rank == size - 1
        self.isFirstShard = size > 0 && group.rank == 0
    }

    /// Run prefill (prompt processing) through the assigned layers.
    ///
    /// Pipeline logic:
    /// 1. If not the first shard, receive activations from the previous rank.
    /// 2. Forward through assigned layers via ``ComputePolicy/compute(input:cache:)``.
    /// 3. If not the last shard, send activations to the next rank.
    /// 4. Last shard returns the activation for sampling.
    ///
    /// - Parameter tokens: Input token IDs (shape `[seq_len]` or `[batch, seq_len]`).
    /// - Returns: Output activation from the last shard in the pipeline.
    public func prefill(tokens: MLXArray) async throws -> MLXArray {
        guard policy.isReady else {
            throw ShardEngineError.notReady
        }

        // Step 1: Receive from previous shard if not first.
        var activation: MLXArray
        if isFirstShard {
            activation = tokens
        } else {
            activation = MLXDistributedWrapper.recvLike(
                tokens,
                from: group.rank - 1,
                group: group
            )
        }

        // Step 2: Forward through assigned layers.
        var cache: [Any] = []
        activation = try policy.compute(input: activation, cache: &cache)

        // Step 3: Send to next shard if not last.
        if !isLastShard {
            _ = MLXDistributedWrapper.send(
                activation,
                to: group.rank + 1,
                group: group
            )
        }

        // Step 4: Last shard returns activation for sampling.
        return activation
    }

    /// Run decode (single-token generation) through the assigned layers.
    ///
    /// Same pipeline logic as ``prefill(tokens:)`` but for a single token.
    ///
    /// - Parameter token: A single token ID (scalar or shape `[1]`).
    /// - Returns: Output activation from the last shard in the pipeline.
    public func decode(token: MLXArray) async throws -> MLXArray {
        guard policy.isReady else {
            throw ShardEngineError.notReady
        }

        // Step 1: Receive from previous shard if not first.
        var activation: MLXArray
        if isFirstShard {
            activation = token
        } else {
            activation = MLXDistributedWrapper.recvLike(
                token,
                from: group.rank - 1,
                group: group
            )
        }

        // Step 2: Forward through assigned layers.
        var cache: [Any] = []
        activation = try policy.compute(input: activation, cache: &cache)

        // Step 3: Send to next shard if not last.
        if !isLastShard {
            _ = MLXDistributedWrapper.send(
                activation,
                to: group.rank + 1,
                group: group
            )
        }

        // Step 4: Last shard returns activation for sampling.
        return activation
    }
}
