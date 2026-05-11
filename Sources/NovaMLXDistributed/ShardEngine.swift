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

    /// Run prefill — selects sequential or wavefront based on config and cluster state.
    public func prefill(tokens: MLXArray, config: PrefillConfig? = nil) async throws -> MLXArray {
        let cfg = config ?? PrefillConfig()
        let tokenCount = tokens.shape.reduce(1, *)
        if group.size > 1 && tokenCount >= cfg.minWavefrontTokens {
            return try await wavefrontPrefill(tokens: tokens, config: cfg)
        }
        return try await sequentialPrefill(tokens: tokens)
    }

    /// Sequential prefill: receive → compute → send.
    public func sequentialPrefill(tokens: MLXArray) async throws -> MLXArray {
        guard policy.isReady else {
            throw ShardEngineError.notReady
        }

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

        var cache: [Any] = []
        activation = try policy.compute(input: activation, cache: &cache)

        if !isLastShard {
            _ = MLXDistributedWrapper.send(
                activation,
                to: group.rank + 1,
                group: group
            )
        }

        return activation
    }

    /// Overlapped wavefront prefill for long prompts on multi-node clusters.
    private func wavefrontPrefill(
        tokens: MLXArray,
        config: PrefillConfig
    ) async throws -> MLXArray {
        guard policy.isReady else {
            throw ShardEngineError.notReady
        }

        let promptLen = tokens.shape.reduce(1, *)
        let worldSize = group.size
        let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
        let nReal = (promptLen - 1 + chunkSize - 1) / chunkSize
        let nLeading = group.rank
        let nTrailing = worldSize - 1 - group.rank

        // Activate send batching
        pipelineQueueSends = true
        clearPrefillSends()
        defer {
            pipelineQueueSends = false
            clearPrefillSends()
        }

        // Leading dummies — pure no-ops
        for _ in 0..<nLeading {}

        // Real chunks
        var cache: [Any] = []
        for i in 0..<nReal {
            let start = i * chunkSize
            let end = min(start + chunkSize, promptLen - 1)

            // Shape template for recv. For non-first ranks, this provides the
            // expected shape to recvLike. TODO: When real forward passes are wired,
            // activation shapes may differ from token shapes — update to use
            // the actual activation shape from the previous rank's output.
            let chunkShape: MLXArray
            if promptLen == 1 {
                chunkShape = tokens
            } else {
                chunkShape = tokens[start..<end]
            }

            var activation: MLXArray
            if isFirstShard {
                activation = chunkShape
            } else {
                activation = MLXDistributedWrapper.recvLike(
                    chunkShape,
                    from: group.rank - 1,
                    group: group
                )
            }

            activation = try policy.compute(input: activation, cache: &cache)

            if !isLastShard {
                MLX.eval(activation)
                prefillSendQueue.enqueue(PendingSend(
                    output: activation,
                    destination: group.rank + 1,
                    group: group
                ))
            }

            flushPrefillSends()
        }

        // Trailing dummies — pure no-ops
        for _ in 0..<nTrailing {}

        // Two final single-token passes (shared cache with chunk loop).
        // Pass 1: complete prompt processing for the last token.
        // Pass 2: generate first response token.
        var lastActivation: MLXArray? = nil
        for _ in 0..<2 {
            let lastToken: MLXArray
            if promptLen == 1 {
                lastToken = tokens
            } else {
                lastToken = tokens[(promptLen - 1)..<promptLen]
            }

            var activation: MLXArray
            if isFirstShard {
                activation = lastToken
            } else {
                activation = MLXDistributedWrapper.recvLike(
                    lastToken,
                    from: group.rank - 1,
                    group: group
                )
            }

            activation = try policy.compute(input: activation, cache: &cache)
            lastActivation = activation

            if !isLastShard {
                MLX.eval(activation)
                _ = MLXDistributedWrapper.send(activation, to: group.rank + 1, group: group)
            }
            flushPrefillSends()
        }

        // Final eval on cache state
        let cacheArrays = cache.compactMap { $0 as? MLXArray }
        if !cacheArrays.isEmpty {
            MLX.eval(cacheArrays)
        }

        // Return the activation from the last final pass (logits on last shard).
        // This is the computed result, not a re-slice of input tokens.
        guard let result = lastActivation else {
            // Fallback for empty prompt edge case
            return tokens
        }
        return result
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
