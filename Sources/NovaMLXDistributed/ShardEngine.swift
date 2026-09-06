import Foundation
import MLX
import MLXLMCommon
import NovaMLXEngine

// MARK: - Sendability helpers

/// Wraps non-Sendable values for capture in @Sendable closures.
/// Safe because `mlxContainer.perform` guarantees serial access.
final class SendableBox<T>: @unchecked Sendable {
    let value: T
    init(_ value: T) { self.value = value }
}

/// Wraps non-Sendable `[KVCache]` for capture in `perform` closures.
final class KVCacheBox: @unchecked Sendable {
    let caches: [KVCache]
    init(_ caches: [KVCache]) { self.caches = caches }
}

// MARK: - ShardEngineError

/// Errors thrown by ShardEngine operations.
public enum ShardEngineError: Error, Sendable {
    /// The compute policy is not ready (weights not bound).
    case notReady
    /// The model container could not be found.
    case modelNotAvailable(String)
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
    /// - Parameter input: Input activation tensor (token IDs or hidden state).
    /// - Returns: Output activation tensor (hidden state or logits).
    func compute(input: MLXArray) async throws -> MLXArray

    /// Run forward computation + argmax sampling remotely.
    ///
    /// Default implementation: compute then argmax locally.
    /// RemoteShardPolicy overrides to send computeAndSample wire message
    /// and receive a 4-byte token ID instead of the full logits tensor.
    ///
    /// This is the key optimization for decode: the worker returns a single
    /// token ID (4 bytes) instead of the full logits tensor (~970KB).
    func computeAndSample(input: MLXArray) async throws -> Int

    /// Release weights from GPU memory.
    func releaseWeights()

    /// Run forward + per-position argmax for speculative verification.
    /// Input: [1, K+1, hidden_size] → Output: K+1 token IDs.
    /// Default: compute then argmax each position.
    func speculativeVerify(input: MLXArray) async throws -> [Int]

    /// Trim KV cache to keep only entries up to the given position.
    func rollbackCache(position: Int) async throws
}

// MARK: - ComputePolicy Default Implementations

extension ComputePolicy {
    public func computeAndSample(input: MLXArray) async throws -> Int {
        let logits = try await compute(input: input)
        return argmaxToken(logits)
    }

    public func speculativeVerify(input: MLXArray) async throws -> [Int] {
        let logits = try await compute(input: input)
        let seqLen = logits.ndim >= 2 ? logits.dim(logits.ndim - 2) : 1
        var ids: [Int] = []
        ids.reserveCapacity(seqLen)
        for pos in 0..<seqLen {
            let posLogits: MLXArray
            if logits.ndim == 3 {
                posLogits = logits[0..., pos, 0...]
            } else if logits.ndim == 2 {
                posLogits = logits[pos, 0...]
            } else {
                posLogits = logits
            }
            ids.append(MLX.argMax(posLogits.flattened()).item(Int.self))
        }
        return ids
    }

    public func rollbackCache(position: Int) async throws {
        // Default: no-op (FitInMemoryPolicy doesn't support rollback)
    }
}

/// Argmax: pick the token with highest logit score.
/// Shared by coordinator (decode loop) and worker (computeAndSample handler).
public func argmaxToken(_ logits: MLXArray) -> Int {
    let lastLogits: MLXArray
    if logits.ndim == 3 {
        lastLogits = logits[0..., -1, 0...]
    } else if logits.ndim == 2 {
        lastLogits = logits[-1, 0...]
    } else {
        lastLogits = logits
    }
    return MLX.argMax(lastLogits.flattened()).item(Int.self)
}

// MARK: - FitInMemoryPolicy

/// A ``ComputePolicy`` that runs the full model forward pass on the coordinator.
///
/// In Phase 1 (single-node), the coordinator loads the entire model and this policy
/// runs all layers in sequence via ``MLXLMCommon/ModelContainer/perform``.
/// KV caches are persisted across decode steps for correct autoregressive generation.
public final class FitInMemoryPolicy: ComputePolicy, @unchecked Sendable {

    /// The shard assignment this policy manages.
    public let assignment: ShardAssignment

    /// Whether weights are currently bound and ready for computation.
    public private(set) var isReady: Bool = false

    private weak var engine: MLXEngine?
    private let modelId: String
    private var kvCaches: [KVCache] = []

    public init(assignment: ShardAssignment, engine: MLXEngine, modelId: String) {
        self.assignment = assignment
        self.engine = engine
        self.modelId = modelId
    }

    public func bindWeights() async throws {
        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else {
            throw ShardEngineError.modelNotAvailable(modelId)
        }
        // Create fresh KV caches for this inference session
        let cacheBox = try await mlxContainer.perform { context in
            return KVCacheBox(try context.model.newCache(parameters: nil))
        }
        self.kvCaches = cacheBox.caches
        isReady = true
    }

    public func compute(input: MLXArray) async throws -> MLXArray {
        guard isReady else {
            throw ShardEngineError.notReady
        }
        guard let container = engine?.getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else {
            throw ShardEngineError.modelNotAvailable(modelId)
        }
        let inputBox = SendableBox(input)
        let cacheBox = KVCacheBox(kvCaches)
        let resultBox = await mlxContainer.perform { context in
            let logits = context.model(inputBox.value, cache: cacheBox.caches)
            MLX.eval(logits)
            return SendableBox(logits)
        }
        return resultBox.value
    }

    public func releaseWeights() {
        kvCaches = []
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

        activation = try await policy.compute(input: activation)

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

            activation = try await policy.compute(input: activation)

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

            activation = try await policy.compute(input: activation)
            lastActivation = activation

            if !isLastShard {
                MLX.eval(activation)
                _ = MLXDistributedWrapper.send(activation, to: group.rank + 1, group: group)
            }
            flushPrefillSends()
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
        activation = try await policy.compute(input: activation)

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
