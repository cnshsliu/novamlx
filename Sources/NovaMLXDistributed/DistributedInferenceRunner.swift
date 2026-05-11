import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils

// MARK: - DistributedInferenceError

/// Errors raised by ``DistributedInferenceRunner``.
public enum DistributedInferenceError: Error, LocalizedError {
    case noWorkersAvailable
    case modelNotLoaded(modelId: String)
    case shardPlanFailed(String)
    case distributedBackendUnavailable
    case coordinatorNotConfigured

    public var errorDescription: String? {
        switch self {
        case .noWorkersAvailable:
            "No workers available in the cluster"
        case .modelNotLoaded(let modelId):
            "Model not loaded: \(modelId)"
        case .shardPlanFailed(let reason):
            "Shard plan computation failed: \(reason)"
        case .distributedBackendUnavailable:
            "MLX distributed backend is not available"
        case .coordinatorNotConfigured:
            "This node is not configured as coordinator"
        }
    }
}

// MARK: - DistributedTokenizer

/// Minimal tokenizer interface needed by the distributed runner.
/// ``MLXEngine.Tokenizer`` already conforms via its encode/decode closures.
public struct DistributedTokenizer: Sendable {
    public let encode: @Sendable (String) -> [Int]
    public let decode: @Sendable ([Int]) -> String

    public init(
        encode: @Sendable @escaping (String) -> [Int],
        decode: @Sendable @escaping ([Int]) -> String
    ) {
        self.encode = encode
        self.decode = decode
    }
}

// MARK: - DistributedInferenceRunner

/// Orchestrates distributed inference across cluster workers using pipeline-parallel sharding.
///
/// On the coordinator node, this runner:
/// 1. Profiles the model via ``ModelAnalyzer`` to get layer memory estimates.
/// 2. Retrieves active workers from ``ClusterManager``.
/// 3. Computes a ``ShardPlan`` proportional to worker memory.
/// 4. Creates a ``ShardEngine`` per shard with ``FitInMemoryPolicy``.
/// 5. Initializes the ``DistributedGroup`` for inter-node communication.
/// 6. Runs prefill then a decode loop through the ShardEngine pipeline.
/// 7. Returns an ``InferenceResult``.
///
/// The current implementation wires the complete code path. Since ``FitInMemoryPolicy``
/// passes tensors through unchanged, actual distributed compute will produce placeholder
/// results until real forward-pass wiring is added.
public final class DistributedInferenceRunner: @unchecked Sendable {

    private let clusterConfig: ClusterConfig

    /// Called to obtain a tokenizer for the given model ID. Returns nil if model is not loaded.
    /// Injected by InferenceService to avoid a direct dependency on NovaMLXEngine.
    private let tokenizerProvider: @Sendable (String) -> DistributedTokenizer?

    /// Called to check whether a model is loaded and get its path.
    /// Returns the model directory path, or nil if the model is not loaded.
    private let modelPathProvider: @Sendable (String) -> String?

    public init(
        clusterConfig: ClusterConfig,
        tokenizerProvider: @Sendable @escaping (String) -> DistributedTokenizer?,
        modelPathProvider: @Sendable @escaping (String) -> String?
    ) {
        self.clusterConfig = clusterConfig
        self.tokenizerProvider = tokenizerProvider
        self.modelPathProvider = modelPathProvider
    }

    // MARK: - Public API

    /// Run distributed inference for a single request.
    ///
    /// - Parameter request: The inference request to process.
    /// - Returns: An ``InferenceResult`` with the generated text.
    public func generate(request: InferenceRequest) async throws -> InferenceResult {
        let modelId = request.model
        let startTime = Date()

        // 1. Get tokenizer (confirms model is loaded)
        guard let tokenizer = tokenizerProvider(modelId) else {
            throw DistributedInferenceError.modelNotLoaded(modelId: modelId)
        }

        // 2. Get model path for profiling
        guard let modelPath = modelPathProvider(modelId) else {
            throw DistributedInferenceError.modelNotLoaded(modelId: modelId)
        }

        // 3. Profile model layers
        let profiles: [LayerProfile]
        do {
            profiles = try await ModelAnalyzer.shared.analyze(modelPath: modelPath)
        } catch {
            throw DistributedInferenceError.shardPlanFailed("Model analysis failed: \(error)")
        }

        guard !profiles.isEmpty else {
            throw DistributedInferenceError.shardPlanFailed("No layer profiles produced")
        }

        // 4. Get active workers from cluster
        let workers = ClusterManager.shared.activeWorkers
        let nodeSpecs = workers.map { $0.spec }

        // If no remote workers, create a local-only node spec for single-node fallback
        let effectiveNodes: [NodeSpec]
        if nodeSpecs.isEmpty {
            let localMemory = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? 0
            effectiveNodes = [NodeSpec(
                nodeId: "local-coordinator",
                totalMemoryBytes: localMemory,
                computeCapability: 1.0,
                hostname: "127.0.0.1",
                port: clusterConfig.coordinatorPort
            )]
            NovaMLXLog.info("[Distributed] No remote workers — using local-only fallback")
        } else {
            effectiveNodes = nodeSpecs
        }

        // 5. Compute shard plan
        let plan = ShardPlan(
            profiles: profiles,
            nodes: effectiveNodes,
            strategy: clusterConfig.strategy
        )

        NovaMLXLog.info("[Distributed] Shard plan: \(plan.assignments.count) shards, \(plan.totalLayers) layers, strategy=\(plan.strategy.rawValue)")

        // 6. Initialize distributed group
        let group: DistributedGroup
        if MLXDistributedWrapper.isCBBackendAvailable {
            let backend = MLXDistributedWrapper.bestAvailableBackend()
            group = MLXDistributedWrapper.initialize(strict: false, backend: backend)
        } else {
            // Without a real distributed backend, use the sentinel group.
            // The ShardEngine will degrade to single-node passthrough.
            group = .uninitialized
        }

        // 7. Create shard engines
        var shardEngines: [ShardEngine] = []
        for assignment in plan.assignments {
            let policy = FitInMemoryPolicy(assignment: assignment)
            let shardEngine = ShardEngine(
                group: group,
                assignment: assignment,
                policy: policy
            )
            shardEngines.append(shardEngine)
        }

        // Bind weights on all shards
        for shardEngine in shardEngines {
            try await shardEngine.policy.bindWeights()
        }

        // 8. Tokenize input
        var promptTokens: [Int] = []
        for msg in request.messages {
            if let content = msg.content {
                promptTokens.append(contentsOf: tokenizer.encode(content))
            }
        }

        guard !promptTokens.isEmpty else {
            return InferenceResult(
                id: request.id,
                model: modelId,
                text: "",
                tokensPerSecond: 0,
                promptTokens: 0,
                completionTokens: 0,
                finishReason: .stop
            )
        }

        let promptArray = MLXArray(promptTokens.map { Int32($0) })

        // 9. Prefill through the first shard engine (coordinator owns rank 0)
        guard let firstShard = shardEngines.first else {
            throw DistributedInferenceError.shardPlanFailed("No shards created")
        }

        let prefillOutput = try await firstShard.prefill(
            tokens: promptArray,
            config: clusterConfig.prefill
        )

        // 10. Decode loop
        let maxTokens = request.maxTokens ?? 512
        var currentToken = prefillOutput
        var generatedTokenIds: [Int] = []
        var stopTokens: Set<String> = []
        if let stop = request.stop {
            stopTokens = Set(stop)
        }

        for _ in 0..<maxTokens {
            let decoded = try await firstShard.decode(token: currentToken)

            // Sample: argmax for now (greedy). Real sampling comes with forward-pass wiring.
            let sampledId = argmax(decoded)

            generatedTokenIds.append(sampledId)

            // Check stop sequences
            let fullText = tokenizer.decode(generatedTokenIds)
            if stopTokens.contains(where: { fullText.hasSuffix($0) }) {
                break
            }

            // Feed sampled token back for next step
            currentToken = MLXArray(Int32(sampledId))
        }

        let text = tokenizer.decode(generatedTokenIds)
        let elapsed = Date().timeIntervalSince(startTime)
        let tps = elapsed > 0 ? Double(generatedTokenIds.count) / elapsed : 0

        NovaMLXLog.info("[Distributed] Completed: \(generatedTokenIds.count) tokens in \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", tps)) tok/s)")

        return InferenceResult(
            id: request.id,
            model: modelId,
            text: text,
            tokensPerSecond: tps,
            promptTokens: promptTokens.count,
            completionTokens: generatedTokenIds.count,
            finishReason: .stop
        )
    }

    // MARK: - Private helpers

    /// Extract the argmax token ID from a logits tensor.
    private func argmax(_ logits: MLXArray) -> Int {
        // logits shape varies; flatten and take argmax
        let flat = logits.flattened()
        let index = MLX.argMax(flat).item(Int.self)
        return index
    }
}
