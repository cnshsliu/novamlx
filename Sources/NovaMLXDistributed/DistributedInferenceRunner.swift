import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine
import Tokenizers

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

/// Orchestrates distributed inference across cluster nodes using pipeline-parallel sharding.
///
/// On the coordinator node, this runner:
/// 1. Profiles the model via ``ModelAnalyzer`` to get layer memory estimates.
/// 2. Builds a node list: Coordinator (rank 0) + available workers.
/// 3. Computes a ``ShardPlan`` proportional to node memory.
/// 4. Creates a ``ShardEngine`` per shard with the right policy:
///    - Coordinator shards → ``SlicedForwardPolicy`` (runs assigned layers locally)
///    - Remote worker shards → ``RemoteShardPolicy`` (TCP delegate to worker)
/// 5. Runs prefill then a decode loop through the shard pipeline.
public final class DistributedInferenceRunner: @unchecked Sendable {

    private let clusterConfig: ClusterConfig

    private let tokenizerProvider: @Sendable (String) -> DistributedTokenizer?
    private let modelPathProvider: @Sendable (String) -> String?
    private weak var engine: MLXEngine?

    /// Cache of the last computed shard plan for observability.
    public private(set) var lastShardPlan: (modelId: String, plan: ShardPlan)?

    public init(
        clusterConfig: ClusterConfig,
        tokenizerProvider: @Sendable @escaping (String) -> DistributedTokenizer?,
        modelPathProvider: @Sendable @escaping (String) -> String?,
        engine: MLXEngine
    ) {
        self.clusterConfig = clusterConfig
        self.tokenizerProvider = tokenizerProvider
        self.modelPathProvider = modelPathProvider
        self.engine = engine
    }

    // MARK: - Public API

    public func generate(request: InferenceRequest) async throws -> InferenceResult {
        let modelId = request.model
        let startTime = Date()

        // 1. Get model path (needed for both profiling and tokenizer fallback)
        guard let modelPath = modelPathProvider(modelId) else {
            throw DistributedInferenceError.modelNotLoaded(modelId: modelId)
        }

        // 2. Get tokenizer — try provider first (engine container), then load from disk
        var tokenizer = tokenizerProvider(modelId)
        if tokenizer == nil {
            NovaMLXLog.info("[Distributed] Tokenizer not in engine, loading from disk: \(modelPath)")
            let modelDir = URL(fileURLWithPath: modelPath)
            let loaded = try await AutoTokenizer.from(modelFolder: modelDir)
            tokenizer = DistributedTokenizer(
                encode: { text in loaded.encode(text: text, addSpecialTokens: true) },
                decode: { tokens in loaded.decode(tokens: tokens, skipSpecialTokens: true) }
            )
        }
        guard let tokenizer = tokenizer else {
            throw DistributedInferenceError.modelNotLoaded(modelId: modelId)
        }

        // Check for pre-loaded shard engines from ClusterModelManager (fast path)
        var shardEngines: [ShardEngine]
        let shouldReleaseWeights: Bool

        if let preloaded = ClusterModelManager.shared.getShardEngines(for: modelId) {
            shardEngines = preloaded
            shouldReleaseWeights = false
            if let plan = ClusterModelManager.shared.shardPlan {
                lastShardPlan = (modelId, plan)
                DistributedInferenceRunnerCache.shared.setPlan(modelId: modelId, plan: plan)
            }
            // Reset KV caches on all local shards for fresh conversation context.
            // Pre-loaded shard engines persist across requests — without reset,
            // the model continues the previous conversation instead of starting fresh.
            for shard in shardEngines {
                if let slicedPolicy = shard.policy as? SlicedForwardPolicy {
                    try? await slicedPolicy.resetCaches()
                }
            }
            NovaMLXLog.info("[Distributed] Using pre-loaded shard engines (\(preloaded.count) shards, caches reset)")
        } else {
            shouldReleaseWeights = true

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

            // 3.5 Ensure model is loaded in main engine for SlicedForwardPolicy
            if let engine = engine, engine.getContainer(for: modelId) == nil {
                NovaMLXLog.info("[Distributed] Loading model \(modelId) into main engine for local shard...")
                let modelDir = URL(fileURLWithPath: modelPath)
                let config = ModelConfig(
                    identifier: ModelIdentifier(id: modelId, family: .qwen)
                )
                _ = try await engine.loadModel(from: modelDir, config: config)
                NovaMLXLog.info("[Distributed] Model \(modelId) loaded in main engine")
            }

            // 4. Build node list: Coordinator (rank 0) + available workers
            let availableWorkers = ClusterManager.shared.workers.values
                .filter { $0.status == .ready || $0.status == .active }

            let localMemory = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? ProcessInfo.processInfo.physicalMemory
            let coordinatorSpec = NodeSpec(
                nodeId: "local-coordinator",
                totalMemoryBytes: localMemory,
                computeCapability: 1.0,
                hostname: "127.0.0.1",
                port: clusterConfig.coordinatorPort
            )
            let effectiveNodes = [coordinatorSpec] + availableWorkers.map(\.spec)

            NovaMLXLog.info("[Distributed] Nodes: \(effectiveNodes.count) (coordinator=\(bytesFormatted(localMemory)), workers=\(availableWorkers.count))")

            // 5. Compute shard plan
            let plan = ShardPlan(
                profiles: profiles,
                nodes: effectiveNodes,
                strategy: clusterConfig.strategy,
                minLayersPerShard: clusterConfig.minLayersPerShard
            )

            lastShardPlan = (modelId, plan)
            DistributedInferenceRunnerCache.shared.setPlan(modelId: modelId, plan: plan)

            for (i, a) in plan.assignments.enumerated() {
                NovaMLXLog.info("[Distributed] Shard \(i): \(a.nodeId) layers \(a.startLayer)..<\(a.endLayer) (\(a.layerCount) layers, \(bytesFormatted(a.memoryEstimate)))")
            }

            // 6. Initialize distributed group
            // Only init if Ring transport is actually being used (RemoteShardPolicy.useRingTransport).
            // JACCL/Ring groups interfere with TCP transport — send/recv calls are NOT no-ops
            // when the group is valid but has no peer nodes.
            let group: DistributedGroup = .uninitialized

            // 7. Create shard engines with proper policies
            shardEngines = []
            for (index, assignment) in plan.assignments.enumerated() {
                let isFirst = index == 0
                let isLast = plan.assignments.count <= 2 ? (index == 0) : (index == plan.assignments.count - 1)
                let policy: ComputePolicy

                if assignment.nodeId == "local-coordinator" {
                    policy = SlicedForwardPolicy(
                        assignment: assignment,
                        engine: engine!,
                        modelId: modelId,
                        isFirst: isFirst,
                        isLast: isLast
                    )
                    NovaMLXLog.info("[Distributed] Shard \(index): local SlicedForward layers \(assignment.startLayer)..<\(assignment.endLayer)")
                } else {
                    let worker = availableWorkers.first { $0.spec.nodeId == assignment.nodeId }
                    let host = worker?.spec.networkHost ?? worker?.spec.hostname ?? assignment.nodeId
                    let endpoint = NodeEndpoint(
                        nodeId: assignment.nodeId,
                        host: host,
                        port: 7010
                    )
                    policy = RemoteShardPolicy(assignment: assignment, workerEndpoint: endpoint, modelId: modelId, modelPath: modelPath, isFirst: isFirst, isLast: isLast)
                    NovaMLXLog.info("[Distributed] Shard \(index): remote \(host):7010 layers \(assignment.startLayer)..<\(assignment.endLayer)")
                }

                shardEngines.append(ShardEngine(
                    group: group,
                    assignment: assignment,
                    policy: policy
                ))
            }

            // Bind weights on all shards (creates KV caches, connects to remote workers)
            for (i, shardEngine) in shardEngines.enumerated() {
                do {
                    try await shardEngine.policy.bindWeights()
                    NovaMLXLog.info("[Distributed] Shard \(i) weights bound")
                } catch {
                    NovaMLXLog.error("[Distributed] Shard \(i) bindWeights failed: \(error)")
                    throw DistributedInferenceError.shardPlanFailed("Shard \(i) bindWeights failed: \(error)")
                }
            }
        }

        // 8. Tokenize input — apply chat template via MLXLMCommon tokenizer
        var promptTokens: [Int] = []
        if let container = engine?.getContainer(for: modelId),
           let mlxTokenizer = await container.mlxContainer?.tokenizer {
            let messageDicts: [[String: any Sendable]] = request.messages.compactMap { msg in
                guard let content = msg.content else { return nil }
                return ["role": msg.role.rawValue, "content": content] as [String: any Sendable]
            }
            if let rendered = try? mlxTokenizer.applyChatTemplate(messages: messageDicts, tools: nil, additionalContext: nil) {
                promptTokens = rendered
                NovaMLXLog.info("[Distributed] Chat template applied: \(promptTokens.count) tokens")
            }
        }
        // Fallback: raw encode if chat template not available
        if promptTokens.isEmpty {
            for msg in request.messages {
                if let content = msg.content {
                    promptTokens.append(contentsOf: tokenizer.encode(content))
                }
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

        // 9. Prefill through the shard pipeline
        guard !shardEngines.isEmpty else {
            throw DistributedInferenceError.shardPlanFailed("No shards created")
        }

        var activation: MLXArray

        // Bypass ShardEngine for prefill — call policy.compute() directly.
        // ShardEngine's MLX distributed recv/send interfere with TCP transport.
        if shardEngines.count == 2,
           let coordPolicy = shardEngines[0].policy as? SlicedForwardPolicy,
           let workerPolicy = shardEngines[1].policy as? RemoteShardPolicy,
           promptTokens.count > 256 {
            activation = try await pipelinedPrefill(
                tokens: promptArray,
                coordPolicy: coordPolicy,
                workerPolicy: workerPolicy
            )
        } else {
            // Direct pipeline: coordinator → worker
            activation = promptArray
            for shard in shardEngines {
                activation = try await shard.policy.compute(input: activation)
            }
        }

        // 10. Decode loop — speculative decoding with N-gram speculation
        let maxTokens = request.maxTokens ?? 512
        var generatedTokenIds: [Int] = []
        var stopTokens: Set<String> = []
        if let stop = request.stop {
            stopTokens = Set(stop)
        }

        // EOS detection: use tokenizer + fallback canonical IDs
        var eosTokenIds: Set<Int> = [151645, 151643] // Qwen3.5/3.6 canonical: <|im_end|>,
        if let container = engine?.getContainer(for: modelId),
           let eosId = container.tokenizer?.eosTokenId {
            eosTokenIds.insert(eosId)
        }

        // Detect remote sampling: last shard is remote → worker does argmax
        let remoteSamplingEnabled = shardEngines.count > 1 && shardEngines.last?.policy is RemoteShardPolicy
        let coordHasMamba = (shardEngines.first?.policy as? SlicedForwardPolicy)?.hasMambaCache ?? false
        let extraPredictionLayers = 5

        // Adaptive speculation: start enabled for 2-shard setups with remote sampling.
        // Runtime accuracy tracking auto-disables if prediction accuracy < 30%.
        let speculationCapable = remoteSamplingEnabled && shardEngines.count == 2
        var speculationEnabled = speculationCapable

        // Adaptive accuracy tracking
        let adaptiveWindow = 20
        let disableThreshold = 0.30
        var recentPredictions: [Bool] = []  // rolling window of last N results
        var correctPredictions = 0
        var totalPredictions = 0
        // Draft length: number of tokens to speculatively predict per iteration.
        // K=1 (single-token) is optimal for 2-node setups where coord ≈ worker speed.
        // K>1 is beneficial when coord is much faster than worker (3+ nodes).
        let draftLength = 1

        NovaMLXLog.info("[Distributed] Remote sampling: \(remoteSamplingEnabled), speculation: \(speculationCapable) (mamba: \(coordHasMamba)), draft: \(draftLength), extraPredLayers: \(extraPredictionLayers)")

        if speculationCapable {
            let coordPolicy = shardEngines[0].policy
            guard let slicedCoord = coordPolicy as? SlicedForwardPolicy else {
                throw DistributedInferenceError.shardPlanFailed("Coord policy is not SlicedForwardPolicy")
            }
            let workerPolicy = shardEngines[1].policy as! RemoteShardPolicy
            var currentPosition = promptTokens.count

            // First token: run head on prefill hidden state (coord has isLast=true)
            let firstHeadResult = await slicedCoord.computeHeadOnly(activation)
            guard let firstResult = firstHeadResult else {
                throw DistributedInferenceError.shardPlanFailed("computeHeadOnly returned nil on prefill output")
            }
            let firstToken = firstResult.tokenId
            if !eosTokenIds.contains(firstToken) {
                generatedTokenIds.append(firstToken)
            }
            // Compute coord activation for first token (embedding + coord layers → hidden for worker)
            activation = try await coordPolicy.compute(input: MLXArray(Int32(firstToken)))
            currentPosition += 1

            // Continuous async pipeline decode loop
            var timingLogInterval = 0
            while generatedTokenIds.count < maxTokens {
                if speculationEnabled {
                    // === SPECULATIVE PATH — coord+worker compute simultaneously ===
                    let t0 = CFAbsoluteTimeGetCurrent()

                    // Step 1: Fire worker compute async (returns hidden state, not token ID)
                    let activationBox = SendableBox(activation)
                    let workerTask = Task {
                        SendableBox(try await workerPolicy.compute(input: activationBox.value))
                    }

                    // Step 2: Draft token N+1 while worker is busy
                    let draft = try? await slicedCoord.draftTokens(
                        from: activation, count: draftLength, extraPredictionLayers: extraPredictionLayers
                    )

                    let tPrecompute = CFAbsoluteTimeGetCurrent()

                    // Step 3: Await worker hidden state, run head locally
                    let workerHidden = try await workerTask.value.value
                    let tWorkerDone = CFAbsoluteTimeGetCurrent()
                    guard let headResult = await slicedCoord.computeHeadOnly(workerHidden) else {
                        break
                    }
                    let actualToken = headResult.tokenId

                    // Step 4: Verify draft against actual token
                    if let draft = draft, draft.count > 0, draft.predictedTokens[0] == actualToken {
                        // HIT — use precomputed activation (zero idle time)
                        activation = draft.precomputedStates[0]
                        recordPrediction(true, &recentPredictions, &correctPredictions, &totalPredictions)
                        currentPosition += 1

                        if eosTokenIds.contains(actualToken) { break }
                        generatedTokenIds.append(actualToken)

                        // Verify subsequent draft tokens
                        for i in 1..<draft.count {
                            let verifyActivation = SendableBox(activation)
                            let verifyTask = Task {
                                SendableBox(try await workerPolicy.compute(input: verifyActivation.value))
                            }
                            let verifyHidden = try await verifyTask.value.value
                            guard let verifyHead = await slicedCoord.computeHeadOnly(verifyHidden) else { break }
                            let verifyToken = verifyHead.tokenId

                            if draft.predictedTokens[i] == verifyToken {
                                activation = draft.precomputedStates[i]
                                recordPrediction(true, &recentPredictions, &correctPredictions, &totalPredictions)
                                currentPosition += 1

                                if eosTokenIds.contains(verifyToken) { break }
                                generatedTokenIds.append(verifyToken)
                            } else {
                                // Mismatch at draft[i]: rollback remaining, recompute
                                recordPrediction(false, &recentPredictions, &correctPredictions, &totalPredictions)
                                try? await slicedCoord.rollbackCache(
                                    position: currentPosition,
                                    speculatedCount: draft.count - i,
                                    mambaSnapshot: draft.mambaSnapshot
                                )
                                activation = try await coordPolicy.compute(input: MLXArray(Int32(verifyToken)))
                                currentPosition += 1

                                if eosTokenIds.contains(verifyToken) { break }
                                generatedTokenIds.append(verifyToken)
                                break
                            }
                        }
                    } else {
                        // MISS — rollback draft cache and recompute
                        recordPrediction(false, &recentPredictions, &correctPredictions, &totalPredictions)
                        if let draft = draft {
                            try? await slicedCoord.rollbackCache(
                                position: currentPosition,
                                speculatedCount: draft.count,
                                mambaSnapshot: draft.mambaSnapshot
                            )
                        }
                        activation = try await coordPolicy.compute(input: MLXArray(Int32(actualToken)))
                        currentPosition += 1

                        if eosTokenIds.contains(actualToken) { break }
                        generatedTokenIds.append(actualToken)
                    }

                    let t5 = CFAbsoluteTimeGetCurrent()

                    // Timing log every 50 tokens
                    timingLogInterval += 1
                    if timingLogInterval % 50 == 1 {
                        let totalMs = (t5 - t0) * 1000
                        let precomputeMs = (tPrecompute - t0) * 1000
                        let workerWaitMs = (tWorkerDone - tPrecompute) * 1000
                        let accuracy = totalPredictions > 0 ? Double(correctPredictions) / Double(totalPredictions) : 0
                        NovaMLXLog.info("[Distributed] Pipeline[\(generatedTokenIds.count)]: \(String(format: "%.1f", totalMs))ms draft=\(String(format: "%.1f", precomputeMs))ms wait=\(String(format: "%.1f", workerWaitMs))ms acc=\(String(format: "%.0f%%", accuracy * 100))(\(correctPredictions)/\(totalPredictions))")
                    }

                    // Adaptive check: disable speculation if accuracy too low
                    if recentPredictions.count >= adaptiveWindow {
                        let rolling = Double(recentPredictions.filter { $0 }.count) / Double(recentPredictions.count)
                        if rolling < disableThreshold {
                            speculationEnabled = false
                            NovaMLXLog.info("[Distributed] Speculation disabled: rolling accuracy \(String(format: "%.0f%%", rolling * 100)) < \(String(format: "%.0f%%", disableThreshold * 100)) threshold")
                        }
                    }

                    // Stop check for text suffix
                    let fullText = tokenizer.decode(generatedTokenIds)
                    if stopTokens.contains(where: { fullText.hasSuffix($0) }) { break }
                } else {
                    // === SEQUENTIAL PATH (speculation disabled) ===
                    // Worker returns hidden state → coord runs head → token
                    let workerHidden = try await workerPolicy.compute(input: activation)
                    guard let headResult = await slicedCoord.computeHeadOnly(workerHidden) else { break }
                    let sampledId = headResult.tokenId
                    if eosTokenIds.contains(sampledId) { break }
                    generatedTokenIds.append(sampledId)
                    let fullText = tokenizer.decode(generatedTokenIds)
                    if stopTokens.contains(where: { fullText.hasSuffix($0) }) { break }
                    activation = try await coordPolicy.compute(input: MLXArray(Int32(sampledId)))
                    currentPosition += 1
                }
            }

            let rate = totalPredictions > 0 ? Double(correctPredictions) / Double(totalPredictions) : 0
            NovaMLXLog.info("[Distributed] Pipeline: \(correctPredictions)/\(totalPredictions) correct (\(String(format: "%.0f%%", rate * 100))), \(generatedTokenIds.count) tokens, speculative=\(speculationEnabled)")
        } else {
            // Original non-speculative decode loop
            for i in 0..<maxTokens {
                let sampledId: Int
                if i == 0 {
                    // First token from prefill output (coord has head)
                    if let slicedCoord = shardEngines.first?.policy as? SlicedForwardPolicy,
                       let headResult = await slicedCoord.computeHeadOnly(activation) {
                        sampledId = headResult.tokenId
                    } else {
                        sampledId = argmax(activation)
                    }
                } else if remoteSamplingEnabled {
                    // Worker returns hidden state → coord runs head
                    let workerPolicy = shardEngines.last!.policy as! RemoteShardPolicy
                    let workerHidden = try await workerPolicy.compute(input: activation)
                    if let slicedCoord = shardEngines.first?.policy as? SlicedForwardPolicy,
                       let headResult = await slicedCoord.computeHeadOnly(workerHidden) {
                        sampledId = headResult.tokenId
                    } else {
                        sampledId = argmax(workerHidden)
                    }
                } else {
                    sampledId = argmax(activation)
                }

                if eosTokenIds.contains(sampledId) { break }
                generatedTokenIds.append(sampledId)
                let fullText = tokenizer.decode(generatedTokenIds)
                if stopTokens.contains(where: { fullText.hasSuffix($0) }) { break }

                if remoteSamplingEnabled {
                    activation = try await shardEngines[0].policy.compute(input: MLXArray(Int32(sampledId)))
                } else {
                    var decodeActivation = MLXArray(Int32(sampledId))
                    for shard in shardEngines {
                        decodeActivation = try await shard.policy.compute(input: decodeActivation)
                    }
                    activation = decodeActivation
                }
            }
        }

        let text = tokenizer.decode(generatedTokenIds)
        let elapsed = Date().timeIntervalSince(startTime)
        let tps = elapsed > 0 ? Double(generatedTokenIds.count) / elapsed : 0

        NovaMLXLog.info("[Distributed] Completed: \(generatedTokenIds.count) tokens in \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", tps)) tok/s)")

        let accuracy = totalPredictions > 0 ? Double(correctPredictions) / Double(totalPredictions) : nil
        DistributedInferenceRunnerCache.shared.recordStats(DistributedInferenceStats(
            tokensPerSecond: tps,
            promptTokens: promptTokens.count,
            completionTokens: generatedTokenIds.count,
            elapsedSeconds: elapsed,
            speculationAccuracy: accuracy
        ))

        // Release weights only if we created them in this request
        if shouldReleaseWeights {
            for shardEngine in shardEngines {
                shardEngine.policy.releaseWeights()
            }
        }

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

    /// Streaming distributed inference — yields tokens as they're decoded.
    public func stream(request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    let modelId = request.model
                    let startTime = Date()

                    // Setup: same as generate() — tokenize + prefill
                    guard let modelPath = modelPathProvider(modelId) else {
                        throw DistributedInferenceError.modelNotLoaded(modelId: modelId)
                    }

                    var tokenizer = tokenizerProvider(modelId)
                    if tokenizer == nil {
                        let modelDir = URL(fileURLWithPath: modelPath)
                        let loaded = try await AutoTokenizer.from(modelFolder: modelDir)
                        tokenizer = DistributedTokenizer(
                            encode: { text in loaded.encode(text: text, addSpecialTokens: true) },
                            decode: { tokens in loaded.decode(tokens: tokens, skipSpecialTokens: true) }
                        )
                    }
                    guard let tokenizer = tokenizer else {
                        throw DistributedInferenceError.modelNotLoaded(modelId: modelId)
                    }

                    var shardEngines: [ShardEngine]
                    let shouldReleaseWeights: Bool

                    if let preloaded = ClusterModelManager.shared.getShardEngines(for: modelId) {
                        shardEngines = preloaded
                        shouldReleaseWeights = false
                        if let plan = ClusterModelManager.shared.shardPlan {
                            lastShardPlan = (modelId, plan)
                            DistributedInferenceRunnerCache.shared.setPlan(modelId: modelId, plan: plan)
                        }
                        for shard in shardEngines {
                            if let slicedPolicy = shard.policy as? SlicedForwardPolicy {
                                try? await slicedPolicy.resetCaches()
                            }
                        }
                        NovaMLXLog.info("[Distributed-Stream] Using pre-loaded shard engines (\(preloaded.count) shards, caches reset)")
                    } else {
                        shouldReleaseWeights = true

                        let profiles: [LayerProfile]
                        do {
                            profiles = try await ModelAnalyzer.shared.analyze(modelPath: modelPath)
                        } catch {
                            throw DistributedInferenceError.shardPlanFailed("Model analysis failed: \(error)")
                        }
                        guard !profiles.isEmpty else {
                            throw DistributedInferenceError.shardPlanFailed("No layer profiles produced")
                        }

                        if let engine = engine, engine.getContainer(for: modelId) == nil {
                            NovaMLXLog.info("[Distributed-Stream] Loading model \(modelId) into main engine for local shard...")
                            let modelDir = URL(fileURLWithPath: modelPath)
                            let config = ModelConfig(
                                identifier: ModelIdentifier(id: modelId, family: .qwen)
                            )
                            _ = try await engine.loadModel(from: modelDir, config: config)
                        }

                        let availableWorkers = ClusterManager.shared.workers.values
                            .filter { $0.status == .ready || $0.status == .active }

                        let localMemory = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? ProcessInfo.processInfo.physicalMemory
                        let coordinatorSpec = NodeSpec(
                            nodeId: "local-coordinator",
                            totalMemoryBytes: localMemory,
                            computeCapability: 1.0,
                            hostname: "127.0.0.1",
                            port: clusterConfig.coordinatorPort
                        )
                        let effectiveNodes = [coordinatorSpec] + availableWorkers.map(\.spec)

                        let plan = ShardPlan(
                            profiles: profiles,
                            nodes: effectiveNodes,
                            strategy: clusterConfig.strategy,
                            minLayersPerShard: clusterConfig.minLayersPerShard
                        )

                        lastShardPlan = (modelId, plan)
                        DistributedInferenceRunnerCache.shared.setPlan(modelId: modelId, plan: plan)

                        let group: DistributedGroup = .uninitialized
                        shardEngines = []
                        for (index, assignment) in plan.assignments.enumerated() {
                            let isFirst = index == 0
                            let isLast = plan.assignments.count <= 2 ? (index == 0) : (index == plan.assignments.count - 1)
                            let policy: ComputePolicy

                            if assignment.nodeId == "local-coordinator" {
                                policy = SlicedForwardPolicy(
                                    assignment: assignment,
                                    engine: engine!,
                                    modelId: modelId,
                                    isFirst: isFirst,
                                    isLast: isLast
                                )
                            } else {
                                let worker = availableWorkers.first { $0.spec.nodeId == assignment.nodeId }
                                let host = worker?.spec.networkHost ?? worker?.spec.hostname ?? assignment.nodeId
                                let endpoint = NodeEndpoint(
                                    nodeId: assignment.nodeId,
                                    host: host,
                                    port: 7010
                                )
                                policy = RemoteShardPolicy(assignment: assignment, workerEndpoint: endpoint, modelId: modelId, modelPath: modelPath, isFirst: isFirst, isLast: isLast)
                            }
                            shardEngines.append(ShardEngine(group: group, assignment: assignment, policy: policy))
                        }

                        for (i, shardEngine) in shardEngines.enumerated() {
                            do {
                                try await shardEngine.policy.bindWeights()
                            } catch {
                                throw DistributedInferenceError.shardPlanFailed("Shard \(i) bindWeights failed: \(error)")
                            }
                        }
                    }

                    // Tokenize input
                    var promptTokens: [Int] = []
                    if let container = engine?.getContainer(for: modelId),
                       let mlxTokenizer = await container.mlxContainer?.tokenizer {
                        let messageDicts: [[String: any Sendable]] = request.messages.compactMap { msg in
                            guard let content = msg.content else { return nil }
                            return ["role": msg.role.rawValue, "content": content] as [String: any Sendable]
                        }
                        if let rendered = try? mlxTokenizer.applyChatTemplate(messages: messageDicts, tools: nil, additionalContext: nil) {
                            promptTokens = rendered
                        }
                    }
                    if promptTokens.isEmpty {
                        for msg in request.messages {
                            if let content = msg.content {
                                promptTokens.append(contentsOf: tokenizer.encode(content))
                            }
                        }
                    }

                    guard !promptTokens.isEmpty else {
                        continuation.yield(Token(id: 0, text: "", finishReason: .stop, promptTokens: promptTokens.count))
                        continuation.finish()
                        return
                    }

                    let promptArray = MLXArray(promptTokens.map { Int32($0) })
                    guard !shardEngines.isEmpty else {
                        throw DistributedInferenceError.shardPlanFailed("No shards created")
                    }

                    // Prefill
                    var activation: MLXArray
                    if shardEngines.count == 2,
                       let coordPolicy = shardEngines[0].policy as? SlicedForwardPolicy,
                       let workerPolicy = shardEngines[1].policy as? RemoteShardPolicy,
                       promptTokens.count > 256 {
                        activation = try await pipelinedPrefill(
                            tokens: promptArray,
                            coordPolicy: coordPolicy,
                            workerPolicy: workerPolicy
                        )
                    } else {
                        activation = promptArray
                        for shard in shardEngines {
                            activation = try await shard.policy.compute(input: activation)
                        }
                    }

                    // Streaming decode loop
                    let maxTokens = request.maxTokens ?? 512
                    var generatedCount = 0
                    var stopTokens: Set<String> = []
                    if let stop = request.stop { stopTokens = Set(stop) }

                    var eosTokenIds: Set<Int> = [151645, 151643]
                    if let container = engine?.getContainer(for: modelId),
                       let eosId = container.tokenizer?.eosTokenId {
                        eosTokenIds.insert(eosId)
                    }

                    let remoteSamplingEnabled = shardEngines.count > 1 && shardEngines.last?.policy is RemoteShardPolicy
                    let extraPredictionLayers = 5

                    let speculationCapable = remoteSamplingEnabled && shardEngines.count == 2
                    var speculationEnabled = speculationCapable
                    let adaptiveWindow = 20
                    let disableThreshold = 0.30
                    var recentPredictions: [Bool] = []
                    var correctPredictions = 0
                    var totalPredictions = 0
                    let draftLength = 1

                    // Running text buffer for stop token suffix check
                    var textBuffer = ""

                    /// Yield a single decoded token, return true if should stop
                    func yieldToken(_ tokenId: Int) -> Bool {
                        let tokenText = tokenizer.decode([tokenId])
                        textBuffer += tokenText
                        generatedCount += 1

                        let isEos = eosTokenIds.contains(tokenId)
                        let hitStop = stopTokens.contains(where: { textBuffer.hasSuffix($0) })
                        let atLimit = generatedCount >= maxTokens
                        let shouldStop = isEos || hitStop || atLimit

                        let finishReason: FinishReason? = shouldStop ? (atLimit && !isEos && !hitStop ? .length : .stop) : nil
                        continuation.yield(Token(
                            id: tokenId,
                            text: tokenText,
                            finishReason: finishReason,
                            promptTokens: promptTokens.count
                        ))
                        return shouldStop
                    }

                    if speculationCapable {
                        let coordPolicy = shardEngines[0].policy
                        guard let slicedCoord = coordPolicy as? SlicedForwardPolicy else {
                            throw DistributedInferenceError.shardPlanFailed("Coord policy is not SlicedForwardPolicy")
                        }
                        let workerPolicy = shardEngines[1].policy as! RemoteShardPolicy
                        var currentPosition = promptTokens.count

                        // First token: run head on prefill hidden state (coord has isLast=true)
                        let firstHeadResult = await slicedCoord.computeHeadOnly(activation)
                        guard let firstResult = firstHeadResult else {
                            throw DistributedInferenceError.shardPlanFailed("computeHeadOnly returned nil on prefill output")
                        }
                        let firstToken = firstResult.tokenId
                        if !eosTokenIds.contains(firstToken) {
                            if yieldToken(firstToken) {
                                if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                                continuation.finish()
                                return
                            }
                        } else {
                            if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                            continuation.finish()
                            return
                        }
                        // Compute coord activation for first token (embedding + coord layers → hidden for worker)
                        activation = try await coordPolicy.compute(input: MLXArray(Int32(firstToken)))
                        currentPosition += 1

                        // Continuous async pipeline decode loop
                        while generatedCount < maxTokens {
                            if speculationEnabled {
                                // Step 1: Fire worker compute async (returns hidden state)
                                let activationBox = SendableBox(activation)
                                let workerTask = Task {
                                    SendableBox(try await workerPolicy.compute(input: activationBox.value))
                                }

                                // Step 2: Draft token N+1 while worker is busy
                                let draft = try? await slicedCoord.draftTokens(
                                    from: activation, count: draftLength, extraPredictionLayers: extraPredictionLayers
                                )

                                // Step 3: Await worker hidden state, run head locally
                                let workerHidden = try await workerTask.value.value
                                guard let headResult = await slicedCoord.computeHeadOnly(workerHidden) else { break }
                                let actualToken = headResult.tokenId

                                // Step 4: Verify draft against actual token
                                if let draft = draft, draft.count > 0, draft.predictedTokens[0] == actualToken {
                                    // HIT — use precomputed activation (zero idle time)
                                    activation = draft.precomputedStates[0]
                                    recordPrediction(true, &recentPredictions, &correctPredictions, &totalPredictions)
                                    currentPosition += 1

                                    if eosTokenIds.contains(actualToken) { break }
                                    if yieldToken(actualToken) { break }

                                    // Verify subsequent draft tokens
                                    for i in 1..<draft.count {
                                        let verifyActivation = SendableBox(activation)
                                        let verifyTask = Task {
                                            SendableBox(try await workerPolicy.compute(input: verifyActivation.value))
                                        }
                                        let verifyHidden = try await verifyTask.value.value
                                        guard let verifyHead = await slicedCoord.computeHeadOnly(verifyHidden) else { break }
                                        let verifyToken = verifyHead.tokenId

                                        if draft.predictedTokens[i] == verifyToken {
                                            activation = draft.precomputedStates[i]
                                            recordPrediction(true, &recentPredictions, &correctPredictions, &totalPredictions)
                                            currentPosition += 1

                                            if eosTokenIds.contains(verifyToken) { break }
                                            if yieldToken(verifyToken) { break }
                                        } else {
                                            recordPrediction(false, &recentPredictions, &correctPredictions, &totalPredictions)
                                            try? await slicedCoord.rollbackCache(
                                                position: currentPosition,
                                                speculatedCount: draft.count - i,
                                                mambaSnapshot: draft.mambaSnapshot
                                            )
                                            activation = try await coordPolicy.compute(input: MLXArray(Int32(verifyToken)))
                                            currentPosition += 1

                                            if eosTokenIds.contains(verifyToken) { break }
                                            if yieldToken(verifyToken) { break }
                                            break
                                        }
                                    }
                                } else {
                                    // MISS — rollback draft cache and recompute
                                    recordPrediction(false, &recentPredictions, &correctPredictions, &totalPredictions)
                                    if let draft = draft {
                                        try? await slicedCoord.rollbackCache(
                                            position: currentPosition,
                                            speculatedCount: draft.count,
                                            mambaSnapshot: draft.mambaSnapshot
                                        )
                                    }
                                    activation = try await coordPolicy.compute(input: MLXArray(Int32(actualToken)))
                                    currentPosition += 1

                                    if eosTokenIds.contains(actualToken) { break }
                                    if yieldToken(actualToken) { break }
                                }

                                // Adaptive accuracy check
                                if recentPredictions.count >= adaptiveWindow {
                                    let rolling = Double(recentPredictions.filter { $0 }.count) / Double(recentPredictions.count)
                                    if rolling < disableThreshold {
                                        speculationEnabled = false
                                    }
                                }
                            } else {
                                // Sequential path — worker returns hidden state → coord runs head
                                let workerHidden = try await workerPolicy.compute(input: activation)
                                guard let headResult = await slicedCoord.computeHeadOnly(workerHidden) else { break }
                                let sampledId = headResult.tokenId
                                if eosTokenIds.contains(sampledId) { break }
                                if yieldToken(sampledId) { break }
                                activation = try await coordPolicy.compute(input: MLXArray(Int32(sampledId)))
                                currentPosition += 1
                            }
                        }
                    } else {
                        // Non-speculative decode
                        for i in 0..<maxTokens {
                            let sampledId: Int
                            if i == 0 {
                                // First token from prefill output (coord has head)
                                if let slicedCoord = shardEngines.first?.policy as? SlicedForwardPolicy,
                                   let headResult = await slicedCoord.computeHeadOnly(activation) {
                                    sampledId = headResult.tokenId
                                } else {
                                    sampledId = argmax(activation)
                                }
                            } else if remoteSamplingEnabled {
                                // Worker returns hidden state → coord runs head
                                let workerPolicy = shardEngines.last!.policy as! RemoteShardPolicy
                                let workerHidden = try await workerPolicy.compute(input: activation)
                                if let slicedCoord = shardEngines.first?.policy as? SlicedForwardPolicy,
                                   let headResult = await slicedCoord.computeHeadOnly(workerHidden) {
                                    sampledId = headResult.tokenId
                                } else {
                                    sampledId = argmax(workerHidden)
                                }
                            } else {
                                sampledId = argmax(activation)
                            }

                            if eosTokenIds.contains(sampledId) { break }
                            if yieldToken(sampledId) { break }

                            if remoteSamplingEnabled {
                                activation = try await shardEngines[0].policy.compute(input: MLXArray(Int32(sampledId)))
                            } else {
                                var decodeActivation = MLXArray(Int32(sampledId))
                                for shard in shardEngines {
                                    decodeActivation = try await shard.policy.compute(input: decodeActivation)
                                }
                                activation = decodeActivation
                            }
                        }
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    let tps = generatedCount > 0 && elapsed > 0 ? Double(generatedCount) / elapsed : 0
                    NovaMLXLog.info("[Distributed] Stream completed: \(generatedCount) tokens in \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", tps)) tok/s)")

                    let accuracy = totalPredictions > 0 ? Double(correctPredictions) / Double(totalPredictions) : nil
                    DistributedInferenceRunnerCache.shared.recordStats(DistributedInferenceStats(
                        tokensPerSecond: tps,
                        promptTokens: promptTokens.count,
                        completionTokens: generatedCount,
                        elapsedSeconds: elapsed,
                        speculationAccuracy: accuracy
                    ))

                    if shouldReleaseWeights {
                        for shardEngine in shardEngines {
                            shardEngine.policy.releaseWeights()
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Private helpers

    /// Pipelined prefill: overlaps Coordinator compute (chunk N+1) with Worker compute (chunk N).
    private func pipelinedPrefill(
        tokens: MLXArray,
        coordPolicy: SlicedForwardPolicy,
        workerPolicy: RemoteShardPolicy
    ) async throws -> MLXArray {
        let tokenCount = tokens.dim(0)
        let chunkSize = min(512, tokenCount)
        let numChunks = (tokenCount + chunkSize - 1) / chunkSize

        NovaMLXLog.info("[Distributed] Pipelined prefill: \(tokenCount) tokens, \(numChunks) chunks of \(chunkSize)")

        var hasPendingResult = false
        let startTime = Date()

        for i in 0..<numChunks {
            let start = i * chunkSize
            let end = min(start + chunkSize, tokenCount)
            let chunk = tokens[start..<end]

            // Collect previous Worker result (overlapped with Coordinator compute)
            if hasPendingResult {
                let _ = try workerPolicy.recvResult()
            }

            // Coordinator processes this chunk (embedding + assigned layers)
            let coordOutput = try await coordPolicy.compute(input: chunk)

            // Send to Worker (fire-and-forget until next iteration)
            try workerPolicy.sendCompute(input: coordOutput)
            hasPendingResult = true
        }

        // Collect final Worker result (includes norm + head on last shard)
        let result = try workerPolicy.recvResult()

        let elapsed = Date().timeIntervalSince(startTime)
        NovaMLXLog.info("[Distributed] Pipelined prefill done: \(numChunks) chunks in \(String(format: "%.2f", elapsed))s")

        return result
    }

    private func argmax(_ logits: MLXArray) -> Int {
        argmaxToken(logits)
    }

    private func bytesFormatted(_ bytes: UInt64) -> String {
        String(format: "%.1fGB", Double(bytes) / 1e9)
    }
}

// MARK: - Adaptive Speculation Helpers

private func recordPrediction(
    _ correct: Bool,
    _ recent: inout [Bool],
    _ correctCount: inout Int,
    _ totalCount: inout Int
) {
    recent.append(correct)
    if recent.count > 20 { recent.removeFirst() }
    correctCount += correct ? 1 : 0
    totalCount += 1
}

// MARK: - DistributedInferenceRunnerCache

/// Thread-safe cache for the last computed shard plan, readable by admin API.
public struct DistributedInferenceStats: Codable, Sendable {
    public let tokensPerSecond: Double
    public let promptTokens: Int
    public let completionTokens: Int
    public let elapsedSeconds: Double
    public let speculationAccuracy: Double?
    public let timestamp: Date

    // Per-component timing breakdown (optional — recorded when profiling is active)
    public let coordComputeMs: Double?
    public let workerComputeMs: Double?
    public let transportMs: Double?
    public let headMs: Double?

    public init(tokensPerSecond: Double, promptTokens: Int, completionTokens: Int,
                elapsedSeconds: Double, speculationAccuracy: Double? = nil,
                coordComputeMs: Double? = nil, workerComputeMs: Double? = nil,
                transportMs: Double? = nil, headMs: Double? = nil) {
        self.tokensPerSecond = tokensPerSecond
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.elapsedSeconds = elapsedSeconds
        self.speculationAccuracy = speculationAccuracy
        self.coordComputeMs = coordComputeMs
        self.workerComputeMs = workerComputeMs
        self.transportMs = transportMs
        self.headMs = headMs
        self.timestamp = Date()
    }
}

public final class DistributedInferenceRunnerCache: @unchecked Sendable {
    public static let shared = DistributedInferenceRunnerCache()
    private let lock = NSLock()
    private var _lastPlan: (modelId: String, plan: ShardPlan)?
    private var _lastStats: DistributedInferenceStats?

    public var lastPlan: (modelId: String, plan: ShardPlan)? {
        lock.withLock { _lastPlan }
    }

    public var lastStats: DistributedInferenceStats? {
        lock.withLock { _lastStats }
    }

    public func setPlan(modelId: String, plan: ShardPlan) {
        lock.withLock { _lastPlan = (modelId, plan) }
    }

    public func recordStats(_ stats: DistributedInferenceStats) {
        lock.withLock { _lastStats = stats }
    }

    private init() {}
}
