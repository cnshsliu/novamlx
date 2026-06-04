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
                // 2-node: both nodes get head for remote sampling. 3+: only last.
                let isLast = plan.assignments.count <= 2 ? true : (index == plan.assignments.count - 1)
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
            // Direct pipeline: coordinator (layers only) → worker (layers only)
            activation = promptArray
            for shard in shardEngines {
                if let sliced = shard.policy as? SlicedForwardPolicy {
                    activation = try await sliced.computeLayersOnly(input: activation)
                } else {
                    activation = try await shard.policy.compute(input: activation)
                }
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

        NovaMLXLog.info("[Distributed] Remote sampling: \(remoteSamplingEnabled), pipeline: computeAndSample (speculativeVerify foundation enabled when numDraftTokens > 0)")

        // === REMOTE SAMPLING PIPELINE ===
        // Worker owns the head (isLast=true). Decode loop:
        //   coord computeLayersOnly(token) → activation (~32ms, single GPU op)
        //   worker computeAndSample(activation) → 4-byte token ID (~34ms)
        //   No hidden state transfer for decode — 4 bytes vs ~16KB.
        if remoteSamplingEnabled && shardEngines.count == 2 {
            guard let slicedCoord = shardEngines[0].policy as? SlicedForwardPolicy else {
                throw DistributedInferenceError.shardPlanFailed("Coord policy is not SlicedForwardPolicy")
            }
            let workerPolicy = shardEngines[1].policy as! RemoteShardPolicy

            // First token: activation is logits from worker (worker has head)
            let firstToken = argmax(activation)
            if !eosTokenIds.contains(firstToken) {
                generatedTokenIds.append(firstToken)
            } else {
                if shouldReleaseWeights { for s in shardEngines { s.policy.releaseWeights() } }
                return InferenceResult(id: request.id, model: modelId, text: "", tokensPerSecond: 0, promptTokens: promptTokens.count, completionTokens: 0, finishReason: .stop)
            }

            // Convert logits → coordinator hidden state for the first decoded token.
            // draftTokens() expects hidden states, not logits.
            activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(firstToken)))

            let decodeStart = CFAbsoluteTimeGetCurrent()
            var timingLogCounter = 0
            var actualToken = firstToken

            // === Speculative Decode State (A next) ===
            let numDraft = request.numDraftTokens ?? 0
            let useSpeculative = numDraft > 0
            var workerCacheOffset = 0   // Approximate logical cache position on worker (for rollback)
            var totalProposed: Int = 0
            var totalAccepted: Int = 0

            // 2.3 Overlap prototype (v2): 真正的 Compute/Communication Overlap
            // 核心思想：在 Worker 执行 speculativeVerify 的同时，Coordinator 立即开始准备下一批 drafts。
            // 注意：现在保存的是 DraftBundle，而不是只保存 token 列表，这样后面可以复用 precomputedStates。
            var nextProposalTask: Task<SlicedForwardPolicy.DraftBundle?, Never>? = nil

            // 用于真正复用 precomputedStates 的变量
            // 当 overlapped proposal 被采用时，我们把 bundle 存到这里，下一轮尝试复用其 hidden state
            var overlappedDraftBundle: SlicedForwardPolicy.DraftBundle? = nil

            // 用于 Coordinator-head 提案跨轮复用 precomputedStates（非 overlapped 也适用）
            var continuationHidden: MLXArray? = nil

            // Overlap 统计（用于观察真实效果）
            var overlappedProposalCount = 0
            var normalProposalCount = 0

            // === 第 1 项打磨：precomputedStates 复用收益统计 ===
            var perfectReuseCount = 0          // 完美命中（整个 batch 直接复用）
            var partialReuseCount = 0          // 部分命中（调用了 forwardRemainingDrafts）
            var continuationReuseCount = 0     // 使用 continuationHidden 跨轮复用
            var totalReusedStates = 0          // 累计复用的 hidden state 数量

            while generatedTokenIds.count < maxTokens {
                let t0 = CFAbsoluteTimeGetCurrent()

                if useSpeculative {
                    let proposalStartTime = CFAbsoluteTimeGetCurrent()
                    // === Real multi-token speculative round (K >= 1) ===
                    var draftTokens: [Int] = []
                    var coordinatorDraftBundle: SlicedForwardPolicy.DraftBundle? = nil
                    var usedOverlappedProposal = false
                    var currentRoundDraftBundle: SlicedForwardPolicy.DraftBundle? = nil

                    // 1. 优先使用上一次在后台准备好的 proposal（overlap 收益点）
                    //    这次我们会保留完整的 DraftBundle，后面会尝试复用 precomputedStates
                    if let task = nextProposalTask {
                        let bundle = await task.value
                        nextProposalTask = nil   // 用完就清掉

                        if let bundle = bundle, !bundle.predictedTokens.isEmpty {
                            currentRoundDraftBundle = bundle
                            draftTokens = bundle.predictedTokens
                            coordinatorDraftBundle = bundle
                            usedOverlappedProposal = true
                            overlappedProposalCount += 1

                            // 关键：保存这个 overlapped bundle，后面尝试复用 precomputedStates
                            overlappedDraftBundle = bundle
                        }
                    }

                    // 2. 如果没有 overlapped 的 proposal，再正常生成（优先 Coordinator-head）
                    if draftTokens.isEmpty {
                        let recentContext = Array(generatedTokenIds.suffix(16)) + [actualToken]
                        if let bundle = try? await slicedCoord.draftTokens(from: activation, count: numDraft) {
                            coordinatorDraftBundle = bundle
                            currentRoundDraftBundle = bundle
                            draftTokens = bundle.predictedTokens
                        } else {
                            draftTokens = engine?.specDecoder.speculate(context: recentContext) ?? []
                        }
                        normalProposalCount += 1
                    }

                    let k = min(numDraft, draftTokens.count)

                    let proposalEndTime = CFAbsoluteTimeGetCurrent()
                    let proposalDurationMs = (proposalEndTime - proposalStartTime) * 1000

                    // 判断这一轮 proposal 是否真正 overlapped（即 proposal 时间主要在 Worker 验证期间发生）
                    let wasOverlapped = usedOverlappedProposal

                    if k == 0 {
                        // No good drafts — fall back to single token this round
                        activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(actualToken)))
                        actualToken = try await workerPolicy.computeAndSample(input: activation)
                        workerCacheOffset += 1

                        if eosTokenIds.contains(actualToken) { break }
                        generatedTokenIds.append(actualToken)
                        // (stop token + timing logging omitted in fallback for brevity)
                        continue
                    }

                    totalProposed += k

                    // === 真正使用 precomputedStates 的核心逻辑（A 方向扩展：跨轮复用）===
                    var activationForBatch: MLXArray

                    if let contHidden = continuationHidden, k > 0 {
                        // 普通 Coordinator-head 提案跨轮复用（非 overlapped 也生效）
                        activationForBatch = try await slicedCoord.forwardRemainingDrafts(
                            startHidden: contHidden,
                            remainingTokens: Array(draftTokens.prefix(k))
                        )
                        continuationHidden = nil
                        continuationReuseCount += 1
                        totalReusedStates += k   // 粗略统计：本次 batch 全靠 continuation
                        NovaMLXLog.info("[Overlap] Used continuationHidden from previous Coordinator-head proposal")
                    } else if let bundle = overlappedDraftBundle,
                       k > 0,
                       !bundle.precomputedStates.isEmpty {

                        // 计算当前 drafts 和 bundle 里 predictedTokens 的最长公共前缀
                        let maxPossible = min(k, bundle.predictedTokens.count)
                        var matchLen = 0

                        for i in 0..<maxPossible {
                            if draftTokens[i] == bundle.predictedTokens[i] {
                                matchLen += 1
                            } else {
                                break
                            }
                        }

                        if matchLen > 0 {
                            // 可以复用前 matchLen 个 precomputed states

                            if matchLen == k {
                                // 完美命中整个 batch，直接用最后一个 precomputed state
                                activationForBatch = bundle.precomputedStates[matchLen - 1]
                                perfectReuseCount += 1
                                totalReusedStates += matchLen
                                NovaMLXLog.info("[Overlap] Full precomputedStates reuse: \(matchLen)/\(k)")
                            } else {
                                // 部分命中：真正实现增量 forward，只计算剩余的 drafts
                                let startHidden = bundle.precomputedStates[matchLen - 1]
                                let remainingDrafts = Array(draftTokens.prefix(k).suffix(from: matchLen))

                                activationForBatch = try await slicedCoord.forwardRemainingDrafts(
                                    startHidden: startHidden,
                                    remainingTokens: remainingDrafts
                                )

                                partialReuseCount += 1
                                totalReusedStates += matchLen
                                NovaMLXLog.info("[Overlap] Partial precomputedStates reuse: forwarded only \(remainingDrafts.count) remaining drafts (saved \(matchLen))")
                            }

                            NovaMLXLog.info("[Overlap] Reused \(matchLen)/\(k) precomputedStates from DraftBundle")
                        } else {
                            // 完全对不上，正常计算
                            let sequenceForCoord = [actualToken] + Array(draftTokens.prefix(k))
                            let seqInput = MLXArray(sequenceForCoord.map { Int32($0) })
                            activationForBatch = try await slicedCoord.computeLayersOnly(input: seqInput)
                        }
                    } else {
                        // 没有可复用的 overlapped bundle，正常路径
                        let sequenceForCoord = [actualToken] + Array(draftTokens.prefix(k))
                        let seqInput = MLXArray(sequenceForCoord.map { Int32($0) })
                        activationForBatch = try await slicedCoord.computeLayersOnly(input: seqInput)
                    }

                    activation = activationForBatch

                    // 用完后清空，避免下次误用
                    overlappedDraftBundle = nil

                    // 用 SendableBox 安全捕获 activation，供两个 Task 使用
                    let activationBox = SendableBox(activation)
                    let numDraftForNext = numDraft
                    let slicedCoordBox = SendableBox(slicedCoord)

                    // 4. 启动 Worker verification（放入 Task，以便 overlap）
                    let verifyTask = Task {
                        try await workerPolicy.speculativeVerify(input: activationBox.value)
                    }

                    // === 2.3 真正的 Overlap 开始 ===
                    // 在 Worker 验证当前 batch 的同时，Coordinator 立即开始准备下一批 drafts
                    // 这次我们保存完整的 DraftBundle（包含 precomputedStates 和 mambaSnapshot）
                    let pendingNextProposalTask = Task<SlicedForwardPolicy.DraftBundle?, Never> { @Sendable in
                        if let bundle = try? await slicedCoordBox.value.draftTokens(from: activationBox.value, count: numDraftForNext) {
                            return bundle
                        } else {
                            // n-gram fallback 时返回 nil bundle
                            return nil
                        }
                    }

                    // 等待 Worker 完成 verification
                    let verifiedTokens = try await verifyTask.value

                    // 5. Compute acceptance: how many of the drafted tokens match the big model's argmax?
                    var accepted = 0
                    for i in 0..<k {
                        let draftT = draftTokens[i]
                        // verifiedTokens[0] corresponds to the original actualToken position
                        // verifiedTokens[1..] correspond to the drafted positions
                        let verifiedT = (i + 1 < verifiedTokens.count) ? verifiedTokens[i + 1] : -1
                        if verifiedT == draftT {
                            accepted += 1
                        } else {
                            break
                        }
                    }

                    totalAccepted += accepted

                    // 6. Append accepted tokens
                    if accepted > 0 {
                        let newlyAccepted = Array(draftTokens.prefix(accepted))
                        generatedTokenIds.append(contentsOf: newlyAccepted)
                        actualToken = newlyAccepted.last ?? actualToken

                        // 如果本轮是 Coordinator-head 提案，保存接受后的 hidden state，用于下一轮增量 forward（A 方向）
                        if coordinatorDraftBundle != nil {
                            continuationHidden = coordinatorDraftBundle?.precomputedStates[accepted - 1]
                        }
                    }

                    // 7. Handle rejection + bonus token
                    let rejected = k - accepted
                    if rejected > 0 {
                        // Worker advanced its KV by (1 + k) during the speculativeVerify forward.
                        // We only want to keep the accepted portion.
                        // Rollback the excess rejected tokens on the worker.
                        let excess = rejected
                        if excess > 0 {
                            let targetPosition = max(0, workerCacheOffset + accepted + 1 - excess)
                            try? await workerPolicy.rollbackCache(position: targetPosition)
                        }

                        // Rejection 发生后，Coordinator 侧也尝试用 DraftBundle 的 mambaSnapshot 做 rollback
                        if let bundle = currentRoundDraftBundle {
                            try? await slicedCoord.rollbackCache(
                                position: workerCacheOffset + accepted + 1 - excess,
                                speculatedCount: excess,
                                mambaSnapshot: bundle.mambaSnapshot
                            )
                        }

                        // Rejection 发生后，之前启动的 next proposal 很可能基于旧 prefix，丢弃它
                        nextProposalTask = nil
                        continuationHidden = nil   // prefix 变了，跨轮复用的 hidden 也失效

                        // Use the verified token at the rejection point as the "bonus" real token
                        let rejectionIndex = accepted + 1
                        if rejectionIndex < verifiedTokens.count {
                            let bonusToken = verifiedTokens[rejectionIndex]
                            if !eosTokenIds.contains(bonusToken) && generatedTokenIds.count < maxTokens {
                                generatedTokenIds.append(bonusToken)
                                actualToken = bonusToken
                            }
                        }
                    } else {
                        // All drafts accepted — the last verified token is a bonus from the big model
                        if verifiedTokens.count > k {
                            let bonus = verifiedTokens.last ?? verifiedTokens[k]
                            if !eosTokenIds.contains(bonus) && generatedTokenIds.count < maxTokens {
                                generatedTokenIds.append(bonus)
                                actualToken = bonus
                            }
                        }
                    }

                    // Update worker cache tracking
                    workerCacheOffset += (1 + k)

                    // Record acceptance (differentiate source for future A/B)
                    if k > 0 {
                        _ = (coordinatorDraftBundle != nil) ? "coord-head" : "ngram"
                        engine?.specDecoder.recordAccepted(tokens: Array(draftTokens.prefix(k)), accepted: accepted)
                        // TODO: later expose per-source acceptance rate in stats
                    }

                    // 把刚启动的下一个 proposal task 保存下来，供下一轮使用（实现 overlap）
                    nextProposalTask = pendingNextProposalTask

                    // Rich speculative round log
                    let effective = accepted + (rejected > 0 ? 1 : 0) // +1 for potential bonus on rejection
                    let overlapTag = usedOverlappedProposal ? " [overlap]" : ""
                    NovaMLXLog.info("[Spec] round: proposed=\(k) accepted=\(accepted) effective=\(effective) tok/round (acc=\(String(format: "%.1f", totalProposed > 0 ? Double(totalAccepted)/Double(totalProposed)*100 : 0))%)\(overlapTag) | proposal=\(String(format: "%.1f", proposalDurationMs))ms overlapped=\(wasOverlapped)")

                    // 每 20 轮输出一次 overlap 统计
                    if (overlappedProposalCount + normalProposalCount) % 20 == 0 && (overlappedProposalCount + normalProposalCount) > 0 {
                        let total = overlappedProposalCount + normalProposalCount
                        let rate = Double(overlappedProposalCount) / Double(total) * 100
                        NovaMLXLog.info("[Overlap] stats: overlapped=\(overlappedProposalCount)/\(total) (\(String(format: "%.1f", rate))%)")
                    }

                    // === precomputedStates 复用收益统计（每 20 轮）===
                    let reuseTotal = perfectReuseCount + partialReuseCount + continuationReuseCount
                    if reuseTotal > 0 && reuseTotal % 20 == 0 {
                        let avgSaved = Double(totalReusedStates) / Double(reuseTotal)
                        NovaMLXLog.info("[Reuse] precomputedStates: perfect=\(perfectReuseCount), partial=\(partialReuseCount), continuation=\(continuationReuseCount), avgSavedStates=\(String(format: "%.1f", avgSaved))")
                    }

                    // Early exit checks
                    if eosTokenIds.contains(actualToken) { break }
                    if !stopTokens.isEmpty {
                        let tailIds = Array(generatedTokenIds.suffix(10))
                        let tailText = tokenizer.decode(tailIds)
                        if stopTokens.contains(where: { tailText.hasSuffix($0) }) { break }
                    }

                    // nextProposalTask 已经在 speculativeVerify 启动后被赋值（pendingNextProposalTask）
                    // 这里不需要重复启动

                } else {
                    // === Classic single-token path (baseline) ===
                    activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(actualToken)))
                    let tAfterCoord = CFAbsoluteTimeGetCurrent()
                    actualToken = try await workerPolicy.computeAndSample(input: activation)
                    let tTotal = CFAbsoluteTimeGetCurrent()
                    workerCacheOffset += 1

                    if eosTokenIds.contains(actualToken) { break }
                    generatedTokenIds.append(actualToken)

                    // Stop token check
                    if !stopTokens.isEmpty {
                        let tailIds = Array(generatedTokenIds.suffix(10))
                        let tailText = tokenizer.decode(tailIds)
                        if stopTokens.contains(where: { tailText.hasSuffix($0) }) { break }
                    }

                    // Timing log every 20 tokens (baseline path)
                    timingLogCounter += 1
                    if timingLogCounter % 20 == 1 {
                        let totalMs = (tTotal - t0) * 1000
                        let coordMs = (tAfterCoord - t0) * 1000
                        let workerMs = (tTotal - tAfterCoord) * 1000
                        NovaMLXLog.info("[Pipeline] token \(generatedTokenIds.count): \(String(format: "%.1f", totalMs))ms coord=\(String(format: "%.1f", coordMs))ms worker=\(String(format: "%.1f", workerMs))ms")
                    }
                }
            }

            let decodeElapsed = CFAbsoluteTimeGetCurrent() - decodeStart
            let decodeTps = Double(generatedTokenIds.count) / decodeElapsed
            NovaMLXLog.info("[Distributed] Pipeline done: \(generatedTokenIds.count) tokens, \(String(format: "%.1f", decodeTps)) tok/s")
        } else {
            // Single-node or multi-shard fallback
            for i in 0..<maxTokens {
                let sampledId: Int
                if i == 0 {
                    sampledId = argmax(activation)
                } else if remoteSamplingEnabled {
                    let workerPolicy = shardEngines.last!.policy as! RemoteShardPolicy
                    let workerHidden = try await workerPolicy.compute(input: activation)
                    sampledId = argmax(workerHidden)
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

        // Record final stats. speculationAccuracy will be populated once the speculative decode loop
        // (using draftTokens + speculativeVerify + rollbackCache) is wired in the remote-sampling path.
        let accuracy: Double? = engine?.specDecoder.acceptanceRate
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
                            let isLast = plan.assignments.count <= 2 ? true : (index == plan.assignments.count - 1)
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
                            if let sliced = shard.policy as? SlicedForwardPolicy {
                                activation = try await sliced.computeLayersOnly(input: activation)
                            } else {
                                activation = try await shard.policy.compute(input: activation)
                            }
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

                    // === REMOTE SAMPLING PIPELINE (streaming) ===
                    if remoteSamplingEnabled && shardEngines.count == 2 {
                        guard let slicedCoord = shardEngines[0].policy as? SlicedForwardPolicy else {
                            throw DistributedInferenceError.shardPlanFailed("Coord policy is not SlicedForwardPolicy")
                        }
                        let workerPolicy = shardEngines[1].policy as! RemoteShardPolicy

                        // First token: activation is logits from worker (worker has head)
                        let firstToken = argmax(activation)
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

                        // Convert logits → coordinator hidden state for draftTokens()
                        activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(firstToken)))

                        var timingLogCounter = 0
                        var actualToken = firstToken

                        // === Speculative Decode State (streaming) ===
                        let numDraft = request.numDraftTokens ?? 0
                        let useSpeculative = numDraft > 0
                        var workerCacheOffset = 0
                        var totalProposed: Int = 0
                        var totalAccepted: Int = 0
                        var recentTokensForNgram: [Int] = []   // maintained for streaming ngram context

                        while generatedCount < maxTokens {
                            let t0 = CFAbsoluteTimeGetCurrent()

                            if useSpeculative {
                                // === Full K>1 speculative round for streaming ===
                                // Prefer Coordinator-head drafts when possible (strong node)
                                let contextForDraft = Array(recentTokensForNgram.suffix(16)) + [actualToken]
                                var draftTokens: [Int] = []
                                var currentRoundDraftBundle: SlicedForwardPolicy.DraftBundle? = nil

                                if let bundle = try? await slicedCoord.draftTokens(from: activation, count: numDraft) {
                                    currentRoundDraftBundle = bundle
                                    draftTokens = bundle.predictedTokens
                                } else {
                                    draftTokens = engine?.specDecoder.speculate(context: contextForDraft) ?? []
                                }

                                // 标记使用（避免 warning），实际在 rejection 时会用到
                                _ = currentRoundDraftBundle

                                let k = min(numDraft, draftTokens.count)

                                if k == 0 {
                                    activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(actualToken)))
                                    actualToken = try await workerPolicy.computeAndSample(input: activation)
                                    workerCacheOffset += 1
                                    if eosTokenIds.contains(actualToken) { break }
                                    if yieldToken(actualToken) { break }
                                    recentTokensForNgram.append(actualToken)
                                    if recentTokensForNgram.count > 32 { recentTokensForNgram.removeFirst() }
                                    continue
                                }

                                totalProposed += k

                                let sequenceForCoord = [actualToken] + Array(draftTokens.prefix(k))
                                let seqInput = MLXArray(sequenceForCoord.map { Int32($0) })
                                activation = try await slicedCoord.computeLayersOnly(input: seqInput)

                                let verifiedTokens = try await workerPolicy.speculativeVerify(input: activation)

                                var accepted = 0
                                for i in 0..<k {
                                    let draftT = draftTokens[i]
                                    let verifiedT = (i + 1 < verifiedTokens.count) ? verifiedTokens[i + 1] : -1
                                    if verifiedT == draftT { accepted += 1 } else { break }
                                }
                                totalAccepted += accepted

                                // Yield accepted tokens one by one for streaming
                                var shouldStop = false
                                for i in 0..<accepted {
                                    let tok = draftTokens[i]
                                    recentTokensForNgram.append(tok)
                                    if recentTokensForNgram.count > 32 { recentTokensForNgram.removeFirst() }
                                    if eosTokenIds.contains(tok) { shouldStop = true; break }
                                    if yieldToken(tok) { shouldStop = true; break }
                                }

                                let rejected = k - accepted
                                if rejected > 0 {
                                    let excess = rejected
                                    let targetPosition = max(0, workerCacheOffset + accepted + 1 - excess)
                                    try? await workerPolicy.rollbackCache(position: targetPosition)

                                    // Streaming 也支持用 DraftBundle 的 mambaSnapshot 做 Coordinator rollback
                                    if let bundle = currentRoundDraftBundle {
                                        try? await slicedCoord.rollbackCache(
                                            position: targetPosition,
                                            speculatedCount: excess,
                                            mambaSnapshot: bundle.mambaSnapshot
                                        )
                                    }

                                    if !shouldStop {
                                        let rejectionIndex = accepted + 1
                                        if rejectionIndex < verifiedTokens.count {
                                            let bonus = verifiedTokens[rejectionIndex]
                                            if !eosTokenIds.contains(bonus) {
                                                recentTokensForNgram.append(bonus)
                                                if recentTokensForNgram.count > 32 { recentTokensForNgram.removeFirst() }
                                                if yieldToken(bonus) { shouldStop = true }
                                            }
                                        }
                                    }
                                } else if !shouldStop {
                                    if verifiedTokens.count > k {
                                        let bonus = verifiedTokens.last ?? verifiedTokens[k]
                                        if !eosTokenIds.contains(bonus) {
                                            recentTokensForNgram.append(bonus)
                                            if recentTokensForNgram.count > 32 { recentTokensForNgram.removeFirst() }
                                            if yieldToken(bonus) { shouldStop = true }
                                        }
                                    }
                                }

                                workerCacheOffset += (1 + k)
                                if k > 0 {
                                    engine?.specDecoder.recordAccepted(tokens: Array(draftTokens.prefix(k)), accepted: accepted)
                                }

                                NovaMLXLog.info("[Spec-Stream] round: proposed=\(k) accepted=\(accepted) (acc=\(String(format: "%.1f", totalProposed > 0 ? Double(totalAccepted)/Double(totalProposed)*100 : 0))%)")

                                if shouldStop { break }

                            } else {
                                // Baseline single token streaming path
                                activation = try await slicedCoord.computeLayersOnly(input: MLXArray(Int32(actualToken)))
                                let tCoordDone = CFAbsoluteTimeGetCurrent()
                                actualToken = try await workerPolicy.computeAndSample(input: activation)
                                let tTotal = CFAbsoluteTimeGetCurrent()
                                workerCacheOffset += 1

                                recentTokensForNgram.append(actualToken)
                                if recentTokensForNgram.count > 32 { recentTokensForNgram.removeFirst() }

                                if eosTokenIds.contains(actualToken) { break }
                                if yieldToken(actualToken) { break }

                                timingLogCounter += 1
                                if timingLogCounter % 20 == 1 {
                                    let totalMs = (tTotal - t0) * 1000
                                    let coordMs = (tCoordDone - t0) * 1000
                                    let workerMs = (tTotal - tCoordDone) * 1000
                                    NovaMLXLog.info("[Pipeline-Stream] token \(generatedCount): \(String(format: "%.1f", totalMs))ms coord=\(String(format: "%.1f", coordMs))ms worker=\(String(format: "%.1f", workerMs))ms")
                                }
                            }
                        }
                    } else {
                        // Single-node or multi-shard fallback
                        for i in 0..<maxTokens {
                            let sampledId: Int
                            if i == 0 {
                                sampledId = argmax(activation)
                            } else if remoteSamplingEnabled {
                                let workerPolicy = shardEngines.last!.policy as! RemoteShardPolicy
                                let workerHidden = try await workerPolicy.compute(input: activation)
                                sampledId = argmax(workerHidden)
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

                    let accuracy: Double? = nil
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
                NovaMLXLog.info("[Distributed] Prefill chunk \(i): collecting previous worker result...")
                let _ = try workerPolicy.recvResult()
                NovaMLXLog.info("[Distributed] Prefill chunk \(i): previous result received")
            }

            // Coordinator processes this chunk (embedding + layers, skip head so we send hidden state to worker)
            let coordOutput = try await coordPolicy.computeLayersOnly(input: chunk)
            NovaMLXLog.info("[Distributed] Prefill chunk \(i): coord output shape=\(coordOutput.shape) dtype=\(coordOutput.dtype)")

            // Send to Worker (fire-and-forget until next iteration)
            try workerPolicy.sendCompute(input: coordOutput)
            NovaMLXLog.info("[Distributed] Prefill chunk \(i): sent to worker")
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
    public let workerWaitMs: Double?
    public let overlapPct: Double?

    public init(tokensPerSecond: Double, promptTokens: Int, completionTokens: Int,
                elapsedSeconds: Double, speculationAccuracy: Double? = nil,
                coordComputeMs: Double? = nil, workerComputeMs: Double? = nil,
                transportMs: Double? = nil, headMs: Double? = nil,
                workerWaitMs: Double? = nil, overlapPct: Double? = nil) {
        self.tokensPerSecond = tokensPerSecond
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.elapsedSeconds = elapsedSeconds
        self.speculationAccuracy = speculationAccuracy
        self.coordComputeMs = coordComputeMs
        self.workerComputeMs = workerComputeMs
        self.transportMs = transportMs
        self.headMs = headMs
        self.workerWaitMs = workerWaitMs
        self.overlapPct = overlapPct
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
