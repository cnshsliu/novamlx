import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils
import NovaMLXEngine
import NovaMLXModelManager
import NovaMLXDistributed
import AsyncAlgorithms

extension Notification.Name {
    /// Posted with the backbone model id when load needs a yes/no on unknown boosters.
    public static let novaMLXAcceleratorPrompt = Notification.Name("novaMLXAcceleratorPrompt")
}

/// Per-model load dedup. Concurrent callers for the same modelId await a
/// single underlying task. Cleanup is synchronous in the actor's isolation
/// context so a failed load releases its slot before the error propagates,
/// allowing immediate retry by piled-up callers.
actor LoadDedup {
    private var inFlight: [String: Task<Void, Error>] = [:]

    /// Number of in-flight loads (test/observability hook).
    var inFlightCount: Int { inFlight.count }

    func ensureSingle(
        modelId: String,
        work: @Sendable @escaping () async throws -> Void
    ) async throws {
        if let existing = inFlight[modelId] {
            try await existing.value
            return
        }
        let task = Task<Void, Error> { try await work() }
        inFlight[modelId] = task
        defer { inFlight.removeValue(forKey: modelId) }
        try await task.value
    }
}

public final class InferenceService: @unchecked Sendable {
    public let engine: MLXEngine
    private let batcher: ContinuousBatcher
    private let fusedScheduler: FusedBatchScheduler
    public let settingsManager: ModelSettingsManager
    public let transcriptionService: TranscriptionService
    public let ttsService: TTSService
    public let imageGenerationService: ImageGenerationService
    /// Async image jobs. Polled on `GET /v1/tasks/{id}` together with video jobs.
    public let imageTaskService: ImageTaskService
    /// Hunyuan3D image-to-3D. Serial queue, answers on `/v1/tasks` as
    /// `3d.generation` plus the `/v1/3d/generations` aliases.
    public let hy3DService: Hy3DService
    public let talkingHeadService: TalkingHeadService
    public let h3VideoService: H3VideoService
    public let decisionService: DecisionService

    // Worker subprocess mode
    public let workerMode: Bool
    private var worker: WorkerSupervisor?
    private var workerLoadedModels: Set<String> = []
    private var workerModelTypes: [String: ModelType] = [:]
    private var workerHybridModels: Set<String> = []
    private var workerNativeMtpModels: Set<String> = []
    private let loadDedup = LoadDedup()
    private let httpLock = NovaMLXLock()
    private var httpToEngine: [String: UUID] = [:]
    private var ttlSweepTask: Task<Void, Never>?
    private var acceleratorDownloads: ModelManager?
    private let acceleratorLock = NovaMLXLock()
    private var acceleratorPrompts: [String: AcceleratorPrompt] = [:]

    // Cluster distributed inference mode
    private let clusterMode: Bool
    private var distributedRunner: DistributedInferenceRunner?

    public init(engine: MLXEngine, settingsManager: ModelSettingsManager, maxBatchSize: Int = 8, workerMode: Bool = false, workerBinaryPath: String? = nil, clusterMode: Bool = false, clusterConfig: ClusterConfig? = nil) {
        self.engine = engine
        self.batcher = ContinuousBatcher(engine: engine, maxBatchSize: maxBatchSize)
        self.fusedScheduler = FusedBatchScheduler(engine: engine)
        self.settingsManager = settingsManager
        self.transcriptionService = TranscriptionService()
        self.ttsService = TTSService()
        self.imageGenerationService = ImageGenerationService()
        self.imageTaskService = ImageTaskService(images: imageGenerationService)
        self.hy3DService = Hy3DService()
        self.talkingHeadService = TalkingHeadService()
        self.h3VideoService = H3VideoService()
        self.decisionService = DecisionService()
        // Share the engine's metrics store with every backend so the status
        // panel's "Realtime Inference Speed" reflects ALL model types, not just LLM.
        self.transcriptionService.metricsStore = engine.metricsStore
        self.ttsService.metricsStore = engine.metricsStore
        self.imageGenerationService.metricsStore = engine.metricsStore
        self.talkingHeadService.metricsStore = engine.metricsStore
        self.h3VideoService.metricsStore = engine.metricsStore
        self.workerMode = workerMode
        self.clusterMode = clusterMode

        if workerMode, let path = workerBinaryPath {
            self.worker = WorkerSupervisor(workerBinaryPath: path)
        }

        if clusterMode, let config = clusterConfig {
            self.distributedRunner = DistributedInferenceRunner(
                clusterConfig: config,
                tokenizerProvider: { [weak self] modelId in
                    guard let self = self else { return nil }
                    let container = self.engine.getContainer(for: modelId)
                    guard let tokenizer = container?.tokenizer else { return nil }
                    return DistributedTokenizer(
                        encode: { text in tokenizer.encode(text) },
                        decode: { tokens in tokenizer.decode(tokens) }
                    )
                },
                modelPathProvider: { modelId in
                    let path = NovaMLXPaths.directory(forModelId:modelId).path
                    var isDir: ObjCBool = false
                    guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
                        return nil
                    }
                    return path
                },
                engine: engine
            )
        }

        engine.settingsProvider = { [settingsManager] modelId in
            settingsManager.getSettings(modelId)
        }

        // TTL sweep: auto-evict models that exceed their idle TTL every 60s
        ttlSweepTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(60))
                guard let self = self else { return }
                self.checkTTLExpirations()
            }
        }
    }

    // MARK: - Worker Lifecycle

    public func startWorker() throws {
        guard workerMode, let worker = worker else { return }
        worker.onCrash = { [weak self] in
            guard let self = self else { return }
            NovaMLXLog.warning("[InferenceService] Worker crashed — clearing in-memory loaded models state")
            self.workerLoadedModels.removeAll()
            self.workerModelTypes.removeAll()
            self.workerHybridModels.removeAll()
            self.workerNativeMtpModels.removeAll()
            // Do NOT saveLoadedModelsList() here — on crash/exit, the persisted file
            // should keep the model list so restoreModels() can reload on next launch.
            // The terminationHandler fires on normal app exit too, which would wipe the list.
        }
        try worker.start()
        NovaMLXLog.info("[InferenceService] Worker mode started")
    }

    public func stopWorker() {
        worker?.stop()
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        let resolvedId = settingsManager.resolveModelId(request.model)
        try Self.rejectMtpAsChat(resolvedId)
        let settings = settingsManager.getSettings(resolvedId)
        var finalRequest = settings.applySamplingOverrides(to: request)
        finalRequest = finalRequest.replacing(model: resolvedId)
        finalRequest = autoInjectDraftModel(finalRequest)
        track(finalRequest)
        defer { untrack(finalRequest) }

        // Cluster mode: check readiness, then route to distributed inference
        if clusterMode, let runner = distributedRunner {
            let modelManager = ClusterModelManager.shared
            let clusterState = modelManager.getStatus()

            switch clusterState.state {
            case .idle:
                // No model activated for distributed — fall through to local inference
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster idle, using local inference")
            case .activating:
                let (ready, total) = clusterState.readinessFraction
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Rejected: cluster preparing (\(ready)/\(total) nodes ready)")
                throw DistributedInferenceError.shardPlanFailed("Cluster is preparing model '\(clusterState.activeModel ?? "?")' (\(ready)/\(total) nodes ready). Please wait.")
            case ClusterModelState.failed:
                // Failed — fall through to local inference
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster failed, using local inference")
            case .ready:
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] -> Distributed (model=\(resolvedId))")
                return try await runner.generate(request: finalRequest)
            }
        }

        // Worker mode: route through subprocess
        if workerMode, let worker = worker {
            let result = try await worker.sendGenerate(finalRequest)
            if result.completionTokens > 0 {
                engine.metricsStore.recordRequest(
                    model: resolvedId,
                    tokens: UInt64(result.completionTokens),
                    inferenceTime: result.tokensPerSecond > 0 ? Double(result.completionTokens) / result.tokensPerSecond : 0
                )
            }
            return result
        }

        let reqTag = finalRequest.id.uuidString.prefix(8)

        // Context window check moved to engine/scheduler layer after tokenization,
        // where we have accurate token counts (includes chat template tokens).
        let container = engine.getContainer(for: resolvedId)

        // VLM models use ContinuousBatcher — they have complex internal position state
        // (3D mRoPE, precomputedPositionIds, ropeDeltas) that our fused decode step
        // can't replicate with a simple model(token, cache:) call.
        let isVLM = container?.config.modelType == .vlm || workerModelTypes[resolvedId] == .vlm

        // Hybrid linear attention models (e.g. Qwen3.5) mix MambaCache + KVCacheSimple layers.
        // FusedBatchScheduler only supports KVCacheSimple — hybrid models must use ContinuousBatcher.
        let hasLinearAttention = container?.config.hasLinearAttention == true
        let nativeMtp = hasNativeMtp(resolvedId)

        // Session, grammar, VLM, hybrid, native MTP head, and draft-model paths use
        // engine TokenIterator. Qwen companion MTP uses MLXLMCommon verify.
        let hasDraftModel = finalRequest.draftModel != nil
        let needsSpecialized = finalRequest.sessionId != nil ||
            finalRequest.jsonSchemaDef != nil ||
            finalRequest.responseFormat == .jsonObject ||
            finalRequest.regexPattern != nil ||
            finalRequest.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention ||
            hasDraftModel ||
            nativeMtp

        if needsSpecialized {
            let reason = isVLM ? "VLM" : (hasLinearAttention ? "hybrid" : (hasDraftModel ? "draft-model" : (nativeMtp ? "native-mtp" : (finalRequest.sessionId != nil ? "session" : "grammar"))))
            NovaMLXLog.info("[Route:\(reqTag)] → ContinuousBatcher (reason=\(reason), model=\(resolvedId))")
            return try await batcher.submit(finalRequest)
        }

        // LLM standard path. One sequence stays on the serial loop (Flint
        // packet-16, n-gram verify). Concurrent KVCacheSimple sequences at the
        // same offset share one model forward.
        NovaMLXLog.info("[Route:\(reqTag)] → FusedBatchScheduler (model=\(resolvedId))")
        return try await fusedScheduler.submit(finalRequest)
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let resolvedId = settingsManager.resolveModelId(request.model)
        let mtpDir = NovaMLXPaths.directory(forModelId:resolvedId)
        if NovaMLXEngine.isMtpDraftConfig(at: mtpDir) {
            return AsyncThrowingStream { continuation in
                continuation.finish(
                    throwing: NovaMLXError.unsupportedModel(
                        "MTP draft head cannot be used as a chat model: \(resolvedId)"))
            }
        }
        let settings = settingsManager.getSettings(resolvedId)
        var finalRequest = settings.applySamplingOverrides(to: request)
        finalRequest = finalRequest.replacing(model: resolvedId, stream: true)
        finalRequest = autoInjectDraftModel(finalRequest)
        track(finalRequest)

        // Cluster mode: route to distributed streaming when cluster is ready
        if clusterMode, let runner = distributedRunner {
            let modelManager = ClusterModelManager.shared
            let clusterState = modelManager.getStatus()

            switch clusterState.state {
            case .idle:
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster idle, using local inference for stream")
            case .activating:
                let (ready, total) = clusterState.readinessFraction
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Rejected: cluster preparing (\(ready)/\(total) nodes ready)")
                return AsyncThrowingStream { $0.finish(throwing: DistributedInferenceError.shardPlanFailed("Cluster is preparing model '\(clusterState.activeModel ?? "?")' (\(ready)/\(total) nodes ready). Please wait.")) }
            case .failed:
                NovaMLXLog.warning("[Route:\(finalRequest.id.uuidString.prefix(8))] Cluster failed, using local inference for stream")
            case .ready:
                NovaMLXLog.info("[Route:\(finalRequest.id.uuidString.prefix(8))] -> Distributed stream (model=\(resolvedId))")
                let clusterKind: InferenceKind = workerModelTypes[resolvedId] == .vlm ? .vlm : .llm
                return Self.trackStream(runner.stream(request: finalRequest), tracker: StreamTracker(model: resolvedId, kind: clusterKind, metricsStore: engine.metricsStore), request: finalRequest)
            }
        }

        // Worker mode: route through subprocess
        if workerMode, let worker = worker {
            let workerKind: InferenceKind = workerModelTypes[resolvedId] == .vlm ? .vlm : .llm
            let tracker = StreamTracker(model: resolvedId, kind: workerKind, metricsStore: engine.metricsStore)
            let upstream = worker.sendStream(finalRequest)
            let capturedRequest = finalRequest
            return AsyncThrowingStream { continuation in
                let task = Task { @Sendable in
                    do {
                        for try await token in upstream {
                            tracker.increment()
                            continuation.yield(token)
                        }
                        tracker.finish(request: capturedRequest)
                        continuation.finish()
                    } catch {
                        tracker.finish(request: capturedRequest, error: error.localizedDescription)
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { _ in task.cancel() }
            }
        }

        let reqTag = finalRequest.id.uuidString.prefix(8)

        // Context window check moved to engine/scheduler layer after tokenization,
        // where we have accurate token counts (includes chat template tokens).
        let container = engine.getContainer(for: resolvedId)

        // VLM models use ContinuousBatcher — they have complex internal position state
        // (3D mRoPE, precomputedPositionIds, ropeDeltas) that our fused decode step
        // can't replicate with a simple model(token, cache:) call.
        let isVLM = container?.config.modelType == .vlm || workerModelTypes[resolvedId] == .vlm

        // Hybrid linear attention models (e.g. Qwen3.5) mix MambaCache + KVCacheSimple layers.
        // FusedBatchScheduler only supports KVCacheSimple — hybrid models must use ContinuousBatcher.
        let hasLinearAttention = container?.config.hasLinearAttention == true
        let nativeMtp = hasNativeMtp(resolvedId)

        // Session, grammar, VLM, hybrid, native MTP, and draft-model paths use engine directly.
        let hasDraftModel = finalRequest.draftModel != nil
        let localKind: InferenceKind = isVLM ? .vlm : .llm
        let needsSpecialized = finalRequest.sessionId != nil ||
            finalRequest.jsonSchemaDef != nil ||
            finalRequest.responseFormat == .jsonObject ||
            finalRequest.regexPattern != nil ||
            finalRequest.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention ||
            hasDraftModel ||
            nativeMtp

        // Wrap any local stream with a log-tracking shim that finalizes the
        // RequestLogEntry on completion (token count + timing + HTTP request id).
        if needsSpecialized {
            let reason = isVLM ? "VLM" : (hasLinearAttention ? "hybrid" : (hasDraftModel ? "draft-model" : (nativeMtp ? "native-mtp" : (finalRequest.sessionId != nil ? "session" : "grammar"))))
            NovaMLXLog.info("[Route:\(reqTag)] → ContinuousBatcher stream (reason=\(reason), model=\(resolvedId))")
            return Self.trackStream(batcher.submitStream(finalRequest), tracker: StreamTracker(model: resolvedId, kind: localKind, metricsStore: engine.metricsStore), request: finalRequest)
        }

        // LLM standard path. One sequence stays on the serial loop (Flint
        // packet-16, n-gram verify). Concurrent KVCacheSimple sequences at the
        // same offset share one model forward.
        NovaMLXLog.info("[Route:\(reqTag)] → FusedBatchScheduler stream (model=\(resolvedId))")
        return Self.trackStream(fusedScheduler.submitStream(finalRequest), tracker: StreamTracker(model: resolvedId, kind: localKind, metricsStore: engine.metricsStore), request: finalRequest)
    }

    /// Wrap an inbound token stream so the request log gets live updates during
    /// streaming and a final record on completion. Isolates token counting from
    /// the scheduler internals (which don't know the HTTP request id).
    private static func trackStream(
        _ upstream: AsyncThrowingStream<Token, Error>,
        tracker: StreamTracker,
        request: InferenceRequest
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @Sendable in
                do {
                    for try await token in upstream {
                        tracker.increment()
                        continuation.yield(token)
                    }
                    tracker.finish(request: request)
                    continuation.finish()
                } catch {
                    tracker.finish(request: request, error: error.localizedDescription)
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    public func abort(requestId: UUID) async {
        if workerMode, let worker = worker {
            worker.sendAbort(requestId: requestId)
        } else {
            engine.abort(requestId: requestId)
            batcher.abort(requestId: requestId)
            fusedScheduler.abort(requestId: requestId)
        }
    }

    /// Cancel the in-flight request shown in the request log. `id` is the HTTP
    /// request id, not the engine UUID.
    public func abort(httpRequestId: String) async {
        let path = RequestLogStore.shared.path(for: httpRequestId)
        let engineId = httpLock.withLock { httpToEngine.removeValue(forKey: httpRequestId) }
        if let engineId {
            await abort(requestId: engineId)
        }
        if path?.contains("/images/") == true {
            imageGenerationService.requestCancel()
        }
        RequestLogStore.shared.cancel(id: httpRequestId)
    }

    private func track(_ request: InferenceRequest) {
        guard let http = request.httpRequestId, !http.isEmpty else { return }
        httpLock.withLock { httpToEngine[http] = request.id }
    }

    private func untrack(_ request: InferenceRequest) {
        guard let http = request.httpRequestId else { return }
        httpLock.withLock {
            if httpToEngine[http] == request.id {
                httpToEngine.removeValue(forKey: http)
            }
        }
    }

    // MARK: - Auto Speculative Decoding (Speed Boost)

    /// Automatically inject draft model for speculative decoding when a compatible
    /// draft model is loaded. Skipped if user explicitly sets draftModel, or for
    /// hybrid/distributed/cluster requests.
    private func autoInjectDraftModel(_ request: InferenceRequest) -> InferenceRequest {
        // User explicitly specified — respect their choice
        guard request.draftModel == nil else { return request }

        // Skip when model is actively served by distributed runner.
        // When cluster is idle (no model activated), requests fall through to
        // local inference where speculative decoding works fine.
        if clusterMode, distributedRunner != nil {
            let clusterState = ClusterModelManager.shared.getStatus()
            if clusterState.state == .ready, clusterState.activeModel == settingsManager.resolveModelId(request.model) {
                return request
            }
        }

        guard isModelLoaded(request.model) else { return request }

        let mainId = settingsManager.resolveModelId(request.model)
        let family = draftFamily(for: request)
        let scenario = Self.requestScenario(request)
        let remembered = settingsManager.getSettings(mainId).acceleratorChoices ?? [:]
        let candidates = DraftModelRegistry.shared.acceleratorCandidates(
            forMainId: mainId,
            family: family,
            isHybrid: isHybridModel(request.model),
            modelType: .llm
        )
        let plan = AcceleratorSpeedGate.plan(
            mainModelId: mainId,
            scenario: scenario,
            candidates: candidates,
            remembered: remembered
        )
        // One primary path. DFlash2, then MTP, then DSpark, then a small draft.
        for candidate in plan.attach {
            let blockKind = candidate.kind == .dflash2 || candidate.kind == .mtp || candidate.kind == .dspark
            if blockKind && !request.allowsMtp {
                NovaMLXLog.info("[SpecBoost] Skipping \(candidate.kind.rawValue) auto-inject for '\(mainId)' (MTP/DFlash off)")
                continue
            }
            guard isModelLoaded(candidate.acceleratorId) else { continue }
            if candidate.kind == .mtp {
                let temp = request.temperature ?? 1
                if temp > 0 {
                    NovaMLXLog.info("[SpecBoost] Skipping MTP auto-inject for '\(mainId)' (temp=\(temp) > 0; Qwen MTP is greedy-only)")
                    continue
                }
            }
            if candidate.kind == .smallDraft {
                let mainDir = NovaMLXPaths.directory(forModelId: mainId)
                let draftDir = NovaMLXPaths.directory(forModelId: candidate.acceleratorId)
                guard let mainVocab = DraftModelRegistry.readVocabSize(from: mainDir),
                      let draftVocab = DraftModelRegistry.readVocabSize(from: draftDir),
                      mainVocab == draftVocab else {
                    NovaMLXLog.warning("[SpecBoost] Vocab mismatch for \(request.model), skipping auto-injection")
                    continue
                }
            }
            NovaMLXLog.info(
                "[SpecBoost] Auto-injecting \(candidate.kind.rawValue) '\(candidate.acceleratorId)' for '\(request.model)'"
            )
            return withDraft(request, draftId: candidate.acceleratorId)
        }
        return request
    }

    private func draftFamily(for request: InferenceRequest) -> ModelFamily {
        inferredFamily(settingsManager.resolveModelId(request.model))
    }

    private static func requestScenario(_ request: InferenceRequest) -> SpecScenario {
        var characters = 0
        var code = false
        for message in request.messages {
            let text = message.content ?? ""
            characters += text.count
            if text.contains("```") { code = true }
        }
        return AcceleratorSpeedGate.scenario(messageCharacters: characters, containsCodeFence: code)
    }

    private func withDraft(_ request: InferenceRequest, draftId: String) -> InferenceRequest {
        request.replacing(draftModel: draftId, numDraftTokens: request.numDraftTokens ?? 4)
    }

    public func loadModel(
        at url: URL,
        config: ModelConfig,
        progress: (@Sendable (LoadPhase) -> Void)? = nil,
        asMtpCompanion: Bool = false
    ) async throws {
        let modelId = config.identifier.id
        if config.modelType == .decision || config.identifier.family == .laya
            || LayaCheckpoint.isLayaDirectory(url)
        {
            if decisionService.isLoaded(modelId) { return }
            try await loadDedup.ensureSingle(modelId: modelId) {
                if self.decisionService.isLoaded(modelId) { return }
                try await self.decisionService.load(from: url, modelId: modelId)
                self.saveLoadedModelsList()
            }
            return
        }
        if !asMtpCompanion, NovaMLXEngine.isMtpDraftConfig(at: url) || NovaMLXEngine.isDFlashDraftConfig(at: url) {
            throw NovaMLXError.mtpCompanionNotLoadable(modelId)
        }

        // Audio models bypass engine entirely — route to specialized services
        if config.modelType == .audio {
            NovaMLXLog.info("[InferenceService] Loading audio model: \(modelId), family=\(config.identifier.family), url=\(url.path)")
            try await loadDedup.ensureSingle(modelId: modelId) {
                let family = config.identifier.family
                if family == .qwen3Tts || family == .dotsTts {
                    NovaMLXLog.info("[InferenceService] Routing to TTSService for \(modelId)")
                    do {
                        try await self.ttsService.loadModel(from: url)
                        NovaMLXLog.info("[InferenceService] TTS model loaded successfully: \(modelId)")
                    } catch {
                        NovaMLXLog.error("[InferenceService] TTS model load FAILED for \(modelId): \(error)")
                        throw error
                    }
                    self.saveLoadedModelsList()
                    return
                }
                NovaMLXLog.info("[InferenceService] Routing to TranscriptionService for \(modelId)")
                _ = try await self.transcriptionService.loadModel(from: url, config: config, progress: progress)
                self.saveLoadedModelsList()
            }
            return
        }

        // Image models stay on the host ImageGenerationService (not the worker MLXEngine).
        let isImageFamily: Bool = {
            switch config.identifier.family {
            case .flux, .flux2, .zImage, .qwenImage, .qwenImage21, .mossImage, .biRefNet, .stableDiffusion: return true
            default: return false
            }
        }()
        if config.modelType == .image || isImageFamily {
            if ImageModelSupport.isUnsupported(modelId) {
                throw NovaMLXError.unsupportedModel(ImageModelSupport.refuseMessage(modelId))
            }
            NovaMLXLog.info(
                "[InferenceService] Loading image model: \(modelId), family=\(config.identifier.family), url=\(url.path)"
            )
            try await loadDedup.ensureSingle(modelId: modelId) {
                _ = try await self.imageGenerationService.loadModel(
                    from: url, config: config, progress: progress
                )
                self.saveLoadedModelsList()
            }
            return
        }

        // Hunyuan3D (image → mesh) stays on the host Hy3DService.
        if config.modelType == .threed || config.identifier.family == .hunyuan3d {
            NovaMLXLog.info(
                "[InferenceService] Loading 3D model: \(modelId), family=\(config.identifier.family), url=\(url.path)"
            )
            try await loadDedup.ensureSingle(modelId: modelId) {
                try await self.hy3DService.loadModel(from: url, config: config)
                self.saveLoadedModelsList()
            }
            return
        }

        try await loadDedup.ensureSingle(modelId: modelId) { [self] in
            if !asMtpCompanion, config.modelType == .llm || config.modelType == .vlm {
                await self.evictOthersForExclusive(keeping: modelId)
            }
            try await self.ensureTieLayoutIfNeeded(
                modelId: modelId, url: url, progress: progress
            )
            do {
                if self.workerMode, let worker = self.worker {
                    let flags = try await worker.sendLoad(modelId: modelId, path: url.path, config: config, progress: progress)
                    self.workerLoadedModels.insert(modelId)
                    self.workerModelTypes[modelId] = config.modelType
                    if flags.hybrid {
                        self.workerHybridModels.insert(modelId)
                    }
                    if flags.nativeMtp {
                        self.workerNativeMtpModels.insert(modelId)
                    }
                } else {
                    _ = try await self.engine.loadModel(from: url, config: config, progress: progress)
                    let settings = self.settingsManager.getSettings(modelId)
                    if settings.isPinned {
                        self.engine.pool.pin(modelId)
                    }
                }
            } catch {
                if ExpertShardConverter.inspect(at: url).status == .ready
                    || ExpertShardConverter.inspect(at: url).status == .incomplete
                {
                    throw NovaMLXError.modelLoadFailed(
                        modelId,
                        underlying: NovaMLXError.apiError(
                            "\(error.localizedDescription) If TIE conversion looks wrong, unload (if needed), remove the TIE layout, and convert again."
                        )
                    )
                }
                throw error
            }
            self.saveLoadedModelsList()
        }

        if !asMtpCompanion, config.modelType == .llm || config.modelType == .vlm {
            await applyAcceleratorGate(
                mainId: modelId,
                family: config.identifier.family,
                modelType: config.modelType
            )
        }
        await applyResourceLimits()
    }

    /// Host download manager used when an `auto_on` or accepted companion is not on disk.
    public func bindAcceleratorDownloads(_ manager: ModelManager) {
        acceleratorDownloads = manager
    }

    public func pendingAcceleratorPrompt(for modelId: String) -> AcceleratorPrompt? {
        let resolved = settingsManager.resolveModelId(modelId)
        return acceleratorLock.withLock { acceleratorPrompts[resolved] }
    }

    /// Remember the load-time answer and attach the best enabled companion.
    /// An empty set stores "no" for every option that was asked.
    public func resolveAcceleratorPrompt(_ prompt: AcceleratorPrompt, enabledIds: Set<String>) async {
        let resolved = settingsManager.resolveModelId(prompt.mainModelId)
        settingsManager.updateSettings(resolved) { settings in
            var choices = settings.acceleratorChoices ?? [:]
            for option in prompt.options {
                choices[option.acceleratorId] = enabledIds.contains(option.acceleratorId)
            }
            settings.acceleratorChoices = choices
        }
        acceleratorLock.withLock { acceleratorPrompts[resolved] = nil }
        await applyAcceleratorGate(
            mainId: resolved, family: prompt.family, modelType: prompt.modelType)
    }

    private func ensureTieLayoutIfNeeded(
        modelId: String,
        url: URL,
        progress: (@Sendable (LoadPhase) -> Void)?
    ) async throws {
        let status = ExpertShardConverter.inspect(
            at: url,
            converting: await TieConversionBroker.shared.snapshot(modelId: modelId)
        )
        if status.status == .incomplete {
            throw NovaMLXError.tieLayoutIncomplete(modelId, status.message ?? "incomplete layout")
        }
        if status.status == .converting {
            progress?(.convertingTIE)
            try await TieConversionBroker.shared.convert(modelId: modelId, at: url)
            return
        }
        guard status.status == .convertible else { return }
        let estimated = MLXEngine.estimateModelWeightSize(at: url) ?? 0
        let gpu = MLXEngine.recommendedGpuBytes()
        if ExpertShardConverter.shouldAutoConvertOnLoad(
            at: url, estimatedBytes: estimated, gpuBudgetBytes: gpu
        ) {
            progress?(.convertingTIE)
            NovaMLXLog.info("[TIE] Auto-converting '\(modelId)' before load (model larger than GPU budget)")
            try await TieConversionBroker.shared.convert(modelId: modelId, at: url)
        }
    }

    private static func rejectMtpAsChat(_ modelId: String) throws {
        let dir = NovaMLXPaths.directory(forModelId:modelId)
        if NovaMLXEngine.isMtpDraftConfig(at: dir) {
            throw NovaMLXError.unsupportedModel(
                "MTP draft head cannot be used as a chat model: \(modelId)"
            )
        }
    }

    /// Load-time gate. `auto_on` downloads and attaches. `auto_off` is silent.
    /// `ask_user` stores a prompt and does not attach until the user answers.
    private func applyAcceleratorGate(
        mainId: String, family: ModelFamily, modelType: ModelType
    ) async {
        let resolved = settingsManager.resolveModelId(mainId)
        if settingsManager.getSettings(resolved).nativeMtpEnabled == false {
            NovaMLXLog.info("[SpecBoost] Skipping accelerator companions for '\(resolved)' (MTP/DFlash off)")
            acceleratorLock.withLock { acceleratorPrompts[resolved] = nil }
            return
        }
        let remembered = settingsManager.getSettings(resolved).acceleratorChoices ?? [:]
        let candidates = DraftModelRegistry.shared.acceleratorCandidates(
            forMainId: resolved, family: family, isHybrid: isHybridModel(resolved), modelType: modelType
        )
        let plan = AcceleratorSpeedGate.plan(
            mainModelId: resolved, scenario: .unspecified, candidates: candidates, remembered: remembered
        )
        var attached: String?
        for candidate in plan.attach {
            if await ensureAccelerator(candidate, mainId: resolved) {
                attached = candidate.acceleratorId
                NovaMLXLog.info(
                    "[SpecBoost] Attached \(candidate.kind.rawValue) '\(candidate.acceleratorId)' for '\(resolved)'"
                )
                break
            }
            NovaMLXLog.warning(
                "[SpecBoost] \(candidate.kind.rawValue) '\(candidate.acceleratorId)' did not attach; trying the next choice"
            )
        }
        if let attached {
            await unloadAcceleratorCompanions(of: resolved, except: attached)
        }
        if plan.ask.isEmpty {
            acceleratorLock.withLock { acceleratorPrompts[resolved] = nil }
            return
        }
        let prompt = AcceleratorPrompt(
            mainModelId: resolved,
            family: family,
            modelType: modelType,
            options: plan.ask.map {
                AcceleratorPromptOption(
                    acceleratorId: $0.candidate.acceleratorId,
                    kind: $0.candidate.kind.rawValue,
                    displayName: $0.candidate.displayName,
                    reason: $0.reason,
                    estimatedSizeMB: $0.candidate.estimatedSizeMB
                )
            },
            autoAttachedId: attached
        )
        acceleratorLock.withLock { acceleratorPrompts[resolved] = prompt }
        NotificationCenter.default.post(name: .novaMLXAcceleratorPrompt, object: resolved)
    }

    private func ensureAccelerator(_ candidate: AcceleratorCandidate, mainId: String) async -> Bool {
        if candidate.kind == .dspark {
            if DSparkDrafterStore.isLoadableDSparkDraft(modelId: candidate.acceleratorId) { return true }
            guard await downloadAccelerator(candidate) else { return false }
            return DSparkDrafterStore.isLoadableDSparkDraft(modelId: candidate.acceleratorId)
        }
        // DFlash2 checkpoints are not chat models. Download only when the speed
        // gate put this id in `plan.attach`. The worker loads the drafter itself.
        if candidate.kind == .dflash2 {
            if dflashCompanionReady(modelId: candidate.acceleratorId) { return true }
            guard await downloadAccelerator(candidate) else { return false }
            return dflashCompanionReady(modelId: candidate.acceleratorId)
        }
        // Qwen MTP heads are MTPDrafterModel checkpoints, not chat models.
        if candidate.kind == .mtp {
            if qwenMtpCompanionReady(modelId: candidate.acceleratorId) { return true }
            guard await downloadAccelerator(candidate) else { return false }
            return qwenMtpCompanionReady(modelId: candidate.acceleratorId)
        }
        if isModelLoaded(candidate.acceleratorId) { return true }
        var dir = NovaMLXPaths.directory(forModelId: candidate.acceleratorId)
        if !FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) {
            guard await downloadAccelerator(candidate) else {
                NovaMLXLog.info(
                    "[SpecBoost] '\(candidate.acceleratorId)' is not on disk; continuing without it"
                )
                return false
            }
            dir = NovaMLXPaths.directory(forModelId: candidate.acceleratorId)
        }
        guard FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path) else {
            return false
        }
        let config = ModelConfig(
            identifier: ModelIdentifier(id: candidate.acceleratorId, family: candidate.family),
            modelType: .llm
        )
        do {
            try await loadModel(at: dir, config: config, asMtpCompanion: true)
            return true
        } catch {
            NovaMLXLog.warning(
                "[SpecBoost] Companion '\(candidate.acceleratorId)' failed for '\(mainId)': \(error.localizedDescription)"
            )
            return false
        }
    }

    private func downloadAccelerator(_ candidate: AcceleratorCandidate) async -> Bool {
        guard let manager = acceleratorDownloads else { return false }
        if manager.isDownloaded(candidate.acceleratorId) { return true }
        if manager.getRecord(candidate.acceleratorId) == nil {
            manager.register(
                id: candidate.acceleratorId,
                family: candidate.family,
                modelType: .llm,
                remoteURL: "https://huggingface.co/\(candidate.acceleratorId)",
                sizeBytes: UInt64(candidate.estimatedSizeMB) * 1_000_000
            )
        }
        if manager.getDownloadStatus(candidate.acceleratorId)?.state != .downloading {
            do {
                try manager.startDownload(candidate.acceleratorId)
            } catch {
                NovaMLXLog.warning(
                    "[SpecBoost] Download failed to start for '\(candidate.acceleratorId)': \(error.localizedDescription)"
                )
                if manager.getDownloadStatus(candidate.acceleratorId)?.state != .downloading {
                    return false
                }
            }
        }
        for _ in 0..<86_400 {
            if manager.isDownloaded(candidate.acceleratorId) { return true }
            let state = manager.getDownloadStatus(candidate.acceleratorId)?.state
            if state == .failed || state == .notDownloaded { return false }
            if state != .downloading && state != nil { return false }
            try? await Task.sleep(for: .milliseconds(500))
        }
        return manager.isDownloaded(candidate.acceleratorId)
    }

    private func unloadAcceleratorCompanions(of mainId: String, except keep: String? = nil) async {
        let family = inferredFamily(mainId)
        let candidates = DraftModelRegistry.shared.acceleratorCandidates(
            forMainId: mainId, family: family, isHybrid: isHybridModel(mainId), modelType: .llm
        )
        for candidate in candidates where candidate.acceleratorId != keep && candidate.acceleratorId != mainId {
            guard isModelLoaded(candidate.acceleratorId) else { continue }
            if candidate.kind == .dspark, DSparkDrafterStore.isLoadableDSparkDraft(modelId: candidate.acceleratorId) {
                continue
            }
            if candidate.kind == .dflash2, dflashCompanionReady(modelId: candidate.acceleratorId) {
                continue
            }
            if candidate.kind == .mtp, qwenMtpCompanionReady(modelId: candidate.acceleratorId) {
                QwenMtpDrafterStore.shared.release(modelId: candidate.acceleratorId)
                continue
            }
            NovaMLXLog.info(
                "[SpecBoost] Unloading \(candidate.kind.rawValue) '\(candidate.acceleratorId)' for '\(mainId)'"
            )
            await unloadModel(ModelIdentifier(id: candidate.acceleratorId, family: candidate.family))
        }
    }

    private func inferredFamily(_ modelId: String) -> ModelFamily {
        if let container = engine.getContainer(for: modelId) {
            return container.identifier.family
        }
        let id = modelId.lowercased()
        if id.contains("qwen") { return .qwen }
        if id.contains("llama") { return .llama }
        if id.contains("gemma") { return .gemma }
        if id.contains("mistral") { return .mistral }
        if id.contains("phi") { return .phi }
        if id.contains("deepseek") { return .deepseek }
        return .other
    }

    /// MTP is only a fallback when DFlash2 is not loaded.
    public static func shouldLoadMtpCompanion(hasDFlash: Bool) -> Bool {
        !hasDFlash
    }

    /// Companions kept with a chat backbone. DFlash2 replaces MTP when present.
    public static func companionKeepIds(
        backboneId: String, dflashId: String?, mtpId: String?, dsparkId: String? = nil
    ) -> Set<String> {
        var keep: Set<String> = [backboneId]
        if let dsparkId {
            keep.insert(dsparkId)
        }
        if let dflashId {
            keep.insert(dflashId)
        } else if dsparkId == nil, let mtpId {
            keep.insert(mtpId)
        }
        return keep
    }

    /// Exclusive mode: one chat/backbone model plus DFlash2, or MTP if no DFlash.
    public static func exclusiveKeepIds(for modelId: String) -> Set<String> {
        companionKeepIds(
            backboneId: modelId,
            dflashId: DraftModelRegistry.shared.dflashCandidate(forMainId: modelId)?.draftModelId,
            mtpId: DraftModelRegistry.shared.mtpCandidate(forMainId: modelId)?.draftModelId,
            dsparkId: DraftModelRegistry.shared.dsparkCandidate(forMainId: modelId)?.draftModelId
        )
    }

    public func evictOthersForExclusive(keeping modelId: String) async {
        let exclusiveOn = await NovaMLXConfiguration.shared.serverConfig.exclusiveAutoUnload
        guard exclusiveOn else {
            NovaMLXLog.info("[Exclusive] skipped — exclusiveAutoUnload is off")
            return
        }
        let keep = Self.exclusiveKeepIds(for: modelId)
        let sideLoaded = Set(
            ttsService.listLoadedModels()
                + transcriptionService.listLoadedModels()
                + imageGenerationService.listLoadedModels()
                + decisionService.listLoadedModels()
        )
        let pinned = Set(settingsManager.getPinnedModelIds())
        for id in listLoadedModels() where Self.shouldExclusiveEvict(
            id: id, keep: keep, sideLoaded: sideLoaded, pinned: pinned
        ) {
            NovaMLXLog.info("[Exclusive] unloading '\(id)' to keep '\(modelId)'")
            await unloadModel(ModelIdentifier(id: id, family: .other))
        }
    }

    /// Chat exclusive-load may replace another LLM/VLM, but never auto-unload
    /// TTS, ASR, or image models sitting beside it. Pinned models stay loaded.
    public static func shouldExclusiveEvict(
        id: String, keep: Set<String>, sideLoaded: Set<String>, pinned: Set<String> = []
    ) -> Bool {
        !keep.contains(id) && !sideLoaded.contains(id) && !pinned.contains(id)
    }

    public func isPinned(_ modelId: String) -> Bool {
        settingsManager.getSettings(settingsManager.resolveModelId(modelId)).isPinned
    }

    /// Persist pin. A pinned model reloads after restart and refuses unload until this is off.
    public func setPinned(_ modelId: String, pinned: Bool) {
        let id = settingsManager.resolveModelId(modelId)
        var settings = settingsManager.getSettings(id)
        settings.isPinned = pinned
        settingsManager.setSettings(id, settings)
        if pinned {
            engine.pool.pin(id)
            saveLoadedModelsList()
        } else {
            engine.pool.unpin(id)
        }
        NovaMLXLog.info("[InferenceService] pin \(id)=\(pinned)")
    }

    /// Nil when unload is allowed. Message when the model is pinned.
    public func pinnedUnloadRefusal(_ modelId: String) -> String? {
        let id = settingsManager.resolveModelId(modelId)
        guard settingsManager.getSettings(id).isPinned else { return nil }
        return "Cannot unload pinned model '\(id)'. Turn Pin off first."
    }

    /// Last-loaded ids first, then any pinned id that is not already in that list.
    public static func restoreIds(lastLoaded: [String], pinned: [String]) -> [String] {
        var seen = Set<String>()
        var ids: [String] = []
        for id in lastLoaded + pinned where seen.insert(id).inserted {
            ids.append(id)
        }
        return ids
    }

    public func applyResourceLimits() async {
        await engine.applyResourceLimits()
    }

    public func chatLoadedCount() -> Int {
        listLoadedModels().filter { id in
            !NovaMLXEngine.isMtpDraftConfig(at: NovaMLXPaths.directory(forModelId:id))
                && !NovaMLXEngine.isDFlashDraftConfig(at: NovaMLXPaths.directory(forModelId:id))
                && !NovaMLXEngine.isDSparkDraftConfig(at: NovaMLXPaths.directory(forModelId:id))
                && !ResourceLimits.isCompanionDraftModelId(id)
        }.count
    }

    public func unloadModel(_ identifier: ModelIdentifier) async {
        let resolvedId = settingsManager.resolveModelId(identifier.id)
        if settingsManager.getSettings(resolvedId).isPinned {
            NovaMLXLog.warning("[InferenceService] refused unload of pinned model \(resolvedId)")
            return
        }

        // TTS models
        if identifier.family == .qwen3Tts || identifier.family == .dotsTts
            || ttsService.listLoadedModels().contains(resolvedId) {
            ttsService.unloadModel()
            saveLoadedModelsList()
            return
        }

        // ASR models (Whisper, Qwen3-ASR) live in TranscriptionService, not MLXEngine.
        if transcriptionService.isLoaded(resolvedId) {
            transcriptionService.unload(modelId: resolvedId)
            saveLoadedModelsList()
            return
        }

        if imageGenerationService.isLoaded(resolvedId) {
            imageGenerationService.unload(modelId: resolvedId)
            saveLoadedModelsList()
            return
        }

        if hy3DService.isLoaded(resolvedId) {
            hy3DService.unload(modelId: resolvedId)
            saveLoadedModelsList()
            return
        }

        if decisionService.isLoaded(resolvedId) {
            decisionService.unload(modelId: resolvedId)
            saveLoadedModelsList()
            return
        }

        if workerMode, let worker = worker {
            try? await worker.sendUnload(modelId: identifier.id)
            workerLoadedModels.remove(identifier.id)
            workerModelTypes.removeValue(forKey: identifier.id)
            workerHybridModels.remove(identifier.id)
            workerNativeMtpModels.remove(identifier.id)
        } else {
            engine.unloadModel(identifier)
        }
        saveLoadedModelsList()
        await applyResourceLimits()
    }

    public func isModelLoaded(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        if workerLoadedModels.contains(resolvedId) { return true }
        if !workerMode, engine.getContainer(for: resolvedId)?.isLoaded == true { return true }
        // DSpark companion drafters ship no tokenizer and never enter the model pool;
        // the engine's DSpark dispatch lazy-loads them from disk. Report an on-disk,
        // parseable one as loaded so an explicit `draft_model` skips the pool
        // auto-load (which cannot load these checkpoints).
        if DSparkDrafterStore.isLoadableDSparkDraft(modelId: resolvedId) { return true }
        // Same for DFlash2: an on-disk checkpoint with weights is "loaded" so
        // auto-inject can set draft_model without pool-loading it as Qwen3.
        if dflashCompanionReady(modelId: resolvedId) { return true }
        // Qwen MTP heads are loaded by QwenMtpDrafterStore during generate.
        if qwenMtpCompanionReady(modelId: resolvedId) { return true }
        // Decision, ASR, TTS and image models stay on the host. In worker mode
        // they are not in workerLoadedModels; treating that as "not loaded"
        // reloaded the 678 MB Laya checkpoint on every request.
        if transcriptionService.isLoaded(resolvedId) { return true }
        if ttsService.listLoadedModels().contains(resolvedId) { return true }
        if imageGenerationService.isLoaded(resolvedId) { return true }
        if hy3DService.isLoaded(resolvedId) { return true }
        if decisionService.isLoaded(resolvedId) { return true }
        return false
    }

    public func isHybridModel(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        if workerMode {
            return workerHybridModels.contains(resolvedId)
        }
        return engine.getContainer(for: resolvedId)?.config.hasLinearAttention ?? false
    }

    public func hasNativeMtp(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        if workerMode {
            return workerNativeMtpModels.contains(resolvedId)
        }
        return engine.getContainer(for: resolvedId)?.hasNativeMtp == true
    }

    /// Standalone MTP draft on disk for this backbone (Spec Boost companion).
    public func hasCompanionMtp(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        guard let mtp = DraftModelRegistry.shared.mtpCandidate(forMainId: resolvedId) else {
            return false
        }
        return isModelLoaded(mtp.draftModelId) || companionMtpOnDisk(mtp.draftModelId)
    }

    /// Show the MTP switch when native weights or a companion pack is present.
    public func mtpSwitchAvailable(_ modelId: String) -> Bool {
        hasNativeMtp(modelId) || hasCompanionMtp(modelId) || hasCompanionDFlash(modelId)
    }

    public func hasCompanionDFlash(_ modelId: String) -> Bool {
        let resolvedId = settingsManager.resolveModelId(modelId)
        guard let dflash = DraftModelRegistry.shared.dflashCandidate(forMainId: resolvedId) else {
            return false
        }
        return isModelLoaded(dflash.draftModelId) || companionMtpOnDisk(dflash.draftModelId)
    }

    public func isMtpEnabled(_ modelId: String) -> Bool {
        settingsManager.getSettings(settingsManager.resolveModelId(modelId)).nativeMtpEnabled != false
    }

    /// Persist the unified MTP switch (native head + companion auto-load/inject).
    public func setMtpEnabled(_ modelId: String, enabled: Bool) async {
        let resolvedId = settingsManager.resolveModelId(modelId)
        settingsManager.updateSettings(resolvedId) { $0.nativeMtpEnabled = enabled }
        if enabled {
            await applyAcceleratorGate(
                mainId: resolvedId,
                family: inferredFamily(resolvedId),
                modelType: .llm
            )
        } else {
            acceleratorLock.withLock { acceleratorPrompts[resolvedId] = nil }
            await unloadAcceleratorCompanions(of: resolvedId)
        }
    }

    private func companionMtpOnDisk(_ draftModelId: String) -> Bool {
        let dir = NovaMLXPaths.directory(forModelId: draftModelId)
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
    }

    /// Check if a model can be loaded given current memory constraints.
    /// Works in both direct and worker mode — uses Metal device info directly.
    public func checkMemoryFeasibility(modelId: String, sizeBytes: UInt64, localURL: URL) async -> MemoryFeasibility? {
        if isModelLoaded(modelId) { return nil }

        let maxGPU = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? 0
        guard maxGPU > 0 else { return nil }

        let estimatedBytes = MLXEngine.estimateModelWeightSize(at: localURL) ?? sizeBytes
        let currentBytes = UInt64(MLX.Memory.activeMemory)
        let available = currentBytes < maxGPU ? maxGPU - currentBytes : 0

        return MemoryFeasibility.evaluate(
            modelId: modelId,
            modelSizeBytes: estimatedBytes,
            currentlyAvailableBytes: available,
            gpuBudgetBytes: maxGPU
        )
    }

    public func listLoadedModels() -> [String] {
        var models: [String]
        if workerMode {
            models = Array(workerLoadedModels)
        } else {
            models = engine.listLoadedModels()
        }
        // Include audio models loaded in transcriptionService
        models.append(contentsOf: transcriptionService.listLoadedModels().filter { !models.contains($0) })
        // Include TTS models
        models.append(contentsOf: ttsService.listLoadedModels().filter { !models.contains($0) })
        // Include image models
        models.append(contentsOf: imageGenerationService.listLoadedModels().filter { !models.contains($0) })
        // Include 3D models
        models.append(contentsOf: hy3DService.listLoadedModels().filter { !models.contains($0) })
        models.append(contentsOf: decisionService.listLoadedModels().filter { !models.contains($0) })
        return models
    }

    public func checkTTLExpirations() {
        let allInfo = engine.pool.allModelInfo()
        let now = Date()
        for info in allInfo {
            if info.pinned || settingsManager.getSettings(info.id).isPinned { continue }

            if let prd = info.perRequestDeadline {
                if prd > now { continue }
                let identifier = ModelIdentifier(id: info.id, family: .other)
                engine.unloadModel(identifier)
                NovaMLXLog.debug("Per-request keep_alive expired for \(info.id), unloaded")
                continue
            }

            let settings = settingsManager.getSettings(info.id)
            guard let ttl = settings.ttlSeconds, ttl > 0 else { continue }
            let idleTime = now.timeIntervalSince(info.lastAccessed)
            if idleTime >= Double(ttl) {
                let identifier = ModelIdentifier(id: info.id, family: .other)
                engine.unloadModel(identifier)
                NovaMLXLog.debug("TTL expired for \(info.id), unloaded after \(ttl)s idle")
            }
        }
        saveLoadedModelsList()
    }

    // MARK: - Loaded Models Persistence

    public func saveLoadedModelsList() {
        let ids = listLoadedModels()
        try? NovaDB.shared.loadedModelsStore.replaceAll(with: ids)
    }

    private func loadLoadedModelsList() -> [String] {
        (try? NovaDB.shared.loadedModelsStore.list()) ?? []
    }

    public typealias RestoreProgressCallback = @Sendable (_ modelId: String, _ phase: RestorePhase) -> Void

    public enum RestorePhase: Sendable { case started, completed, failed, skipped }

    public func restoreModels(modelManager: ModelManager,
                              progress: RestoreProgressCallback? = nil) async {
        let ids = Self.restoreIds(
            lastLoaded: loadLoadedModelsList(),
            pinned: settingsManager.getPinnedModelIds()
        )
        guard !ids.isEmpty else { return }
        NovaMLXLog.info("[InferenceService] Restoring \(ids.count) previously loaded model(s)...")
        for modelId in ids {
            guard let record = modelManager.getRecord(modelId) else {
                NovaMLXLog.warning("[InferenceService] Skipping restore of '\(modelId)' — not found in registry")
                progress?(modelId, .skipped)
                continue
            }
            progress?(modelId, .started)
            if NovaMLXPaths.triggersRemovableVolumeTCC(record.localURL) {
                NovaMLXLog.info("[InferenceService] Skipping restore of '\(modelId)' — on a removable volume")
                progress?(modelId, .skipped)
                continue
            }
            if NovaMLXEngine.isMtpDraftConfig(at: record.localURL) {
                NovaMLXLog.info("[InferenceService] Skipping restore of MTP companion '\(modelId)'")
                progress?(modelId, .skipped)
                continue
            }
            let config = ModelConfig(
                identifier: ModelIdentifier(id: modelId, family: record.family),
                modelType: record.modelType
            )
            do {
                try await loadModel(at: record.localURL, config: config)
                NovaMLXLog.info("[InferenceService] Restored model: \(modelId) (type: \(record.modelType))")
                progress?(modelId, .completed)
            } catch {
                NovaMLXLog.warning("[InferenceService] Failed to restore model \(modelId): \(error)")
                progress?(modelId, .failed)
            }
        }
        saveLoadedModelsList()
    }

    public var stats: InferenceStats {
        let gpuMem: UInt64
        if workerMode, let worker = worker, let memStats = worker.latestMemoryStats {
            gpuMem = memStats.currentBytes
        } else {
            gpuMem = engine.gpuActiveMemory
        }
        let modelCount = workerMode ? workerLoadedModels.count : engine.loadedModelCount
        let activeReqs = workerMode
            ? (worker?.activeRequestCount ?? 0)
            : (batcher.activeRequests + fusedScheduler.activeRequestCount)
        let workerCpu = workerMode ? (worker?.latestMemoryStats?.cpuUsage ?? 0) : 0
        return InferenceStats(
            loadedModels: modelCount,
            activeRequests: activeReqs,
            gpuMemoryUsed: gpuMem,
            recentTokensPerSecond: engine.metricsStore.recentTokensPerSecond,
            liveActivity: engine.metricsStore.liveActivity,
            totalTokensGenerated: engine.metricsStore.metrics.totalTokensAllTime,
            workerCpuUsage: workerCpu
        )
    }

    public var batcherMetrics: BatcherMetrics {
        batcher.metrics
    }

    public var fusedSchedulerMetrics: FusedSchedulerMetrics {
        let local = fusedScheduler.metrics
        // The menu-bar app decodes in the worker. Host counters stay at zero;
        // the worker copies its scheduler into the 5s memory-stats message.
        guard workerMode, let reported = worker?.latestMemoryStats else { return local }
        return FusedSchedulerMetrics(
            activeSequences: reported.fusedActiveSequences,
            queueDepth: local.queueDepth,
            totalDecodeSteps: local.totalDecodeSteps,
            totalTokensViaFused: local.totalTokensViaFused,
            peakBatchWidth: reported.fusedPeakBatchWidth,
            totalPreemptions: local.totalPreemptions,
            specAcceptanceRate: local.specAcceptanceRate,
            specCacheSize: local.specCacheSize,
            peakFusedForwardWidth: reported.fusedPeakForwardWidth,
            totalFusedForwards: reported.fusedForwardCount
        )
    }

    public func resolveModelId(_ input: String) -> String {
        settingsManager.resolveModelId(input)
    }

    public func forkSession(from sourceId: String, into targetId: String, modelId: String) async throws {
        let resolvedModelId = settingsManager.resolveModelId(modelId)
        try await engine.forkSession(from: sourceId, into: targetId, modelId: resolvedModelId)
    }

    public func countTokens(model: String, messages: [ChatMessage]) -> Int? {
        let resolvedId = settingsManager.resolveModelId(model)
        guard let container = engine.getContainer(for: resolvedId),
              let tokenizer = container.tokenizer else { return nil }
        var total = 0
        for msg in messages {
            if let content = msg.content {
                total += tokenizer.encode(content).count
            }
        }
        return total
    }

    public func getContextWindow(for modelId: String) -> Int? {
        let resolvedId = settingsManager.resolveModelId(modelId)
        let settings = settingsManager.getSettings(resolvedId)
        return settings.maxContextWindow
    }
}

private final class StreamTracker: @unchecked Sendable {
    let model: String
    let metricsStore: MetricsStore
    let kind: InferenceKind
    let startTime = Date()
    var tokenCount = 0
    private var lastLiveUpdate: Date = Date()

    init(model: String, kind: InferenceKind, metricsStore: MetricsStore) {
        self.model = model
        self.kind = kind
        self.metricsStore = metricsStore
    }

    func increment() {
        tokenCount += 1
        let now = Date()
        // Update live TPS every second during streaming to keep the status chart fresh
        if now.timeIntervalSince(lastLiveUpdate) >= 1.0 {
            let elapsed = now.timeIntervalSince(startTime)
            if elapsed > 0.01 {
                let tps = Double(tokenCount) / elapsed
                metricsStore.updateLiveTps(tps)
                metricsStore.reportActivity(model: model, kind: kind, speed: tps, unit: "tok/s")
            }
            lastLiveUpdate = now
        }
    }

    func finish(request: InferenceRequest? = nil, error: String? = nil) {
        let elapsed = Date().timeIntervalSince(startTime)
        if tokenCount > 0 && elapsed > 0 {
            let tps = Double(tokenCount) / elapsed
            metricsStore.updateLiveTps(tps)
            metricsStore.reportActivity(model: model, kind: kind, speed: tps, unit: "tok/s")
            metricsStore.recordRequest(model: model, tokens: UInt64(tokenCount), inferenceTime: elapsed)
        }
        // Finalize the request-log entry so the Requests page shows final stats.
        if let req = request {
            RequestLogStore.shared.finish(
                request: req, model: model, kind: kind,
                tps: tokenCount > 0 && elapsed > 0 ? Double(tokenCount) / elapsed : 0,
                promptTokens: 0, completionTokens: tokenCount,
                durationMs: elapsed * 1000, error: error)
        }
        // Keep the activity record briefly so the final speed is visible; it will
        // also age out via the store's staleness threshold. Don't clear here since
        // staleness handling in the store covers it.
    }
}

public struct InferenceStats: Sendable {
    public let loadedModels: Int
    public let activeRequests: Int
    public let gpuMemoryUsed: UInt64
    public let recentTokensPerSecond: Double
    /// Live, in-progress inference across ALL backends (LLM, VLM, ASR, TTS, image).
    /// nil when idle.
    public let liveActivity: LiveActivity?
    public let totalTokensGenerated: UInt64
    public let workerCpuUsage: Double
    public init(loadedModels: Int = 0, activeRequests: Int = 0, gpuMemoryUsed: UInt64 = 0, recentTokensPerSecond: Double = 0, liveActivity: LiveActivity? = nil, totalTokensGenerated: UInt64 = 0, workerCpuUsage: Double = 0) {
        self.loadedModels = loadedModels
        self.activeRequests = activeRequests
        self.gpuMemoryUsed = gpuMemoryUsed
        self.recentTokensPerSecond = recentTokensPerSecond
        self.liveActivity = liveActivity
        self.totalTokensGenerated = totalTokensGenerated
        self.workerCpuUsage = workerCpuUsage
    }
}
