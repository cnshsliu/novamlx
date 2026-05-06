import Foundation
import MLX
import NovaMLXCore
import NovaMLXModelManager
import NovaMLXUtils
import NovaMLXEngine

public actor AutoLoadCoordinator {

    public struct Options: Sendable {
        public var evictOnConflict: Bool
        public var allowDownload: Bool
        public var coldLoadDeadline: Date?
        public var progress: (@Sendable (LoadPhase) -> Void)?

        public init(
            evictOnConflict: Bool = true,
            allowDownload: Bool = false,
            coldLoadDeadline: Date? = nil,
            progress: (@Sendable (LoadPhase) -> Void)? = nil
        ) {
            self.evictOnConflict = evictOnConflict
            self.allowDownload = allowDownload
            self.coldLoadDeadline = coldLoadDeadline
            self.progress = progress
        }
    }

    private let inference: InferenceService
    private let embeddings: EmbeddingService
    private let models: ModelManager
    private let settings: ModelSettingsManager?
    private let defaultTTLSeconds: Int?
    private var inFlight: [String: Task<Void, Error>] = [:]
    private let globalGate: AsyncSemaphore

    public init(
        inference: InferenceService,
        embeddings: EmbeddingService,
        models: ModelManager,
        settings: ModelSettingsManager? = nil,
        defaultTTLSeconds: Int? = nil
    ) {
        self.inference = inference
        self.embeddings = embeddings
        self.models = models
        self.settings = settings
        self.defaultTTLSeconds = defaultTTLSeconds
        self.globalGate = AsyncSemaphore(value: 1)
    }

    public func ensureLoaded(_ modelId: String, options: Options) async throws {
        // Fast path: already loaded
        if inference.isModelLoaded(modelId) || embeddings.isLoaded(modelId) {
            return
        }

        // Lookup record
        guard let record = models.getRecord(modelId) else {
            throw NovaMLXError.modelNotFound(modelId)
        }

        // Not downloaded
        if !models.isDownloaded(modelId) {
            if options.allowDownload {
                throw NovaMLXError.modelNotLoaded(
                    "\(modelId): allowDownload=true not yet implemented in v1.1.0"
                )
            } else {
                throw NovaMLXError.modelNotLoaded(
                    "\(modelId): not downloaded. Use POST /admin/models/download first."
                )
            }
        }

        // Per-model dedup
        if let existing = inFlight[modelId] {
            try await existing.value
            return
        }

        let task = Task<Void, Error> { [self] in
            do {
                try await self.performLoad(modelId: modelId, record: record, options: options)
                options.progress?(.ready)
            } catch {
                options.progress?(.failed)
                throw error
            }
        }
        inFlight[modelId] = task
        // Synchronous cleanup inside actor isolation — avoids the async-cleanup
        // race where piled-up callers see a stale (possibly failed) task.
        defer { inFlight.removeValue(forKey: modelId) }
        try await task.value
    }

    private func performLoad(
        modelId: String,
        record: ModelRecord,
        options: Options
    ) async throws {
        options.progress?(.queued)
        await globalGate.wait()
        defer { globalGate.signal() }

        // Re-check after acquiring gate — another request may have loaded it
        if inference.isModelLoaded(modelId) || embeddings.isLoaded(modelId) {
            return
        }

        // Feasibility check
        options.progress?(.feasibilityChecking)
        if let feasibility = await inference.checkMemoryFeasibility(
            modelId: modelId,
            sizeBytes: record.sizeBytes,
            localURL: record.localURL
        ), !feasibility.canLoad {
            if options.evictOnConflict {
                options.progress?(.evicting)
                try await evictForFit(needBytes: record.sizeBytes, excluding: modelId, localURL: record.localURL)
            } else {
                throw NovaMLXError.insufficientMemory(
                    neededMB: feasibility.modelSizeMB,
                    availableMB: feasibility.availableMB,
                    modelId: modelId
                )
            }
        }

        // Cold-load deadline check
        if let deadline = options.coldLoadDeadline, Date() > deadline {
            throw NovaMLXError.modelLoadInProgress(
                modelId: modelId,
                etaSeconds: 60
            )
        }

        // Dispatch to inference or embeddings
        let modelConfig = ModelConfig(
            identifier: ModelIdentifier(id: modelId, family: record.family),
            modelType: record.modelType
        )

        if record.modelType == .embedding {
            _ = try await embeddings.loadModel(
                from: record.localURL,
                config: modelConfig,
                progress: options.progress
            )
        } else {
            try await inference.loadModel(
                at: record.localURL,
                config: modelConfig,
                progress: options.progress
            )
        }

        applyDefaultTTLIfNeeded(modelId: modelId)
    }

    /// Apply the configured auto-load TTL when the model has no explicit TTL set.
    /// Existing per-model `ttlSeconds` values are preserved (never overridden).
    private func applyDefaultTTLIfNeeded(modelId: String) {
        guard let settings = settings, let defaultTTL = defaultTTLSeconds else { return }
        let current = settings.getSettings(modelId)
        if current.ttlSeconds == nil {
            settings.updateSettings(modelId) { s in s.ttlSeconds = defaultTTL }
        }
    }

    private func evictForFit(needBytes: UInt64, excluding modelId: String, localURL: URL) async throws {
        // Loop: evict LRU non-pinned model, recheck, until fit OR only pinned remain
        let maxIterations = inference.engine.pool.allModelInfo().count + 1
        for _ in 0..<maxIterations {
            let freed = inference.engine.pool.evictLRU(excluding: modelId)
            guard freed != nil else {
                let current = UInt64(MLX.Memory.activeMemory)
                let maxGPU = MLX.GPU.maxRecommendedWorkingSetBytes().map { UInt64($0) } ?? 0
                let availableMB = current < maxGPU ? (maxGPU - current) / 1_048_576 : 0
                throw NovaMLXError.insufficientMemory(
                    neededMB: needBytes / 1_048_576,
                    availableMB: availableMB,
                    modelId: modelId
                )
            }

            // Recheck feasibility with the real model URL
            if let feasibility = await inference.checkMemoryFeasibility(
                modelId: modelId,
                sizeBytes: needBytes,
                localURL: localURL
            ), feasibility.canLoad {
                return
            }
        }
        throw NovaMLXError.insufficientMemory(
            neededMB: needBytes / 1_048_576,
            availableMB: 0,
            modelId: modelId
        )
    }
}
