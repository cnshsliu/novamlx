import Foundation
import Hummingbird
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXUtils

// MARK: - Auto-Load Helpers
// Extracted from APIServer.swift for modularity.

extension NovaMLXAPIServer {

    enum LoadOutcome: Sendable {
        case alreadyLoaded
        case justLoaded(coldLoadMs: Int)
        /// Streaming: load is required but deferred to inside the response body
        /// so that withSSEKeepAlive's heartbeat covers the load window.
        case deferred
    }

    static func ensureModelReady(
        modelId: String,
        isStreaming: Bool,
        cfg: ServerConfig,
        inference: InferenceService,
        embeddings: EmbeddingService,
        coordinator: AutoLoadCoordinator,
        request: Request
    ) async throws -> LoadOutcome {
        // Fast path: already loaded
        if inference.isModelLoaded(modelId) || embeddings.isLoaded(modelId) {
            return .alreadyLoaded
        }

        // Auto-load disabled — throw original error
        if !cfg.autoLoad.enabled {
            throw NovaMLXError.modelNotLoaded(modelId)
        }

        // X-Wait-Cold-Load: false → fire-and-forget + immediate 503
        let waitForColdLoad = parseWaitColdLoadHeader(request)
        if waitForColdLoad == false {
            Task.detached {
                try? await coordinator.ensureLoaded(
                    modelId,
                    options: AutoLoadCoordinator.Options(
                        evictOnConflict: cfg.autoLoad.evictOnConflict,
                        allowDownload: cfg.autoLoad.allowDownload
                    )
                )
            }
            throw NovaMLXError.modelLoadInProgress(modelId: modelId, etaSeconds: 60)
        }

        // For streaming requests: defer the actual load to inside the response
        // body. The streaming handler wraps inference.stream(...) with
        // loadAwareStream, and withSSEKeepAlive's heartbeat fires while the
        // load runs — so the client never sees dead air during a 30-90s cold
        // load. Without this, the client would block waiting for the response
        // headers/body to start while the eager load completes here.
        if isStreaming {
            return .deferred
        }

        // Non-streaming: do the load now (the connection blocks anyway).
        let deadline = computeColdLoadDeadline(request: request, cfg: cfg)
        let started = Date()

        let options = AutoLoadCoordinator.Options(
            evictOnConflict: cfg.autoLoad.evictOnConflict,
            allowDownload: cfg.autoLoad.allowDownload,
            coldLoadDeadline: deadline
        )

        try await withColdLoadTimeout(deadline: deadline, modelId: modelId) {
            try await coordinator.ensureLoaded(modelId, options: options)
        }

        let coldLoadMs = Int(Date().timeIntervalSince(started) * 1000)
        return .justLoaded(coldLoadMs: coldLoadMs)
    }

    /// Wraps an inference token stream with an optional pre-load step. When the
    /// model isn't loaded and auto-load is enabled, the load runs *before* the
    /// inference stream begins yielding tokens. Combined with withSSEKeepAlive,
    /// this produces SSE `:keep-alive\n\n` heartbeat traffic during the load
    /// window so the connection stays open through cold-load delays.
    static func loadAwareStream(
        modelId: String,
        inference: InferenceService,
        coordinator: AutoLoadCoordinator,
        autoLoadCfg: AutoLoadConfig,
        inferenceStreamProducer: @Sendable @escaping () -> AsyncThrowingStream<Token, Error>
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    if !inference.isModelLoaded(modelId), autoLoadCfg.enabled {
                        try await coordinator.ensureLoaded(
                            modelId,
                            options: AutoLoadCoordinator.Options(
                                evictOnConflict: autoLoadCfg.evictOnConflict,
                                allowDownload: autoLoadCfg.allowDownload
                            )
                        )
                    }
                    for try await token in inferenceStreamProducer() {
                        if Task.isCancelled { break }
                        continuation.yield(token)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    static func computeColdLoadDeadline(request: Request, cfg: ServerConfig) -> Date {
        if let headerVal = request.headers[.init("X-Request-Timeout")!]?.first,
           let secs = Double(String(headerVal)) {
            let capped = min(secs, cfg.autoLoad.coldLoadTimeoutMaxSeconds)
            return Date().addingTimeInterval(capped)
        }

        let base = cfg.requestTimeout
        let multiplied = base * cfg.autoLoad.coldLoadTimeoutMultiplier
        let withFloor = max(multiplied, cfg.autoLoad.coldLoadTimeoutSeconds)
        let capped = min(withFloor, cfg.autoLoad.coldLoadTimeoutMaxSeconds)
        return Date().addingTimeInterval(capped)
    }

    static func withColdLoadTimeout<T: Sendable>(
        deadline: Date,
        modelId: String,
        operation: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await operation()
            }
            group.addTask {
                let interval = deadline.timeIntervalSinceNow
                if interval > 0 {
                    try await Task.sleep(for: .seconds(interval))
                }
                throw NovaMLXError.modelLoadInProgress(
                    modelId: modelId,
                    etaSeconds: 60
                )
            }
            defer { group.cancelAll() }
            guard let result = try await group.next() else {
                throw NovaMLXError.inferenceFailed("cold-load timeout race lost")
            }
            return result
        }
    }

    static func parseWaitColdLoadHeader(_ request: Request) -> Bool? {
        guard let v = request.headers[.init("X-Wait-Cold-Load")!]?.first?.lowercased() else {
            return nil
        }
        return v == "false" || v == "0" || v == "no" ? false : (v == "true" || v == "1" || v == "yes" ? true : nil)
    }
}
