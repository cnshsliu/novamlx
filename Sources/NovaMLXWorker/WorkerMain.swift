import Foundation
import Darwin
import MLX
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils
import NovaMLXEngine

@main
struct NovaMLXWorker {
    static func main() async {
        // Workers read prefix-cache kill switch from the SQLite configStore.
        // The worker process owns no other DB consumer, so we open the same
        // ~/.nova SQLite files the host app uses (read-only access would be a
        // future improvement; for now the worker is a trusted child process).
        // Legacy config.json is auto-imported by NovaDB.setup on first run.
        //
        // NovaDB MUST be set up BEFORE constructing MLXEngine — MetricsStore
        // (created inside MLXEngine.init) reads from NovaDB.shared.metricsStore
        // during its own init, and would otherwise trap on the nil IUO.
        do {
            try NovaDB.shared.setup(baseDir: NovaMLXPaths.baseDir)
        } catch {
            NovaMLXLog.warning("[Worker] failed to init NovaDB (\(error))")
        }

        let engine = MLXEngine()
        let batcher = ContinuousBatcher(engine: engine, maxBatchSize: 8)
        let fusedScheduler = FusedBatchScheduler(engine: engine)
        let writer = LineWriter()

        do {
            try await NovaMLXConfiguration.shared.loadFromStore()
            let cfg = await NovaMLXConfiguration.shared.serverConfig
            engine.setPrefixCacheEnabled(cfg.prefixCacheEnabled)
            await engine.applyResourceLimits(gpuRaw: cfg.maxGpuMemory, ramRaw: cfg.maxProcessMemory)
            // Env beats the config files. ServerConfig mirrors those files for
            // the admin API; passing it as an explicit override would hide
            // NOVA_PREFIX_CACHE_ROOT.
            let prefix = PrefixCacheLocation.resolveProduction()
            NovaMLXLog.info(
                "[Worker] prefixCacheEnabled=\(cfg.prefixCacheEnabled) root=\(prefix.root.path) "
                    + "ttl=\(Int(prefix.ttlSeconds))s hot=\(prefix.hotCacheMaxBytes) "
                    + "gpu=\(cfg.maxGpuMemory) ram=\(cfg.maxProcessMemory)"
            )
        } catch {
            NovaMLXLog.warning("[Worker] failed to load config (\(error)) — prefix cache stays at default")
        }

        // Start memory enforcer inside worker (1s polling, evicts models under pressure)
        await engine.startMemoryEnforcer()

        NovaMLXLog.info("[Worker] Started, waiting for commands on stdin")

        let lines = AsyncStream<String> { continuation in
            Thread {
                while let line = Swift.readLine() {
                    continuation.yield(line)
                }
                continuation.finish()
            }.start()
        }

        await withTaskGroup(of: Void.self) { group in
            // Background task: report memory + CPU stats to parent every 5s
            group.addTask {
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(5))
                    guard !Task.isCancelled else { return }
                    if let enforcer = engine.memoryEnforcer {
                        let status = await enforcer.status
                        let cpu = SystemMonitor.shared.currentStats()
                        let fused = fusedScheduler.metrics
                        writer.write(WorkerMessage(
                            type: WorkerMessageType.memoryStats,
                            memoryCurrentBytes: status.currentBytes,
                            memorySoftLimitBytes: status.softLimitBytes,
                            memoryHardLimitBytes: status.hardLimitBytes,
                            memoryUtilization: status.utilization,
                            memoryEvictions: status.totalEvictions,
                            cpuUsage: cpu.cpuUsage,
                            fusedActiveSequences: fused.activeSequences,
                            fusedPeakBatchWidth: fused.peakBatchWidth,
                            fusedPeakForwardWidth: fused.peakFusedForwardWidth,
                            fusedForwardCount: fused.totalFusedForwards
                        ))
                    }
                }
            }

            for await line in lines {
                guard !line.isEmpty else { continue }

                guard let data = line.data(using: .utf8),
                      let msg = try? JSONDecoder().decode(WorkerMessage.self, from: data) else {
                    writer.write(WorkerMessage(type: WorkerMessageType.error, errorMessage: "Invalid JSON"))
                    continue
                }

                switch msg.type {
                case WorkerMessageType.load:
                    await handleLoad(msg, engine: engine, writer: writer)

                case WorkerMessageType.unload:
                    await handleUnload(msg, engine: engine, writer: writer)

                case WorkerMessageType.generate:
                    group.addTask {
                        await handleGenerate(msg, engine: engine, batcher: batcher, fusedScheduler: fusedScheduler, writer: writer)
                    }

                case WorkerMessageType.stream:
                    let reqTagRead = (msg.requestId ?? "?").prefix(8)
                    NovaMLXLog.info("[Worker:\(reqTagRead)] stdin read + decoded — dispatching to handleStream")
                    group.addTask {
                        await handleStream(msg, engine: engine, batcher: batcher, fusedScheduler: fusedScheduler, writer: writer)
                    }

                case WorkerMessageType.abort:
                    if let reqId = msg.requestId, let uuid = UUID(uuidString: reqId) {
                        engine.abort(requestId: uuid)
                        batcher.abort(requestId: uuid)
                        fusedScheduler.abort(requestId: uuid)
                    }

                case WorkerMessageType.ping:
                    writer.write(WorkerMessage(type: WorkerMessageType.pong))

                default:
                    writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: "Unknown message type: \(msg.type)"))
                }
            }
        }

        NovaMLXLog.info("[Worker] Stdin closed, exiting")
    }

    // MARK: - Load

    private static func handleLoad(_ msg: WorkerMessage, engine: MLXEngine, writer: LineWriter) async {
        guard let modelId = msg.modelId, let path = msg.modelPath, let config = msg.modelConfig else {
            writer.write(WorkerMessage(type: WorkerMessageType.error, errorMessage: "Missing modelId/path/config"))
            return
        }
        do {
            let url = URL(fileURLWithPath: path)
            NovaMLXLog.info("[Worker] Loading \(modelId), MLX active=\(MLX.Memory.activeMemory / 1_048_576)MB, peak=\(MLX.Memory.peakMemory / 1_048_576)MB")
            if let reason = ExpertShardConverter.validationError(at: url) {
                throw NovaMLXError.tieLayoutIncomplete(modelId, reason)
            }
            // NovaMLX-TIE: bind BEFORE loadModel so shards move into tie-shards/
            // before MLX's eager loadWeights scans the directory. If we bind
            // after, loadWeights sees the per-shard files and loads everything
            // eagerly (defeats the purpose).
            let tierPolicy = await TieredOffloadPolicy.bindIfTiered(
                modelDir: url, metrics: engine.metricsStore
            )
            _ = try await engine.loadModel(from: url, config: config)
            let loaded = engine.getContainer(for: modelId)
            let isHybrid = loaded?.config.hasLinearAttention ?? false
            let nativeMtp = loaded?.hasNativeMtp ?? false
            if let container = loaded {
                container.tierPolicy = tierPolicy
                await container.applyTierPolicyToModel()
            }
            await engine.applyResourceLimits()
            writer.write(WorkerMessage(
                type: WorkerMessageType.loaded, modelId: modelId,
                hasLinearAttention: isHybrid, hasNativeMtp: nativeMtp))
            NovaMLXLog.info("[Worker] Loaded model: \(modelId), MLX active=\(MLX.Memory.activeMemory / 1_048_576)MB, peak=\(MLX.Memory.peakMemory / 1_048_576)MB")
        } catch {
            writer.write(WorkerMessage(type: WorkerMessageType.error, modelId: modelId, errorMessage: error.localizedDescription))
            NovaMLXLog.error("[Worker] Failed to load \(modelId): \(error)")
        }
    }

    // MARK: - Unload

    private static func handleUnload(_ msg: WorkerMessage, engine: MLXEngine, writer: LineWriter) async {
        guard let modelId = msg.modelId else { return }
        let identifier = ModelIdentifier(id: modelId, family: .other)
        engine.unloadModel(identifier)
        await engine.applyResourceLimits()
        writer.write(WorkerMessage(type: WorkerMessageType.unloaded, modelId: modelId))
        NovaMLXLog.info("[Worker] Unloaded model: \(modelId)")
    }

    // MARK: - Generate

    private static func handleGenerate(
        _ msg: WorkerMessage,
        engine: MLXEngine,
        batcher: ContinuousBatcher,
        fusedScheduler: FusedBatchScheduler,
        writer: LineWriter
    ) async {
        guard let codableReq = msg.request else {
            writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: "Missing request"))
            return
        }

        let request = codableReq.toInferenceRequest()
        let reqTag = request.id.uuidString.prefix(8)
        try? await NovaMLXConfiguration.shared.loadFromStore()
        await engine.applyResourceLimits()

        do {
            let result: InferenceResult
            switch resolveScheduler(request: request, engine: engine) {
            case .continuousBatcher(let reason):
                NovaMLXLog.info("[Worker:\(reqTag)] → ContinuousBatcher (reason=\(reason))")
                result = try await batcher.submit(request)
            case .fused:
                if let kind = GrammarFusedRoute.hostKind(of: request) {
                    NovaMLXLog.info("[Worker:\(reqTag)] → FusedBatchScheduler (grammar=\(kind))")
                } else {
                    NovaMLXLog.info("[Worker:\(reqTag)] → FusedBatchScheduler")
                }
                result = try await fusedScheduler.submit(request)
            }
            NovaMLXLog.info("[Worker:\(reqTag)] writing result tokens=\(result.completionTokens) chars=\(result.text.utf8.count)")
            let wrote = writer.write(WorkerMessage(type: WorkerMessageType.result, requestId: msg.requestId, result: result))
            NovaMLXLog.info("[Worker:\(reqTag)] result written ok=\(wrote)")
        } catch {
            writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: error.localizedDescription))
        }
        engine.reapVLMWorkingSets()
    }

    // MARK: - Stream

    private static func handleStream(
        _ msg: WorkerMessage,
        engine: MLXEngine,
        batcher: ContinuousBatcher,
        fusedScheduler: FusedBatchScheduler,
        writer: LineWriter
    ) async {
        guard let codableReq = msg.request else {
            writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: "Missing request"))
            return
        }

        let request = codableReq.toInferenceRequest()
        let reqTag = request.id.uuidString.prefix(8)
        try? await NovaMLXConfiguration.shared.loadFromStore()
        await engine.applyResourceLimits()

        let tokenStream: AsyncThrowingStream<Token, Error>
        switch resolveScheduler(request: request, engine: engine) {
        case .continuousBatcher(let reason):
            NovaMLXLog.info("[Worker:\(reqTag)] → ContinuousBatcher stream (reason=\(reason))")
            tokenStream = batcher.submitStream(request)
        case .fused:
            if let kind = GrammarFusedRoute.hostKind(of: request) {
                NovaMLXLog.info("[Worker:\(reqTag)] → FusedBatchScheduler stream (grammar=\(kind))")
            } else {
                NovaMLXLog.info("[Worker:\(reqTag)] → FusedBatchScheduler stream")
            }
            tokenStream = fusedScheduler.submitStream(request)
        }

        var promptTokens = 0
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var firstTokenLogged = false
        let workerStreamStart = ContinuousClock.now

        do {
            for try await token in tokenStream {
                completionTokens += 1
                if !firstTokenLogged {
                    firstTokenLogged = true
                    let elapsed = ContinuousClock.now - workerStreamStart
                    let ms = Double(elapsed.components.seconds) * 1000 + Double(elapsed.components.attoseconds) / 1e15
                    NovaMLXLog.info("[Worker:\(reqTag)] First token yielded by scheduler (after \(String(format: "%.0f", ms))ms since dispatch)")
                }
                if completionTokens % 50 == 0 {
                    let elapsed = ContinuousClock.now - workerStreamStart
                    let sec = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
                    let tps = sec > 0 ? Double(completionTokens) / sec : 0
                    NovaMLXLog.info("[Worker:\(reqTag)] Progress: \(completionTokens) tokens, \(String(format: "%.1f", tps)) tok/s")
                }
                if let fr = token.finishReason {
                    finishReason = fr
                }
                if let pt = token.promptTokens {
                    promptTokens = pt
                }
                writer.write(WorkerMessage(
                    type: WorkerMessageType.token,
                    requestId: msg.requestId,
                    token: token
                ))
            }

            writer.write(WorkerMessage(
                type: WorkerMessageType.done,
                requestId: msg.requestId,
                finishReason: finishReason,
                promptTokens: promptTokens,
                completionTokens: completionTokens
            ))
            let totalElapsed = ContinuousClock.now - workerStreamStart
            let sec = Double(totalElapsed.components.seconds) + Double(totalElapsed.components.attoseconds) / 1e18
            NovaMLXLog.info("[Worker:\(reqTag)] Stream done — tokens=\(completionTokens) finish=\(finishReason.rawValue) total=\(String(format: "%.2f", sec))s")
        } catch {
            NovaMLXLog.error("[Worker:\(reqTag)] Stream errored: \(error) — \(type(of: error))")
            writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: error.localizedDescription))
        }
    }

    // MARK: - Routing

    private static func resolveScheduler(
        request: InferenceRequest, engine: MLXEngine
    ) -> GrammarFusedRoute.Choice {
        let container = engine.getContainer(for: request.model)

        let isVLM = container?.config.modelType == .vlm
        let hasLinearAttention = container?.config.hasLinearAttention == true
        let hasDraftModel = request.draftModel != nil
        let nativeMtp = container?.hasNativeMtp == true
        // Explicit use_mtp=false forces the serial TokenIterator path (ContinuousBatcher).
        // Needed so greedy lossless compares (plain vs DSpark) share one decode kernel:
        // FusedBatchScheduler's single-token path and DSpark's dflashVerify can flip
        // near-tie argmax even when both are greedy step-by-step.
        let forceSerialDecode = request.useNativeMtp == false

        return GrammarFusedRoute.choose(GrammarFusedRoute.inputs(
            request: request,
            isVLM: isVLM,
            hasLinearAttention: hasLinearAttention,
            hasDraftModel: hasDraftModel,
            hasNativeMTP: nativeMtp,
            forceSerialDecode: forceSerialDecode))
    }
}

// MARK: - Thread-Safe Line Writer

final class LineWriter: @unchecked Sendable {
    private let lock = NSLock()

    /// Length-prefixed frame: 10 decimal digits, a newline, then that many
    /// JSON bytes. A raw `0x0A` inside the JSON cannot split the message, and
    /// a short `write` is retried until every byte is in the pipe.
    @discardableResult
    func write(_ message: WorkerMessage) -> Bool {
        let payload: Data
        do {
            payload = try JSONEncoder().encode(message)
        } catch {
            NovaMLXLog.error("[Worker] failed to encode \(message.type): \(error)")
            guard message.type != WorkerMessageType.error else { return false }
            return write(WorkerMessage(
                type: WorkerMessageType.error,
                requestId: message.requestId,
                errorMessage: "Failed to encode worker \(message.type): \(error)"
            ))
        }
        let header = String(format: "%010d\n", payload.count)
        var frame = Data(header.utf8)
        frame.append(payload)
        let ok = writeRaw(frame)
        if !ok {
            NovaMLXLog.error("[Worker] stdout write failed type=\(message.type) bytes=\(frame.count) errno=\(errno)")
        }
        return ok
    }

    private func writeRaw(_ data: Data) -> Bool {
        let fd = STDOUT_FILENO
        let flags = fcntl(fd, F_GETFL)
        if flags >= 0, (flags & O_NONBLOCK) != 0 {
            _ = fcntl(fd, F_SETFL, flags & ~O_NONBLOCK)
        }
        lock.lock()
        defer { lock.unlock() }
        return data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return false }
            var offset = 0
            let bytes = raw.count
            while offset < bytes {
                let wrote = Darwin.write(fd, base.advanced(by: offset), bytes - offset)
                if wrote > 0 {
                    offset += wrote
                } else if wrote < 0 && errno == EINTR {
                    continue
                } else if wrote < 0 && errno == EAGAIN {
                    continue
                } else {
                    return false
                }
            }
            return true
        }
    }
}
