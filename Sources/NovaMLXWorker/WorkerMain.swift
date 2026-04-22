import Foundation
import NovaMLXCore
import NovaMLXUtils
import NovaMLXEngine

@main
struct NovaMLXWorker {
    static func main() async {
        let engine = MLXEngine()
        let batcher = ContinuousBatcher(engine: engine, maxBatchSize: 8)
        let fusedScheduler = FusedBatchScheduler(engine: engine, maxConcurrentPerModel: 4)
        let writer = LineWriter()

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
                    handleUnload(msg, engine: engine, writer: writer)

                case WorkerMessageType.generate:
                    group.addTask {
                        await handleGenerate(msg, engine: engine, batcher: batcher, fusedScheduler: fusedScheduler, writer: writer)
                    }

                case WorkerMessageType.stream:
                    group.addTask {
                        await handleStream(msg, engine: engine, batcher: batcher, fusedScheduler: fusedScheduler, writer: writer)
                    }

                case WorkerMessageType.abort:
                    if let reqId = msg.requestId, let uuid = UUID(uuidString: reqId) {
                        engine.abort(requestId: uuid)
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
            _ = try await engine.loadModel(from: url, config: config)
            writer.write(WorkerMessage(type: WorkerMessageType.loaded, modelId: modelId))
            NovaMLXLog.info("[Worker] Loaded model: \(modelId)")
        } catch {
            writer.write(WorkerMessage(type: WorkerMessageType.error, modelId: modelId, errorMessage: error.localizedDescription))
            NovaMLXLog.error("[Worker] Failed to load \(modelId): \(error)")
        }
    }

    // MARK: - Unload

    private static func handleUnload(_ msg: WorkerMessage, engine: MLXEngine, writer: LineWriter) {
        guard let modelId = msg.modelId else { return }
        let identifier = ModelIdentifier(id: modelId, family: .other)
        engine.unloadModel(identifier)
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

        do {
            let result: InferenceResult
            let scheduler = resolveScheduler(request: request, engine: engine)
            switch scheduler {
            case .batcher:
                NovaMLXLog.info("[Worker:\(reqTag)] → ContinuousBatcher")
                result = try await batcher.submit(request)
            case .fused:
                NovaMLXLog.info("[Worker:\(reqTag)] → FusedBatchScheduler")
                result = try await fusedScheduler.submit(request)
            }
            writer.write(WorkerMessage(type: WorkerMessageType.result, requestId: msg.requestId, result: result))
        } catch {
            writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: error.localizedDescription))
        }
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

        let tokenStream: AsyncThrowingStream<Token, Error>
        let scheduler = resolveScheduler(request: request, engine: engine)
        switch scheduler {
        case .batcher:
            NovaMLXLog.info("[Worker:\(reqTag)] → ContinuousBatcher stream")
            tokenStream = batcher.submitStream(request)
        case .fused:
            NovaMLXLog.info("[Worker:\(reqTag)] → FusedBatchScheduler stream")
            tokenStream = fusedScheduler.submitStream(request)
        }

        let promptTokens = 0
        var completionTokens = 0
        var finishReason: FinishReason = .stop

        do {
            for try await token in tokenStream {
                completionTokens += 1
                if let fr = token.finishReason {
                    finishReason = fr
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
        } catch {
            writer.write(WorkerMessage(type: WorkerMessageType.error, requestId: msg.requestId, errorMessage: error.localizedDescription))
        }
    }

    // MARK: - Routing

    private enum Scheduler { case batcher, fused }

    private static func resolveScheduler(request: InferenceRequest, engine: MLXEngine) -> Scheduler {
        let container = engine.getContainer(for: request.model)

        let isVLM = container?.config.modelType == .vlm
        let hasLinearAttention = container?.config.hasLinearAttention == true

        let needsSpecialized = request.sessionId != nil ||
            request.jsonSchemaDef != nil ||
            request.responseFormat == .jsonObject ||
            request.regexPattern != nil ||
            request.gbnfGrammar != nil ||
            isVLM ||
            hasLinearAttention

        return needsSpecialized ? .batcher : .fused
    }
}

// MARK: - Thread-Safe Line Writer

final class LineWriter: @unchecked Sendable {
    private let lock = NSLock()

    func write(_ message: WorkerMessage) {
        guard let data = try? JSONEncoder().encode(message),
              let line = String(data: data, encoding: .utf8) else { return }
        var output = line
        output.append("\n")
        lock.lock()
        output.withUTF8 { buf in
            _ = Darwin.write(STDOUT_FILENO, buf.baseAddress, buf.count)
        }
        lock.unlock()
    }
}
