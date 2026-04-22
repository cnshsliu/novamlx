import Foundation
import NovaMLXCore
import NovaMLXUtils

public final class WorkerSupervisor: @unchecked Sendable {
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private let lock = NSLock()

    // Pending generate requests: requestId → continuation
    private var pendingRequests: [String: CheckedContinuation<WorkerMessage, Error>] = [:]

    // Active stream handlers: requestId → AsyncThrowingStream continuation
    private var streamContinuations: [String: AsyncThrowingStream<Token, Error>.Continuation] = [:]

    // Stream metadata accumulation
    private var streamFinishReasons: [String: FinishReason] = [:]
    private var streamCompletionTokens: [String: Int] = [:]

    private let workerBinaryPath: String

    public private(set) var isRunning = false

    public init(workerBinaryPath: String) {
        self.workerBinaryPath = workerBinaryPath
    }

    // MARK: - Lifecycle

    public func start() throws {
        guard !isRunning else { return }

        let stdin = Pipe()
        let stdout = Pipe()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: workerBinaryPath)
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = FileHandle.nullDevice

        // Terminate worker when parent dies
        proc.terminationHandler = { [weak self] _ in
            Task { @MainActor in
                self?.handleWorkerCrash()
            }
        }

        try proc.run()

        self.process = proc
        self.stdinPipe = stdin
        self.stdoutPipe = stdout
        self.isRunning = true

        startReader(stdout: stdout)

        NovaMLXLog.info("[WorkerSupervisor] Worker started (pid=\(proc.processIdentifier))")
    }

    public func stop() {
        lock.lock()
        defer { lock.unlock() }

        if let pipe = stdoutPipe {
            pipe.fileHandleForReading.readabilityHandler = nil
        }
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        isRunning = false
        readBuffer = Data()

        // Fail all pending requests
        for (_, cont) in pendingRequests {
            cont.resume(throwing: NovaMLXError.inferenceFailed("Worker stopped"))
        }
        pendingRequests.removeAll()

        for (_, cont) in streamContinuations {
            cont.finish(throwing: NovaMLXError.inferenceFailed("Worker stopped"))
        }
        streamContinuations.removeAll()
    }

    // MARK: - Send Messages

    public func sendLoad(modelId: String, path: String, config: ModelConfig) async throws {
        let msg = WorkerMessage(type: WorkerMessageType.load, modelId: modelId, modelPath: path, modelConfig: config)
        let response = try await sendAndWait(msg, requestId: modelId)
        guard response.type == WorkerMessageType.loaded else {
            throw NovaMLXError.modelLoadFailed(modelId, underlying: NovaMLXError.apiError(response.errorMessage ?? "Unknown load error"))
        }
    }

    public func sendUnload(modelId: String) async throws {
        let msg = WorkerMessage(type: WorkerMessageType.unload, modelId: modelId)
        let response = try await sendAndWait(msg, requestId: "unload-\(modelId)")
        guard response.type == WorkerMessageType.unloaded else {
            throw NovaMLXError.inferenceFailed(response.errorMessage ?? "Unload failed")
        }
    }

    public func sendGenerate(_ request: InferenceRequest) async throws -> InferenceResult {
        let codable = CodableInferenceRequest(from: request)
        let requestId = request.id.uuidString
        let msg = WorkerMessage(type: WorkerMessageType.generate, requestId: requestId, request: codable)

        let response = try await sendAndWait(msg, requestId: requestId)
        switch response.type {
        case WorkerMessageType.result:
            guard let result = response.result else {
                throw NovaMLXError.inferenceFailed("Worker returned result without data")
            }
            return result
        case WorkerMessageType.error:
            throw NovaMLXError.inferenceFailed(response.errorMessage ?? "Unknown error")
        default:
            throw NovaMLXError.inferenceFailed("Unexpected response type: \(response.type)")
        }
    }

    public func sendStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        let codable = CodableInferenceRequest(from: request)
        let requestId = request.id.uuidString
        let msg = WorkerMessage(type: WorkerMessageType.stream, requestId: requestId, request: codable)

        return AsyncThrowingStream { continuation in
            self.lock.lock()
            self.streamContinuations[requestId] = continuation
            self.streamCompletionTokens[requestId] = 0
            self.lock.unlock()

            continuation.onTermination = { [weak self] _ in
                self?.lock.lock()
                self?.streamContinuations.removeValue(forKey: requestId)
                self?.streamFinishReasons.removeValue(forKey: requestId)
                self?.streamCompletionTokens.removeValue(forKey: requestId)
                self?.lock.unlock()
            }

            do {
                try self.writeMessage(msg)
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    public func sendAbort(requestId: UUID) {
        let msg = WorkerMessage(type: WorkerMessageType.abort, requestId: requestId.uuidString)
        try? writeMessage(msg)
    }

    // MARK: - Private

    private func writeMessage(_ msg: WorkerMessage) throws {
        guard let pipe = stdinPipe else {
            throw NovaMLXError.inferenceFailed("Worker not running")
        }
        var data = try JSONEncoder().encode(msg)
        data.append(0x0A) // newline
        pipe.fileHandleForWriting.write(data)
    }

    private func sendAndWait(_ msg: WorkerMessage, requestId: String) async throws -> WorkerMessage {
        guard isRunning else {
            throw NovaMLXError.inferenceFailed("Worker not running")
        }

        return try await withCheckedThrowingContinuation { continuation in
            lock.lock()
            pendingRequests[requestId] = continuation
            lock.unlock()

            do {
                try writeMessage(msg)
            } catch {
                lock.lock()
                pendingRequests.removeValue(forKey: requestId)
                lock.unlock()
                continuation.resume(throwing: error)
            }
        }
    }

    private var readBuffer = Data()

    private func startReader(stdout: Pipe) {
        let handle = stdout.fileHandleForReading
        NovaMLXLog.info("[WorkerSupervisor] Reader started (readabilityHandler)")

        handle.readabilityHandler = { [weak self] handle in
            guard let self = self else { return }
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                self.handleWorkerCrash()
                return
            }

            self.readBuffer.append(data)

            while let newlineIdx = self.readBuffer.firstIndex(of: 0x0A) {
                let lineData = self.readBuffer[..<newlineIdx]
                self.readBuffer = Data(self.readBuffer[(newlineIdx + 1)...])

                guard !lineData.isEmpty else { continue }
                guard let msg = try? JSONDecoder().decode(WorkerMessage.self, from: lineData) else {
                    NovaMLXLog.warning("[WorkerSupervisor] Decode failed: \(String(data: lineData, encoding: .utf8) ?? "?")")
                    continue
                }

                self.handleMessage(msg)
            }
        }
    }

    private func handleMessage(_ msg: WorkerMessage) {
        guard let requestId = msg.requestId else {
            // No requestId — likely load/unload response
            if msg.type == WorkerMessageType.loaded || msg.type == WorkerMessageType.unloaded {
                let key = msg.modelId ?? ""
                lock.lock()
                if let cont = pendingRequests.removeValue(forKey: key) {
                    cont.resume(returning: msg)
                }
                lock.unlock()
            }
            return
        }

        switch msg.type {
        case WorkerMessageType.result, WorkerMessageType.error:
            lock.lock()
            if let cont = pendingRequests.removeValue(forKey: requestId) {
                cont.resume(returning: msg)
            }
            lock.unlock()

        case WorkerMessageType.token:
            let tokenToYield: Token?
            lock.lock()
            if let token = msg.token, let cont = streamContinuations[requestId] {
                if let fr = token.finishReason {
                    streamFinishReasons[requestId] = fr
                }
                if var count = streamCompletionTokens[requestId] {
                    count += 1
                    streamCompletionTokens[requestId] = count
                }
                tokenToYield = token
            } else {
                tokenToYield = nil
            }
            lock.unlock()

            if let token = tokenToYield, let cont = streamContinuations[requestId] {
                cont.yield(token)
            }

        case WorkerMessageType.done:
            let contToFinish: AsyncThrowingStream<Token, Error>.Continuation?
            lock.lock()
            contToFinish = streamContinuations.removeValue(forKey: requestId)
            streamFinishReasons.removeValue(forKey: requestId)
            streamCompletionTokens.removeValue(forKey: requestId)
            lock.unlock()

            contToFinish?.finish()
            NovaMLXLog.info("[WorkerSupervisor] Done processed: hadCont=\(contToFinish != nil) requestId=\(requestId)")

        default:
            break
        }
    }

    private func handleWorkerCrash() {
        lock.lock()
        let wasRunning = isRunning
        isRunning = false

        // Fail all pending requests
        for (_, cont) in pendingRequests {
            cont.resume(throwing: NovaMLXError.inferenceFailed("Worker crashed"))
        }
        pendingRequests.removeAll()

        // Fail all active streams
        for (_, cont) in streamContinuations {
            cont.finish(throwing: NovaMLXError.inferenceFailed("Worker crashed"))
        }
        streamContinuations.removeAll()
        streamFinishReasons.removeAll()
        streamCompletionTokens.removeAll()

        lock.unlock()

        if wasRunning {
            NovaMLXLog.error("[WorkerSupervisor] Worker crashed! All pending requests failed.")
            // TODO: auto-restart worker
        }
    }
}
