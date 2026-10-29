import Foundation
import Darwin
import NovaMLXCore
import NovaMLXUtils

public struct WorkerMemoryStats: Sendable {
    public let currentBytes: UInt64
    public let softLimitBytes: UInt64
    public let hardLimitBytes: UInt64
    public let utilization: Double
    public let evictions: UInt64
    public let cpuUsage: Double
    public var fusedActiveSequences: Int = 0
    public var fusedPeakBatchWidth: Int = 0
    public var fusedPeakForwardWidth: Int = 0
    public var fusedForwardCount: UInt64 = 0
}

public final class WorkerSupervisor: @unchecked Sendable {
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var stderrFile: FileHandle?
    private var lastStderrLines: [String] = []
    private let lock = NSLock()

    // Pending generate requests: requestId → continuation
    private var pendingRequests: [String: CheckedContinuation<WorkerMessage, Error>] = [:]

    // Active stream handlers: requestId → AsyncThrowingStream continuation
    private var streamContinuations: [String: AsyncThrowingStream<Token, Error>.Continuation] = [:]

    // Stream metadata accumulation
    private var streamFinishReasons: [String: FinishReason] = [:]
    private var streamCompletionTokens: [String: Int] = [:]
    // Per-request dispatch timestamp — used to measure IPC latency from
    // sendStream() writeMessage → first WorkerMessageType.token reply.
    private var streamDispatchStarts: [String: ContinuousClock.Instant] = [:]
    private var streamFirstTokenLogged: [String: Bool] = [:]

    // Progress callbacks for model loads
    private var loadProgressCallbacks: [String: @Sendable (LoadPhase) -> Void] = [:]

    // Latest memory stats from worker
    public private(set) var latestMemoryStats: WorkerMemoryStats?

    private let workerBinaryPath: String

    public private(set) var isRunning = false
    /// Set before the stdout/stderr handles are closed so an intentional
    /// stop is not reported as a worker crash.
    private var readersStopped = false
    /// After an oversized stdout record, skip bytes until the next newline
    /// so the following message can still be decoded.
    private var discardStdoutUntilNewline = false
    private static let maxStdoutLineBytes = 16 * 1024 * 1024
    private static let readChunkBytes = 64 * 1024
    private var pendingLoadModelId: String?
    private var lastCrashTime: Date?
    private static let restartCooldown: TimeInterval = 2.0

    public var onCrash: (() -> Void)?

    /// Number of requests currently being processed by the worker
    public var activeRequestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pendingRequests.count + streamContinuations.count
    }

    public init(workerBinaryPath: String) {
        self.workerBinaryPath = workerBinaryPath
    }

    // MARK: - Lifecycle

    public func start() throws {
        guard !isRunning else { return }

        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: workerBinaryPath)
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = stderr

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
        self.stderrPipe = stderr
        self.isRunning = true
        self.readersStopped = false
        self.discardStdoutUntilNewline = false

        startReader(stdout: stdout)
        startStderrReader(stderr, pid: proc.processIdentifier)

        NovaMLXLog.info("[WorkerSupervisor] Worker started (pid=\(proc.processIdentifier))")
    }

    /// Restart the worker if it's not running. Returns true if worker is ready.
    public func ensureRunning() -> Bool {
        lock.lock()
        let running = isRunning
        lock.unlock()
        if running { return true }

        // Respect crash cooldown to avoid rapid restart loops
        if let lastCrash = lastCrashTime {
            let elapsed = Date().timeIntervalSince(lastCrash)
            if elapsed < Self.restartCooldown {
                Thread.sleep(forTimeInterval: Self.restartCooldown - elapsed)
            }
        }

        NovaMLXLog.warning("[WorkerSupervisor] Worker not running — attempting restart")
        do {
            try start()
            NovaMLXLog.info("[WorkerSupervisor] Worker restarted successfully")
            return true
        } catch {
            NovaMLXLog.error("[WorkerSupervisor] Worker restart failed: \(error)")
            return false
        }
    }

    public func stop() {
        lock.lock()
        readersStopped = true
        let stdout = stdoutPipe
        let stderr = stderrPipe
        let proc = process
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil
        isRunning = false
        readBuffer = Data()
        let pending = pendingRequests
        pendingRequests.removeAll()
        let streams = streamContinuations
        clearStreamStateLocked()
        lock.unlock()

        // Unblock the dedicated readers. `read(upToCount:)` returns on close.
        try? stdout?.fileHandleForReading.close()
        try? stderr?.fileHandleForReading.close()
        proc?.terminate()
        failWaiters(pending: pending, streams: streams, error: NovaMLXError.inferenceFailed("Worker stopped"))
    }

    // MARK: - Send Messages

    public struct LoadFlags: Sendable {
        public let hybrid: Bool
        public let nativeMtp: Bool
    }

    @discardableResult
    public func sendLoad(modelId: String, path: String, config: ModelConfig, progress: (@Sendable (LoadPhase) -> Void)? = nil) async throws -> LoadFlags {
        let msg = WorkerMessage(type: WorkerMessageType.load, modelId: modelId, modelPath: path, modelConfig: config)
        pendingLoadModelId = modelId

        // Store progress callback so handleMessage can forward loadProgress frames.
        // Must set before entering async context.
        setProgressCallback(progress, for: modelId)

        let response = try await sendAndWait(msg, requestId: modelId)
        pendingLoadModelId = nil

        // Clean up progress callback
        removeProgressCallback(for: modelId)

        guard response.type == WorkerMessageType.loaded else {
            throw NovaMLXError.modelLoadFailed(modelId, underlying: NovaMLXError.apiError(response.errorMessage ?? "Unknown load error"))
        }
        return LoadFlags(
            hybrid: response.hasLinearAttention ?? false,
            nativeMtp: response.hasNativeMtp ?? false
        )
    }

    /// Stores a progress callback. Safe to call from any context (sync).
    private func setProgressCallback(_ callback: (@Sendable (LoadPhase) -> Void)?, for modelId: String) {
        guard let callback = callback else { return }
        lock.lock()
        loadProgressCallbacks[modelId] = callback
        lock.unlock()
    }

    /// Removes a progress callback. Safe to call from any context (sync).
    private func removeProgressCallback(for modelId: String) {
        lock.lock()
        loadProgressCallbacks.removeValue(forKey: modelId)
        lock.unlock()
    }

    public func sendUnload(modelId: String) async throws {
        let msg = WorkerMessage(type: WorkerMessageType.unload, modelId: modelId)
        let response = try await sendAndWait(msg, requestId: modelId)
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
        let reqTag = String(requestId.prefix(8))
        let msg = WorkerMessage(type: WorkerMessageType.stream, requestId: requestId, request: codable)
        let dispatchStart = ContinuousClock.now

        return AsyncThrowingStream { continuation in
            if !self.ensureRunning() {
                NovaMLXLog.error("[Sup:\(reqTag)] sendStream: worker not running, restart failed")
                continuation.finish(throwing: NovaMLXError.inferenceFailed("Worker not running and restart failed"))
                return
            }
            self.lock.lock()
            self.streamContinuations[requestId] = continuation
            self.streamCompletionTokens[requestId] = 0
            self.streamDispatchStarts[requestId] = dispatchStart
            self.lock.unlock()
            NovaMLXLog.info("[Sup:\(reqTag)] sendStream: queuing stream msg to worker (model=\(request.model))")

            continuation.onTermination = { [weak self] reason in
                self?.lock.lock()
                self?.streamContinuations.removeValue(forKey: requestId)
                self?.streamFinishReasons.removeValue(forKey: requestId)
                self?.streamCompletionTokens.removeValue(forKey: requestId)
                self?.streamDispatchStarts.removeValue(forKey: requestId)
                self?.streamFirstTokenLogged.removeValue(forKey: requestId)
                self?.lock.unlock()
                // Client disconnect (SSE writer error, relay cancel): tell the
                // worker to abort the generation. Without this the worker keeps
                // decoding to EOS/token-cap after every disconnected reader —
                // a zombie holding the serial decode slot, so the NEXT request
                // queues behind work nobody will read. `.finished` is normal
                // completion (worker already ended) — no abort needed.
                if case .cancelled = reason, let uuid = UUID(uuidString: requestId) {
                    self?.sendAbort(requestId: uuid)
                }
            }

            do {
                try self.writeMessage(msg)
                NovaMLXLog.info("[Sup:\(reqTag)] sendStream: msg written to worker stdin pipe")
            } catch {
                NovaMLXLog.error("[Sup:\(reqTag)] sendStream: writeMessage failed: \(error)")
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
        if !isRunning {
            guard ensureRunning() else {
                throw NovaMLXError.inferenceFailed("Worker not running and restart failed")
            }
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

    private func startStderrReader(_ pipe: Pipe, pid: Int32) {
        let url = NovaMLXPaths.workerStderrFile
        let fm = FileManager.default
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let size = (try? fm.attributesOfItem(atPath: url.path)[.size] as? NSNumber)?.intValue,
            size > 16 * 1024 * 1024
        {
            try? fm.removeItem(at: url)
        }
        if !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        let file = try? FileHandle(forWritingTo: url)
        file?.seekToEndOfFile()
        let stamp = ISO8601DateFormatter().string(from: Date())
        if let banner = "\n===== worker pid=\(pid) \(stamp) =====\n".data(using: .utf8) {
            file?.write(banner)
        }
        lock.lock()
        stderrFile = file
        lastStderrLines.removeAll()
        lock.unlock()

        let handle = pipe.fileHandleForReading
        handle.readabilityHandler = nil
        let thread = Thread { [weak self] in
            self?.readLoop(handle, kind: .stderr)
        }
        thread.name = "novamlx-worker-stderr"
        thread.start()
    }

    private enum PipeKind { case stdout, stderr }

    /// Blocking read loop.
    ///
    /// `FileHandle.read(upToCount:)` waits until that many bytes arrive or the
    /// pipe closes. A `loaded` ack and the 5s memory-stats lines are a few
    /// hundred bytes, so the host sat in that call forever: the worker logged
    /// `Loaded model` and `/health` stayed at `loadedModels: 0`.
    /// `FileHandle.availableData` inside `readabilityHandler` is the other
    /// failure: the handler suspends the dispatch source and a result larger
    /// than the pipe buffer never finishes.
    ///
    /// POSIX `read` on a blocking pipe returns whatever is already buffered
    /// (at least one byte) and blocks only when the pipe is empty. Short
    /// control messages are delivered immediately, and a large OCR line is
    /// drained while the worker is still writing it.
    private func startReader(stdout: Pipe) {
        let handle = stdout.fileHandleForReading
        handle.readabilityHandler = nil
        NovaMLXLog.info("[WorkerSupervisor] Reader started (dedicated thread)")
        let thread = Thread { [weak self] in
            self?.readLoop(handle, kind: .stdout)
        }
        thread.name = "novamlx-worker-stdout"
        thread.start()
    }

    private func readLoop(_ handle: FileHandle, kind: PipeKind) {
        var buffer = Data()
        let fd = handle.fileDescriptor
        var scratch = [UInt8](repeating: 0, count: Self.readChunkBytes)
        while true {
            let n = scratch.withUnsafeMutableBytes { raw -> Int in
                guard let base = raw.baseAddress else { return 0 }
                return Darwin.read(fd, base, raw.count)
            }
            if n > 0 {
                buffer.append(contentsOf: scratch.prefix(n))
            } else if n < 0 && errno == EINTR {
                continue
            } else {
                // n == 0 is EOF. Any other error (EBADF from stop()'s close)
                // ends the loop too.
                notePipeClosed(kind)
                return
            }
            switch kind {
            case .stdout:
                drainStdout(&buffer)
            case .stderr:
                drainStderr(&buffer)
            }
        }
    }

    private func notePipeClosed(_ kind: PipeKind) {
        lock.lock()
        let stopped = readersStopped
        lock.unlock()
        guard !stopped, kind == .stdout else { return }
        handleWorkerCrash()
    }

    private func drainStdout(_ buffer: inout Data) {
        if discardStdoutUntilNewline {
            guard let newline = buffer.firstIndex(of: 0x0A) else {
                buffer.removeAll(keepingCapacity: true)
                return
            }
            buffer.removeSubrange(..<buffer.index(after: newline))
            discardStdoutUntilNewline = false
        }
        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = Data(buffer[..<newline])
            buffer.removeSubrange(..<buffer.index(after: newline))
            guard !line.isEmpty else { continue }
            guard let msg = try? JSONDecoder().decode(WorkerMessage.self, from: line) else {
                let preview = String(data: line.prefix(180), encoding: .utf8) ?? "?"
                NovaMLXLog.warning("[WorkerSupervisor] Decode failed (\(line.count) bytes): \(preview)")
                continue
            }
            handleMessage(msg)
        }
        if buffer.count > Self.maxStdoutLineBytes {
            NovaMLXLog.error("[WorkerSupervisor] stdout record exceeded \(Self.maxStdoutLineBytes) bytes without a newline")
            buffer.removeAll(keepingCapacity: true)
            discardStdoutUntilNewline = true
            failPendingLineOverflow()
        }
    }

    private func drainStderr(_ buffer: inout Data) {
        var lines: [Data] = []
        while let newline = buffer.firstIndex(of: 0x0A) {
            lines.append(Data(buffer[..<newline]))
            buffer.removeSubrange(..<buffer.index(after: newline))
        }
        if buffer.count > Self.maxStdoutLineBytes {
            buffer.removeAll(keepingCapacity: true)
        }
        guard !lines.isEmpty else { return }
        var echoed: [String] = []
        echoed.reserveCapacity(lines.count)
        lock.lock()
        for lineData in lines where !lineData.isEmpty {
            stderrFile?.write(lineData)
            stderrFile?.write(Data([0x0A]))
            let stored = String(data: lineData.prefix(2_048), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            guard !stored.isEmpty else { continue }
            lastStderrLines.append(stored)
            if lastStderrLines.count > 80 {
                lastStderrLines.removeFirst(lastStderrLines.count - 80)
            }
            echoed.append(stored)
        }
        lock.unlock()
        for line in echoed {
            NovaMLXLog.error("[Worker stderr] \(line)")
        }
    }

    private func resumeOffReader(
        _ continuation: CheckedContinuation<WorkerMessage, Error>,
        returning message: WorkerMessage
    ) {
        let box = ContinuationResume(continuation)
        DispatchQueue.global(qos: .userInitiated).async {
            box.resume(returning: message)
        }
    }

    private func failPendingLineOverflow() {
        lock.lock()
        let pending = pendingRequests
        pendingRequests.removeAll()
        let streams = streamContinuations
        clearStreamStateLocked()
        lock.unlock()
        failWaiters(
            pending: pending,
            streams: streams,
            error: NovaMLXError.inferenceFailed(
                "Worker result exceeded \(Self.maxStdoutLineBytes) bytes"))
    }

    private func handleMessage(_ msg: WorkerMessage) {
        // Handle memoryStats from worker (no requestId needed)
        if msg.type == WorkerMessageType.memoryStats {
            lock.lock()
            latestMemoryStats = WorkerMemoryStats(
                currentBytes: msg.memoryCurrentBytes ?? 0,
                softLimitBytes: msg.memorySoftLimitBytes ?? 0,
                hardLimitBytes: msg.memoryHardLimitBytes ?? 0,
                utilization: msg.memoryUtilization ?? 0,
                evictions: msg.memoryEvictions ?? 0,
                cpuUsage: msg.cpuUsage ?? 0,
                fusedActiveSequences: msg.fusedActiveSequences ?? 0,
                fusedPeakBatchWidth: msg.fusedPeakBatchWidth ?? 0,
                fusedPeakForwardWidth: msg.fusedPeakForwardWidth ?? 0,
                fusedForwardCount: msg.fusedForwardCount ?? 0
            )
            lock.unlock()
            return
        }

        guard let requestId = msg.requestId else {
            // No requestId — likely load/unload/loadProgress response
            if msg.type == WorkerMessageType.loaded || msg.type == WorkerMessageType.unloaded {
                let key = msg.modelId ?? ""
                lock.lock()
                let callback = loadProgressCallbacks.removeValue(forKey: key)
                let cont = pendingRequests.removeValue(forKey: key)
                lock.unlock()
                // Same rule as generate results: the host's load bookkeeping
                // continues on the resuming thread. Doing it here stalls this
                // reader, so the next stdout line (memory stats, OCR result)
                // sits unread.
                if let cont {
                    resumeOffReader(cont, returning: msg)
                }
                if let callback = callback, msg.type == WorkerMessageType.loaded {
                    callback(.ready)
                }
            } else if msg.type == WorkerMessageType.loadProgress {
                // Forward load progress from worker to host
                let key = msg.modelId ?? ""
                let phase: LoadPhase? = msg.loadPhase.flatMap { LoadPhase(rawValue: $0) }
                lock.lock()
                let callback = loadProgressCallbacks[key]
                lock.unlock()
                if let callback = callback, let phase = phase {
                    callback(phase)
                }
            } else if msg.type == WorkerMessageType.error {
                // Worker rejected load/unload — resume the waiting continuation
                let key = msg.modelId ?? ""
                lock.lock()
                let cont = pendingRequests.removeValue(forKey: key)
                lock.unlock()
                if let cont {
                    resumeOffReader(cont, returning: msg)
                } else {
                    NovaMLXLog.warning("[WorkerSupervisor] Unhandled worker error (no pending request for '\(key)'): \(msg.errorMessage ?? "unknown")")
                }
            }
            return
        }

        switch msg.type {
        case WorkerMessageType.result, WorkerMessageType.error:
            lock.lock()
            let cont = pendingRequests.removeValue(forKey: requestId)
            lock.unlock()
            guard let cont else { return }
            // The HTTP task continues on the resuming thread until its next
            // await. Packaging the OCR body here used to run inside the pipe
            // reader and wedge the host.
            resumeOffReader(cont, returning: msg)

        case WorkerMessageType.token:
            let tokenToYield: Token?
            var ipcLatencyMs: Double? = nil
            var isFirstToken = false
            lock.lock()
            if let token = msg.token, streamContinuations[requestId] != nil {
                if let fr = token.finishReason {
                    streamFinishReasons[requestId] = fr
                }
                if var count = streamCompletionTokens[requestId] {
                    count += 1
                    streamCompletionTokens[requestId] = count
                    // Log every 50 tokens OR the first one (for IPC latency).
                    if count == 1 || count % 50 == 0 {
                        isFirstToken = (count == 1)
                        if let start = streamDispatchStarts[requestId] {
                            let dur = ContinuousClock.now - start
                            ipcLatencyMs = Double(dur.components.seconds) * 1000 + Double(dur.components.attoseconds) / 1e15
                        }
                    }
                }
                tokenToYield = token
            } else {
                tokenToYield = nil
            }
            lock.unlock()

            if let token = tokenToYield, let cont = streamContinuations[requestId] {
                cont.yield(token)
                if isFirstToken, let ms = ipcLatencyMs {
                    NovaMLXLog.info("[Sup:\(String(requestId.prefix(8)))] First token from worker after \(String(format: "%.0f", ms))ms IPC latency")
                } else if let ms = ipcLatencyMs {
                    let count = streamCompletionTokens[requestId] ?? 0
                    NovaMLXLog.info("[Sup:\(String(requestId.prefix(8)))] Progress: \(count) tokens from worker (cumulative IPC \(String(format: "%.0f", ms))ms)")
                }
            }

        case WorkerMessageType.done:
            let contToFinish: AsyncThrowingStream<Token, Error>.Continuation?
            var totalTokens = 0
            var finishReason: FinishReason? = nil
            var totalTime: Duration? = nil
            lock.lock()
            contToFinish = streamContinuations.removeValue(forKey: requestId)
            finishReason = streamFinishReasons.removeValue(forKey: requestId)
            totalTokens = streamCompletionTokens.removeValue(forKey: requestId) ?? 0
            if let start = streamDispatchStarts.removeValue(forKey: requestId) {
                totalTime = ContinuousClock.now - start
            }
            streamFirstTokenLogged.removeValue(forKey: requestId)
            lock.unlock()

            contToFinish?.finish()
            let totalSec = totalTime.map { Double($0.components.seconds) + Double($0.components.attoseconds) / 1e18 } ?? 0
            let avgTps = totalSec > 0 ? Double(totalTokens) / totalSec : 0
            NovaMLXLog.info("[Sup:\(String(requestId.prefix(8)))] Done processed — tokens=\(totalTokens) finish=\(finishReason?.rawValue ?? "nil") avgTPS=\(String(format: "%.1f", avgTps)) total=\(String(format: "%.2f", totalSec))s hadCont=\(contToFinish != nil)")

        default:
            break
        }
    }

    /// `onTermination` and `activeRequestCount` take this lock. Never resume
    /// or finish a continuation while holding it.
    private func clearStreamStateLocked() {
        streamContinuations.removeAll()
        streamFinishReasons.removeAll()
        streamCompletionTokens.removeAll()
        streamDispatchStarts.removeAll()
        streamFirstTokenLogged.removeAll()
    }

    private func failWaiters(
        pending: [String: CheckedContinuation<WorkerMessage, Error>],
        streams: [String: AsyncThrowingStream<Token, Error>.Continuation],
        error: Error
    ) {
        for (_, cont) in pending {
            cont.resume(throwing: error)
        }
        for (_, cont) in streams {
            cont.finish(throwing: error)
        }
    }

    private func handleWorkerCrash() {
        lock.lock()
        let wasRunning = isRunning
        let crashedModelId = pendingLoadModelId
        isRunning = false
        lastCrashTime = Date()
        pendingLoadModelId = nil
        let pending = pendingRequests
        pendingRequests.removeAll()
        let streams = streamContinuations
        clearStreamStateLocked()
        if let pipe = stderrPipe {
            pipe.fileHandleForReading.readabilityHandler = nil
        }
        stderrPipe = nil
        let proc = process
        process = nil
        let stderrTail = Array(lastStderrLines.suffix(20))
        lock.unlock()

        let crashMsg = crashedModelId.map { "Worker crashed while loading \($0)" } ?? "Worker crashed"
        failWaiters(
            pending: pending, streams: streams,
            error: NovaMLXError.inferenceFailed(crashMsg))

        if wasRunning {
            var detail = crashedModelId.map { " While loading: \($0)" } ?? ""
            if let proc {
                let how: String
                switch proc.terminationReason {
                case .exit: how = "exit"
                case .uncaughtSignal: how = "signal"
                @unknown default: how = "term"
                }
                if proc.isRunning {
                    detail += " pid=\(proc.processIdentifier) still-running"
                } else {
                    detail += " pid=\(proc.processIdentifier) \(how)=\(proc.terminationStatus)"
                }
            }
            if !stderrTail.isEmpty {
                detail += " stderr=\(stderrTail.joined(separator: " | "))"
            }
            NovaMLXLog.error("[WorkerSupervisor] Worker crashed!\(detail)")
            onCrash?()
        }
    }
}

/// Resumes a worker continuation off the stdout reader thread.
private final class ContinuationResume: @unchecked Sendable {
    private var continuation: CheckedContinuation<WorkerMessage, Error>?

    init(_ continuation: CheckedContinuation<WorkerMessage, Error>) {
        self.continuation = continuation
    }

    func resume(returning message: WorkerMessage) {
        continuation?.resume(returning: message)
        continuation = nil
    }
}
