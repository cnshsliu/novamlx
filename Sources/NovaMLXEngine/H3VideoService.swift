import CryptoKit
import Foundation
import NovaMLXCore
import NovaMLXUtils

public enum H3VideoJobStatus: String, Codable, Sendable {
    case pending
    case inProgress = "in_progress"
    case completed
    case failed
    case cancelled
    case expired

    public var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled, .expired: return true
        case .pending, .inProgress: return false
        }
    }
}

/// One video generation job. Persisted as `<jobsDir>/<id>/job.json`; the
/// rendered clip lives next to it as `out.mp4`.
public struct H3VideoJob: Codable, Sendable {
    public var id: String
    public var status: H3VideoJobStatus
    public var model: String
    public var prompt: String
    public var width: Int
    public var height: Int
    public var durationSeconds: Int?
    public var seed: Int?
    public var steps: Int
    public var layers: Int
    public var reuse: Int
    /// Original image references (http(s) or data: URLs), downloaded into the
    /// job directory when the runner picks the job up.
    public var firstFrameURL: String?
    public var lastFrameURL: String?
    public var referenceImageURLs: [String]
    public var callbackURL: String?
    public var error: String?
    public var createdAt: Date
    public var startedAt: Date?
    public var finishedAt: Date?
    public var expiredAt: Date?
}

/// Live progress of a running job, parsed from h3's stderr
/// (`\r<phase> <completed>/<total>` updates). Held in memory only — job.json
/// is written on status transitions, not per progress tick.
public struct H3VideoProgress: Codable, Sendable, Equatable {
    public var phase: String
    public var completed: Int
    public var total: Int
    /// 0–100, estimated from phase weights (denoise dominates runtime).
    public var percent: Int
}

/// Parses h3's carriage-return progress stream into `H3VideoProgress`.
///
/// Phase order/weights come from the measured runtime breakdown (standard
/// tier, M4 Max): denoise is ~75% of wall time, video VAE ~12%, the rest is
/// small. Phases not in the table keep the previous estimate.
public struct H3ProgressParser: Sendable {
    /// Cumulative percent at which each phase starts.
    static let phaseStart: [(phase: String, start: Double, end: Double)] = [
        ("tokenizer", 0, 1),
        ("text encoder", 1, 7),
        ("Qwen vision", 7, 9),
        ("video VAE encoder", 9, 11),
        ("audio VAE encoder", 11, 12),
        ("denoise", 12, 87),
        ("video VAE load", 87, 88),
        ("preview VAE load", 88, 89),
        ("video VAE", 89, 97),
        ("audio VAE", 97, 99),
        ("FFmpeg", 99, 100),
    ]

    private var buffer = ""

    public init() {}

    /// Feed a raw stderr chunk; returns the newest parsed progress (if any).
    public mutating func feed(_ chunk: String) -> H3VideoProgress? {
        buffer += chunk
        // Progress updates are \r-separated; keep the incomplete tail.
        // Note: "\r\n" is a single grapheme cluster in Swift, so compare
        // with isNewline rather than == "\r" / == "\n".
        var latest: H3VideoProgress? = nil
        var consumedUpTo = buffer.startIndex
        for i in buffer.indices where buffer[i].isNewline {
            if let parsed = Self.parseLine(String(buffer[consumedUpTo..<i])) {
                latest = parsed
            }
            consumedUpTo = buffer.index(after: i)
        }
        buffer = String(buffer[consumedUpTo...])
        return latest
    }

    /// Parse one `<phase> <completed>/<total>` segment.
    static func parseLine(_ line: String) -> H3VideoProgress? {        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard let match = trimmed.range(
            of: #"^(.+?)\s+(\d+)/(\d+)$"#, options: .regularExpression)
        else { return nil }
        let parts = String(trimmed[match]).split(separator: " ")
        guard parts.count >= 2,
              let counts = parts.last,
              let slash = counts.firstIndex(of: "/"),
              let completed = Int(counts[..<slash]),
              let total = Int(counts[counts.index(after: slash)...]),
              total > 0
        else { return nil }
        let phase = parts.dropLast().joined(separator: " ")
        // Unknown phases (future h3 versions) are skipped — the caller keeps
        // the last known estimate instead of jumping backwards.
        guard let entry = phaseStart.first(where: { $0.phase == phase }) else { return nil }
        let percent = Int(
            (entry.start + (entry.end - entry.start) * Double(completed) / Double(total))
                .rounded())
        return H3VideoProgress(phase: phase, completed: completed, total: total, percent: percent)
    }
}

/// Reference wrapper so a @Sendable stream callback can mutate the parser
/// (SafeProcess's drain queue is serial, so no locking is needed).
private final class H3ProgressParserBox: @unchecked Sendable {
    var parser = H3ProgressParser()
}

/// Everything the service needs to validate and enqueue a job. The API layer
/// maps the OpenRouter wire shape onto this.
public struct H3VideoJobRequest: Sendable {
    public var model: String
    public var prompt: String
    public var durationSeconds: Int?
    public var size: String?
    public var resolution: String?
    public var aspectRatio: String?
    public var seed: Int?
    public var steps: Int?
    public var layers: Int?
    public var reuse: Int?
    public var firstFrameURL: String?
    public var lastFrameURL: String?
    public var referenceImageURLs: [String]
    public var callbackURL: String?

    public init(
        model: String,
        prompt: String,
        durationSeconds: Int? = nil,
        size: String? = nil,
        resolution: String? = nil,
        aspectRatio: String? = nil,
        seed: Int? = nil,
        steps: Int? = nil,
        layers: Int? = nil,
        reuse: Int? = nil,
        firstFrameURL: String? = nil,
        lastFrameURL: String? = nil,
        referenceImageURLs: [String] = [],
        callbackURL: String? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.durationSeconds = durationSeconds
        self.size = size
        self.resolution = resolution
        self.aspectRatio = aspectRatio
        self.seed = seed
        self.steps = steps
        self.layers = layers
        self.reuse = reuse
        self.firstFrameURL = firstFrameURL
        self.lastFrameURL = lastFrameURL
        self.referenceImageURLs = referenceImageURLs
        self.callbackURL = callbackURL
    }
}

/// OpenRouter-compatible async video generation backed by the local h3.c CLI.
///
/// h3 saturates the GPU for minutes per clip, so execution is strictly serial:
/// submissions land in a bounded FIFO queue (full → 429) and a single runner
/// task dequeues one job at a time, spawning the CLI via SafeProcess with a
/// wall-time deadline. Job state is file-based (`~/.nova/videos/<id>/`) so it
/// survives restarts; on init, leftover pending/in_progress jobs are failed
/// with "interrupted by server restart".
public final class H3VideoService: @unchecked Sendable {

    public var metricsStore: MetricsStore?

    public let config: H3VideoConfig

    private let lock = NovaMLXLock()
    private var jobs: [String: H3VideoJob] = [:]
    private var runnerTask: Task<Void, Never>?
    /// Live progress for running jobs (in-memory only).
    private var progressById: [String: H3VideoProgress] = [:]
    /// Cancellation handles for running h3 processes.
    private var cancelTokens: [String: SafeProcess.CancelToken] = [:]

    /// Test hook: when false, submit only enqueues — the serial runner never
    /// starts, so queueing behaviour is deterministic without a real GPU run.
    var runnerEnabled = true

    public static let primaryModelId = "minimax/h3"
    public static let modelAliases: Set<String> = ["minimax/h3", "minimax/hailuo-3"]
    /// Sizes the h3 build is known to produce well. 864x480 is the model's
    /// native training aspect.
    public static let supportedSizes: [(width: Int, height: Int)] = [
        (864, 480), (480, 864), (512, 512), (640, 640),
    ]
    public static let defaultSize = (width: 864, height: 480)
    public static let maxDurationSeconds = 10
    public static let framesPerSecond = 24

    public init(config: H3VideoConfig = H3VideoConfig.load()) {
        self.config = config
        try? FileManager.default.createDirectory(
            atPath: config.jobsDir, withIntermediateDirectories: true)
        recoverInterruptedJobs()
    }

    public var isReady: Bool { config.isReady }

    // MARK: - Validation / mapping (pure, testable)

    public static func validateModel(_ model: String) throws {
        guard modelAliases.contains(model) else {
            throw NovaMLXError.apiError(
                "Unsupported video model '\(model)'. Supported: \(modelAliases.sorted().joined(separator: ", "))")
        }
    }

    /// Resolve the requested output size. An explicit `size` ("WxH") must be
    /// one of `supportedSizes`; otherwise `resolution` + `aspect_ratio` map to
    /// the nearest supported size (landscape → 864x480, portrait → 480x864,
    /// square → 512x512, or 640x640 for 720p+).
    public static func resolveSize(
        size: String?,
        resolution: String?,
        aspectRatio: String?
    ) throws -> (width: Int, height: Int) {
        if let size {
            let parts = size.lowercased().split(separator: "x", omittingEmptySubsequences: false)
            guard parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]) else {
                throw NovaMLXError.apiError("Invalid size '\(size)' (expected \"<width>x<height>\")")
            }
            guard let match = supportedSizes.first(where: { $0.width == w && $0.height == h }) else {
                let legal = supportedSizes.map { "\($0.width)x\($0.height)" }.joined(separator: ", ")
                throw NovaMLXError.apiError("Unsupported size '\(size)'. Supported sizes: \(legal)")
            }
            return match
        }

        var aspect: Double = 16.0 / 9.0
        if let aspectRatio {
            let parts = aspectRatio.split(separator: ":", omittingEmptySubsequences: false)
            guard parts.count == 2, let w = Double(parts[0]), let h = Double(parts[1]), w > 0, h > 0 else {
                throw NovaMLXError.apiError("Invalid aspect_ratio '\(aspectRatio)' (expected \"<w>:<h>\")")
            }
            aspect = w / h
        }

        let hiRes: Bool
        if let resolution {
            let r = resolution.lowercased()
            if r.contains("1080") || r.contains("720") || r == "high" {
                hiRes = true
            } else if r.contains("480") || r.contains("360") || r == "low" {
                hiRes = false
            } else {
                throw NovaMLXError.apiError(
                    "Unsupported resolution '\(resolution)'. Supported: 480p, 720p, 1080p")
            }
        } else {
            hiRes = false
        }

        if aspect > 1.05 { return (864, 480) }
        if aspect < 0.95 { return (480, 864) }
        return hiRes ? (640, 640) : (512, 512)
    }

    /// h3 CLI argument vector for a job. Static and pure so tests can pin it
    /// without a GPU.
    public static func buildArguments(
        config: H3VideoConfig,
        job: H3VideoJob,
        outputPath: String,
        firstFramePath: String? = nil,
        lastFramePath: String? = nil,
        referenceImagePaths: [String] = []
    ) -> [String] {
        var args = [
            "-d", config.weightsDir,
            "--width", String(job.width),
            "--height", String(job.height),
            "--steps", String(job.steps),
            "--layers", String(job.layers),
            "--reuse", String(job.reuse),
            "-p", job.prompt,
            "-o", outputPath,
        ]
        if let seconds = job.durationSeconds {
            args += ["--seconds", String(seconds)]
        }
        if let seed = job.seed {
            args += ["--seed", String(seed)]
        }
        if let firstFramePath {
            args += ["--first-frame", firstFramePath]
        }
        if let lastFramePath {
            args += ["--last-frame", lastFramePath]
        }
        for ref in referenceImagePaths {
            args += ["--ref-image", ref]
        }
        return args
    }

    public static func isValidJobId(_ id: String) -> Bool {
        id.range(of: "^[a-zA-Z0-9-]+$", options: .regularExpression) != nil
    }

    // MARK: - Submit / query / cancel

    /// Validate, enqueue, and persist a job. Returns immediately; generation
    /// happens on the serial runner.
    @discardableResult
    public func submit(_ req: H3VideoJobRequest) throws -> H3VideoJob {
        try Self.validateModel(req.model)
        guard !req.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw NovaMLXError.apiError("'prompt' is required and must be non-empty")
        }
        if let duration = req.durationSeconds {
            guard (1...Self.maxDurationSeconds).contains(duration) else {
                throw NovaMLXError.apiError(
                    "'duration' must be between 1 and \(Self.maxDurationSeconds) seconds (got \(duration))")
            }
        }
        let (width, height) = try Self.resolveSize(
            size: req.size, resolution: req.resolution, aspectRatio: req.aspectRatio)

        guard isReady else {
            throw NovaMLXError.configurationError(
                "h3 video pipeline is not configured: need executable at '\(config.binaryPath)' and weights at '\(config.weightsDir)'")
        }

        return try lock.withLock {
            cleanupExpiredJobsLocked(now: Date())

            let active = jobs.values.filter { $0.status == .pending || $0.status == .inProgress }.count
            guard active < config.maxQueuedJobs else {
                throw NovaMLXError.tooManyRequests(
                    "video generation queue is full (\(active)/\(config.maxQueuedJobs) jobs pending or running); poll existing jobs and retry")
            }

            let job = H3VideoJob(
                id: "job-" + UUID().uuidString.lowercased(),
                status: .pending,
                model: Self.primaryModelId,
                prompt: req.prompt,
                width: width,
                height: height,
                durationSeconds: req.durationSeconds,
                seed: req.seed,
                steps: req.steps ?? config.steps,
                layers: req.layers ?? config.layers,
                reuse: req.reuse ?? config.reuse,
                firstFrameURL: req.firstFrameURL,
                lastFrameURL: req.lastFrameURL,
                referenceImageURLs: req.referenceImageURLs,
                callbackURL: req.callbackURL,
                error: nil,
                createdAt: Date(),
                startedAt: nil,
                finishedAt: nil,
                expiredAt: nil
            )
            try FileManager.default.createDirectory(
                atPath: jobDirURL(for: job.id).path, withIntermediateDirectories: true)
            try persistLocked(job)
            jobs[job.id] = job
            NovaMLXLog.info("H3Video: submitted \(job.id) (\(width)x\(height), \(job.durationSeconds.map { "\($0)s" } ?? "default duration"))")
            ensureRunnerLocked()
            return job
        }
    }

    public func job(id: String) -> H3VideoJob? {
        guard Self.isValidJobId(id) else { return nil }
        return lock.withLock { jobs[id] }
    }

    /// Live progress for a running job (nil once finished or never started).
    public func progress(id: String) -> H3VideoProgress? {
        guard Self.isValidJobId(id) else { return nil }
        return lock.withLock { progressById[id] }
    }

    /// Absolute file URL of the finished clip. Only completed jobs have
    /// content, and only index 0 exists (h3 renders one clip per job).
    public func contentURL(id: String, index: Int = 0) -> URL? {
        guard Self.isValidJobId(id), index == 0 else { return nil }
        guard let job = job(id: id), job.status == .completed else { return nil }
        let url = outputURL(for: id)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return url
    }

    /// Cancel a queued job immediately, or ask the runner to kill a running
    /// one (SIGTERM, SIGKILL after 2 s). A running job flips to `cancelled`
    /// once the runner observes the process exit — poll to confirm.
    @discardableResult
    public func cancel(id: String) throws -> H3VideoJob {
        guard Self.isValidJobId(id) else {
            throw NovaMLXError.modelNotFound("video job '\(id)'")
        }
        return try lock.withLock {
            guard var job = jobs[id] else {
                throw NovaMLXError.modelNotFound("video job '\(id)'")
            }
            switch job.status {
            case .pending:
                job.status = .cancelled
                job.finishedAt = Date()
                try persistLocked(job)
                jobs[id] = job
                NovaMLXLog.info("H3Video: cancelled queued job \(id)")
                fireWebhookLocked(for: job)
                return job
            case .inProgress:
                guard let token = cancelTokens[id] else {
                    throw NovaMLXError.apiError(
                        "Job \(id) is finishing up and can no longer be cancelled")
                }
                token.terminate()
                NovaMLXLog.info("H3Video: terminate requested for running job \(id)")
                return job
            case .completed, .failed, .cancelled, .expired:
                return job
            }
        }
    }

    /// Live snapshot for progress reporting / metrics.
    public func activeJob() -> H3VideoJob? {
        lock.withLock { jobs.values.first { $0.status == .inProgress } }
    }

    // MARK: - Paths

    private func jobDirURL(for id: String) -> URL {
        URL(fileURLWithPath: config.jobsDir).appendingPathComponent(id)
    }

    private func jobFileURL(for id: String) -> URL {
        jobDirURL(for: id).appendingPathComponent("job.json")
    }

    private func outputURL(for id: String) -> URL {
        jobDirURL(for: id).appendingPathComponent("out.mp4")
    }

    // MARK: - Serial runner

    private func ensureRunnerLocked() {
        guard runnerEnabled, runnerTask == nil else { return }
        runnerTask = Task { [weak self] in
            await self?.runQueue()
        }
    }

    private func runQueue() async {
        while true {
            // Dequeue and runner-shutdown share one lock acquisition: a submit
            // landing between them would otherwise see a stale runnerTask and
            // strand its job forever.
            let next: H3VideoJob? = lock.withLock {
                if let job = jobs.values
                    .filter({ $0.status == .pending })
                    .sorted(by: { $0.createdAt < $1.createdAt })
                    .first {
                    return job
                }
                runnerTask = nil
                return nil
            }
            guard let job = next else { return }
            await run(job)
        }
    }

    private func run(_ job: H3VideoJob) async {
        let modelId = Self.primaryModelId
        do {
            try transition(job.id) { job in
                job.status = .inProgress
                job.startedAt = Date()
            }
        } catch {
            return // job vanished (e.g. TTL cleanup) — nothing to run
        }

        metricsStore?.reportActivity(model: modelId, kind: .video, speed: 0, unit: "job")
        let heartbeat = Task { [metricsStore] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                if Task.isCancelled { break }
                metricsStore?.touchActivity(model: modelId)
            }
        }
        defer {
            heartbeat.cancel()
            metricsStore?.clearActivity(forModel: modelId)
        }

        do {
            let dir = jobDirURL(for: job.id)
            // Reference images: download http(s) or decode data: URLs into the
            // job directory, then point the CLI at the local files.
            var firstFramePath: String? = nil
            var lastFramePath: String? = nil
            var refPaths: [String] = []
            if let ref = job.firstFrameURL {
                firstFramePath = try await Self.resolveImageReference(ref, into: dir, name: "first_frame").path
            }
            if let ref = job.lastFrameURL {
                lastFramePath = try await Self.resolveImageReference(ref, into: dir, name: "last_frame").path
            }
            for (i, ref) in job.referenceImageURLs.enumerated() {
                refPaths.append(try await Self.resolveImageReference(ref, into: dir, name: "ref_\(i)").path)
            }

            let out = outputURL(for: job.id)
            let args = Self.buildArguments(
                config: config,
                job: job,
                outputPath: out.path,
                firstFramePath: firstFramePath,
                lastFramePath: lastFramePath,
                referenceImagePaths: refPaths
            )
            NovaMLXLog.info("H3Video: \(config.binaryPath) \(args.joined(separator: " "))")

            let binaryPath = config.binaryPath
            let deadline = config.processTimeoutSeconds
            // h3 resolves h3_shaders.metal relative to its CWD, so run it from
            // the binary's own directory (same as gen.sh's `cd`).
            let workingDir = URL(fileURLWithPath: binaryPath).deletingLastPathComponent()
            let jobId = job.id
            let token = SafeProcess.CancelToken()
            lock.withLock { cancelTokens[jobId] = token }
            defer {
                lock.withLock {
                    cancelTokens[jobId] = nil
                    progressById[jobId] = nil
                }
            }
            // The drain queue is serial, so the parser needs no locking; the
            // box exists because @Sendable closures can't mutate a captured var.
            let parserBox = H3ProgressParserBox()
            let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<SafeProcess.Result, Error>) in
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        let result = try SafeProcess.run(
                            binaryPath,
                            arguments: args,
                            currentDirectory: workingDir,
                            deadline: deadline,
                            allowTimeout: false,
                            stderrStream: { [weak self] chunk in
                                guard let self, let text = String(data: chunk, encoding: .utf8) else { return }
                                guard let parsed = parserBox.parser.feed(text) else { return }
                                self.lock.withLock { self.progressById[jobId] = parsed }
                            },
                            cancelToken: token
                        )
                        continuation.resume(returning: result)
                    } catch {
                        continuation.resume(throwing: error)
                    }
                }
            }
            if token.wasTerminated {
                try transition(job.id) { job in
                    job.status = .cancelled
                    job.error = "cancelled by user"
                    job.finishedAt = Date()
                }
                NovaMLXLog.info("H3Video: \(job.id) cancelled")
                return
            }
            guard result.succeeded else {
                let tail = result.stderrText.split(separator: "\n").suffix(5).joined(separator: " | ")
                throw NovaMLXError.inferenceFailed("h3 failed (\(result.status)): \(tail)")
            }
            guard FileManager.default.fileExists(atPath: out.path) else {
                throw NovaMLXError.inferenceFailed("h3 exited cleanly but produced no output at \(out.path)")
            }

            let elapsed = Date().timeIntervalSince(job.createdAt)
            try transition(job.id) { job in
                job.status = .completed
                job.finishedAt = Date()
            }
            NovaMLXLog.info("H3Video: \(job.id) completed in \(String(format: "%.1f", elapsed))s")
        } catch {
            let message = error.localizedDescription
            try? transition(job.id) { job in
                job.status = .failed
                job.error = message
                job.finishedAt = Date()
            }
            NovaMLXLog.error("H3Video: \(job.id) failed: \(message)")
        }
    }

    /// Apply a mutation to a job's state, persist, and fire the webhook when
    /// the job lands in a terminal state.
    private func transition(_ id: String, _ mutate: (inout H3VideoJob) -> Void) throws {
        let finished: H3VideoJob? = try lock.withLock {
            guard var job = jobs[id] else {
                throw NovaMLXError.modelNotFound("video job '\(id)'")
            }
            mutate(&job)
            try persistLocked(job)
            jobs[id] = job
            return job.status.isTerminal ? job : nil
        }
        if let finished {
            fireWebhook(for: finished)
        }
    }

    // MARK: - Persistence / recovery / TTL

    /// Must be called with `lock` held.
    private func persistLocked(_ job: H3VideoJob) throws {
        let data = try JSONEncoder().encode(job)
        try data.write(to: jobFileURL(for: job.id), options: .atomic)
    }

    /// Startup recovery: any job left pending/in_progress by a previous
    /// process is failed — the h3 child died with the server.
    private func recoverInterruptedJobs() {
        let fm = FileManager.default
        let base = URL(fileURLWithPath: config.jobsDir)
        guard let entries = try? fm.contentsOfDirectory(atPath: base.path) else { return }
        for entry in entries where Self.isValidJobId(entry) {
            let file = base.appendingPathComponent(entry).appendingPathComponent("job.json")
            guard let data = try? Data(contentsOf: file),
                  var job = try? JSONDecoder().decode(H3VideoJob.self, from: data)
            else { continue }
            if job.status == .pending || job.status == .inProgress {
                job.status = .failed
                job.error = "interrupted by server restart"
                job.finishedAt = Date()
                try? persistLocked(job)
            }
            lock.withLock { jobs[job.id] = job }
        }
    }

    /// Lazy expiry, called from submit: terminal jobs older than
    /// `jobTTLSeconds` flip to `expired` (job.json tombstone kept, mp4
    /// deleted); tombstones older than another TTL are removed entirely.
    func cleanupExpiredJobs(now: Date = Date()) {
        lock.withLock { cleanupExpiredJobsLocked(now: now) }
    }

    /// Must be called with `lock` held.
    func cleanupExpiredJobsLocked(now: Date) {
        let fm = FileManager.default
        for (id, job) in jobs {
            if job.status == .expired {
                if let expiredAt = job.expiredAt,
                   now.timeIntervalSince(expiredAt) >= config.jobTTLSeconds {
                    try? fm.removeItem(at: jobDirURL(for: id))
                    jobs[id] = nil
                }
                continue
            }
            guard job.status.isTerminal else { continue }
            let age = now.timeIntervalSince(job.finishedAt ?? job.createdAt)
            guard age >= config.jobTTLSeconds else { continue }
            var tombstone = job
            tombstone.status = .expired
            tombstone.expiredAt = now
            try? fm.removeItem(at: outputURL(for: id))
            try? persistLocked(tombstone)
            jobs[id] = tombstone
        }
    }

    // MARK: - Reference images

    /// Decode a `data:` URL into raw bytes + a file extension derived from the
    /// media type. Returns nil for malformed input.
    static func decodeDataURL(_ string: String) -> (data: Data, ext: String)? {
        guard string.hasPrefix("data:"),
              let comma = string.firstIndex(of: ",") else { return nil }
        let meta = string[string.index(string.startIndex, offsetBy: 5)..<comma]
        let payload = String(string[string.index(after: comma)...])
        let isBase64 = meta.contains(";base64")
        let data: Data?
        if isBase64 {
            data = Data(base64Encoded: payload)
        } else {
            data = payload.removingPercentEncoding.flatMap { Data($0.utf8) }
        }
        guard let data else { return nil }
        let subtype = meta.split(separator: "/").last.map(String.init)?
            .split(separator: ";").first.map(String.init) ?? "png"
        let ext = subtype == "jpeg" ? "jpg" : subtype
        return (data, ext)
    }

    /// Turn an image reference (http(s) URL or data: URL) into a local file
    /// inside the job directory.
    static func resolveImageReference(_ ref: String, into dir: URL, name: String) async throws -> URL {
        if ref.hasPrefix("data:") {
            guard let (data, ext) = decodeDataURL(ref) else {
                throw NovaMLXError.apiError("Malformed data: URL for image reference '\(name)'")
            }
            let url = dir.appendingPathComponent("\(name).\(ext)")
            try data.write(to: url)
            return url
        }
        if ref.hasPrefix("http://") || ref.hasPrefix("https://") {
            guard let remote = URL(string: ref) else {
                throw NovaMLXError.apiError("Malformed image URL: \(ref)")
            }
            let (data, response) = try await URLSession.shared.data(from: remote)
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw NovaMLXError.downloadFailed(ref, underlying: NovaMLXError.apiError("HTTP \(http.statusCode)"))
            }
            let ext = remote.pathExtension.isEmpty ? "png" : remote.pathExtension
            let url = dir.appendingPathComponent("\(name).\(ext)")
            try data.write(to: url)
            return url
        }
        throw NovaMLXError.apiError(
            "Unsupported image reference '\(ref.prefix(64))': expected an http(s) or data: URL")
    }

    // MARK: - Webhook

    private func fireWebhookLocked(for job: H3VideoJob) {
        // cancel() already holds the lock; hop out before posting.
        fireWebhook(for: job)
    }

    /// OpenRouter-style completion webhook: POST
    /// `{type: "video.generation.<status>", created_at, data: {...}}` to the
    /// job's callback_url, with an idempotency key and — when a signing
    /// secret is configured — an HMAC-SHA256 signature header.
    private func fireWebhook(for job: H3VideoJob) {
        guard let callback = job.callbackURL, let url = URL(string: callback) else { return }
        var data: [String: Any] = [
            "id": job.id,
            "model": job.model,
            "status": job.status.rawValue,
        ]
        if job.status == .completed {
            data["unsigned_urls"] = ["/v1/videos/\(job.id)/content?index=0"]
            data["usage"] = ["cost": 0, "is_byok": false]
        }
        if let error = job.error {
            data["error"] = error
        }
        let envelope: [String: Any] = [
            "type": "video.generation.\(job.status.rawValue)",
            "created_at": Int(Date().timeIntervalSince1970),
            "data": data,
        ]
        guard let body = try? JSONSerialization.data(withJSONObject: envelope) else { return }

        let secret = config.webhookSigningSecret
        Task.detached {
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("\(job.id)-\(job.status.rawValue)", forHTTPHeaderField: "X-OpenRouter-Idempotency-Key")
            if let secret {
                let ts = Int(Date().timeIntervalSince1970)
                request.setValue(
                    Self.webhookSignatureHeader(secret: secret, timestamp: ts, body: body),
                    forHTTPHeaderField: "X-OpenRouter-Signature")
            }
            request.httpBody = body
            do {
                let (_, response) = try await URLSession.shared.data(for: request)
                if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                    NovaMLXLog.warning("H3Video: webhook for \(job.id) got HTTP \(http.statusCode)")
                }
            } catch {
                NovaMLXLog.warning("H3Video: webhook for \(job.id) failed: \(error.localizedDescription)")
            }
        }
    }

    /// `t=<unix>,v1=<hex HMAC-SHA256 of "<unix>,<rawBody>">` — matches the
    /// OpenRouter webhook signing scheme.
    static func webhookSignatureHeader(secret: String, timestamp: Int, body: Data) -> String {
        let key = SymmetricKey(data: Data(secret.utf8))
        var payload = Data("\(timestamp),".utf8)
        payload.append(body)
        let mac = HMAC<SHA256>.authenticationCode(for: payload, using: key)
        let hex = mac.map { String(format: "%02x", $0) }.joined()
        return "t=\(timestamp),v1=\(hex)"
    }
}
