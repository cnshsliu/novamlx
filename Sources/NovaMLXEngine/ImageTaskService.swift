import Foundation
import NovaMLXCore

/// Live sampler progress for one image task. Same fields as a video job's
/// progress so both modalities poll as one object.
public struct ImageTaskProgress: Sendable, Equatable {
    public var phase: String
    public var completed: Int
    public var total: Int
    public var percent: Int

    public init(phase: String, completed: Int, total: Int, percent: Int) {
        self.phase = phase
        self.completed = completed
        self.total = total
        self.percent = percent
    }
}

/// One image generation accepted onto the shared task queue.
///
/// Status names are `H3VideoJobStatus` (OpenRouter's video job states).
/// The HTTP query for this id is `GET /v1/tasks/{id}`, the same resource
/// video jobs answer on.
public struct ImageTaskJob: Sendable {
    public var id: String
    public var status: H3VideoJobStatus
    public var model: String
    public var prompt: String
    public var negativePrompt: String
    public var n: Int
    public var width: Int
    public var height: Int
    public var seed: UInt64?
    public var steps: Int?
    public var responseFormat: String
    public var error: String?
    public var createdAt: Date
    public var startedAt: Date?
    public var finishedAt: Date?
    public var result: ImageGenerationResult?

    public init(
        id: String,
        status: H3VideoJobStatus,
        model: String,
        prompt: String,
        negativePrompt: String,
        n: Int,
        width: Int,
        height: Int,
        seed: UInt64?,
        steps: Int?,
        responseFormat: String,
        error: String? = nil,
        createdAt: Date,
        startedAt: Date? = nil,
        finishedAt: Date? = nil,
        result: ImageGenerationResult? = nil
    ) {
        self.id = id
        self.status = status
        self.model = model
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.n = n
        self.width = width
        self.height = height
        self.seed = seed
        self.steps = steps
        self.responseFormat = responseFormat
        self.error = error
        self.createdAt = createdAt
        self.startedAt = startedAt
        self.finishedAt = finishedAt
        self.result = result
    }
}

/// Serial image-generation queue. Submit returns immediately; sampling runs
/// off the HTTP connection. One sampler at a time, matching
/// `ImageGenerationService`'s single in-flight generation.
public final class ImageTaskService: @unchecked Sendable {
    /// Finished tasks stay queryable for a day, then flip to `expired` and
    /// drop the image bytes. Same order of magnitude as DashScope's task id.
    public static let jobTTL: TimeInterval = 24 * 60 * 60

    private let images: ImageGenerationService
    private let lock = NSLock()
    private var jobs: [String: ImageTaskJob] = [:]
    private var runner: Task<Void, Never>?
    private let autostart: Bool

    /// `autostart` is false only in tests that need a queued job to stay pending.
    public init(images: ImageGenerationService, autostart: Bool = true) {
        self.images = images
        self.autostart = autostart
    }

    public static func isValidId(_ id: String) -> Bool {
        id.range(of: "^img-[a-zA-Z0-9-]+$", options: .regularExpression) != nil
    }

    @discardableResult
    public func submit(
        model: String,
        prompt: String,
        negativePrompt: String,
        n: Int,
        width: Int,
        height: Int,
        seed: UInt64?,
        steps: Int?,
        responseFormat: String
    ) -> ImageTaskJob {
        let job = ImageTaskJob(
            id: "img-" + UUID().uuidString.lowercased(),
            status: .pending,
            model: model,
            prompt: prompt,
            negativePrompt: negativePrompt,
            n: n,
            width: width,
            height: height,
            seed: seed,
            steps: steps,
            responseFormat: responseFormat,
            createdAt: Date()
        )
        lock.withLock {
            expireLocked(now: Date())
            jobs[job.id] = job
            if autostart { ensureRunnerLocked() }
        }
        return job
    }

    public func job(id: String) -> ImageTaskJob? {
        guard Self.isValidId(id) else { return nil }
        return lock.withLock {
            guard var job = jobs[id] else { return nil }
            if job.status.isTerminal, let finished = job.finishedAt,
               Date().timeIntervalSince(finished) >= Self.jobTTL {
                job.status = .expired
                job.result = nil
                jobs[id] = job
            }
            return job
        }
    }

    /// Nil unless the job is sampling. `step`/`total` stay 0 until the
    /// pipeline reports its first step.
    public func progress(id: String) -> ImageTaskProgress? {
        guard Self.isValidId(id) else { return nil }
        let running = lock.withLock { jobs[id]?.status == .inProgress }
        guard running else { return nil }
        let step = images.inFlight()
        let completed = step?.step ?? 0
        let total = step?.total ?? 0
        let percent = total > 0 ? min(99, completed * 100 / total) : 0
        return ImageTaskProgress(phase: "sampling", completed: completed, total: total, percent: percent)
    }

    @discardableResult
    public func cancel(id: String) throws -> ImageTaskJob {
        guard Self.isValidId(id) else {
            throw NovaMLXError.modelNotFound("image task '\(id)'")
        }
        let outcome: (ImageTaskJob, Bool) = try lock.withLock {
            guard var job = jobs[id] else {
                throw NovaMLXError.modelNotFound("image task '\(id)'")
            }
            switch job.status {
            case .pending:
                job.status = .cancelled
                job.finishedAt = Date()
                jobs[id] = job
                return (job, false)
            case .inProgress:
                return (job, true)
            case .completed, .failed, .cancelled, .expired:
                return (job, false)
            }
        }
        if outcome.1 {
            images.requestCancel()
        }
        return outcome.0
    }

    private func ensureRunnerLocked() {
        guard runner == nil else { return }
        runner = Task { [weak self] in
            await self?.runQueue()
        }
    }

    private func runQueue() async {
        while true {
            let next: ImageTaskJob? = lock.withLock {
                if let job = jobs.values
                    .filter({ $0.status == .pending })
                    .sorted(by: { $0.createdAt < $1.createdAt })
                    .first {
                    return job
                }
                runner = nil
                return nil
            }
            guard let job = next else { return }
            await run(job)
        }
    }

    private func run(_ job: ImageTaskJob) async {
        let started = lock.withLock { () -> Bool in
            guard var current = jobs[job.id], current.status == .pending else { return false }
            current.status = .inProgress
            current.startedAt = Date()
            jobs[job.id] = current
            return true
        }
        guard started else { return }
        do {
            let result = try await images.generate(
                modelId: job.model,
                prompt: job.prompt,
                negativePrompt: job.negativePrompt,
                n: job.n,
                width: job.width,
                height: job.height,
                seed: job.seed,
                steps: job.steps
            )
            lock.withLock {
                guard var current = jobs[job.id], current.status == .inProgress else { return }
                current.status = .completed
                current.finishedAt = Date()
                current.result = result
                jobs[job.id] = current
            }
        } catch {
            let message = String(describing: error)
            let cancelled = message.localizedCaseInsensitiveContains("cancelled")
            lock.withLock {
                guard var current = jobs[job.id] else { return }
                if current.status == .cancelled { return }
                current.status = cancelled ? .cancelled : .failed
                current.error = message
                current.finishedAt = Date()
                jobs[job.id] = current
            }
        }
    }

    /// Must be called with `lock` held.
    private func expireLocked(now: Date) {
        for (id, var job) in jobs {
            guard job.status.isTerminal, let finished = job.finishedAt else { continue }
            guard now.timeIntervalSince(finished) >= Self.jobTTL else { continue }
            job.status = .expired
            job.result = nil
            jobs[id] = job
        }
    }
}
