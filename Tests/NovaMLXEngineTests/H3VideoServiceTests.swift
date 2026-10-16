import Foundation
import Testing
@testable import NovaMLXEngine
import NovaMLXCore
import NovaMLXUtils

/// Pure-logic tests for the h3 video job service: argument construction,
/// config decoding, size mapping, and the file-based job state machine.
/// The real h3 CLI is never invoked (minutes per run, external drive); one
/// test drives the full queue against a fake shell-script binary instead.
struct H3VideoServiceTests {

    // MARK: - Helpers

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("h3video_tests_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func makeConfig(jobsDir: URL, maxQueuedJobs: Int = 4) -> H3VideoConfig {
        H3VideoConfig(
            binaryPath: "/bin/echo",
            weightsDir: jobsDir.path, // any existing dir passes the readiness check
            jobsDir: jobsDir.appendingPathComponent("videos").path,
            steps: 20,
            layers: 45,
            reuse: 2,
            processTimeoutSeconds: 30,
            jobTTLSeconds: 86400,
            maxQueuedJobs: maxQueuedJobs,
            webhookSigningSecret: nil
        )
    }

    private func makeRequest(model: String = "minimax/h3", prompt: String = "a fox") -> H3VideoJobRequest {
        H3VideoJobRequest(model: model, prompt: prompt)
    }

    private func writeJob(_ job: H3VideoJob, jobsDir: URL, withOutput: Bool = false) throws {
        let dir = jobsDir.appendingPathComponent(job.id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(job).write(to: dir.appendingPathComponent("job.json"))
        if withOutput {
            try Data("fake mp4".utf8).write(to: dir.appendingPathComponent("out.mp4"))
        }
    }

    private func makeJob(id: String, status: H3VideoJobStatus, createdAt: Date = Date(), finishedAt: Date? = nil) -> H3VideoJob {
        H3VideoJob(
            id: id,
            status: status,
            model: "minimax/h3",
            prompt: "a fox",
            width: 864,
            height: 480,
            durationSeconds: nil,
            seed: nil,
            steps: 20,
            layers: 45,
            reuse: 2,
            firstFrameURL: nil,
            lastFrameURL: nil,
            referenceImageURLs: [],
            callbackURL: nil,
            error: nil,
            createdAt: createdAt,
            startedAt: nil,
            finishedAt: finishedAt,
            expiredAt: nil
        )
    }

    // MARK: - Argument construction

    @Test func argumentsCarryWeightsSizeQualityAndPrompt() {
        let dir = URL(fileURLWithPath: "/tmp/h3jobs")
        let config = H3VideoConfig(jobsDir: dir.path)
        let job = makeJob(id: "job-a", status: .pending)
        let args = H3VideoService.buildArguments(
            config: config, job: job, outputPath: "/tmp/h3jobs/job-a/out.mp4")

        #expect(args[0] == "-d")
        #expect(args[1] == H3VideoConfig.defaultWeightsDir)
        for (flag, value) in [("--width", "864"), ("--height", "480"), ("--steps", "20"), ("--layers", "45"), ("--reuse", "2")] {
            guard let i = args.firstIndex(of: flag) else {
                Issue.record("missing flag \(flag)")
                continue
            }
            #expect(args[i + 1] == value)
        }
        #expect(args[args.firstIndex(of: "-p")! + 1] == "a fox")
        #expect(args[args.firstIndex(of: "-o")! + 1] == "/tmp/h3jobs/job-a/out.mp4")
        // No optional flags when the job doesn't set them.
        #expect(!args.contains("--seconds"))
        #expect(!args.contains("--seed"))
        #expect(!args.contains("--first-frame"))
        #expect(!args.contains("--last-frame"))
        #expect(!args.contains("--ref-image"))
    }

    @Test func argumentsIncludeOptionalFlags() {
        let config = H3VideoConfig(jobsDir: "/tmp/h3jobs")
        var job = makeJob(id: "job-b", status: .pending)
        job.durationSeconds = 5
        job.seed = 42
        let args = H3VideoService.buildArguments(
            config: config,
            job: job,
            outputPath: "/tmp/h3jobs/job-b/out.mp4",
            firstFramePath: "/tmp/h3jobs/job-b/first_frame.png",
            lastFramePath: "/tmp/h3jobs/job-b/last_frame.png",
            referenceImagePaths: ["/tmp/h3jobs/job-b/ref_0.png", "/tmp/h3jobs/job-b/ref_1.png"]
        )
        #expect(args[args.firstIndex(of: "--seconds")! + 1] == "5")
        #expect(args[args.firstIndex(of: "--seed")! + 1] == "42")
        #expect(args[args.firstIndex(of: "--first-frame")! + 1] == "/tmp/h3jobs/job-b/first_frame.png")
        #expect(args[args.firstIndex(of: "--last-frame")! + 1] == "/tmp/h3jobs/job-b/last_frame.png")
        #expect(args.filter { $0 == "--ref-image" }.count == 2)
    }

    // MARK: - Config

    @Test func configDefaults() {
        let cfg = H3VideoConfig()
        #expect(cfg.binaryPath == "/Users/lucas/dev/h3.c/h3")
        #expect(cfg.weightsDir == "/Volumes/Samsung768/h3/MiniMax-H3")
        #expect(cfg.jobsDir.hasSuffix(".nova/videos"))
        #expect(cfg.steps == 20)
        #expect(cfg.layers == 45)
        #expect(cfg.reuse == 2)
        #expect(cfg.processTimeoutSeconds == 3600)
        #expect(cfg.jobTTLSeconds == 86400)
        #expect(cfg.maxQueuedJobs == 4)
        #expect(cfg.webhookSigningSecret == nil)
    }

    @Test func configDecodesPartialJSON() throws {
        let json = #"{"steps": 4, "layers": 40, "webhookSigningSecret": "s3cret"}"#
        let cfg = try JSONDecoder().decode(H3VideoConfig.self, from: Data(json.utf8))
        #expect(cfg.steps == 4)
        #expect(cfg.layers == 40)
        #expect(cfg.webhookSigningSecret == "s3cret")
        // Untouched fields fall back to defaults, not zeros.
        #expect(cfg.reuse == 2)
        #expect(cfg.binaryPath == H3VideoConfig.defaultBinaryPath)
        #expect(cfg.maxQueuedJobs == 4)
    }

    @Test func configEnvironmentOverridesFileAndDefaults() {
        let cfg = H3VideoConfig.load(
            fileURL: URL(fileURLWithPath: "/nonexistent/h3_video.json"),
            environment: [
                "NOVA_H3_BINARY": "/opt/h3/h3",
                "NOVA_H3_STEPS": "8",
                "NOVA_H3_MAX_QUEUED": "2",
            ])
        #expect(cfg.binaryPath == "/opt/h3/h3")
        #expect(cfg.steps == 8)
        #expect(cfg.maxQueuedJobs == 2)
        // Everything else stays at defaults.
        #expect(cfg.layers == 45)
        #expect(cfg.jobTTLSeconds == 86400)
    }

    // MARK: - Size mapping / validation

    @Test func explicitSizeMustBeSupported() throws {
        #expect(try H3VideoService.resolveSize(size: "480x864", resolution: nil, aspectRatio: nil) == (480, 864))
        #expect(throws: NovaMLXError.self) {
            try H3VideoService.resolveSize(size: "1920x1080", resolution: nil, aspectRatio: nil)
        }
        #expect(throws: NovaMLXError.self) {
            try H3VideoService.resolveSize(size: "not-a-size", resolution: nil, aspectRatio: nil)
        }
    }

    @Test func invalidAspectRatioAndResolutionThrow() {
        #expect(throws: NovaMLXError.self) {
            try H3VideoService.resolveSize(size: nil, resolution: nil, aspectRatio: "widescreen")
        }
        #expect(throws: NovaMLXError.self) {
            try H3VideoService.resolveSize(size: nil, resolution: "8k", aspectRatio: nil)
        }
    }

    @Test func jobIdsRejectPathTraversal() {
        #expect(H3VideoService.isValidJobId("job-abc-123"))
        #expect(!H3VideoService.isValidJobId("../etc/passwd"))
        #expect(!H3VideoService.isValidJobId("job/../../x"))
        #expect(!H3VideoService.isValidJobId(""))
    }

    // MARK: - data: URLs and webhook signing

    @Test func decodeDataURL() {
        let png = H3VideoService.decodeDataURL("data:image/png;base64,aGVsbG8=")
        #expect(png?.ext == "png")
        #expect(png?.data == Data("hello".utf8))

        let jpeg = H3VideoService.decodeDataURL("data:image/jpeg;base64,aGVsbG8=")
        #expect(jpeg?.ext == "jpg")

        #expect(H3VideoService.decodeDataURL("https://example.com/x.png") == nil)
        #expect(H3VideoService.decodeDataURL("data:image/png;base64,!!!not-base64!!!") == nil)
    }

    @Test func webhookSignatureFormat() {
        let body = Data(#"{"type":"video.generation.completed"}"#.utf8)
        let header = H3VideoService.webhookSignatureHeader(secret: "secret", timestamp: 1_700_000_000, body: body)
        #expect(header.hasPrefix("t=1700000000,v1="))
        #expect(header.count == "t=1700000000,v1=".count + 64) // hex SHA256
        // Deterministic, and key-dependent.
        #expect(header == H3VideoService.webhookSignatureHeader(secret: "secret", timestamp: 1_700_000_000, body: body))
        #expect(header != H3VideoService.webhookSignatureHeader(secret: "other", timestamp: 1_700_000_000, body: body))
    }

    // MARK: - Submit validation and queueing

    @Test func submitRejectsBadModelEmptyPromptAndLongDuration() throws {
        let dir = try makeTempDir()
        let service = H3VideoService(config: makeConfig(jobsDir: dir))
        service.runnerEnabled = false

        #expect(throws: NovaMLXError.self) {
            try service.submit(makeRequest(model: "openai/sora-2"))
        }
        #expect(throws: NovaMLXError.self) {
            try service.submit(makeRequest(prompt: "   "))
        }
        var longReq = makeRequest()
        longReq.durationSeconds = 11
        #expect(throws: NovaMLXError.self) {
            try service.submit(longReq)
        }
        // Alias is accepted.
        let job = try service.submit(makeRequest(model: "minimax/hailuo-3"))
        #expect(job.status == .pending)
        #expect(job.model == "minimax/h3")
    }

    @Test func submitPersistsJobJSON() throws {
        let dir = try makeTempDir()
        let config = makeConfig(jobsDir: dir)
        let service = H3VideoService(config: config)
        service.runnerEnabled = false

        let job = try service.submit(makeRequest())
        let file = URL(fileURLWithPath: config.jobsDir)
            .appendingPathComponent(job.id).appendingPathComponent("job.json")
        let persisted = try JSONDecoder().decode(H3VideoJob.self, from: Data(contentsOf: file))
        #expect(persisted.id == job.id)
        #expect(persisted.status == .pending)
        #expect(service.job(id: job.id)?.status == .pending)
    }

    @Test func queueFullReturnsTooManyRequests() throws {
        let dir = try makeTempDir()
        let service = H3VideoService(config: makeConfig(jobsDir: dir, maxQueuedJobs: 1))
        service.runnerEnabled = false

        _ = try service.submit(makeRequest())
        #expect(throws: NovaMLXError.self) {
            try service.submit(makeRequest())
        }
        do {
            _ = try service.submit(makeRequest())
            Issue.record("expected queue-full throw")
        } catch NovaMLXError.tooManyRequests {
            // expected
        }
    }

    @Test func cancelQueuedJob() throws {
        let dir = try makeTempDir()
        let service = H3VideoService(config: makeConfig(jobsDir: dir))
        service.runnerEnabled = false

        let job = try service.submit(makeRequest())
        let cancelled = try service.cancel(id: job.id)
        #expect(cancelled.status == .cancelled)
        #expect(service.job(id: job.id)?.status == .cancelled)
        // Terminal jobs are returned unchanged; unknown ids 404.
        #expect(try service.cancel(id: job.id).status == .cancelled)
        #expect(throws: NovaMLXError.self) {
            try service.cancel(id: "job-does-not-exist")
        }
    }

    // MARK: - Restart recovery and TTL

    @Test func restartMarksLeftoverJobsFailed() throws {
        let dir = try makeTempDir()
        let config = makeConfig(jobsDir: dir)
        let jobsDir = URL(fileURLWithPath: config.jobsDir)

        try writeJob(makeJob(id: "job-pending", status: .pending), jobsDir: jobsDir)
        try writeJob(makeJob(id: "job-running", status: .inProgress), jobsDir: jobsDir)
        try writeJob(makeJob(id: "job-done", status: .completed, finishedAt: Date()), jobsDir: jobsDir, withOutput: true)

        let service = H3VideoService(config: config)
        service.runnerEnabled = false

        let pending = service.job(id: "job-pending")
        #expect(pending?.status == .failed)
        #expect(pending?.error == "interrupted by server restart")
        let running = service.job(id: "job-running")
        #expect(running?.status == .failed)
        #expect(running?.error == "interrupted by server restart")
        // Completed jobs survive untouched.
        #expect(service.job(id: "job-done")?.status == .completed)
        #expect(service.contentURL(id: "job-done") != nil)
    }

    @Test func ttlCleanupExpiresOldJobsThenRemovesTombstones() throws {
        let dir = try makeTempDir()
        var config = makeConfig(jobsDir: dir)
        config.jobTTLSeconds = 3600
        let jobsDir = URL(fileURLWithPath: config.jobsDir)

        let old = Date().addingTimeInterval(-7200)
        try writeJob(
            makeJob(id: "job-old", status: .completed, createdAt: old, finishedAt: old),
            jobsDir: jobsDir, withOutput: true)
        try writeJob(
            makeJob(id: "job-fresh", status: .completed, finishedAt: Date()),
            jobsDir: jobsDir, withOutput: true)

        let service = H3VideoService(config: config)
        service.runnerEnabled = false
        service.cleanupExpiredJobs()

        // Old job: tombstone kept, mp4 deleted, content gone.
        #expect(service.job(id: "job-old")?.status == .expired)
        #expect(service.contentURL(id: "job-old") == nil)
        #expect(!FileManager.default.fileExists(atPath: jobsDir.appendingPathComponent("job-old/out.mp4").path))
        // Fresh job untouched.
        #expect(service.job(id: "job-fresh")?.status == .completed)
        #expect(service.contentURL(id: "job-fresh") != nil)

        // A tombstone older than another TTL is removed entirely.
        var tombstone = service.job(id: "job-old")!
        tombstone.expiredAt = Date().addingTimeInterval(-7200)
        try JSONEncoder().encode(tombstone).write(
            to: jobsDir.appendingPathComponent("job-old/job.json"))
        let reloaded = H3VideoService(config: config)
        reloaded.runnerEnabled = false
        reloaded.cleanupExpiredJobs()
        #expect(reloaded.job(id: "job-old") == nil)
        #expect(!FileManager.default.fileExists(atPath: jobsDir.appendingPathComponent("job-old").path))
    }

    // MARK: - End-to-end against a fake h3 binary

    @Test func runnerCompletesJobWithFakeBinary() async throws {
        let dir = try makeTempDir()
        // A stand-in for h3: parses -o and writes a fake mp4.
        let script = dir.appendingPathComponent("fake-h3.sh")
        try """
        #!/bin/sh
        out=""
        prev=""
        for a in "$@"; do
          if [ "$prev" = "-o" ]; then out="$a"; fi
          prev="$a"
        done
        echo "fake mp4 data" > "$out"
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        var config = makeConfig(jobsDir: dir)
        config.binaryPath = script.path
        let service = H3VideoService(config: config)

        var req = makeRequest()
        req.durationSeconds = 2
        req.seed = 1
        req.referenceImageURLs = ["data:image/png;base64,aGVsbG8="]
        let job = try service.submit(req)

        // Poll until the serial runner finishes the job.
        var final: H3VideoJob?
        for _ in 0..<100 {
            if let current = service.job(id: job.id), current.status.isTerminal {
                final = current
                break
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        #expect(final?.status == .completed)
        #expect(final?.finishedAt != nil)
        let content = service.contentURL(id: job.id)
        #expect(content != nil)
        #expect(try String(contentsOf: content!, encoding: .utf8).contains("fake mp4 data"))
        // The data: reference was decoded into the job directory.
        #expect(FileManager.default.fileExists(
            atPath: URL(fileURLWithPath: config.jobsDir)
                .appendingPathComponent(job.id).appendingPathComponent("ref_0.png").path))
    }

    @Test func runnerMarksJobFailedWhenBinaryFails() async throws {
        let dir = try makeTempDir()
        let script = dir.appendingPathComponent("fail-h3.sh")
        try """
        #!/bin/sh
        echo "boom: out of memory" >&2
        exit 3
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        var config = makeConfig(jobsDir: dir)
        config.binaryPath = script.path
        let service = H3VideoService(config: config)

        let job = try service.submit(makeRequest())
        var final: H3VideoJob?
        for _ in 0..<100 {
            if let current = service.job(id: job.id), current.status.isTerminal {
                final = current
                break
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        #expect(final?.status == .failed)
        #expect(final?.error?.contains("boom: out of memory") == true)
        #expect(service.contentURL(id: job.id) == nil)
    }

    // MARK: - Progress parsing

    @Test func progressLineParsesPhaseCountsAndPercent() {
        let denoise = H3ProgressParser.parseLine("denoise                    10/20  ")
        #expect(denoise?.phase == "denoise")
        #expect(denoise?.completed == 10)
        #expect(denoise?.total == 20)
        // denoise spans 12–87%; halfway lands near the middle.
        #expect(denoise != nil && denoise!.percent > 40 && denoise!.percent < 60)

        let start = H3ProgressParser.parseLine("tokenizer                     0/1   ")
        #expect(start?.percent == 0)

        let ffmpeg = H3ProgressParser.parseLine("FFmpeg                       22/22")
        #expect(ffmpeg?.percent == 100)

        // Unknown phases and non-progress lines are ignored.
        #expect(H3ProgressParser.parseLine("h3: video VAE cache hit") == nil)
        #expect(H3ProgressParser.parseLine("future phase                3/9") == nil)
    }

    @Test func progressParserHandlesCarriageReturnStream() {
        var parser = H3ProgressParser()
        #expect(parser.feed("tokenizer                     0/1   \rtokenizer") == H3VideoProgress(phase: "tokenizer", completed: 0, total: 1, percent: 0))
        // The incomplete tail is buffered and completed by the next chunk.
        let second = parser.feed("                     1/1   \r\ndenoise                      3/20  \r")
        #expect(second?.phase == "denoise")
        #expect(second?.completed == 3)
    }

    // MARK: - Cancelling a running job

    @Test func cancelRunningJobTerminatesProcess() async throws {
        let dir = try makeTempDir()
        // A stand-in that emits one progress line then sleeps past the test.
        let script = dir.appendingPathComponent("slow-h3.sh")
        try """
        #!/bin/sh
        printf 'denoise                      1/20  \\r' >&2
        sleep 60
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        var config = makeConfig(jobsDir: dir)
        config.binaryPath = script.path
        let service = H3VideoService(config: config)

        let job = try service.submit(makeRequest())

        // Wait for the runner to pick the job up.
        var sawRunning = false
        for _ in 0..<100 {
            if service.job(id: job.id)?.status == .inProgress { sawRunning = true; break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        #expect(sawRunning)

        _ = try service.cancel(id: job.id)

        var final: H3VideoJob?
        for _ in 0..<100 {
            if let current = service.job(id: job.id), current.status.isTerminal {
                final = current
                break
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        #expect(final?.status == .cancelled)
        #expect(final?.error == "cancelled by user")
    }

    @Test func progressIsExposedWhileRunning() async throws {
        let dir = try makeTempDir()
        let script = dir.appendingPathComponent("progress-h3.sh")
        try """
        #!/bin/sh
        printf 'tokenizer                     1/1   \\r' >&2
        printf 'denoise                      5/20  \\r' >&2
        sleep 60
        """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        var config = makeConfig(jobsDir: dir)
        config.binaryPath = script.path
        let service = H3VideoService(config: config)

        let job = try service.submit(makeRequest())
        defer { _ = try? service.cancel(id: job.id) }

        var seen: H3VideoProgress?
        for _ in 0..<100 {
            if let p = service.progress(id: job.id) { seen = p; break }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        #expect(seen?.phase == "denoise")
        #expect(seen?.completed == 5)
        #expect(seen != nil && seen!.percent > 0)
    }
}
