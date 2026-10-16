import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Configuration for the h3.c async video generation backend
/// (`POST /v1/videos`, OpenRouter-compatible).
///
/// Resolution order: built-in defaults → `~/.nova/h3_video.json` →
/// `NOVA_H3_*` environment variables.
public struct H3VideoConfig: Codable, Sendable {
    /// Absolute path to the `h3` CLI binary.
    public var binaryPath: String
    /// Absolute path to the MiniMax-H3 weights directory (passed as `-d`).
    public var weightsDir: String
    /// Directory holding one subdirectory per job (`<id>/job.json` + `out.mp4`).
    public var jobsDir: String
    /// Quality preset — the acceptance baseline for local generations.
    public var steps: Int
    public var layers: Int
    public var reuse: Int
    /// Hard wall-time cap for one h3 invocation, in seconds.
    public var processTimeoutSeconds: Double
    /// Seconds a finished job is kept before its mp4 is deleted and the
    /// job.json tombstone flips to `expired`.
    public var jobTTLSeconds: Double
    /// Max jobs allowed in pending + in_progress; beyond this submit → 429.
    public var maxQueuedJobs: Int
    /// When set, webhook POSTs carry an `X-OpenRouter-Signature` HMAC header.
    public var webhookSigningSecret: String?

    public init(
        binaryPath: String = H3VideoConfig.defaultBinaryPath,
        weightsDir: String = H3VideoConfig.defaultWeightsDir,
        jobsDir: String = H3VideoConfig.defaultJobsDir,
        steps: Int = 20,
        layers: Int = 45,
        reuse: Int = 2,
        processTimeoutSeconds: Double = 3600,
        jobTTLSeconds: Double = 86400,
        maxQueuedJobs: Int = 4,
        webhookSigningSecret: String? = nil
    ) {
        self.binaryPath = binaryPath
        self.weightsDir = weightsDir
        self.jobsDir = jobsDir
        self.steps = steps
        self.layers = layers
        self.reuse = reuse
        self.processTimeoutSeconds = processTimeoutSeconds
        self.jobTTLSeconds = jobTTLSeconds
        self.maxQueuedJobs = maxQueuedJobs
        self.webhookSigningSecret = webhookSigningSecret
    }

    public static let defaultBinaryPath = "/Users/lucas/dev/h3.c/h3"
    public static let defaultWeightsDir = "/Volumes/Samsung768/h3/MiniMax-H3"
    public static var defaultJobsDir: String {
        NovaMLXPaths.baseDir.appendingPathComponent("videos").path
    }

    /// Partial JSON (e.g. a hand-edited h3_video.json with two keys) fills the
    /// rest from defaults instead of failing to decode.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = H3VideoConfig()
        binaryPath = try c.decodeIfPresent(String.self, forKey: .binaryPath) ?? d.binaryPath
        weightsDir = try c.decodeIfPresent(String.self, forKey: .weightsDir) ?? d.weightsDir
        jobsDir = try c.decodeIfPresent(String.self, forKey: .jobsDir) ?? d.jobsDir
        steps = try c.decodeIfPresent(Int.self, forKey: .steps) ?? d.steps
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? d.layers
        reuse = try c.decodeIfPresent(Int.self, forKey: .reuse) ?? d.reuse
        processTimeoutSeconds = try c.decodeIfPresent(Double.self, forKey: .processTimeoutSeconds) ?? d.processTimeoutSeconds
        jobTTLSeconds = try c.decodeIfPresent(Double.self, forKey: .jobTTLSeconds) ?? d.jobTTLSeconds
        maxQueuedJobs = try c.decodeIfPresent(Int.self, forKey: .maxQueuedJobs) ?? d.maxQueuedJobs
        webhookSigningSecret = try c.decodeIfPresent(String.self, forKey: .webhookSigningSecret) ?? d.webhookSigningSecret
    }

    private enum CodingKeys: String, CodingKey {
        case binaryPath, weightsDir, jobsDir, steps, layers, reuse
        case processTimeoutSeconds, jobTTLSeconds, maxQueuedJobs, webhookSigningSecret
    }

    /// Load config: defaults → JSON file → environment overrides. Both the
    /// file path and the environment are injectable for tests.
    public static func load(
        fileURL: URL? = nil,
        environment: [String: String]? = nil
    ) -> H3VideoConfig {
        var cfg = H3VideoConfig()

        // ~/.nova/h3_video.json (base dir honours the NOVA_DIR convention)
        let file = fileURL ?? NovaMLXPaths.baseDir.appendingPathComponent("h3_video.json")
        if let data = try? Data(contentsOf: file),
           let parsed = try? JSONDecoder().decode(H3VideoConfig.self, from: data) {
            cfg = parsed
        }

        let env = environment ?? ProcessInfo.processInfo.environment
        if let v = env["NOVA_H3_BINARY"] { cfg.binaryPath = v }
        if let v = env["NOVA_H3_WEIGHTS_DIR"] { cfg.weightsDir = v }
        if let v = env["NOVA_H3_JOBS_DIR"] { cfg.jobsDir = v }
        if let v = env["NOVA_H3_STEPS"].flatMap(Int.init) { cfg.steps = v }
        if let v = env["NOVA_H3_LAYERS"].flatMap(Int.init) { cfg.layers = v }
        if let v = env["NOVA_H3_REUSE"].flatMap(Int.init) { cfg.reuse = v }
        if let v = env["NOVA_H3_TIMEOUT"].flatMap(Double.init) { cfg.processTimeoutSeconds = v }
        if let v = env["NOVA_H3_JOB_TTL"].flatMap(Double.init) { cfg.jobTTLSeconds = v }
        if let v = env["NOVA_H3_MAX_QUEUED"].flatMap(Int.init) { cfg.maxQueuedJobs = v }
        if let v = env["NOVA_H3_WEBHOOK_SECRET"] { cfg.webhookSigningSecret = v }
        return cfg
    }

    /// True when the configured CLI and weights directory both exist on disk.
    public var isReady: Bool {
        FileManager.default.isExecutableFile(atPath: binaryPath)
            && FileManager.default.fileExists(atPath: weightsDir)
    }
}
