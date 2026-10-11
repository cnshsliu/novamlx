import CoreGraphics
import Foundation
import ImageIO
import NovaMLXCore
import NovaMLXUtils

/// Configuration for the LTX-2.5 talking-head (audio-to-video) pipeline.
///
/// The default engine is the in-process Swift runner (`LTX2A2VRunner`, vendored
/// xocialize/ltx-2-mlx-swift — pure Swift on the same MLX Metal core). The
/// external Python CLI (`ltx-2-mlx` in a uv venv) remains available as a
/// fallback via `engine: "python"` / `NOVA_TALKINGHEAD_ENGINE=python` until
/// the M3 parity gates fully close.
///
/// Resolution order: built-in defaults → `~/.nova/talking_head.json` →
/// `NOVA_TALKINGHEAD_*` environment variables → per-request API parameters.
public struct TalkingHeadConfig: Codable, Sendable {
    /// Absolute path to the `ltx-2-mlx` executable inside its uv venv.
    public var cliPath: String
    /// Absolute path to the LTX-2.5 pack directory (safetensors + configs).
    public var modelDir: String
    /// Prompt used when the request does not carry one.
    public var defaultPrompt: String
    public var defaultWidth: Int
    public var defaultHeight: Int
    public var defaultFrameRate: Int
    /// Longest single generation segment, in seconds. The LTX pipeline caps a
    /// clip at 257 frames (~10.7 s at 24 fps); longer audio is sliced, generated
    /// per segment, and concatenated.
    public var maxSegmentSeconds: Double
    /// Hard wall-time cap for one CLI invocation, in seconds.
    public var processTimeoutSeconds: Double
    /// True to use `a2v --distilled` (fast, experimental). False uses the full
    /// `a2v` pipeline (CFG, ~5x slower) — requires the dev transformer weights.
    public var distilled: Bool

    public init(
        cliPath: String = TalkingHeadConfig.defaultCLIPath,
        modelDir: String = TalkingHeadConfig.defaultModelDir,
        defaultPrompt: String = "A person talks calmly to the camera in a studio portrait, natural small head movements, clear lip movements",
        defaultWidth: Int = 576,
        defaultHeight: Int = 1024,
        defaultFrameRate: Int = 24,
        maxSegmentSeconds: Double = 10.0,
        processTimeoutSeconds: Double = 3600,
        distilled: Bool = true
    ) {
        self.cliPath = cliPath
        self.modelDir = modelDir
        self.defaultPrompt = defaultPrompt
        self.defaultWidth = defaultWidth
        self.defaultHeight = defaultHeight
        self.defaultFrameRate = defaultFrameRate
        self.maxSegmentSeconds = maxSegmentSeconds
        self.processTimeoutSeconds = processTimeoutSeconds
        self.distilled = distilled
    }

    public static let defaultCLIPath = NSString(string: "~/agent-tools/ltx-a2v/ltx-2-mlx/.venv/bin/ltx-2-mlx").expandingTildeInPath
    public static let defaultModelDir = "/Volumes/WD/nova-models/ltx-2.5-q8"

    /// Partial JSON (e.g. a hand-edited talking_head.json with two keys)
    /// fills the rest from defaults instead of failing to decode.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = TalkingHeadConfig()
        cliPath = try c.decodeIfPresent(String.self, forKey: .cliPath) ?? d.cliPath
        modelDir = try c.decodeIfPresent(String.self, forKey: .modelDir) ?? d.modelDir
        defaultPrompt = try c.decodeIfPresent(String.self, forKey: .defaultPrompt) ?? d.defaultPrompt
        defaultWidth = try c.decodeIfPresent(Int.self, forKey: .defaultWidth) ?? d.defaultWidth
        defaultHeight = try c.decodeIfPresent(Int.self, forKey: .defaultHeight) ?? d.defaultHeight
        defaultFrameRate = try c.decodeIfPresent(Int.self, forKey: .defaultFrameRate) ?? d.defaultFrameRate
        maxSegmentSeconds = try c.decodeIfPresent(Double.self, forKey: .maxSegmentSeconds) ?? d.maxSegmentSeconds
        processTimeoutSeconds = try c.decodeIfPresent(Double.self, forKey: .processTimeoutSeconds) ?? d.processTimeoutSeconds
        distilled = try c.decodeIfPresent(Bool.self, forKey: .distilled) ?? d.distilled
    }

    private enum CodingKeys: String, CodingKey {
        case cliPath, modelDir, defaultPrompt, defaultWidth, defaultHeight
        case defaultFrameRate, maxSegmentSeconds, processTimeoutSeconds, distilled
    }

    /// Load config: defaults → JSON file → environment overrides.
    public static func load() -> TalkingHeadConfig {
        var cfg = TalkingHeadConfig()

        // ~/.nova/talking_head.json (base dir honours the NOVA_DIR convention)
        let fileURL = NovaMLXPaths.baseDir.appendingPathComponent("talking_head.json")
        if let data = try? Data(contentsOf: fileURL),
           let parsed = try? JSONDecoder().decode(TalkingHeadConfig.self, from: data) {
            cfg = parsed
        }

        // Environment overrides — handy for tests and multi-instance setups.
        let env = ProcessInfo.processInfo.environment
        if let v = env["NOVA_TALKINGHEAD_CLI"] { cfg.cliPath = v }
        if let v = env["NOVA_TALKINGHEAD_MODEL_DIR"] { cfg.modelDir = v }
        if let v = env["NOVA_TALKINGHEAD_TIMEOUT"].flatMap(Double.init) { cfg.processTimeoutSeconds = v }
        return cfg
    }

    /// True when the configured CLI and model pack both exist on disk
    /// (Python-engine readiness).
    public var isReady: Bool {
        FileManager.default.isExecutableFile(atPath: cliPath)
            && FileManager.default.fileExists(atPath: modelDir.appending("/embedded_config.json"))
    }

    /// True when the model pack alone exists — all the Swift engine needs
    /// (in-process runner; no external CLI).
    public var isSwiftReady: Bool {
        FileManager.default.fileExists(atPath: modelDir.appending("/embedded_config.json"))
    }
}

public struct TalkingHeadSegmentResult: Sendable {
    public let index: Int
    public let audioSeconds: Double
    public let frames: Int
    public let elapsed: Double
}

public struct TalkingHeadResult: Sendable {
    /// Final mp4 (concatenated when the audio needed multiple segments).
    public let outputURL: URL
    public let model: String
    public let prompt: String
    public let seed: UInt64
    public let width: Int
    public let height: Int
    public let frameRate: Int
    public let audioSeconds: Double
    public let segments: [TalkingHeadSegmentResult]
    public let elapsed: Double
}

/// Audio-to-video talking-head generation (in-process Swift; Python CLI fallback).
///
/// Single-flight like `ImageGenerationService`: one GPU job at a time; a second
/// concurrent request fails fast with the active model name instead of queueing
/// (video jobs run 5–15 min, queueing would be a footgun for API timeouts).
public final class TalkingHeadService: @unchecked Sendable {

    public var metricsStore: MetricsStore?

    public let config: TalkingHeadConfig

    private let lock = NovaMLXLock()
    private var isGenerating = false
    private var progress: JobProgress?

    public struct JobProgress: Sendable {
        public let modelId: String
        public let segment: Int
        public let totalSegments: Int
        public let startedAt: Date
    }

    /// Which a2v engine serves requests. The Swift in-process runner is the
    /// default (pure-Swift mandate); "python" keeps the external ltx-2-mlx CLI
    /// as an escape hatch (config or env) until M3 parity gates fully close.
    public enum Engine: String, Codable, Sendable {
        case swift
        case python
    }

    public init(config: TalkingHeadConfig = TalkingHeadConfig.load()) {
        self.config = config
        if let e = ProcessInfo.processInfo.environment["NOVA_TALKINGHEAD_ENGINE"],
           let parsed = Engine(rawValue: e.lowercased()) {
            engine = parsed
        }
    }

    public private(set) var engine: Engine = .swift

    public func inFlight() -> JobProgress? {
        lock.withLock { isGenerating ? progress : nil }
    }

    /// Engine-aware readiness: the Swift runner needs only the pack.
    public var isReady: Bool {
        engine == .swift ? config.isSwiftReady : config.isReady
    }

    /// Generate a lip-synced video from a portrait image + narration audio.
    ///
    /// Audio longer than `config.maxSegmentSeconds` is sliced with ffmpeg,
    /// each slice is generated separately, and the clips are concatenated;
    /// the output video duration always tracks the audio duration.
    public func generate(
        image: URL,
        audio: URL,
        prompt: String? = nil,
        width: Int? = nil,
        height: Int? = nil,
        frameRate: Int? = nil,
        seed: UInt64? = nil,
        distilled: Bool? = nil
    ) async throws -> TalkingHeadResult {
        guard isReady else {
            throw NovaMLXError.configurationError(
                "Talking-head pipeline is not configured: \(engine == .swift ? "need model pack at \'\(config.modelDir)\'" : "need executable at \'\(config.cliPath)\' and model pack at \'\(config.modelDir)\'")"
            )
        }
        guard FileManager.default.fileExists(atPath: image.path) else {
            throw NovaMLXError.apiError("Image file not found: \(image.path)")
        }
        guard FileManager.default.fileExists(atPath: audio.path) else {
            throw NovaMLXError.apiError("Audio file not found: \(audio.path)")
        }

        let blocker = lock.withLock { () -> String? in
            if isGenerating {
                return progress?.modelId ?? "talking-head"
            }
            isGenerating = true
            return nil
        }
        if let blocker {
            throw NovaMLXError.apiError("Talking-head generation already in progress for \(blocker)")
        }
        defer {
            lock.withLock {
                isGenerating = false
                progress = nil
            }
        }

        let modelId = "ltx-2.5-a2v"
        let useDistilled = distilled ?? config.distilled
        let w = Self.roundTo64(width ?? config.defaultWidth)
        let h = Self.roundTo64(height ?? config.defaultHeight)
        let fps = max(1, frameRate ?? config.defaultFrameRate)
        let resolvedPrompt = (prompt?.isEmpty == false ? prompt : config.defaultPrompt)!
        let resolvedSeed = seed ?? UInt64(Date().timeIntervalSince1970 * 1000)

        // Two-stage pipelines require W/H multiples of 64; snap silently and
        // report the actual values back in the result.
        if let requested = width, Self.roundTo64(requested) != requested {
            NovaMLXLog.warning("TalkingHead: width \(requested) rounded to \(w) (must be a multiple of 64)")
        }
        if let requested = height, Self.roundTo64(requested) != requested {
            NovaMLXLog.warning("TalkingHead: height \(requested) rounded to \(h) (must be a multiple of 64)")
        }

        let audioSeconds = try VideoSlicePipeline.probeDuration(audio)
        guard audioSeconds > 0.1 else {
            throw NovaMLXError.apiError("Audio duration could not be determined (is ffprobe installed?)")
        }

        // Slice plan: even chunks under the per-clip frame cap. The VAE
        // upsamples time 8x, so pixel frames must have the form 8k+1.
        let maxFrames = 257
        let segmentCount = max(1, Int(ceil(audioSeconds / config.maxSegmentSeconds)))
        let segSeconds = audioSeconds / Double(segmentCount)
        if segSeconds * Double(fps) > Double(maxFrames - 1) {
            throw NovaMLXError.apiError("Audio too long: \(String(format: "%.1f", audioSeconds))s exceeds the \(String(format: "%.1f", config.maxSegmentSeconds))s per-segment limit")
        }

        let startedAt = Date()
        lock.withLock {
            progress = JobProgress(modelId: modelId, segment: 0, totalSegments: segmentCount, startedAt: startedAt)
        }

        metricsStore?.reportActivity(model: modelId, kind: .video, speed: 0, unit: "seg/s")
        let heartbeat = Task { [metricsStore, modelId] in
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

        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx_talkinghead_\(Int(Date().timeIntervalSince1970))")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)

        var segmentFiles: [URL] = []
        var segmentInfos: [TalkingHeadSegmentResult] = []

        for i in 0..<segmentCount {
            lock.withLock { progress = JobProgress(modelId: modelId, segment: i, totalSegments: segmentCount, startedAt: startedAt) }

            let segStart = Double(i) * segSeconds
            let segDur = min(segSeconds, audioSeconds - segStart)

            // Frame count first: pixel frames must be 8k+1, which can round
            // ABOVE the exact audio length (2.5 s @ 24 fps → 60 → 65 frames
            // = 2.71 s). The audio encoder refuses a clip longer than the
            // track, so pad the slice with silence to the frame duration.
            let frames = Self.framesFor(seconds: segDur, fps: fps)
            let targetSeconds = Double(frames) / Double(fps)

            let segAudio = workDir.appendingPathComponent("audio_\(i).wav")
            try VideoSlicePipeline.run(VideoSlicePipeline.ffmpegPath(), [
                "-y",
                "-ss", String(format: "%.3f", segStart), "-t", String(format: "%.3f", segDur),
                "-i", audio.path,
                "-af", "apad",
                "-t", String(format: "%.3f", targetSeconds),
                segAudio.path,
            ])

            let segOut = workDir.appendingPathComponent("segment_\(i).mp4")

            // Every segment gets the SAME seed + the same reference photo, so
            // face / hair / lighting stay consistent across the slices of one
            // episode (fixed-seed strategy requested by the video pipeline).
            let segStart2 = Date()
            switch engine {
            case .swift:
                let cgImage = try Self.loadCGImage(image)
                _ = try await LTX2A2VRunner.shared.generate(
                    image: cgImage,
                    wavURL: segAudio,
                    prompt: resolvedPrompt,
                    width: w,
                    height: h,
                    frames: frames,
                    fps: fps,
                    seed: resolvedSeed,
                    packDir: URL(fileURLWithPath: config.modelDir),
                    outputURL: segOut)
            case .python:
                _ = try await Self.runCLI(
                    config: config,
                    distilled: useDistilled,
                    prompt: resolvedPrompt,
                    image: image,
                    audio: segAudio,
                    width: w,
                    height: h,
                    frames: frames,
                    frameRate: fps,
                    seed: resolvedSeed,
                    output: segOut
                )
            }
            segmentFiles.append(segOut)
            segmentInfos.append(TalkingHeadSegmentResult(
                index: i,
                audioSeconds: segDur,
                frames: frames,
                elapsed: Date().timeIntervalSince(segStart2)
            ))
            metricsStore?.reportActivity(model: modelId, kind: .video, speed: Double(i + 1), unit: "seg")
        }

        // Concatenate when needed; the generator muxes each segment's own
        // audio, so a stream-copy concat keeps the original timing per segment.
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx_video_\(Int(Date().timeIntervalSince1970)).mp4")
        if segmentFiles.count == 1 {
            try FileManager.default.copyItem(at: segmentFiles[0], to: outputURL)
        } else {
            try Self.concatClips(segmentFiles, to: outputURL)
        }

        let elapsed = Date().timeIntervalSince(startedAt)
        NovaMLXLog.info("TalkingHead complete: \(String(format: "%.1f", audioSeconds))s audio, \(segmentCount) segment(s), \(String(format: "%.1f", elapsed))s total")

        return TalkingHeadResult(
            outputURL: outputURL,
            model: modelId,
            prompt: resolvedPrompt,
            seed: resolvedSeed,
            width: w,
            height: h,
            frameRate: fps,
            audioSeconds: audioSeconds,
            segments: segmentInfos,
            elapsed: elapsed
        )
    }

    // MARK: - Internals

    /// Image URL → CGImage for the Swift a2v runner.
    private static func loadCGImage(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else {
            throw NovaMLXError.apiError("Cannot decode image: \(url.lastPathComponent)")
        }
        return cg
    }

    static func roundTo64(_ value: Int) -> Int {
        max(64, (value / 64) * 64)
    }

    /// Pixel frame count for a clip duration: the temporal VAE upsamples 8x,
    /// so the count must be 8k+1 (e.g. 10 s at 24 fps → 241 = 8×30+1).
    static func framesFor(seconds: Double, fps: Int) -> Int {
        let exact = Int((seconds * Double(fps)).rounded())
        let k = max(1, Int(ceil(Double(max(0, exact - 1)) / 8.0)))
        return min(257, 8 * k + 1)
    }

    /// Run one `ltx-2-mlx a2v` invocation on a background thread.
    private static func runCLI(
        config: TalkingHeadConfig,
        distilled: Bool,
        prompt: String,
        image: URL,
        audio: URL,
        width: Int,
        height: Int,
        frames: Int,
        frameRate: Int,
        seed: UInt64,
        output: URL
    ) async throws -> SafeProcess.Result {
        var args = [
            "a2v",
            "--model", config.modelDir,
            "--prompt", prompt,
            "--audio", audio.path,
            "--image", image.path,
            "--height", String(height),
            "--width", String(width),
            "--frames", String(frames),
            "--frame-rate", String(frameRate),
            "--seed", String(seed),
            "--output", output.path,
        ]
        if distilled { args.append("--distilled") }
        let finalArgs = args

        NovaMLXLog.info("TalkingHead: \(config.cliPath) \(finalArgs.joined(separator: " "))")

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let result = try SafeProcess.run(
                        config.cliPath,
                        arguments: finalArgs,
                        deadline: config.processTimeoutSeconds,
                        allowTimeout: false
                    )
                    if result.succeeded {
                        continuation.resume(returning: result)
                    } else {
                        let tail = result.stderrText.split(separator: "\n").suffix(5).joined(separator: " | ")
                        continuation.resume(throwing: NovaMLXError.inferenceFailed(
                            "ltx-2-mlx a2v failed (\(result.status)): \(tail)"
                        ))
                    }
                } catch {
                    continuation.resume(throwing: NovaMLXError.inferenceFailed("ltx-2-mlx a2v failed: \(error.localizedDescription)"))
                }
            }
        }
    }

    /// Concatenate same-codec mp4 clips with the concat demuxer (stream copy).
    private static func concatClips(_ clips: [URL], to output: URL) throws {
        let listURL = output.deletingLastPathComponent().appendingPathComponent("concat_\(UUID().uuidString).txt")
        // Apostrophes in paths would break the concat file's quoting; these are
        // our own temp paths, but escape anyway.
        let list = clips.map { "file '\($0.path.replacingOccurrences(of: "'", with: "'\\''"))'" }.joined(separator: "\n")
        try list.write(to: listURL, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: listURL) }
        try VideoSlicePipeline.run(VideoSlicePipeline.ffmpegPath(), [
            "-y", "-f", "concat", "-safe", "0", "-i", listURL.path, "-c", "copy", output.path,
        ])
    }
}
