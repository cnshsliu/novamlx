import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Qwen3-TTS Base voice cloning via mlx-audio (Python).
/// CustomVoice / VoiceDesign checkpoints are rejected — they are a different recipe.
public enum Qwen3TTSCloneService: Sendable {
    public static let defaultModelId = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    public static let dotsModelId = "smcleod/dots.tts-soar-mlx"

    public struct EngineOption: Equatable, Sendable, Identifiable {
        public let id: String
        public let family: ModelFamily
        public init(id: String, family: ModelFamily) {
            self.id = id
            self.family = family
        }
    }

    public static func isBaseModelId(_ id: String) -> Bool {
        let l = id.lowercased()
        if l.contains("customvoice") || l.contains("custom-voice") { return false }
        if l.contains("voicedesign") || l.contains("voice-design") { return false }
        return l.contains("qwen3-tts") && l.contains("base")
    }

    public static func isAllowedCloneEngine(id: String, family: ModelFamily) -> Bool {
        switch family {
        case .dotsTts: return true
        case .qwen3Tts: return isBaseModelId(id)
        default: return false
        }
    }

    /// Pinned catalog engines first, then extras. No silent substitution.
    public static func mergeEngineList(
        pinned: [EngineOption],
        extra: [EngineOption]
    ) -> [EngineOption] {
        var seen = Set<String>()
        var out: [EngineOption] = []
        for opt in pinned + extra {
            guard isAllowedCloneEngine(id: opt.id, family: opt.family) else { continue }
            guard seen.insert(opt.id).inserted else { continue }
            out.append(opt)
        }
        return out
    }

    public static func isQwen3TTSDirectory(_ dir: URL) -> Bool {
        if isBaseModelId(dir.lastPathComponent) { return true }
        let cfg = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        let modelType = ((obj["model_type"] as? String) ?? "").lowercased()
        if modelType.contains("qwen3_tts") || modelType.contains("qwen3-tts") { return true }
        if let archs = obj["architectures"] as? [String],
           archs.contains(where: { $0.lowercased().contains("qwen3tts") })
        {
            return true
        }
        let speechTok = dir.appendingPathComponent("speech_tokenizer/config.json")
        return FileManager.default.fileExists(atPath: speechTok.path)
            && dir.lastPathComponent.lowercased().contains("tts")
    }

    public static func ensureModel(at dir: URL? = nil) async throws {
        let model = dir?.path ?? localModelPath() ?? defaultModelId
        try runPython(arguments: ["--prefetch", "--model", model])
    }

    public static func synthesize(
        text: String,
        refAudio: URL,
        refText: String,
        modelDir: URL,
        output: URL? = nil
    ) async throws -> Data {
        let dest = output ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx_qwen3tts_\(UUID().uuidString).wav")
        try runPython(arguments: [
            "--model", modelDir.path,
            "--text", text,
            "--ref-audio", refAudio.path,
            "--ref-text", refText,
            "--output", dest.path,
        ])
        let data = try Data(contentsOf: dest)
        if output == nil {
            try? FileManager.default.removeItem(at: dest)
        }
        return data
    }

    public static func decodePrefetchJSON(_ data: Data) throws -> Bool {
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return obj?["ok"] as? Bool == true
    }

    private static func runPython(arguments: [String]) throws {
        let script = try scriptURL()
        let python = pythonExecutable()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = [script.path] + arguments
        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr
        var env = ProcessInfo.processInfo.environment
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        proc.environment = env
        try proc.run()
        proc.waitUntilExit()
        let out = String(data: stdout.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let err = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        if !out.isEmpty {
            NovaMLXLog.info("[Qwen3TTS] \(out.trimmingCharacters(in: .whitespacesAndNewlines).prefix(500))")
        }
        if proc.terminationStatus != 0 {
            NovaMLXLog.error("[Qwen3TTS] clone failed: \(err.prefix(800))")
            throw NovaMLXError.apiError(Self.shortCloneError(stderr: err, stdout: out))
        }
    }

    static func shortCloneError(stderr: String, stdout: String) -> String {
        for blob in [stderr, stdout] {
            if let msg = extractJSONError(blob) {
                return "Qwen3-TTS clone failed: \(msg)"
            }
        }
        let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Qwen3-TTS clone failed"
        }
        if let last = trimmed.split(whereSeparator: \.isNewline).last {
            return "Qwen3-TTS clone failed: \(String(last).prefix(240))"
        }
        return "Qwen3-TTS clone failed: \(trimmed.prefix(240))"
    }

    private static func extractJSONError(_ blob: String) -> String? {
        guard let start = blob.range(of: "{\"error\"", options: .backwards) else { return nil }
        let json = String(blob[start.lowerBound...])
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let msg = obj["error"] as? String
        else { return nil }
        return msg
    }

    private static func pythonExecutable() -> String {
        let candidates = [
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3",
        ]
        for path in candidates where FileManager.default.isExecutableFile(atPath: path) {
            return path
        }
        return "python3"
    }

    private static func localModelPath() -> String? {
        let dir = NovaMLXPaths.directory(forModelId: defaultModelId)
        if isQwen3TTSDirectory(dir) { return dir.path }
        let cfg = dir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfg.path) { return dir.path }
        return nil
    }

    private static func scriptURL() throws -> URL {
        if let bundle = ResourceBundleLocator.find(bundleName: "NovaMLX_NovaMLXUtils") {
            let subs = ["scripts", "Resources/scripts", nil] as [String?]
            for sub in subs {
                if let url = bundle.url(
                    forResource: "qwen3_tts_clone", withExtension: "py", subdirectory: sub)
                {
                    return url
                }
            }
        }
        let sourceRelatives = [
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("NovaMLXUtils/Resources/scripts/qwen3_tts_clone.py"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Sources/NovaMLXUtils/Resources/scripts/qwen3_tts_clone.py"),
        ]
        for url in sourceRelatives where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        throw NovaMLXError.apiError("qwen3_tts_clone.py not found in app resources")
    }
}
