import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct AlignedWord: Sendable {
    public let text: String
    public let start: Double
    public let end: Double

    public init(text: String, start: Double, end: Double) {
        self.text = text
        self.start = start
        self.end = end
    }
}

/// Word-level timestamps via mlx-audio Qwen3-ForcedAligner (Python).
public enum ForcedAlignerService: Sendable {
    public static let defaultModelId = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    public static func isModelOnDisk() -> Bool {
        localModelPath() != nil
    }

    /// Download/load Qwen3-ForcedAligner so the first alignment is not a surprise download.
    public static func ensureModel(at dir: URL? = nil) async throws {
        let script = try scriptURL()
        let python = pythonExecutable()
        let model = dir?.path ?? localModelPath() ?? defaultModelId
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = [script.path, "--prefetch", "--model", model]
        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr
        proc.environment = ProcessInfo.processInfo.environment
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let err = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw NovaMLXError.apiError("Forced aligner prefetch failed: \(err)")
        }
    }

    public static func align(
        audioURL: URL,
        text: String,
        language: String? = nil
    ) async throws -> [AlignedWord] {
        let script = try scriptURL()
        let python = pythonExecutable()
        let lang = resolvedLanguage(text: text, requested: language)
        let model = localModelPath() ?? defaultModelId

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: python)
        proc.arguments = [
            script.path,
            "--audio", audioURL.path,
            "--text", text,
            "--language", lang,
            "--model", model,
        ]
        let stdout = Pipe()
        let stderr = Pipe()
        proc.standardOutput = stdout
        proc.standardError = stderr
        proc.environment = ProcessInfo.processInfo.environment

        try proc.run()
        proc.waitUntilExit()

        let outData = stdout.fileHandleForReading.readDataToEndOfFile()
        let errData = stderr.fileHandleForReading.readDataToEndOfFile()
        if proc.terminationStatus != 0 {
            let err = String(data: errData, encoding: .utf8) ?? "aligner failed"
            throw NovaMLXError.apiError("Forced aligner failed: \(err)")
        }
        return try decodeWords(outData)
    }

    public static func decodeWords(_ data: Data) throws -> [AlignedWord] {
        let rows = try JSONDecoder().decode([Row].self, from: data)
        return rows.map { AlignedWord(text: $0.text, start: $0.start, end: $0.end) }
    }

    public static func resolvedLanguage(text: String, requested: String?) -> String {
        if let requested, !requested.isEmpty { return requested }
        return text.unicodeScalars.contains(where: { isCJK($0) }) ? "Chinese" : "English"
    }

    private struct Row: Decodable {
        let text: String
        let start: Double
        let end: Double
    }

    private static func isCJK(_ s: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(s.value)
            || (0x3400...0x4DBF).contains(s.value)
            || (0x3040...0x30FF).contains(s.value)
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
        let cfg = dir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfg.path) {
            return dir.path
        }
        return nil
    }

    private static func scriptURL() throws -> URL {
        if let bundle = ResourceBundleLocator.find(bundleName: "NovaMLX_NovaMLXUtils") {
            let subs = ["scripts", "Resources/scripts", nil] as [String?]
            for sub in subs {
                if let url = bundle.url(
                    forResource: "forced_align", withExtension: "py", subdirectory: sub)
                {
                    return url
                }
            }
        }
        let sourceRelatives = [
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("NovaMLXUtils/Resources/scripts/forced_align.py"),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("Sources/NovaMLXUtils/Resources/scripts/forced_align.py"),
        ]
        for url in sourceRelatives where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        throw NovaMLXError.apiError("forced_align.py not found in app resources")
    }
}

/// Reference-clip language for Voice Clone (script + ASR + ForcedAligner).
public enum VoiceCloneLanguage: String, CaseIterable, Identifiable, Sendable {
    case chinese
    case english

    public var id: String { rawValue }

    public var alignerLanguage: String {
        switch self {
        case .chinese: return "Chinese"
        case .english: return "English"
        }
    }

    public var asrCode: String {
        switch self {
        case .chinese: return "zh"
        case .english: return "en"
        }
    }

    public var defaultTranscript: String {
        switch self {
        case .chinese:
            return "今天天气不错，阳光明媚，微风轻拂。我喜欢在这样的日子里，到公园里散散步，听听鸟儿的歌唱，感受大自然的美好。"
        case .english:
            return "The weather today is quite pleasant, with a gentle breeze blowing through the trees. I enjoy taking a walk in the park on days like this, listening to the birds singing and feeling the warmth of the sun."
        }
    }

    /// Replace the script when it is empty or still the previous language's default.
    public static func shouldReplaceTranscript(_ current: String, switchingFrom old: VoiceCloneLanguage) -> Bool {
        let t = current.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty || t == old.defaultTranscript
    }
}
