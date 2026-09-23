import Foundation
import NovaMLXAudio
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

/// Word-level timestamps from the Swift Qwen3-ForcedAligner.
/// Weights must already be on disk. This does not download.
public enum ForcedAlignerService: Sendable {
    public static let defaultModelId = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    private static let lock = NSLock()
    private final class Cache: @unchecked Sendable {
        var model: Qwen3ASRModel?
        var path: String?
    }
    private static let cache = Cache()

    public static func isModelOnDisk() -> Bool {
        localModelPath() != nil
    }

    /// Load Qwen3-ForcedAligner so the first alignment is not a cold read.
    public static func ensureModel(at dir: URL? = nil) async throws {
        let url = try modelDirectory(dir)
        if lock.withLock({ cache.path == url.path && cache.model != nil }) { return }
        let model = try await Qwen3ASRModel.fromModelDirectory(url)
        guard model.config.isForcedAligner else {
            throw NovaMLXError.apiError("Not a Qwen3 ForcedAligner checkpoint: \(url.path)")
        }
        lock.withLock {
            cache.model = model
            cache.path = url.path
        }
    }

    public static func align(
        audioURL: URL,
        text: String,
        language: String? = nil
    ) async throws -> [AlignedWord] {
        try await ensureModel()
        let lang = resolvedLanguage(text: text, requested: language)
        let model = lock.withLock { cache.model }
        guard let model else {
            throw NovaMLXError.apiError("Forced aligner is not loaded")
        }
        let (_, audio) = try loadAudioArray(from: audioURL, sampleRate: 16000)
        let rows = try model.align(audio: audio, text: text, language: lang)
        return rows.map { AlignedWord(text: $0.text, start: $0.start, end: $0.end) }
    }

    public static func alignedWords(in text: String, language: String) -> [String] {
        Qwen3ForceAlignText.words(in: text, language: language)
    }

    public static func alignPrompt(words: [String], audioTokens: Int) -> String {
        Qwen3ForceAlignText.prompt(words: words, audioTokens: audioTokens)
    }

    public static func fixAlignTimestamps(_ data: [Int]) -> [Int] {
        Qwen3ForceAlignText.fixTimestamps(data)
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

    private static func localModelPath() -> String? {
        let dir = NovaMLXPaths.directory(forModelId: defaultModelId)
        let cfg = dir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfg.path) {
            return dir.path
        }
        return nil
    }

    private static func modelDirectory(_ dir: URL?) throws -> URL {
        if let dir {
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else {
                throw NovaMLXError.apiError("Forced aligner weights not found at \(dir.path)")
            }
            return dir
        }
        if let path = localModelPath() {
            return URL(fileURLWithPath: path)
        }
        throw NovaMLXError.apiError(
            "Forced aligner is not downloaded. Download \(defaultModelId) before aligning."
        )
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
