import Foundation
import NovaMLXAudio
import NovaMLXCore
import NovaMLXUtils

/// Qwen3-TTS Base voice cloning in Swift. The vocoder decodes generated
/// codec frames only. CustomVoice / VoiceDesign checkpoints are rejected.
public enum Qwen3TTSCloneService: Sendable {
    private static let lock = NSLock()
    private final class Holder: @unchecked Sendable {
        var model: Qwen3TTSCloneModel?
        var path: String?
    }
    private static let holder = Holder()
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
        let url = try modelDirectory(dir)
        if lock.withLock({ holder.path == url.path && holder.model != nil }) { return }
        let model = try await Qwen3TTSCloneModel.load(directory: url)
        lock.withLock {
            holder.model = model
            holder.path = url.path
        }
    }

    public static func synthesize(
        text: String,
        refAudio: URL,
        refText: String,
        modelDir: URL,
        output: URL? = nil
    ) async throws -> Data {
        try await ensureModel(at: modelDir)
        let model = lock.withLock { holder.model }
        guard let model else {
            throw NovaMLXError.apiError("Qwen3-TTS clone model is not loaded")
        }
        let (_, audio) = try loadAudioArray(from: refAudio, sampleRate: model.sampleRate)
        let samples = try model.synthesize(
            text: text, refAudio: audio, refText: refText, language: nil, temperature: 0.6
        )
        let dest = output ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx_qwen3tts_\(UUID().uuidString).wav")
        try AudioUtils.writeWavFile(samples: samples, sampleRate: model.sampleRate, fileURL: dest)
        let data = try Data(contentsOf: dest)
        if output == nil {
            try? FileManager.default.removeItem(at: dest)
        }
        return data
    }

    public static func uniqueRefPrefix(_ refText: String, _ text: String) -> String {
        Qwen3TTSCloneModel.uniqueRefPrefix(refText, text)
    }

    public static func prefixCutSamples(out: Int, ref: Int, text: String, refText: String) -> Int? {
        Qwen3TTSCloneModel.prefixCutSamples(out: out, ref: ref, text: text, refText: refText)
    }

    public static func decodePrefetchJSON(_ data: Data) throws -> Bool {
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return obj?["ok"] as? Bool == true
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

    private static func localModelPath() -> String? {
        let dir = NovaMLXPaths.directory(forModelId: defaultModelId)
        if isQwen3TTSDirectory(dir) { return dir.path }
        let cfg = dir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: cfg.path) { return dir.path }
        return nil
    }

    private static func modelDirectory(_ dir: URL?) throws -> URL {
        if let dir {
            guard isQwen3TTSDirectory(dir) || FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("config.json").path
            ) else {
                throw NovaMLXError.apiError("Qwen3-TTS weights not found at \(dir.path)")
            }
            return dir
        }
        if let path = localModelPath() {
            return URL(fileURLWithPath: path)
        }
        throw NovaMLXError.apiError(
            "Qwen3-TTS Base is not downloaded. Download \(defaultModelId) before cloning."
        )
    }
}
