import Foundation
import os.log
import MLX
import MLXLMCommon
import Tokenizers
import DotsTTS
import NovaMLXCore
import NovaMLXUtils
import NovaMLXAudio

private let ttsLog = Logger(subsystem: "com.novamlx", category: "TTS")
private func ttsPrint(_ msg: String) { NovaMLXLog.info("[TTS] \(msg)") }

public enum TTSEngine: String, CaseIterable, Sendable {
    case neural = "Neural TTS"
    case system = "System TTS"
}

public struct MacOSVoice: Identifiable, Hashable, Sendable {
    public let name: String
    public let locale: String
    public var id: String { "\(name) (\(locale))" }
}

public final class TTSService: @unchecked Sendable {
    private let lock = NovaMLXLock()
    private var pipeline: DotsTTSPipeline?
    private var qwen3ModelDir: URL?

    /// Shared metrics store. Set by InferenceService after construction so the
    /// status panel can show live TTS activity. Weak to avoid a retain cycle.
    public var metricsStore: MetricsStore?
    private var loadedModelId: String?

    public init() {}

    public func listLoadedModels() -> [String] {
        lock.withLock {
            if let id = loadedModelId { return [id] }
            return []
        }
    }

    public func loadModel(from dir: URL) async throws {
        ttsLog.info("[TTS] ====== START loadModel from \(dir.path) ======")

        let hadPrevious = lock.withLock { pipeline != nil || qwen3ModelDir != nil }
        if hadPrevious {
            ttsLog.info("[TTS] Replacing existing TTS model, clearing GPU cache...")
            lock.withLock { pipeline = nil; qwen3ModelDir = nil; loadedModelId = nil }
            MLX.Memory.clearCache()
        }

        let dirContents = try? FileManager.default.contentsOfDirectory(atPath: dir.path)
        ttsLog.info("[TTS] Directory contents: \(dirContents ?? [])")

        let modelId = dir.pathComponents.last.flatMap { p in
            dir.pathComponents.count >= 2 ? "\(dir.pathComponents.dropLast().last!)/\(p)" : p
        } ?? dir.lastPathComponent

        if Qwen3TTSCloneService.isQwen3TTSDirectory(dir) {
            ttsLog.info("[TTS] Loading Qwen3-TTS Base clone backend: \(dir.path)")
            try await Qwen3TTSCloneService.ensureModel(at: dir)
            lock.withLock {
                self.pipeline = nil
                self.qwen3ModelDir = dir
                self.loadedModelId = modelId
            }
            ttsLog.info("[TTS] ====== loadModel COMPLETE (Qwen3-TTS) for \(modelId) ======")
            return
        }

        ttsLog.info("[TTS] Loading DotsTTS pipeline...")
        let loadedPipeline: DotsTTSPipeline
        do {
            // tokenizer.json lives in backbone/ subdirectory for dots.tts models
            let tokenizerDir = dir.appendingPathComponent("backbone")
            let fallbackDir = FileManager.default.fileExists(atPath: tokenizerDir.appendingPathComponent("tokenizer.json").path) ? tokenizerDir : dir
            let tokenizer = try await AutoTokenizer.from(modelFolder: fallbackDir)
            loadedPipeline = try DotsTTSPipeline(modelRepo: dir, tokenizer: tokenizer)
            ttsLog.info("[TTS] DotsTTSPipeline loaded successfully")
        } catch {
            ttsLog.error("[TTS] DotsTTSPipeline load FAILED: \(error)")
            throw error
        }

        lock.withLock {
            self.pipeline = loadedPipeline
            self.qwen3ModelDir = nil
            self.loadedModelId = modelId
        }

        ttsLog.info("[TTS] ====== loadModel COMPLETE for \(modelId) ======")
    }

    public func isModelLoaded() -> Bool {
        lock.withLock { pipeline != nil || qwen3ModelDir != nil }
    }

    public func unloadModel() {
        lock.withLock {
            pipeline = nil
            qwen3ModelDir = nil
            loadedModelId = nil
        }
        MLX.Memory.clearCache()
        ttsLog.info("[TTS] Model unloaded")
    }

    public static func listMacOSVoices() -> [MacOSVoice] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-v", "?"]

        let pipe = Pipe()
        process.standardOutput = pipe

        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            let localePattern = try NSRegularExpression(pattern: "\\s+([a-z]{2}_[A-Z]{2})\\s+#")
            return output.split(separator: "\n").compactMap { line in
                let lineStr = String(line)
                guard let match = localePattern.firstMatch(in: lineStr, range: NSRange(lineStr.startIndex..., in: lineStr)),
                      let localeRange = Range(match.range(at: 1), in: lineStr) else { return nil }
                let locale = String(lineStr[localeRange])
                let nameEnd = lineStr.index(lineStr.startIndex, offsetBy: (match.range(at: 1).location))
                let name = lineStr[..<nameEnd].trimmingCharacters(in: .whitespaces)
                return MacOSVoice(name: name, locale: locale)
            }
        } catch {
            return [MacOSVoice(name: "Tingting", locale: "zh_CN")]
        }
    }

    // MARK: - Synthesize (DotsTTS)

    public func synthesize(
        text: String,
        voice: String = "Tingting",
        rate: Int = 175
    ) async throws -> Data {
        return try await synthesize(text: text, voice: voice, rate: rate, engine: nil, voiceProfile: nil)
    }

    public func synthesize(
        text: String,
        voice: String = "Tingting",
        rate: Int = 175,
        engine: TTSEngine? = nil,
        voiceProfile: VoiceProfile? = nil
    ) async throws -> Data {
        let resolvedEngine: TTSEngine
        if let engine {
            resolvedEngine = engine
        } else {
            resolvedEngine = lock.withLock { pipeline != nil || qwen3ModelDir != nil } ? .neural : .system
        }

        switch resolvedEngine {
        case .neural:
            let profile = voiceProfile ?? findDefaultVoiceProfile()
            guard let profile else {
                throw NovaMLXError.apiError("No voice profile available. Clone a voice first.")
            }
            ttsPrint("Using voice profile: \(profile.name)")

            metricsStore?.reportActivity(
                model: "TTS", kind: .tts, speed: 0, unit: "×RT")
            defer { metricsStore?.clearActivity(forModel: "TTS") }

            if let qwenDir = lock.withLock({ qwen3ModelDir }) {
                ttsPrint("Using Qwen3-TTS Base for clone synthesis (\(qwenDir.lastPathComponent))")
                guard let wavURL = VoiceProfileManager.shared.refAudioURL(for: profile) else {
                    throw NovaMLXError.apiError("Failed to load voice profile audio")
                }
                let wavData = try await Qwen3TTSCloneService.synthesize(
                    text: text,
                    refAudio: wavURL,
                    refText: profile.refTranscript,
                    modelDir: qwenDir
                )
                ttsPrint("Generated WAV: \(wavData.count) bytes")
                return wavData
            }

            let pipe = lock.withLock { pipeline }
            guard let pipe else {
                throw NovaMLXError.apiError("No neural TTS model loaded. Load a model or switch to System TTS.")
            }

            ttsPrint("Using DotsTTS for synthesis")
            guard let refAudio = VoiceProfileManager.shared.loadRefAudio(for: profile) else {
                ttsLog.error("[TTS] Failed to load reference audio for profile \(profile.name)")
                throw NovaMLXError.apiError("Failed to load voice profile audio")
            }

            let params = DotsTTSPipeline.Params()
            let audio = pipe.generate(
                targetText: text,
                refAudio48k: refAudio,
                refTranscript: profile.refTranscript,
                params: params
            )

            let wavData = try Self.mlxArrayToWAVData(audio, sampleRate: 48000)
            ttsPrint("Generated WAV: \(wavData.count) bytes")
            return wavData

        case .system:
            ttsLog.info("[TTS] Using macOS system TTS (voice=\(voice))")
            return try synthesizeWithMacOS(text: text, voice: voice, rate: rate)
        }
    }

    // MARK: - Private

    private static func mlxArrayToWAVData(_ audio: MLXArray, sampleRate: Int) throws -> Data {
        let samples = audio.asArray(Float.self)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tts_out_\(UUID().uuidString).wav")
        try AudioUtils.writeWavFile(samples: samples, sampleRate: sampleRate, fileURL: tempURL)
        let data = try Data(contentsOf: tempURL)
        try? FileManager.default.removeItem(at: tempURL)
        return data
    }

    private func findDefaultVoiceProfile() -> VoiceProfile? {
        let profiles = VoiceProfileManager.shared.listProfiles()
        return profiles.first
    }

    private func synthesizeWithMacOS(text: String, voice: String, rate: Int) throws -> Data {
        guard !text.isEmpty else {
            throw NovaMLXError.apiError("Text cannot be empty")
        }

        let tempFile = "/tmp/tts_\(UUID().uuidString).aiff"
        let url = URL(fileURLWithPath: tempFile)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-v", voice, "-r", String(rate), "-o", tempFile, text]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw NovaMLXError.apiError("TTS synthesis failed with exit code \(process.terminationStatus)")
        }
        guard FileManager.default.fileExists(atPath: tempFile) else {
            throw NovaMLXError.apiError("TTS failed to generate audio file")
        }

        let audioData = try Data(contentsOf: url)
        try? FileManager.default.removeItem(at: url)
        return audioData
    }
}
