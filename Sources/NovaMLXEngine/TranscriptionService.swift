import AVFoundation
import Foundation
import MLX
import NovaMLXAudio
import NovaMLXCore
import NovaMLXUtils

public enum AudioModel: @unchecked Sendable {
    case qwen3ASR(Qwen3ASRModel)
    case whisper(WhisperModel)
}

public final class TranscriptionContainer: @unchecked Sendable {
    public let identifier: ModelIdentifier
    public let config: ModelConfig
    public private(set) var isLoaded: Bool
    public var model: AudioModel?

    public init(identifier: ModelIdentifier, config: ModelConfig) {
        self.identifier = identifier
        self.config = config
        self.isLoaded = false
    }

    public func setLoaded(model: AudioModel) {
        self.model = model
        isLoaded = true
        NovaMLXLog.info("Audio model loaded: \(identifier.displayName)")
    }

    public func unload() {
        model = nil
        isLoaded = false
        MLX.Memory.clearCache()
        NovaMLXLog.info("Audio model unloaded: \(identifier.displayName)")
    }
}

public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String?
    public let duration: Double?

    public init(text: String, language: String? = nil, duration: Double? = nil) {
        self.text = text
        self.language = language
        self.duration = duration
    }
}

public final class TranscriptionService: @unchecked Sendable {
    private var containers: [String: TranscriptionContainer] = [:]
    private let lock = NovaMLXLock()

    /// Shared metrics store. Set by InferenceService after construction so the
    /// status panel can show live ASR activity.
    public var metricsStore: MetricsStore?

    public init() {
        self.containers = [:]
    }

    public func loadModel(from url: URL, config: ModelConfig, progress: (@Sendable (LoadPhase) -> Void)? = nil) async throws -> TranscriptionContainer {
        let container = TranscriptionContainer(
            identifier: config.identifier,
            config: config
        )
        NovaMLXLog.info("Loading audio model from: \(url.path)")

        let family = config.identifier.family
        if family == .qwen3Tts {
            // TTS handled by TTSService, not TranscriptionService
            throw NovaMLXError.apiError("TTS models should be loaded via TTSService, not TranscriptionService.")
        } else if family == .whisper {
            let model = try await WhisperModel.fromModelDirectory(url)
            container.setLoaded(model: .whisper(model))
        } else if family == .qwen3Asr {
            let model = try await Qwen3ASRModel.fromModelDirectory(url)
            container.setLoaded(model: .qwen3ASR(model))
        } else {
            throw NovaMLXError.apiError("Unknown audio model family: \(family.rawValue). Supported: whisper, qwen3Asr.")
        }

        lock.withLock {
            containers[config.identifier.id] = container
        }

        return container
    }

    public func unload(modelId: String) {
        lock.withLock {
            guard let container = containers.removeValue(forKey: modelId) else { return }
            container.unload()
        }
    }

    public func isLoaded(_ modelId: String) -> Bool {
        lock.withLock {
            containers[modelId]?.isLoaded ?? false
        }
    }

    public func listLoadedModels() -> [String] {
        lock.withLock {
            containers.filter { $0.value.isLoaded }.map { $0.key }
        }
    }

    public func transcribe(modelId: String, audioData: Data, language: String? = nil, responseFormat: String = "json") async throws -> TranscriptionResult {
        guard let container = lock.withLock({ containers[modelId] }),
              container.isLoaded,
              let model = container.model else {
            throw NovaMLXError.modelNotFound(modelId)
        }

        // Decode audio bytes to 16kHz mono PCM via AVFoundation temp file
        let audioURL = try writeTempAudio(audioData)
        defer { try? FileManager.default.removeItem(at: audioURL) }

        let (_, audioArray) = try loadAudioArray(from: audioURL, sampleRate: 16000)

        let startTime = Date()
        // Report live activity so the status panel shows ASR is running.
        metricsStore?.reportActivity(model: modelId, kind: .asr, speed: 0, unit: "×RT")
        defer { metricsStore?.clearActivity(forModel: modelId) }

        let output: STTOutput
        switch model {
        case .qwen3ASR(let asrModel):
            output = asrModel.generate(
                audio: audioArray,
                maxTokens: 8192,
                temperature: 0.0,
                language: language
            )
        case .whisper(let whisperModel):
            // Whisper needs mel spectrogram (n_fft=400, hop=160)
            let mel = computeMelSpectrogram(audio: audioArray, sampleRate: 16000, nFft: 400, hopLength: 160, nMels: whisperModel.dims.nMels)
            // Pad or trim to 3000 mel frames (30 seconds)
            let padded = Self.padOrTrim2D(mel, length: 3000)
            // MLX Conv1d uses NLC: [batch, length, channels] = [1, 3000, nMels]
            let result = whisperModel.generate(mel: padded.reshaped([1, padded.dim(0), padded.dim(1)]), language: language, temperature: 0.0)
            output = STTOutput(
                text: result.text,
                language: result.language,
                generationTokens: result.tokens.count
            )
        }
        let elapsed = Date().timeIntervalSince(startTime)

        return TranscriptionResult(
            text: output.text,
            language: output.language,
            duration: elapsed
        )
    }

    public func transcribeStream(modelId: String, audioData: Data, language: String? = nil) -> AsyncThrowingStream<String, Error> {
        guard let container = lock.withLock({ containers[modelId] }),
              container.isLoaded,
              let model = container.model else {
            return AsyncThrowingStream { $0.finish(throwing: NovaMLXError.modelNotFound(modelId)) }
        }

        let sendableSelf = self
        let sendableModel = SendableBox(model)
        let sendableLanguage = language

        return AsyncThrowingStream { continuation in
            let task = Task.detached {
                let model = sendableModel.value
                do {
                    let audioURL = try sendableSelf.writeTempAudio(audioData)
                    defer { try? FileManager.default.removeItem(at: audioURL) }
                    let (_, audioArray) = try loadAudioArray(from: audioURL, sampleRate: 16000)

                    let sttStream: AsyncThrowingStream<STTGeneration, Error>
                    switch model {
                    case .qwen3ASR(let asrModel):
                        sttStream = asrModel.generateStream(
                            audio: audioArray,
                            maxTokens: 8192,
                            temperature: 0.0,
                            language: sendableLanguage
                        )
                    case .whisper(let whisperModel):
                        let mel = computeMelSpectrogram(audio: audioArray, sampleRate: 16000, nFft: 400, hopLength: 160, nMels: whisperModel.dims.nMels)
                        let padded = TranscriptionService.padOrTrim2D(mel, length: 3000)
                        sttStream = whisperModel.generateStream(mel: padded.reshaped([1, padded.dim(0), padded.dim(1)]), language: sendableLanguage, temperature: 0.0)
                    }

                    for try await event in sttStream {
                        guard !Task.isCancelled else { break }
                        switch event {
                        case .token(let text):
                            continuation.yield(text)
                        case .info, .result:
                            break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Audio Decoding

    private func writeTempAudio(_ data: Data) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("novamlx_audio_\(UUID().uuidString).wav")

        // Try decoding as WAV/CAF/AIFF first (native formats)
        // If the data is already a valid audio file, write it directly
        if let ext = guessAudioExtension(data) {
            let url = tempDir.appendingPathComponent("novamlx_audio_\(UUID().uuidString).\(ext)")
            try data.write(to: url)
            // Verify it's readable
            if let _ = try? AVAudioFile(forReading: url) {
                return url
            }
            try? FileManager.default.removeItem(at: url)
        }

        // Fallback: write raw data and let AVFoundation try to parse it
        try data.write(to: tempFile)
        if let _ = try? AVAudioFile(forReading: tempFile) {
            return tempFile
        }
        try? FileManager.default.removeItem(at: tempFile)

        // Last resort: treat as raw PCM and wrap in WAV
        return try writeRawPCMAsWAV(data)
    }

    private func guessAudioExtension(_ data: Data) -> String? {
        guard data.count >= 4 else { return nil }
        let header = [UInt8](data.prefix(4))

        // WAV: RIFF....WAVE
        if header[0] == 0x52, header[1] == 0x49, header[2] == 0x46, header[3] == 0x46 {
            return "wav"
        }
        // FLAC: fLaC
        if header[0] == 0x66, header[1] == 0x4C, header[2] == 0x61, header[3] == 0x43 {
            return "flac"
        }
        // MP3: ID3 or 0xFF 0xFB / 0xFF 0xF3 / 0xFF 0xF2
        if header[0] == 0x49, header[1] == 0x44, header[2] == 0x33 {
            return "mp3"
        }
        if header[0] == 0xFF, (header[1] & 0xE0) == 0xE0 {
            return "mp3"
        }
        // OGG: OggS
        if header[0] == 0x4F, header[1] == 0x67, header[2] == 0x67, header[3] == 0x53 {
            return "ogg"
        }
        // CAF: caff
        if header[0] == 0x63, header[1] == 0x61, header[2] == 0x66, header[3] == 0x66 {
            return "caf"
        }
        // AIFF: FORM....AIFF
        if header[0] == 0x46, header[1] == 0x4F, header[2] == 0x52, header[3] == 0x4D {
            return "aiff"
        }
        return nil
    }

    private func writeRawPCMAsWAV(_ pcmData: Data) throws -> URL {
        // Assume 16kHz mono 16-bit PCM
        let sampleRate: Double = 16000
        let channels: UInt32 = 1
        let bitsPerChannel: UInt32 = 16
        let bytesPerFrame = channels * (bitsPerChannel / 8)
        let frameCount = UInt32(pcmData.count) / bytesPerFrame

        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("novamlx_raw_\(UUID().uuidString).wav")

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: true
        ) else {
            throw NovaMLXError.apiError("Failed to create audio format for raw PCM")
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
            throw NovaMLXError.apiError("Failed to create audio buffer for raw PCM")
        }

        buffer.frameLength = AVAudioFrameCount(frameCount)
        pcmData.withUnsafeBytes { rawBuffer in
            if let base = rawBuffer.baseAddress {
                memcpy(buffer.mutableAudioBufferList.pointee.mBuffers.mData!, base, pcmData.count)
            }
        }

        let audioFile = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: format.commonFormat,
            interleaved: format.isInterleaved
        )
        try audioFile.write(from: buffer)

        return url
    }

    private static func padOrTrim2D(_ array: MLXArray, length: Int) -> MLXArray {
        let currentLen = array.dim(0)
        if currentLen > length {
            return array[0..<length, 0...]
        } else if currentLen < length {
            let padSize = length - currentLen
            let padding = MLXArray.zeros([padSize, array.dim(1)])
            return concatenated([array, padding], axis: 0)
        }
        return array
    }
}
