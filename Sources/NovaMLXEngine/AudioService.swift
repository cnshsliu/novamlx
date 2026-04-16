import AVFoundation
import Foundation
import Speech
import NovaMLXCore
import NovaMLXUtils

public final class AudioService: @unchecked Sendable {
    private let lock = NovaMLXLock()
    private var synthesizer: AVSpeechSynthesizer?

    public struct TranscriptionResult: Codable, Sendable {
        public let text: String
        public let language: String?
        public let duration: Double?
        public let segments: [Segment]?

        public struct Segment: Codable, Sendable {
            public let text: String
            public let start: Double
            public let end: Double
        }
    }

    public struct SynthesisRequest: Codable, Sendable {
        public let text: String
        public let voice: String?
        public let speed: Double?
        public let language: String?

        public init(text: String, voice: String? = nil, speed: Double? = nil, language: String? = nil) {
            self.text = text
            self.voice = voice
            self.speed = speed
            self.language = language
        }
    }

    public struct AudioChunk: Sendable {
        public let data: Data
        public let isFinal: Bool
    }

    public init() {}

    public func transcribe(audioData: Data, language: String? = nil) async throws -> TranscriptionResult {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("novamlx_audio_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempFile) }

        try audioData.write(to: tempFile)

        let audioURL = tempFile
        let supportedLocales = SFSpeechRecognizer.supportedLocales()

        let locale: Locale
        if let lang = language {
            locale = Locale(identifier: lang)
        } else {
            locale = try await detectLocale(audioURL: audioURL, supportedLocales: supportedLocales)
        }

        guard supportedLocales.contains(locale) else {
            throw NovaMLXError.inferenceFailed("Language \(locale.identifier) not supported for transcription")
        }

        guard let recognizer = SFSpeechRecognizer(locale: locale) else {
            throw NovaMLXError.inferenceFailed("Could not create speech recognizer for \(locale.identifier)")
        }

        let request = SFSpeechURLRecognitionRequest(url: audioURL)
        request.requiresOnDeviceRecognition = true

        var segments: [TranscriptionResult.Segment] = []
        var fullText = ""

        let result = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<SFSpeechRecognitionResult?, Error>) in
            recognizer.recognitionTask(with: request) { recognitionResult, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                if let recognitionResult, recognitionResult.isFinal {
                    continuation.resume(returning: SendableBox(recognitionResult).value)
                } else if recognitionResult == nil {
                    continuation.resume(returning: nil)
                }
            }
        }

        if let result {
            fullText = result.bestTranscription.formattedString
            for segment in result.bestTranscription.segments {
                segments.append(TranscriptionResult.Segment(
                    text: segment.substring,
                    start: segment.timestamp,
                    end: segment.timestamp + segment.duration
                ))
            }
        }

        let asset = AVURLAsset(url: audioURL)
        let duration = try await asset.load(.duration).seconds

        return TranscriptionResult(
            text: fullText,
            language: locale.language.languageCode?.identifier,
            duration: duration > 0 ? duration : nil,
            segments: segments.isEmpty ? nil : segments
        )
    }

    private func detectLocale(audioURL: URL, supportedLocales: Set<Locale>) async throws -> Locale {
        let priorityLocales: [String] = [
            "en-US", "zh-CN", "es-ES", "hi-IN", "ar-SA",
            "fr-FR", "de-DE", "ja-JP", "pt-BR", "ru-RU",
            "ko-KR", "it-IT", "nl-NL", "pl-PL", "tr-TR",
            "th-TH", "vi-VN", "uk-UA", "id-ID", "ms-MY",
        ]

        let validLocales = priorityLocales.compactMap { localeId -> Locale? in
            let locale = Locale(identifier: localeId)
            guard supportedLocales.contains(locale),
                  SFSpeechRecognizer(locale: locale) != nil
            else { return nil }
            return locale
        }

        guard !validLocales.isEmpty else { return Locale(identifier: "en-US") }

        let bestLocaleBox = UnsafeMutablePointer<(locale: Locale, confidence: Double)>.allocate(capacity: 1)
        bestLocaleBox.initialize(to: (validLocales[0], 0.0))
        defer { bestLocaleBox.deallocate() }

        await withTaskGroup(of: (Int, Double).self) { group in
            for (index, locale) in validLocales.enumerated() {
                let box = SendableBox(audioURL)
                group.addTask { @Sendable in
                    guard let recognizer = SFSpeechRecognizer(locale: locale) else {
                        return (index, 0.0)
                    }
                    let request = SFSpeechURLRecognitionRequest(url: box.value)
                    request.requiresOnDeviceRecognition = true
                    do {
                        let confidence = try await Self.measureRecognitionConfidenceStatic(recognizer: recognizer, request: request)
                        return (index, confidence)
                    } catch {
                        return (index, 0.0)
                    }
                }
            }

            for await (index, confidence) in group {
                let current = bestLocaleBox.pointee
                if confidence > current.confidence {
                    bestLocaleBox.pointee = (validLocales[index], confidence)
                }
                if bestLocaleBox.pointee.confidence > 0.8 {
                    group.cancelAll()
                }
            }
        }

        return bestLocaleBox.pointee.locale
    }

    private static func measureRecognitionConfidenceStatic(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechURLRecognitionRequest
    ) async throws -> Double {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Double, Error>) in
            recognizer.recognitionTask(with: request) { result, _ in
                if result == nil {
                    continuation.resume(returning: 0.0)
                    return
                }
                if let result, result.isFinal {
                    let segments = result.bestTranscription.segments
                    guard !segments.isEmpty else {
                        continuation.resume(returning: 0.0)
                        return
                    }
                    let avgConfidence = segments.map { Double($0.confidence) }.reduce(0, +) / Double(segments.count)
                    continuation.resume(returning: avgConfidence)
                }
            }
        }
    }

    private func measureRecognitionConfidence(
        recognizer: SFSpeechRecognizer,
        request: SFSpeechURLRecognitionRequest
    ) async throws -> Double {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Double, Error>) in
            recognizer.recognitionTask(with: request) { result, _ in
                if result == nil {
                    continuation.resume(returning: 0.0)
                    return
                }
                if let result, result.isFinal {
                    let segments = result.bestTranscription.segments
                    guard !segments.isEmpty else {
                        continuation.resume(returning: 0.0)
                        return
                    }
                    let avgConfidence = segments.map { Double($0.confidence) }.reduce(0, +) / Double(segments.count)
                    continuation.resume(returning: avgConfidence)
                } else if result == nil {
                    continuation.resume(returning: 0.0)
                }
            }
        }
    }

    public func synthesize(request: SynthesisRequest) async throws -> Data {
        let buffer = try await synthesizeToBuffer(request: request)
        return try pcmBufferToWav(buffer)
    }

    public func synthesizeStream(request: SynthesisRequest) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let utterance = AVSpeechUtterance(attributedString: NSAttributedString(string: request.text))

                    if let voiceId = request.voice {
                        let voices = AVSpeechSynthesisVoice.speechVoices()
                        if let voice = voices.first(where: { $0.identifier == voiceId || $0.name.lowercased() == voiceId.lowercased() }) {
                            utterance.voice = voice
                        }
                    } else if let lang = request.language {
                        utterance.voice = AVSpeechSynthesisVoice(language: lang)
                    }

                    if let speed = request.speed {
                        utterance.rate = AVSpeechUtteranceMinimumSpeechRate + Float(speed) * (AVSpeechUtteranceMaximumSpeechRate - AVSpeechUtteranceMinimumSpeechRate)
                    }

                    let wavHeader = try Self.buildWavHeader(sampleRate: 22050, channels: 1, bitsPerSample: 16, dataChunkSize: 0)
                    continuation.yield(AudioChunk(data: wavHeader, isFinal: false))

                    let buffers = try await Self.collectBuffers(utterance: utterance)

                    for (index, buffer) in buffers.enumerated() {
                        let pcmData = Self.pcmBufferToRawData(buffer)
                        let isFinal = index == buffers.count - 1
                        continuation.yield(AudioChunk(data: pcmData, isFinal: isFinal))
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    private static func collectBuffers(utterance: AVSpeechUtterance) async throws -> [AVAudioPCMBuffer] {
        final class BufferCollector: @unchecked Sendable {
            private var buffers: [AVAudioPCMBuffer] = []
            func append(_ buffer: AVAudioPCMBuffer) { buffers.append(buffer) }
            func collect() -> [AVAudioPCMBuffer] { let b = buffers; buffers = []; return b }
        }

        let collector = BufferCollector()

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[AVAudioPCMBuffer], Error>) in
            let synth = AVSpeechSynthesizer()
            synth.write(utterance) { audioBuffer in
                if let buf = audioBuffer as? AVAudioPCMBuffer, buf.frameLength > 0 {
                    collector.append(SendableBox(buf).value)
                } else if let buf = audioBuffer as? AVAudioPCMBuffer, buf.frameLength == 0 {
                    continuation.resume(returning: collector.collect())
                }
            }
        }
    }

    public func synthesizeToBuffer(request: SynthesisRequest) async throws -> AVAudioPCMBuffer {
        let utterance = AVSpeechUtterance(attributedString: NSAttributedString(string: request.text))

        if let voiceId = request.voice {
            let voices = AVSpeechSynthesisVoice.speechVoices()
            if let voice = voices.first(where: { $0.identifier == voiceId || $0.name.lowercased() == voiceId.lowercased() }) {
                utterance.voice = voice
            }
        } else if let lang = request.language {
            utterance.voice = AVSpeechSynthesisVoice(language: lang)
        }

        if let speed = request.speed {
            utterance.rate = AVSpeechUtteranceMinimumSpeechRate + Float(speed) * (AVSpeechUtteranceMaximumSpeechRate - AVSpeechUtteranceMinimumSpeechRate)
        }

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<AVAudioPCMBuffer, Error>) in
            lock.withLock {
                let synth = AVSpeechSynthesizer()
                self.synthesizer = synth

                synth.write(utterance) { audioBuffer in
                    if let buf = audioBuffer as? AVAudioPCMBuffer {
                        continuation.resume(returning: SendableBox(buf).value)
                    } else {
                        continuation.resume(throwing: NovaMLXError.inferenceFailed("TTS synthesis failed"))
                    }
                }
            }
        }
    }

    private static func buildWavHeader(sampleRate: Int, channels: Int, bitsPerSample: Int, dataChunkSize: UInt32) throws -> Data {
        var header = Data()
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = UInt16(channels * bitsPerSample / 8)

        header.append(contentsOf: [UInt8]("RIFF".utf8))
        header.append(contentsOf: withUnsafeBytes(of: UInt32(36 + dataChunkSize).littleEndian) { Array($0) })
        header.append(contentsOf: [UInt8]("WAVE".utf8))
        header.append(contentsOf: [UInt8]("fmt ".utf8))
        header.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: UInt16(channels).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: blockAlign.littleEndian) { Array($0) })
        header.append(contentsOf: withUnsafeBytes(of: UInt16(bitsPerSample).littleEndian) { Array($0) })
        header.append(contentsOf: [UInt8]("data".utf8))
        header.append(contentsOf: withUnsafeBytes(of: dataChunkSize.littleEndian) { Array($0) })

        return header
    }

    private static func pcmBufferToRawData(_ buffer: AVAudioPCMBuffer) -> Data {
        let frameCount = buffer.frameLength
        guard frameCount > 0, let channelData = buffer.floatChannelData else { return Data() }

        var data = Data()
        data.reserveCapacity(Int(frameCount) * Int(buffer.format.channelCount) * 2)

        for frame in 0..<Int(frameCount) {
            for channel in 0..<Int(buffer.format.channelCount) {
                let sample = channelData[channel][frame]
                let intSample = Int16(max(-32768, min(32767, sample * 32767)))
                data.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
            }
        }
        return data
    }

    private func pcmBufferToWav(_ buffer: AVAudioPCMBuffer) throws -> Data {
        let format = buffer.format
        let frameCount = buffer.frameLength

        let bytesPerFrame = UInt32(format.streamDescription.pointee.mBytesPerFrame)
        let dataSize = frameCount * bytesPerFrame

        let header = try Self.buildWavHeader(
            sampleRate: Int(format.sampleRate),
            channels: Int(format.channelCount),
            bitsPerSample: Int(format.streamDescription.pointee.mBitsPerChannel),
            dataChunkSize: dataSize
        )

        var wav = header

        if let channelData = buffer.floatChannelData {
            for frame in 0..<Int(frameCount) {
                for channel in 0..<Int(format.channelCount) {
                    let sample = channelData[channel][frame]
                    let intSample = Int16(max(-32768, min(32767, sample * 32767)))
                    wav.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
                }
            }
        }

        return wav
    }

    public static func supportedVoices() -> [[String: String]] {
        AVSpeechSynthesisVoice.speechVoices().map { voice in
            [
                "identifier": voice.identifier,
                "name": voice.name,
                "language": voice.language,
                "quality": voice.quality == .enhanced ? "enhanced" : voice.quality == .premium ? "premium" : "default",
            ]
        }
    }

    public static func supportedLanguages() -> [String] {
        Array(Set(AVSpeechSynthesisVoice.speechVoices().map(\.language))).sorted()
    }
}
