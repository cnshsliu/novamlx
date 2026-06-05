import Foundation
import os.log
import NovaMLXCore
import NovaMLXUtils

private let ttsLog = Logger(subsystem: "com.novamlx", category: "TTS")

/// Text-to-Speech Service
/// Currently uses macOS built-in 'say' command
/// TODO: Add Qwen3-TTS MLX model support
public final class TTSService: @unchecked Sendable {
    private let lock = NSLock()

    public init() {}

    /// Synthesize speech from text
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - voice: Voice name (e.g., "Tingting" for Chinese, "Alex" for English)
    ///   - rate: Speech rate (default: 175, range: 50-400)
    /// - Returns: Audio data (WAV format)
    public func synthesize(
        text: String,
        voice: String = "Tingting",
        rate: Int = 175
    ) async throws -> Data {
        guard !text.isEmpty else {
            throw NovaMLXError.apiError("Text cannot be empty")
        }

        let tempFile = "/tmp/tts_\(UUID().uuidString).aiff"
        let url = URL(fileURLWithPath: tempFile)

        // Use macOS say command
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = [
            "-v", voice,
            "-r", String(rate),
            "-o", tempFile,
            text
        ]

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()

            guard process.terminationStatus == 0 else {
                throw NovaMLXError.apiError("TTS synthesis failed with exit code \(process.terminationStatus)")
            }

            guard FileManager.default.fileExists(atPath: tempFile) else {
                throw NovaMLXError.apiError("TTS failed to generate audio file")
            }

            let audioData = try Data(contentsOf: url)

            // Cleanup
            try? FileManager.default.removeItem(at: url)

            ttsLog.info("TTS synthesis completed: \(text.count) chars, \(audioData.count) bytes")

            return audioData
        } catch {
            // Cleanup on error
            try? FileManager.default.removeItem(at: url)
            throw error
        }
    }

    /// List available voices
    public func listVoices() -> [(name: String, language: String)] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-v", "?"]

        let pipe = Pipe()
        process.standardOutput = pipe

        do {
            try process.run()
            process.waitUntilExit()

            let outputData = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: outputData, encoding: .utf8) ?? ""

            // Parse voice list
            var voices: [(name: String, language: String)] = []
            let lines = output.split(separator: "\n")

            for line in lines {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }

                // Format: "Name (Language) langcode # Sample"
                // Example: "Tingting (Chinese (China mainland)) zh_CN    # 你好！我叫婷婷。"
                if let parenIndex = trimmed.firstIndex(of: "("),
                   let langStart = trimmed[parenIndex...].firstIndex(of: " ") {
                    let name = String(trimmed[..<parenIndex]).trimmingCharacters(in: .whitespaces)
                    let langEnd = trimmed[langStart...].firstIndex(of: ")") ?? trimmed.endIndex
                    let langCode = trimmed[trimmed.index(after: langStart)..<langEnd]
                        .trimmingCharacters(in: .whitespaces)
                    voices.append((name: name, language: String(langCode)))
                }
            }

            return voices
        } catch {
            ttsLog.error("Failed to list voices: \(error)")
            return []
        }
    }
}

/// TTS Synthesis Result
public struct TTSResult {
    public let audioData: Data
    public let format: String
    public let voice: String

    public init(audioData: Data, format: String = "aiff", voice: String) {
        self.audioData = audioData
        self.format = format
        self.voice = voice
    }
}
