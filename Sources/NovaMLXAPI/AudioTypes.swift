import Foundation

struct TranscriptionRequest: Codable, Sendable {
    let model: String
    let file: String        // base64-encoded audio data
    let language: String?
    let responseFormat: String?
    let temperature: Double?
    let stream: Bool?

    enum CodingKeys: String, CodingKey {
        case model, file, language, temperature, stream
        case responseFormat = "response_format"
    }

    var resolvedResponseFormat: String {
        responseFormat ?? "json"
    }
}

struct TranscriptionResponse: Codable, Sendable {
    let text: String
    let language: String?
    let duration: Double?
}

// MARK: - Text-to-Speech Types

struct TTSRequest: Codable, Sendable {
    let model: String
    let input: String
    let voice: String?
    let responseFormat: String?
    let speed: Float?
    let refAudio: String?
    let refTranscript: String?
    let numSteps: Int?
    let guidance: Float?
    let speakerScale: Float?

    enum CodingKeys: String, CodingKey {
        case model, input, voice, speed
        case responseFormat = "response_format"
        case refAudio = "ref_audio"
        case refTranscript = "ref_transcript"
        case numSteps = "num_steps"
        case guidance, speakerScale
    }

    var resolvedResponseFormat: String {
        responseFormat ?? "wav"
    }
}

struct TTSResponse: Codable, Sendable {
    // For streaming responses
}
