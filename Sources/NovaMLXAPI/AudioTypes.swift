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
