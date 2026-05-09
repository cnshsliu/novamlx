import Foundation

struct ImageGenerationRequest: Codable, Sendable {
    let prompt: String
    let model: String
    let n: Int?
    let size: String?
    let responseFormat: String?
    let quality: String?
    let style: String?
    let seed: Int?
    let negativePrompt: String?

    enum CodingKeys: String, CodingKey {
        case prompt, model, n, size, quality, style, seed
        case responseFormat = "response_format"
        case negativePrompt = "negative_prompt"
    }

    var resolvedN: Int { min(max(n ?? 1, 1), 4) }
    var resolvedSize: (width: Int, height: Int) {
        switch size ?? "1024x1024" {
        case "256x256": return (256, 256)
        case "512x512": return (512, 512)
        case "1024x1024": return (1024, 1024)
        default: return (1024, 1024)
        }
    }
    var resolvedResponseFormat: String { responseFormat ?? "b64_json" }
}

struct ImageGenerationResponse: Codable, Sendable {
    let created: Int
    let data: [ImageData]
    let model: String
}

struct ImageData: Codable, Sendable {
    let b64Json: String?
    let url: String?
    let revisedPrompt: String?

    enum CodingKeys: String, CodingKey {
        case b64Json = "b64_json"
        case url
        case revisedPrompt = "revised_prompt"
    }
}

// MARK: - Image Edit

/// OpenAI-compatible request type for POST /v1/images/edits.
/// Parsed from multipart/form-data fields.
struct ImageEditRequest {
    let image: Data
    let mask: Data?
    let prompt: String
    let model: String
    let n: Int?
    let size: String?
    let responseFormat: String?

    var resolvedN: Int { min(max(n ?? 1, 1), 4) }
    var resolvedSize: (width: Int, height: Int) {
        switch size ?? "1024x1024" {
        case "256x256": return (256, 256)
        case "512x512": return (512, 512)
        case "1024x1024": return (1024, 1024)
        default: return (1024, 1024)
        }
    }
    var resolvedResponseFormat: String { responseFormat ?? "b64_json" }
}

// MARK: - Image Variation

/// OpenAI-compatible request type for POST /v1/images/variations.
/// Parsed from multipart/form-data fields.
struct ImageVariationRequest {
    let image: Data
    let model: String
    let n: Int?
    let size: String?
    let responseFormat: String?

    var resolvedN: Int { min(max(n ?? 1, 1), 4) }
    var resolvedSize: (width: Int, height: Int) {
        switch size ?? "1024x1024" {
        case "256x256": return (256, 256)
        case "512x512": return (512, 512)
        case "1024x1024": return (1024, 1024)
        default: return (1024, 1024)
        }
    }
    var resolvedResponseFormat: String { responseFormat ?? "b64_json" }
}
