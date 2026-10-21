import Foundation
import NovaMLXEngine

struct ImageGenerationRequest: Codable, Sendable {
    let prompt: String
    let model: String
    let n: Int?
    let size: String?
    let responseFormat: String?
    let quality: String?
    let style: String?
    let seed: Int?
    let steps: Int?
    let negativePrompt: String?

    enum CodingKeys: String, CodingKey {
        case prompt, model, n, size, quality, style, seed, steps
        case responseFormat = "response_format"
        case negativePrompt = "negative_prompt"
    }

    var resolvedN: Int { min(max(n ?? 1, 1), 4) }
    var resolvedSize: (width: Int, height: Int) { novaImageSize(size) }
    var resolvedResponseFormat: String { responseFormat ?? "b64_json" }
}

func novaImageSize(_ size: String?) -> (width: Int, height: Int) {
    let raw = size ?? "1024x1024"
    switch raw {
    case "256x256": return (256, 256)
    case "512x512": return (512, 512)
    case "1024x1024": return (1024, 1024)
    case "768x1344": return (768, 1344)
    case "1344x768": return (1344, 768)
    case "864x1152": return (864, 1152)
    case "1152x864": return (1152, 864)
    case "1920x1080": return (1920, 1080)
    default:
        let parts = raw.lowercased().split(separator: "x", omittingEmptySubsequences: false)
        if parts.count == 2, let width = Int(parts[0]), let height = Int(parts[1]),
           (256...2816).contains(width), (256...2816).contains(height),
           width.isMultiple(of: 16), height.isMultiple(of: 16)
        {
            return (width, height)
        }
        return (1024, 1024)
    }
}

/// Billing fields for one image call. Pixel size is the first rendered image,
/// not the request `size` string. Token counts are intentionally absent.
struct ImageUsage: Codable, Sendable {
    let images: Int
    let operation: String
    let width: Int
    let height: Int
    let pixels: Int
    let sizeTier: String
    let steps: Int
    let durationMs: Int
    let model: String

    enum CodingKeys: String, CodingKey {
        case images, operation, width, height, pixels, steps, model
        case sizeTier = "size_tier"
        case durationMs = "duration_ms"
    }

    static func billed(
        images: Int,
        operation: String,
        width: Int,
        height: Int,
        steps: Int,
        durationMs: Int,
        model: String
    ) -> ImageUsage {
        ImageUsage(
            images: images,
            operation: operation,
            width: width,
            height: height,
            pixels: width * height,
            sizeTier: "\(width)x\(height)",
            steps: steps,
            durationMs: durationMs,
            model: model
        )
    }

    static func billed(from result: ImageGenerationResult, images: Int) -> ImageUsage {
        billed(
            images: images,
            operation: result.operation,
            width: result.width,
            height: result.height,
            steps: result.steps,
            durationMs: result.durationMs,
            model: result.model
        )
    }
}

struct ImageGenerationResponse: Codable, Sendable {
    let created: Int
    let data: [ImageData]
    let model: String
    let usage: ImageUsage
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
    var resolvedSize: (width: Int, height: Int) { novaImageSize(size) }
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
    var resolvedSize: (width: Int, height: Int) { novaImageSize(size) }
    var resolvedResponseFormat: String { responseFormat ?? "b64_json" }
}
