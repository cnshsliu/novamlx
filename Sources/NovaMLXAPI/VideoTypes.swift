import Foundation
import NovaMLXEngine

// MARK: - OpenRouter-compatible async video generation types
//
// Wire shape mirrors OpenRouter's video API (POST /v1/videos → 202, poll
// GET /v1/videos/{id}, download GET /v1/videos/{id}/content). Deviations from
// the upstream spec, all surfaced honestly in the payloads:
// - usage.cost is always 0 and is_byok false (generation is local)
// - generate_audio is accepted but ignored (h3 always renders an AAC track)
// - provider.options is not forwarded to any vendor; the private `h3`
//   namespace carries our quality overrides (steps/layers/reuse)

struct VideoGenerationRequest: Codable, Sendable {
    let model: String
    let prompt: String
    let duration: Int?
    let resolution: String?
    let aspectRatio: String?
    let size: String?
    let frameImages: [VideoFrameImage]?
    let inputReferences: [VideoInputReference]?
    let generateAudio: Bool?
    let seed: Int?
    let callbackUrl: String?
    let provider: VideoProvider?

    enum CodingKeys: String, CodingKey {
        case model, prompt, duration, resolution, size, seed, provider
        case aspectRatio = "aspect_ratio"
        case frameImages = "frame_images"
        case inputReferences = "input_references"
        case generateAudio = "generate_audio"
        case callbackUrl = "callback_url"
    }

    /// Map the wire shape onto the engine-layer job request.
    func toJobRequest() -> H3VideoJobRequest {
        var firstFrame: String? = nil
        var lastFrame: String? = nil
        for frame in frameImages ?? [] {
            switch frame.frameType {
            case "first_frame": firstFrame = frame.imageUrl.url
            case "last_frame": lastFrame = frame.imageUrl.url
            default: break
            }
        }
        let h3 = provider?.options?.h3Overrides
        return H3VideoJobRequest(
            model: model,
            prompt: prompt,
            durationSeconds: duration,
            size: size,
            resolution: resolution,
            aspectRatio: aspectRatio,
            seed: seed,
            steps: h3?.steps,
            layers: h3?.layers,
            reuse: h3?.reuse,
            firstFrameURL: firstFrame,
            lastFrameURL: lastFrame,
            referenceImageURLs: (inputReferences ?? []).map { $0.imageUrl.url },
            callbackURL: callbackUrl
        )
    }
}

struct VideoFrameImage: Codable, Sendable {
    let frameType: String?
    let imageUrl: VideoImageURL

    enum CodingKeys: String, CodingKey {
        case frameType = "frame_type"
        case imageUrl = "image_url"
    }
}

struct VideoInputReference: Codable, Sendable {
    let imageUrl: VideoImageURL

    enum CodingKeys: String, CodingKey {
        case imageUrl = "image_url"
    }
}

struct VideoImageURL: Codable, Sendable {
    let url: String
}

struct VideoProvider: Codable, Sendable {
    let options: VideoProviderOptions?
}

/// `provider.options` — accepted for spec compatibility but never forwarded.
/// Only the private `h3` namespace is read, as a quality-preset override.
struct VideoProviderOptions: Codable, Sendable {
    let raw: [String: AnyCodable]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        raw = (try? container.decode([String: AnyCodable].self)) ?? [:]
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(raw)
    }

    var h3Overrides: (steps: Int?, layers: Int?, reuse: Int?) {
        guard case .dictionary(let h3) = raw["h3"] else { return (nil, nil, nil) }
        func intValue(_ key: String) -> Int? {
            if case .int(let v) = h3[key] { return v }
            return nil
        }
        return (intValue("steps"), intValue("layers"), intValue("reuse"))
    }
}

// MARK: - Responses

struct VideoUsage: Codable, Sendable, Equatable {
    let cost: Double
    let isByok: Bool

    enum CodingKeys: String, CodingKey {
        case cost
        case isByok = "is_byok"
    }
}

struct VideoJobResponse: Codable, Sendable {
    let id: String
    let generationId: String?
    let pollingUrl: String
    let status: String
    let model: String
    let width: Int
    let height: Int
    let duration: Int?
    let unsignedUrls: [String]?
    let usage: VideoUsage?
    let error: String?
    let createdAt: Int
    /// Live progress while the job runs (extension beyond the OpenRouter
    /// wire shape; nil unless in_progress).
    let progress: VideoJobProgress?

    enum CodingKeys: String, CodingKey {
        case id, status, model, width, height, duration, error, progress
        case generationId = "generation_id"
        case pollingUrl = "polling_url"
        case unsignedUrls = "unsigned_urls"
        case usage
        case createdAt = "created_at"
    }
}

struct VideoJobProgress: Codable, Sendable, Equatable {
    let phase: String
    let completed: Int
    let total: Int
    let percent: Int
}

extension VideoJobResponse {
    /// URLs are path-relative — clients resolve them against the same host
    /// they POSTed to (there is no configured public base URL).
    init(job: H3VideoJob, progress: H3VideoProgress? = nil) {
        id = job.id
        generationId = job.status == .completed ? job.id : nil
        pollingUrl = "/v1/videos/\(job.id)"
        status = job.status.rawValue
        model = job.model
        width = job.width
        height = job.height
        duration = job.durationSeconds
        unsignedUrls = job.status == .completed
            ? ["/v1/videos/\(job.id)/content?index=0"]
            : nil
        usage = job.status == .completed ? VideoUsage(cost: 0, isByok: false) : nil
        error = job.error
        createdAt = Int(job.createdAt.timeIntervalSince1970)
        self.progress = progress.map {
            VideoJobProgress(phase: $0.phase, completed: $0.completed, total: $0.total, percent: $0.percent)
        }
    }
}

struct VideoModelInfo: Codable, Sendable {
    let id: String
    let name: String
    let supportedDurations: [Int]
    let supportedResolutions: [String]
    let supportedAspectRatios: [String]
    let supportedSizes: [String]
    let supportsFrameImages: Bool
    let supportsInputReferences: Bool
    let supportsAudioGeneration: Bool

    enum CodingKeys: String, CodingKey {
        case id, name
        case supportedDurations = "supported_durations"
        case supportedResolutions = "supported_resolutions"
        case supportedAspectRatios = "supported_aspect_ratios"
        case supportedSizes = "supported_sizes"
        case supportsFrameImages = "supports_frame_images"
        case supportsInputReferences = "supports_input_references"
        case supportsAudioGeneration = "supports_audio_generation"
    }
}

struct VideoModelsResponse: Codable, Sendable {
    let data: [VideoModelInfo]

    /// Static capability table for the h3 backend.
    static let h3 = VideoModelsResponse(data: [
        VideoModelInfo(
            id: H3VideoService.primaryModelId,
            name: "MiniMax H3 (local h3.c)",
            supportedDurations: Array(1...H3VideoService.maxDurationSeconds),
            supportedResolutions: ["480p", "720p"],
            supportedAspectRatios: ["16:9", "9:16", "1:1"],
            supportedSizes: H3VideoService.supportedSizes.map { "\($0.width)x\($0.height)" },
            supportsFrameImages: true,
            supportsInputReferences: true,
            supportsAudioGeneration: true
        ),
        VideoModelInfo(
            id: "minimax/hailuo-3",
            name: "MiniMax Hailuo 3 (alias of minimax/h3, local h3.c)",
            supportedDurations: Array(1...H3VideoService.maxDurationSeconds),
            supportedResolutions: ["480p", "720p"],
            supportedAspectRatios: ["16:9", "9:16", "1:1"],
            supportedSizes: H3VideoService.supportedSizes.map { "\($0.width)x\($0.height)" },
            supportsFrameImages: true,
            supportsInputReferences: true,
            supportsAudioGeneration: true
        ),
    ])
}
