import Foundation
import NovaMLXCore
import NovaMLXEngine

// MARK: - Task-style 3D generation types (Hunyuan3D-2.1)
//
// Wire shape mirrors the media-task family: POST /v1/3d/generations → 202,
// poll GET /v1/3d/generations/{id} (or the shared GET /v1/tasks/{id} with
// task = "3d.generation"), download GET /v1/3d/generations/{id}/content.
// Deviations surfaced honestly in the payloads:
// - usage.cost is always 0 and is_byok false (generation is local), but the
//   metering fields (meshes, size_tier) are stable for upstream billing
// - image-to-3D only this round; `prompt` is accepted and ignored

struct ThreeDGenerationRequest: Codable, Sendable {
    let model: String
    let image: String
    let prompt: String?
    /// shape-only when false (default); shape + paint when true.
    let texture: Bool?
    /// Optional paint model id; defaults to the loaded paint slot.
    let paintModel: String?
    let seed: Int?
    let octreeResolution: Int?
    let numInferenceSteps: Int?
    let guidanceScale: Double?
    /// Only "glb" this round.
    let outputFormat: String?

    enum CodingKeys: String, CodingKey {
        case model, image, prompt, texture, seed
        case paintModel = "paint_model"
        case octreeResolution = "octree_resolution"
        case numInferenceSteps = "num_inference_steps"
        case guidanceScale = "guidance_scale"
        case outputFormat = "output_format"
    }

    init(
        model: String,
        image: String,
        prompt: String? = nil,
        texture: Bool? = nil,
        paintModel: String? = nil,
        seed: Int? = nil,
        octreeResolution: Int? = nil,
        numInferenceSteps: Int? = nil,
        guidanceScale: Double? = nil,
        outputFormat: String? = nil
    ) {
        self.model = model
        self.image = image
        self.prompt = prompt
        self.texture = texture
        self.paintModel = paintModel
        self.seed = seed
        self.octreeResolution = octreeResolution
        self.numInferenceSteps = numInferenceSteps
        self.guidanceScale = guidanceScale
        self.outputFormat = outputFormat
    }

    /// Resolve the `image` field to bytes. Accepts a data URI
    /// (`data:image/...;base64,...`), raw base64, an http(s) URL, or a local
    /// file path.
    func resolveImageData() async throws -> Data {
        if image.hasPrefix("data:") {
            guard let comma = image.firstIndex(of: ",") else {
                throw NovaMLXError.apiError("'image' data URI has no base64 payload")
            }
            let payload = String(image[image.index(after: comma)...])
            guard let data = Data(base64Encoded: payload) else {
                throw NovaMLXError.apiError("'image' data URI payload is not valid base64")
            }
            return data
        }
        if image.hasPrefix("http://") || image.hasPrefix("https://") {
            guard let url = URL(string: image) else {
                throw NovaMLXError.apiError("'image' URL is malformed")
            }
            let (data, response): (Data, URLResponse)
            do {
                (data, response) = try await URLSession.shared.data(from: url)
            } catch {
                throw NovaMLXError.apiError("failed to fetch 'image' URL: \(error.localizedDescription)")
            }
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw NovaMLXError.apiError("failed to fetch 'image' URL: HTTP \(http.statusCode)")
            }
            return data
        }
        if image.hasPrefix("/") {
            guard let data = try? Data(contentsOf: URL(fileURLWithPath: image)) else {
                throw NovaMLXError.apiError("cannot read 'image' local file \(image)")
            }
            return data
        }
        guard let data = Data(base64Encoded: image) else {
            throw NovaMLXError.apiError("'image' must be a data URI, base64 string, http(s) URL, or local file path")
        }
        return data
    }
}

/// Usage block for completed 3D jobs. `meshes` and `size_tier` are the stable
/// metering fields for Level-2 (tknet/peer billing); `cost` stays 0 because
/// generation is local.
struct ThreeDUsage: Codable, Sendable, Equatable {
    let meshes: Int
    let sizeTier: String
    let cost: Double
    let isByok: Bool

    enum CodingKeys: String, CodingKey {
        case meshes, cost
        case sizeTier = "size_tier"
        case isByok = "is_byok"
    }
}

struct ThreeDJobResponse: Codable, Sendable {
    let id: String
    let generationId: String?
    let pollingUrl: String
    let status: String
    let model: String
    let texture: Bool
    let unsignedUrls: [String]?
    let usage: ThreeDUsage?
    let error: String?
    let createdAt: Int
    let verts: Int?
    let faces: Int?
    /// Live progress while the job runs (nil unless in_progress).
    let progress: VideoJobProgress?

    enum CodingKeys: String, CodingKey {
        case id, status, model, texture, usage, error, progress, verts, faces
        case generationId = "generation_id"
        case pollingUrl = "polling_url"
        case unsignedUrls = "unsigned_urls"
        case createdAt = "created_at"
    }
}

extension ThreeDJobResponse {
    init(job: Hy3DJob, progress: Hy3DProgress? = nil) {
        id = job.id
        generationId = job.status == .completed ? job.id : nil
        pollingUrl = "/v1/3d/generations/\(job.id)"
        status = job.status.rawValue
        model = job.model
        texture = job.texture
        unsignedUrls = job.status == .completed
            ? ["/v1/3d/generations/\(job.id)/content"]
            : nil
        usage = job.status == .completed
            ? ThreeDUsage(meshes: 1, sizeTier: job.sizeTier ?? "shape", cost: 0, isByok: false)
            : nil
        error = job.error
        createdAt = Int(job.createdAt.timeIntervalSince1970)
        verts = job.verts
        faces = job.faces
        self.progress = progress.map {
            VideoJobProgress(phase: $0.phase, completed: $0.completed, total: $0.total, percent: $0.percent)
        }
    }
}

struct ThreeDModelInfo: Codable, Sendable {
    let id: String
    let loaded: Bool
    /// `shape` or `paint`.
    let slot: String

    enum CodingKeys: String, CodingKey { case id, loaded, slot }
}

struct ThreeDModelsResponse: Codable, Sendable {
    let data: [ThreeDModelInfo]
}
