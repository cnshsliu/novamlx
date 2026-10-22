import Foundation
import NovaMLXEngine

/// One poll payload for image and video.
///
/// Video jobs keep `GET /v1/videos/{id}` (OpenRouter). Both modalities also
/// answer `GET /v1/tasks/{id}` (DashScope's shared task resource). Status
/// strings are the video job's: pending, in_progress, completed, failed,
/// cancelled, expired.
struct MediaTaskUsage: Codable, Sendable {
    var images: Int?
    /// 3D jobs count meshes instead of images.
    var meshes: Int?
    var sizeTier: String?
    var cost: Double?
    var isByok: Bool?

    enum CodingKeys: String, CodingKey {
        case images, meshes, cost
        case sizeTier = "size_tier"
        case isByok = "is_byok"
    }
}

struct MediaTaskResponse: Codable, Sendable {
    var id: String
    var object: String
    var task: String
    var status: String
    var pollingUrl: String
    var model: String
    var createdAt: Int
    var progress: VideoJobProgress?
    var error: String?
    var contentUrl: String?
    var usage: MediaTaskUsage?

    enum CodingKeys: String, CodingKey {
        case id, object, task, status, model, progress, error, usage
        case pollingUrl = "polling_url"
        case createdAt = "created_at"
        case contentUrl = "content_url"
    }

    init(image job: ImageTaskJob, progress: ImageTaskProgress?) {
        id = job.id
        object = "task"
        task = job.operation == "edit" ? "image.edit" : "image.generation"
        status = job.status.rawValue
        pollingUrl = "/v1/tasks/\(job.id)"
        model = job.model
        createdAt = Int(job.createdAt.timeIntervalSince1970)
        self.progress = progress.map {
            VideoJobProgress(phase: $0.phase, completed: $0.completed, total: $0.total, percent: $0.percent)
        }
        error = job.error
        contentUrl = job.status == .completed ? "/v1/tasks/\(job.id)/content" : nil
        if let result = job.result, job.status == .completed {
            let usage = ImageUsage.billed(from: result, images: result.images.count)
            self.usage = MediaTaskUsage(images: usage.images, sizeTier: usage.sizeTier, cost: nil, isByok: nil)
        } else {
            usage = nil
        }
    }

    init(video job: H3VideoJob, progress: H3VideoProgress?) {
        id = job.id
        object = "task"
        task = "video.generation"
        status = job.status.rawValue
        pollingUrl = "/v1/tasks/\(job.id)"
        model = job.model
        createdAt = Int(job.createdAt.timeIntervalSince1970)
        self.progress = progress.map {
            VideoJobProgress(phase: $0.phase, completed: $0.completed, total: $0.total, percent: $0.percent)
        }
        error = job.error
        contentUrl = job.status == .completed ? "/v1/tasks/\(job.id)/content" : nil
        usage = job.status == .completed ? MediaTaskUsage(images: nil, sizeTier: nil, cost: 0, isByok: false) : nil
    }

    init(threeD job: Hy3DJob, progress: Hy3DProgress?) {
        id = job.id
        object = "task"
        task = "3d.generation"
        status = job.status.rawValue
        pollingUrl = "/v1/tasks/\(job.id)"
        model = job.model
        createdAt = Int(job.createdAt.timeIntervalSince1970)
        self.progress = progress.map {
            VideoJobProgress(phase: $0.phase, completed: $0.completed, total: $0.total, percent: $0.percent)
        }
        error = job.error
        contentUrl = job.status == .completed ? "/v1/tasks/\(job.id)/content" : nil
        usage = job.status == .completed
            ? MediaTaskUsage(images: nil, meshes: 1, sizeTier: job.sizeTier ?? "shape", cost: 0, isByok: false)
            : nil
    }
}
