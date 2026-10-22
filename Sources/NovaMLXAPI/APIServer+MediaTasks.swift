import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdRouter
import NIOCore
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference

func imageGenerationBody(job: ImageTaskJob, result: ImageGenerationResult) throws -> ImageGenerationResponse {
    let imageData: [ImageData]
    switch job.responseFormat {
    case "url":
        imageData = try result.images.enumerated().map { (i, b64) in
            guard let data = Data(base64Encoded: b64) else {
                throw NovaMLXError.apiError("Failed to decode generated image")
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("novamlx_img_\(job.id)_\(i).png")
            try data.write(to: url)
            return ImageData(b64Json: nil, url: url.absoluteString, revisedPrompt: job.prompt)
        }
    default:
        imageData = result.images.map { b64 in
            ImageData(b64Json: b64, url: nil, revisedPrompt: job.prompt)
        }
    }
    return ImageGenerationResponse(
        created: Int(job.createdAt.timeIntervalSince1970),
        data: imageData,
        model: job.model,
        usage: .billed(from: result, images: imageData.count)
    )
}

extension NovaMLXAPIServer {
    /// Image and video jobs share `GET /v1/tasks/{id}`. Video's
    /// `/v1/videos/{id}` routes stay as the OpenRouter alias.
    /// Not inlined: `APIServer.start`'s router closure overflows the stack
    /// if these handlers are nested inside it.
    @inline(never)
    static func mediaTaskRoutes(inference: InferenceService) -> some MiddlewareProtocol<Request, Response, NovaRequestContext> {
        RouteGroup("/v1/tasks") {
            Get("{id}/content") { _, context in
                let taskId = try context.parameters.require("id")
                if let job = inference.imageTaskService.job(id: taskId) {
                    guard job.status == .completed, let result = job.result else {
                        throw NovaMLXError.apiError("task '\(taskId)' is \(job.status.rawValue)")
                    }
                    return try Self.jsonResponse(imageGenerationBody(job: job, result: result))
                }
                if inference.h3VideoService.job(id: taskId) != nil {
                    return try Self.videoFileResponse(jobId: taskId, inference: inference, index: 0)
                }
                if inference.hy3DService.job(id: taskId) != nil {
                    return try Self.glbFileResponse(jobId: taskId, hy3d: inference.hy3DService)
                }
                throw NovaMLXError.modelNotFound("task '\(taskId)'")
            }

            Get("{id}") { _, context in
                let taskId = try context.parameters.require("id")
                if let job = inference.imageTaskService.job(id: taskId) {
                    return try Self.jsonResponse(
                        MediaTaskResponse(image: job, progress: inference.imageTaskService.progress(id: taskId)))
                }
                if let job = inference.h3VideoService.job(id: taskId) {
                    return try Self.jsonResponse(
                        MediaTaskResponse(video: job, progress: inference.h3VideoService.progress(id: taskId)))
                }
                if let job = inference.hy3DService.job(id: taskId) {
                    return try Self.jsonResponse(
                        MediaTaskResponse(threeD: job, progress: inference.hy3DService.progress(id: taskId)))
                }
                throw NovaMLXError.modelNotFound("task '\(taskId)'")
            }

            Post("{id}/cancel") { _, context in
                let taskId = try context.parameters.require("id")
                if inference.imageTaskService.job(id: taskId) != nil {
                    let job = try inference.imageTaskService.cancel(id: taskId)
                    return try Self.jsonResponse(
                        MediaTaskResponse(image: job, progress: inference.imageTaskService.progress(id: taskId)))
                }
                if inference.h3VideoService.job(id: taskId) != nil {
                    let job = try inference.h3VideoService.cancel(id: taskId)
                    return try Self.jsonResponse(
                        MediaTaskResponse(video: job, progress: inference.h3VideoService.progress(id: taskId)))
                }
                if inference.hy3DService.job(id: taskId) != nil {
                    let job = try inference.hy3DService.cancel(id: taskId)
                    return try Self.jsonResponse(
                        MediaTaskResponse(threeD: job, progress: inference.hy3DService.progress(id: taskId)))
                }
                throw NovaMLXError.modelNotFound("task '\(taskId)'")
            }
        }
    }

    /// MP4 body for a completed video job. Shared by `/v1/videos/{id}/content`
    /// and `/v1/tasks/{id}/content`.
    static func videoFileResponse(jobId: String, inference: InferenceService, index: Int) throws -> Response {
        guard let fileURL = inference.h3VideoService.contentURL(id: jobId, index: index) else {
            throw NovaMLXError.modelNotFound("video content for job '\(jobId)'")
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = (attrs[.size] as? NSNumber)?.intValue
        let body = ResponseBody(contentLength: fileSize) { writer in
            let handle = try FileHandle(forReadingFrom: fileURL)
            while true {
                let chunk = try handle.read(upToCount: 256 * 1024) ?? Data()
                if chunk.isEmpty { break }
                do { try await writer.write(ByteBuffer(data: chunk)) }
                catch { break }
            }
            try? handle.close()
            try await writer.finish(nil)
        }
        return Response(
            status: .ok,
            headers: [
                .contentType: "video/mp4",
                .contentDisposition: "attachment; filename=\"\(jobId).mp4\"",
            ],
            body: body
        )
    }
}
