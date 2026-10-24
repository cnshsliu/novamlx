import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdRouter
import NIOCore
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

extension NovaMLXAPIServer {
    /// Task-style 3D generation routes (Hunyuan3D-2.1). Mirrors the video
    /// feature: 202 on create, poll by id, chunk-streamed content download.
    /// Jobs also answer on the shared `GET /v1/tasks/{id}` (see
    /// `mediaTaskRoutes`) as `task = "3d.generation"`.
    /// Not inlined: `APIServer.start`'s router closure overflows the stack
    /// if these handlers are nested inside it.
    @inline(never)
    static func threeDRoutes(
        inference: InferenceService,
        modelManager: ModelManager
    ) -> some MiddlewareProtocol<Request, Response, NovaRequestContext> {
        RouteGroup("/v1/3d") {
            // Registered BEFORE "/v1/3d/generations/{id}" so "models" isn't
            // captured as an id.
            Get("/models") { _, _ in
                let loaded = inference.hy3DService.listLoadedModels()
                let downloaded = modelManager.downloadedModels()
                    .filter { $0.family == .hunyuan3d || $0.modelType == .threed }
                    .map(\.id)
                var data: [ThreeDModelInfo] = loaded.map { id in
                    let isPaint = id.lowercased().contains("paint")
                    return ThreeDModelInfo(id: id, loaded: true, slot: isPaint ? "paint" : "shape")
                }
                for id in downloaded where !loaded.contains(id) {
                    let isPaint = id.lowercased().contains("paint")
                    data.append(ThreeDModelInfo(id: id, loaded: false, slot: isPaint ? "paint" : "shape"))
                }
                return try Self.jsonResponse(ThreeDModelsResponse(data: data))
            }

            Post("/generations") { request, _ in
                let body = try await request.body.collect(upTo: .max)
                let req: ThreeDGenerationRequest
                do {
                    req = try JSONDecoder().decode(ThreeDGenerationRequest.self, from: body)
                } catch {
                    throw NovaMLXError.apiError("invalid 3D generation request: \(error.localizedDescription)")
                }
                if let format = req.outputFormat, format.lowercased() != "glb" {
                    throw NovaMLXError.apiError("output_format '\(format)' is not supported (only glb)")
                }
                let imageData = try await req.resolveImageData()
                let job = try inference.hy3DService.submit(
                    model: req.model,
                    paintModel: req.paintModel,
                    texture: req.texture ?? false,
                    image: imageData,
                    seed: req.seed.flatMap { UInt64(exactly: max(0, $0)) },
                    octreeResolution: req.octreeResolution ?? 256,
                    steps: req.numInferenceSteps,
                    guidance: req.guidanceScale.map(Float.init)
                )
                if let rid = HTTPHelpers.requestID(from: request) {
                    RequestLogStore.shared.finish(
                        model: job.model, kind: .threed, status: .success,
                        tps: 0, promptTokens: 0, completionTokens: 0,
                        durationMs: 0, requestId: rid)
                }
                return try Self.jsonResponse(ThreeDJobResponse(job: job), httpStatus: .accepted)
            }

            Get("/generations/{id}") { _, context in
                let jobId = try context.parameters.require("id")
                guard let job = inference.hy3DService.job(id: jobId) else {
                    throw NovaMLXError.modelNotFound("3D job '\(jobId)'")
                }
                return try Self.jsonResponse(
                    ThreeDJobResponse(job: job, progress: inference.hy3DService.progress(id: jobId)))
            }

            Get("/generations/{id}/content") { _, context in
                let jobId = try context.parameters.require("id")
                return try Self.glbFileResponse(jobId: jobId, hy3d: inference.hy3DService)
            }

            Post("/generations/{id}/cancel") { _, context in
                let jobId = try context.parameters.require("id")
                let job = try inference.hy3DService.cancel(id: jobId)
                return try Self.jsonResponse(ThreeDJobResponse(job: job))
            }
        }
    }

    /// GLB body for a completed 3D job. Streamed in chunks like the video
    /// content route — textured GLBs run to tens of MB.
    static func glbFileResponse(jobId: String, hy3d: Hy3DService) throws -> Response {
        guard let fileURL = hy3d.contentURL(id: jobId) else {
            throw NovaMLXError.modelNotFound("3D content for job '\(jobId)'")
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = (attrs[.size] as? NSNumber)?.intValue
        let body = ResponseBody(contentLength: fileSize) { writer in
            let handle = try FileHandle(forReadingFrom: fileURL)
            while true {
                let chunk = try handle.read(upToCount: 256 * 1024) ?? Data()
                if chunk.isEmpty { break }
                do { try await writer.write(ByteBuffer(data: chunk)) }
                catch { break } // client went away
            }
            try? handle.close()
            try await writer.finish(nil)
        }
        return Response(
            status: .ok,
            headers: [
                .contentType: "model/gltf-binary",
                .contentDisposition: "attachment; filename=\"\(jobId).glb\"",
            ],
            body: body
        )
    }
}
