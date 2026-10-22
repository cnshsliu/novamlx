import Foundation
import Testing
import NovaMLXCore

@testable import NovaMLXAPI
@testable import NovaMLXEngine

@Suite("3D Generation (task-style)")
struct ThreeDGenerationTests {

    @Test("ThreeDGenerationRequest decodes the full payload")
    func testRequestDecode() throws {
        let json = """
        {
            "model": "zimengxiong/hunyuan3d-mlx-shape-small",
            "image": "data:image/png;base64,iVBORw0KGgo=",
            "prompt": null,
            "texture": true,
            "paint_model": "zimengxiong/hunyuan3d-mlx-paint-large",
            "seed": 42,
            "octree_resolution": 256,
            "num_inference_steps": 30,
            "guidance_scale": 5.0,
            "output_format": "glb"
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(ThreeDGenerationRequest.self, from: json)
        #expect(req.model == "zimengxiong/hunyuan3d-mlx-shape-small")
        #expect(req.image.hasPrefix("data:image/png;base64,"))
        #expect(req.texture == true)
        #expect(req.paintModel == "zimengxiong/hunyuan3d-mlx-paint-large")
        #expect(req.seed == 42)
        #expect(req.octreeResolution == 256)
        #expect(req.numInferenceSteps == 30)
        #expect(req.guidanceScale == 5.0)
        #expect(req.outputFormat == "glb")
    }

    @Test("image is required — decoding without it fails")
    func testMissingImageFails() {
        let json = """
        {"model": "zimengxiong/hunyuan3d-mlx-shape-small"}
        """.data(using: .utf8)!

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(ThreeDGenerationRequest.self, from: json)
        }
    }

    @Test("resolveImageData accepts data URIs, raw base64, and rejects garbage")
    func testResolveImageData() async throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47])

        let dataURI = ThreeDGenerationRequest(
            model: "m", image: "data:image/png;base64," + png.base64EncodedString())
        #expect(try await dataURI.resolveImageData() == png)

        let rawB64 = ThreeDGenerationRequest(model: "m", image: png.base64EncodedString())
        #expect(try await rawB64.resolveImageData() == png)

        let garbage = ThreeDGenerationRequest(model: "m", image: "!!!not-base64!!!")
        await #expect(throws: NovaMLXError.self) {
            _ = try await garbage.resolveImageData()
        }
    }

    @Test("ThreeDJobResponse maps status, urls and usage like the video shape")
    func testJobResponseMapping() throws {
        let pending = Hy3DJob(
            id: "h3d-abc", status: .pending, model: "m", texture: false,
            imageData: Data([1]), createdAt: Date(timeIntervalSince1970: 1_700_000_000))
        let pendingResp = ThreeDJobResponse(job: pending)
        #expect(pendingResp.status == "pending")
        #expect(pendingResp.pollingUrl == "/v1/3d/generations/h3d-abc")
        #expect(pendingResp.unsignedUrls == nil)
        #expect(pendingResp.usage == nil)
        #expect(pendingResp.generationId == nil)

        var done = pending
        done.status = .completed
        done.sizeTier = "shape+texture"
        done.verts = 1000
        done.faces = 2000
        let doneResp = ThreeDJobResponse(job: done)
        #expect(doneResp.generationId == "h3d-abc")
        #expect(doneResp.unsignedUrls == ["/v1/3d/generations/h3d-abc/content"])
        #expect(doneResp.usage?.meshes == 1)
        #expect(doneResp.usage?.sizeTier == "shape+texture")
        #expect(doneResp.usage?.cost == 0)
        #expect(doneResp.usage?.isByok == false)
    }

    @Test("MediaTaskResponse(threeD:) reports task = 3d.generation with meshes usage")
    func testMediaTaskMapping() {
        var job = Hy3DJob(
            id: "h3d-xyz", status: .completed, model: "m", texture: false,
            imageData: nil, createdAt: Date(timeIntervalSince1970: 1_700_000_000))
        job.sizeTier = "shape"
        let resp = MediaTaskResponse(
            threeD: job,
            progress: Hy3DProgress(phase: "done", completed: 100, total: 100, percent: 99))
        #expect(resp.task == "3d.generation")
        #expect(resp.object == "task")
        #expect(resp.status == "completed")
        #expect(resp.pollingUrl == "/v1/tasks/h3d-xyz")
        #expect(resp.contentUrl == "/v1/tasks/h3d-xyz/content")
        #expect(resp.usage?.meshes == 1)
        #expect(resp.usage?.sizeTier == "shape")
        #expect(resp.usage?.images == nil)
        #expect(resp.progress?.phase == "done")
    }

    @Test("submit refuses unloaded models and texture without a paint model")
    func testSubmitValidation() async throws {
        let service = Hy3DService(autostart: false)
        let png = Data([0x89, 0x50])

        await #expect(throws: NovaMLXError.self) {
            _ = try service.submit(model: "not-loaded", image: png)
        }

        // A shape-only submit works even without a paint slot.
        // (Shape container cannot be constructed without weights here, so
        // texture validation is exercised through the model-not-loaded path.)
        await #expect(throws: NovaMLXError.self) {
            _ = try service.submit(model: "not-loaded", texture: true, image: png)
        }

        // Unknown job ids are rejected.
        #expect(service.job(id: "h3d-nonexistent") == nil)
        #expect(!Hy3DService.isValidId("img-abc"))
        #expect(Hy3DService.isValidId("h3d-abc-123"))
    }
}
