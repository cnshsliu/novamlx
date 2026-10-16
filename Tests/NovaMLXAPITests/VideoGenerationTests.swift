import Foundation
import Testing
import NovaMLXCore

@testable import NovaMLXAPI
@testable import NovaMLXEngine

@Suite("Video Generation (OpenRouter-compatible)")
struct VideoGenerationTests {

    @Test("VideoGenerationRequest decodes the full OpenRouter payload")
    func testRequestDecode() throws {
        let json = """
        {
            "model": "minimax/h3",
            "prompt": "a red fox in snow",
            "duration": 4,
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "size": "864x480",
            "frame_images": [
                {"frame_type": "first_frame", "image_url": {"url": "https://example.com/first.png"}},
                {"frame_type": "last_frame", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
            ],
            "input_references": [
                {"image_url": {"url": "https://example.com/ref1.png"}},
                {"image_url": {"url": "https://example.com/ref2.png"}}
            ],
            "generate_audio": true,
            "seed": 42,
            "callback_url": "https://example.com/hook",
            "provider": {"options": {"h3": {"steps": 4, "layers": 40, "reuse": 3}, "other_vendor": {"x": 1}}}
        }
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(VideoGenerationRequest.self, from: json)
        #expect(req.model == "minimax/h3")
        #expect(req.prompt == "a red fox in snow")
        #expect(req.duration == 4)
        #expect(req.resolution == "480p")
        #expect(req.aspectRatio == "16:9")
        #expect(req.size == "864x480")
        #expect(req.frameImages?.count == 2)
        #expect(req.frameImages?[0].frameType == "first_frame")
        #expect(req.frameImages?[0].imageUrl.url == "https://example.com/first.png")
        #expect(req.inputReferences?.count == 2)
        #expect(req.generateAudio == true)
        #expect(req.seed == 42)
        #expect(req.callbackUrl == "https://example.com/hook")
    }

    @Test("VideoGenerationRequest tolerates a minimal payload")
    func testRequestDefaults() throws {
        let json = """
        {"model": "minimax/h3", "prompt": "hello"}
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(VideoGenerationRequest.self, from: json)
        #expect(req.duration == nil)
        #expect(req.size == nil)
        #expect(req.frameImages == nil)
        #expect(req.inputReferences == nil)
        #expect(req.generateAudio == nil)
        #expect(req.seed == nil)
        #expect(req.callbackUrl == nil)
        #expect(req.provider == nil)
    }

    @Test("toJobRequest maps frame_images, input_references and provider.options.h3")
    func testJobRequestMapping() throws {
        let json = """
        {
            "model": "minimax/hailuo-3",
            "prompt": "a cat",
            "duration": 2,
            "seed": 7,
            "frame_images": [
                {"frame_type": "first_frame", "image_url": {"url": "https://example.com/f.png"}},
                {"frame_type": "last_frame", "image_url": {"url": "https://example.com/l.png"}}
            ],
            "input_references": [{"image_url": {"url": "https://example.com/r.png"}}],
            "callback_url": "https://example.com/hook",
            "provider": {"options": {"h3": {"steps": 4, "layers": 40, "reuse": 3}}}
        }
        """.data(using: .utf8)!

        let job = try JSONDecoder().decode(VideoGenerationRequest.self, from: json).toJobRequest()
        #expect(job.model == "minimax/hailuo-3")
        #expect(job.prompt == "a cat")
        #expect(job.durationSeconds == 2)
        #expect(job.seed == 7)
        #expect(job.firstFrameURL == "https://example.com/f.png")
        #expect(job.lastFrameURL == "https://example.com/l.png")
        #expect(job.referenceImageURLs == ["https://example.com/r.png"])
        #expect(job.callbackURL == "https://example.com/hook")
        #expect(job.steps == 4)
        #expect(job.layers == 40)
        #expect(job.reuse == 3)
    }

    @Test("provider.options without the h3 namespace yields no overrides")
    func testProviderOptionsIgnored() throws {
        let json = """
        {"model": "minimax/h3", "prompt": "x", "provider": {"options": {"other": {"steps": 99}}}}
        """.data(using: .utf8)!

        let job = try JSONDecoder().decode(VideoGenerationRequest.self, from: json).toJobRequest()
        #expect(job.steps == nil)
        #expect(job.layers == nil)
        #expect(job.reuse == nil)
    }

    @Test("Size/resolution/aspect validation mapping")
    func testSizeMapping() throws {
        // Explicit supported size passes through.
        #expect(try H3VideoService.resolveSize(size: "512x512", resolution: nil, aspectRatio: nil) == (512, 512))
        #expect(try H3VideoService.resolveSize(size: "864x480", resolution: nil, aspectRatio: nil) == (864, 480))
        // Unsupported explicit size → 400 with the legal values listed.
        #expect(throws: NovaMLXError.self) {
            try H3VideoService.resolveSize(size: "999x999", resolution: nil, aspectRatio: nil)
        }
        // Aspect-ratio mapping to the nearest supported size.
        #expect(try H3VideoService.resolveSize(size: nil, resolution: nil, aspectRatio: "16:9") == (864, 480))
        #expect(try H3VideoService.resolveSize(size: nil, resolution: nil, aspectRatio: "9:16") == (480, 864))
        #expect(try H3VideoService.resolveSize(size: nil, resolution: nil, aspectRatio: "1:1") == (512, 512))
        #expect(try H3VideoService.resolveSize(size: nil, resolution: "720p", aspectRatio: "1:1") == (640, 640))
        // Default is the model's native landscape frame.
        #expect(try H3VideoService.resolveSize(size: nil, resolution: nil, aspectRatio: nil) == (864, 480))
    }

    @Test("VideoJobResponse encodes the OpenRouter job shape")
    func testJobResponseEncode() throws {
        var job = H3VideoJob(
            id: "job-abc123",
            status: .completed,
            model: "minimax/h3",
            prompt: "a red fox",
            width: 864,
            height: 480,
            durationSeconds: 4,
            seed: 42,
            steps: 20,
            layers: 45,
            reuse: 2,
            firstFrameURL: nil,
            lastFrameURL: nil,
            referenceImageURLs: [],
            callbackURL: nil,
            error: nil,
            createdAt: Date(timeIntervalSince1970: 1_715_234_567),
            startedAt: nil,
            finishedAt: nil,
            expiredAt: nil
        )
        let encoded = try JSONEncoder().encode(VideoJobResponse(job: job))
        let json = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]

        #expect(json["id"] as? String == "job-abc123")
        #expect(json["generation_id"] as? String == "job-abc123")
        #expect(json["polling_url"] as? String == "/v1/videos/job-abc123")
        #expect(json["status"] as? String == "completed")
        #expect(json["unsigned_urls"] as? [String] == ["/v1/videos/job-abc123/content?index=0"])
        let usage = json["usage"] as! [String: Any]
        #expect(usage["cost"] as? Double == 0)
        #expect(usage["is_byok"] as? Bool == false)
        #expect(json["created_at"] as? Int == 1_715_234_567)

        // A pending job has no content URLs and no usage.
        job.status = .pending
        let pending = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(VideoJobResponse(job: job))) as! [String: Any]
        #expect(pending["status"] as? String == "pending")
        #expect(pending["unsigned_urls"] == nil)
        #expect(pending["usage"] == nil)
        #expect(pending["generation_id"] == nil)
    }

    @Test("VideoModelsResponse lists the h3 capability table")
    func testModelsResponseEncode() throws {
        let encoded = try JSONEncoder().encode(VideoModelsResponse.h3)
        let json = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
        let data = json["data"] as! [[String: Any]]
        #expect(data.count == 2)
        let h3 = data.first { $0["id"] as? String == "minimax/h3" }!
        #expect(h3["supported_sizes"] as? [String] == ["864x480", "480x864", "512x512", "640x640"])
        #expect(h3["supported_aspect_ratios"] as? [String] == ["16:9", "9:16", "1:1"])
        #expect((h3["supported_durations"] as? [Int])?.count == H3VideoService.maxDurationSeconds)
        #expect(h3["supports_frame_images"] as? Bool == true)
        #expect(data.contains { $0["id"] as? String == "minimax/hailuo-3" })
    }
}
