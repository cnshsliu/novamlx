import Testing
import Foundation
import NovaMLXCore

@testable import NovaMLXAPI
@testable import NovaMLXEngine

@Suite("Image Generation")
struct ImageGenerationTests {

    @Test("ImageGenerationRequest decodes all fields")
    func testImageGenerationRequestDecode() throws {
        let json = """
        {"prompt":"a red cat","model":"sdxl-turbo","n":2,"size":"512x512","response_format":"url","quality":"hd","style":"vivid","seed":42,"negative_prompt":"blurry"}
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(ImageGenerationRequest.self, from: json)
        #expect(req.prompt == "a red cat")
        #expect(req.model == "sdxl-turbo")
        #expect(req.n == 2)
        #expect(req.size == "512x512")
        #expect(req.responseFormat == "url")
        #expect(req.quality == "hd")
        #expect(req.style == "vivid")
        #expect(req.seed == 42)
        #expect(req.negativePrompt == "blurry")
    }

    @Test("ImageGenerationRequest defaults and size parsing")
    func testImageGenerationRequestDefaults() throws {
        let json = """
        {"prompt":"hello","model":"test"}
        """.data(using: .utf8)!

        let req = try JSONDecoder().decode(ImageGenerationRequest.self, from: json)
        #expect(req.resolvedN == 1)
        #expect(req.resolvedSize == (1024, 1024))
        #expect(req.resolvedResponseFormat == "b64_json")
        #expect(req.negativePrompt == nil)
    }

    @Test("ImageGenerationResponse encodes with b64_json")
    func testImageGenerationResponseEncode() throws {
        let response = ImageGenerationResponse(
            created: 1715234567,
            data: [
                ImageData(b64Json: "iVBORw0KGgo=", url: nil, revisedPrompt: "a sunset"),
            ],
            model: "sdxl-turbo"
        )

        let encoded = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]

        #expect(json["created"] as? Int == 1715234567)
        #expect(json["model"] as? String == "sdxl-turbo")

        let data = json["data"] as! [[String: Any]]
        #expect(data.count == 1)
        #expect(data[0]["b64_json"] as? String == "iVBORw0KGgo=")
        #expect(data[0]["revised_prompt"] as? String == "a sunset")
        #expect(data[0]["url"] == nil)
    }

    @Test("ImageEditRequest resolves defaults")
    func testImageEditRequestDefaults() {
        let req = ImageEditRequest(
            image: Data(), mask: nil, prompt: "make it red",
            model: "sdxl-turbo", n: nil, size: nil, responseFormat: nil
        )
        #expect(req.resolvedN == 1)
        #expect(req.resolvedSize == (1024, 1024))
        #expect(req.resolvedResponseFormat == "b64_json")
    }

    @Test("ImageEditRequest resolves custom size and n")
    func testImageEditRequestCustom() {
        let req = ImageEditRequest(
            image: Data(), mask: Data(), prompt: "add stars",
            model: "sdxl-turbo", n: 3, size: "512x512", responseFormat: "url"
        )
        #expect(req.resolvedN == 3)
        #expect(req.resolvedSize == (512, 512))
        #expect(req.resolvedResponseFormat == "url")
    }

    @Test("ImageVariationRequest resolves defaults")
    func testImageVariationRequestDefaults() {
        let req = ImageVariationRequest(
            image: Data(), model: "sdxl-turbo", n: nil, size: nil, responseFormat: nil
        )
        #expect(req.resolvedN == 1)
        #expect(req.resolvedSize == (1024, 1024))
        #expect(req.resolvedResponseFormat == "b64_json")
    }
}
