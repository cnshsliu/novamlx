import XCTest
@testable import NovaMLXCore

final class SpeculativeDecodingTests: XCTestCase {

    // MARK: - InferenceRequest draftModel fields

    func testInferenceRequestDefaultDraftModelIsNil() {
        let req = InferenceRequest(model: "test", messages: [ChatMessage(role: .user, content: "hi")])
        XCTAssertNil(req.draftModel, "draftModel should default to nil")
        XCTAssertNil(req.numDraftTokens, "numDraftTokens should default to nil")
    }

    func testInferenceRequestWithDraftModel() {
        let req = InferenceRequest(
            model: "Qwen3.5-9B",
            messages: [ChatMessage(role: .user, content: "hello")],
            draftModel: "Qwen3-0.6B",
            numDraftTokens: 5
        )
        XCTAssertEqual(req.draftModel, "Qwen3-0.6B")
        XCTAssertEqual(req.numDraftTokens, 5)
    }

    func testInferenceRequestWithDraftModelOnly() {
        let req = InferenceRequest(
            model: "main-model",
            messages: [],
            draftModel: "draft-model"
        )
        XCTAssertEqual(req.draftModel, "draft-model")
        XCTAssertNil(req.numDraftTokens, "numDraftTokens should be nil when not specified")
    }

    // MARK: - CodableInferenceRequest round-trip

    func testCodableRoundTripWithDraftModel() {
        let original = InferenceRequest(
            model: "main",
            messages: [ChatMessage(role: .user, content: "test")],
            draftModel: "draft",
            numDraftTokens: 8
        )
        let codable = CodableInferenceRequest(from: original)
        let restored = codable.toInferenceRequest()

        XCTAssertEqual(restored.model, "main")
        XCTAssertEqual(restored.draftModel, "draft")
        XCTAssertEqual(restored.numDraftTokens, 8)
    }

    func testCodableRoundTripWithoutDraftModel() {
        let original = InferenceRequest(
            model: "main",
            messages: [ChatMessage(role: .user, content: "test")]
        )
        let codable = CodableInferenceRequest(from: original)
        let restored = codable.toInferenceRequest()

        XCTAssertNil(restored.draftModel)
        XCTAssertNil(restored.numDraftTokens)
    }

    func testCodableJSONRoundTrip() throws {
        let original = CodableInferenceRequest(from: InferenceRequest(
            model: "test-model",
            messages: [ChatMessage(role: .user, content: "hello")],
            temperature: 0.7,
            maxTokens: 100,
            draftModel: "small-model",
            numDraftTokens: 3
        ))

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(CodableInferenceRequest.self, from: data)
        let restored = decoded.toInferenceRequest()

        XCTAssertEqual(restored.model, "test-model")
        XCTAssertEqual(restored.temperature, 0.7)
        XCTAssertEqual(restored.maxTokens, 100)
        XCTAssertEqual(restored.draftModel, "small-model")
        XCTAssertEqual(restored.numDraftTokens, 3)
    }

    // MARK: - Speculative decoding is opt-in

    func testSpeculativeDecodingIsOptIn() {
        // Without draftModel, regular decode should be used (no overhead)
        let req = InferenceRequest(
            model: "model",
            messages: [ChatMessage(role: .user, content: "normal request")]
        )
        XCTAssertNil(req.draftModel, "Speculative decoding should be opt-in via draftModel parameter")
    }
}
