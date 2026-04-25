import XCTest
@testable import NovaMLXAPI
import NovaMLXCore

final class OCROptimizerTests: XCTestCase {

    // MARK: - Model Detection

    func testDetectDeepSeekOCR() {
        XCTAssertTrue(OCROptimizer.isOCRModel("deepseekocr"))
        XCTAssertTrue(OCROptimizer.isOCRModel("DeepSeekOCR"))
        XCTAssertTrue(OCROptimizer.isOCRModel("my-deepseekocr-v2"))
    }

    func testDetectDeepSeekOCR2() {
        XCTAssertTrue(OCROptimizer.isOCRModel("deepseekocr_2"))
        XCTAssertTrue(OCROptimizer.isOCRModel("DeepSeek-OCR-2"))
    }

    func testDetectDotsOCR() {
        XCTAssertTrue(OCROptimizer.isOCRModel("dots_ocr"))
        XCTAssertTrue(OCROptimizer.isOCRModel("dots-ocr"))
    }

    func testDetectGLMOCR() {
        XCTAssertTrue(OCROptimizer.isOCRModel("glm_ocr"))
        XCTAssertTrue(OCROptimizer.isOCRModel("glm-ocr"))
    }

    func testNonOCRModelRejected() {
        XCTAssertFalse(OCROptimizer.isOCRModel("qwen3-4b"))
        XCTAssertFalse(OCROptimizer.isOCRModel("llama-3"))
        XCTAssertFalse(OCROptimizer.isOCRModel("gpt-4"))
    }

    // MARK: - Prompt Injection

    func testPromptInjectionImageOnly() {
        let messages = [
            ChatMessage(role: .user, content: "", images: ["data:image/png;base64,abc"]),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "deepseekocr")
        XCTAssertEqual(result[0].content, "Convert the document to markdown.")
        XCTAssertEqual(result[0].images?.count, 1)
    }

    func testNoPromptInjectionWhenUserProvidesText() {
        let messages = [
            ChatMessage(role: .user, content: "Extract tables", images: ["data:image/png;base64,abc"]),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "deepseekocr")
        XCTAssertEqual(result[0].content, "Extract tables")
    }

    func testNoPromptInjectionForNonOCRModel() {
        let messages = [
            ChatMessage(role: .user, content: "", images: ["data:image/png;base64,abc"]),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "qwen3-4b")
        XCTAssertEqual(result[0].content, "")
    }

    func testDotsOCRPrompt() {
        let messages = [
            ChatMessage(role: .user, content: nil, images: ["http://img.png"]),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "dots_ocr")
        XCTAssertEqual(result[0].content, "Convert this page to clean Markdown while preserving reading order.")
    }

    func testGLMOCRPrompt() {
        let messages = [
            ChatMessage(role: .user, content: nil, images: ["http://img.png"]),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "glm_ocr")
        XCTAssertEqual(result[0].content, "Text Recognition:")
    }

    func testNoPromptInjectionWithoutImages() {
        let messages = [
            ChatMessage(role: .user, content: ""),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "deepseekocr")
        XCTAssertEqual(result[0].content, "")
    }

    func testSystemMessageUntouched() {
        let messages = [
            ChatMessage(role: .system, content: "You are helpful"),
            ChatMessage(role: .user, content: "", images: ["http://img.png"]),
        ]
        let result = OCROptimizer.applyPrompt(messages: messages, modelName: "deepseekocr")
        XCTAssertEqual(result[0].content, "You are helpful")
        XCTAssertEqual(result[1].content, "Convert the document to markdown.")
    }

    // MARK: - Stop Sequences

    func testStopSequencesMergedForOCRModel() {
        let userStop = ["</output>"]
        let result = OCROptimizer.applyStopSequences(userStop, modelName: "deepseekocr")
        XCTAssertNotNil(result)
        XCTAssertTrue(result!.contains("</output>"))
        XCTAssertTrue(result!.contains("<|im_start|>"))
        XCTAssertTrue(result!.contains("<|im_end|>"))
    }

    func testStopSequencesNoDupe() {
        let userStop = ["<|im_start|>"]
        let result = OCROptimizer.applyStopSequences(userStop, modelName: "deepseekocr")
        let count = result!.filter { $0 == "<|im_start|>" }.count
        XCTAssertEqual(count, 1)
    }

    func testStopSequencesUnchangedForNonOCR() {
        let userStop = ["</output>"]
        let result = OCROptimizer.applyStopSequences(userStop, modelName: "qwen3-4b")
        XCTAssertEqual(result, ["</output>"])
    }

    func testStopSequencesAddedWhenNoneProvided() {
        let result = OCROptimizer.applyStopSequences(nil, modelName: "deepseekocr")
        XCTAssertNotNil(result)
        XCTAssertTrue(result!.contains("<|im_start|>"))
    }

    // MARK: - Sampling Overrides

    func testSamplingDefaultsDeepSeekOCR() {
        let overrides = OCROptimizer.samplingOverrides(
            modelName: "deepseekocr", userTemperature: nil,
            userMaxTokens: nil, userRepetitionPenalty: nil
        )
        XCTAssertEqual(overrides.temperature, 0.0)
        XCTAssertEqual(overrides.maxTokens, 8192)
        XCTAssertNil(overrides.repetitionPenalty)
    }

    func testSamplingDefaultsGLMOCR() {
        let overrides = OCROptimizer.samplingOverrides(
            modelName: "glm_ocr", userTemperature: nil,
            userMaxTokens: nil, userRepetitionPenalty: nil
        )
        XCTAssertEqual(overrides.temperature, 0.0)
        XCTAssertEqual(overrides.maxTokens, 4096)
        XCTAssertEqual(overrides.repetitionPenalty, 1.1)
    }

    func testUserOverridesTakePriority() {
        let overrides = OCROptimizer.samplingOverrides(
            modelName: "deepseekocr", userTemperature: 0.5,
            userMaxTokens: 2048, userRepetitionPenalty: 1.2
        )
        XCTAssertEqual(overrides.temperature, 0.5)
        XCTAssertEqual(overrides.maxTokens, 2048)
        XCTAssertEqual(overrides.repetitionPenalty, 1.2)
    }

    func testPartialUserOverride() {
        let overrides = OCROptimizer.samplingOverrides(
            modelName: "deepseekocr", userTemperature: 0.5,
            userMaxTokens: nil, userRepetitionPenalty: nil
        )
        XCTAssertEqual(overrides.temperature, 0.5)
        XCTAssertEqual(overrides.maxTokens, 8192) // default
    }

    func testNoOverridesForNonOCRModel() {
        let overrides = OCROptimizer.samplingOverrides(
            modelName: "qwen3-4b", userTemperature: nil,
            userMaxTokens: nil, userRepetitionPenalty: nil
        )
        XCTAssertNil(overrides.temperature)
        XCTAssertNil(overrides.maxTokens)
        XCTAssertNil(overrides.repetitionPenalty)
    }
}
