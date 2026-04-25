import Foundation
import NovaMLXCore

// MARK: - OCR Model Auto-Optimization

enum OCROptimizer {

    // MARK: - OCR Model Detection

    private static let ocrModelPatterns: Set<String> = [
        "deepseekocr", "deepseekocr_2", "deepseek-ocr",
        "dots_ocr", "dots-ocr",
        "glm_ocr", "glm-ocr",
    ]

    static func isOCRModel(_ modelName: String) -> Bool {
        let lower = modelName.lowercased()
        return ocrModelPatterns.contains { lower.contains($0) }
    }

    private static func ocrType(for modelName: String) -> String? {
        let lower = modelName.lowercased()
        if lower.contains("deepseekocr_2") || lower.contains("deepseek-ocr-2") { return "deepseekocr_2" }
        if lower.contains("deepseekocr") || lower.contains("deepseek-ocr") { return "deepseekocr" }
        if lower.contains("dots_ocr") || lower.contains("dots-ocr") { return "dots_ocr" }
        if lower.contains("glm_ocr") || lower.contains("glm-ocr") { return "glm_ocr" }
        return nil
    }

    // MARK: - Default Prompts

    private static let defaultPrompts: [String: String] = [
        "deepseekocr": "Convert the document to markdown.",
        "deepseekocr_2": "Convert the document to markdown.",
        "dots_ocr": "Convert this page to clean Markdown while preserving reading order.",
        "glm_ocr": "Text Recognition:",
    ]

    // MARK: - Extra Stop Sequences

    static let extraStopSequences: [String] = [
        "<|im_start|>", "<|im_end|>", "<|endofassistant|>",
    ]

    // MARK: - Sampling Defaults

    private struct OCRSamplingDefaults: Sendable {
        let temperature: Double
        let maxTokens: Int
        let repetitionPenalty: Float?

        static let deepseekocr = OCRSamplingDefaults(temperature: 0.0, maxTokens: 8192, repetitionPenalty: nil)
        static let deepseekocr_2 = OCRSamplingDefaults(temperature: 0.0, maxTokens: 8192, repetitionPenalty: nil)
        static let dots_ocr = OCRSamplingDefaults(temperature: 0.0, maxTokens: 8192, repetitionPenalty: nil)
        static let glm_ocr = OCRSamplingDefaults(temperature: 0.0, maxTokens: 4096, repetitionPenalty: 1.1)
    }

    private static let samplingDefaults: [String: OCRSamplingDefaults] = [
        "deepseekocr": .deepseekocr,
        "deepseekocr_2": .deepseekocr_2,
        "dots_ocr": .dots_ocr,
        "glm_ocr": .glm_ocr,
    ]

    // MARK: - Prompt Injection

    /// Inject default OCR prompt for image-only requests.
    /// Only modifies messages when user sends images WITHOUT any text.
    static func applyPrompt(messages: [ChatMessage], modelName: String) -> [ChatMessage] {
        guard let ocrType = ocrType(for: modelName),
              let defaultPrompt = defaultPrompts[ocrType] else { return messages }

        var result = messages
        for i in result.indices where result[i].role == .user {
            let hasImages = !(result[i].images?.isEmpty ?? true)
            let hasText = !(result[i].content?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
            if hasImages && !hasText {
                result[i] = ChatMessage(
                    role: .user, content: defaultPrompt,
                    images: result[i].images, name: result[i].name,
                    toolCallId: result[i].toolCallId, toolCalls: result[i].toolCalls
                )
            }
        }
        return result
    }

    // MARK: - Stop Sequences

    /// Merge OCR extra stop sequences with user-provided ones.
    static func applyStopSequences(_ existing: [String]?, modelName: String) -> [String]? {
        guard isOCRModel(modelName) else { return existing }
        var merged = existing ?? []
        for seq in extraStopSequences where !merged.contains(seq) {
            merged.append(seq)
        }
        return merged.isEmpty ? nil : merged
    }

    // MARK: - Sampling Overrides

    struct SamplingOverrides {
        var temperature: Double?
        var maxTokens: Int?
        var repetitionPenalty: Float?
    }

    /// Get OCR-specific sampling overrides. Only returns values for params the user didn't set.
    static func samplingOverrides(
        modelName: String,
        userTemperature: Double?,
        userMaxTokens: Int?,
        userRepetitionPenalty: Float?
    ) -> SamplingOverrides {
        guard let ocrType = ocrType(for: modelName),
              let defaults = samplingDefaults[ocrType] else {
            return SamplingOverrides()
        }
        return SamplingOverrides(
            temperature: userTemperature ?? defaults.temperature,
            maxTokens: userMaxTokens ?? defaults.maxTokens,
            repetitionPenalty: userRepetitionPenalty ?? defaults.repetitionPenalty
        )
    }
}
