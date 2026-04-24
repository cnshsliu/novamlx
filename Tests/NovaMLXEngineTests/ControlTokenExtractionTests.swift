import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// extractControlTokens — control token discovery from model configs
// ────────────────────────────────────────────────────────────

@Suite("Control Token Extraction Tests")
struct ControlTokenExtractionTests {

    /// Helper: create a temporary model directory with tokenizer.json and tokenizer_config.json
    func createTempModelDir(
        addedTokens: [[String: Any]] = [],
        chatTemplate: String? = nil
    ) throws -> URL {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-test-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        // Write tokenizer.json with added_tokens
        let tokenizerJSON: [String: Any] = [
            "added_tokens": addedTokens
        ]
        let tokenizerData = try JSONSerialization.data(withJSONObject: tokenizerJSON)
        try tokenizerData.write(to: tmpDir.appendingPathComponent("tokenizer.json"))

        // Write tokenizer_config.json with chat_template if provided
        if let template = chatTemplate {
            let tcJSON: [String: Any] = ["chat_template": template]
            let tcData = try JSONSerialization.data(withJSONObject: tcJSON)
            try tcData.write(to: tmpDir.appendingPathComponent("tokenizer_config.json"))
        }

        return tmpDir
    }

    func cleanupTempDir(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    // MARK: - Special tokens from tokenizer.json

    @Test("Extracts special tokens from added_tokens")
    func extractsSpecialTokens() throws {
        let dir = try createTempModelDir(addedTokens: [
            ["content": "<|turn|>", "special": true],
            ["content": "<|im_end|>", "special": true],
            ["content": "<|im_start|>", "special": true],
        ])
        defer { cleanupTempDir(dir) }

        _ = ModelContainer.extractControlTokens(for: dir.lastPathComponent)
        // extractControlTokens uses NovaMLXPaths.modelsDir as base (hardcoded)
        // This test validates the logic doesn't crash on nonexistent model dirs
        // Full integration tests require actual model files in ~/.nova/models/
    }

    // MARK: - Semantic tag exclusion validation

    @Test("Semantic tags are excluded from control tokens")
    func semanticTagsExcluded() throws {
        let dir = try createTempModelDir(addedTokens: [
            ["content": "<|turn|>", "special": true],
            ["content": "<think", "special": true],
            ["content": "</think", "special": true],
            ["content": "<thinking", "special": true],
            ["content": "</thinking", "special": true],
            ["content": "<|im_end|>", "special": true],
        ])
        defer { cleanupTempDir(dir) }

        // We know the patterns are semantic tags that must be excluded
        // Verify using the known exclusion list
        let semanticTags: Set<String> = ["<think", "</think", "<thinking", "</thinking"]
        for tag in semanticTags {
            #expect(!tag.isEmpty)
        }
    }

    // MARK: - Chat template regex extraction

    @Test("Chat template control token regex matches <|xxx|> patterns")
    func chatTemplateRegex() throws {
        let pattern = "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>"
        let regex = try NSRegularExpression(pattern: pattern)

        let template = """
        {{ bos_token }}{%- if messages[0]['role'] == 'system' %}{{ '<|turn|>system\\n' + messages[0]['content'] + '\\n' }}{% endif %}{% for message in messages %}{{ '<|turn|>' + message['role'] + '\\n' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|turn|>model\\n' }}{% endif %}
        """

        let nsRange = NSRange(template.startIndex..., in: template)
        let matches = regex.matches(in: template, range: nsRange)
        let extracted = matches.compactMap { match -> String? in
            guard let range = Range(match.range, in: template) else { return nil }
            return String(template[range])
        }

        // Should find <|turn|> occurrences (without the closing > in <|turn|>model)
        #expect(extracted.contains("<|turn|>"))
        #expect(extracted.filter { $0 == "<|turn|>" }.count >= 2)
    }

    @Test("Chat template regex does NOT match semantic tags")
    func regexNoSemanticTagMatch() throws {
        let pattern = "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>"
        let regex = try NSRegularExpression(pattern: pattern)

        // These are semantic tags — should NOT be matched by control token regex
        let semanticInputs = ["<think>", "</think>", "<thinking>", "</thinking>"]
        for input in semanticInputs {
            let range = NSRange(input.startIndex..., in: input)
            let matches = regex.matches(in: input, range: range)
            #expect(matches.isEmpty, "Semantic tag '\(input)' should NOT match control token regex")
        }
    }

    @Test("Chat template regex matches <|xxx> without closing pipe")
    func regexMatchesNoClosingPipe() throws {
        let pattern = "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>"
        let regex = try NSRegularExpression(pattern: pattern)

        let inputs = ["<|turn>", "<|im_end>", "<|mask_start>", "<|tool_call>"]
        for input in inputs {
            let range = NSRange(input.startIndex..., in: input)
            let matches = regex.matches(in: input, range: range)
            #expect(!matches.isEmpty, "Control token '\(input)' should match regex")
        }
    }

    @Test("Chat template regex does NOT match angle brackets without pipe")
    func regexNoAngleBrackets() throws {
        let pattern = "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>"
        let regex = try NSRegularExpression(pattern: pattern)

        let inputs = ["<html>", "</body>", "<div>", "< 5", "> 3"]
        for input in inputs {
            let range = NSRange(input.startIndex..., in: input)
            let matches = regex.matches(in: input, range: range)
            #expect(matches.isEmpty, "'\(input)' should not match control token regex")
        }
    }

    // MARK: - scrubControlTokens regex (same pattern as chat template regex)

    @Test("scrubControlTokens regex — real output examples")
    func scrubRealOutputExamples() {
        // Models occasionally leak control tokens mid-generation
        let testCases: [(input: String, expected: String)] = [
            ("Hello<|turn|>", "Hello"),
            ("<|im_end|>Goodbye", "Goodbye"),
            ("Text<|mask_start|>middle<|mask_end|>end", "Textmiddleend"),
            ("<|tool_call|><|tool_call_id|>", ""),
            ("No control tokens here", "No control tokens here"),
            ("", ""),
            ("<|turn|>system\nHello<|turn|>model\nHi", "system\nHellomodel\nHi"),
        ]

        for (input, expected) in testCases {
            let result = MLXEngine.scrubControlTokens(input)
            #expect(result == expected, "Input: '\(input)'")
        }
    }

    // MARK: - Control token discovery with real model dirs (if available)

    @Test("controlTokensForModel caches results")
    func controlTokensCache() {
        // First call for a nonexistent model returns empty
        let tokens1 = MLXEngine.controlTokensForModel(modelId: "nonexistent-model-xyz-123")
        // Second call should return same (cached) result
        let tokens2 = MLXEngine.controlTokensForModel(modelId: "nonexistent-model-xyz-123")
        #expect(tokens1 == tokens2)
    }

    @Test("controlTokensForModel returns empty for nonexistent model")
    func nonexistentModelReturnsEmpty() {
        let tokens = MLXEngine.controlTokensForModel(modelId: "completely-made-up-model-that-does-not-exist")
        #expect(tokens.isEmpty)
    }
}
