import Foundation
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils
import Tokenizers

private struct TokenizerBridge: MLXLMCommon.Tokenizer {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        upstream.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        upstream.convertIdToToken(id)
    }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        // The 3-arg Tokenizers API always sets addGenerationPrompt=true and
        // only then merges additionalContext into Jinja. Honor an explicit
        // flag on the Swift parameter so GDN session continuation can render
        // a closed prefix (no trailing assistant header).
        var addGenerationPrompt = true
        var context = additionalContext
        if let additionalContext,
            let flag = additionalContext["add_generation_prompt"] as? Bool
        {
            addGenerationPrompt = flag
            context = additionalContext.filter { $0.key != "add_generation_prompt" }
        }
        do {
            return try upstream.applyChatTemplate(
                messages: messages,
                chatTemplate: nil,
                addGenerationPrompt: addGenerationPrompt,
                truncation: false,
                maxLength: nil,
                tools: tools,
                additionalContext: context)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}

final class LocalTokenizerLoader: MLXLMCommon.TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        if FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path)
        {
            let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
            return TokenizerBridge(upstream)
        }
        if isDFlashDraftConfig(at: directory) {
            let fallbacks = [
                "mlx-community/Qwen3.8-27B-8bit",
                "mlx-community/Qwen3.8-27B-4bit",
                "orcarouter/Qwen3.8-27B-Uncensored-MLX",
            ]
            for id in fallbacks {
                let fb = NovaMLXPaths.modelsDir.appendingPathComponent(id)
                if FileManager.default.fileExists(
                    atPath: fb.appendingPathComponent("tokenizer.json").path)
                {
                    NovaMLXLog.info("[Tokenizer] DFlash drafter has no tokenizer; using \(id)")
                    let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: fb)
                    return TokenizerBridge(upstream)
                }
            }
        }
        let upstream = try await Tokenizers.AutoTokenizer.from(modelFolder: directory)
        return TokenizerBridge(upstream)
    }
}
