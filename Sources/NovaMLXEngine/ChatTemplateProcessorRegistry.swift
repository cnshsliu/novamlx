import Foundation
import NovaMLXCore

/// Factory: (ModelFamily, detected template format) → ChatTemplateProcessor.
/// Family-first selection with format as discriminator within family.
enum ChatTemplateProcessorRegistry {

    static func processor(family: ModelFamily, chatTemplate: String?) -> ChatTemplateProcessor {
        let format = ChatTemplateFormat.detect(from: chatTemplate)

        switch (family, format) {
        // Qwen: split by format — ChatML (2/3/3.5) vs turnRole (3.6+)
        case (.qwen, .turnRole):    return QwenTurnRoleProcessor()
        case (.qwen, .imStartEnd):  return QwenChatMLProcessor()
        case (.qwen, _):            return QwenChatMLProcessor()

        // Gemma: all Gemma versions use startOfTurn format
        case (.gemma, _):           return GemmaProcessor()

        // Bailing: uses <|turn>role\n format (Ling series)
        case (.bailing, _):         return BailingProcessor()

        // GPT-OSS: uses <|start|>role<|channel|>type<|message|> format (OpenAI Harmony)
        case (.gptOss, _):          return HarmonyProcessor()

        // Unknown families (llama, mistral, phi, starcoder, claude, other):
        // format-based fallback. Future family-specific processors go here.
        default:                    return DefaultProcessor(format: format)
        }
    }
}
