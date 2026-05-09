import Foundation
import NovaMLXCore

/// Computes and caches per-model capabilities by inspecting on-disk model files.
/// Thread-safe via lock. Template parsing happens once per model; subsequent
/// calls hit the cache.
final class ModelCapabilitiesDetector: @unchecked Sendable {
    private var cache: [String: ModelCapabilities] = [:]
    private let lock = NSLock()

    func capabilities(for modelId: String, modelType: ModelType, localURL: URL) -> ModelCapabilities {
        lock.lock()
        if let hit = cache[modelId] {
            lock.unlock()
            return hit
        }
        lock.unlock()

        let caps = compute(modelType: modelType, localURL: localURL)

        lock.lock()
        cache[modelId] = caps
        lock.unlock()
        return caps
    }

    func invalidate(_ modelId: String) {
        lock.lock()
        cache.removeValue(forKey: modelId)
        lock.unlock()
    }

    // MARK: - Detection

    private func compute(modelType: ModelType, localURL: URL) -> ModelCapabilities {
        let template = loadChatTemplate(from: localURL) ?? ""

        let vision = modelType == .vlm
        let tools = Self.detectTools(template: template)
        let thinking = Self.detectImplicitThinking(template: template)
        let reasoning = Self.detectReasoning(template: template, thinking: thinking)
        let audio = modelType == .audio
        let imageGeneration = modelType == .image

        return ModelCapabilities(
            reasoning: reasoning,
            thinking: thinking,
            tools: tools,
            vision: vision,
            audio: audio,
            imageGeneration: imageGeneration
        )
    }

    /// Load chat_template from tokenizer_config.json (Jinja string, not rendered).
    private func loadChatTemplate(from modelDir: URL) -> String? {
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: tcPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        if let str = json["chat_template"] as? String {
            return str
        }
        if let arr = json["chat_template"] as? [[String: Any]],
           let first = arr.first,
           let tmpl = first["template"] as? String {
            return tmpl
        }
        return nil
    }

    // MARK: - Static detection (testable without file I/O)

    /// Deterministic: template references tool_call or <tools> Jinja blocks.
    static func detectTools(template: String) -> Bool {
        template.contains("tool_call") || template.contains("<tools>")
    }

    /// Mirrors ModelContainer.isImplicitThinkingModel - true when the template
    /// injects an opening think tag into the generation prompt.
    static func detectImplicitThinking(template: String) -> Bool {
        let injectionLiterals: [String] = [
            "'<think>\\n'",
            "\"<think>\\n\"",
            "'<thinking>\\n'",
            "\"<thinking>\\n\"",
            "'<think>'",
            "\"<think>\"",
            "'<thinking>'",
            "\"<thinking>\"",
            "'<think\\n'",
            "\"<think\\n\""
        ]
        if injectionLiterals.contains(where: { template.contains($0) }) {
            return true
        }
        if template.contains("<think>\n") || template.contains("<thinking>\n") || template.contains("<think\n") {
            return true
        }
        return false
    }

    /// A model supports reasoning_effort if its template references any thinking
    /// markers (implicit injection OR explicit think-tag handling).
    static func detectReasoning(template: String, thinking: Bool) -> Bool {
        if thinking { return true }
        if template.contains("<think") || template.contains("<thinking") {
            return true
        }
        if template.contains("begin_of_thought") || template.contains("end_of_thought") {
            return true
        }
        return false
    }
}
