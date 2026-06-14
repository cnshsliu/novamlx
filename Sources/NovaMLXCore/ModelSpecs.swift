import Foundation

public struct ModelFamilySpec: Sendable {
    public let family: String
    public let contextWindow: Int
    public let displayName: String

    public init(family: String, contextWindow: Int, displayName: String) {
        self.family = family
        self.contextWindow = contextWindow
        self.displayName = displayName
    }
}

public enum ModelSpecs {
    public static let defaultContextWindow = 32_768

    public static let families: [ModelFamilySpec] = [
        // Specific patterns first, general patterns after
        ModelFamilySpec(family: "deepseek-v4-pro", contextWindow: 1_000_000, displayName: "DeepSeek V4 Pro"),
        ModelFamilySpec(family: "deepseek-v4", contextWindow: 1_000_000, displayName: "DeepSeek V4"),
        ModelFamilySpec(family: "deepseek-r1", contextWindow: 128_000, displayName: "DeepSeek R1"),
        ModelFamilySpec(family: "deepseek-chat", contextWindow: 128_000, displayName: "DeepSeek Chat"),
        ModelFamilySpec(family: "deepseek", contextWindow: 128_000, displayName: "DeepSeek"),

        ModelFamilySpec(family: "glm-5", contextWindow: 1_000_000, displayName: "GLM 5"),
        ModelFamilySpec(family: "glm-4", contextWindow: 128_000, displayName: "GLM 4"),
        ModelFamilySpec(family: "glm", contextWindow: 128_000, displayName: "GLM"),

        ModelFamilySpec(family: "gpt-4.5", contextWindow: 512_000, displayName: "GPT-4.5"),
        ModelFamilySpec(family: "gpt-4o", contextWindow: 128_000, displayName: "GPT-4o"),
        ModelFamilySpec(family: "gpt-4-turbo", contextWindow: 128_000, displayName: "GPT-4 Turbo"),
        ModelFamilySpec(family: "gpt-4", contextWindow: 128_000, displayName: "GPT-4"),
        ModelFamilySpec(family: "gpt-3.5", contextWindow: 16_385, displayName: "GPT-3.5"),

        ModelFamilySpec(family: "claude-opus", contextWindow: 200_000, displayName: "Claude Opus"),
        ModelFamilySpec(family: "claude-sonnet", contextWindow: 200_000, displayName: "Claude Sonnet"),
        ModelFamilySpec(family: "claude-haiku", contextWindow: 200_000, displayName: "Claude Haiku"),
        ModelFamilySpec(family: "claude", contextWindow: 200_000, displayName: "Claude"),

        ModelFamilySpec(family: "llama-4", contextWindow: 1_000_000, displayName: "Llama 4"),
        ModelFamilySpec(family: "llama-3.3", contextWindow: 128_000, displayName: "Llama 3.3"),
        ModelFamilySpec(family: "llama-3", contextWindow: 128_000, displayName: "Llama 3"),
        ModelFamilySpec(family: "llama", contextWindow: 128_000, displayName: "Llama"),

        ModelFamilySpec(family: "qwen3", contextWindow: 128_000, displayName: "Qwen 3"),
        ModelFamilySpec(family: "qwen2.5", contextWindow: 128_000, displayName: "Qwen 2.5"),
        ModelFamilySpec(family: "qwen", contextWindow: 128_000, displayName: "Qwen"),

        ModelFamilySpec(family: "gemini-2.5", contextWindow: 1_000_000, displayName: "Gemini 2.5"),
        ModelFamilySpec(family: "gemini-2", contextWindow: 1_000_000, displayName: "Gemini 2"),
        ModelFamilySpec(family: "gemini", contextWindow: 1_000_000, displayName: "Gemini"),

        ModelFamilySpec(family: "grok-3", contextWindow: 128_000, displayName: "Grok 3"),
        ModelFamilySpec(family: "grok", contextWindow: 128_000, displayName: "Grok"),

        ModelFamilySpec(family: "mistral-large", contextWindow: 128_000, displayName: "Mistral Large"),
        ModelFamilySpec(family: "mistral-medium", contextWindow: 32_000, displayName: "Mistral Medium"),
        ModelFamilySpec(family: "mistral-small", contextWindow: 32_000, displayName: "Mistral Small"),
        ModelFamilySpec(family: "mistral", contextWindow: 32_000, displayName: "Mistral"),

        ModelFamilySpec(family: "mixtral", contextWindow: 32_000, displayName: "Mixtral"),
    ]

    public static func contextWindow(for modelName: String) -> Int {
        let lower = modelName.lowercased()
        for spec in families {
            if lower.contains(spec.family) {
                return spec.contextWindow
            }
        }
        return defaultContextWindow
    }

    public static func familyName(for modelName: String) -> String? {
        let lower = modelName.lowercased()
        for spec in families {
            if lower.contains(spec.family) {
                return spec.displayName
            }
        }
        return nil
    }

    public static func lbContextWindow(from providers: [TokenhubProvider]) -> (window: Int, mixed: Bool) {
        // Post-Task-6: includeInLoadBalance flag is gone — every enabled
        // provider is a candidate for the LB pool.
        let lbProviders = providers.filter { $0.isEnabled }
        guard !lbProviders.isEmpty else { return (defaultContextWindow, false) }

        let windows = lbProviders.map { $0.effectiveContextWindow }
        let minWindow = windows.min() ?? defaultContextWindow
        let maxWindow = windows.max() ?? defaultContextWindow
        return (minWindow, minWindow != maxWindow)
    }
}
