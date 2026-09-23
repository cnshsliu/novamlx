import Foundation

/// Image checkpoints NovaMLX will not load or advertise.
public enum ImageModelSupport: Sendable {
    public static let unsupportedIds: Set<String> = [
        "mzbac/flux1.schnell.4bit.mlx",
    ]

    public static func isUnsupported(_ id: String) -> Bool {
        if unsupportedIds.contains(id) { return true }
        let lower = id.lowercased()
        return lower.contains("flux1.schnell")
            || lower.contains("flux.1-schnell")
            || lower.contains("flux1-schnell")
    }

    public static func refuseMessage(_ id: String) -> String {
        "\(id) is not supported. FLUX.1 schnell crashes in VAE decode. Use black-forest-labs/FLUX.2-klein-4B, Qwen/Qwen-Image, Qwen/Qwen-Image-2.1, or mzbac/Z-Image-Turbo-8bit."
    }
}

/// Qwen-Image-2.1 is a separate checkpoint from Qwen-Image 1.x.
public enum QwenImage21Support: Sendable {
    public static func matches(id: String) -> Bool {
        let lower = id.lowercased()
        let tokens = [
            "qwen-image-2.1",
            "qwen-image-21",
            "qwen_image_2.1",
            "qwen_image_21",
            "qwenimage21",
            "qwen-image-2-1",
        ]
        return tokens.contains { lower.contains($0) }
    }
}
