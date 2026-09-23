import Foundation

enum QwenImageDemoPage {
    static var html: String {
        if let url = Bundle.module.url(forResource: "qwen-image-demo", withExtension: "html", subdirectory: "Resources"),
           let text = try? String(contentsOf: url, encoding: .utf8),
           !text.isEmpty
        {
            return text
        }
        return """
        <!doctype html><meta charset="utf-8"><title>Qwen-Image demo missing</title>
        <p>The Qwen-Image 2.1 demo page was not bundled.</p>
        """
    }
}
