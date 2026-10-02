import Foundation

enum PlaygroundPage {
    static var html: String {
        if let url = Bundle.module.url(forResource: "playground", withExtension: "html", subdirectory: "Resources"),
           let text = try? String(contentsOf: url, encoding: .utf8),
           !text.isEmpty
        {
            return text
        }
        return """
        <!doctype html><meta charset="utf-8"><title>Playground missing</title>
        <p>The playground page was not bundled.</p>
        """
    }
}
