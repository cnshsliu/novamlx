import Foundation

enum LayaDemoPage {
    static var html: String {
        if let url = Bundle.module.url(forResource: "laya-demo", withExtension: "html", subdirectory: "Resources"),
           let text = try? String(contentsOf: url, encoding: .utf8),
           !text.isEmpty
        {
            return text
        }
        return """
        <!doctype html><meta charset="utf-8"><title>Laya demo missing</title>
        <p>The decision demo page was not bundled.</p>
        """
    }
}
