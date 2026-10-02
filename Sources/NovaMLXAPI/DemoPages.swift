import Foundation
import Hummingbird

/// Demo pages are ONE PER MODEL TYPE (/demo/chat, /demo/vlm, /demo/image,
/// /demo/audio, /demo/decision, /demo/embed), each with an in-page model
/// switcher. Legacy paths (/demo/playground, /demo/qwen-image, /demo/laya)
/// keep serving the same pages.
enum DemoPage {
    static func html(resource: String) -> String {
        if let url = Bundle.module.url(forResource: resource, withExtension: "html", subdirectory: "Resources"),
           let text = try? String(contentsOf: url, encoding: .utf8),
           !text.isEmpty
        {
            return text
        }
        return """
        <!doctype html><meta charset="utf-8"><title>Demo missing</title>
        <p>The \(resource) page was not bundled.</p>
        """
    }
}

extension NovaMLXAPIServer {
    /// Serve a bundled demo page as UTF-8 HTML.
    static func demoResponse(_ resource: String) -> Response {
        Response(
            status: .ok,
            headers: [.contentType: "text/html; charset=utf-8"],
            body: .init(byteBuffer: ByteBuffer(string: DemoPage.html(resource: resource)))
        )
    }
}
