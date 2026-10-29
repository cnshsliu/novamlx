import Foundation

/// swift-jinja rejects numeric attribute access (`content.0.type`).
/// GLM chat templates use that form for list items. `[0]` is the same lookup.
enum GlmJinja {
    /// Rewrites `.12` after an identifier, `]`, or `)` into `[12]`.
    /// A digit before the dot (`1.0`) is left alone.
    static func rewriteNumericIndexes(_ template: String) -> (String, Int) {
        guard let regex = try? NSRegularExpression(
            pattern: #"(?<=[A-Za-z_)\])])\.([0-9]+)"#
        ) else {
            return (template, 0)
        }
        let range = NSRange(template.startIndex..., in: template)
        let count = regex.numberOfMatches(in: template, range: range)
        guard count > 0 else { return (template, 0) }
        let replaced = regex.stringByReplacingMatches(
            in: template, range: range, withTemplate: #"[$1]"#)
        return (replaced, count)
    }
}
