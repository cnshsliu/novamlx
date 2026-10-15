import Foundation

/// API-key references into the `env` file beside `peer.json`.
///
/// A stored key whose text is ALL UPPERCASE (`OPENROUTER_KEY`, `_FOO`) is
/// treated as the *name* of a variable in `~/.config/tknet-peer/env`, not as
/// the key itself. Anything else — `sk-…`, `gsk_…`, mixed case — is a literal
/// key. The env file is re-read on every resolution so edits apply without a
/// restart.
///
/// File format: one `KEY=VALUE` per line. `#` comments and blank lines are
/// skipped. Surrounding whitespace and matched single/double quotes on the
/// value are stripped.
public enum EnvKeyFile {
    /// `TKNETPEER_CONFIG_DIR` overrides the whole directory, matching
    /// PeerAppState and the UI-test isolation convention.
    public static var defaultDirectory: URL {
        if let override = ProcessInfo.processInfo.environment["TKNETPEER_CONFIG_DIR"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/tknet-peer", isDirectory: true)
    }

    /// Default location of the env file.
    public static var defaultURL: URL {
        defaultDirectory.appendingPathComponent("env")
    }

    /// True when `value` looks like an env variable name: uppercase letters,
    /// digits, and underscores only, with at least one letter. Empty strings
    /// and pure digits are not references.
    public static func isReference(_ value: String) -> Bool {
        guard !value.isEmpty else { return false }
        var hasLetter = false
        for scalar in value.unicodeScalars {
            switch scalar {
            case "A"..."Z": hasLetter = true
            case "0"..."9", "_": continue
            default: return false
            }
        }
        return hasLetter
    }

    /// Parses `KEY=VALUE` text. Later duplicates win.
    public static func parse(_ text: String) -> [String: String] {
        var map: [String: String] = [:]
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            guard let eq = line.firstIndex(of: "=") else { continue }
            let name = String(line[..<eq]).trimmingCharacters(in: .whitespaces)
            guard !name.isEmpty else { continue }
            map[name] = stripQuotes(String(line[line.index(after: eq)...]).trimmingCharacters(in: .whitespaces))
        }
        return map
    }

    /// Reads and parses the env file. Missing or unreadable file is an empty
    /// map — a first run just has no references resolvable yet.
    public static func load(url: URL) -> [String: String] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [:] }
        return parse(text)
    }

    /// Resolves a stored key value. Literal keys pass through unchanged (nil
    /// when empty). A reference returns the env value, or nil when the name is
    /// not defined — the caller must surface that to the user.
    public static func resolve(_ value: String, url: URL) -> String? {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard isReference(trimmed) else { return trimmed }
        return load(url: url)[trimmed]
    }

    private static func stripQuotes(_ value: String) -> String {
        guard value.count >= 2 else { return value }
        let first = value.first, last = value.last
        if (first == "\"" && last == "\"") || (first == "'" && last == "'") {
            return String(value.dropFirst().dropLast())
        }
        return value
    }
}
