import Foundation

// MARK: - Endpoint Normalizer
//
// Tokenhub accepts ANY endpoint from ANY provider. Users paste what they
// copy: a bare base URL, a base with a trailing slash, a FULL request URL
// that already ends in the target path, a query-stringed gateway URL, or
// a scheme-less host. This turns every one of those into the URL the
// proxy should actually call — appendingPathComponent alone breaks
// query strings and doubles path suffixes.

public enum EndpointNormalizer {

    /// Build the request URL for an endpoint.
    ///
    /// - If the endpoint's path already ends with `suffix` (e.g. the user
    ///   pasted `.../v1/chat/completions` and the suffix is
    ///   `chat/completions`), the URL is used as-is.
    /// - Otherwise the suffix is appended to the path.
    /// - Query items and fragment are preserved (Azure-style
    ///   `?api-version=...`, gateway `?key=...`).
    /// - A scheme-less endpoint is treated as https.
    /// - Trailing slashes are ignored.
    ///
    /// Returns nil for endpoints that cannot be a URL at all.
    public static func url(endpoint: String, suffix: String) -> URL? {
        var raw = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty { return nil }
        if !raw.contains("://") { raw = "https://" + raw }

        // URLComponents handles query/fragment; appendingPathComponent does not.
        guard var comps = URLComponents(string: raw) else { return nil }
        guard let _ = comps.host, let scheme = comps.scheme,
              ["http", "https"].contains(scheme.lowercased()) else { return nil }

        let suffixClean = suffix.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let trimmed = comps.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let path = trimmed.isEmpty ? "" : "/" + trimmed

        if path.hasSuffix("/" + suffixClean) {
            comps.path = path
        } else {
            comps.path = path.isEmpty ? "/" + suffixClean : path + "/" + suffixClean
        }

        return comps.url
    }
}
