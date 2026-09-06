import Foundation

/// Hidden operator gate. The catalog admin UI is shown only when this file
/// exists in the user's home directory.
public enum CatalogAdminGate: Sendable {
    public static let flagFileName = "liukehongistheking.txt"

    public static func isEnabled(
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> Bool {
        let url = home.appendingPathComponent(flagFileName)
        return FileManager.default.fileExists(atPath: url.path)
    }
}

public enum CatalogAdminError: Error, LocalizedError, Equatable {
    case invalid(String)
    case repoNotFound
    case gitFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let msg): return msg
        case .repoNotFound:
            return "Could not find catalog/models.json. Set NOVAMLX_REPO to the git checkout."
        case .gitFailed(let msg): return msg
        }
    }
}

extension CatalogFile {
    public static let catalogCapabilities = [
        "tools", "vision", "thinking", "audio", "imageGeneration",
    ]

    public static func utcNow() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        var raw = formatter.string(from: Date())
        if raw.contains(".") {
            raw = raw.replacingOccurrences(
                of: #"\.\d+Z$"#, with: "Z", options: .regularExpression
            )
        }
        return raw
    }

    public func encodedPretty() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
        var data = try encoder.encode(self)
        if data.last != 0x0A { data.append(0x0A) }
        return data
    }

    /// Same rules as `catalog/admin.py` `validate()`.
    public func validate() -> [String] {
        var errors: [String] = []
        if schemaVersion != 1 {
            errors.append("schemaVersion must be 1")
        }
        var seen = Set<String>()
        for (i, entry) in models.enumerated() {
            let prefix = "models[\(i)]"
            if entry.id.trimmingCharacters(in: .whitespaces).isEmpty {
                errors.append("\(prefix).id is required")
            }
            if entry.url.trimmingCharacters(in: .whitespaces).isEmpty {
                errors.append("\(prefix).url is required")
            }
            if entry.name.trimmingCharacters(in: .whitespaces).isEmpty {
                errors.append("\(prefix).name is required")
            }
            if seen.contains(entry.id) {
                errors.append("duplicate id: \(entry.id)")
            }
            seen.insert(entry.id)
            if entry.id.contains("*"), !ModelCatalogPolicy.isIdPattern(entry.id) {
                errors.append(
                    "\(prefix).id family pattern must be owner/prefix* with a non-empty repo stem (e.g. mlx-community/Qwen3.8-*)"
                )
            }
            if let added = entry.addedAt, added.trimmingCharacters(in: .whitespaces).count < 10 {
                errors.append("\(prefix).addedAt must be an ISO-8601 string")
            }
        }
        return errors
    }

    public func validated() throws -> CatalogFile {
        let errors = validate()
        if !errors.isEmpty {
            throw CatalogAdminError.invalid(errors.joined(separator: "\n"))
        }
        return self
    }
}

extension CatalogEntry {
    public static func defaultURL(forId id: String) -> String {
        if ModelCatalogPolicy.isIdPattern(id) {
            let prefix = String(id.dropLast())
            guard let slash = prefix.firstIndex(of: "/") else {
                return "https://huggingface.co/models?search=\(id)"
            }
            let owner = String(prefix[..<slash])
            var stem = String(prefix[prefix.index(after: slash)...])
            while stem.hasSuffix("-") { stem.removeLast() }
            var comps = URLComponents(string: "https://huggingface.co/models")!
            comps.queryItems = [
                URLQueryItem(name: "search", value: stem),
                URLQueryItem(name: "author", value: owner),
            ]
            return comps.string ?? "https://huggingface.co/models"
        }
        return "https://huggingface.co/\(id)"
    }

    public static func guessedQuant(fromId id: String) -> String? {
        let lower = id.lowercased()
        if lower.contains("mxfp4") { return "mxfp4" }
        if lower.contains("6bit") { return "6bit" }
        if lower.contains("8bit") || lower.contains("q8") { return "8bit" }
        if lower.contains("4bit") || lower.contains("q4") { return "4bit" }
        if lower.contains("2bit") { return "2bit" }
        if lower.contains("fp16") { return "fp16" }
        return nil
    }

    public static func displaySize(bytes: UInt64) -> String? {
        guard bytes > 0 else { return nil }
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1 {
            return String(format: "~%.1f GB", gb)
        }
        let mb = Double(bytes) / 1_048_576
        return String(format: "~%.0f MB", mb)
    }

    /// Build a verified-catalog row from a local model the operator has tested.
    public static func verifiedDraft(
        id: String,
        url: String,
        name: String? = nil,
        category: ModelType,
        family: ModelFamily,
        format: CatalogFormat = .mlx,
        sizeBytes: UInt64 = 0,
        description: String? = nil
    ) -> CatalogEntry {
        let trimmed = id.trimmingCharacters(in: .whitespaces)
        let resolvedURL = url.trimmingCharacters(in: .whitespaces).isEmpty
            ? defaultURL(forId: trimmed)
            : url
        let display = name?.trimmingCharacters(in: .whitespaces).nilIfEmpty
            ?? trimmed.split(separator: "/").last.map(String.init)
            ?? trimmed
        return CatalogEntry(
            id: trimmed,
            url: resolvedURL,
            name: display,
            category: category,
            family: family,
            format: format,
            description: description,
            quant: guessedQuant(fromId: trimmed),
            size: displaySize(bytes: sizeBytes),
            sizeBytes: sizeBytes > 0 ? sizeBytes : nil,
            tags: ["MLX"],
            capabilities: [],
            testedOn: version,
            status: .verified,
            addedAt: CatalogFile.utcNow()
        )
    }
}

private extension String {
    var nilIfEmpty: String? {
        let t = trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }
}
