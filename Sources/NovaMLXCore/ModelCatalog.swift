import Foundation

public enum CatalogFormat: String, Codable, Sendable, Equatable, CaseIterable {
    case mlx
    case gguf
}

public enum CatalogStatus: String, Codable, Sendable, Equatable, CaseIterable {
    case verified
    case preview
}

public struct CatalogEntry: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let url: String
    public let name: String
    public let category: ModelType
    public let family: ModelFamily
    public let format: CatalogFormat
    public let description: String?
    public let revision: String?
    public let quant: String?
    public let size: String?
    public let sizeBytes: UInt64?
    public let minRamGB: Int?
    public let tags: [String]
    public let capabilities: [String]
    public let testedOn: String?
    public let status: CatalogStatus
    /// When this row was added to the catalog (ISO-8601). Newest-first browse.
    public let addedAt: String?

    public init(
        id: String,
        url: String,
        name: String,
        category: ModelType,
        family: ModelFamily,
        format: CatalogFormat,
        description: String? = nil,
        revision: String? = nil,
        quant: String? = nil,
        size: String? = nil,
        sizeBytes: UInt64? = nil,
        minRamGB: Int? = nil,
        tags: [String] = [],
        capabilities: [String] = [],
        testedOn: String? = nil,
        status: CatalogStatus = .verified,
        addedAt: String? = nil
    ) {
        self.id = id
        self.url = url
        self.name = name
        self.category = category
        self.family = family
        self.format = format
        self.description = description
        self.revision = revision
        self.quant = quant
        self.size = size
        self.sizeBytes = sizeBytes
        self.minRamGB = minRamGB
        self.tags = tags
        self.capabilities = capabilities
        self.testedOn = testedOn
        self.status = status
        self.addedAt = addedAt
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        url = try c.decode(String.self, forKey: .url)
        name = try c.decode(String.self, forKey: .name)
        category = try c.decode(ModelType.self, forKey: .category)
        family = try c.decode(ModelFamily.self, forKey: .family)
        format = try c.decode(CatalogFormat.self, forKey: .format)
        description = try c.decodeIfPresent(String.self, forKey: .description)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        quant = try c.decodeIfPresent(String.self, forKey: .quant)
        size = try c.decodeIfPresent(String.self, forKey: .size)
        sizeBytes = try c.decodeIfPresent(UInt64.self, forKey: .sizeBytes)
        minRamGB = try c.decodeIfPresent(Int.self, forKey: .minRamGB)
        tags = try c.decodeIfPresent([String].self, forKey: .tags) ?? []
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
        testedOn = try c.decodeIfPresent(String.self, forKey: .testedOn)
        status = try c.decodeIfPresent(CatalogStatus.self, forKey: .status) ?? .verified
        addedAt = try c.decodeIfPresent(String.self, forKey: .addedAt)
    }
}

public struct CatalogFile: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let updatedAt: String?
    public let models: [CatalogEntry]

    public init(schemaVersion: Int, updatedAt: String? = nil, models: [CatalogEntry]) {
        self.schemaVersion = schemaVersion
        self.updatedAt = updatedAt
        self.models = models
    }

    public static func decode(_ data: Data) throws -> CatalogFile {
        try JSONDecoder().decode(CatalogFile.self, from: data)
    }
}

public enum ModelCatalogPolicy {
    /// `owner/prefix*` family allowlist, e.g. `mlx-community/Qwen3.8-*`.
    /// Trailing `*` only; the repo stem before `*` must be non-empty, so
    /// `mlx-community/*` is rejected.
    public static func isIdPattern(_ id: String) -> Bool {
        guard id.hasSuffix("*"), id.filter({ $0 == "*" }).count == 1 else { return false }
        let prefix = String(id.dropLast())
        let parts = prefix.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2 else { return false }
        return !parts[0].isEmpty && !parts[1].isEmpty
    }

    public static func containsIdPattern(_ catalog: [CatalogEntry]) -> Bool {
        catalog.contains { isIdPattern($0.id) }
    }

    /// True when `candidate` is the catalog id itself, or matches a family glob.
    public static func idMatches(_ pattern: String, candidate: String) -> Bool {
        if !isIdPattern(pattern) {
            return pattern == candidate
        }
        if isIdPattern(candidate) { return false }
        return candidate.hasPrefix(String(pattern.dropLast()))
    }

    /// True when the typed query is about a family glob, so Hub may expand
    /// variants (`qwen3.8`, `mlx-community/Qwen3.8-27B-4bit`). Unrelated
    /// queries (`ornith`) stay on the local catalog.
    public static func shouldExpandFamilyGlobs(query: String, catalog: [CatalogEntry]) -> Bool {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty, containsIdPattern(catalog) else { return false }
        let tokens = queryTokens(needle)
        return catalog.contains { entry in
            guard isIdPattern(entry.id) else { return false }
            if entry.id.lowercased().contains(needle) { return true }
            if entry.name.lowercased().contains(needle) { return true }
            if matchesTokens(tokens, entry: entry) { return true }
            let stem = hubSearchQuery(forPattern: entry.id).lowercased()
            if !stem.isEmpty && (stem.contains(needle) || needle.contains(stem)) { return true }
            let prefix = String(entry.id.dropLast()).lowercased()
            return needle.hasPrefix(prefix)
        }
    }

    /// Hub search string for a family glob: `mlx-community/Qwen3.8-*` → `Qwen3.8`.
    public static func hubSearchQuery(forPattern id: String) -> String {
        let raw = isIdPattern(id) ? String(id.dropLast()) : id
        guard let slash = raw.firstIndex(of: "/") else { return raw }
        var stem = String(raw[raw.index(after: slash)...])
        while stem.hasSuffix("-") { stem.removeLast() }
        return stem.isEmpty ? raw : stem
    }

    public static func isDownloadAllowed(
        id: String,
        catalog: [CatalogEntry],
        allowUnlisted: Bool
    ) -> Bool {
        if allowUnlisted { return true }
        if isIdPattern(id) { return false }
        return catalog.contains { idMatches($0.id, candidate: id) }
    }

    public static func entry(id: String, in catalog: [CatalogEntry]) -> CatalogEntry? {
        if let exact = catalog.first(where: { $0.id == id }) {
            return exact
        }
        return catalog
            .filter { isIdPattern($0.id) && idMatches($0.id, candidate: id) }
            .max(by: { $0.id.count < $1.id.count })
    }

    public static func refuseMessage(id: String) -> String {
        "\(id) is not in the NovaMLX verified catalog. Turn on Download Models → Allow unverified downloads if you want to try it anyway."
    }

    public static func search(
        _ catalog: [CatalogEntry],
        query: String,
        category: ModelType?
    ) -> [CatalogEntry] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let needle = trimmed.lowercased()
        let tokens = queryTokens(needle)

        let filtered = catalog.enumerated().filter { _, entry in
            if let category, entry.category != category {
                return false
            }
            if needle.isEmpty {
                return true
            }
            if entry.id.lowercased().contains(needle) { return true }
            if entry.name.lowercased().contains(needle) { return true }
            if let description = entry.description, description.lowercased().contains(needle) {
                return true
            }
            if entry.family.rawValue.lowercased().contains(needle) { return true }
            if entry.tags.contains(where: { $0.lowercased().contains(needle) }) { return true }
            return matchesTokens(tokens, entry: entry)
        }

        // Newest `addedAt` first; undated rows keep file order after dated ones.
        return filtered.sorted { lhs, rhs in
            let ld = addedAtDate(lhs.element.addedAt)
            let rd = addedAtDate(rhs.element.addedAt)
            if ld != rd { return ld > rd }
            return lhs.offset < rhs.offset
        }.map(\.element)
    }

    /// Split a query into lowercase tokens so `qwen3.8 flash` matches an
    /// entry that contains both words across id / name / description / tags.
    static func queryTokens(_ needle: String) -> [String] {
        needle.split { $0.isWhitespace || $0 == "," }.map(String.init).filter { !$0.isEmpty }
    }

    static func searchableText(for entry: CatalogEntry) -> String {
        var parts = [entry.id, entry.name, entry.family.rawValue]
        if let description = entry.description { parts.append(description) }
        parts.append(contentsOf: entry.tags)
        return parts.joined(separator: " ").lowercased()
    }

    static func matchesTokens(_ tokens: [String], entry: CatalogEntry) -> Bool {
        guard !tokens.isEmpty else { return false }
        let haystack = searchableText(for: entry)
        return tokens.allSatisfy { haystack.contains($0) }
    }

    static func addedAtDate(_ raw: String?) -> Date {
        guard let raw, !raw.isEmpty else { return .distantPast }
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        if let date = iso.date(from: raw) { return date }
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return iso.date(from: raw) ?? .distantPast
    }

    /// Hub-level heuristic: keep repos that advertise MLX, drop GGUF/transformers-only.
    /// Not a substitute for the verified catalog — used when Advanced Hub search is on.
    public static func looksLikeMLXRepo(id: String, tags: [String]) -> Bool {
        let lowerID = id.lowercased()
        let lowerTags = tags.map { $0.lowercased() }
        if lowerID.contains("mlx") { return true }
        if lowerTags.contains(where: { $0 == "mlx" || $0.contains("mlx") }) { return true }
        return false
    }
}
