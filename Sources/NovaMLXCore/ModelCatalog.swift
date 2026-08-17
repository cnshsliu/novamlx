import Foundation

public enum CatalogFormat: String, Codable, Sendable, Equatable {
    case mlx
    case gguf
}

public enum CatalogStatus: String, Codable, Sendable, Equatable {
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
        status: CatalogStatus = .verified
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
    public static func isDownloadAllowed(
        id: String,
        catalog: [CatalogEntry],
        allowUnlisted: Bool
    ) -> Bool {
        if allowUnlisted { return true }
        return catalog.contains { $0.id == id }
    }

    public static func entry(id: String, in catalog: [CatalogEntry]) -> CatalogEntry? {
        catalog.first { $0.id == id }
    }

    public static func refuseMessage(id: String) -> String {
        "\(id) is not in the NovaMLX verified catalog. Turn on Settings → Allow unverified downloads if you want to try it anyway."
    }

    public static func search(
        _ catalog: [CatalogEntry],
        query: String,
        category: ModelType?
    ) -> [CatalogEntry] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        let needle = trimmed.lowercased()

        return catalog.filter { entry in
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
            return false
        }
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
