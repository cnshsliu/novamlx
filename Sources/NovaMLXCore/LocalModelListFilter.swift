import Foundation

/// Tri-state facet for a Local Inference list attribute.
/// `any` keeps every row. `yes` and `no` keep rows where the attribute matches.
public enum LocalModelFacet: String, Sendable, Equatable, CaseIterable {
    case any
    case yes
    case no
}

/// Downloaded / verified / loaded flags for one Local Inference row.
///
/// `isLoaded` means the weights are resident in memory. On-disk draft
/// accelerators stay on `companionReady` and are not part of this row.
public struct LocalModelRowState: Sendable, Equatable, Identifiable {
    public let id: String
    public let isDownloaded: Bool
    public let isVerified: Bool
    public let isLoaded: Bool

    public init(id: String, isDownloaded: Bool, isVerified: Bool, isLoaded: Bool) {
        self.id = id
        self.isDownloaded = isDownloaded
        self.isVerified = isVerified
        self.isLoaded = isLoaded
    }

    /// Draw the loaded badge only while the model is resident.
    public var showsLoadedBadge: Bool { isLoaded }
}

public enum LocalModelListFilter {
    /// Exact catalog weight whose status is not `unsupported`.
    /// A family glob is an allowlist, not a tested weight, so it does not count.
    public static func isVerifiedWeight(id: String, catalog: [CatalogEntry]) -> Bool {
        catalog.contains { entry in
            entry.id == id
                && entry.status != .unsupported
                && !ModelCatalogPolicy.isIdPattern(entry.id)
        }
    }

    public static func matches(
        _ row: LocalModelRowState,
        downloaded: LocalModelFacet,
        verified: LocalModelFacet
    ) -> Bool {
        passes(downloaded, value: row.isDownloaded) && passes(verified, value: row.isVerified)
    }

    /// Catalog weights that are not already downloaded, loaded, or parked on an
    /// unmounted disk. Order follows `catalog`. Duplicate ids keep the first row.
    public static func notDownloadedWeights(
        catalog: [CatalogEntry],
        excluding ids: Set<String>
    ) -> [CatalogEntry] {
        var seen = Set<String>()
        return catalog.filter { entry in
            guard entry.status != .unsupported else { return false }
            guard !ModelCatalogPolicy.isIdPattern(entry.id) else { return false }
            guard !ids.contains(entry.id) else { return false }
            return seen.insert(entry.id).inserted
        }
    }

    private static func passes(_ facet: LocalModelFacet, value: Bool) -> Bool {
        switch facet {
        case .any: return true
        case .yes: return value
        case .no: return !value
        }
    }
}
