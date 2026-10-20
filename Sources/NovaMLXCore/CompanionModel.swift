import Foundation

/// Speculative companions (DSpark, DFlash / DFlash2, MTP heads) that accelerate
/// another checkpoint and are not standalone chat, VLM, audio, or image models.
///
/// Name checks go through `ResourceLimits.isCompanionDraftModelId`. Catalog rows
/// use the `Companion` tag (a bare `MTP` / `DSpark` / `DFlash` tag is not enough:
/// backbones carry those too). On-disk draft configs are reported by the caller
/// via `configSaysDraft` so this type stays free of weight-file I/O.
public enum CompanionModel {
    public static func isAuxiliary(
        id: String,
        catalog: [CatalogEntry] = [],
        configSaysDraft: Bool = false
    ) -> Bool {
        if ResourceLimits.isCompanionDraftModelId(id) { return true }
        if configSaysDraft { return true }
        guard let entry = ModelCatalogPolicy.entry(id: id, in: catalog) else { return false }
        return entry.tags.contains { $0.caseInsensitiveCompare("Companion") == .orderedSame }
    }
}
