import Foundation
import NovaMLXCore

extension CompanionModel {
    /// Same predicate as `CompanionModel.isAuxiliary(id:catalog:configSaysDraft:)`,
    /// with the draft bit read from `config.json` in `directory`.
    public static func isAuxiliary(
        id: String,
        catalog: [CatalogEntry] = [],
        directory: URL
    ) -> Bool {
        let draft = isDSparkDraftConfig(at: directory)
            || isDFlashDraftConfig(at: directory)
            || isMtpDraftConfig(at: directory)
        return isAuxiliary(id: id, catalog: catalog, configSaysDraft: draft)
    }
}
