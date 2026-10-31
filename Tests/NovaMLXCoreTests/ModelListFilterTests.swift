import Foundation
import Testing
@testable import NovaMLXCore

@Suite("Model list filters")
struct ModelListFilterTests {
    private func entry(
        id: String,
        tags: [String] = [],
        status: CatalogStatus = .verified
    ) -> CatalogEntry {
        CatalogEntry(
            id: id,
            url: "https://example.com/\(id)",
            name: id,
            category: .llm,
            family: .qwen,
            format: .mlx,
            tags: tags,
            status: status
        )
    }

    @Test("Known companion ids are auxiliary without a catalog row")
    func companionIds() {
        #expect(CompanionModel.isAuxiliary(id: "incoai/Qwen3.8-27B-DFlash2"))
        #expect(CompanionModel.isAuxiliary(id: "mlx-community/Qwen3.8-27B-MTP-8bit"))
        #expect(CompanionModel.isAuxiliary(id: "mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"))
        #expect(!CompanionModel.isAuxiliary(id: "mlx-community/Qwen3.8-27B-8bit"))
        #expect(!CompanionModel.isAuxiliary(id: "mlx-community/Qwen3-0.6B-4bit"))
    }

    @Test("Catalog Companion tag hides a drafter; MTP or DSpark tags do not hide a backbone")
    func catalogTags() {
        let head = entry(id: "org/custom-spec-head", tags: ["Companion"])
        #expect(CompanionModel.isAuxiliary(id: "org/custom-spec-head", catalog: [head]))
        #expect(!CompanionModel.isAuxiliary(id: "org/custom-spec-head"))

        let backbone = entry(
            id: "mlx-community/DeepSeek-V4.1-Flash",
            tags: ["DSpark", "MTP", "Vision"]
        )
        #expect(!CompanionModel.isAuxiliary(id: backbone.id, catalog: [backbone]))

        let pattern = entry(id: "acme/SpecHead-*", tags: ["Companion", "MTP"])
        #expect(CompanionModel.isAuxiliary(id: "acme/SpecHead-4bit", catalog: [pattern]))
    }

    @Test("Caller-supplied draft config marks an otherwise ordinary id")
    func configBit() {
        #expect(CompanionModel.isAuxiliary(id: "org/plain-head", configSaysDraft: true))
        #expect(!CompanionModel.isAuxiliary(id: "org/plain-head", configSaysDraft: false))
    }

    @Test("Disk menu uses volume names and keeps each configured root")
    func diskLabels() {
        let home = URL(fileURLWithPath: "/Users/me/.nova/models")
        let disk1 = URL(fileURLWithPath: "/Volumes/Disk1/Models")
        let disk2 = URL(fileURLWithPath: "/Volumes/Disk2/Models")
        let options = ModelRootFilter.options(for: [home, disk1, disk2, disk1])
        #expect(options.map(\.label) == ["models", "Disk1", "Disk2"])
        #expect(options.map(\.path) == [home.path, disk1.path, disk2.path])
    }

    @Test("Two folders on one volume stay distinct")
    func collidingVolumeLabels() {
        let models = URL(fileURLWithPath: "/Volumes/Disk1/Models")
        let backup = URL(fileURLWithPath: "/Volumes/Disk1/backup")
        let options = ModelRootFilter.options(for: [models, backup])
        #expect(options.map(\.label) == ["Disk1 (Models)", "Disk1 (backup)"])
    }

    @Test("Same volume and folder name falls back to the full path")
    func identicalDecoratedLabels() {
        let a = URL(fileURLWithPath: "/Volumes/Disk1/a/Models")
        let b = URL(fileURLWithPath: "/Volumes/Disk1/b/Models")
        let options = ModelRootFilter.options(for: [a, b])
        #expect(options.map(\.label) == [a.path, b.path])
    }

    @Test("Verified means an exact catalog weight, not a family glob or unsupported row")
    func verifiedWeight() {
        let verified = entry(id: "mlx-community/Qwen3-0.6B-4bit", status: .verified)
        let preview = entry(id: "mlx-community/Qwen3-1.7B-4bit", status: .preview)
        let unsupported = entry(id: "org/broken", status: .unsupported)
        let glob = CatalogEntry(
            id: "mlx-community/Qwen3-*",
            url: "https://example.com/qwen",
            name: "Qwen3",
            category: .llm,
            family: .qwen,
            format: .mlx,
            status: .verified
        )
        let catalog = [verified, preview, unsupported, glob]
        #expect(LocalModelListFilter.isVerifiedWeight(id: verified.id, catalog: catalog))
        #expect(LocalModelListFilter.isVerifiedWeight(id: preview.id, catalog: catalog))
        #expect(!LocalModelListFilter.isVerifiedWeight(id: unsupported.id, catalog: catalog))
        #expect(!LocalModelListFilter.isVerifiedWeight(id: "mlx-community/Qwen3-8B-4bit", catalog: catalog))
        #expect(!LocalModelListFilter.isVerifiedWeight(id: "org/local-only", catalog: catalog))
    }

    @Test("Downloaded and verified facets narrow the local list")
    func statusFacets() {
        let onDiskVerified = LocalModelRowState(
            id: "org/verified", isDownloaded: true, isVerified: true, isLoaded: true
        )
        let onDiskPlain = LocalModelRowState(
            id: "org/plain", isDownloaded: true, isVerified: false, isLoaded: false
        )
        let catalogOnly = LocalModelRowState(
            id: "org/catalog", isDownloaded: false, isVerified: true, isLoaded: false
        )
        let rows = [onDiskVerified, onDiskPlain, catalogOnly]

        let all = rows.filter { LocalModelListFilter.matches($0, downloaded: .any, verified: .any) }
        #expect(all.map(\.id) == ["org/verified", "org/plain", "org/catalog"])

        let downloaded = rows.filter { LocalModelListFilter.matches($0, downloaded: .yes, verified: .any) }
        #expect(downloaded.map(\.id) == ["org/verified", "org/plain"])

        let notDownloaded = rows.filter { LocalModelListFilter.matches($0, downloaded: .no, verified: .any) }
        #expect(notDownloaded.map(\.id) == ["org/catalog"])

        let verifiedOnly = rows.filter { LocalModelListFilter.matches($0, downloaded: .any, verified: .yes) }
        #expect(verifiedOnly.map(\.id) == ["org/verified", "org/catalog"])

        let unverifiedOnDisk = rows.filter { LocalModelListFilter.matches($0, downloaded: .yes, verified: .no) }
        #expect(unverifiedOnDisk.map(\.id) == ["org/plain"])

        let impossible = rows.filter { LocalModelListFilter.matches($0, downloaded: .no, verified: .no) }
        #expect(impossible.isEmpty)
    }

    @Test("Loaded badge is present only while the model is resident")
    func loadedBadge() {
        let resident = LocalModelRowState(
            id: "org/resident", isDownloaded: true, isVerified: true, isLoaded: true
        )
        let idle = LocalModelRowState(
            id: "org/idle", isDownloaded: true, isVerified: false, isLoaded: false
        )
        #expect(resident.showsLoadedBadge)
        #expect(!idle.showsLoadedBadge)
    }

    @Test("Not-downloaded rows are catalog weights missing from disk")
    func notDownloadedCatalog() {
        let onDisk = entry(id: "org/on-disk", status: .verified)
        let missing = entry(id: "org/missing", status: .preview)
        let duplicate = entry(id: "org/missing", status: .verified)
        let unsupported = entry(id: "org/broken", status: .unsupported)
        let glob = CatalogEntry(
            id: "org/Family-*",
            url: "https://example.com/family",
            name: "Family",
            category: .llm,
            family: .qwen,
            format: .mlx
        )
        let weights = LocalModelListFilter.notDownloadedWeights(
            catalog: [onDisk, missing, duplicate, unsupported, glob],
            excluding: ["org/on-disk"]
        )
        #expect(weights.map(\.id) == ["org/missing"])
        #expect(weights.first?.status == .preview)
    }

    @Test("Disk match is prefix-safe and All keeps every model")
    func diskMatch() {
        let model = URL(fileURLWithPath: "/Volumes/Disk1/Models/org/repo")
        #expect(ModelRootFilter.matches(modelURL: model, selectedRoot: ""))
        #expect(ModelRootFilter.matches(modelURL: model, selectedRoot: "/Volumes/Disk1/Models"))
        #expect(ModelRootFilter.matches(modelURL: model, selectedRoot: "/Volumes/Disk1/Models/"))
        #expect(!ModelRootFilter.matches(modelURL: model, selectedRoot: "/Volumes/Disk2/Models"))
        #expect(!ModelRootFilter.matches(
            modelURL: URL(fileURLWithPath: "/Volumes/Disk10/Models/org/repo"),
            selectedRoot: "/Volumes/Disk1"
        ))
    }
}
