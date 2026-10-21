import Foundation
import Testing
@testable import NovaMLXCore

@Suite("Model list filters")
struct ModelListFilterTests {
    private func entry(id: String, tags: [String]) -> CatalogEntry {
        CatalogEntry(
            id: id,
            url: "https://example.com/\(id)",
            name: id,
            category: .llm,
            family: .qwen,
            format: .mlx,
            tags: tags
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

    @Test("Move destinations skip the current root and unwritable disks")
    func moveDestinations() {
        let home = URL(fileURLWithPath: "/Users/me/.nova/models")
        let disk1 = URL(fileURLWithPath: "/Volumes/Disk1/Models")
        let disk2 = URL(fileURLWithPath: "/Volumes/Disk2/Models")
        let nested = URL(fileURLWithPath: "/Volumes/Disk1/Models/extra")
        let model = URL(fileURLWithPath: "/Volumes/Disk1/Models/org/repo")
        let statuses = [
            status(home, mounted: true, writable: true),
            status(disk1, mounted: true, writable: true),
            status(disk2, mounted: false, writable: false),
            status(disk2, mounted: true, writable: false, priority: 3),
        ]
        let dests = ModelRootFilter.moveDestinations(modelURL: model, statuses: statuses)
        #expect(dests.map(\.path) == [home.path])

        let deep = URL(fileURLWithPath: "/Volumes/Disk1/Models/extra/org/repo")
        let nestedDests = ModelRootFilter.moveDestinations(
            modelURL: deep,
            statuses: [
                status(disk1, mounted: true, writable: true),
                status(nested, mounted: true, writable: true, priority: 1),
            ]
        )
        #expect(nestedDests.map(\.path) == [disk1.path])

        let unknown = URL(fileURLWithPath: "/tmp/not-a-model-root/org/repo")
        #expect(ModelRootFilter.moveDestinations(modelURL: unknown, statuses: statuses).isEmpty)
    }

    private func status(
        _ root: URL,
        mounted: Bool,
        writable: Bool,
        priority: Int = 0
    ) -> NovaMLXPaths.ModelDirStatus {
        NovaMLXPaths.ModelDirStatus(
            path: root.path,
            priority: priority,
            isRemovable: root.path.hasPrefix("/Volumes/"),
            mounted: mounted,
            exists: mounted,
            writable: writable,
            freeBytes: writable ? 1_000 : nil
        )
    }
}
