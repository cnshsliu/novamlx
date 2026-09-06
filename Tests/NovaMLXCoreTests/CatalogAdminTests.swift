import Foundation
import Testing
@testable import NovaMLXCore

@Suite("Catalog admin")
struct CatalogAdminTests {
    @Test("Gate is off when the flag file is missing")
    func gateOff() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-gate-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        #expect(!CatalogAdminGate.isEnabled(home: tmp))
    }

    @Test("Gate is on when the flag file exists")
    func gateOn() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("catalog-gate-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Data().write(to: tmp.appendingPathComponent(CatalogAdminGate.flagFileName))
        #expect(CatalogAdminGate.isEnabled(home: tmp))
    }

    @Test("Rejects duplicate ids and org-wide globs")
    func validateRejects() throws {
        let a = CatalogEntry(
            id: "org/foo", url: "https://huggingface.co/org/foo", name: "Foo",
            category: .llm, family: .qwen, format: .mlx
        )
        let dup = CatalogFile(schemaVersion: 1, models: [a, a])
        #expect(dup.validate().contains { $0.contains("duplicate id") })

        let glob = CatalogEntry(
            id: "org/*", url: "https://huggingface.co/org", name: "All",
            category: .llm, family: .qwen, format: .mlx
        )
        let badGlob = CatalogFile(schemaVersion: 1, models: [glob])
        #expect(badGlob.validate().contains { $0.contains("family pattern") })
    }

    @Test("Accepts a family glob and writes pretty JSON")
    func validateFamilyGlob() throws {
        let entry = CatalogEntry(
            id: "mlx-community/Qwen3.8-*",
            url: "https://huggingface.co/models?search=Qwen3.8",
            name: "Qwen3.8 family",
            category: .llm, family: .qwen, format: .mlx,
            status: .verified
        )
        let file = CatalogFile(schemaVersion: 1, updatedAt: "2026-08-27T00:00:00Z", models: [entry])
        #expect(file.validate().isEmpty)
        let data = try file.encodedPretty()
        let text = String(data: data, encoding: .utf8) ?? ""
        #expect(text.contains("Qwen3.8-*"))
        #expect(text.hasSuffix("\n"))
    }

    @Test("Verified draft fills Hub URL, quant, and verified status")
    func verifiedDraft() {
        let entry = CatalogEntry.verifiedDraft(
            id: "ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit",
            url: "",
            category: .llm,
            family: .other,
            sizeBytes: 2_147_483_648
        )
        #expect(entry.status == .verified)
        #expect(entry.url == "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-8bit")
        #expect(entry.quant == "8bit")
        #expect(entry.size != nil)
        #expect(entry.testedOn == version)
        #expect(CatalogEntry.defaultURL(forId: "mlx-community/Qwen3.8-*").contains("Qwen3.8"))
    }
}
