import Foundation
import Testing
@testable import NovaMLXModelManager

@Suite("Aria2Downloader")
struct Aria2DownloaderTests {
    @Test("locates the system aria2c binary")
    func locatesBinary() {
        let url = Aria2Downloader.locateBinary()
        #expect(url != nil)
        if let url {
            #expect(FileManager.default.isExecutableFile(atPath: url.path))
            #expect(url.lastPathComponent == "aria2c")
        }
    }

    @Test("writes an aria2 input list with out= and dir=")
    func writesInputList() throws {
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("aria2-list-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dest) }

        let files = [
            Aria2DownloadFile(
                url: URL(string: "https://example.com/a.safetensors")!,
                relativePath: "model-00001-of-00002.safetensors",
                expectedSize: 100
            ),
            Aria2DownloadFile(
                url: URL(string: "https://example.com/b.json")!,
                relativePath: "config.json",
                expectedSize: 10
            ),
        ]
        let list = dest.appendingPathComponent("in.txt")
        try Aria2Downloader.writeInputList(files: files, destination: dest, to: list)
        let body = try String(contentsOf: list, encoding: .utf8)
        #expect(body.contains("https://example.com/a.safetensors"))
        #expect(body.contains("  out=model-00001-of-00002.safetensors"))
        #expect(body.contains("  out=config.json"))
        #expect(body.contains("  dir=\(dest.path)"))
    }

    @Test("disk snapshot reports allocated size and completeness")
    func diskSnapshot() throws {
        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent("aria2-snap-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dest, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dest) }

        let complete = dest.appendingPathComponent("done.bin")
        try Data(repeating: 1, count: 64).write(to: complete)

        let partial = dest.appendingPathComponent("partial.bin")
        try Data(repeating: 2, count: 32).write(to: partial)
        try Data().write(to: URL(fileURLWithPath: partial.path + ".aria2"))

        let files = [
            Aria2DownloadFile(url: URL(string: "https://example.com/done.bin")!, relativePath: "done.bin", expectedSize: 64),
            Aria2DownloadFile(url: URL(string: "https://example.com/partial.bin")!, relativePath: "partial.bin", expectedSize: 100),
            Aria2DownloadFile(url: URL(string: "https://example.com/missing.bin")!, relativePath: "missing.bin", expectedSize: 50),
        ]
        let snap = Aria2Downloader.diskSnapshot(files: files, destination: dest)
        #expect(snap.files.count == 3)
        #expect(snap.files[0].isComplete)
        #expect(snap.files[0].downloadedBytes == 64)
        #expect(!snap.files[1].isComplete)
        #expect(snap.files[1].downloadedBytes > 0)
        #expect(snap.files[1].downloadedBytes <= 100)
        #expect(!snap.files[2].isComplete)
        #expect(snap.files[2].downloadedBytes == 0)
        #expect(snap.downloadedBytes == snap.files.map(\.downloadedBytes).reduce(0, +))
    }
}
