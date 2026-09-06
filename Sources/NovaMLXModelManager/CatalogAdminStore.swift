import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Reads/writes `catalog/models.json` in the git checkout, copies the live
/// cache + bundled snapshot, and pushes to GitHub.
public struct CatalogAdminStore: Sendable {
    public let repoRoot: URL

    public var catalogURL: URL {
        repoRoot.appendingPathComponent("catalog/models.json")
    }

    public var bundleURL: URL {
        repoRoot.appendingPathComponent(
            "Sources/NovaMLXUtils/Resources/catalog/models.json"
        )
    }

    public init(repoRoot: URL) {
        self.repoRoot = repoRoot
    }

    public static func discover(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        bundleURL: URL = Bundle.main.bundleURL,
        cwd: URL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    ) -> CatalogAdminStore? {
        var candidates: [URL] = []
        if let raw = environment["NOVAMLX_REPO"]?.trimmingCharacters(in: .whitespaces),
           !raw.isEmpty {
            candidates.append(URL(fileURLWithPath: raw, isDirectory: true))
        }
        var cursor: URL? = bundleURL
        if bundleURL.pathExtension == "app" {
            cursor = bundleURL.deletingLastPathComponent()
        }
        while let dir = cursor {
            candidates.append(dir)
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            cursor = parent
        }
        candidates.append(home.appendingPathComponent("dev/novamlx", isDirectory: true))
        candidates.append(cwd)
        var walk: URL? = cwd
        while let dir = walk {
            candidates.append(dir)
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            walk = parent
        }

        var seen = Set<String>()
        for root in candidates {
            let path = root.standardizedFileURL.path
            if seen.contains(path) { continue }
            seen.insert(path)
            let catalog = root.appendingPathComponent("catalog/models.json")
            if FileManager.default.isReadableFile(atPath: catalog.path) {
                return CatalogAdminStore(repoRoot: root)
            }
        }
        return nil
    }

    public func load() throws -> CatalogFile {
        let data = try Data(contentsOf: catalogURL)
        return try CatalogFile.decode(data)
    }

    /// Validate, write the git file, copy bundle + `~/.nova` cache.
    public func save(_ file: CatalogFile, cacheURL: URL = NovaMLXPaths.catalogCacheFile) throws -> CatalogFile {
        let stamped = CatalogFile(
            schemaVersion: 1,
            updatedAt: CatalogFile.utcNow(),
            models: file.models
        )
        try stamped.validated()
        let data = try stamped.encodedPretty()
        try writeAtomically(data, to: catalogURL)
        try FileManager.default.createDirectory(
            at: bundleURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try writeAtomically(data, to: bundleURL)
        try FileManager.default.createDirectory(
            at: cacheURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try writeAtomically(data, to: cacheURL)
        return stamped
    }

    /// Stage catalog files, commit, push current branch.
    public func pushToGitHub() throws -> String {
        let files = [
            "catalog/models.json",
            "Sources/NovaMLXUtils/Resources/catalog/models.json",
        ]
        _ = try git(["add"] + files)
        let status = (try? git(["status", "--porcelain", "--"] + files)) ?? ""
        if !status.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            _ = try git([
                "commit", "-m", "catalog: update verified models",
            ])
        }
        let pushed = try git(["push"])
        return pushed.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "Pushed catalog/models.json to GitHub"
            : pushed
    }

    private func writeAtomically(_ data: Data, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url, options: .atomic)
    }

    @discardableResult
    private func git(_ args: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = args
        process.currentDirectoryURL = repoRoot
        process.environment = ProcessInfo.processInfo.environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let text = String(data: data, encoding: .utf8) ?? ""
        if process.terminationStatus != 0 {
            throw CatalogAdminError.gitFailed(text.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return text
    }
}
