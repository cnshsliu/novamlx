import Foundation
import os.log

public enum NovaMLXPaths {
    private static let log = Logger(subsystem: "com.novamlx", category: "Paths")

    /// Splits a models-path file: one directory per line, `#` comments ignored.
    /// First path is the default download root (portable / internal disk);
    /// later paths are extra scan/load roots (external disks).
    public static func parseModelsPathContents(_ content: String) -> [String] {
        content.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
    }

    private static func readPathConfigLines(_ name: String) -> [String] {
        let content = (try? String(contentsOf: pathConfigFile(name), encoding: .utf8)) ?? ""
        return parseModelsPathContents(content)
    }

    /// Location of `~/.config/novamlx/<name>`.
    public static func pathConfigFile(_ name: String) -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".config/novamlx/\(name)")
    }

    /// Persist an ordered models-path list (index 0 = highest priority).
    /// Validates, dedupes (first occurrence wins), and writes atomically.
    /// `file` is injectable for tests; production uses the default location.
    public static func setModelsDirs(_ paths: [String], to file: URL? = nil) throws {
        let cleaned = normalizeModelsDirPaths(paths)
        guard let first = cleaned.first else {
            throw ModelStorageError.emptyDirectoryList
        }
        // The primary root must be usable: create it now if missing so a typo
        // fails here instead of at the next download.
        let primary = URL(fileURLWithPath: first, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: primary, withIntermediateDirectories: true)
        } catch {
            throw ModelStorageError.primaryNotWritable(first)
        }
        guard FileManager.default.isWritableFile(atPath: primary.path) else {
            throw ModelStorageError.primaryNotWritable(first)
        }

        let target = file ?? pathConfigFile("models-path")
        let content = cleaned.joined(separator: "\n") + "\n"
        let fm = FileManager.default
        try fm.createDirectory(
            at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = target.appendingPathExtension("tmp")
        try content.write(to: tmp, atomically: false, encoding: .utf8)
        if fm.fileExists(atPath: target.path) {
            _ = try fm.replaceItemAt(target, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: target)
        }
        log.info("[Paths] models-path updated: \(cleaned)")
    }

    /// Tilde-expand, standardize, drop blanks, dedupe (order preserved).
    public static func normalizeModelsDirPaths(_ paths: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for raw in paths {
            let expanded = (raw as NSString).expandingTildeInPath
                .trimmingCharacters(in: .whitespaces)
            guard !expanded.isEmpty else { continue }
            let key = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
            guard seen.insert(key).inserted else { continue }
            result.append(key)
        }
        return result
    }

    private static func readPathConfig(_ name: String) -> String? {
        readPathConfigLines(name).first
    }

    /// Validate configured paths. Returns error messages for any path that's configured but inaccessible.
    /// Call this at startup before using any paths. If non-empty, show dialog and quit.
    public static func validateConfiguredPaths() -> [String] {
        var errors: [String] = []
        let fm = FileManager.default

        // Check baseDir if explicitly configured
        if let path = readPathConfig("path") {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            if !fm.fileExists(atPath: url.path) {
                errors.append("Data directory not found: \(url.path)\n(Check ~/.config/novamlx/path)")
            } else if !fm.isWritableFile(atPath: url.path) {
                errors.append("Data directory not writable: \(url.path)\n(Check permissions)")
            }
        }

        // Primary models dir (first line) must exist. Extra lines are optional
        // so an unplugged external disk does not block startup.
        // Never stat `/Volumes/...` here — that pops the "access a removable
        // volume" TCC sheet on every launch.
        let modelPaths = readPathConfigLines("models-path")
        if let path = modelPaths.first {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            if !triggersRemovableVolumeTCC(url) {
                if !fm.fileExists(atPath: url.path) {
                    errors.append("Models directory not found: \(url.path)\n(Check ~/.config/novamlx/models-path)")
                } else if !fm.isReadableFile(atPath: url.path) {
                    errors.append("Models directory not readable: \(url.path)\n(Check permissions)")
                }
            }
        }

        return errors
    }

    /// True for paths on an external disk (`/Volumes/<name>/...` other than the
    /// boot volume). `FileManager.fileExists` / `checkResourceIsReachable` on
    /// these URLs is what shows the macOS removable-volume TCC alert.
    public static func triggersRemovableVolumeTCC(_ url: URL) -> Bool {
        let path = url.standardizedFileURL.path
        let prefix = "/Volumes/"
        guard path.hasPrefix(prefix) else { return false }
        let rest = path.dropFirst(prefix.count)
        guard let slash = rest.firstIndex(of: "/") else {
            return !isBootVolumeName(String(rest))
        }
        let volume = String(rest[..<slash])
        return !volume.isEmpty && !isBootVolumeName(volume)
    }

    public static func isBootVolumeName(_ name: String) -> Bool {
        guard !name.isEmpty else { return false }
        if let boot = bootVolumeName(), boot == name { return true }
        return name == "Macintosh HD"
    }

    public static func bootVolumeName() -> String? {
        (try? URL(fileURLWithPath: "/").resourceValues(forKeys: [.volumeNameKey]))?.volumeName
    }

    /// Resolution order: config file > NOVA_DIR env > default ~/.nova
    public static let baseDir: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        // 1. Config file: ~/.config/novamlx/path
        if let path = readPathConfig("path") {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            log.info("[Paths] NOVA_DIR from config: \(url.path)")
            return url
        }
        // 2. Env var: NOVA_DIR
        if let envPath = ProcessInfo.processInfo.environment["NOVA_DIR"] {
            let url = URL(fileURLWithPath: envPath, isDirectory: true)
            log.info("[Paths] NOVA_DIR from env: \(url.path)")
            return url
        }
        // 3. Default: ~/.nova
        return home.appendingPathComponent(".nova")
    }()

    /// Primary model root (first line of `~/.config/novamlx/models-path`).
    /// Computed so `setModelsDirs` takes effect without a process restart.
    public static var modelsDir: URL {
        if let path = readPathConfig("models-path") {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return baseDir.appendingPathComponent("models")
    }

    /// Every configured model root in priority order, whether or not it
    /// currently exists / is mounted. First element is `modelsDir`.
    public static var configuredModelsDirs: [URL] {
        let configured = readPathConfigLines("models-path").map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        var seen = Set<String>()
        var urls: [URL] = []
        for url in configured + [modelsDir] {
            let key = url.standardizedFileURL.path
            guard seen.insert(key).inserted else { continue }
            urls.append(url)
        }
        return urls
    }

    /// All configured model roots that currently exist. First is `modelsDir`.
    /// Removable volumes are kept without probing (TCC) — check mount state
    /// via `isVolumeMounted` before touching their contents.
    public static var modelsDirs: [URL] {
        let fm = FileManager.default
        return configuredModelsDirs.filter { url in
            if triggersRemovableVolumeTCC(url) { return true }
            return fm.fileExists(atPath: url.path)
        }
    }

    /// True when the volume hosting `url` is mounted. Non-removable paths are
    /// always "mounted". Uses `mountedVolumeURLs`, which does not pop the
    /// removable-volume TCC sheet (unlike `fileExists` on the volume root).
    public static func isVolumeMounted(_ url: URL) -> Bool {
        guard triggersRemovableVolumeTCC(url) else { return true }
        let prefix = "/Volumes/"
        let rest = url.standardizedFileURL.path.dropFirst(prefix.count)
        let volume = rest.split(separator: "/").first.map(String.init) ?? ""
        guard !volume.isEmpty else { return false }
        let mounts = FileManager.default.mountedVolumeURLs(
            includingResourceValuesForKeys: nil, options: []) ?? []
        return mounts.contains { $0.lastPathComponent == volume }
    }

    /// Resolve an on-disk model folder across every configured root
    /// (`org/name`, a flat folder, or `hub/models/org/name`). Roots are
    /// searched in priority order; unmounted removable volumes are skipped
    /// without probing (no TCC prompt). Missing models fall back to the
    /// primary root so downloads have a destination.
    public static func directory(forModelId id: String, roots: [URL]? = nil) -> URL {
        candidateDirectories(forModelId: id, roots: roots).first
            ?? modelsDir.appendingPathComponent(id, isDirectory: true)
    }

    /// Every existing on-disk location for `id`, in root-priority order.
    /// Both layouts (`<root>/<id>` and `<root>/hub/models/<id>`) are checked
    /// per root. Unmounted removable volumes are skipped without probing.
    public static func candidateDirectories(forModelId id: String, roots: [URL]? = nil) -> [URL] {
        let search = roots ?? modelsDirs
        let fm = FileManager.default
        var seen = Set<String>()
        var matches: [URL] = []
        for root in search {
            guard isVolumeMounted(root) else { continue }
            let candidates = [
                root.appendingPathComponent(id, isDirectory: true),
                root.appendingPathComponent("hub/models/\(id)", isDirectory: true),
            ]
            for candidate in candidates {
                let key = candidate.standardizedFileURL.path
                guard seen.insert(key).inserted else { continue }
                if fm.fileExists(atPath: candidate.path) {
                    matches.append(candidate)
                }
            }
        }
        return matches
    }

    /// Pick the download root by strict priority: the first configured root
    /// that is mounted, writable, and has more than `estimatedBytes` free.
    /// A root whose directory does not exist yet is still eligible when its
    /// nearest existing ancestor is writable (the download creates it).
    /// Throws `ModelStorageError.insufficientSpace` when NO root qualifies.
    public static func downloadRoot(estimatedBytes: UInt64, roots: [URL]? = nil) throws -> URL {
        let search = roots ?? configuredModelsDirs
        let fm = FileManager.default
        var reasons: [String] = []
        for root in search {
            guard isVolumeMounted(root) else {
                reasons.append("\(root.path): volume not mounted")
                continue
            }
            // Probe the root itself, or the nearest existing ancestor when the
            // root still needs to be created.
            var probe = root
            while !fm.fileExists(atPath: probe.path) {
                let parent = probe.deletingLastPathComponent()
                if parent.path == probe.path { break }
                probe = parent
            }
            guard fm.fileExists(atPath: probe.path) else {
                reasons.append("\(root.path): does not exist and cannot be created")
                continue
            }
            guard fm.isWritableFile(atPath: probe.path) else {
                reasons.append("\(root.path): not writable")
                continue
            }
            let free = (try? fm.attributesOfFileSystem(forPath: probe.path)[.systemFreeSize] as? Int64) ?? 0
            guard free > Int64(estimatedBytes) else {
                reasons.append("\(root.path): only \(free) bytes free, need > \(estimatedBytes)")
                continue
            }
            return root
        }
        throw ModelStorageError.insufficientSpace(neededBytes: estimatedBytes, reasons: reasons)
    }

    /// Per-root status for the storage management API / settings UI.
    public struct ModelDirStatus: Sendable {
        public let path: String
        public let priority: Int
        public let isRemovable: Bool
        public let mounted: Bool
        public let exists: Bool
        public let writable: Bool
        public let freeBytes: UInt64?

        public init(path: String, priority: Int, isRemovable: Bool, mounted: Bool,
                    exists: Bool, writable: Bool, freeBytes: UInt64?) {
            self.path = path
            self.priority = priority
            self.isRemovable = isRemovable
            self.mounted = mounted
            self.exists = exists
            self.writable = writable
            self.freeBytes = freeBytes
        }
    }

    /// Status of every configured root, in priority order. Removable volumes
    /// are only probed when mounted (TCC-safe).
    public static func modelDirStatuses(roots: [URL]? = nil) -> [ModelDirStatus] {
        let search = roots ?? configuredModelsDirs
        let fm = FileManager.default
        return search.enumerated().map { index, root in
            let removable = triggersRemovableVolumeTCC(root)
            let mounted = isVolumeMounted(root)
            guard mounted else {
                return ModelDirStatus(
                    path: root.path, priority: index, isRemovable: true,
                    mounted: false, exists: false, writable: false, freeBytes: nil)
            }
            let exists = fm.fileExists(atPath: root.path)
            let writable = exists && fm.isWritableFile(atPath: root.path)
            let free = exists
                ? (try? fm.attributesOfFileSystem(forPath: root.path)[.systemFreeSize] as? Int64)
                : nil
            return ModelDirStatus(
                path: root.path, priority: index, isRemovable: removable,
                mounted: true, exists: exists, writable: writable,
                freeBytes: free.map { UInt64(max($0, 0)) })
        }
    }

    public static var logFile: URL { baseDir.appendingPathComponent("novamlx.log") }
    public static var workerStderrFile: URL { baseDir.appendingPathComponent("worker.stderr") }
    public static var configFile: URL { baseDir.appendingPathComponent("config.json") }
    public static var metricsFile: URL { baseDir.appendingPathComponent("metrics.json") }
    public static var sessionsDir: URL { baseDir.appendingPathComponent("sessions") }
    public static var prefixCacheBaseDir: URL { baseDir.appendingPathComponent("prefix_cache") }
    public static var chatHistoryDir: URL { baseDir.appendingPathComponent("chat_history") }
    public static var modelfilesDir: URL { baseDir.appendingPathComponent("modelfiles") }
    public static var tokenhubDir: URL { baseDir.appendingPathComponent("tokenhub") }

    public static var voicesDir: URL {
        let dir = baseDir.appendingPathComponent("voices")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public static var catalogCacheDir: URL {
        let dir = baseDir.appendingPathComponent("cache/catalog", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
    public static var catalogCacheFile: URL {
        catalogCacheDir.appendingPathComponent("models.json")
    }

    // Auth & subscription
    public static var sessionFile: URL { baseDir.appendingPathComponent("session") }
    public static var authCacheFile: URL { baseDir.appendingPathComponent("auth_cache.json") }

    public static func prefixCacheDir(for modelId: String) -> URL {
        prefixCacheBaseDir.appendingPathComponent(
            modelId.replacingOccurrences(of: "/", with: "_"), isDirectory: true)
    }

    /// The old (incorrect) directory under Application Support — used only for migration/cleanup.
    public static var legacyAppSupportDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("NovaMLX")
    }
}

/// Errors from model storage directory management and download placement.
public enum ModelStorageError: Error, LocalizedError {
    /// No directory was provided for the models-path list.
    case emptyDirectoryList
    /// The first (primary) directory cannot be created or written.
    case primaryNotWritable(String)
    /// Every configured root failed the download preflight; `reasons` holds
    /// one human-readable entry per root, in priority order.
    case insufficientSpace(neededBytes: UInt64, reasons: [String])

    public var errorDescription: String? {
        switch self {
        case .emptyDirectoryList:
            return "At least one model storage directory is required."
        case .primaryNotWritable(let path):
            return "Primary models directory is not writable: \(path)"
        case .insufficientSpace(let needed, let reasons):
            let detail = reasons.joined(separator: "; ")
            return "No storage directory can hold this download "
                + "(\(needed) bytes needed). \(detail)"
        }
    }
}
