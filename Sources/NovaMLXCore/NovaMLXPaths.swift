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
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configFile = home.appendingPathComponent(".config/novamlx/\(name)")
        guard let content = try? String(contentsOf: configFile, encoding: .utf8) else { return [] }
        return parseModelsPathContents(content)
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
    /// Small / portable models download here. Load still searches `modelsDirs`.
    public static let modelsDir: URL = {
        if let path = readPathConfig("models-path") {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            log.info("[Paths] modelsDir from config: \(url.path)")
            return url
        }
        return baseDir.appendingPathComponent("models")
    }()

    /// All configured model roots that currently exist. First is `modelsDir`.
    public static var modelsDirs: [URL] {
        let configured = readPathConfigLines("models-path").map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        var seen = Set<String>()
        var urls: [URL] = []
        let fm = FileManager.default
        for url in configured + [modelsDir] {
            let key = url.standardizedFileURL.path
            guard seen.insert(key).inserted else { continue }
            if triggersRemovableVolumeTCC(url) {
                urls.append(url)
                continue
            }
            guard fm.fileExists(atPath: url.path) else { continue }
            urls.append(url)
        }
        return urls
    }

    /// Models at or above this size download to the extra root with most free space.
    public static let largeModelDownloadThresholdBytes: UInt64 = 64 * 1_073_741_824

    /// Resolve an on-disk model folder across every configured root
    /// (`org/name`, a flat folder, or `hub/models/org/name`). Missing models
    /// fall back to the primary root so downloads have a destination.
    public static func directory(forModelId id: String, roots: [URL]? = nil) -> URL {
        let search = roots ?? modelsDirs
        let fm = FileManager.default
        for root in search {
            let candidates = [
                root.appendingPathComponent(id, isDirectory: true),
                root.appendingPathComponent("hub/models/\(id)", isDirectory: true),
            ]
            for candidate in candidates {
                if triggersRemovableVolumeTCC(candidate) { continue }
                if fm.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
        }
        return modelsDir.appendingPathComponent(id, isDirectory: true)
    }

    /// Primary root for small models; extra disk for large checkpoints.
    public static func downloadRoot(estimatedBytes: UInt64, roots: [URL]? = nil) -> URL {
        let search = roots ?? modelsDirs
        guard estimatedBytes >= largeModelDownloadThresholdBytes, search.count > 1 else {
            return modelsDir
        }
        var best: (url: URL, free: Int64)?
        for root in search.dropFirst() {
            if triggersRemovableVolumeTCC(root) {
                if best == nil { best = (root, Int64.max / 4) }
                continue
            }
            guard FileManager.default.isWritableFile(atPath: root.path) else { continue }
            let free = (try? FileManager.default.attributesOfFileSystem(forPath: root.path)[.systemFreeSize] as? Int64) ?? 0
            if free > Int64(estimatedBytes), best == nil || free > best!.free {
                best = (root, free)
            }
        }
        return best?.url ?? modelsDir
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
