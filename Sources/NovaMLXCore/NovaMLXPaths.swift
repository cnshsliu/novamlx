import Foundation
import os.log

public enum NovaMLXPaths {
    private static let log = Logger(subsystem: "com.novamlx", category: "paths")

    private static func readPathConfig(_ name: String) -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configFile = home.appendingPathComponent(".config/novamlx/\(name)")
        guard let content = try? String(contentsOf: configFile, encoding: .utf8),
              let path = content.split(separator: "\n").first.map(String.init),
              !path.trimmingCharacters(in: .whitespaces).isEmpty
        else { return nil }
        return path.trimmingCharacters(in: .whitespaces)
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

        // Check modelsDir if explicitly configured
        if let path = readPathConfig("models-path") {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            if !fm.fileExists(atPath: url.path) {
                errors.append("Models directory not found: \(url.path)\n(Check ~/.config/novamlx/models-path)")
            } else if !fm.isReadableFile(atPath: url.path) {
                errors.append("Models directory not readable: \(url.path)\n(Check permissions)")
            }
        }

        return errors
    }

    /// Resolution order: config file > NOVA_DIR env > default ~/.nova
    public static let baseDir: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        // 1. Config file: ~/.config/novamlx/path
        if let path = readPathConfig("path") {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            log.info("NOVA_DIR from config: \(url.path)")
            return url
        }
        // 2. Env var: NOVA_DIR
        if let envPath = ProcessInfo.processInfo.environment["NOVA_DIR"] {
            let url = URL(fileURLWithPath: envPath, isDirectory: true)
            log.info("NOVA_DIR from env: \(url.path)")
            return url
        }
        // 3. Default: ~/.nova
        return home.appendingPathComponent(".nova")
    }()

    public static let modelsDir: URL = {
        // 1. Config file: ~/.config/novamlx/models-path
        if let path = readPathConfig("models-path") {
            let url = URL(fileURLWithPath: path, isDirectory: true)
            log.info("modelsDir from config: \(url.path)")
            return url
        }
        // 2. Fallback: <baseDir>/models
        return baseDir.appendingPathComponent("models")
    }()

    public static var logFile: URL { baseDir.appendingPathComponent("novamlx.log") }
    public static var configFile: URL { baseDir.appendingPathComponent("config.json") }
    public static var metricsFile: URL { baseDir.appendingPathComponent("metrics.json") }
    public static var sessionsDir: URL { baseDir.appendingPathComponent("sessions") }
    public static var loadedModelsFile: URL { baseDir.appendingPathComponent("loaded_models.json") }
    public static var prefixCacheBaseDir: URL { baseDir.appendingPathComponent("prefix_cache") }
    public static var chatHistoryDir: URL { baseDir.appendingPathComponent("chat_history") }

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
