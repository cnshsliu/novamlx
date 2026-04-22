import Foundation

public enum NovaMLXPaths {
    public static let baseDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".nova")

    public static var modelsDir: URL { baseDir.appendingPathComponent("models") }
    public static var cacheDir: URL { baseDir.appendingPathComponent("cache") }
    public static var logFile: URL { baseDir.appendingPathComponent("novamlx.log") }
    public static var configFile: URL { baseDir.appendingPathComponent("config.json") }
    public static var metricsFile: URL { baseDir.appendingPathComponent("metrics.json") }
    public static var sessionsDir: URL { baseDir.appendingPathComponent("sessions") }
    public static var loadedModelsFile: URL { baseDir.appendingPathComponent("loaded_models.json") }
    public static var prefixCacheBaseDir: URL { baseDir.appendingPathComponent("prefix_cache") }
    public static var chatHistoryDir: URL { baseDir.appendingPathComponent("chat_history") }

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
