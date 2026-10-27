import Foundation

/// Defaults for the long-term prefix-cache store.
///
/// The store is process-wide (one directory per model), so these knobs live
/// with server settings, environment variables, and `~/.config/novamlx/`
/// files — the same places as the model roots — rather than on a single
/// `GenerateParameters` value.
public enum PrefixCachePersistenceDefaults {
    public static let ttlSeconds: Double = 86_400
    /// 256 MiB of decoded prefix blocks kept in RAM in front of the files.
    public static let hotCacheMaxBytes: Int = 256 * 1024 * 1024
    public static let ssdMaxSizeBytes: UInt64 = 5 * 1024 * 1024 * 1024
}

/// Where long-term prefix KV lives, and for how long.
///
/// Preferred roots are tried in order. A missing external volume (for
/// example `/Volumes/Samsung768/...` left at the office) is skipped. The
/// built-in fallback, normally `~/.nova/prefix_cache`, is used when none of
/// the preferred roots are mounted and writable. Nothing here requires an
/// external disk to be present.
public enum PrefixCacheLocation {

    public struct Resolution: Sendable, Equatable {
        public var root: URL
        public var ttlSeconds: Double
        public var hotCacheMaxBytes: Int
        public var ssdMaxSizeBytes: UInt64
        /// Preferred roots that were considered, before the fallback.
        public var preferred: [URL]
        public var fallback: URL

        public init(
            root: URL,
            ttlSeconds: Double,
            hotCacheMaxBytes: Int,
            ssdMaxSizeBytes: UInt64,
            preferred: [URL],
            fallback: URL
        ) {
            self.root = root
            self.ttlSeconds = ttlSeconds
            self.hotCacheMaxBytes = hotCacheMaxBytes
            self.ssdMaxSizeBytes = ssdMaxSizeBytes
            self.preferred = preferred
            self.fallback = fallback
        }
    }

    /// Inputs for ``resolve(_:isUsable:)``. All fields are injectable so
    /// tests never touch a real external volume or the user config directory.
    public struct Inputs: Sendable {
        public var environment: [String: String]
        public var pathFileText: String?
        public var settingsJSON: Data?
        /// Highest priority. `nil` means "not set" and falls through to the
        /// environment, then the path file. An empty array means "no
        /// preferred root" — resolution goes straight to the fallback.
        public var explicitRoots: [String]?
        public var explicitTTLSeconds: Double?
        public var explicitHotCacheMaxBytes: Int?
        public var explicitSSDMaxSizeBytes: UInt64?
        public var fallbackRoot: URL

        public init(
            environment: [String: String] = [:],
            pathFileText: String? = nil,
            settingsJSON: Data? = nil,
            explicitRoots: [String]? = nil,
            explicitTTLSeconds: Double? = nil,
            explicitHotCacheMaxBytes: Int? = nil,
            explicitSSDMaxSizeBytes: UInt64? = nil,
            fallbackRoot: URL
        ) {
            self.environment = environment
            self.pathFileText = pathFileText
            self.settingsJSON = settingsJSON
            self.explicitRoots = explicitRoots
            self.explicitTTLSeconds = explicitTTLSeconds
            self.explicitHotCacheMaxBytes = explicitHotCacheMaxBytes
            self.explicitSSDMaxSizeBytes = explicitSSDMaxSizeBytes
            self.fallbackRoot = fallbackRoot
        }
    }

    /// Values mirrored into `ServerConfig` and the config files.
    public struct Persisted: Sendable, Equatable {
        public var roots: [String]?
        public var ttlSeconds: Double?
        public var hotCacheMaxBytes: Int?
        public var ssdMaxSizeBytes: UInt64?

        public init(
            roots: [String]? = nil,
            ttlSeconds: Double? = nil,
            hotCacheMaxBytes: Int? = nil,
            ssdMaxSizeBytes: UInt64? = nil
        ) {
            self.roots = roots
            self.ttlSeconds = ttlSeconds
            self.hotCacheMaxBytes = hotCacheMaxBytes
            self.ssdMaxSizeBytes = ssdMaxSizeBytes
        }

        public var isEmpty: Bool {
            roots == nil && ttlSeconds == nil && hotCacheMaxBytes == nil && ssdMaxSizeBytes == nil
        }
    }

    private struct SettingsFile: Codable {
        var ttlSeconds: Double?
        var hotCacheMaxBytes: Int?
        var ssdMaxSizeBytes: UInt64?
    }

    /// First usable preferred root, otherwise `fallback`.
    ///
    /// `fallback` is returned even when it does not exist yet. The cache
    /// store creates it. An unmounted preferred volume must not throw.
    public static func selectRoot(
        preferred: [URL],
        fallback: URL,
        isUsable: (URL) -> Bool
    ) -> URL {
        for url in preferred where isUsable(url) {
            return url
        }
        return fallback
    }

    /// Mounted, and either writable itself or creatable under a writable ancestor.
    ///
    /// Unmounted removable volumes return false without stating the volume,
    /// so an absent `/Volumes/Samsung768` does not raise a TCC prompt and
    /// does not fail resolution.
    public static func isUsable(_ url: URL, fileManager: FileManager = .default) -> Bool {
        if NovaMLXPaths.triggersRemovableVolumeTCC(url), !NovaMLXPaths.isVolumeMounted(url) {
            return false
        }
        var probe = url
        while !fileManager.fileExists(atPath: probe.path) {
            let parent = probe.deletingLastPathComponent()
            if parent.path == probe.path { return false }
            probe = parent
        }
        return fileManager.isWritableFile(atPath: probe.path)
    }

    /// Split a root list on newlines or `;`. A colon splits only when every
    /// piece is absolute (`/` or `~`), so a single macOS path is left intact.
    public static func splitRoots(_ raw: String) -> [String] {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }
        let chunks: [String]
        if trimmed.contains("\n") || trimmed.contains(";") {
            chunks = trimmed.split { $0 == "\n" || $0 == "\r" || $0 == ";" }.map(String.init)
        } else if trimmed.contains(":") {
            let parts = trimmed.split(separator: ":").map(String.init)
            if parts.count > 1, parts.allSatisfy({ $0.hasPrefix("/") || $0.hasPrefix("~") }) {
                chunks = parts
            } else {
                chunks = [trimmed]
            }
        } else {
            chunks = [trimmed]
        }
        return NovaMLXPaths.normalizeModelsDirPaths(chunks)
    }

    public static func resolve(
        _ inputs: Inputs,
        isUsable: (URL) -> Bool = { isUsable($0) }
    ) -> Resolution {
        let envRoots = inputs.environment["NOVA_PREFIX_CACHE_ROOT"].map(splitRoots) ?? []
        let fileRoots = NovaMLXPaths.normalizeModelsDirPaths(
            NovaMLXPaths.parseModelsPathContents(inputs.pathFileText ?? ""))
        let preferredStrings: [String]
        if let explicit = inputs.explicitRoots {
            preferredStrings = NovaMLXPaths.normalizeModelsDirPaths(explicit)
        } else if !envRoots.isEmpty {
            preferredStrings = envRoots
        } else {
            preferredStrings = fileRoots
        }
        let preferred = preferredStrings.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let root = selectRoot(preferred: preferred, fallback: inputs.fallbackRoot, isUsable: isUsable)
        let file = decodeSettings(inputs.settingsJSON)

        let ttl = sanitizeTTL(
            inputs.explicitTTLSeconds
                ?? doubleValue(inputs.environment["NOVA_PREFIX_CACHE_TTL"])
                ?? file?.ttlSeconds
                ?? PrefixCachePersistenceDefaults.ttlSeconds)
        let hot = sanitizeBytes(
            inputs.explicitHotCacheMaxBytes
                ?? intValue(inputs.environment["NOVA_PREFIX_CACHE_HOT_BYTES"])
                ?? file?.hotCacheMaxBytes
                ?? PrefixCachePersistenceDefaults.hotCacheMaxBytes)
        let ssd = inputs.explicitSSDMaxSizeBytes
            ?? uintValue(inputs.environment["NOVA_PREFIX_CACHE_MAX_BYTES"])
            ?? file?.ssdMaxSizeBytes
            ?? PrefixCachePersistenceDefaults.ssdMaxSizeBytes

        return Resolution(
            root: root,
            ttlSeconds: ttl,
            hotCacheMaxBytes: hot,
            ssdMaxSizeBytes: max(ssd, 1),
            preferred: preferred,
            fallback: inputs.fallbackRoot
        )
    }

    /// Read env + `prefix-cache-path` + `prefix-cache.json` and resolve a root.
    /// Does not create directories.
    public static func resolveProduction(
        explicit: Persisted = Persisted(),
        environment: [String: String] = ProcessInfo.processInfo.environment,
        configDirectory: URL? = nil,
        fallbackRoot: URL = NovaMLXPaths.prefixCacheBaseDir
    ) -> Resolution {
        let directory = configDirectory ?? defaultConfigDirectory()
        let pathText = try? String(
            contentsOf: directory.appendingPathComponent("prefix-cache-path"),
            encoding: .utf8)
        let json = try? Data(contentsOf: directory.appendingPathComponent("prefix-cache.json"))
        // Explicit fields win per knob. Unset fields fall through to the
        // environment, then the files above.
        let inputs = Inputs(
            environment: environment,
            pathFileText: pathText,
            settingsJSON: json,
            explicitRoots: explicit.roots,
            explicitTTLSeconds: explicit.ttlSeconds,
            explicitHotCacheMaxBytes: explicit.hotCacheMaxBytes,
            explicitSSDMaxSizeBytes: explicit.ssdMaxSizeBytes,
            fallbackRoot: fallbackRoot
        )
        return resolve(inputs)
    }

    public static func modelDirectory(modelId: String, root: URL) -> URL {
        root.appendingPathComponent(
            modelId.replacingOccurrences(of: "/", with: "_"),
            isDirectory: true)
    }

    public static func defaultConfigDirectory() -> URL {
        NovaMLXPaths.pathConfigFile("prefix-cache-path").deletingLastPathComponent()
    }

    public static func loadPersisted(from directory: URL) -> Persisted {
        let pathURL = directory.appendingPathComponent("prefix-cache-path")
        let jsonURL = directory.appendingPathComponent("prefix-cache.json")
        let fm = FileManager.default
        var persisted = Persisted()
        if fm.fileExists(atPath: pathURL.path),
           let text = try? String(contentsOf: pathURL, encoding: .utf8) {
            persisted.roots = NovaMLXPaths.normalizeModelsDirPaths(
                NovaMLXPaths.parseModelsPathContents(text))
        }
        if fm.fileExists(atPath: jsonURL.path),
           let data = try? Data(contentsOf: jsonURL),
           let file = decodeSettings(data) {
            persisted.ttlSeconds = file.ttlSeconds
            persisted.hotCacheMaxBytes = file.hotCacheMaxBytes
            persisted.ssdMaxSizeBytes = file.ssdMaxSizeBytes
        }
        return persisted
    }

    /// Write whichever fields are non-nil. A nil field leaves the existing
    /// file value in place so an unrelated settings save does not clear the
    /// prefix-cache root. Does not create external volumes.
    public static func persist(_ settings: Persisted, to directory: URL) throws {
        if settings.isEmpty { return }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        if let roots = settings.roots {
            let url = directory.appendingPathComponent("prefix-cache-path")
            let body = roots.joined(separator: "\n") + (roots.isEmpty ? "" : "\n")
            try writeAtomic(body.data(using: .utf8) ?? Data(), to: url)
        }
        if settings.ttlSeconds != nil || settings.hotCacheMaxBytes != nil || settings.ssdMaxSizeBytes != nil {
            let url = directory.appendingPathComponent("prefix-cache.json")
            var file = SettingsFile()
            if fm.fileExists(atPath: url.path),
               let data = try? Data(contentsOf: url),
               let existing = decodeSettings(data) {
                file = existing
            }
            if let ttl = settings.ttlSeconds { file.ttlSeconds = sanitizeTTL(ttl) }
            if let hot = settings.hotCacheMaxBytes { file.hotCacheMaxBytes = sanitizeBytes(hot) }
            if let ssd = settings.ssdMaxSizeBytes { file.ssdMaxSizeBytes = max(ssd, 1) }
            let data = try JSONEncoder().encode(file)
            try writeAtomic(data, to: url)
        }
    }

    private static func writeAtomic(_ data: Data, to url: URL) throws {
        let fm = FileManager.default
        let tmp = url.appendingPathExtension("tmp")
        try data.write(to: tmp, options: .atomic)
        if fm.fileExists(atPath: url.path) {
            _ = try fm.replaceItemAt(url, withItemAt: tmp)
        } else {
            try fm.moveItem(at: tmp, to: url)
        }
    }

    private static func decodeSettings(_ data: Data?) -> SettingsFile? {
        guard let data, !data.isEmpty else { return nil }
        return try? JSONDecoder().decode(SettingsFile.self, from: data)
    }

    private static func doubleValue(_ raw: String?) -> Double? {
        guard let raw, let value = Double(raw.trimmingCharacters(in: .whitespaces)) else { return nil }
        return value
    }

    private static func intValue(_ raw: String?) -> Int? {
        guard let raw = raw?.trimmingCharacters(in: .whitespaces), !raw.isEmpty else { return nil }
        return Int(raw)
    }

    private static func uintValue(_ raw: String?) -> UInt64? {
        guard let raw = raw?.trimmingCharacters(in: .whitespaces), !raw.isEmpty else { return nil }
        return UInt64(raw)
    }

    private static func sanitizeTTL(_ value: Double) -> Double {
        if value.isNaN || value < 0 { return 0 }
        return value
    }

    private static func sanitizeBytes(_ value: Int) -> Int {
        max(value, 0)
    }
}
