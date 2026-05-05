import Foundation
import NovaMLXCore

/// Data-driven configuration for chat-template processing and family routing.
///
/// Loads `Resources/template-registry.json` (bundled in NovaMLXUtils) and
/// overlays any user override at `~/.nova/templates/registry.json`
/// (per-key precedence — user's keys win, missing keys fall through to bundle).
///
/// Adding a new family or fixing a problematic quant becomes a JSON edit
/// instead of a Swift recompile. Hot-reload is supported via `reload()`.
///
/// Lives in NovaMLXUtils so both NovaMLXEngine (template selection) and
/// NovaMLXModelManager (family detection) can consult it without circular
/// module dependencies.
public final class ChatTemplateRegistry: @unchecked Sendable {
    public static let shared = ChatTemplateRegistry()

    private let lock = NovaMLXLock()
    private var cached: Document = .empty
    private var loadedAt: Date?

    private init() {
        reload()
    }

    /// Re-read the bundled and user JSON files. Call after editing the user
    /// registry to pick up changes without restarting.
    public func reload() {
        let bundle = Self.loadBundled()
        let user = Self.loadUserOverride()
        let merged = bundle.merging(user)
        lock.withLock {
            self.cached = merged
            self.loadedAt = Date()
        }
        let bundleVer = bundle.version
        let userPresent = user.familyCount > 0 || !user.exactTemplateOverrides.isEmpty
        NovaMLXLog.info("[ChatTemplateRegistry] loaded — bundle v\(bundleVer), \(merged.familyCount) families, \(merged.exactTemplateOverrides.count) exact overrides\(userPresent ? " (user override present)" : "")")
    }

    // MARK: - Public lookup API

    /// Returns the bundled template name to force for `modelId`, or nil.
    /// Resolution order: exact → repo prefix (longest first) → family+arch.
    public func templateOverride(modelId: String, family: ModelFamily, architecture: String?) -> String? {
        let snap = lock.withLock { cached }
        if let exact = snap.exactTemplateOverrides[modelId] { return exact }
        let prefixHit = snap.repoPrefixOverrides
            .filter { modelId.hasPrefix($0.key) }
            .max(by: { $0.key.count < $1.key.count })?.value
        if let p = prefixHit { return p }
        if let arch = architecture {
            let key = "\(family.rawValue)_\(arch)"
            if let f = snap.byFamilyArchOverrides[key] { return f }
        }
        return nil
    }

    /// Returns the family-specific behavior config, falling back to the
    /// "other" entry if the family has no explicit configuration.
    public func familyConfig(for family: ModelFamily) -> FamilyConfig {
        let snap = lock.withLock { cached }
        return snap.families[family.rawValue]
            ?? snap.families["other"]
            ?? FamilyConfig.minimalFallback
    }

    /// Family detection extensions — merged into ModelDiscovery's hardcoded
    /// maps. Editing the JSON adds support for new model_type / architecture
    /// without a Swift recompile.
    public func familyByModelType(_ modelType: String) -> ModelFamily? {
        let snap = lock.withLock { cached }
        let normalized = modelType.lowercased().replacingOccurrences(of: "-", with: "_")
        if let raw = snap.familyByModelType[normalized],
           let f = ModelFamily(rawValue: raw) {
            return f
        }
        return nil
    }

    public func familyByArchitecture(_ arch: String) -> ModelFamily? {
        let snap = lock.withLock { cached }
        if let raw = snap.familyByArchitecture[arch],
           let f = ModelFamily(rawValue: raw) {
            return f
        }
        return nil
    }

    // MARK: - JSON I/O

    private static func loadBundled() -> Document {
        // Bundled at Sources/NovaMLXUtils/Resources/template-registry.json
        if let url = Bundle.module.url(forResource: "template-registry", withExtension: "json"),
           let data = try? Data(contentsOf: url) {
            return Document.parse(data, source: "bundle") ?? .empty
        }
        // Dev fallback (when SRCROOT is set, e.g. xcodebuild / Xcode tests)
        let dev = "Sources/NovaMLXUtils/Resources/template-registry.json"
        if let base = ProcessInfo.processInfo.environment["SRCROOT"] {
            let url = URL(fileURLWithPath: base).appendingPathComponent(dev)
            if let data = try? Data(contentsOf: url) {
                return Document.parse(data, source: "src-root") ?? .empty
            }
        }
        NovaMLXLog.warning("[ChatTemplateRegistry] bundled template-registry.json not found — using empty defaults")
        return .empty
    }

    private static func loadUserOverride() -> Document {
        let url = NovaMLXPaths.baseDir
            .appendingPathComponent("templates")
            .appendingPathComponent("registry.json")
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url) else {
            return .empty
        }
        guard let doc = Document.parse(data, source: "user-override") else {
            NovaMLXLog.warning("[ChatTemplateRegistry] user override at \(url.path) is invalid JSON — ignoring")
            return .empty
        }
        return doc
    }

    // MARK: - Internal types

    public struct FamilyConfig: Sendable {
        public let label: String
        public let expectedFormats: [String]
        public let expectedEosTokens: [String]
        public let hallucinationPatterns: [String]
        public let leakageBlacklist: [String]
        public let thinkingMarkers: [String]
        public let preserveThinkingSupported: Bool
        public let softSwitchSupported: Bool

        public static let minimalFallback = FamilyConfig(
            label: "minimal",
            expectedFormats: [],
            expectedEosTokens: [],
            hallucinationPatterns: ["(?m)^\\s*user\\s*\\n", "(?m)^\\s*assistant\\s*\\n"],
            leakageBlacklist: [],
            thinkingMarkers: [],
            preserveThinkingSupported: false,
            softSwitchSupported: false
        )
    }

    /// In-memory representation of registry.json. Kept private so the JSON
    /// schema can evolve without breaking call-sites.
    private struct Document {
        var version: Int
        var families: [String: FamilyConfig]
        var exactTemplateOverrides: [String: String]
        var byFamilyArchOverrides: [String: String]
        var repoPrefixOverrides: [String: String]
        var familyByModelType: [String: String]
        var familyByArchitecture: [String: String]

        var familyCount: Int { families.count }

        static let empty = Document(
            version: 0,
            families: [:],
            exactTemplateOverrides: [:],
            byFamilyArchOverrides: [:],
            repoPrefixOverrides: [:],
            familyByModelType: [:],
            familyByArchitecture: [:]
        )

        /// User-overrides win on a per-key basis (NOT whole sub-tree replacement).
        func merging(_ other: Document) -> Document {
            var result = self
            result.version = other.version > 0 ? other.version : self.version
            for (k, v) in other.families { result.families[k] = v }
            for (k, v) in other.exactTemplateOverrides { result.exactTemplateOverrides[k] = v }
            for (k, v) in other.byFamilyArchOverrides { result.byFamilyArchOverrides[k] = v }
            for (k, v) in other.repoPrefixOverrides { result.repoPrefixOverrides[k] = v }
            for (k, v) in other.familyByModelType { result.familyByModelType[k] = v }
            for (k, v) in other.familyByArchitecture { result.familyByArchitecture[k] = v }
            return result
        }

        static func parse(_ data: Data, source: String) -> Document? {
            guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                NovaMLXLog.warning("[ChatTemplateRegistry] failed to parse \(source) as JSON object")
                return nil
            }
            let version = obj["version"] as? Int ?? 0

            // families
            var families: [String: FamilyConfig] = [:]
            if let famDict = obj["families"] as? [String: Any] {
                for (key, value) in famDict where key != "_doc" {
                    guard let v = value as? [String: Any] else { continue }
                    families[key] = FamilyConfig(
                        label: v["label"] as? String ?? key,
                        expectedFormats: v["expectedFormats"] as? [String] ?? [],
                        expectedEosTokens: v["expectedEosTokens"] as? [String] ?? [],
                        hallucinationPatterns: v["hallucinationPatterns"] as? [String] ?? [],
                        leakageBlacklist: v["leakageBlacklist"] as? [String] ?? [],
                        thinkingMarkers: v["thinkingMarkers"] as? [String] ?? [],
                        preserveThinkingSupported: v["preserveThinkingSupported"] as? Bool ?? false,
                        softSwitchSupported: v["softSwitchSupported"] as? Bool ?? false
                    )
                }
            }

            // template overrides
            var exact: [String: String] = [:]
            var byArch: [String: String] = [:]
            var byRepo: [String: String] = [:]
            if let tplDict = obj["templateOverrides"] as? [String: Any] {
                if let m = tplDict["exact"] as? [String: String] { exact = m }
                if let m = tplDict["byFamilyArch"] as? [String: String] { byArch = m.filter { $0.key != "_doc" } }
                if let m = tplDict["byRepoPrefix"] as? [String: String] { byRepo = m.filter { $0.key != "_doc" } }
            }

            // family detection
            var byMt: [String: String] = [:]
            var byArchFam: [String: String] = [:]
            if let det = obj["familyDetection"] as? [String: Any] {
                if let m = det["byModelType"] as? [String: String] { byMt = m }
                if let m = det["byArchitecture"] as? [String: String] { byArchFam = m }
            }

            return Document(
                version: version,
                families: families,
                exactTemplateOverrides: exact,
                byFamilyArchOverrides: byArch,
                repoPrefixOverrides: byRepo,
                familyByModelType: byMt,
                familyByArchitecture: byArchFam
            )
        }
    }
}
