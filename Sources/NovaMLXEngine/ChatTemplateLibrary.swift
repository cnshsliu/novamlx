import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Four-level chat template resolution:
/// 1. **User override file** — `~/.nova/templates/{sanitizedModelId}.jinja`
///    (raw .jinja content for one specific model, highest priority).
/// 2. **Registry exact / repo-prefix / family-arch override** — looked up in
///    `ChatTemplateRegistry` (data-driven, hot-reloadable). Returns the name
///    of a bundled .jinja file in `Resources/ChatTemplates/`.
/// 3. **Downloaded** — whatever came with the model (fallback, returns nil so
///    `ensureChatTemplate` keeps the on-disk template).
///
/// Adding a fix for a new problematic model = edit `registry.json`, no recompile.
enum ChatTemplateLibrary {

    /// Resolve template. Returns nil when no override is configured (caller
    /// should keep the model's downloaded template).
    static func resolve(
        modelId: String,
        family: ModelFamily,
        architecture: String?,
        downloadedTemplate: String?
    ) -> String? {
        // Level 1: User .jinja override (raw content, per-model)
        if let t = loadUserOverride(modelId: modelId) {
            NovaMLXLog.info("[ChatTemplateLibrary] \(modelId): level 1 (user .jinja override)")
            return t
        }

        // Level 2: Registry-driven override (bundled template name)
        if let name = ChatTemplateRegistry.shared.templateOverride(
            modelId: modelId, family: family, architecture: architecture
        ), let t = loadBundleTemplate(name: name) {
            NovaMLXLog.info("[ChatTemplateLibrary] \(modelId): level 2 (registry: \(name))")
            return t
        }

        // Level 3: Use downloaded
        return downloadedTemplate
    }

    // MARK: - Level 1

    private static func loadUserOverride(modelId: String) -> String? {
        let sanitized = modelId.replacingOccurrences(of: "/", with: "_")
        let url = NovaMLXPaths.baseDir
            .appendingPathComponent("templates")
            .appendingPathComponent("\(sanitized).jinja")
        return try? String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - Bundle loading

    private static func loadBundleTemplate(name: String) -> String? {
        // Templates are in Sources/NovaMLXEngine/ChatTemplates/{name}.jinja
        // SPM copies ChatTemplates/ as resources when declared in Package.swift
        if let url = Bundle.module.url(forResource: name, withExtension: "jinja", subdirectory: "ChatTemplates") {
            return try? String(contentsOf: url, encoding: .utf8)
        }
        // Fallback: flat layout
        if let url = Bundle.module.url(forResource: name, withExtension: "jinja") {
            return try? String(contentsOf: url, encoding: .utf8)
        }
        // Fallback: try reading from source tree (dev mode)
        let devPath = "Sources/NovaMLXEngine/ChatTemplates/\(name).jinja"
        if let base = ProcessInfo.processInfo.environment["SRCROOT"] {
            let url = URL(fileURLWithPath: base).appendingPathComponent(devPath)
            return try? String(contentsOf: url, encoding: .utf8)
        }
        return nil
    }
}
