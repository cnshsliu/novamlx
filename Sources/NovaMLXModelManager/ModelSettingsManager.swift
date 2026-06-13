import Foundation
import NovaMLXCore
import NovaMLXDB
import NovaMLXUtils

public final class ModelSettingsManager: @unchecked Sendable {
    private var _settings: [String: ModelSettings]
    private let lock = NovaMLXLock()

    public init(baseDirectory: URL) {
        self._settings = [:]
        try? FileManager.default.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        importLegacyJSONIfNeeded(file: baseDirectory.appendingPathComponent("model_settings.json"))
        load()
    }

    /// One-shot import of the legacy `model_settings.json` into the SQLite
    /// store. Lives here (not in NovaDB) because ModelSettings is a
    /// NovaMLXCore type and NovaMLXDB cannot import NovaMLXCore without
    /// creating a circular dependency. Idempotent: if the store already has
    /// rows, the file is left untouched; otherwise we parse, upsert, and
    /// rename the file to `.migrated` so we never run again.
    private func importLegacyJSONIfNeeded(file: URL) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: file.path) else { return }

        // Skip if store already populated — SQLite is source of truth.
        if let existing = try? NovaDB.shared.modelSettingsStore.list(), !existing.isEmpty {
            return
        }

        struct LegacyContainer: Decodable {
            let version: Int?
            let models: [String: ModelSettings]
        }
        guard let data = try? Data(contentsOf: file),
              let container = try? JSONDecoder().decode(LegacyContainer.self, from: data) else {
            NovaMLXLog.warning("[ModelSettings] Failed to parse legacy \(file.lastPathComponent); leaving file in place")
            return
        }

        for (modelId, settings) in container.models {
            let record = Self.toRecord(modelId: modelId, settings: settings)
            try? NovaDB.shared.modelSettingsStore.upsert(record)
        }
        NovaMLXLog.info("[ModelSettings] Imported \(container.models.count) entries from legacy \(file.lastPathComponent)")

        // Rename to .migrated so we never re-import. If a .migrated file
        // already exists from a prior run, just remove the source.
        let migrated = file.appendingPathExtension("migrated")
        if fm.fileExists(atPath: migrated.path) {
            try? fm.removeItem(at: file)
        } else {
            try? fm.moveItem(at: file, to: migrated)
        }
    }

    public func getSettings(_ modelId: String) -> ModelSettings {
        lock.withLock { _settings[modelId] ?? ModelSettings() }
    }

    public func setSettings(_ modelId: String, _ settings: ModelSettings) {
        lock.withLock {
            if settings.isDefault {
                for (key, _) in _settings where key != modelId {
                    _settings[key]?.isDefault = false
                }
            }
            _settings[modelId] = settings
        }
        save(modelId: modelId)
        NovaMLXLog.info("[ModelSettings] Updated settings for \(modelId)")
    }

    public func updateSettings(_ modelId: String, _ update: (inout ModelSettings) -> Void) {
        lock.withLock {
            var settings = _settings[modelId] ?? ModelSettings()
            let wasDefault = settings.isDefault
            update(&settings)
            if !wasDefault && settings.isDefault {
                for (key, _) in _settings where key != modelId {
                    _settings[key]?.isDefault = false
                }
            }
            _settings[modelId] = settings
        }
        save(modelId: modelId)
    }

    public func removeSettings(_ modelId: String) {
        _ = lock.withLock { _settings.removeValue(forKey: modelId) }
        do {
            try NovaDB.shared.modelSettingsStore.delete(modelId: modelId)
        } catch {
            NovaMLXLog.warning("[ModelSettings] Failed to delete \(modelId) from store: \(error)")
        }
    }

    public func getDefaultModelId() -> String? {
        lock.withLock {
            for (modelId, settings) in _settings where settings.isDefault {
                return modelId
            }
            return nil
        }
    }

    public func getPinnedModelIds() -> [String] {
        lock.withLock {
            _settings.filter { $0.value.isPinned }.map(\.key)
        }
    }

    public func getAllSettings() -> [String: ModelSettings] {
        lock.withLock { _settings }
    }

    public func resolveAlias(_ alias: String) -> String? {
        lock.withLock {
            for (modelId, settings) in _settings where settings.modelAlias == alias {
                return modelId
            }
            return nil
        }
    }

    public func resolveModelId(_ input: String) -> String {
        if let alias = resolveAlias(input) { return alias }
        return input
    }

    // MARK: - SQLite-backed load/save (Phase D2 cutover)

    private func load() {
        guard let records = try? NovaDB.shared.modelSettingsStore.list() else { return }
        lock.withLock {
            for record in records {
                _settings[record.modelId] = Self.toDomain(record)
            }
        }
        NovaMLXLog.info("[ModelSettings] Loaded \(_settings.count) model settings from SQLite")
    }

    /// Persist a single model's settings to the store. The manager keeps an
    /// in-memory cache (`_settings`) for fast lookup; this writes through to
    /// SQLite so settings survive restarts.
    private func save(modelId: String) {
        let snapshot = lock.withLock { _settings[modelId] }
        guard let settings = snapshot else {
            try? NovaDB.shared.modelSettingsStore.delete(modelId: modelId)
            return
        }
        let record = Self.toRecord(modelId: modelId, settings: settings)
        do {
            try NovaDB.shared.modelSettingsStore.upsert(record)
        } catch {
            NovaMLXLog.warning("[ModelSettings] Failed to upsert \(modelId): \(error)")
        }
    }

    // MARK: - Mapping

    private static func toRecord(modelId: String, settings: ModelSettings) -> ModelSettingsRecord {
        // Pack the full ModelSettings into samplingParams JSON so we don't
        // lose fields the schema doesn't have dedicated columns for. The
        // top-level columns mirror what's queryable (alias, defaults, pins,
        // ttl, context window).
        let samplingJSON: String? = {
            guard let data = try? JSONEncoder().encode(settings),
                  let s = String(data: data, encoding: .utf8) else { return nil }
            return s
        }()

        return ModelSettingsRecord(
            modelId: modelId,
            alias: settings.modelAlias,
            isDefault: settings.isDefault,
            isPinned: settings.isPinned,
            samplingParams: samplingJSON,
            ttlSeconds: settings.ttlSeconds,
            contextWindow: settings.maxContextWindow,
            draftModel: nil,
            updatedAt: Date()
        )
    }

    private static func toDomain(_ record: ModelSettingsRecord) -> ModelSettings {
        // Preferred path: decode the full snapshot from samplingParams.
        if let json = record.samplingParams,
           let data = json.data(using: .utf8),
           let decoded = try? JSONDecoder().decode(ModelSettings.self, from: data) {
            return decoded
        }
        // Fallback: reconstruct from the top-level columns.
        var settings = ModelSettings()
        settings.modelAlias = record.alias
        settings.isDefault = record.isDefault
        settings.isPinned = record.isPinned
        settings.ttlSeconds = record.ttlSeconds
        settings.maxContextWindow = record.contextWindow
        return settings
    }
}
