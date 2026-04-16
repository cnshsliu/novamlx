import Foundation
import NovaMLXCore
import NovaMLXUtils

public final class ModelSettingsManager: @unchecked Sendable {
    private let settingsFile: URL
    private var _settings: [String: ModelSettings]
    private let lock = NovaMLXLock()

    public init(baseDirectory: URL) {
        self.settingsFile = baseDirectory.appendingPathComponent("model_settings.json")
        self._settings = [:]
        try? FileManager.default.createDirectory(at: baseDirectory, withIntermediateDirectories: true)
        load()
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
        save()
        NovaMLXLog.info("Updated settings for \(modelId)")
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
        save()
    }

    public func removeSettings(_ modelId: String) {
        _ = lock.withLock { _settings.removeValue(forKey: modelId) }
        save()
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

    private func load() {
        guard let data = try? Data(contentsOf: settingsFile) else { return }
        guard let container = try? JSONDecoder().decode(SettingsContainer.self, from: data) else { return }
        lock.withLock { _settings = container.models }
        NovaMLXLog.info("Loaded model settings for \(_settings.count) models")
    }

    private func save() {
        let data = lock.withLock {
            let container = SettingsContainer(version: 1, models: _settings)
            return (try? JSONEncoder().encode(container)) ?? Data()
        }
        // .atomic write handles temp file + rename atomically, no need for manual moveItem
        try? data.write(to: settingsFile, options: .atomic)
    }
}

private struct SettingsContainer: Codable {
    let version: Int
    let models: [String: ModelSettings]
}
