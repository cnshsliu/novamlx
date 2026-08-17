import Testing
import Foundation
@testable import NovaMLXModelManager
@testable import NovaMLXCore

@Suite("ModelSettings Persistence", .serialized)
struct SettingsPersistenceTests {

    @Test("kvBits persists across manager restarts")
    func kvBitsPersistsAcrossRestarts() {
        IsolatedTestDB.run { tempDir in
            // Phase 1: Set kvBits and save
            let manager1 = ModelSettingsManager(baseDirectory: tempDir)
            manager1.updateSettings("mlx-community/Phi-3.5-mini-instruct-4bit") { settings in
                settings.kvBits = 4
                settings.kvGroupSize = 64
            }

            // Verify it was saved
            let saved = manager1.getSettings("mlx-community/Phi-3.5-mini-instruct-4bit")
            #expect(saved.kvBits == 4)
            #expect(saved.kvGroupSize == 64)

            // Phase 2: Simulate app restart — create new manager reading same file
            let manager2 = ModelSettingsManager(baseDirectory: tempDir)
            let restored = manager2.getSettings("mlx-community/Phi-3.5-mini-instruct-4bit")
            #expect(restored.kvBits == 4)
            #expect(restored.kvGroupSize == 64)
        }
    }

    @Test("kvBits=8 persists correctly")
    func kvBits8Persists() {
        IsolatedTestDB.run { tempDir in
            let manager = ModelSettingsManager(baseDirectory: tempDir)
            manager.updateSettings("test-model") { $0.kvBits = 8; $0.kvGroupSize = 64 }

            let manager2 = ModelSettingsManager(baseDirectory: tempDir)
            #expect(manager2.getSettings("test-model").kvBits == 8)
        }
    }

    @Test("kvBits=2 persists correctly")
    func kvBits2Persists() {
        IsolatedTestDB.run { tempDir in
            let manager = ModelSettingsManager(baseDirectory: tempDir)
            manager.updateSettings("test-model") { $0.kvBits = 2; $0.kvGroupSize = 64 }

            let manager2 = ModelSettingsManager(baseDirectory: tempDir)
            #expect(manager2.getSettings("test-model").kvBits == 2)
        }
    }

    @Test("Clearing kvBits persists as nil")
    func clearingKvBitsPersists() {
        IsolatedTestDB.run { tempDir in
            // Set 4-bit first
            let manager = ModelSettingsManager(baseDirectory: tempDir)
            manager.updateSettings("test-model") { $0.kvBits = 4; $0.kvGroupSize = 64 }

            // Clear it (user selects "Off")
            manager.updateSettings("test-model") { $0.kvBits = nil; $0.kvGroupSize = nil }

            // Verify nil persists across restart
            let manager2 = ModelSettingsManager(baseDirectory: tempDir)
            let restored = manager2.getSettings("test-model")
            #expect(restored.kvBits == nil)
            #expect(restored.kvGroupSize == nil)
        }
    }

    @Test("Model ID with slash persists correctly")
    func modelIdWithSlashPersists() {
        IsolatedTestDB.run { tempDir in
            let modelId = "mlx-community/Phi-3.5-mini-instruct-4bit"
            let manager = ModelSettingsManager(baseDirectory: tempDir)
            manager.updateSettings(modelId) { $0.kvBits = 4 }

            let manager2 = ModelSettingsManager(baseDirectory: tempDir)
            #expect(manager2.getSettings(modelId).kvBits == 4)
        }
    }

    @Test("Default settings have nil kvBits")
    func defaultSettingsNilKvBits() {
        IsolatedTestDB.run { dir in
            let manager = ModelSettingsManager(baseDirectory: dir)
            let settings = manager.getSettings("nonexistent-model")
            #expect(settings.kvBits == nil)
            #expect(settings.kvGroupSize == nil)
        }
    }

    @Test("Multiple models persist independently")
    func multipleModelsPersist() {
        IsolatedTestDB.run { tempDir in
            let manager = ModelSettingsManager(baseDirectory: tempDir)
            manager.updateSettings("model-a") { $0.kvBits = 4 }
            manager.updateSettings("model-b") { $0.kvBits = 8 }
            manager.updateSettings("model-c") { $0.kvBits = 2 }

            let manager2 = ModelSettingsManager(baseDirectory: tempDir)
            #expect(manager2.getSettings("model-a").kvBits == 4)
            #expect(manager2.getSettings("model-b").kvBits == 8)
            #expect(manager2.getSettings("model-c").kvBits == 2)
        }
    }
}
