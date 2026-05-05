import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXEngine

/// Verifies the operational kill switch wired through `ServerConfig.prefixCacheEnabled`
/// and `MLXEngine.setPrefixCacheEnabled(_:)`. When disabled, the engine must not
/// instantiate any `PrefixCacheManager` and `getOrCreatePrefixCacheManager` must
/// return `nil` for every model.
@Suite("Prefix Cache Kill Switch")
struct PrefixCacheKillSwitchTests {

    @Test("ServerConfig defaults prefixCacheEnabled to true")
    func defaultIsEnabled() {
        let cfg = ServerConfig()
        #expect(cfg.prefixCacheEnabled == true)
    }

    @Test("ServerConfig honors prefixCacheEnabled = false")
    func canDisable() {
        let cfg = ServerConfig(prefixCacheEnabled: false)
        #expect(cfg.prefixCacheEnabled == false)
    }

    @Test("ServerConfig decodes legacy JSON without the field as enabled")
    func legacyJSONDecodesEnabled() throws {
        // Older configs predate prefixCacheEnabled. Missing key must default
        // to `true` to preserve historical behavior.
        let legacyJSON = """
        { "host": "127.0.0.1", "port": 6590, "adminPort": 6591 }
        """
        let data = Data(legacyJSON.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: data)
        #expect(cfg.prefixCacheEnabled == true)
    }

    @Test("ServerConfig decodes prefixCacheEnabled = false from JSON")
    func disabledFromJSON() throws {
        let json = """
        { "host": "127.0.0.1", "port": 6590, "adminPort": 6591, "prefixCacheEnabled": false }
        """
        let data = Data(json.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: data)
        #expect(cfg.prefixCacheEnabled == false)
    }

    @Test("Engine getOrCreatePrefixCacheManager returns non-nil by default")
    func engineDefaultIsEnabled() {
        let engine = MLXEngine()
        #expect(engine.isPrefixCacheEnabled == true)
        let manager = engine.getOrCreatePrefixCacheManager(modelId: "test-model-default")
        #expect(manager != nil)
        // Cleanup so the test does not leave a manager behind for any
        // subsequent suite.
        engine.setPrefixCacheEnabled(false)
    }

    @Test("Engine setPrefixCacheEnabled(false) blocks manager creation")
    func engineDisabledBlocksCreation() {
        let engine = MLXEngine()
        engine.setPrefixCacheEnabled(false)
        #expect(engine.isPrefixCacheEnabled == false)
        let manager = engine.getOrCreatePrefixCacheManager(modelId: "test-model-disabled")
        #expect(manager == nil)
        // Stats must also be unavailable when no manager exists.
        #expect(engine.getPrefixCacheStats(for: "test-model-disabled") == nil)
    }

    @Test("Re-enabling after disable allows new managers")
    func reEnableAfterDisable() {
        let engine = MLXEngine()
        engine.setPrefixCacheEnabled(false)
        #expect(engine.getOrCreatePrefixCacheManager(modelId: "round-trip") == nil)
        engine.setPrefixCacheEnabled(true)
        let manager = engine.getOrCreatePrefixCacheManager(modelId: "round-trip")
        #expect(manager != nil)
        engine.setPrefixCacheEnabled(false)
    }
}
