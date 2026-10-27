import Foundation
import Testing
@testable import NovaMLXCore

@Suite("PrefixCache Location")
struct PrefixCacheLocationTests {

    @Test("absent external volume falls back to an internal directory")
    func missingVolumeFallsBack() throws {
        let fm = FileManager.default
        let base = fm.temporaryDirectory.appendingPathComponent("pc-root-\(UUID().uuidString)", isDirectory: true)
        let models = base.appendingPathComponent("Models", isDirectory: true)
        try fm.createDirectory(at: models, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: base) }

        let samsung = "/Volumes/NoSuchVolume12345/NovaMLX/prefix_cache"
        let internalRoot = models.appendingPathComponent("prefix_cache").path
        let fallback = base.appendingPathComponent("nova/prefix_cache", isDirectory: true)
        let resolved = PrefixCacheLocation.resolve(
            PrefixCacheLocation.Inputs(
                pathFileText: "\(samsung)\n\(internalRoot)\n",
                fallbackRoot: fallback
            )
        )
        #expect(resolved.root.standardizedFileURL.path == URL(fileURLWithPath: internalRoot, isDirectory: true).standardizedFileURL.path)
        #expect(resolved.preferred.count == 2)
    }

    @Test("no usable preferred root returns the internal fallback without throwing")
    func nothingUsableUsesFallback() {
        let fallback = URL(fileURLWithPath: "/tmp/novamlx-prefix-fallback-\(UUID().uuidString)", isDirectory: true)
        let resolved = PrefixCacheLocation.resolve(
            PrefixCacheLocation.Inputs(
                pathFileText: "/Volumes/NoSuchVolume12345/NovaMLX/prefix_cache\n/Volumes/OtherDiskMissing/prefix_cache\n",
                fallbackRoot: fallback
            )
        )
        #expect(resolved.root.standardizedFileURL.path == fallback.standardizedFileURL.path)
    }

    @Test("mounted preferred root wins over the fallback")
    func preferredWinsWhenUsable() throws {
        let fm = FileManager.default
        let base = fm.temporaryDirectory.appendingPathComponent("pc-win-\(UUID().uuidString)", isDirectory: true)
        let preferred = base.appendingPathComponent("fast", isDirectory: true)
        try fm.createDirectory(at: preferred, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: base) }
        let fallback = base.appendingPathComponent("slow", isDirectory: true)
        let chosen = PrefixCacheLocation.selectRoot(
            preferred: [preferred],
            fallback: fallback,
            isUsable: PrefixCacheLocation.isUsable
        )
        #expect(chosen.standardizedFileURL.path == preferred.standardizedFileURL.path)
    }

    @Test("env root and TTL beat the config files; explicit beats env")
    func precedence() {
        let json = #"{"ttlSeconds":99,"hotCacheMaxBytes":32,"ssdMaxSizeBytes":1000}"#.data(using: .utf8)
        let fromFiles = PrefixCacheLocation.resolve(
            PrefixCacheLocation.Inputs(
                environment: [
                    "NOVA_PREFIX_CACHE_ROOT": "/Volumes/Samsung768/NovaMLX/prefix_cache;/tmp/internal-prefix",
                    "NOVA_PREFIX_CACHE_TTL": "15",
                ],
                pathFileText: "/Volumes/Other/cache\n",
                settingsJSON: json,
                fallbackRoot: URL(fileURLWithPath: "/tmp/built-in-prefix", isDirectory: true)
            ),
            isUsable: { url in
                !url.path.contains("Samsung") && !url.path.contains("Other")
            }
        )
        #expect(fromFiles.root.path.hasSuffix("/internal-prefix"))
        #expect(!fromFiles.root.path.contains("Samsung"))
        #expect(fromFiles.ttlSeconds == 15)
        #expect(fromFiles.hotCacheMaxBytes == 32)
        #expect(fromFiles.ssdMaxSizeBytes == 1000)

        let explicit = PrefixCacheLocation.resolve(
            PrefixCacheLocation.Inputs(
                environment: [
                    "NOVA_PREFIX_CACHE_ROOT": "/tmp/from-env",
                    "NOVA_PREFIX_CACHE_TTL": "15",
                    "NOVA_PREFIX_CACHE_HOT_BYTES": "8",
                ],
                settingsJSON: json,
                explicitRoots: ["/tmp/from-settings"],
                explicitTTLSeconds: 3,
                fallbackRoot: URL(fileURLWithPath: "/tmp/built-in-prefix", isDirectory: true)
            ),
            isUsable: { _ in true }
        )
        #expect(explicit.root.path.hasSuffix("/from-settings"))
        #expect(explicit.ttlSeconds == 3)
        #expect(explicit.hotCacheMaxBytes == 8)
    }

    @Test("settings files round-trip and drive production resolution")
    func persistAndResolve() throws {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("pc-cfg-\(UUID().uuidString)", isDirectory: true)
        let internalRoot = dir.appendingPathComponent("internal", isDirectory: true)
        try fm.createDirectory(at: internalRoot, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: dir) }

        try PrefixCacheLocation.persist(
            PrefixCacheLocation.Persisted(
                roots: ["/Volumes/NoSuchVolume12345/NovaMLX/prefix_cache", internalRoot.path],
                ttlSeconds: 120,
                hotCacheMaxBytes: 4096
            ),
            to: dir
        )
        let loaded = PrefixCacheLocation.loadPersisted(from: dir)
        #expect(loaded.roots?.count == 2)
        #expect(loaded.ttlSeconds == 120)
        #expect(loaded.hotCacheMaxBytes == 4096)

        // A later save of only the TTL must keep the roots and the hot budget.
        try PrefixCacheLocation.persist(
            PrefixCacheLocation.Persisted(ttlSeconds: 45),
            to: dir
        )
        let merged = PrefixCacheLocation.loadPersisted(from: dir)
        #expect(merged.roots?.count == 2)
        #expect(merged.ttlSeconds == 45)
        #expect(merged.hotCacheMaxBytes == 4096)

        let fallback = dir.appendingPathComponent("fallback", isDirectory: true)
        let resolved = PrefixCacheLocation.resolveProduction(
            environment: [:],
            configDirectory: dir,
            fallbackRoot: fallback
        )
        #expect(resolved.root.standardizedFileURL.path == internalRoot.standardizedFileURL.path)
        #expect(resolved.ttlSeconds == 45)
        #expect(resolved.hotCacheMaxBytes == 4096)
    }

    @Test("ServerConfig decodes prefix-cache knobs and defaults the rest")
    func serverConfigDecodesPrefixCache() throws {
        #expect(ServerConfig().prefixCacheRoots == nil)
        #expect(ServerConfig().prefixCacheTTLSeconds == nil)
        #expect(ServerConfig().prefixCacheHotBytes == nil)
        #expect(ServerConfig().prefixCacheEnabled == true)

        let json = """
        {"prefixCacheTTLSeconds": 30,
         "prefixCacheRoots": ["/Volumes/Samsung768/NovaMLX/prefix_cache", "~/Models/prefix_cache"],
         "prefixCacheHotBytes": 1024,
         "prefixCacheMaxBytes": 2048}
        """
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: Data(json.utf8))
        #expect(cfg.prefixCacheTTLSeconds == 30)
        #expect(cfg.prefixCacheRoots == [
            "/Volumes/Samsung768/NovaMLX/prefix_cache",
            "~/Models/prefix_cache",
        ])
        #expect(cfg.prefixCacheHotBytes == 1024)
        #expect(cfg.prefixCacheMaxBytes == 2048)
        #expect(cfg.prefixCacheEnabled == true)
        #expect(cfg.port == 6590)

        let encoded = try JSONDecoder().decode(
            ServerConfig.self,
            from: JSONEncoder().encode(cfg)
        )
        #expect(encoded.prefixCacheTTLSeconds == 30)
        #expect(encoded.prefixCacheRoots?.count == 2)
        #expect(encoded.prefixCacheHotBytes == 1024)
    }
}
