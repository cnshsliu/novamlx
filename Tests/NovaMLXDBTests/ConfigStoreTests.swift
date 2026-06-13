import Testing
import Foundation
import GRDB
import NovaMLXDB
import NovaMLXCore

@Suite("ConfigStore")
struct ConfigStoreTests {

    private func makeTmpDir() throws -> URL {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("novamlx-cfg-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        return tmp
    }

    @Test("v2 migration adds all 7 server-field columns with correct defaults")
    func v2MigrationAddsColumns() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let record = try nova.configStore.get()
        #expect(record.maxConcurrentRequests == 16)
        #expect(record.requestTimeout == 300)
        #expect(record.contextScalingTarget == nil)
        #expect(record.tlsKeyPassword == nil)
        #expect(record.maxRequestSizeMB == 100)
        #expect(record.maxProcessMemory == "auto")
        #expect(record.prefixCacheEnabled == true)
    }

    @Test("ConfigStore.update persists new fields")
    func updatePersistsNewFields() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        try nova.configStore.update { rec in
            rec.maxConcurrentRequests = 32
            rec.requestTimeout = 600
            rec.contextScalingTarget = 8192
            rec.tlsKeyPassword = "hunter2"
            rec.maxRequestSizeMB = 200
            rec.maxProcessMemory = "12G"
            rec.prefixCacheEnabled = false
        }

        let fetched = try nova.configStore.get()
        #expect(fetched.maxConcurrentRequests == 32)
        #expect(fetched.requestTimeout == 600)
        #expect(fetched.contextScalingTarget == 8192)
        #expect(fetched.tlsKeyPassword == "hunter2")
        #expect(fetched.maxRequestSizeMB == 200)
        #expect(fetched.maxProcessMemory == "12G")
        #expect(fetched.prefixCacheEnabled == false)
    }

    @Test("Configuration.syncToStore round-trips through ConfigStore")
    func syncToStoreRoundTrip() async throws {
        let tmp = try makeTmpDir()
        let nova = NovaDB.shared
        try nova.setup(baseDir: tmp)

        let config = NovaMLXConfiguration.shared
        // Configure state directly via setters
        await config.setServerConfig(ServerConfig(
            host: "1.2.3.4", port: 7000, adminPort: 7001,
            maxConcurrentRequests: 24, requestTimeout: 450,
            contextScalingTarget: 4096,
            tlsCertPath: "/tmp/cert.pem", tlsKeyPath: "/tmp/key.pem",
            tlsKeyPassword: "secret",
            maxRequestSizeMB: 150, maxProcessMemory: "8G",
            prefixCacheEnabled: false
        ))
        await config.setDefaultModel("test-model")
        await config.setHuggingfaceEndpoint("https://test.hf.co")
        await config.setModelsDirectory(URL(fileURLWithPath: "/tmp/models"))

        // Bridge: syncToStore pushes to configStore
        await config.syncToStore()

        // Verify by reading the store directly
        let record = try nova.configStore.get()
        #expect(record.host == "1.2.3.4")
        #expect(record.port == 7000)
        #expect(record.adminPort == 7001)
        #expect(record.tlsEnabled == true)
        #expect(record.tlsCertPath == "/tmp/cert.pem")
        #expect(record.tlsKeyPassword == "secret")
        #expect(record.maxConcurrentRequests == 24)
        #expect(record.contextScalingTarget == 4096)
        #expect(record.maxProcessMemory == "8G")
        #expect(record.prefixCacheEnabled == false)
        #expect(record.defaultModel == "test-model")
        #expect(record.modelsDir == "/tmp/models")
        #expect(record.hfEndpoint == "https://test.hf.co")
    }
}
