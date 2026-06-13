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
}
