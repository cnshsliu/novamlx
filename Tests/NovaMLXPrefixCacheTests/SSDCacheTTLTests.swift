import Foundation
import Testing
@testable import NovaMLXPrefixCache

@Suite("PrefixCache SSD TTL")
struct SSDCacheTTLTests {

    private final class Clock: @unchecked Sendable {
        var date: Date
        init(_ date: Date) { self.date = date }
        func now() -> Date { date }
    }

    private func writeHeader(url: URL, metadata: [String: String]) throws {
        let object: [String: Any] = ["__metadata__": metadata]
        let json = try JSONSerialization.data(withJSONObject: object)
        var len = UInt64(json.count).littleEndian
        var data = Data()
        withUnsafeBytes(of: len) { data.append(contentsOf: $0) }
        data.append(json)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
    }

    @Test("scan drops a block older than the TTL and keeps a fresh one")
    func scanPrunesExpired() throws {
        let clock = Clock(Date(timeIntervalSince1970: 10_000))
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("pc-ttl-scan-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let expired = dir.appendingPathComponent("a/abcd.safetensors")
        let fresh = dir.appendingPathComponent("b/bb01.safetensors")
        try writeHeader(url: expired, metadata: [
            "block_hash": "abcd",
            "token_count": "64",
            "created_at": "1000",
        ])
        try writeHeader(url: fresh, metadata: [
            "block_hash": "bb01",
            "token_count": "64",
            "created_at": "9990",
        ])

        let store = SSDCacheStore(cacheDir: dir, maxSize: 10_000_000, ttl: 30, now: { clock.now() })
        #expect(store.blockCount == 1)
        #expect(store.hasBlock([0xbb, 0x01]))
        #expect(!store.hasBlock([0xab, 0xcd]))
        #expect(!FileManager.default.fileExists(atPath: expired.path))
        #expect(FileManager.default.fileExists(atPath: fresh.path))
    }

    @Test("a live block expires when the clock passes the TTL")
    func expiresWhileRunning() throws {
        let clock = Clock(Date(timeIntervalSince1970: 5_000))
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("pc-ttl-live-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("c/cc10.safetensors")
        try writeHeader(url: file, metadata: [
            "block_hash": "cc10",
            "token_count": "64",
            "created_at": "5000",
        ])

        let store = SSDCacheStore(cacheDir: dir, maxSize: 10_000_000, ttl: 60, now: { clock.now() })
        #expect(store.hasBlock([0xcc, 0x10]))

        clock.date = Date(timeIntervalSince1970: 5_061)
        #expect(!store.hasBlock([0xcc, 0x10]))
        #expect(store.blockCount == 0)

        store.discardExpired(now: clock.date)
        #expect(store.blockCount == 0)
    }
}
