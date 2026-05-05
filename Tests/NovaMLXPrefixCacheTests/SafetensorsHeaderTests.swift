import Testing
import Foundation
import MLX
import CommonCrypto
@testable import NovaMLXPrefixCache

@Suite("Safetensors Header Reader Tests")
struct SafetensorsHeaderTests {

    private func makeTempStore() -> (SSDCacheStore, URL) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("safetensors-hdr-test-\(UUID().uuidString)")
        let store = SSDCacheStore(cacheDir: tmp, maxSize: 1024 * 1024 * 1024)
        return (store, tmp)
    }

    private func cleanup(_ dir: URL) {
        try? FileManager.default.removeItem(at: dir)
    }

    private func makeHash(_ seed: String) -> BlockHash {
        var digest = [UInt8](repeating: 0, count: Int(CC_SHA1_DIGEST_LENGTH))
        let data = Data(seed.utf8)
        data.withUnsafeBytes { ptr in
            _ = CC_SHA1(ptr.baseAddress, CC_LONG(data.count), &digest)
        }
        return Array(digest)
    }

    @Test("readSafetensorsMetadata extracts block_hash from real cache block")
    func readsMetadataFromRealBlock() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let hash = makeHash("hdr-test-block")
        let cacheData: [[MLXArray]] = [
            [MLXArray(0...5), MLXArray(0...5)]
        ]
        store.saveBlock(
            hash: hash,
            cacheData: cacheData,
            layerClasses: ["KVCacheSimple"],
            metaStates: [["offset:64"]],
            tokenCount: 64,
            modelName: "test-model"
        )

        // Wait for async write to land on disk
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            if store.loadBlock(hash) != nil { break }
            usleep(50_000)
        }

        let hex = hash.map { String(format: "%02x", $0) }.joined()
        let subdir = String(hex.first!)
        let fileURL = dir.appendingPathComponent(subdir).appendingPathComponent("\(hex).safetensors")

        let meta = try readSafetensorsMetadata(url: fileURL)
        let expectedHashHex = hex
        #expect(meta["block_hash"] == expectedHashHex)
        #expect(meta["token_count"] == "64")
        #expect(meta["model_name"] == "test-model")
    }

    @Test("readSafetensorsMetadata completes in <50ms for any block size")
    func headerReadIsFast() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        // Write a block with larger arrays to simulate real KV cache size
        let hash = makeHash("speed-test-block")
        let bigArray = MLXArray.zeros([256, 1024])
        let cacheData: [[MLXArray]] = Array(repeating: [bigArray, bigArray], count: 4)
        store.saveBlock(
            hash: hash,
            cacheData: cacheData,
            layerClasses: Array(repeating: "KVCacheSimple", count: 4),
            metaStates: Array(repeating: ["offset:64"], count: 4),
            tokenCount: 64,
            modelName: "speed-model"
        )

        // Wait for async write
        let deadline = Date().addingTimeInterval(3)
        while Date() < deadline {
            if store.loadBlock(hash) != nil { break }
            usleep(50_000)
        }

        let hex = hash.map { String(format: "%02x", $0) }.joined()
        let subdir = String(hex.first!)
        let fileURL = dir.appendingPathComponent(subdir).appendingPathComponent("\(hex).safetensors")

        let t0 = Date()
        let meta = try readSafetensorsMetadata(url: fileURL)
        let elapsed = Date().timeIntervalSince(t0) * 1000

        #expect(!meta.isEmpty)
        // Header read must be < 50ms even for multi-MB files
        #expect(elapsed < 50.0)
    }
}
