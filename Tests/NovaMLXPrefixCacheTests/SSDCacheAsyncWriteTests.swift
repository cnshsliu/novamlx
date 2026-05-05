import Testing
import Foundation
import MLX
import CommonCrypto
@testable import NovaMLXPrefixCache

@Suite("SSDCache Async Write Tests")
struct SSDCacheAsyncWriteTests {

    private func makeTempStore() -> (SSDCacheStore, URL) {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-async-test-\(UUID().uuidString)")
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

    @Test("saveBlock: index updated synchronously before file lands on disk")
    func indexUpdatedSynchronously() throws {
        let (store, dir) = makeTempStore()
        defer { cleanup(dir) }

        let hash = makeHash("async-test-1")
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

        // Index entry must be visible immediately (added before writeQueue.async)
        #expect(store.hasBlock(hash))

        // File may not exist yet — wait up to 2s for async write to complete
        let deadline = Date().addingTimeInterval(2)
        var loaded: (arrays: [String: [MLXArray]], metadata: [String: String], numLayers: Int)? = nil
        while Date() < deadline {
            loaded = store.loadBlock(hash)
            if loaded != nil { break }
            usleep(50_000)
        }
        #expect(loaded != nil)
        #expect(loaded!.metadata["block_hash"] != nil)
    }
}
