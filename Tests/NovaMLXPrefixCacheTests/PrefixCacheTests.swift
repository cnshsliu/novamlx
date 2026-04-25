import Testing
import Foundation
@testable import NovaMLXPrefixCache

@Suite("PrefixCache Tests")
struct PrefixCacheTests {

    @Test("BlockHasher produces consistent hashes")
    func blockHasherConsistency() {
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        let hash1 = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: tokens, modelName: "test-model")
        let hash2 = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: tokens, modelName: "test-model")
        #expect(hash1 == hash2)
    }

    @Test("BlockHasher produces different hashes for different tokens")
    func blockHasherDifference() {
        let hash1 = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3], modelName: "test")
        let hash2 = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: [4, 5, 6], modelName: "test")
        #expect(hash1 != hash2)
    }

    @Test("BlockHasher chain hashes depend on parent")
    func blockHasherChain() {
        let tokens = [1, 2, 3]
        let parentA = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: [10, 20], modelName: "test")
        let parentB = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: [30, 40], modelName: "test")
        let hashA = BlockHasher.computeBlockHash(parentHash: parentA, tokenIds: tokens, modelName: "test")
        let hashB = BlockHasher.computeBlockHash(parentHash: parentB, tokenIds: tokens, modelName: "test")
        #expect(hashA != hashB)
    }

    @Test("BlockHasher computes correct number of chain hashes")
    func chainHashesCount() {
        let tokens = Array(0..<256)
        let hashes = BlockHasher.computeChainHashes(tokenIds: tokens, blockSize: 64, modelName: "test")
        #expect(hashes.count == 4)

        let smallTokens = Array(0..<32)
        let smallHashes = BlockHasher.computeChainHashes(tokenIds: smallTokens, blockSize: 64, modelName: "test")
        #expect(smallHashes.isEmpty)
    }

    @Test("BlockHasher model name isolation")
    func blockHasherModelIsolation() {
        let hash1 = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3], modelName: "model-a")
        let hash2 = BlockHasher.computeBlockHash(parentHash: nil, tokenIds: [1, 2, 3], modelName: "model-b")
        #expect(hash1 != hash2)
    }

    @Test("PagedBlockPool allocates and frees blocks")
    func poolAllocateFree() {
        let pool = PagedBlockPool(blockSize: 64, maxBlocks: 100, initialBlocks: 10, modelName: "test")
        let block = pool.allocateBlock()
        #expect(block != nil)
        #expect(block!.refCount == 1)

        let (total, allocated, free, _) = pool.getStats()
        #expect(total == 10)
        #expect(allocated == 1)
        #expect(free == 8)
    }

    @Test("PagedBlockPool findSharedPrefix returns empty for new tokens")
    func poolNoMatch() {
        let pool = PagedBlockPool(blockSize: 64, maxBlocks: 100, initialBlocks: 10, modelName: "test")
        let tokens = Array(0..<128)
        let (ids, remaining) = pool.findSharedPrefix(tokenIds: tokens)
        #expect(ids.isEmpty)
        #expect(remaining == 128)
    }

    @Test("PagedBlockPool register and find prefix")
    func poolRegisterAndFind() {
        let pool = PagedBlockPool(blockSize: 4, maxBlocks: 100, initialBlocks: 10, modelName: "test")
        let block = pool.allocateBlock()!
        let tokens = Array(0..<4)
        pool.registerBlock(block: block, tokenIds: tokens, parentHash: nil)

        let searchTokens = Array(0..<8)
        let (ids, remaining) = pool.findSharedPrefix(tokenIds: searchTokens)
        #expect(ids.count == 1)
        #expect(remaining == 4)
    }

    @Test("PagedBlockPool reference counting")
    func poolRefCount() {
        let pool = PagedBlockPool(blockSize: 4, maxBlocks: 100, initialBlocks: 10, modelName: "test")
        let block = pool.allocateBlock()!
        #expect(block.refCount == 1)

        pool.incrementRef(block.blockId)
        #expect(block.refCount == 2)
        #expect(block.isShared())

        pool.freeBlock(block.blockId)
        #expect(block.refCount == 1)

        pool.freeBlock(block.blockId)
        let (_, _, free, _) = pool.getStats()
        #expect(free == 9)
    }

    @Test("PagedBlockPool block table fork increments refs")
    func poolForkTable() {
        let pool = PagedBlockPool(blockSize: 4, maxBlocks: 100, initialBlocks: 10, modelName: "test")
        let block = pool.allocateBlock()!
        var table = pool.createBlockTable(requestId: "req-1")
        table.blockIds.append(block.blockId)
        table.numTokens = 4

        let forked = pool.forkBlockTable(source: table, newRequestId: "req-2")
        #expect(forked != nil)
        #expect(block.refCount == 2)
    }

    @Test("PagedBlockPool clear resets everything")
    func poolClear() {
        let pool = PagedBlockPool(blockSize: 4, maxBlocks: 100, initialBlocks: 10, modelName: "test")
        let _ = pool.allocateBlock()
        let _ = pool.allocateBlock()
        pool.clear()

        let (total, _, free, _) = pool.getStats()
        #expect(total == 10)
        #expect(free == 9)
    }

    @Test("PrefixCacheConfig defaults are reasonable")
    func configDefaults() {
        let config = PrefixCacheConfig.default
        #expect(config.blockSize == 64)
        #expect(config.maxBlocks == 4096)
        #expect(config.initialBlocks == 256)
        #expect(config.ssdCacheDir == nil)
        #expect(config.ssdMaxSizeBytes == 5 * 1024 * 1024 * 1024)
    }

    @Test("PrefixCacheStats default init")
    func statsDefaults() {
        let stats = PrefixCacheStats()
        #expect(stats.hits == 0)
        #expect(stats.misses == 0)
        #expect(stats.tokensSaved == 0)
        #expect(stats.totalBlocks == 0)
    }

    @Test("FreeBlockQueue operations")
    func freeQueueOps() {
        let blocks = (0..<5).map { CacheBlock(blockId: $0) }
        let queue = FreeBlockQueue(blocks: blocks)

        #expect(queue.numFree == 5)

        let first = queue.popLeft()
        #expect(first != nil)
        #expect(first!.blockId == 0)
        #expect(queue.numFree == 4)

        let batch = queue.popLeftN(2)
        #expect(batch.count == 2)
        #expect(queue.numFree == 2)

        queue.append(first!)
        #expect(queue.numFree == 3)
    }

    @Test("BlockHashMap operations")
    func hashMapOps() {
        let map = BlockHashMap()
        let hash: BlockHash = [0x01, 0x02, 0x03]
        let block = CacheBlock(blockId: 42)

        #expect(map.get(hash) == nil)
        map.insert(hash, block: block)
        #expect(map.get(hash) != nil)
        #expect(map.get(hash)!.blockId == 42)
        #expect(map.count == 1)

        let removed = map.remove(hash)
        #expect(removed != nil)
        #expect(map.count == 0)
    }

    @Test("SSDCacheIndex LRU eviction")
    func ssdIndexEviction() {
        let index = SSDCacheIndex(maxSize: 1000)
        let hash1: BlockHash = [0x01]
        let hash2: BlockHash = [0x02]

        index.add(SSDIndexEntry(blockHash: hash1, filePath: URL(fileURLWithPath: "/a"), fileSize: 800, tokenCount: 64, lastAccess: Date()))
        index.add(SSDIndexEntry(blockHash: hash2, filePath: URL(fileURLWithPath: "/b"), fileSize: 800, tokenCount: 64, lastAccess: Date()))

        let evicted = index.evictLRU()
        #expect(evicted.count == 1)
        #expect(evicted[0].blockHash == hash1)
        #expect(index.count == 1)
    }
}
