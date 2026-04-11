import Foundation
import NovaMLXUtils

final class FreeBlockQueue: @unchecked Sendable {
    private let sentinelHead = CacheBlock(blockId: -1)
    private let sentinelTail = CacheBlock(blockId: -2)
    private(set) var numFree: Int = 0

    init(blocks: [CacheBlock]) {
        numFree = blocks.count
        for i in 0..<blocks.count {
            if i > 0 { blocks[i].prevFree = blocks[i - 1] }
            if i < blocks.count - 1 { blocks[i].nextFree = blocks[i + 1] }
        }
        if blocks.isEmpty {
            sentinelHead.nextFree = sentinelTail
            sentinelTail.prevFree = sentinelHead
        } else {
            sentinelHead.nextFree = blocks[0]
            blocks[0].prevFree = sentinelHead
            sentinelTail.prevFree = blocks[blocks.count - 1]
            blocks[blocks.count - 1].nextFree = sentinelTail
        }
    }

    func popLeft() -> CacheBlock? {
        guard let block = sentinelHead.nextFree, block !== sentinelTail else { return nil }
        sentinelHead.nextFree = block.nextFree
        block.nextFree?.prevFree = sentinelHead
        block.prevFree = nil
        block.nextFree = nil
        numFree -= 1
        return block
    }

    func popLeftN(_ n: Int) -> [CacheBlock] {
        guard n > 0 else { return [] }
        var result: [CacheBlock] = []
        var current = sentinelHead.nextFree
        for _ in 0..<n {
            guard let block = current, block !== sentinelTail else { break }
            result.append(block)
            let next = block.nextFree
            block.prevFree = nil
            block.nextFree = nil
            current = next
        }
        sentinelHead.nextFree = current
        current?.prevFree = sentinelHead
        numFree -= result.count
        return result
    }

    func remove(_ block: CacheBlock) {
        block.prevFree?.nextFree = block.nextFree
        block.nextFree?.prevFree = block.prevFree
        block.prevFree = nil
        block.nextFree = nil
        numFree -= 1
    }

    func append(_ block: CacheBlock) {
        let last = sentinelTail.prevFree!
        last.nextFree = block
        block.prevFree = last
        block.nextFree = sentinelTail
        sentinelTail.prevFree = block
        numFree += 1
    }

    func appendN(_ blocks: [CacheBlock]) {
        guard !blocks.isEmpty else { return }
        let last = sentinelTail.prevFree!
        for block in blocks {
            block.prevFree = last
            last.nextFree = block
        }
        blocks[blocks.count - 1].nextFree = sentinelTail
        sentinelTail.prevFree = blocks[blocks.count - 1]
        numFree += blocks.count
    }
}

final class BlockHashMap: @unchecked Sendable {
    private var map: [String: CacheBlock] = [:]

    private func key(_ hash: BlockHash) -> String {
        hash.map { String(format: "%02x", $0) }.joined()
    }

    func get(_ hash: BlockHash) -> CacheBlock? {
        map[key(hash)]
    }

    func insert(_ hash: BlockHash, block: CacheBlock) {
        map[key(hash)] = block
    }

    func remove(_ hash: BlockHash) -> CacheBlock? {
        map.removeValue(forKey: key(hash))
    }

    var count: Int { map.count }

    func clear() { map.removeAll() }
}

public final class PagedBlockPool: @unchecked Sendable {
    public let blockSize: Int
    public let maxBlocks: Int
    public let modelName: String
    private let initialBlocks: Int

    private var blocks: [CacheBlock] = []
    private var freeQueue: FreeBlockQueue
    private var hashMap = BlockHashMap()
    private var allocatedBlocks: [Int: CacheBlock] = [:]
    private var requestTables: [String: BlockTable] = [:]
    private var currentAllocatedCount: Int
    private let lock = NovaMLXLock()

    private var nullBlock: CacheBlock!

    public init(blockSize: Int = 64, maxBlocks: Int = 4096, initialBlocks: Int = 256, modelName: String = "") {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.modelName = modelName
        self.initialBlocks = min(initialBlocks, maxBlocks)

        let count = self.initialBlocks
        self.currentAllocatedCount = count
        self.blocks = (0..<count).map { CacheBlock(blockId: $0) }
        self.freeQueue = FreeBlockQueue(blocks: self.blocks)

        self.nullBlock = self.freeQueue.popLeft()!
        self.nullBlock.isNull = true
        self.nullBlock.refCount = 1
        self.allocatedBlocks[self.nullBlock.blockId] = self.nullBlock
    }

    public func allocateBlock() -> CacheBlock? {
        lock.withLock {
            if freeQueue.numFree == 0 {
                _ = growBlocks(min(256, maxBlocks - currentAllocatedCount))
            }
            guard let block = freeQueue.popLeft() else { return nil }
            if block.blockHash != nil {
                _ = hashMap.remove(block.blockHash!)
                block.resetHash()
            }
            block.refCount = 1
            block.touch()
            allocatedBlocks[block.blockId] = block
            return block
        }
    }

    public func freeBlock(_ blockId: Int) {
        lock.withLock {
            guard let block = allocatedBlocks[blockId], !block.isNull else { return }
            block.refCount -= 1
            if block.refCount <= 0 {
                if block.blockHash != nil { _ = hashMap.remove(block.blockHash!) }
                allocatedBlocks.removeValue(forKey: blockId)
                freeQueue.append(block)
            }
        }
    }

    public func incrementRef(_ blockId: Int) {
        lock.withLock {
            guard let block = allocatedBlocks[blockId] else { return }
            block.refCount += 1
            block.touch()
        }
    }

    public func findSharedPrefix(tokenIds: [Int]) -> (blockIds: [Int], remainingTokenCount: Int) {
        lock.withLock {
            let numFullBlocks = tokenIds.count / blockSize
            var matchedIds: [Int] = []
            var parentHash: BlockHash? = nil

            for i in 0..<numFullBlocks {
                let start = i * blockSize
                let end = start + blockSize
                let blockTokens = Array(tokenIds[start..<end])
                let hash = BlockHasher.computeBlockHash(
                    parentHash: parentHash,
                    tokenIds: blockTokens,
                    modelName: modelName
                )
                guard let block = hashMap.get(hash) else { break }
                matchedIds.append(block.blockId)
                parentHash = hash
            }

            let cachedTokens = matchedIds.count * blockSize
            return (matchedIds, tokenIds.count - cachedTokens)
        }
    }

    public func registerBlock(block: CacheBlock, tokenIds: [Int], parentHash: BlockHash?) {
        lock.withLock {
            let hash = BlockHasher.computeBlockHash(
                parentHash: parentHash,
                tokenIds: tokenIds,
                modelName: modelName
            )
            block.blockHash = hash
            block.tokenCount = tokenIds.count
            hashMap.insert(hash, block: block)
        }
    }

    public func getBlock(_ blockId: Int) -> CacheBlock? {
        lock.withLock { allocatedBlocks[blockId] }
    }

    public func createBlockTable(requestId: String) -> BlockTable {
        lock.withLock {
            let table = BlockTable(requestId: requestId)
            requestTables[requestId] = table
            return table
        }
    }

    public func deleteBlockTable(requestId: String) {
        lock.withLock {
            guard let table = requestTables.removeValue(forKey: requestId) else { return }
            for blockId in table.blockIds { freeBlock(blockId) }
        }
    }

    public func forkBlockTable(source: BlockTable, newRequestId: String) -> BlockTable? {
        lock.withLock {
            let forked = source.copy(newRequestId: newRequestId)
            for blockId in forked.blockIds { incrementRef(blockId) }
            requestTables[newRequestId] = forked
            return forked
        }
    }

    public var freeBlockCount: Int { lock.withLock { freeQueue.numFree } }

    public func getStats() -> (totalBlocks: Int, allocated: Int, free: Int, shared: Int) {
        lock.withLock {
            let allocated = allocatedBlocks.count - 1
            let shared = allocatedBlocks.values.filter { $0.refCount > 1 }.count
            return (currentAllocatedCount, allocated, freeQueue.numFree, shared)
        }
    }

    public func clear() {
        lock.withLock {
            let count = min(initialBlocks, maxBlocks)
            currentAllocatedCount = count
            blocks = (0..<count).map { CacheBlock(blockId: $0) }
            freeQueue = FreeBlockQueue(blocks: blocks)
            hashMap.clear()
            requestTables.removeAll()
            allocatedBlocks.removeAll()
            nullBlock = freeQueue.popLeft()!
            nullBlock.isNull = true
            nullBlock.refCount = 1
            allocatedBlocks[nullBlock.blockId] = nullBlock
        }
    }

    private func growBlocks(_ additional: Int) -> Int {
        let available = maxBlocks - currentAllocatedCount
        let toCreate = min(additional, available)
        guard toCreate > 0 else { return 0 }
        let startId = currentAllocatedCount
        let newBlocks = (startId..<(startId + toCreate)).map { CacheBlock(blockId: $0) }
        blocks.append(contentsOf: newBlocks)
        freeQueue.appendN(newBlocks)
        currentAllocatedCount += toCreate
        return toCreate
    }
}
