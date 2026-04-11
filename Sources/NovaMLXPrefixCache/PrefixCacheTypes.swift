import Foundation

public typealias BlockHash = [UInt8]

public struct PrefixCacheConfig: Sendable {
    public let blockSize: Int
    public let maxBlocks: Int
    public let initialBlocks: Int
    public let ssdCacheDir: URL?
    public let ssdMaxSizeBytes: UInt64
    public let hotCacheMaxBytes: Int

    public init(
        blockSize: Int = 64,
        maxBlocks: Int = 4096,
        initialBlocks: Int = 256,
        ssdCacheDir: URL? = nil,
        ssdMaxSizeBytes: UInt64 = 100 * 1024 * 1024 * 1024,
        hotCacheMaxBytes: Int = 0
    ) {
        self.blockSize = blockSize
        self.maxBlocks = maxBlocks
        self.initialBlocks = min(initialBlocks, maxBlocks)
        self.ssdCacheDir = ssdCacheDir
        self.ssdMaxSizeBytes = ssdMaxSizeBytes
        self.hotCacheMaxBytes = hotCacheMaxBytes
    }

    public static let `default` = PrefixCacheConfig()
}

public final class CacheBlock: @unchecked Sendable {
    public let blockId: Int
    public var refCount: Int
    public var blockHash: BlockHash?
    public var tokenCount: Int
    public var lastAccess: Date

    public var prevFree: CacheBlock?
    public var nextFree: CacheBlock?
    public var isNull: Bool

    public init(blockId: Int) {
        self.blockId = blockId
        self.refCount = 0
        self.blockHash = nil
        self.tokenCount = 0
        self.lastAccess = Date()
        self.prevFree = nil
        self.nextFree = nil
        self.isNull = false
    }

    public func isShared() -> Bool { refCount > 1 }

    public func touch() { lastAccess = Date() }

    public func resetHash() { blockHash = nil }
}

public struct BlockTable: Sendable {
    public let requestId: String
    public var blockIds: [Int]
    public var numTokens: Int

    public init(requestId: String) {
        self.requestId = requestId
        self.blockIds = []
        self.numTokens = 0
    }

    public func copy(newRequestId: String) -> BlockTable {
        var copy = BlockTable(requestId: newRequestId)
        copy.blockIds = blockIds
        copy.numTokens = numTokens
        return copy
    }
}

public struct PrefixCacheStats: Sendable {
    public var hits: Int
    public var misses: Int
    public var tokensSaved: Int
    public var evictions: Int
    public var totalBlocks: Int
    public var allocatedBlocks: Int
    public var freeBlocks: Int
    public var sharedBlocks: Int
    public var ssdBlockCount: Int
    public var ssdTotalSize: UInt64
    public var hotCacheBlockCount: Int
    public var hotCacheTotalBytes: Int

    public init(
        hits: Int = 0, misses: Int = 0, tokensSaved: Int = 0, evictions: Int = 0,
        totalBlocks: Int = 0, allocatedBlocks: Int = 0, freeBlocks: Int = 0,
        sharedBlocks: Int = 0, ssdBlockCount: Int = 0, ssdTotalSize: UInt64 = 0,
        hotCacheBlockCount: Int = 0, hotCacheTotalBytes: Int = 0
    ) {
        self.hits = hits
        self.misses = misses
        self.tokensSaved = tokensSaved
        self.evictions = evictions
        self.totalBlocks = totalBlocks
        self.allocatedBlocks = allocatedBlocks
        self.freeBlocks = freeBlocks
        self.sharedBlocks = sharedBlocks
        self.ssdBlockCount = ssdBlockCount
        self.ssdTotalSize = ssdTotalSize
        self.hotCacheBlockCount = hotCacheBlockCount
        self.hotCacheTotalBytes = hotCacheTotalBytes
    }
}

struct BlockMetadata: Codable, Sendable {
    let blockHashHex: String
    let tokenCount: Int
    let numLayers: Int
    let modelName: String
    let layerCacheTypes: [String]?
    let layerMetaStates: [[String]]?
    let createdAt: Double
}
