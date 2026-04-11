import CryptoKit
import Foundation

public enum BlockHasher {
    private static let rootSeed = Data("NovaMLX-root".utf8)

    public static func computeBlockHash(
        parentHash: BlockHash?,
        tokenIds: [Int],
        modelName: String
    ) -> BlockHash {
        var hasher = Insecure.SHA1()
        if !modelName.isEmpty {
            hasher.update(data: Data(modelName.utf8))
        }
        if let parent = parentHash {
            hasher.update(data: Data(parent))
        } else {
            hasher.update(data: rootSeed)
        }
        withUnsafeBytes(of: tokenIds.count) { hasher.update(bufferPointer: $0) }
        for tokenId in tokenIds {
            withUnsafeBytes(of: Int32(tokenId)) { hasher.update(bufferPointer: $0) }
        }
        let digest = hasher.finalize()
        return Array(digest)
    }

    public static func computeChainHashes(
        tokenIds: [Int],
        blockSize: Int,
        modelName: String
    ) -> [BlockHash] {
        let numFullBlocks = tokenIds.count / blockSize
        guard numFullBlocks > 0 else { return [] }

        var hashes: [BlockHash] = []
        hashes.reserveCapacity(numFullBlocks)

        var parentHash: BlockHash? = nil
        for i in 0..<numFullBlocks {
            let start = i * blockSize
            let end = start + blockSize
            let blockTokens = Array(tokenIds[start..<end])
            let hash = computeBlockHash(
                parentHash: parentHash,
                tokenIds: blockTokens,
                modelName: modelName
            )
            hashes.append(hash)
            parentHash = hash
        }
        return hashes
    }
}
