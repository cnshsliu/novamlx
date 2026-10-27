import Foundation
import MLX

/// Long-term prefix block store. `SSDCacheStore` is the file-backed
/// implementation. Tests supply a scripted store so a slow `loadBlock` can
/// prove `fetchPrefix` does not wait on it.
public protocol PrefixCacheDurableStore: AnyObject {
    func hasBlock(_ hash: BlockHash) -> Bool
    func saveBlock(
        hash: BlockHash,
        cacheData: [[MLXArray]],
        layerClasses: [String],
        metaStates: [[String]],
        tokenCount: Int,
        modelName: String
    )
    func loadBlock(_ hash: BlockHash) -> (arrays: [String: [MLXArray]], metadata: [String: String], numLayers: Int)?
    func discardExpired(now: Date)
    func clear()
    var blockCount: Int { get }
    var totalSize: UInt64 { get }
}

extension SSDCacheStore: PrefixCacheDurableStore {}
