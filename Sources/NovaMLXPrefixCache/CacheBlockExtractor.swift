import Foundation
import MLX
import MLXLMCommon

/// A cache wrapper whose stored prefix-cache identity is another layer.
///
/// Decode proxies keep the scheduler's trim target stable, but prefix cache
/// records the concrete class (`RotatingKVCache` vs full-attention `KVCache`).
/// Classifying the wrapper itself restores a sliding window as full attention.
public protocol PrefixCacheLayerUnwrapping: KVCache {
    var prefixCacheLayer: any KVCache { get }
}

public enum CacheBlockExtractor {

    /// The layer prefix cache should record. Wrappers unwrap once.
    public static func canonicalLayer(_ layer: KVCache) -> KVCache {
        if let unwrapping = layer as? any PrefixCacheLayerUnwrapping {
            return unwrapping.prefixCacheLayer
        }
        return layer
    }

    /// True when any layer is a sliding-window ring, including through a wrapper.
    public static func usesRotatingKV(_ cache: [KVCache]) -> Bool {
        cache.contains { canonicalLayer($0) is RotatingKVCache }
    }

    public static func extractBlockSlices(
        cache: [KVCache],
        startIdx: Int,
        endIdx: Int
    ) -> [[MLXArray]] {
        var result: [[MLXArray]] = []
        result.reserveCapacity(cache.count)

        for wrapped in cache {
            let layer = canonicalLayer(wrapped)
            let state = layer.state
            var sliced: [MLXArray] = []
            sliced.reserveCapacity(state.count)

            // KV-type caches have sequence-dependent 4D arrays [B, heads, seqLen, dim].
            // MambaCache/ArraysCache have fixed-size state — must NOT be sliced.
            let isKVType = layer is KVCacheSimple
                || layer is RotatingKVCache
                || layer is ChunkedKVCache
                || layer is QuantizedKVCache

            if isKVType {
                for array in state {
                    if array.ndim == 4 && array.shape[2] >= endIdx && startIdx < endIdx {
                        let slice = array[.ellipsis, ..<endIdx, 0...]
                        sliced.append(slice)
                    } else if array.ndim == 4 && array.shape[2] > startIdx {
                        let actualEnd = min(endIdx, array.shape[2])
                        let slice = array[.ellipsis, startIdx..<actualEnd, 0...]
                        sliced.append(slice)
                    } else {
                        sliced.append(array)
                    }
                }
            } else {
                // Fixed-size state (MambaCache, ArraysCache, etc.) — store as-is
                for array in state {
                    sliced.append(array)
                }
            }
            result.append(sliced)
        }
        return result
    }

    public static func reconstructKVCache(
        blockDataList: [[[MLXArray]]],
        layerClasses: [String],
        metaStatesList: [[String]]
    ) -> [KVCache]? {
        let numLayers = layerClasses.count
        guard numLayers > 0, !blockDataList.isEmpty else { return nil }

        var mergedLayers: [[MLXArray]] = Array(repeating: [], count: numLayers)
        let isKVMask: [Bool] = (0..<numLayers).map { i in
            let cls = i < layerClasses.count ? layerClasses[i] : "KVCache"
            return cls == "KVCache" || cls == "RotatingKVCache" || cls == "ChunkedKVCache" || cls == "QuantizedKVCache"
        }

        for blockData in blockDataList {
            for i in 0..<min(blockData.count, numLayers) {
                let layerState = blockData[i]
                if mergedLayers[i].isEmpty {
                    mergedLayers[i] = layerState
                } else if mergedLayers[i].count == layerState.count {
                    if isKVMask[i] {
                        // KV caches: concatenate along sequence axis
                        for j in 0..<layerState.count {
                            mergedLayers[i][j] = concatenated([mergedLayers[i][j], layerState[j]], axis: 2)
                        }
                    } else {
                        // Fixed-size state (MambaCache, etc.): use latest block only
                        mergedLayers[i] = layerState
                    }
                }
            }
        }

        var result: [KVCache] = []
        result.reserveCapacity(numLayers)

        for i in 0..<numLayers {
            let className = i < layerClasses.count ? layerClasses[i] : "KVCache"
            let metaState = i < metaStatesList.count ? metaStatesList[i] : []
            let stateArrays = mergedLayers[i]

            var cache: KVCache
            switch className {
            case "RotatingKVCache":
                var maxSize = 2048
                if metaState.count >= 2, let v = Int(metaState[1]) { maxSize = v }
                cache = RotatingKVCache(maxSize: maxSize)
                if !metaState.isEmpty { cache.metaState = metaState }
                if stateArrays.count == 2 { cache.state = stateArrays }
            case "QuantizedKVCache":
                cache = QuantizedKVCache()
                if !metaState.isEmpty { cache.metaState = metaState }
                if !stateArrays.isEmpty { cache.state = stateArrays }
            case "ChunkedKVCache":
                cache = ChunkedKVCache()
                if !metaState.isEmpty { cache.metaState = metaState }
                if stateArrays.count == 2 { cache.state = stateArrays }
            case "MambaCache":
                let mamba = MambaCache()
                if !stateArrays.isEmpty { mamba.state = stateArrays }
                cache = mamba
            case "ArraysCache":
                let ac = ArraysCache(size: max(stateArrays.count, 1))
                if !stateArrays.isEmpty { ac.state = stateArrays }
                cache = ac
            default:
                cache = KVCacheSimple()
                if stateArrays.count == 2 { cache.state = stateArrays }
            }
            result.append(cache)
        }
        return result
    }

    public static func getLayerClasses(from cache: [KVCache]) -> [String] {
        cache.map { wrapped -> String in
            switch canonicalLayer(wrapped) {
            case is ChunkedKVCache: "ChunkedKVCache"
            case is KVCacheSimple: "KVCache"
            case is RotatingKVCache: "RotatingKVCache"
            case is QuantizedKVCache: "QuantizedKVCache"
            case is MambaCache: "MambaCache"
            case is ArraysCache: "ArraysCache"
            case is CacheList: "CacheList"
            default: "KVCache"
            }
        }
    }

    public static func getMetaStates(from cache: [KVCache]) -> [[String]] {
        cache.map { canonicalLayer($0).metaState }
    }
}
