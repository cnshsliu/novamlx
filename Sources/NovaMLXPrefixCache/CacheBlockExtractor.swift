import Foundation
import MLX
import MLXLMCommon

public enum CacheBlockExtractor {

    public static func extractBlockSlices(
        cache: [KVCache],
        startIdx: Int,
        endIdx: Int
    ) -> [[MLXArray]] {
        var result: [[MLXArray]] = []
        result.reserveCapacity(cache.count)

        for layer in cache {
            let state = layer.state
            var sliced: [MLXArray] = []
            sliced.reserveCapacity(state.count)

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
        for blockData in blockDataList {
            for i in 0..<min(blockData.count, numLayers) {
                let layerState = blockData[i]
                if mergedLayers[i].isEmpty {
                    mergedLayers[i] = layerState
                } else if mergedLayers[i].count == layerState.count {
                    for j in 0..<layerState.count {
                        mergedLayers[i][j] = concatenated([mergedLayers[i][j], layerState[j]], axis: 2)
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
        cache.map { layer -> String in
            switch layer {
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
        cache.map { $0.metaState }
    }
}
