import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NovaMLXUtils

public final class FusedBatchKVCache: KVCache {
    private struct PerSequenceEntry {
        var cache: KVCacheSimple
        var logicalLength: Int
        let id: UUID
        var finished: Bool
    }

    private var entries: [PerSequenceEntry] = []
    private let maxBatchSize: Int

    public init(maxBatchSize: Int = 8) {
        self.maxBatchSize = maxBatchSize
    }

    public var offset: Int { maxLength }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { false }

    public var state: [MLXArray] {
        get { [] }
        set { _ = newValue }
    }

    public var metaState: [String] {
        get { [""] }
        set { _ = newValue }
    }

    public func innerState() -> [MLXArray] { [] }
    public func trim(_ n: Int) -> Int { 0 }

    public func copy() -> any KVCache {
        fatalError("FusedBatchKVCache does not support copy")
    }

    public func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let active = entries.filter { !$0.finished }
        guard !active.isEmpty else { return .none }

        if active.count == 1 {
            return active[0].cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
        }

        let offsets = active.map(\.cache.offset)
        let maxOffset = offsets.max() ?? 0

        if n == 1 && offsets.allSatisfy({ $0 == maxOffset }) { return .none }

        return .array(createBatchedMask(batchSize: active.count, n: n, offsets: offsets, maxLen: maxOffset + n))
    }

    public var activeCount: Int { entries.filter { !$0.finished }.count }
    public var batchSize: Int { entries.count }

    public var maxLength: Int {
        entries.filter { !$0.finished }.map(\.logicalLength).max() ?? 0
    }

    @discardableResult
    public func addSequence(id: UUID, prefillCache: KVCacheSimple) -> Int {
        let entry = PerSequenceEntry(
            cache: prefillCache,
            logicalLength: prefillCache.offset,
            id: id,
            finished: false
        )
        entries.append(entry)
        return entries.count - 1
    }

    public func removeSequence(id: UUID) {
        if let idx = entries.firstIndex(where: { $0.id == id }) {
            entries[idx].finished = true
        }
    }

    /// Remove all finished entries and free their slots.
    public func compact() {
        entries.removeAll { $0.finished }
    }

    public func getCache(id: UUID) -> KVCacheSimple? {
        entries.first { $0.id == id }?.cache
    }

    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let active = entries.enumerated().filter { !$0.element.finished }
        guard !active.isEmpty else { return (keys, values) }

        if active.count == 1, let (idx, _) = active.first {
            entries[idx].logicalLength = entries[idx].cache.offset + keys.dim(2)
            return entries[idx].cache.update(keys: keys, values: values)
        }

        var resultKeys: [MLXArray] = []
        var resultValues: [MLXArray] = []
        let splitKeys = keys.split(parts: active.count, axis: 0)
        let splitVals = values.split(parts: active.count, axis: 0)

        for (batchIdx, (entryIdx, entry)) in active.enumerated() {
            guard batchIdx < splitKeys.count else { break }
            let (rk, rv) = entry.cache.update(keys: splitKeys[batchIdx], values: splitVals[batchIdx])
            entries[entryIdx].logicalLength = entry.cache.offset
            resultKeys.append(rk)
            resultValues.append(rv)
        }

        let maxLen = resultKeys.map { $0.dim(2) }.max() ?? 0
        return (padAndStack(resultKeys, maxLen: maxLen), padAndStack(resultValues, maxLen: maxLen))
    }

    private func padAndStack(_ arrays: [MLXArray], maxLen: Int) -> MLXArray {
        let padded = arrays.map { arr -> MLXArray in
            let cur = arr.dim(2)
            guard cur < maxLen else { return arr }
            var shape = arr.shape
            shape[2] = maxLen - cur
            return concatenated([arr, MLXArray.zeros(shape, dtype: arr.dtype)], axis: 2)
        }
        return stacked(padded, axis: 0)
    }
}

func createBatchedMask(batchSize: Int, n: Int, offsets: [Int], maxLen: Int) -> MLXArray {
    var masks: [MLXArray] = []
    for i in 0..<batchSize {
        let offset = offsets[i]
        let seqLen = offset + n

        let rinds = MLXArray(Int32(0) ..< Int32(maxLen))[.newAxis]
        let linds = MLXArray(Int32(offset) ..< Int32(offset + n))[0..., .newAxis]
        var mask = linds .>= rinds
        mask = mask & (rinds .< Int32(seqLen))

        masks.append(mask)
    }
    return stacked(masks, axis: 0)
}

public final class FusedBatchDecoder: @unchecked Sendable {
    public struct SequenceHandle: Sendable {
        public let id: UUID
        public let index: Int
    }

    private var caches: [Int: FusedBatchKVCache] = [:]
    private let lock = NovaMLXLock()

    public init() {}

    public func createCache(modelId: String, maxBatchSize: Int = 8) -> FusedBatchKVCache {
        let key = modelId.hashValue
        let cache = FusedBatchKVCache(maxBatchSize: maxBatchSize)
        lock.withLock { caches[key] = cache }
        return cache
    }

    public func getCache(modelId: String) -> FusedBatchKVCache? {
        lock.withLock { caches[modelId.hashValue] }
    }
}
