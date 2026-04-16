import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXRandom
import NovaMLXCore
import NovaMLXUtils

public final class CompiledSampler: @unchecked Sendable {
    private let compiledSample: @Sendable (MLXArray) -> MLXArray
    private let temperature: Float
    private let topP: Float
    private let topK: Int
    private let minP: Float
    private let randomState: MLXRandom.RandomState

    public init(temperature: Float, topP: Float = 1.0, topK: Int = 0, minP: Float = 0.0) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.randomState = MLXRandom.RandomState()

        if temperature == 0 {
            compiledSample = compile { logits in
                argMax(logits, axis: -1)
            }
        } else if topP > 0 && topP < 1 {
            let tp = MLXArray(topP)
            let temp = MLXArray(temperature)
            compiledSample = compile { logits in
                var l = logits
                if l.dtype == .bfloat16 { l = l.asType(.float32) }
                let negInf = MLXArray(-Float.infinity)
                let logprobs = logSoftmax(l)
                let sortedIndices = argSort(logprobs, axis: -1)
                let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
                let cumulativeProbs = cumsum(exp(sortedLogprobs), axis: -1)
                let filtered = MLX.where(cumulativeProbs .> (1 - tp), sortedLogprobs, negInf)
                let restored = putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
                return categorical(restored * (1 / temp))
            }
        } else {
            let temp = MLXArray(temperature)
            compiledSample = compile { logits in
                var l = logits
                if l.dtype == .bfloat16 { l = l.asType(.float32) }
                return categorical(logSoftmax(l) * (1 / temp))
            }
        }
    }

    public func sample(logits: MLXArray) -> MLXArray {
        withRandomState(randomState) {
            compiledSample(logits)
        }
    }
}

public final class NGramSpeculator: @unchecked Sendable {
    public struct Config: Sendable {
        public let ngramSize: Int
        public let maxDraft: Int
        public let minHits: Int
        public let maxCacheSize: Int

        public init(
            ngramSize: Int = 4,
            maxDraft: Int = 5,
            minHits: Int = 1,
            maxCacheSize: Int = 65536
        ) {
            self.ngramSize = ngramSize
            self.maxDraft = maxDraft
            self.minHits = minHits
            self.maxCacheSize = maxCacheSize
        }
    }

    private struct CacheEntry {
        var continuations: [Int]
        var hits: Int
    }

    private var cache: [String: CacheEntry] = [:]
    private let config: Config
    private var tokenBuffer: [Int] = []
    private let lock = NovaMLXLock()

    public init(config: Config = Config()) {
        self.config = config
    }

    private func hashKey(_ tokens: [Int]) -> String {
        tokens.map { String($0) }.joined(separator: ":")
    }

    public func record(token: Int) {
        lock.withLock {
            tokenBuffer.append(token)
            if tokenBuffer.count > config.ngramSize + config.maxDraft + 10 {
                tokenBuffer = Array(tokenBuffer.suffix(config.ngramSize + config.maxDraft + 10))
            }
            updateCache()
        }
    }

    public func record(tokens: [Int]) {
        lock.withLock {
            tokenBuffer.append(contentsOf: tokens)
            if tokenBuffer.count > config.ngramSize + config.maxDraft + 10 {
                tokenBuffer = Array(tokenBuffer.suffix(config.ngramSize + config.maxDraft + 10))
            }
            for i in 0..<tokens.count {
                if tokenBuffer.count >= config.ngramSize + 1 + i {
                    let start = tokenBuffer.count - config.ngramSize - 1 - i
                    if start >= 0 {
                        let ngram = Array(tokenBuffer[start..<start + config.ngramSize])
                        let continuation = tokenBuffer[start + config.ngramSize]
                        let key = hashKey(ngram)
                        if var entry = cache[key] {
                            entry.hits += 1
                            if entry.continuations.count < config.maxDraft {
                                if !entry.continuations.contains(continuation) {
                                    entry.continuations.append(continuation)
                                }
                            }
                            cache[key] = entry
                        } else {
                            cache[key] = CacheEntry(continuations: [continuation], hits: 1)
                        }
                    }
                }
            }
            evictIfNeeded()
        }
    }

    private func updateCache() {
        guard tokenBuffer.count > config.ngramSize else { return }
        let start = tokenBuffer.count - config.ngramSize - 1
        guard start >= 0 else { return }
        let ngram = Array(tokenBuffer[start..<start + config.ngramSize])
        let continuation = tokenBuffer[start + config.ngramSize]
        let key = hashKey(ngram)

        if var entry = cache[key] {
            entry.hits += 1
            if entry.continuations.count < config.maxDraft {
                if !entry.continuations.contains(continuation) {
                    entry.continuations.append(continuation)
                }
            }
            cache[key] = entry
        } else {
            cache[key] = CacheEntry(continuations: [continuation], hits: 1)
        }
        evictIfNeeded()
    }

    private func evictIfNeeded() {
        guard cache.count > config.maxCacheSize else { return }
        let sorted = cache.sorted { $0.value.hits < $1.value.hits }
        let toRemove = cache.count - config.maxCacheSize / 2
        for i in 0..<toRemove {
            cache.removeValue(forKey: sorted[i].key)
        }
    }

    public func speculate() -> [Int] {
        lock.withLock {
            guard tokenBuffer.count >= config.ngramSize else { return [] }
            let ngram = Array(tokenBuffer.suffix(config.ngramSize))
            let key = hashKey(ngram)

            guard let entry = cache[key], entry.hits >= config.minHits else {
                return []
            }

            return Array(entry.continuations.prefix(config.maxDraft))
        }
    }

    public func speculate(context: [Int]) -> [Int] {
        lock.withLock {
            guard context.count >= config.ngramSize else { return [] }
            let ngram = Array(context.suffix(config.ngramSize))
            let key = hashKey(ngram)

            guard let entry = cache[key], entry.hits >= config.minHits else {
                return []
            }

            return Array(entry.continuations.prefix(config.maxDraft))
        }
    }

    public var stats: (cacheSize: Int, totalHits: Int, bufferSize: Int) {
        lock.withLock {
            let hits = cache.values.reduce(0) { $0 + $1.hits }
            return (cache.count, hits, tokenBuffer.count)
        }
    }
}

public final class SpeculativeDecoder: @unchecked Sendable {
    private let speculator: NGramSpeculator
    private let compiledSampler: CompiledSampler?
    private var totalSpeculated: UInt64 = 0
    private var totalAccepted: UInt64 = 0
    private let lock = NovaMLXLock()

    public init(ngramConfig: NGramSpeculator.Config = .init(), samplerConfig: CompiledSampler? = nil) {
        self.speculator = NGramSpeculator(config: ngramConfig)
        self.compiledSampler = samplerConfig
    }

    public func speculate(context: [Int]) -> [Int] {
        speculator.speculate(context: context)
    }

    public func recordAccepted(tokens: [Int], accepted: Int) {
        speculator.record(tokens: tokens)
        lock.withLock {
            totalSpeculated += UInt64(tokens.count)
            totalAccepted += UInt64(accepted)
        }
    }

    public func recordToken(_ token: Int) {
        speculator.record(token: token)
    }

    public var acceptanceRate: Double {
        lock.withLock {
            totalSpeculated > 0 ? Double(totalAccepted) / Double(totalSpeculated) : 0
        }
    }

    public var speculatorStats: (cacheSize: Int, totalHits: Int, bufferSize: Int) {
        speculator.stats
    }
}
