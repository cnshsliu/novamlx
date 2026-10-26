import Foundation
import NovaMLXCore

/// Token-level shared prefix for a Flint session.
///
/// This is `rec` `prefix.py` (`select_prefix`, checkpoints, `prepare_prefix`,
/// `PrefillCancelled`, `_take_prefix`). It is not `NovaMLXPrefixCache`: that
/// store is a block-hash SSD cache and does not implement this protocol.
///
/// Checkpoints keep at most two positions, matching `PromptCache.checkpoint`.
/// When the prefill KV snapshot is still trimmable, a match shorter than the
/// cached prompt is reused up to `common` tokens so two chats that share a
/// system and tool prefix report `cached_prompt_tokens > 0`.
enum FlintPrefixEngine {
    /// Official `BLOCK_TOKENS` / prefill chunk (`cache.py`).
    static let productionChunkSize = 65536

    struct Checkpoint: Equatable, Sendable {
        var position: Int
    }

    struct Cache: Equatable, Sendable {
        var tokens: [Int] = []
        var checkpoints: [Checkpoint] = []

        mutating func checkpoint() {
            guard !tokens.isEmpty else { return }
            checkpoints = Array(checkpoints.suffix(1))
            checkpoints.append(Checkpoint(position: tokens.count))
        }
    }

    struct Selection: Equatable, Sendable {
        var cache: Cache?
        var common: Int
        var cached: Int
    }

    struct Preparation: Equatable, Sendable {
        var origin: Cache
        var tokens: [Int]
        var cache: Cache?
    }

    /// `select_prefix`. `trimmable` extends a partial match to `common` when
    /// the KV snapshot can drop the divergent suffix. Sliding-ring checkpoints
    /// stay on the official rewind (`common=9`, `reused=8`) when this is false.
    static func select(prefix: Cache?, tokens: [Int], trimmable: Bool) -> Selection {
        var prefix = prefix
        var common = 0
        if var current = prefix {
            for pair in zip(current.tokens, tokens) {
                if pair.0 != pair.1 { break }
                common += 1
            }
            if common != current.tokens.count {
                if let checkpoint = current.checkpoints.reversed().first(where: { $0.position <= common }) {
                    if trimmable, common > checkpoint.position {
                        current.tokens = Array(current.tokens.prefix(common))
                        current.checkpoints = current.checkpoints.filter { $0.position < common }
                        prefix = current
                    } else {
                        current.tokens = Array(current.tokens.prefix(checkpoint.position))
                        current.checkpoints = current.checkpoints.filter { $0.position < checkpoint.position }
                        prefix = current
                    }
                } else if trimmable && common > 0 {
                    current.tokens = Array(current.tokens.prefix(common))
                    current.checkpoints = current.checkpoints.filter { $0.position < common }
                    prefix = current
                } else {
                    prefix = nil
                }
            } else {
                prefix = current
            }
        }
        let cached = prefix.map { $0.tokens.count } ?? 0
        return Selection(cache: prefix, common: common, cached: cached)
    }

    static func prefillLog(prompt: Int, common: Int, reused: Int) -> String {
        "Prefill: prompt=\(prompt) common_prefix=\(common) reused=\(reused) new=\(prompt - reused)"
    }
}

/// One Flint model's prompt cache and the background `prepare_prefix` job.
struct FlintPrefixSession: Equatable, Sendable {
    var promptCache: FlintPrefixEngine.Cache?
    var preparation: FlintPrefixEngine.Preparation?
    var chunkSize: Int
    var lastCommon = 0
    var lastCached = 0

    init(chunkSize: Int = FlintPrefixEngine.productionChunkSize) {
        self.chunkSize = chunkSize
    }

    /// `_take_prefix` then the caller's prefill runs `select`.
    mutating func takePrefix(ids: [Int], trimmable: Bool) -> FlintPrefixEngine.Cache? {
        let prompt = promptCache
        promptCache = nil
        let pending = preparation
        preparation = nil
        var selected = prompt
        if let pending, let prepared = pending.cache {
            var candidate: FlintPrefixEngine.Cache? = prepared
            if Array(ids.prefix(prepared.tokens.count)) != prepared.tokens {
                candidate = FlintPrefixEngine.select(
                    prefix: prepared, tokens: ids, trimmable: trimmable
                ).cache
            }
            if let candidate, candidate.tokens.count > pending.origin.tokens.count {
                selected = candidate
            }
            // A mismatch that cannot beat the origin keeps `prompt` (the anchor).
            // The preparation is cleared, matching `_take_prefix`.
            _ = pending
        }
        return selected
    }

    /// Record the selection `prefill` would log. Does not replace `promptCache`
    /// until `commit` — a cancelled prefill must not poison the anchor.
    mutating func noteSelection(_ selection: FlintPrefixEngine.Selection) {
        lastCommon = selection.common
        lastCached = selection.cached
    }

    /// Successful prefill replaces the anchor with the prompt that was prefilled.
    mutating func commit(tokens: [Int]) {
        var cache = FlintPrefixEngine.Cache(tokens: tokens)
        if !tokens.isEmpty {
            cache.checkpoint()
        }
        promptCache = cache
    }

    /// `_schedule_prefix`. Short generations never schedule.
    mutating func schedulePrefix(ids: [Int], generated: [Int]) {
        guard generated.count >= chunkSize, let promptCache else { return }
        preparation = FlintPrefixEngine.Preparation(
            origin: promptCache, tokens: ids + generated, cache: nil)
    }

    struct PrepareResult: Equatable, Sendable {
        var preparedTokens: Int
    }

    /// `prepare_prefix`. `advance` is one chunk. A cancel whose resulting
    /// token count is not the chunk target leaves the committed preparation
    /// cache untouched.
    mutating func preparePrefix(
        shouldStop: () -> Bool,
        advance: (FlintPrefixEngine.Cache, Int) throws -> FlintPrefixEngine.Cache
    ) rethrows -> PrepareResult? {
        guard preparation != nil else { return nil }
        let originCount = preparation!.origin.tokens.count
        let end = originCount + (preparation!.tokens.count - originCount) / chunkSize * chunkSize
        var prepared = 0
        while currentPreparedCount(originCount) < end {
            if shouldStop() { break }
            let prefix = preparation!.cache ?? preparation!.origin
            let target = prefix.tokens.count + chunkSize
            do {
                let next = try advance(prefix, target)
                if next.tokens.count != target { break }
                preparation!.cache = next
                prepared += chunkSize
            } catch let error as FlintPrefixCancellation {
                _ = error
                break
            }
        }
        return PrepareResult(preparedTokens: prepared)
    }

    /// `_generate_tokens` on `PrefillCancelled`. When the prefix in use was
    /// the prepared cache, the original anchor is restored and the partial
    /// session cache is stashed back on the preparation.
    mutating func prefillCancelled(
        pending: FlintPrefixEngine.Preparation?,
        prefixWasPrepared: Bool,
        sessionCache: FlintPrefixEngine.Cache
    ) {
        if let pending, prefixWasPrepared {
            var restored = pending
            restored.cache = sessionCache
            promptCache = pending.origin
            preparation = restored
        } else {
            promptCache = sessionCache
        }
    }

    private func currentPreparedCount(_ originCount: Int) -> Int {
        preparation?.cache?.tokens.count ?? originCount
    }
}

struct FlintPrefixCancellation: Error {}

extension FlintPrefixSession {
    /// Chunk advance used by tests and by the scheduler's token bookkeeping:
    /// checkpoint, then append the next full chunk. A `shouldStop` mid-chunk
    /// throws and does not return a completed target.
    static func advanceChunk(
        prefix: FlintPrefixEngine.Cache,
        target: Int,
        tokens: [Int],
        cancel: Bool
    ) throws -> FlintPrefixEngine.Cache {
        if cancel { throw FlintPrefixCancellation() }
        var cache = prefix
        if !cache.tokens.isEmpty { cache.checkpoint() }
        let start = cache.tokens.count
        guard target <= tokens.count, start <= target else { throw FlintPrefixCancellation() }
        cache.tokens.append(contentsOf: tokens[start..<target])
        return cache
    }
}
