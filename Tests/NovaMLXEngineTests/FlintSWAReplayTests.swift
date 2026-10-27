import Foundation
import MLX
import MLXFast
import MLXLMCommon
import Testing

@testable import NovaMLXEngine

@Suite("Flint SWA replay and YOCO")
struct FlintSWAReplayTests {

    private func token(_ id: Int) -> MLXArray {
        var raw = [Float](repeating: 0, count: 4)
        raw[0] = Float(id)
        return MLXArray(raw).reshaped(1, 1, 1, 4)
    }

    private func chunk(_ ids: [Int]) -> MLXArray {
        concatenated(ids.map(token), axis: 2)
    }

    private func identities(_ cache: KVCache) -> [Float] {
        guard let pair = FlintRecCacheBridge.orderedPair(from: cache) else {
            return []
        }
        let keys = pair.0.asType(.float32)
        return (0 ..< keys.dim(2)).map { keys[0, 0, $0, 0].item(Float.self) }
    }

    @Test("a prompt longer than the window keeps the temporal tail")
    func testTailIdentity() {
        let window = 4
        let cache = FlintSlidingKVCache(windowSize: window)
        for id in 0 ..< 6 {
            let step = token(id)
            let (keys, _) = cache.update(keys: step, values: step)
            #expect(keys.dim(2) == min(id + 1, window))
        }
        #expect(cache.storedLength == window)
        #expect(cache.offset == 6)
        #expect(cache.evictedTokenCount == 2)
        #expect(identities(cache) == [2, 3, 4, 5])
        #expect(FlintRecCacheBridge.orderedPair(from: cache)?.0.dim(2) == window)
    }

    @Test("the window edge drops the oldest token on the next write")
    func testWindowEdge() {
        let cache = FlintSlidingKVCache(windowSize: 4)
        let first = chunk([0, 1, 2, 3])
        let (seen, _) = cache.update(keys: first, values: first)
        #expect(seen.dim(2) == 4)
        #expect(cache.storedLength == 4)
        #expect(cache.evictedTokenCount == 0)
        #expect(identities(cache) == [0, 1, 2, 3])
        #expect(cache.trim(1) == 1)
        #expect(identities(cache) == [0, 1, 2])

        let again = FlintSlidingKVCache(windowSize: 4)
        _ = again.update(keys: first, values: first)
        let extra = token(4)
        let (live, _) = again.update(keys: extra, values: extra)
        #expect(live.dim(2) == 4)
        #expect(again.evictedTokenCount == 1)
        #expect(again.trim(1) == 0)
        #expect(identities(again) == [1, 2, 3, 4])
    }

    @Test("a chunk that crosses the window returns the span and stores the tail")
    func testChunkAcrossEdge() {
        let cache = FlintSlidingKVCache(windowSize: 4)
        let head = chunk([0, 1, 2])
        _ = cache.update(keys: head, values: head)
        let tail = chunk([3, 4, 5])
        let (attention, _) = cache.update(keys: tail, values: tail)
        #expect(attention.dim(2) == 6)
        #expect(cache.storedLength == 4)
        #expect(cache.offset == 6)
        let keys = attention.asType(.float32)
        let seen = (0 ..< 6).map { keys[0, 0, $0, 0].item(Float.self) }
        #expect(seen == [0, 1, 2, 3, 4, 5])
        #expect(identities(cache) == [2, 3, 4, 5])
    }

    @Test("state restores the absolute offset and the temporal tail")
    func testStateRoundTrip() {
        let cache = FlintSlidingKVCache(windowSize: 4)
        for id in 0 ..< 6 {
            let step = token(id)
            _ = cache.update(keys: step, values: step)
        }
        let restored = FlintSlidingKVCache(windowSize: 4)
        restored.metaState = cache.metaState
        restored.state = cache.state
        #expect(restored.offset == 6)
        #expect(restored.storedLength == 4)
        #expect(identities(restored) == [2, 3, 4, 5])
    }

    @Test("a checkpoint restore is isolated from later writes")
    func testCheckpointRestore() {
        let cache = FlintSlidingKVCache(windowSize: 4)
        let full = KVCacheSimple()
        let head = chunk([0, 1, 2, 3, 4, 5])
        _ = cache.update(keys: head, values: head)
        _ = full.update(keys: head, values: head)
        let shot = FlintSlidingCheckpoint.capture(position: 6, caches: [cache, full])
        let extra = chunk([6, 7])
        _ = cache.update(keys: extra, values: extra)
        _ = full.update(keys: extra, values: extra)
        #expect(identities(cache) == [4, 5, 6, 7])

        let sliding = FlintSlidingKVCache(windowSize: 4)
        let other = KVCacheSimple()
        _ = sliding.update(keys: chunk([9, 9, 9, 9, 9, 9, 9, 9]), values: chunk([9, 9, 9, 9, 9, 9, 9, 9]))
        _ = other.update(keys: chunk([9, 9, 9, 9, 9, 9, 9, 9]), values: chunk([9, 9, 9, 9, 9, 9, 9, 9]))
        shot.restore(into: [sliding, other])
        #expect(FlintSWAReplay.trimKeepingSliding([sliding, other], drop: 2))
        #expect(identities(sliding) == [2, 3, 4, 5])
        #expect(sliding.offset == 6)
        #expect(other.offset == 6)
        #expect(identities(cache) == [4, 5, 6, 7])
    }

    @Test("replay chooses trim, a checkpoint, or a full prefill")
    func testDecide() {
        #expect(FlintSWAReplay.decide(snap: 10, kept: 7, prompt: 12, evicted: false, checkpoints: [])
            == .trim(3))
        #expect(FlintSWAReplay.decide(snap: 10, kept: 10, prompt: 12, evicted: true, checkpoints: [8])
            == .trim(0))
        #expect(FlintSWAReplay.decide(snap: 20, kept: 15, prompt: 18, evicted: true, checkpoints: [8, 16])
            == .replay(8))
        #expect(FlintSWAReplay.decide(snap: 20, kept: 15, prompt: 18, evicted: true, checkpoints: [])
            == .full)
        #expect(FlintSWAReplay.decide(snap: 10, kept: 10, prompt: 10, evicted: true, checkpoints: [10])
            == .full)
        #expect(FlintSWAReplay.decide(snap: 10, kept: 0, prompt: 4, evicted: true, checkpoints: [8])
            == .full)
    }

    @Test("tiled window attention matches one SDPA over the same mask")
    func testTiledAttentionMatches() throws {
        let heads = 2
        let window = 4
        let queries = MLXRandom.normal([1, heads, 6, 8]).asType(.float32)
        let keys = MLXRandom.normal([1, heads, 8, 8]).asType(.float32)
        let values = MLXRandom.normal([1, heads, 8, 8]).asType(.float32)
        let sinks = MLXArray.zeros([heads]).asType(.float32)
        let mask = MLXFast.ScaledDotProductAttentionMaskMode.array(
            createCausalMask(n: 6, offset: 2, windowSize: window))
        let full = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: 1, mask: mask, sinks: sinks)
        let tiled = try #require(
            FlintSWAAttention.replay(
                queries: queries, keys: keys, values: values,
                scale: 1, mask: mask, sinks: sinks, window: window, tile: 3))
        let causalTiles = try #require(
            FlintSWAAttention.replay(
                queries: queries, keys: keys, values: values,
                scale: 1, mask: .causal, sinks: sinks, window: window, tile: 3))
        eval(full, tiled, causalTiles)
        #expect(abs(full - tiled).max().item(Float.self) < 1e-4)
        #expect(abs(full - causalTiles).max().item(Float.self) < 1e-4)

        let short = MLXRandom.normal([1, heads, 3, 8]).asType(.float32)
        #expect(
            FlintSWAAttention.replay(
                queries: short, keys: short, values: short,
                scale: 1, mask: .causal, sinks: sinks, window: window) == nil)
    }

    @Test("YOCO aliases the last same-type layer and stores no KV")
    func testYOCO() throws {
        let identity = FlintYOCOLayout.make(
            layerTypes: ["sliding_attention", "full_attention"], sharedTail: 0)
        #expect(identity.source == [0, 1])
        #expect(!identity.sharesKV)

        let mixed = FlintYOCOLayout.make(
            layerTypes: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            sharedTail: 1)
        #expect(mixed.source == [0, 1, 2, 1])
        #expect(mixed.isAlias(3))
        #expect(!mixed.isAlias(2))

        let json = """
        {
          "model_type": "flint",
          "num_hidden_layers": 4,
          "head_dim": 8,
          "hidden_size": 32,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "moe_intermediate_size": 16,
          "shared_expert_intermediate_size": 16,
          "mlp_only_layers": [0, 1, 2, 3],
          "sliding_window": 4,
          "rms_norm_eps": 1e-06,
          "vocab_size": 32,
          "rope_theta": 10000.0,
          "tie_word_embeddings": true,
          "num_kv_shared_layers": 1,
          "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        }
        """
        let config = try JSONDecoder.json5().decode(FlintConfiguration.self, from: Data(json.utf8))
        #expect(config.numKVSharedLayers == 1)
        let model = FlintModel(config)
        let caches = try model.newCache(parameters: nil)
        #expect(caches[0] is FlintSlidingKVCache)
        #expect(caches[2] is FlintSlidingKVCache)
        let alias = try #require(caches[3] as? FlintYOCOAliasCache)
        #expect(alias.source == 1)
        #expect(alias.state.isEmpty)

        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let logits = model(tokens, cache: caches)
        eval(logits)
        let more = MLXArray([Int32(5), 6, 7, 8]).reshaped(1, 4)
        let next = model(more, cache: caches)
        eval(next)
        let sliding = try #require(caches[0] as? FlintSlidingKVCache)
        #expect(sliding.storedLength == 4)
        #expect(sliding.offset == 8)
        #expect(caches[1].offset == 8)
        #expect(caches[2].offset == 8)
        #expect(alias.offset == 0)
        #expect(alias.state.isEmpty)
        #expect(caches[1].state[0].dim(2) == 8)
    }
}
