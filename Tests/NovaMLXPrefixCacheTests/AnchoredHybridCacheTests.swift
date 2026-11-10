import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import NovaMLXPrefixCache

/// Anchored hybrid (GDN) prefix caching: full blocks carry KV only, the
/// anchor entry carries the partial KV tail plus the recurrent state at the
/// prompt boundary, and a fetch must either restore both consistently
/// (anchored hit) or miss — never a KV-only restore for a hybrid model.
@Suite("Anchored hybrid prefix cache")
struct AnchoredHybridCacheTests {

    private func makeHybridCache() -> [KVCache] {
        // 2 KV layers + 2 mamba layers, mirroring a hybrid stack's shape.
        let kv0 = KVCacheSimple()
        _ = kv0.update(
            keys: MLXArray.zeros([1, 2, 256, 8], dtype: .float16),
            values: MLXArray.zeros([1, 2, 256, 8], dtype: .float16))
        let kv1 = KVCacheSimple()
        _ = kv1.update(
            keys: MLXArray.zeros([1, 2, 256, 8], dtype: .float16),
            values: MLXArray.zeros([1, 2, 256, 8], dtype: .float16))
        let m0 = MambaCache()
        m0[0] = MLXArray.zeros([1, 3, 16], dtype: .float16)
        m0[1] = MLXArray.zeros([1, 4, 4, 4])
        let m1 = MambaCache()
        m1[0] = MLXArray.zeros([1, 3, 16], dtype: .float16)
        m1[1] = MLXArray.zeros([1, 4, 4, 4])
        return [kv0, m0, kv1, m1]
    }

    private func makeManager() -> PrefixCacheManager {
        PrefixCacheManager(
            config: PrefixCacheConfig(
                blockSize: 8,
                maxBlocks: 64,
                initialBlocks: 8,
                ssdCacheDir: nil,
                ssdMaxSizeBytes: 0,
                hotCacheMaxBytes: 256 * 1024 * 1024,
                ttl: 3600),
            modelName: "anchored-test")
    }

    @Test("anchored store restores a consistent cache on a prefix-extension fetch")
    func anchoredRoundTrip() throws {
        let manager = makeManager()
        let prompt = Array(0..<21)  // 2 full blocks of 8 + tail of 5
        let cache = makeHybridCache()
        // Prefill-boundary anchor: distinct values so restore is checkable.
        let anchor = cache.map { layer -> [MLXArray] in
            layer is MambaCache ? layer.state : []
        }
        for (i, layer) in cache.enumerated() where layer is MambaCache {
            let m = layer as! MambaCache
            m[1] = MLXArray([Float](repeating: Float(i + 1), count: 64))
                .reshaped([1, 4, 4, 4])
        }
        manager.storeAnchoredCache(tokenIds: prompt, cache: cache, fixedStateAnchor: anchor)

        // Fetch with a prompt that strictly extends the anchored prompt.
        let extended = prompt + [100, 101, 102]
        let result = manager.fetchPrefix(tokenIds: extended, requireAnchor: true)
        #expect(result.cachedTokenCount == 21, "anchor should cover 21 tokens")
        #expect(result.remainingTokenIds == [100, 101, 102])
        let restored = try #require(result.cache)
        // Mamba layers restored at the anchor marker values (0.0 / 2.0), not
        // the post-generation 1.0/3.0 markers.
        let m0Restored = restored[1] as! MambaCache
        let m1Restored = restored[3] as! MambaCache
        let v0 = (m0Restored[1] ?? MLXArray.zeros([1])).asArray(Float.self).max()!
        let v1 = (m1Restored[1] ?? MLXArray.zeros([1])).asArray(Float.self).max()!
        #expect(v0 == 0.0)
        #expect(v1 == 0.0, "anchor was captured before the marker overwrite")
    }

    @Test("requireAnchor rejects a KV-only chain hit")
    func requireAnchorMissesWithoutAnchor() {
        let manager = makeManager()
        let prompt = Array(0..<16)  // exactly 2 blocks, no tail
        let cache = makeHybridCache()
        let anchor = cache.map { $0 is MambaCache ? $0.state : [] }
        manager.storeAnchoredCache(tokenIds: prompt, cache: cache, fixedStateAnchor: anchor)

        // A different prompt sharing only the first block: chain hits 1 block
        // but the anchor (at the 2-block tip with empty tail) does not apply.
        let otherPrompt = Array(0..<8) + Array(50..<58)
        let result = manager.fetchPrefix(tokenIds: otherPrompt, requireAnchor: true)
        #expect(result.cachedTokenCount == 0, "KV-only hit must be rejected for hybrid")
    }

    @Test("exact-prompt repeat is a miss (nothing left to prefill)")
    func exactRepeatIsMiss() {
        let manager = makeManager()
        let prompt = Array(0..<21)
        let cache = makeHybridCache()
        let anchor = cache.map { $0 is MambaCache ? $0.state : [] }
        manager.storeAnchoredCache(tokenIds: prompt, cache: cache, fixedStateAnchor: anchor)
        let result = manager.fetchPrefix(tokenIds: prompt, requireAnchor: true)
        #expect(result.cachedTokenCount == 0)
        #expect(result.remainingTokenIds == prompt)
    }
}
