import Testing
import Foundation
import NovaMLXCore
import MLXLMCommon
@testable import NovaMLXEngine

/// Regression tests for `MLXEngine.defensiveEosTokenIds(for:container:)` —
/// the family-canonical EOS injection that protects against malformed
/// `config.json` files shipping incomplete `eos_token_id` arrays.
///
/// Tracked as todo.markdown §2.9.
///
/// **Risk being defended against**: Gemma's VLM streaming loop in
/// `MLXEngine.swift:~1795/2075` stops on token 106 (`<end_of_turn>`) only
/// because the canonical Gemma `config.json` lists `"eos_token_id": [1, 106]`.
/// A future quant author shipping `eos_token_id: []` would silently produce
/// runaway generation up to the full `maxTokens` budget. The helper
/// unconditionally unions `{1, 106}` for `.gemma` family models and emits a
/// once-per-modelId warning so we can detect malformed quants in production
/// via `~/.nova/novamlx.log`.
@Suite("Defensive EOS Injection (todo §2.9)")
struct DefensiveEosInjectionTests {

    // NOTE: explicit module qualification on every overlap — `MLXLMCommon`
    // re-exports `ModelContainer` and a `.gemma` case (on `ToolCallFormat`),
    // colliding with NovaMLX's own `ModelContainer` and `ModelFamily.gemma`.
    private func makeContainer(
        id: String,
        family: NovaMLXCore.ModelFamily
    ) -> NovaMLXEngine.ModelContainer {
        let identifier = ModelIdentifier(id: id, family: family)
        let config = ModelConfig(identifier: identifier)
        return NovaMLXEngine.ModelContainer(identifier: identifier, config: config)
    }

    private func makeUpstreamConfig(eosTokenIds: Set<Int>) -> ModelConfiguration {
        // Use the directory-based init which exposes `eosTokenIds`.
        ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/synthetic-test-config"),
            eosTokenIds: eosTokenIds
        )
    }

    // MARK: - Gemma: malformed configs get defensively patched

    @Test("Gemma with empty eos_token_id gets {1, 106} injected")
    func gemmaEmptyEosTokenIdInjected() {
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(id: "gemma-3-4b-it-empty-eos", family: NovaMLXCore.ModelFamily.gemma)
        let modelConfig = makeUpstreamConfig(eosTokenIds: [])

        let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(result.contains(1), "Gemma <bos> token 1 must be injected")
        #expect(result.contains(106), "Gemma <end_of_turn> token 106 must be injected")
        #expect(result == [1, 106], "Empty input should produce exactly the canonical set")
    }

    @Test("Gemma missing only token 106 still gets it injected")
    func gemmaMissingEndOfTurnInjected() {
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(id: "gemma-3-4b-it-no-106", family: NovaMLXCore.ModelFamily.gemma)
        // Some malformed configs include <bos> but forget <end_of_turn>.
        let modelConfig = makeUpstreamConfig(eosTokenIds: [1])

        let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(result.contains(106), "Token 106 must be injected when missing")
        #expect(result.contains(1), "Existing token 1 must be preserved")
    }

    @Test("Gemma with canonical config returns the original set unchanged")
    func gemmaCanonicalConfigUnchanged() {
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(id: "gemma-3-4b-it-canonical", family: NovaMLXCore.ModelFamily.gemma)
        let modelConfig = makeUpstreamConfig(eosTokenIds: [1, 106])

        let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(result == [1, 106], "Canonical Gemma config must round-trip without modification")
    }

    @Test("Gemma with extra EOS tokens beyond canonical preserves all of them")
    func gemmaExtraEosPreserved() {
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(id: "gemma-3-4b-it-extra", family: NovaMLXCore.ModelFamily.gemma)
        // Some Gemma quants add extra padding/instruct tokens to eos_token_id.
        let modelConfig = makeUpstreamConfig(eosTokenIds: [1, 106, 107, 256000])

        let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(result.contains(1))
        #expect(result.contains(106))
        #expect(result.contains(107), "Custom extra EOS token must be preserved")
        #expect(result.contains(256000), "Custom extra EOS token must be preserved")
    }

    // MARK: - Non-Gemma families: pass through unchanged

    @Test("Qwen family passes through eos_token_id unchanged (no defensive injection)")
    func qwenPassthrough() {
        let container = makeContainer(id: "Qwen3.6-35B-A3B-4bit", family: NovaMLXCore.ModelFamily.qwen)
        // Qwen3.6's actual canonical eos_token_id from its config.json
        let modelConfig = makeUpstreamConfig(eosTokenIds: [248046, 248044])

        let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(result == [248046, 248044], "Qwen must not get Gemma's defensive injection")
        #expect(!result.contains(1), "Token 1 must not leak into Qwen's set")
        #expect(!result.contains(106), "Token 106 must not leak into Qwen's set")
    }

    @Test("Llama family passes through empty eos_token_id unchanged")
    func llamaEmptyPassthrough() {
        let container = makeContainer(id: "synthetic-llama", family: NovaMLXCore.ModelFamily.llama)
        let modelConfig = makeUpstreamConfig(eosTokenIds: [])

        let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(result.isEmpty, "No defensive injection for non-Gemma families")
    }

    @Test("All non-Gemma families round-trip without modification")
    func allNonGemmaFamiliesPassthrough() {
        let nonGemma: [ModelFamily] = ModelFamily.allCases.filter { $0 != .gemma }
        let probe: Set<Int> = [42, 99]

        for family in nonGemma {
            let container = makeContainer(id: "probe-\(family.rawValue)", family: family)
            let modelConfig = makeUpstreamConfig(eosTokenIds: probe)
            let result = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

            #expect(result == probe, "Family \(family.rawValue) must not get defensive injection")
        }
    }

    // MARK: - Idempotency

    @Test("Repeated calls on same container produce stable output")
    func idempotentOnRepeatedCalls() {
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(id: "gemma-idempotent", family: NovaMLXCore.ModelFamily.gemma)
        let modelConfig = makeUpstreamConfig(eosTokenIds: [])

        let first = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)
        let second = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)
        let third = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)

        #expect(first == second, "Repeated invocations must be deterministic")
        #expect(second == third, "Repeated invocations must be deterministic")
    }

    @Test("Once-per-model warning cache prevents duplicate warnings")
    func warningCacheDeduplicates() {
        MLXEngine._resetDefensiveEosWarningCache()
        let modelId = "gemma-warning-cache-test"
        let container = makeContainer(id: modelId, family: NovaMLXCore.ModelFamily.gemma)
        let modelConfig = makeUpstreamConfig(eosTokenIds: [])

        // First call: would trigger warning, model gets added to warned set.
        _ = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)
        #expect(MLXEngine.defensiveEosWarnedModels.contains(modelId),
                "Model must be in warned set after first defensive injection")

        // Second call on same modelId: should not double-add (set semantics)
        _ = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)
        #expect(MLXEngine.defensiveEosWarnedModels.filter { $0 == modelId }.count == 1,
                "Model must appear exactly once in warned set despite multiple calls")
    }

    @Test("Cache reset hook clears warned models for test isolation")
    func resetHookClearsCache() {
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(id: "gemma-reset-test", family: NovaMLXCore.ModelFamily.gemma)
        let modelConfig = makeUpstreamConfig(eosTokenIds: [])

        _ = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)
        #expect(!MLXEngine.defensiveEosWarnedModels.isEmpty)

        MLXEngine._resetDefensiveEosWarningCache()
        #expect(MLXEngine.defensiveEosWarnedModels.isEmpty,
                "Reset hook must clear all warned-model state")
    }

    // MARK: - Realistic malformed-quant scenarios

    @Test("Realistic scenario: Gemma quant author drops eos_token_id entirely")
    func realisticMalformedGemmaQuant() {
        // Simulates a real-world failure mode: a community quantizer copies
        // a model but ships `config.json` without `eos_token_id`. Upstream
        // BaseConfiguration.swift would decode `eosTokenIds` as nil → empty
        // set on `ModelConfiguration`. Without §2.9, the VLM streaming loop
        // (MLXEngine.swift:~2075) would never terminate on token 106.
        MLXEngine._resetDefensiveEosWarningCache()
        let container = makeContainer(
            id: "community-gemma-3-4b-vlm-malformed-q4_0",
            family: NovaMLXCore.ModelFamily.gemma
        )
        let malformedConfig = makeUpstreamConfig(eosTokenIds: [])

        let safeSet = MLXEngine.defensiveEosTokenIds(
            for: malformedConfig,
            container: container
        )

        // The VLM loop's `stopTokenIds.contains(tokenId)` must now succeed
        // when the model emits token 106, even though the upstream config
        // had no EOS at all.
        #expect(safeSet.contains(106),
                "Defensive injection must protect against malformed Gemma quants — this is the entire point of §2.9")
    }
}
