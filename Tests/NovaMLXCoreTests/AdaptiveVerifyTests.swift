import Foundation
import Testing
@testable import NovaMLXCore

@Suite("Adaptive verify length")
struct AdaptiveVerifyTests {
    private let defaults = AdaptiveVerifyConfig()

    @Test("Unset environment is adaptive and fixed strings turn the scheduler off")
    func parsesModeFromEnvironment() {
        let adaptive = AdaptiveVerifyConfig.resolved(environment: [:])
        #expect(adaptive.mode == .adaptive)
        #expect(adaptive.minVerify == 1)
        #expect(adaptive.maxVerify == nil)
        #expect(adaptive.fixedCost == 2)
        #expect(adaptive.marginalCost == 0.40)
        #expect(adaptive.slowMarginalCost == 0.55)
        #expect(adaptive.fastKernelMaxDrafts == 4)
        #expect(adaptive.marginScale == 2)
        #expect(adaptive.confidenceTemperature == 1)

        let off = AdaptiveVerifyConfig.resolved(environment: [
            "NOVAMLX_ADAPTIVE_VERIFY": "off",
            "NOVAMLX_ADAPTIVE_VERIFY_MIN": "2",
            "NOVAMLX_ADAPTIVE_VERIFY_MAX": "5",
            "NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST": "3.5",
            "NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST": "0.2",
            "NOVAMLX_ADAPTIVE_VERIFY_SLOW_MARGINAL_COST": "0.9",
            "NOVAMLX_ADAPTIVE_VERIFY_FAST_DRAFTS": "3",
            "NOVAMLX_ADAPTIVE_VERIFY_MARGIN_SCALE": "4",
            "NOVAMLX_ADAPTIVE_VERIFY_TEMPERATURE": "1.5",
        ])
        #expect(off.mode == .fixed)
        #expect(off.minVerify == 2)
        #expect(off.maxVerify == 5)
        #expect(off.fixedCost == 3.5)
        #expect(off.marginalCost == 0.2)
        #expect(off.slowMarginalCost == 0.9)
        #expect(off.fastKernelMaxDrafts == 3)
        #expect(off.marginScale == 4)
        #expect(off.confidenceTemperature == 1.5)

        let request = AdaptiveVerifyConfig.resolved(
            environment: ["NOVAMLX_ADAPTIVE_VERIFY": "fixed"],
            modeOverride: "adaptive",
            minOverride: 1,
            maxOverride: 6
        )
        #expect(request.mode == .adaptive)
        #expect(request.maxVerify == 6)

        #expect(AdaptiveVerifyConfig.parseMode("0") == .fixed)
        #expect(AdaptiveVerifyConfig.parseMode("true") == .adaptive)
        #expect(AdaptiveVerifyConfig.parseMode("sometimes") == nil)
        #expect(AdaptiveVerifyConfig.parseMode(nil) == nil)
    }

    @Test("Invalid numeric knobs stay on the defaults")
    func ignoresUnusableKnobs() {
        let config = AdaptiveVerifyConfig.resolved(environment: [
            "NOVAMLX_ADAPTIVE_VERIFY_MIN": "0",
            "NOVAMLX_ADAPTIVE_VERIFY_MAX": "nope",
            "NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST": "-1",
            "NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST": "nan",
        ])
        #expect(config.minVerify == 1)
        #expect(config.maxVerify == nil)
        #expect(config.fixedCost == 2)
        #expect(config.marginalCost == 0.40)
    }

    @Test("Low-confidence tail stops inside the fast kernel window")
    func earlyStopOnLowTail() {
        let conf: [Float] = [0.92, 0.88, 0.84, 0.7, 0.25, 0.15, 0.1]
        let k = defaults.verifyLength(confidences: conf, draftCount: 7, fixedLength: 7)
        #expect(k == 4)
    }

    @Test("A DeepSeek-like survival curve verifies about three drafts, not the whole block")
    func deepseekishLength() {
        let conf: [Float] = [0.85, 0.8, 0.75, 0.55, 0.35, 0.2, 0.15]
        let k = defaults.verifyLength(confidences: conf, draftCount: 7, fixedLength: 7)
        #expect(k == 3)
    }

    @Test("High confidence extends past the fast kernel up to the draft budget")
    func highConfidenceExtends() {
        let conf = [Float](repeating: 0.97, count: 7)
        let k = defaults.verifyLength(confidences: conf, draftCount: 7, fixedLength: 4)
        #expect(k == 7)
    }

    @Test("Collapsed confidence verifies the minimum")
    func collapseStopsAtOne() {
        let conf: [Float] = [0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        let k = defaults.verifyLength(confidences: conf, draftCount: 7, fixedLength: 7)
        #expect(k == 1)
    }

    @Test("min and max clamp the scheduled prefix")
    func minMaxClamps() {
        let collapsed: [Float] = [0.4, 0.2, 0.1, 0.1, 0.1]
        var raised = defaults
        raised.minVerify = 3
        #expect(raised.verifyLength(confidences: collapsed, draftCount: 5, fixedLength: 5) == 3)

        let high = [Float](repeating: 0.97, count: 7)
        var capped = defaults
        capped.maxVerify = 3
        #expect(capped.verifyLength(confidences: high, draftCount: 7, fixedLength: 7) == 3)
    }

    @Test("A higher temperature shortens a confident block and fixed mode ignores it")
    func temperatureChangesLength() {
        let confident = [Float](repeating: 0.97, count: 7)
        let baseline = defaults.verifyLength(
            confidences: confident, draftCount: 7, fixedLength: 4)
        let softened = defaults.verifyLength(
            confidences: confident, draftCount: 7, fixedLength: 4,
            positionTemperatures: [Float](repeating: 3, count: 7))
        #expect(baseline == 7)
        #expect(softened < baseline)

        let aggressive = [Float](repeating: 0.5, count: 7)
        let fixed = AdaptiveVerifyConfig.fixed.verifyLength(
            confidences: [0.1, 0.1, 0.1, 0.1],
            draftCount: 7,
            fixedLength: 4,
            positionTemperatures: aggressive)
        #expect(fixed == 4)
    }

    @Test("Fixed mode and a missing head both keep the historical block length")
    func fixedAndMissingSignal() {
        let low: [Float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        let fixed = AdaptiveVerifyConfig.fixed.verifyLength(
            confidences: low, draftCount: 7, fixedLength: 4)
        #expect(fixed == 4)

        let missing = defaults.verifyLength(
            confidences: [0, 0, 0, 0], draftCount: 6, fixedLength: 4)
        #expect(missing == 4)

        let empty = defaults.verifyLength(confidences: [], draftCount: 6, fixedLength: 4)
        #expect(empty == 4)

        #expect(defaults.verifyLength(confidences: [0.9], draftCount: 0, fixedLength: 4) == 0)
    }

    @Test("A tie in expected throughput keeps the shorter prefix")
    func tieKeepsShorter() {
        let conf: [Float] = [0.5, 0.5, 0.5, 0.5]
        let k = defaults.verifyLength(confidences: conf, draftCount: 4, fixedLength: 4)
        #expect(k == 1)
    }

    @Test("Margin heuristic is 0.5 on a tie and saturates on a wide gap")
    func marginHeuristic() {
        let tie = AdaptiveVerifyConfig.marginProbability(top1: 3, top2: 3, scale: 2)
        #expect(abs(tie - 0.5) < 1e-5)
        let wide = AdaptiveVerifyConfig.marginProbability(top1: 8, top2: 0, scale: 2)
        #expect(wide > 0.98)
        let mapped = AdaptiveVerifyConfig.confidences(fromMargins: [0, 2, 8], scale: 2)
        #expect(mapped.count == 3)
        #expect(abs(mapped[0] - 0.5) < 1e-5)
        #expect(mapped[2] > mapped[1])
    }

    @Test("Request fields survive sampling overrides and the worker codec")
    func requestFieldsRoundTrip() {
        let original = InferenceRequest(
            model: "main",
            messages: [ChatMessage(role: .user, content: "hi")],
            draftModel: "draft",
            numDraftTokens: 4,
            adaptiveVerify: "fixed",
            adaptiveVerifyMin: 2,
            adaptiveVerifyMax: 6
        )
        var settings = ModelSettings()
        settings.temperature = 0.2
        let overridden = settings.applySamplingOverrides(to: original)
        #expect(overridden.adaptiveVerify == "fixed")
        #expect(overridden.adaptiveVerifyMin == 2)
        #expect(overridden.adaptiveVerifyMax == 6)
        #expect(overridden.temperature == 0.2)

        let restored = CodableInferenceRequest(from: overridden).toInferenceRequest()
        #expect(restored.adaptiveVerify == "fixed")
        #expect(restored.adaptiveVerifyMin == 2)
        #expect(restored.adaptiveVerifyMax == 6)
        #expect(restored.draftModel == "draft")
    }
}
