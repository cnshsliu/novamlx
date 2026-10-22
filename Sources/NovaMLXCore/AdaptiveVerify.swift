import Foundation

// Confidence-scheduled verification length for DFlash2 and DSpark.
//
// Each confidence is P(draft position i is accepted | positions 0..<i were).
// The scheduler picks the prefix length that maximizes expected committed
// tokens per unit of target-verify cost. Committed tokens are always the
// greedy matching prefix plus the target's bonus argmax, the same rule a
// fixed block uses. A shorter prefix does not change that rule; it only
// chooses how many draft positions the target scores.
//
// Cost is not linear in verify width. On one Apple Silicon stream the target
// forward is dominated by reading weights, so a fixed cost keeps medium
// confidence from collapsing to a single token. Drafts past the Metal vector
// SDPA window (query length that stays on the fast kernel; 4 drafts for the
// GQA shapes this engine already special-cases) pay a higher marginal cost.
// High confidence can still cross that window. Low confidence stops before
// the tail, which is the fixed-block waste (verify 7, accept ~4).

public struct AdaptiveVerifyConfig: Sendable, Equatable {
    public enum Mode: String, Sendable, Equatable {
        case adaptive
        case fixed
    }

    public var mode: Mode
    /// Shortest prefix the scheduler may return when it has a confidence signal.
    public var minVerify: Int
    /// Longest prefix. Nil means the caller’s draft budget.
    public var maxVerify: Int?
    /// Token-equivalent cost of launching one target verify, before any draft token.
    public var fixedCost: Double
    /// Extra cost of each draft token inside the fast SDPA window.
    public var marginalCost: Double
    /// Extra cost of each draft token past `fastKernelMaxDrafts`.
    public var slowMarginalCost: Double
    /// Drafts that stay on the fast vector SDPA kernel. 4 matches
    /// DSpark’s GQA-6 / head-256 window (verify width = anchor + drafts ≤ 5).
    public var fastKernelMaxDrafts: Int
    /// DFlash2 has no confidence head. Gap between the top two draft scores,
    /// in logit units, is mapped with a sigmoid of scale `marginScale`.
    public var marginScale: Float
    /// Softens a calibrated head: sigmoid(logit(p) / temperature). 1 leaves p unchanged.
    public var confidenceTemperature: Float

    public init(
        mode: Mode = .adaptive,
        minVerify: Int = 1,
        maxVerify: Int? = nil,
        fixedCost: Double = 2.0,
        marginalCost: Double = 0.40,
        slowMarginalCost: Double = 0.55,
        fastKernelMaxDrafts: Int = 4,
        marginScale: Float = 2,
        confidenceTemperature: Float = 1
    ) {
        self.mode = mode
        let minClamped = max(minVerify, 1)
        self.minVerify = minClamped
        if let maxVerify {
            self.maxVerify = max(maxVerify, minClamped)
        } else {
            self.maxVerify = nil
        }
        self.fixedCost = fixedCost > 0 ? fixedCost : 2.0
        self.marginalCost = max(marginalCost, 0)
        self.slowMarginalCost = max(slowMarginalCost, 0)
        self.fastKernelMaxDrafts = max(fastKernelMaxDrafts, 0)
        self.marginScale = max(marginScale, 1e-4)
        self.confidenceTemperature = max(confidenceTemperature, 1e-4)
    }

    /// Fixed block: ignore confidences and verify the caller’s historical cap.
    public static let fixed = AdaptiveVerifyConfig(mode: .fixed)

    public var summary: String {
        let maxText = maxVerify.map(String.init) ?? "draft"
        return "mode=\(mode.rawValue) min=\(minVerify) max=\(maxText) "
            + "fixedCost=\(fixedCost) marginal=\(marginalCost) "
            + "slowMarginal=\(slowMarginalCost) fastDrafts=\(fastKernelMaxDrafts) "
            + "marginScale=\(marginScale) temperature=\(confidenceTemperature)"
    }

    /// Environment and optional per-request overrides. Unset mode is adaptive.
    /// Recognized mode strings: adaptive, on, 1, true; fixed, off, 0, false.
    public static func resolved(
        environment: [String: String]? = nil,
        modeOverride: String? = nil,
        minOverride: Int? = nil,
        maxOverride: Int? = nil
    ) -> AdaptiveVerifyConfig {
        let env = environment ?? ProcessInfo.processInfo.environment
        var config = AdaptiveVerifyConfig()
        if let mode = parseMode(modeOverride ?? env["NOVAMLX_ADAPTIVE_VERIFY"]) {
            config.mode = mode
        }
        if let minOverride {
            config.minVerify = max(minOverride, 1)
        } else if let raw = env["NOVAMLX_ADAPTIVE_VERIFY_MIN"], let value = Int(raw) {
            config.minVerify = max(value, 1)
        }
        if let maxOverride {
            config.maxVerify = max(maxOverride, config.minVerify)
        } else if let raw = env["NOVAMLX_ADAPTIVE_VERIFY_MAX"], let value = Int(raw), value > 0 {
            config.maxVerify = max(value, config.minVerify)
        }
        if let value = envDouble(env["NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST"]), value > 0 {
            config.fixedCost = value
        }
        if let value = envDouble(env["NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST"]), value >= 0 {
            config.marginalCost = value
        }
        if let value = envDouble(env["NOVAMLX_ADAPTIVE_VERIFY_SLOW_MARGINAL_COST"]), value >= 0 {
            config.slowMarginalCost = value
        }
        if let raw = env["NOVAMLX_ADAPTIVE_VERIFY_FAST_DRAFTS"], let value = Int(raw), value >= 0 {
            config.fastKernelMaxDrafts = value
        }
        if let value = envFloat(env["NOVAMLX_ADAPTIVE_VERIFY_MARGIN_SCALE"]), value > 0 {
            config.marginScale = value
        }
        if let value = envFloat(env["NOVAMLX_ADAPTIVE_VERIFY_TEMPERATURE"]), value > 0 {
            config.confidenceTemperature = value
        }
        return config
    }

    /// How many draft tokens to send to the target.
    ///
    /// `fixedLength` is the historical block (DFlash cap policy, or DSpark’s
    /// vector-SDPA cap). Fixed mode and a missing confidence signal both
    /// return that length, clamped to the draft. Adaptive mode searches
    /// prefixes of the confidence vector.
    public func verifyLength(confidences: [Float], draftCount: Int, fixedLength: Int) -> Int {
        let available = max(draftCount, 0)
        guard available > 0 else { return 0 }
        let fallback = clampedFixed(fixedLength, available: available)
        if mode == .fixed { return fallback }
        let signaled = min(available, confidences.count)
        guard signaled > 0, confidences.prefix(signaled).contains(where: { $0 != 0 && !$0.isNaN }) else {
            return fallback
        }
        let ceiling = min(signaled, maxVerify ?? signaled)
        guard ceiling > 0 else { return 0 }
        let floor = min(max(minVerify, 1), ceiling)

        var bestK = floor
        var bestScore = -Double.infinity
        var survival = 1.0
        var expectedAccepted = 0.0
        for k in 1 ... ceiling {
            let p = adjustedProbability(confidences[k - 1])
            survival *= p
            expectedAccepted += survival
            let expectedTokens = 1 + expectedAccepted
            let cost = cost(of: k)
            let score = expectedTokens / cost
            if k >= floor, score > bestScore + 1e-9 {
                bestScore = score
                bestK = k
            }
        }
        return bestK
    }

    /// Sigmoid of the top-1 / top-2 score gap. A tie is 0.5. DFlash2 uses this
    /// in place of a trained confidence head.
    public static func marginProbability(top1: Float, top2: Float, scale: Float) -> Float {
        let denom = max(scale, 1e-4)
        let x = (top1 - top2) / denom
        if x >= 0 {
            let z = exp(-x)
            return 1 / (1 + z)
        }
        let z = exp(x)
        return z / (1 + z)
    }

    public static func confidences(fromMargins margins: [Float], scale: Float) -> [Float] {
        margins.map { marginProbability(top1: $0, top2: 0, scale: scale) }
    }

    public static func parseMode(_ raw: String?) -> Mode? {
        guard let text = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !text.isEmpty
        else { return nil }
        switch text {
        case "adaptive", "on", "1", "true", "yes":
            return .adaptive
        case "fixed", "off", "0", "false", "no":
            return .fixed
        default:
            return nil
        }
    }

    private func clampedFixed(_ fixedLength: Int, available: Int) -> Int {
        var cap = fixedLength
        if let maxVerify { cap = min(cap, maxVerify) }
        return min(available, max(cap, 0))
    }

    private func cost(of k: Int) -> Double {
        var total = fixedCost
        if k <= 0 { return max(total, 1e-6) }
        let fast = min(k, fastKernelMaxDrafts)
        let slow = max(k - fastKernelMaxDrafts, 0)
        total += Double(fast) * marginalCost + Double(slow) * slowMarginalCost
        return max(total, 1e-6)
    }

    private func adjustedProbability(_ raw: Float) -> Double {
        var p = Double(raw)
        if p.isNaN || p < 0 { p = 0 }
        if p > 1 { p = 1 }
        let temperature = Double(confidenceTemperature)
        if abs(temperature - 1) > 1e-6, p > 0, p < 1 {
            let logit = log(p / (1 - p))
            let x = logit / temperature
            if x >= 0 {
                p = 1 / (1 + exp(-x))
            } else {
                let z = exp(x)
                p = z / (1 + z)
            }
        }
        if p < 1e-6 { p = 1e-6 }
        if p > 1 - 1e-6 { p = 1 - 1e-6 }
        return p
    }

    private static func envDouble(_ raw: String?) -> Double? {
        guard let raw, let value = Double(raw), value.isFinite else { return nil }
        return value
    }

    private static func envFloat(_ raw: String?) -> Float? {
        guard let value = envDouble(raw) else { return nil }
        return Float(value)
    }
}
