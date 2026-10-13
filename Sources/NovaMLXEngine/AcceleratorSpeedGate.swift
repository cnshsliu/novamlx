import Foundation
import NovaMLXCore

// MARK: - Kinds and decisions

/// Speculative companion kinds. One primary path runs per request.
///
/// Preference, best first: DFlash2, then MTP, then DSpark, then a small
/// autoregressive draft. A lower-ranked companion is not started while a
/// higher-ranked one is already on. DSpark joint verify (PR #14) is unchanged;
/// this order only chooses which companion is primary.
public enum AcceleratorKind: String, Sendable, Codable, CaseIterable {
    case dflash2
    case mtp
    case dspark
    case smallDraft
}

public enum AcceleratorPreference {
    /// Best first. Do not reorder without a measured reason.
    public static let order: [AcceleratorKind] = [.dflash2, .mtp, .dspark, .smallDraft]

    public static func rank(_ kind: AcceleratorKind) -> Int {
        order.firstIndex(of: kind) ?? order.count
    }
}

/// Rough request shape used when a profile differs by length or task.
public enum SpecScenario: String, Sendable, Codable {
    /// Short chat reply. The DFlash2 one-wait note measured this separately.
    case chatShort
    /// Long-prompt decode (the case a block drafter is meant to speed up).
    case chatLong
    /// Code or math, where acceptance and wording noise differ from chat.
    case codeMath
    /// Load time, or the caller did not name a scenario.
    case unspecified
}

/// Whether a companion should be attached for this model and scenario.
public enum AcceleratorGateDecision: String, Sendable, Codable {
    /// Measured faster than plain decode. Download if needed and attach.
    case autoOn = "auto_on"
    /// Measured slower, or no better. Do not load and do not ask.
    case autoOff = "auto_off"
    /// No confident profile. Ask at load time. Never treat this as auto-on.
    case askUser = "ask_user"
}

public struct AcceleratorCandidate: Sendable, Equatable, Codable {
    public let kind: AcceleratorKind
    public let acceleratorId: String
    public let displayName: String
    public let estimatedSizeMB: Int
    public let family: ModelFamily

    public init(
        kind: AcceleratorKind,
        acceleratorId: String,
        displayName: String,
        estimatedSizeMB: Int,
        family: ModelFamily
    ) {
        self.kind = kind
        self.acceleratorId = acceleratorId
        self.displayName = displayName
        self.estimatedSizeMB = estimatedSizeMB
        self.family = family
    }
}

public struct AcceleratorGateResult: Sendable, Equatable {
    public let candidate: AcceleratorCandidate
    public let decision: AcceleratorGateDecision
    public let reason: String

    public init(candidate: AcceleratorCandidate, decision: AcceleratorGateDecision, reason: String) {
        self.candidate = candidate
        self.decision = decision
        self.reason = reason
    }
}

/// What to do at load (or at the next request) after the gate and any remembered answer.
public struct AcceleratorAttachPlan: Sendable, Equatable {
    /// Known-faster, or previously accepted, best first. Try in order; stop at the first that loads.
    public let attach: [AcceleratorCandidate]
    /// No profile yet, and nothing better is already selected. Show these at load time.
    public let ask: [AcceleratorGateResult]
    /// Known-worse, or the user already said no.
    public let skipped: [AcceleratorGateResult]

    public init(attach: [AcceleratorCandidate], ask: [AcceleratorGateResult], skipped: [AcceleratorGateResult]) {
        self.attach = attach
        self.ask = ask
        self.skipped = skipped
    }
}

public struct AcceleratorPromptOption: Sendable, Equatable, Codable {
    public let acceleratorId: String
    public let kind: String
    public let displayName: String
    public let reason: String
    public let estimatedSizeMB: Int

    public init(
        acceleratorId: String,
        kind: String,
        displayName: String,
        reason: String,
        estimatedSizeMB: Int
    ) {
        self.acceleratorId = acceleratorId
        self.kind = kind
        self.displayName = displayName
        self.reason = reason
        self.estimatedSizeMB = estimatedSizeMB
    }
}

public struct AcceleratorPrompt: Sendable, Equatable, Codable, Identifiable {
    public let mainModelId: String
    public let family: ModelFamily
    public let modelType: ModelType
    public let options: [AcceleratorPromptOption]
    /// Companion already attached because its profile is `auto_on`.
    public let autoAttachedId: String?

    public var id: String { mainModelId }

    public init(
        mainModelId: String,
        family: ModelFamily,
        modelType: ModelType,
        options: [AcceleratorPromptOption],
        autoAttachedId: String?
    ) {
        self.mainModelId = mainModelId
        self.family = family
        self.modelType = modelType
        self.options = options
        self.autoAttachedId = autoAttachedId
    }
}

// MARK: - Gate

/// Explicit speed profiles for speculative companions.
///
/// Missing data is `ask_user`, never `auto_on`. A remembered "yes" does not
/// override `auto_off`.
public enum AcceleratorSpeedGate {

    public static func decision(
        mainModelId: String,
        acceleratorId: String,
        kind: AcceleratorKind,
        scenario: SpecScenario
    ) -> (decision: AcceleratorGateDecision, reason: String) {
        let matched = rules.filter {
            $0.matches(mainModelId: mainModelId, acceleratorId: acceleratorId)
        }
        let specific = matched.filter { $0.scenario == scenario }
        let pool = specific.isEmpty ? matched.filter { $0.scenario == nil } : specific
        let hinted = pool.filter { !$0.mainHints.isEmpty }
        if let rule = (hinted.isEmpty ? pool : hinted).first {
            return (rule.decision, rule.reason)
        }
        return (
            .askUser,
            "No measured tokens-per-second profile for \(acceleratorId) on this model, so it is not loaded automatically."
        )
    }

    /// `remembered` maps accelerator id → the user's last answer.
    /// A `true` answer loads an `ask_user` companion next time. It does not load an `auto_off` one.
    /// Unknown companions that rank below an already-on companion are not asked about; they would not run.
    public static func plan(
        mainModelId: String,
        scenario: SpecScenario = .unspecified,
        candidates: [AcceleratorCandidate],
        remembered: [String: Bool] = [:]
    ) -> AcceleratorAttachPlan {
        var on: [AcceleratorCandidate] = []
        var ask: [AcceleratorGateResult] = []
        var skipped: [AcceleratorGateResult] = []

        for candidate in candidates {
            let (gate, reason) = decision(
                mainModelId: mainModelId,
                acceleratorId: candidate.acceleratorId,
                kind: candidate.kind,
                scenario: scenario
            )
            switch gate {
            case .autoOff:
                skipped.append(AcceleratorGateResult(candidate: candidate, decision: gate, reason: reason))
            case .autoOn:
                on.append(candidate)
            case .askUser:
                if let choice = remembered[candidate.acceleratorId] {
                    if choice {
                        on.append(candidate)
                    } else {
                        skipped.append(AcceleratorGateResult(
                            candidate: candidate,
                            decision: gate,
                            reason: "Saved choice is off. \(reason)"
                        ))
                    }
                } else {
                    ask.append(AcceleratorGateResult(candidate: candidate, decision: gate, reason: reason))
                }
            }
        }

        let indexed = on.enumerated().sorted { lhs, rhs in
            let left = AcceleratorPreference.rank(lhs.element.kind)
            let right = AcceleratorPreference.rank(rhs.element.kind)
            if left != right { return left < right }
            return lhs.offset < rhs.offset
        }
        on = indexed.map(\.element)
        let bestOn = on.map { AcceleratorPreference.rank($0.kind) }.min()
        if let bestOn {
            let (keep, drop) = ask.stablePartition { AcceleratorPreference.rank($0.candidate.kind) < bestOn }
            ask = keep
            skipped.append(contentsOf: drop.map {
                AcceleratorGateResult(
                    candidate: $0.candidate,
                    decision: $0.decision,
                    reason: "A higher-priority booster is already selected. \($0.reason)"
                )
            })
        }
        return AcceleratorAttachPlan(attach: on, ask: ask, skipped: skipped)
    }

    /// Rough scenario from the user text. Load time passes `unspecified`.
    public static func scenario(messageCharacters: Int, containsCodeFence: Bool) -> SpecScenario {
        if containsCodeFence { return .codeMath }
        if messageCharacters >= 6_000 { return .chatLong }
        if messageCharacters > 0 && messageCharacters < 400 { return .chatShort }
        return .unspecified
    }

    // MARK: Seeded profiles

    private struct Rule {
        var mainHints: [String]
        var acceleratorHint: String
        /// `nil` applies when no scenario-specific rule matches.
        var scenario: SpecScenario?
        var decision: AcceleratorGateDecision
        var reason: String

        func matches(mainModelId: String, acceleratorId: String) -> Bool {
            let accel = acceleratorId.lowercased()
            guard accel.contains(acceleratorHint) else { return false }
            if mainHints.isEmpty { return true }
            return mainHints.contains { hint in
                Self.mainMatches(mainModelId, hint: hint)
            }
        }

        private static func mainMatches(_ modelId: String, hint: String) -> Bool {
            let leaf = (modelId.split(separator: "/").last.map(String.init) ?? modelId).lowercased()
            switch hint {
            case "qwen3-8b":
                let is8 = leaf.contains("qwen3-8b") || leaf.contains("qwen3_8b")
                let is38 = leaf.contains("qwen3.8") || leaf.contains("qwen3_8-")
                return is8 && !is38
            case "qwen3-14b":
                return leaf.contains("qwen3-14b") || leaf.contains("qwen3_14b")
            case "qwen3.8-27b":
                return leaf.contains("qwen3.8-27b") || leaf.contains("qwen3_8-27b")
            case "deepseek-v4.1-flash":
                return leaf.contains("deepseek-v4.1-flash")
            default:
                return leaf.contains(hint) || modelId.lowercased().contains(hint)
            }
        }
    }

    /// Sources are called out in each reason. Anything not listed is `ask_user`.
    private static let rules: [Rule] = [
        Rule(
            mainHints: ["qwen3-8b"],
            acceleratorHint: "dspark_qwen3_8b_block7",
            scenario: nil,
            decision: .autoOn,
            reason: "Qwen3-8B DSpark was faster than plain decode on Apple Silicon (about 1.2–1.8×, PR #14)."
        ),
        Rule(
            mainHints: ["qwen3-14b"],
            acceleratorHint: "dspark_qwen3_14b_block7",
            scenario: nil,
            decision: .autoOn,
            reason: "Qwen3-14B DSpark was faster than plain decode on Apple Silicon (about 1.1–1.8×, PR #14). Code runs were noisier but still above 1×."
        ),
        Rule(
            mainHints: ["qwen3.8-27b"],
            acceleratorHint: "qwen3.8-27b-dflash2",
            scenario: .chatShort,
            decision: .autoOff,
            reason: "Short greedy chat on Qwen3.8-27B was slower with DFlash2 (about 22 tok/s) than without it (about 30 tok/s). No later short-chat win is recorded."
        ),
        Rule(
            mainHints: ["qwen3.8-27b"],
            acceleratorHint: "qwen3.8-27b-dflash2",
            scenario: .chatLong,
            decision: .askUser,
            reason: "Long-prompt decode is the case DFlash2 is meant to speed up, but a passing median versus plain decode is not recorded."
        ),
        Rule(
            mainHints: ["qwen3.8-27b"],
            acceleratorHint: "qwen3.8-27b-dflash2",
            scenario: .codeMath,
            decision: .askUser,
            reason: "No code or math speed profile is recorded for this DFlash2 companion."
        ),
        Rule(
            mainHints: ["qwen3.8-27b"],
            acceleratorHint: "qwen3.8-27b-dflash2",
            scenario: nil,
            decision: .askUser,
            reason: "Short chat with this DFlash2 companion was slower, and a long-prompt win is not recorded. Choose whether to load it."
        ),
        Rule(
            mainHints: ["qwen3.8-27b"],
            acceleratorHint: "qwen3.8-27b-speculator.dspark",
            scenario: nil,
            decision: .askUser,
            reason: "No measured tokens-per-second win versus plain decode for the RedHat DSpark speculator."
        ),
        Rule(
            mainHints: ["qwen3.8-27b"],
            acceleratorHint: "qwen3.8-27b-mtp",
            scenario: nil,
            decision: .askUser,
            reason: "No measured tokens-per-second win versus plain decode for the Qwen3.8 MTP companion. It is also greedy-only."
        ),
        Rule(
            mainHints: ["deepseek-v4.1-flash"],
            acceleratorHint: "deepseek-v4.1-flash-dspark-drafter",
            scenario: nil,
            decision: .askUser,
            reason: "The split DSpark drafter has no recorded win over the in-graph head, so it is not loaded automatically."
        ),
        Rule(
            mainHints: [],
            acceleratorHint: "qwen3-0.6b",
            scenario: nil,
            decision: .askUser,
            reason: "No measured tokens-per-second win versus plain decode for the Qwen3 0.6B draft."
        ),
        Rule(
            mainHints: [],
            acceleratorHint: "llama-3.2-1b",
            scenario: nil,
            decision: .askUser,
            reason: "No measured tokens-per-second win versus plain decode for the Llama 3.2 1B draft."
        ),
        Rule(
            mainHints: [],
            acceleratorHint: "gemma-2-2b",
            scenario: nil,
            decision: .askUser,
            reason: "No measured tokens-per-second win versus plain decode for the Gemma 2 2B draft."
        ),
    ]
}

private extension Array {
    /// Stable split. `true` values stay in order, then `false` values stay in order.
    func stablePartition(_ keep: (Element) -> Bool) -> (keep: [Element], drop: [Element]) {
        var yes: [Element] = []
        var no: [Element] = []
        for element in self {
            if keep(element) { yes.append(element) } else { no.append(element) }
        }
        return (yes, no)
    }
}
