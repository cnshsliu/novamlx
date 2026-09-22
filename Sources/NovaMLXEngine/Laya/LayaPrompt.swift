import Foundation

public enum LayaQuestionKind: String, Sendable, Equatable {
    case choice
    case score
    case noul

    public var typeIndex: Int {
        switch self {
        case .choice: return 0
        case .score: return 1
        case .noul: return 2
        }
    }

    public static func name(for index: Int) -> String {
        switch index {
        case 0: return "choice"
        case 1: return "score"
        default: return "noul"
        }
    }
}

public struct LayaQuestionSpec: Sendable, Equatable {
    public var kind: LayaQuestionKind
    public var instructions: String
    /// Choice labels in order. Score levels in order. Noul ignores this.
    public var choiceLabels: [String]
    public var choiceDetails: [String?]
    public var scoreLevels: [String]
    public var noulFalse: String?
    public var noulTrue: String?

    public init(
        kind: LayaQuestionKind,
        instructions: String,
        choiceLabels: [String] = [],
        choiceDetails: [String?] = [],
        scoreLevels: [String] = [],
        noulFalse: String? = nil,
        noulTrue: String? = nil
    ) {
        self.kind = kind
        self.instructions = instructions
        self.choiceLabels = choiceLabels
        self.choiceDetails = choiceDetails
        self.scoreLevels = scoreLevels
        self.noulFalse = noulFalse
        self.noulTrue = noulTrue
    }
}

public struct LayaSequence: Sendable, Equatable {
    public var ids: [Int]
    public var markers: [Int]
    public var qtype: Int
}

public protocol LayaTokenizing: Sendable {
    func encode(_ text: String) -> [Int]
    var clsTokenId: Int { get }
    var sepTokenId: Int { get }
    var padTokenId: Int { get }
    var maskTokenId: Int { get }
    var maskToken: String { get }
}

public enum LayaPrompt {
    public static func renderOptions(_ q: LayaQuestionSpec) -> [String] {
        switch q.kind {
        case .choice:
            return zip(q.choiceLabels, paddedDetails(q)).map { label, detail in
                if let detail, !detail.isEmpty {
                    return "\(label): \(detail)"
                }
                return label
            }
        case .score:
            return q.scoreLevels.enumerated().map { i, level in
                "level \(i): \(level)"
            }
        case .noul:
            let no = (q.noulFalse?.isEmpty == false)
                ? q.noulFalse!
                : "no, the statement does not hold"
            let yes = (q.noulTrue?.isEmpty == false)
                ? q.noulTrue!
                : "yes, the statement holds"
            return ["false: \(no)", "true: \(yes)"]
        }
    }

    public static func buildSequence(
        tokenizer: any LayaTokenizing,
        state: String,
        question: LayaQuestionSpec,
        maxLen: Int = 512,
        headMaxLen: Int = 192
    ) -> LayaSequence {
        let mask = tokenizer.maskToken
        let options = renderOptions(question)
        let instructions = question.instructions.replacingOccurrences(of: mask, with: " ")
        var head = tokenizer.encode("\(question.kind.rawValue) question: \(instructions)")
        var optionIds: [[Int]] = options.map { option in
            let body = tokenizer.encode(" " + option.replacingOccurrences(of: mask, with: " "))
            return [tokenizer.maskTokenId] + Array(body.prefix(48))
        }
        var budget = headMaxLen - optionIds.reduce(0) { $0 + $1.count }
        if budget < 16 {
            let per = max(4, (headMaxLen - 16) / max(1, optionIds.count))
            optionIds = optionIds.map { Array($0.prefix(per)) }
            budget = headMaxLen - optionIds.reduce(0) { $0 + $1.count }
        }
        head = Array(head.prefix(max(8, budget)))
        var ids = [tokenizer.clsTokenId] + head + [tokenizer.sepTokenId]
        var markers: [Int] = []
        for option in optionIds {
            markers.append(ids.count)
            ids.append(contentsOf: option)
        }
        ids.append(tokenizer.sepTokenId)
        let room = max(0, maxLen - ids.count - 1)
        let stateText = state.replacingOccurrences(of: mask, with: " ")
        let stateIds = Array(tokenizer.encode(stateText).prefix(room))
        ids.append(contentsOf: stateIds)
        ids.append(tokenizer.sepTokenId)
        let clipped = Array(ids.prefix(maxLen))
        return LayaSequence(
            ids: clipped,
            markers: markers.filter { $0 < maxLen },
            qtype: question.kind.typeIndex
        )
    }

    public static func confidence(from probs: [Double]) -> Double {
        let k = probs.count
        guard k >= 2 else { return 1 }
        let entropy = -probs.reduce(0.0) { partial, p in
            let clipped = min(1, max(1e-12, p))
            return partial + clipped * log(clipped)
        }
        let value = 1 - entropy / log(Double(k))
        return min(1, max(0, value))
    }

    public static func tempBucket(qtype: Int, optionCount: Int) -> String {
        let size: String
        if optionCount <= 2 { size = "2" }
        else if optionCount <= 5 { size = "3-5" }
        else if optionCount <= 10 { size = "6-10" }
        else { size = "11+" }
        return "\(LayaQuestionKind.name(for: qtype)):\(size)"
    }

    public static func round4(_ value: Double) -> Double {
        (value * 10_000).rounded() / 10_000
    }

    private static func paddedDetails(_ q: LayaQuestionSpec) -> [String?] {
        q.choiceLabels.indices.map { index in
            index < q.choiceDetails.count ? q.choiceDetails[index] : nil
        }
    }
}
