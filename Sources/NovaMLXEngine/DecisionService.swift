import Foundation
import NovaMLXCore
import NovaMLXUtils

public final class DecisionService: @unchecked Sendable {
    private var agents: [String: LayaAgent] = [:]
    private let lock = NovaMLXLock()

    public init() {}

    public func isLoaded(_ modelId: String) -> Bool {
        lock.withLock { agents[modelId] != nil }
    }

    public func listLoadedModels() -> [String] {
        lock.withLock { Array(agents.keys) }
    }

    public func load(from url: URL, modelId: String) async throws {
        let agent = try await LayaAgent(directory: url, modelId: modelId)
        lock.withLock { agents[modelId] = agent }
        NovaMLXLog.info("Laya decision model loaded: \(modelId)")
    }

    public func unload(modelId: String) {
        lock.withLock { _ = agents.removeValue(forKey: modelId) }
        NovaMLXLog.info("Laya decision model unloaded: \(modelId)")
    }

    public func predict(
        modelId: String,
        state: String,
        questions: [String: LayaQuestionSpec]
    ) throws -> LayaPredictResult {
        let agent = lock.withLock { agents[modelId] }
        guard let agent else { throw NovaMLXError.modelNotFound(modelId) }
        return try agent.predict(state: state, questions: questions)
    }

    public static func questions(from raw: [String: Any]) throws -> [String: LayaQuestionSpec] {
        var out: [String: LayaQuestionSpec] = [:]
        for (id, value) in raw {
            guard let dict = value as? [String: Any] else {
                throw NovaMLXError.apiError("Question \(id) must be an object")
            }
            guard let kind = LayaQuestionKind(rawValue: dict["type"] as? String ?? "") else {
                throw NovaMLXError.apiError("Question \(id) type must be choice, score, or noul")
            }
            let instructions = dict["instructions"] as? String ?? ""
            guard !instructions.isEmpty else {
                throw NovaMLXError.apiError("Question \(id) is missing instructions")
            }
            switch kind {
            case .choice:
                let (labels, details) = try choiceCriteria(dict["criteria"], id: id)
                out[id] = LayaQuestionSpec(
                    kind: .choice,
                    instructions: instructions,
                    choiceLabels: labels,
                    choiceDetails: details
                )
            case .score:
                guard let levels = dict["criteria"] as? [String], !levels.isEmpty else {
                    throw NovaMLXError.apiError("Question \(id) score criteria must be a nonempty list")
                }
                out[id] = LayaQuestionSpec(
                    kind: .score, instructions: instructions, scoreLevels: levels
                )
            case .noul:
                let map = dict["criteria"] as? [String: String]
                out[id] = LayaQuestionSpec(
                    kind: .noul,
                    instructions: instructions,
                    noulFalse: map?["false"],
                    noulTrue: map?["true"]
                )
            }
        }
        return out
    }

    private static func choiceCriteria(_ raw: Any?, id: String) throws -> ([String], [String?]) {
        if let list = raw as? [String] {
            guard !list.isEmpty, Set(list).count == list.count else {
                throw NovaMLXError.apiError("Question \(id) choice labels must be unique and nonempty")
            }
            return (list, Array(repeating: nil, count: list.count))
        }
        if let list = raw as? [[String: String]] {
            let labels = list.compactMap { $0["label"] ?? $0.keys.first }
            guard !labels.isEmpty, Set(labels).count == labels.count else {
                throw NovaMLXError.apiError("Question \(id) choice labels must be unique and nonempty")
            }
            let details = list.map { $0["description"] ?? $0["detail"] }
            return (labels, details)
        }
        throw NovaMLXError.apiError(
            "Question \(id) choice criteria must be a list of labels"
        )
    }
}
