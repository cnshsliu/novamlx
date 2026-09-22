import Foundation
import MLX
import MLXNN
import NovaMLXCore
import Tokenizers

public struct LayaAnswer: Sendable, Equatable {
    public var type: String
    public var confidence: Double
    public var actProbability: Double
    public var choice: String?
    public var probabilities: [String: Double]
    public var score: Double?
    public var legend: [String: String]
    public var noul: Double?
}

public struct LayaPredictResult: Sendable {
    public var modelId: String
    public var answers: [String: LayaAnswer]
    public var inputTokens: Int
}

struct LayaRuntimeTokenizer: LayaTokenizing {
    let backend: any Tokenizers.Tokenizer
    let clsTokenId: Int
    let sepTokenId: Int
    let padTokenId: Int
    let maskTokenId: Int
    let maskToken: String

    func encode(_ text: String) -> [Int] {
        backend.encode(text: text, addSpecialTokens: false)
    }

    static func load(from tokenizerDir: URL) async throws -> LayaRuntimeTokenizer {
        let backend = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        let configURL = tokenizerDir.appendingPathComponent("tokenizer_config.json")
        let data = try Data(contentsOf: configURL)
        let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        func token(_ key: String) throws -> (String, Int) {
            var value = raw[key]
            if let dict = value as? [String: Any] {
                value = dict["content"]
            }
            guard let text = value as? String, let id = backend.convertTokenToId(text) else {
                throw NovaMLXError.configurationError("Laya tokenizer is missing \(key)")
            }
            return (text, id)
        }
        let cls = try token("cls_token")
        let sep = try token("sep_token")
        let pad = try token("pad_token")
        let mask = try token("mask_token")
        return LayaRuntimeTokenizer(
            backend: backend,
            clsTokenId: cls.1,
            sepTokenId: sep.1,
            padTokenId: pad.1,
            maskTokenId: mask.1,
            maskToken: mask.0
        )
    }
}

public enum LayaCheckpoint {
    public static let defaultModelId = "aac6fef/laya-multilingual-mlx"

    public static func isLayaDirectory(_ url: URL) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: url.appendingPathComponent("rl_agent_config.json").path)
            && fm.fileExists(atPath: url.appendingPathComponent("model.safetensors").path)
            && fm.fileExists(atPath: url.appendingPathComponent("encoder/config.json").path)
    }
}

public final class LayaAgent: @unchecked Sendable {
    public let modelId: String
    public let maxLen: Int
    let tokenizer: LayaRuntimeTokenizer
    let model: LayaDecisionModel
    let temperature: [Double]
    let temperatureByOptions: [String: Double]
    let headMaxLen: Int

    public init(directory: URL, modelId: String) async throws {
        guard LayaCheckpoint.isLayaDirectory(directory) else {
            throw NovaMLXError.configurationError("Not a Laya checkpoint: \(directory.path)")
        }
        let agentJSON = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("rl_agent_config.json"))
        ) as? [String: Any] ?? [:]
        let encoderJSON = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("encoder/config.json"))
        ) as? [String: Any] ?? [:]
        let encoder = try LayaEncoderConfig.parse(encoderJSON)
        let headLayers = agentJSON["head_layers"] as? Int ?? 2
        let actCosts = agentJSON["act_costs"] as? [String: Any] ?? [:]
        let maxLen = agentJSON["max_len"] as? Int ?? 512
        let headMaxLen = agentJSON["head_max_len"] as? Int ?? 192
        guard 4 < headMaxLen, headMaxLen < maxLen, maxLen <= encoder.maxPositionEmbeddings else {
            throw NovaMLXError.configurationError("Invalid Laya context: head \(headMaxLen) max \(maxLen)")
        }
        var temps = (agentJSON["temperature"] as? [Double]) ?? [1, 1, 1]
        if temps.count != 3, let ints = agentJSON["temperature"] as? [Int], ints.count == 3 {
            temps = ints.map(Double.init)
        }
        guard temps.count == 3, temps.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw NovaMLXError.configurationError("Laya calibration temperatures are invalid")
        }
        var buckets: [String: Double] = [:]
        if let raw = agentJSON["temperature_by_options"] as? [String: Double] {
            buckets = raw
        } else if let raw = agentJSON["temperature_by_options"] as? [String: Int] {
            buckets = raw.mapValues(Double.init)
        }
        let tokenizer = try await LayaRuntimeTokenizer.load(
            from: directory.appendingPathComponent("tokenizer", isDirectory: true)
        )
        let model = LayaDecisionModel(
            encoderConfig: encoder,
            headLayers: headLayers,
            actClasses: actCosts.count + 1
        )
        let weights = try MLX.loadArrays(
            url: directory.appendingPathComponent("model.safetensors")
        )
        let cast = weights.mapValues { $0.asType(.float16) }
        try model.update(parameters: ModuleParameters.unflattened(cast), verify: .all)
        self.modelId = modelId
        self.maxLen = maxLen
        self.headMaxLen = headMaxLen
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temps
        self.temperatureByOptions = buckets
    }

    public func predict(state: String, questions: [String: LayaQuestionSpec]) throws -> LayaPredictResult {
        guard !questions.isEmpty else {
            throw NovaMLXError.apiError("questions must not be empty")
        }
        var items: [(id: String, spec: LayaQuestionSpec, sequence: LayaSequence)] = []
        for (id, spec) in questions {
            let sequence = LayaPrompt.buildSequence(
                tokenizer: tokenizer,
                state: state,
                question: spec,
                maxLen: maxLen,
                headMaxLen: headMaxLen
            )
            let expected = LayaPrompt.renderOptions(spec).count
            guard sequence.markers.count == expected else {
                throw NovaMLXError.apiError("Question \(id) does not fit the token budget")
            }
            items.append((id, spec, sequence))
        }
        var answers: [String: LayaAnswer] = [:]
        let batchSize = 16
        var start = 0
        while start < items.count {
            let end = min(items.count, start + batchSize)
            let chunk = Array(items[start..<end])
            let (logits, action) = try forward(chunk.map(\.sequence))
            let logitRows = logits.asArray(Float.self)
            let actionRows = action.asArray(Float.self)
            let width = chunk.map { $0.sequence.markers.count }.max() ?? 1
            let padded = max(2, width)
            let actClasses = action.dim(1)
            for (row, item) in chunk.enumerated() {
                let k = item.sequence.markers.count
                let scale = temperatureByOptions[
                    LayaPrompt.tempBucket(qtype: item.sequence.qtype, optionCount: k)
                ] ?? temperature[item.sequence.qtype]
                var z = (0..<k).map { i in
                    Double(logitRows[row * padded + i]) / max(1e-3, scale)
                }
                let peak = z.max() ?? 0
                z = z.map { exp($0 - peak) }
                let sum = z.reduce(0, +)
                let p = z.map { $0 / sum }
                var act = (0..<actClasses).map { i in Double(actionRows[row * actClasses + i]) }
                let actPeak = act.max() ?? 0
                act = act.map { exp($0 - actPeak) }
                let actSum = act.reduce(0, +)
                let actP = act.map { $0 / actSum }
                var answer = LayaAnswer(
                    type: item.spec.kind.rawValue,
                    confidence: LayaPrompt.round4(LayaPrompt.confidence(from: p)),
                    actProbability: LayaPrompt.round4(actP.first ?? 0),
                    choice: nil,
                    probabilities: [:],
                    score: nil,
                    legend: [:],
                    noul: nil
                )
                switch item.spec.kind {
                case .choice:
                    let labels = item.spec.choiceLabels
                    let best = p.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
                    answer.choice = labels[best]
                    for (label, value) in zip(labels, p) {
                        answer.probabilities[label] = LayaPrompt.round4(value)
                    }
                case .score:
                    let expected = p.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element }
                    answer.score = LayaPrompt.round4(expected)
                    for (i, level) in item.spec.scoreLevels.enumerated() {
                        answer.legend[String(i)] = level
                        answer.probabilities[String(i)] = LayaPrompt.round4(p[i])
                    }
                case .noul:
                    let yes = p.count > 1 ? p[1] : 0
                    answer.noul = LayaPrompt.round4(yes)
                    answer.confidence = LayaPrompt.round4(max(yes, 1 - yes))
                }
                answers[item.id] = answer
            }
            start = end
        }
        return LayaPredictResult(
            modelId: modelId,
            answers: answers,
            inputTokens: items.reduce(0) { $0 + $1.sequence.ids.count }
        )
    }

    private func forward(_ sequences: [LayaSequence]) throws -> (MLXArray, MLXArray) {
        let length = sequences.map { $0.ids.count }.max() ?? 1
        let markerCount = max(2, sequences.map { $0.markers.count }.max() ?? 1)
        let batch = sequences.count
        var ids = [Int32](repeating: Int32(tokenizer.padTokenId), count: batch * length)
        var mask = [Int32](repeating: 0, count: batch * length)
        var markers = [Int32](repeating: 0, count: batch * markerCount)
        var markerMask = [Int32](repeating: 0, count: batch * markerCount)
        var qtype = [Int32](repeating: 0, count: batch)
        for (row, seq) in sequences.enumerated() {
            for (i, id) in seq.ids.enumerated() {
                ids[row * length + i] = Int32(id)
                mask[row * length + i] = 1
            }
            for (i, marker) in seq.markers.enumerated() {
                markers[row * markerCount + i] = Int32(marker)
                markerMask[row * markerCount + i] = 1
            }
            qtype[row] = Int32(seq.qtype)
        }
        let logits: MLXArray
        let action: MLXArray
        (logits, action) = model(
            inputIds: MLXArray(ids).reshaped([batch, length]),
            attentionMask: MLXArray(mask).reshaped([batch, length]),
            markerPos: MLXArray(markers).reshaped([batch, markerCount]),
            markerMask: MLXArray(markerMask).reshaped([batch, markerCount]),
            qtype: MLXArray(qtype)
        )
        eval(logits, action)
        return (logits, action)
    }
}
