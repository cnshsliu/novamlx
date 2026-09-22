import Foundation
import NovaMLXCore
import NovaMLXEngine

struct DecisionHTTPResponse: Encodable {
    struct Answer: Encodable {
        let type: String
        let confidence: Double
        let choice: String?
        let score: Double?
        let noul: Double?
        let probabilities: [String: Double]
        let legend: [String: String]
        let action: Action
    }

    struct Action: Encodable {
        let actProbability: Double
        enum CodingKeys: String, CodingKey {
            case actProbability = "act_probability"
        }
    }

    struct Usage: Encodable {
        let inputTokens: Int
        let outputTokens: Int
        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
    }

    let model: String
    let answers: [String: Answer]
    let usage: Usage
}

extension NovaMLXAPIServer {
    static func decisionState(_ value: Any?) throws -> String {
        if let text = value as? String { return text }
        if let value {
            let data = try JSONSerialization.data(withJSONObject: value)
            if let text = String(data: data, encoding: .utf8) { return text }
        }
        throw NovaMLXError.apiError("decisions state must be a string or JSON value")
    }
}
