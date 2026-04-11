import Foundation
import NovaMLXUtils

public struct StructuredOutputOptions: Codable, Sendable {
    public let jsonSchema: JSONSchemaFormat?
    public let regex: String?
    public let choice: [String]?
    public let grammar: String?

    private enum CodingKeys: String, CodingKey {
        case jsonSchema = "json"
        case regex, choice, grammar
    }

    public init(jsonSchema: JSONSchemaFormat? = nil, regex: String? = nil, choice: [String]? = nil, grammar: String? = nil) {
        self.jsonSchema = jsonSchema
        self.regex = regex
        self.choice = choice
        self.grammar = grammar
    }
}

public struct JSONSchemaFormat: Codable, Sendable {
    public let name: String?
    public let schema: [String: AnyCodable]?
    public let strict: Bool?

    public init(name: String? = nil, schema: [String: AnyCodable]? = nil, strict: Bool? = nil) {
        self.name = name
        self.schema = schema
        self.strict = strict
    }
}

public struct ResponseFormatJsonSchema: Codable, Sendable {
    public let type: String
    public let jsonSchema: JSONSchemaFormat?

    private enum CodingKeys: String, CodingKey {
        case type
        case jsonSchema = "json_schema"
    }

    public init(type: String = "json_schema", jsonSchema: JSONSchemaFormat? = nil) {
        self.type = type
        self.jsonSchema = jsonSchema
    }
}
