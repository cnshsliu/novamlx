import Foundation

public struct RerankRequest: Codable, Sendable {
    public let model: String
    public let query: String
    public let documents: [RerankDocument]
    public let topN: Int?
    public let returnDocuments: Bool?

    private enum CodingKeys: String, CodingKey {
        case model, query, documents
        case topN = "top_n"
        case returnDocuments = "return_documents"
    }

    public init(
        model: String,
        query: String,
        documents: [RerankDocument],
        topN: Int? = nil,
        returnDocuments: Bool? = nil
    ) {
        self.model = model
        self.query = query
        self.documents = documents
        self.topN = topN
        self.returnDocuments = returnDocuments
    }
}

public enum RerankDocument: Codable, Sendable {
    case text(String)
    case object(RerankDocumentObject)

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let str = try? container.decode(String.self) {
            self = .text(str)
        } else {
            self = .object(try container.decode(RerankDocumentObject.self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let s): try container.encode(s)
        case .object(let obj): try container.encode(obj)
        }
    }

    public var text: String {
        switch self {
        case .text(let s): s
        case .object(let obj): obj.text
        }
    }
}

public struct RerankDocumentObject: Codable, Sendable {
    public let text: String

    public init(text: String) {
        self.text = text
    }
}

public struct RerankResponse: Codable, Sendable {
    public let id: String
    public let results: [RerankResult]
    public let model: String
    public let usage: RerankUsage

    public init(id: String, results: [RerankResult], model: String, usage: RerankUsage) {
        self.id = id
        self.results = results
        self.model = model
        self.usage = usage
    }
}

public struct RerankResult: Codable, Sendable {
    public let index: Int
    public let relevanceScore: Double
    public let document: RerankDocument?

    private enum CodingKeys: String, CodingKey {
        case index
        case relevanceScore = "relevance_score"
        case document
    }

    public init(index: Int, relevanceScore: Double, document: RerankDocument? = nil) {
        self.index = index
        self.relevanceScore = relevanceScore
        self.document = document
    }
}

public struct RerankUsage: Codable, Sendable {
    public let totalTokens: Int

    private enum CodingKeys: String, CodingKey {
        case totalTokens = "total_tokens"
    }

    public init(totalTokens: Int) {
        self.totalTokens = totalTokens
    }
}
