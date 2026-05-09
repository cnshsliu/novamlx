import Foundation

/// A reusable recipe that wraps a base model with a system prompt, sampling
/// parameters, and tool definitions.  Stored as JSON in ~/.nova/modelfiles/.
///
/// When a chat request uses a modelfile name as the `model` field, the server
/// resolves it to the base model for loading/caching but injects the stored
/// overrides (system prompt, params, tools) into the request.
public struct Modelfile: Codable, Sendable, Equatable {
    /// Human-readable name, used as filename and lookup key.
    /// Must match `^[a-zA-Z0-9_-]{1,64}$`.
    public let name: String
    /// The real model ID to load (e.g. "mlx-community/Llama-3.2-3B-Instruct-4bit").
    public let baseModel: String
    /// Prepended as a system message (if set) before the user's messages.
    public let systemPrompt: String?
    /// Sampling overrides — only non-nil values are applied.
    public let parameters: ModelfileParameters?
    /// Tool JSON objects to inject into the request's `tools` array.
    public let tools: [[String: AnyCodableModelfile]]?
    /// Optional one-line description shown in `nova modelfile list`.
    public let description: String?

    public init(
        name: String,
        baseModel: String,
        systemPrompt: String? = nil,
        parameters: ModelfileParameters? = nil,
        tools: [[String: AnyCodableModelfile]]? = nil,
        description: String? = nil
    ) {
        self.name = name
        self.baseModel = baseModel
        self.systemPrompt = systemPrompt
        self.parameters = parameters
        self.tools = tools
        self.description = description
    }
}

/// Sampling parameters that a modelfile can override.
/// All fields optional — only non-nil values replace the request defaults.
public struct ModelfileParameters: Codable, Sendable, Equatable {
    public let temperature: Double?
    public let topP: Double?
    public let topK: Int?
    public let minP: Double?
    public let maxTokens: Int?
    public let frequencyPenalty: Double?
    public let presencePenalty: Double?
    public let repetitionPenalty: Double?
    public let stop: [String]?
    public let seed: UInt64?

    public init(
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        minP: Double? = nil,
        maxTokens: Int? = nil,
        frequencyPenalty: Double? = nil,
        presencePenalty: Double? = nil,
        repetitionPenalty: Double? = nil,
        stop: [String]? = nil,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.maxTokens = maxTokens
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.repetitionPenalty = repetitionPenalty
        self.stop = stop
        self.seed = seed
    }
}

// MARK: - AnyCodable for Modelfile tool definitions

/// Minimal AnyCodable wrapper so tool JSON can round-trip through Codable.
public enum AnyCodableModelfile: Codable, Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([AnyCodableModelfile])
    case dict([String: AnyCodableModelfile])

    public init(_ value: Any) {
        switch value {
        case is NSNull: self = .null
        case let b as Bool: self = .bool(b)
        case let i as Int: self = .int(i)
        case let d as Double: self = .double(d)
        case let s as String: self = .string(s)
        case let arr as [Any]: self = .array(arr.map { AnyCodableModelfile($0) })
        case let dict as [String: Any]:
            var result: [String: AnyCodableModelfile] = [:]
            for (k, v) in dict { result[k] = AnyCodableModelfile(v) }
            self = .dict(result)
        default: self = .null
        }
    }

    public func toAny() -> Any {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let a): return a.map { $0.toAny() }
        case .dict(let d): return d.mapValues { $0.toAny() }
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let b = try? container.decode(Bool.self) {
            self = .bool(b)
        } else if let i = try? container.decode(Int.self) {
            self = .int(i)
        } else if let d = try? container.decode(Double.self) {
            self = .double(d)
        } else if let s = try? container.decode(String.self) {
            self = .string(s)
        } else if let arr = try? container.decode([AnyCodableModelfile].self) {
            self = .array(arr)
        } else if let dict = try? container.decode([String: AnyCodableModelfile].self) {
            self = .dict(dict)
        } else {
            self = .null
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let b): try container.encode(b)
        case .int(let i): try container.encode(i)
        case .double(let d): try container.encode(d)
        case .string(let s): try container.encode(s)
        case .array(let a): try container.encode(a)
        case .dict(let d): try container.encode(d)
        }
    }
}

// MARK: - Name validation

extension Modelfile {
    nonisolated(unsafe) private static let validNamePattern = /^[a-zA-Z0-9_-]{1,64}$/

    /// Validate a modelfile name. Returns nil if valid, or an error message.
    public static func validateName(_ name: String) -> String? {
        if name.isEmpty { return "Name must not be empty" }
        if name.count > 64 { return "Name must be ≤ 64 characters" }
        if name.wholeMatch(of: validNamePattern) == nil {
            return "Name must contain only letters, digits, hyphens, and underscores"
        }
        return nil
    }
}
