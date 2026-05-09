import Foundation

public enum KeepAliveValue: Codable, Sendable, Equatable {
    case seconds(Int)
    case infinite
    case immediate

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) {
            if intVal == 0 {
                self = .immediate
            } else if intVal < 0 {
                self = .infinite
            } else {
                self = .seconds(intVal)
            }
        } else if let strVal = try? container.decode(String.self) {
            switch strVal.lowercased().trimmingCharacters(in: .whitespaces) {
            case "infinite", "infinity":
                self = .infinite
            default:
                guard let seconds = Self.parseDuration(strVal) else {
                    throw DecodingError.dataCorruptedError(
                        in: container,
                        debugDescription: "Invalid keep_alive value: '\(strVal)'. Use integer seconds, duration string (e.g. '5m', '1h'), 0 for immediate, or -1 for infinite."
                    )
                }
                self = .seconds(seconds)
            }
        } else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "keep_alive must be an integer (seconds) or duration string (e.g. '5m', '1h', '0', '-1')"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .seconds(let s): try container.encode(s)
        case .infinite: try container.encode(-1)
        case .immediate: try container.encode(0)
        }
    }

    public func deadline(from reference: Date = Date()) -> Date {
        switch self {
        case .seconds(let s): return reference.addingTimeInterval(Double(s))
        case .infinite: return Date.distantFuture
        case .immediate: return reference
        }
    }

    private static func parseDuration(_ str: String) -> Int? {
        let lowered = str.lowercased().trimmingCharacters(in: .whitespaces)
        guard let last = lowered.last, let value = Int(lowered.dropLast()), value >= 0 else { return nil }
        switch last {
        case "s": return value
        case "m": return value * 60
        case "h": return value * 3600
        case "d": return value * 86400
        default: return nil
        }
    }
}
