import HTTPTypes
import Hummingbird

enum ClientType: Equatable, Sendable, CustomStringConvertible {
    case agentTool(name: String)
    case generalChat

    var shouldScaleContext: Bool {
        switch self {
        case .agentTool: return true
        case .generalChat: return false
        }
    }

    var description: String {
        switch self {
        case .agentTool(let name): return "agentTool(\(name))"
        case .generalChat: return "generalChat"
        }
    }
}

enum ClientDetector {
    /// Ordered list of known agent User-Agent substrings (case-insensitive).
    private static let userAgentPatterns: [(agent: String, pattern: String)] = [
        ("Claude Code", "claude"),
        ("OpenCode", "opencode"),
        ("OpenClaw", "openclaw"),
        ("Hermes", "hermes"),
    ]

    private static let anthropicVersionHeader = HTTPField.Name("anthropic-version")!

    /// Detect client type from HTTP request headers.
    ///
    /// Priority:
    /// 1. `Anthropic-Version` header → Claude Code (definitive)
    /// 2. `User-Agent` substring match → known agent tool
    /// 3. Fallback → generalChat (no scaling)
    static func detect(request: Request) -> ClientType {
        if request.headers[anthropicVersionHeader] != nil {
            return .agentTool(name: "Claude Code")
        }

        if let userAgent = request.headers[.userAgent] {
            let lower = userAgent.lowercased()
            for (agentName, pattern) in userAgentPatterns {
                if lower.contains(pattern) {
                    return .agentTool(name: agentName)
                }
            }
        }

        return .generalChat
    }
}
