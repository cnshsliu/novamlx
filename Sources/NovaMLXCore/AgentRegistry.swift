import Foundation

public enum AgentConfigFormat: String, Sendable {
    case codex
    case claudeCode
    case hermes
    case opencode
    case curl
    case python
    case node
}

public struct AgentSpec: Identifiable, Sendable, Equatable {
    public let id: String
    public let displayName: String
    public let icon: String
    public let hasApp: Bool
    public let hasCLI: Bool
    public let appName: String?
    public let cliCommand: String
    public let configFormat: AgentConfigFormat

    public init(
        id: String,
        displayName: String,
        icon: String,
        hasApp: Bool,
        hasCLI: Bool,
        appName: String? = nil,
        cliCommand: String,
        configFormat: AgentConfigFormat
    ) {
        self.id = id
        self.displayName = displayName
        self.icon = icon
        self.hasApp = hasApp
        self.hasCLI = hasCLI
        self.appName = appName
        self.cliCommand = cliCommand
        self.configFormat = configFormat
    }
}

public enum AgentRegistry {
    public static let all: [AgentSpec] = [
        AgentSpec(
            id: "codex",
            displayName: "Codex",
            icon: "laptopcomputer.and.arrow.down",
            hasApp: true,
            hasCLI: true,
            appName: "Codex",
            cliCommand: "codex",
            configFormat: .codex
        ),
        AgentSpec(
            id: "claude-code",
            displayName: "Claude Code",
            icon: "brain.head.profile",
            hasApp: false,
            hasCLI: true,
            cliCommand: "claude",
            configFormat: .claudeCode
        ),
        AgentSpec(
            id: "hermes",
            displayName: "Hermes",
            icon: "bolt.horizontal.fill",
            hasApp: false,
            hasCLI: true,
            cliCommand: "hermes",
            configFormat: .hermes
        ),
        AgentSpec(
            id: "opencode",
            displayName: "OpenCode",
            icon: "terminal.fill",
            hasApp: false,
            hasCLI: true,
            cliCommand: "opencode",
            configFormat: .opencode
        ),
        AgentSpec(
            id: "curl",
            displayName: "cURL",
            icon: "terminal",
            hasApp: false,
            hasCLI: true,
            cliCommand: "curl",
            configFormat: .curl
        ),
        AgentSpec(
            id: "python",
            displayName: "Python",
            icon: "doc.text",
            hasApp: false,
            hasCLI: true,
            cliCommand: "python3",
            configFormat: .python
        ),
        AgentSpec(
            id: "node",
            displayName: "Node.js",
            icon: "doc.text",
            hasApp: false,
            hasCLI: true,
            cliCommand: "node",
            configFormat: .node
        ),
    ]

    public static func byId(_ id: String) -> AgentSpec? {
        all.first { $0.id == id }
    }
}
