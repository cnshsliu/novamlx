import Foundation
import NovaMLXCore
import NovaMLXUtils

public enum MCPTransport: String, Codable, Sendable {
    case stdio
    case sse
    case streamableHTTP = "streamable-http"
}

public struct MCPServerConfig: Codable, Sendable {
    public let transport: MCPTransport
    public let command: String?
    public let args: [String]?
    public let url: String?
    public let headers: [String: String]?
    public let enabled: Bool?
    public let timeout: TimeInterval?

    public init(
        transport: MCPTransport = .stdio,
        command: String? = nil,
        args: [String]? = nil,
        url: String? = nil,
        headers: [String: String]? = nil,
        enabled: Bool? = nil,
        timeout: TimeInterval? = nil
    ) {
        self.transport = transport
        self.command = command
        self.args = args
        self.url = url
        self.headers = headers
        self.enabled = enabled
        self.timeout = timeout
    }
}

public struct MCPConfig: Codable, Sendable {
    public let servers: [String: MCPServerConfig]
    public let maxToolCalls: Int?
    public let defaultTimeout: TimeInterval?

    private enum CodingKeys: String, CodingKey {
        case servers
        case maxToolCalls = "max_tool_calls"
        case defaultTimeout = "default_timeout"
    }

    public init(
        servers: [String: MCPServerConfig] = [:],
        maxToolCalls: Int? = nil,
        defaultTimeout: TimeInterval? = nil
    ) {
        self.servers = servers
        self.maxToolCalls = maxToolCalls
        self.defaultTimeout = defaultTimeout
    }

    public static func load(from url: URL) throws -> MCPConfig {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(MCPConfig.self, from: data)
    }
}

public struct MCPTool: Sendable {
    public let name: String
    public let description: String?
    public let inputSchema: [String: AnyCodable]?
    public let serverName: String

    public init(name: String, description: String? = nil, inputSchema: [String: AnyCodable]? = nil, serverName: String) {
        self.name = name
        self.description = description
        self.inputSchema = inputSchema
        self.serverName = serverName
    }

    public var namespacedName: String {
        "\(serverName)__\(name)"
    }

    public func toOpenAITool() -> MCPToolInfo {
        MCPToolInfo(
            name: namespacedName,
            description: description ?? "",
            server: serverName,
            parameters: inputSchema
        )
    }
}

public struct MCPToolResult: Codable, Sendable {
    public let content: String
    public let isError: Bool
    public let errorMessage: String?

    public init(content: String, isError: Bool = false, errorMessage: String? = nil) {
        self.content = content
        self.isError = isError
        self.errorMessage = errorMessage
    }
}

public enum MCPServerState: String, Codable, Sendable {
    case disconnected
    case connecting
    case connected
    case error
}

public struct MCPServerStatus: Codable, Sendable {
    public let name: String
    public let state: MCPServerState
    public let transport: String
    public let toolsCount: Int
    public let error: String?

    public init(name: String, state: MCPServerState, transport: String, toolsCount: Int = 0, error: String? = nil) {
        self.name = name
        self.state = state
        self.transport = transport
        self.toolsCount = toolsCount
        self.error = error
    }
}

public struct MCPToolInfo: Codable, Sendable {
    public let name: String
    public let description: String
    public let server: String
    public let parameters: [String: AnyCodable]?
}

public struct MCPToolsResponse: Codable, Sendable {
    public let tools: [MCPToolInfo]
    public let count: Int
}

public struct MCPServersResponse: Codable, Sendable {
    public let servers: [MCPServerStatus]
}

public struct MCPExecuteRequest: Codable, Sendable {
    public let toolName: String
    public let arguments: [String: AnyCodable]

    private enum CodingKeys: String, CodingKey {
        case toolName = "tool_name"
        case arguments
    }
}

public struct MCPExecuteResponse: Codable, Sendable {
    public let toolName: String
    public let content: String
    public let isError: Bool
    public let errorMessage: String?

    private enum CodingKeys: String, CodingKey {
        case toolName = "tool_name"
        case content
        case isError = "is_error"
        case errorMessage = "error_message"
    }
}
