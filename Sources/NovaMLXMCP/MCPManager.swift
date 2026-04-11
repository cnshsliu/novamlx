import Foundation
import NovaMLXCore
import NovaMLXUtils

public final class MCPManager: @unchecked Sendable {
    private var clients: [String: MCPClient]
    private var config: MCPConfig?
    private let lock = NovaMLXLock()

    public init() {
        self.clients = [:]
    }

    public func loadConfig(_ config: MCPConfig) async {
        self.config = config
        for (name, serverConfig) in config.servers {
            guard serverConfig.enabled != false else { continue }
            let client = MCPClient(serverName: name, config: serverConfig)
            lock.withLock { clients[name] = client }

            do {
                try await client.connect()
            } catch {
                NovaMLXLog.error("Failed to connect MCP server '\(name)': \(error)")
            }
        }
    }

    public func loadConfig(from url: URL) async throws {
        let config = try MCPConfig.load(from: url)
        await loadConfig(config)
    }

    public func disconnectAll() {
        lock.withLock {
            for (_, client) in clients {
                client.disconnect()
            }
            clients.removeAll()
        }
    }

    public func getAllTools() -> [MCPTool] {
        lock.withLock {
            clients.values.flatMap { $0.tools }
        }
    }

    public func getToolsForOpenAI() -> [[String: Any]] {
        getAllTools().map { tool in
            var params: [String: Any] = [:]
            if let schema = tool.inputSchema {
                params = schema.mapValues { $0.value }
            }
            return [
                "type": "function",
                "function": [
                    "name": tool.namespacedName,
                    "description": tool.description ?? "",
                    "parameters": params
                ] as [String: Any]
            ]
        }
    }

    public func getServerStatuses() -> [MCPServerStatus] {
        lock.withLock {
            clients.map { (name, client) in
                MCPServerStatus(
                    name: name,
                    state: client.state,
                    transport: client.config.transport.rawValue,
                    toolsCount: client.tools.count
                )
            }
        }
    }

    public func executeTool(namespacedName: String, arguments: [String: Any]) async throws -> MCPToolResult {
        let parts = namespacedName.components(separatedBy: "__")
        guard parts.count >= 2 else {
            throw NovaMLXError.apiError("Invalid tool name format: \(namespacedName). Expected 'server__tool'.")
        }
        let serverName = parts[0]
        let toolName = parts.dropFirst().joined(separator: "__")

        let client = lock.withLock { clients[serverName] }
        guard let client else {
            throw NovaMLXError.modelNotFound("MCP server '\(serverName)' not found")
        }

        return try await client.execute(toolName: toolName, arguments: arguments)
    }

    public var isConnected: Bool {
        lock.withLock { clients.values.contains { $0.state == .connected } }
    }

    public var connectedServerCount: Int {
        lock.withLock { clients.values.filter { $0.state == .connected }.count }
    }

    public var totalServerCount: Int {
        lock.withLock { clients.count }
    }
}
