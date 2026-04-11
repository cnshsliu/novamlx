import Foundation
import NovaMLXCore
import NovaMLXUtils

private struct SendableDict: @unchecked Sendable {
    let dictionary: [String: Any]
    init(_ dict: [String: Any]) { self.dictionary = dict }
}

public final class MCPClient: @unchecked Sendable {
    public let serverName: String
    public let config: MCPServerConfig
    public private(set) var state: MCPServerState
    public private(set) var tools: [MCPTool]

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var readBuffer: Data
    private let lock = NovaMLXLock()
    private var requestId: Int
    private let urlSession: URLSession

    public init(serverName: String, config: MCPServerConfig) {
        self.serverName = serverName
        self.config = config
        self.state = .disconnected
        self.tools = []
        self.readBuffer = Data()
        self.requestId = 0
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = config.timeout ?? 30.0
        self.urlSession = URLSession(configuration: sessionConfig)
    }

    public func connect() async throws {
        lock.withLock { state = .connecting }

        switch config.transport {
        case .stdio:
            try connectStdio()
        case .streamableHTTP, .sse:
            try connectHTTP()
        }

        try await initialize()
        let discoveredTools = try await listTools()
        lock.withLock {
            self.tools = discoveredTools
            self.state = .connected
        }
        NovaMLXLog.info("MCP client '\(serverName)' connected with \(discoveredTools.count) tools via \(config.transport.rawValue)")
    }

    public func disconnect() {
        lock.withLock {
            process?.terminate()
            process = nil
            stdinPipe = nil
            stdoutPipe = nil
            stderrPipe = nil
            state = .disconnected
            tools = []
            readBuffer = Data()
        }
    }

    public func execute(toolName: String, arguments: [String: Any]) async throws -> MCPToolResult {
        let timeout = config.timeout ?? 30.0
        let argsData = try JSONSerialization.data(withJSONObject: arguments)
        let argsStr = String(data: argsData, encoding: .utf8) ?? "{}"

        let params: [String: Any] = [
            "name": toolName,
            "arguments": argsStr
        ]

        let result = try await sendRequest(method: "tools/call", params: params, timeout: timeout)

        guard let dict = result as? [String: Any],
              let content = dict["content"] as? [[String: Any]] else {
            return MCPToolResult(content: String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8) ?? "{}")
        }

        let isError = dict["isError"] as? Bool ?? false
        let textParts = content.compactMap { item -> String? in
            guard item["type"] as? String == "text" else { return nil }
            return item["text"] as? String
        }

        return MCPToolResult(
            content: textParts.joined(),
            isError: isError,
            errorMessage: isError ? textParts.joined() : nil
        )
    }

    private func connectStdio() throws {
        guard let command = config.command else {
            throw NovaMLXError.configurationError("MCP stdio transport requires 'command'")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: command)
        if let args = config.args {
            process.arguments = args
        }

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        try process.run()

        lock.withLock {
            self.process = process
            self.stdinPipe = stdinPipe
            self.stdoutPipe = stdoutPipe
            self.stderrPipe = stderrPipe
        }
    }

    private func initialize() async throws {
        let params: [String: Any] = [
            "protocolVersion": "2024-11-05",
            "capabilities": [:],
            "clientInfo": ["name": "NovaMLX", "version": "1.0.0"]
        ]
        _ = try await sendRequest(method: "initialize", params: params, timeout: config.timeout ?? 30.0)
        try sendNotification(method: "notifications/initialized")
    }

    private func listTools() async throws -> [MCPTool] {
        let result = try await sendRequest(method: "tools/list", params: [:], timeout: config.timeout ?? 30.0)

        guard let dict = result as? [String: Any],
              let toolsArray = dict["tools"] as? [[String: Any]] else {
            return []
        }

        return toolsArray.compactMap { toolDict -> MCPTool? in
            guard let name = toolDict["name"] as? String else { return nil }
            let description = toolDict["description"] as? String
            let schema = toolDict["inputSchema"] as? [String: Any]
            return MCPTool(
                name: name,
                description: description,
                inputSchema: schema?.mapValues { AnyCodable($0) },
                serverName: serverName
            )
        }
    }

    private func connectHTTP() throws {
        guard config.url != nil else {
            throw NovaMLXError.configurationError("MCP HTTP transport requires 'url'")
        }
    }

    private func sendRequestHTTP(message: [String: Any], timeout: TimeInterval) async throws -> [String: Any] {
        guard let urlString = config.url else {
            throw NovaMLXError.configurationError("MCP HTTP transport requires 'url'")
        }
        guard let url = URL(string: urlString) else {
            throw NovaMLXError.configurationError("Invalid MCP server URL: \(urlString)")
        }

        let data = try JSONSerialization.data(withJSONObject: message)
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = data
        request.timeoutInterval = timeout

        if let headers = config.headers {
            for (key, value) in headers {
                request.setValue(value, forHTTPHeaderField: key)
            }
        }

        let (responseData, response) = try await urlSession.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NovaMLXError.inferenceFailed("Invalid MCP HTTP response")
        }

        let contentType = httpResponse.value(forHTTPHeaderField: "Content-Type") ?? ""

        if contentType.contains("text/event-stream") {
            return try parseSSEResponse(responseData, expectedId: message["id"] as? Int)
        }

        guard httpResponse.statusCode == 200 else {
            let body = String(data: responseData, encoding: .utf8) ?? ""
            throw NovaMLXError.inferenceFailed("MCP HTTP error \(httpResponse.statusCode): \(body)")
        }

        guard let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any] else {
            throw NovaMLXError.inferenceFailed("Invalid JSON in MCP HTTP response")
        }

        if let error = json["error"] as? [String: Any] {
            let msg = error["message"] as? String ?? "Unknown MCP error"
            throw NovaMLXError.inferenceFailed("MCP error: \(msg)")
        }

        return json
    }

    private func parseSSEResponse(_ data: Data, expectedId: Int?) throws -> [String: Any] {
        guard let text = String(data: data, encoding: .utf8) else {
            throw NovaMLXError.inferenceFailed("Invalid SSE response encoding")
        }

        for line in text.components(separatedBy: "\n\n") {
            var eventData = ""
            for eventLine in line.components(separatedBy: "\n") {
                if eventLine.hasPrefix("data: ") {
                    eventData += String(eventLine.dropFirst(6))
                }
            }
            if !eventData.isEmpty,
               let eventJSON = eventData.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: eventJSON) as? [String: Any] {
                if let id = expectedId, json["id"] as? Int == id {
                    return json
                } else if expectedId == nil {
                    return json
                }
            }
        }

        throw NovaMLXError.inferenceFailed("No matching SSE event found for request")
    }

    private func sendRequest(method: String, params: [String: Any], timeout: TimeInterval) async throws -> Any {
        var currentId: Int = 0
        lock.withLock {
            requestId += 1
            currentId = requestId
        }

        var message: [String: Any] = [
            "jsonrpc": "2.0",
            "id": currentId,
            "method": method
        ]
        if !params.isEmpty {
            message["params"] = params
        }

        let response: [String: Any]

        switch config.transport {
        case .streamableHTTP, .sse:
            response = try await sendRequestHTTP(message: message, timeout: timeout)
        case .stdio:
            let data = try JSONSerialization.data(withJSONObject: message)
            let header = "Content-Length: \(data.count)\r\n\r\n"

            guard let headerData = header.data(using: .utf8),
                  let pipe = lock.withLock({ stdinPipe }) else {
                throw NovaMLXError.configurationError("MCP client not connected")
            }

            pipe.fileHandleForWriting.write(headerData)
            pipe.fileHandleForWriting.write(data)

            response = try await readResponse(id: currentId, timeout: timeout)
        }

        if let error = response["error"] as? [String: Any] {
            let msg = error["message"] as? String ?? "Unknown MCP error"
            throw NovaMLXError.inferenceFailed("MCP error: \(msg)")
        }

        return response["result"] ?? [:]
    }

    private func sendNotification(method: String) throws {
        let message: [String: Any] = [
            "jsonrpc": "2.0",
            "method": method
        ]

        switch config.transport {
        case .streamableHTTP, .sse:
            guard let urlString = config.url, let url = URL(string: urlString) else { return }
            let data = try JSONSerialization.data(withJSONObject: message)
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = data
            if let headers = config.headers {
                for (key, value) in headers {
                    request.setValue(value, forHTTPHeaderField: key)
                }
            }
            Task { _ = try? await urlSession.data(for: request) }
        case .stdio:
            let data = try JSONSerialization.data(withJSONObject: message)
            let header = "Content-Length: \(data.count)\r\n\r\n"
            guard let headerData = header.data(using: .utf8),
                  let pipe = lock.withLock({ stdinPipe }) else { return }
            pipe.fileHandleForWriting.write(headerData)
            pipe.fileHandleForWriting.write(data)
        }
    }

    private func readResponse(id expectedId: Int, timeout: TimeInterval) async throws -> [String: Any] {
        try await withThrowingTaskGroup(of: SendableDict.self) { group in
            let id = expectedId
            group.addTask {
                while true {
                    let message = try await self.readOneMessage()
                    if let msgId = message.dictionary["id"] as? Int, msgId == id {
                        return message
                    }
                }
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                throw NovaMLXError.inferenceFailed("MCP request timed out after \(timeout)s")
            }

            guard let firstResult = try await group.next() else {
                throw NovaMLXError.inferenceFailed("MCP request failed")
            }
            group.cancelAll()
            return firstResult.dictionary
        }
    }

    private func readOneMessage() async throws -> SendableDict {
        guard let pipe = lock.withLock({ stdoutPipe }) else {
            throw NovaMLXError.configurationError("MCP client not connected")
        }

        var contentLength: Int?
        while contentLength == nil {
            let byte = try pipe.fileHandleForReading.read(upToCount: 1)
            guard let byte, !byte.isEmpty else {
                throw NovaMLXError.inferenceFailed("MCP connection closed")
            }
            lock.withLock { readBuffer.append(byte) }

            let bufferStr = String(data: lock.withLock { readBuffer }, encoding: .utf8) ?? ""
            if let range = bufferStr.range(of: "\r\n\r\n") {
                let headerStr = String(bufferStr[..<range.lowerBound])
                for line in headerStr.components(separatedBy: "\r\n") {
                    if line.hasPrefix("Content-Length:") {
                        contentLength = Int(line.dropFirst(15).trimmingCharacters(in: .whitespaces))
                    }
                }
                let headerEnd = bufferStr[..<range.lowerBound].utf8.count + 4
                lock.withLock { readBuffer = Data(readBuffer[headerEnd...]) }
            }
        }

        guard let length = contentLength else {
            throw NovaMLXError.inferenceFailed("No Content-Length in MCP response")
        }

        while lock.withLock({ readBuffer.count }) < length {
            let chunk = try pipe.fileHandleForReading.read(upToCount: length - lock.withLock({ readBuffer.count }))
            guard let chunk, !chunk.isEmpty else {
                throw NovaMLXError.inferenceFailed("MCP connection closed while reading body")
            }
            lock.withLock { readBuffer.append(chunk) }
        }

        let bodyData = lock.withLock { Data(readBuffer[0..<length]) }
        lock.withLock { readBuffer = Data(readBuffer[length...]) }

        guard let json = try JSONSerialization.jsonObject(with: bodyData) as? [String: Any] else {
            throw NovaMLXError.inferenceFailed("Invalid JSON in MCP response")
        }
        return SendableDict(json)
    }
}
