import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXAPI

// MARK: - Helper: recursive AnyCodable → Any (mirrors APIServer.unwrapAnyCodable)

private func unwrapAnyCodable(_ ac: AnyCodable) -> Any {
    switch ac {
    case .string(let s): return s
    case .int(let i): return i
    case .double(let d): return d
    case .bool(let b): return b
    case .null: return NSNull()
    case .array(let a): return a.map { unwrapAnyCodable($0) }
    case .dictionary(let d): return d.mapValues { unwrapAnyCodable($0) }
    }
}

private func anyToAnyCodable(_ value: Any) -> AnyCodable {
    switch value {
    case let s as String: return .string(s)
    case let i as Int: return .int(i)
    case let d as Double: return .double(d)
    case let b as Bool: return .bool(b)
    case let a as [Any]: return .array(a.map { anyToAnyCodable($0) })
    case let d as [String: Any]: return .dictionary(d.mapValues { anyToAnyCodable($0) })
    default: return .string("\(value)")
    }
}

// MARK: - Realistic Tool Definitions

private let listFilesTool: [String: AnyCodable] = [
    "type": .string("function"),
    "function": .dictionary([
        "name": .string("list_files"),
        "description": .string("List files in a directory"),
        "parameters": .dictionary([
            "type": .string("object"),
            "properties": .dictionary([
                "path": .dictionary([
                    "type": .string("string"),
                    "description": .string("Directory path"),
                ]),
                "recursive": .dictionary([
                    "type": .string("boolean"),
                    "description": .string("Recursive listing"),
                ]),
            ]),
            "required": .array([.string("path")]),
        ]),
    ]),
]

private let readFileTool: [String: AnyCodable] = [
    "type": .string("function"),
    "function": .dictionary([
        "name": .string("read_file"),
        "description": .string("Read file contents"),
        "parameters": .dictionary([
            "type": .string("object"),
            "properties": .dictionary([
                "path": .dictionary(["type": .string("string"), "description": .string("File path")]),
                "offset": .dictionary(["type": .string("integer"), "description": .string("Start line")]),
                "limit": .dictionary(["type": .string("integer"), "description": .string("Max lines")]),
            ]),
            "required": .array([.string("path")]),
        ]),
    ]),
]

private let writeToolJSON = """
{"type":"function","function":{"name":"write_file","description":"Write content to a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"content":{"type":"string","description":"Content to write"}},"required":["path","content"]}}}
"""

private let execToolJSON = """
{"type":"function","function":{"name":"execute_command","description":"Execute a shell command","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Shell command"},"timeout":{"type":"integer","description":"Timeout seconds"}},"required":["command"]}}}
"""

@Suite("Tool Calling Tests")
struct ToolCallingTests {

    // MARK: - Core Types

    @Test("ToolCallResult encodes with all fields")
    func toolCallResultEncoding() throws {
        let tc = ToolCallResult(
            id: "call_abc123",
            functionName: "read_file",
            arguments: #"{"path": "README.md"}"#
        )
        let data = try JSONEncoder().encode(tc)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(json?["id"] as? String == "call_abc123")
        #expect(json?["functionName"] as? String == "read_file")
        let argsStr = json?["arguments"] as? String ?? ""
        let args = try JSONSerialization.jsonObject(with: Data(argsStr.utf8)) as? [String: Any]
        #expect(args?["path"] as? String == "README.md")
    }

    @Test("InferenceRequest carries tools field")
    func inferenceRequestCarriesTools() {
        let tools: [[String: Any]] = [
            ["type": "function", "function": ["name": "calc", "description": "Math"]],
        ]
        let req = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hi")],
            tools: tools
        )
        #expect(req.tools != nil)
        #expect(req.tools?.count == 1)
        #expect(req.tools?.first?["type"] as? String == "function")
    }

    @Test("InferenceRequest without tools defaults to nil")
    func inferenceRequestNoTools() {
        let req = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hi")]
        )
        #expect(req.tools == nil)
    }

    @Test("ChatMessage carries toolCalls and toolCallId")
    func chatMessageToolFields() {
        let tc = ToolCallResult(id: "call_1", functionName: "search", arguments: #"{"q": "test"}"#)
        let msg = ChatMessage(
            role: .assistant,
            content: nil,
            toolCalls: [tc]
        )
        #expect(msg.toolCalls?.count == 1)
        #expect(msg.toolCalls?.first?.functionName == "search")

        let toolMsg = ChatMessage(
            role: .tool,
            content: #"{"result": "found"}"#,
            name: "search",
            toolCallId: "call_1"
        )
        #expect(toolMsg.toolCallId == "call_1")
        #expect(toolMsg.name == "search")
    }

    @Test("InferenceResult carries toolCalls with finishReason")
    func inferenceResultToolCalls() {
        let tc = ToolCallResult(id: "call_x", functionName: "exec", arguments: #"{"cmd": "ls"}"#)
        let result = InferenceResult(
            id: UUID(),
            model: "test",
            text: "",
            toolCalls: [tc],
            tokensPerSecond: 50.0,
            promptTokens: 100,
            completionTokens: 20,
            finishReason: .toolCalls
        )
        #expect(result.toolCalls?.count == 1)
        #expect(result.finishReason == .toolCalls)
        #expect(result.toolCalls?.first?.id == "call_x")
    }

    // MARK: - IPC Serialization (CodableInferenceRequest)

    @Test("CodableInferenceRequest preserves tools through serialization")
    func codableRequestToolsRoundTrip() throws {
        let tools: [[String: Any]] = [
            [
                "type": "function",
                "function": [
                    "name": "calculator",
                    "description": "Calculate math",
                    "parameters": [
                        "type": "object",
                        "properties": ["expression": ["type": "string"]],
                        "required": ["expression"],
                    ],
                ],
            ],
        ]
        let original = InferenceRequest(
            model: "test-model",
            messages: [ChatMessage(role: .user, content: "2+2")],
            tools: tools,
            temperature: 0.7,
            maxTokens: 100
        )

        let codable = CodableInferenceRequest(from: original)
        #expect(codable.toolsJSON != nil)

        let decoded = codable.toInferenceRequest()
        #expect(decoded.tools != nil)
        #expect(decoded.tools?.count == 1)
        #expect(decoded.model == "test-model")
        #expect(decoded.temperature == 0.7)
        #expect(decoded.maxTokens == 100)

        // Verify nested structure preserved
        let tool = decoded.tools?.first
        #expect(tool?["type"] as? String == "function")
        let function = tool?["function"] as? [String: Any]
        #expect(function?["name"] as? String == "calculator")
        let params = function?["parameters"] as? [String: Any]
        #expect(params?["type"] as? String == "object")
    }

    @Test("CodableInferenceRequest handles nil tools")
    func codableRequestNilTools() {
        let original = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hello")]
        )
        let codable = CodableInferenceRequest(from: original)
        #expect(codable.toolsJSON == nil)
        let decoded = codable.toInferenceRequest()
        #expect(decoded.tools == nil)
    }

    @Test("CodableInferenceRequest handles empty tools array")
    func codableRequestEmptyTools() {
        let original = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hello")],
            tools: []
        )
        let codable = CodableInferenceRequest(from: original)
        #expect(codable.toolsJSON == nil) // empty array → nil
        let decoded = codable.toInferenceRequest()
        #expect(decoded.tools == nil)
    }

    @Test("CodableInferenceRequest preserves multiple tools")
    func codableRequestMultipleTools() throws {
        let tools: [[String: Any]] = [
            ["type": "function", "function": ["name": "tool_a"]],
            ["type": "function", "function": ["name": "tool_b"]],
            ["type": "function", "function": ["name": "tool_c"]],
        ]
        let original = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "go")],
            tools: tools
        )
        let decoded = CodableInferenceRequest(from: original).toInferenceRequest()
        #expect(decoded.tools?.count == 3)
    }

    @Test("CodableInferenceRequest preserves all other fields alongside tools")
    func codableRequestAllFields() throws {
        let tc = ToolCallResult(id: "c1", functionName: "f", arguments: "{}")
        let original = InferenceRequest(
            id: UUID(),
            model: "m",
            messages: [
                ChatMessage(role: .system, content: "sys"),
                ChatMessage(role: .user, content: "hi"),
                ChatMessage(role: .assistant, content: "hey", toolCalls: [tc]),
                ChatMessage(role: .tool, content: "result", name: "f", toolCallId: "c1"),
            ],
            tools: [["type": "function", "function": ["name": "f"]]],
            temperature: 0.3,
            maxTokens: 200,
            topP: 0.95,
            topK: 40,
            minP: 0.05,
            frequencyPenalty: 0.1,
            presencePenalty: 0.2,
            repetitionPenalty: 1.1,
            seed: 42,
            stream: true,
            stop: ["END"],
            sessionId: "sess-123",
            responseFormat: .jsonObject,
            jsonSchemaDef: ["type": "object"],
            regexPattern: "\\d+",
            gbnfGrammar: nil,
            thinkingBudget: 5000
        )

        let decoded = CodableInferenceRequest(from: original).toInferenceRequest()
        #expect(decoded.model == "m")
        #expect(decoded.messages.count == 4)
        #expect(decoded.tools?.count == 1)
        #expect(decoded.temperature == 0.3)
        #expect(decoded.maxTokens == 200)
        #expect(decoded.topP == 0.95)
        #expect(decoded.topK == 40)
        #expect(decoded.minP == 0.05)
        #expect(decoded.seed == 42)
        #expect(decoded.stream == true)
        #expect(decoded.stop == ["END"])
        #expect(decoded.sessionId == "sess-123")
        #expect(decoded.responseFormat == .jsonObject)
        #expect(decoded.thinkingBudget == 5000)
    }

    // MARK: - ModelSettings.applySamplingOverrides preserves tools

    @Test("applySamplingOverrides preserves tools")
    func settingsOverridesPreserveTools() {
        let tools: [[String: Any]] = [
            ["type": "function", "function": ["name": "calc"]],
        ]
        let request = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hi")],
            tools: tools,
            temperature: nil // let settings override
        )
        let settings = ModelSettings(temperature: 0.5)
        let result = settings.applySamplingOverrides(to: request)

        #expect(result.tools != nil)
        #expect(result.tools?.count == 1)
        #expect(result.temperature == 0.5) // overridden
        #expect(result.model == "test") // preserved
    }

    @Test("applySamplingOverrides handles nil tools")
    func settingsOverridesNilTools() {
        let request = InferenceRequest(
            model: "test",
            messages: [ChatMessage(role: .user, content: "hi")]
        )
        let settings = ModelSettings()
        let result = settings.applySamplingOverrides(to: request)
        #expect(result.tools == nil)
    }

    // MARK: - OpenAI API Types

    @Test("OpenAI request decodes tools from JSON")
    func openAIRequestDecodesTools() throws {
        let json = """
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Calculate math",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {"type": "string"}
                            },
                            "required": ["expression"]
                        }
                    }
                }
            ],
            "max_tokens": 300
        }
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIRequest.self, from: data)

        #expect(req.tools != nil)
        #expect(req.tools?.count == 1)

        let tool = req.tools?.first
        #expect(tool?["type"] == .string("function"))
        let function = tool.flatMap { ac -> [String: AnyCodable]? in
            if case .dictionary(let d) = ac["function"] { return d }
            return nil
        }
        #expect(function?["name"] == .string("calculator"))
        #expect(function?["description"] == .string("Calculate math"))
    }

    @Test("OpenAI request with tool_choice auto")
    func openAIRequestToolChoiceAuto() throws {
        let json = """
        {
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": "auto"
        }
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(OpenAIRequest.self, from: data)
        #expect(req.toolChoice == .string("auto"))
    }

    @Test("OpenAI response with tool_calls encodes correctly")
    func openAIResponseToolCalls() throws {
        let response = OpenAIResponse(
            id: "chatcmpl-1",
            model: "test",
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(
                        role: "assistant",
                        content: "Let me look that up.",
                        toolCalls: [
                            OpenAIToolCall(
                                id: "call_abc",
                                function: OpenAIFunctionCall(
                                    name: "list_files",
                                    arguments: #"{"path": "/src"}"#
                                )
                            ),
                        ]
                    ),
                    finishReason: "tool_calls"
                ),
            ],
            usage: OpenAIUsage(promptTokens: 100, completionTokens: 20)
        )

        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        let choice = (json?["choices"] as? [[String: Any]])?.first
        #expect(choice?["finish_reason"] as? String == "tool_calls")

        let msg = choice?["message"] as? [String: Any]
        let toolCalls = msg?["tool_calls"] as? [[String: Any]]
        #expect(toolCalls?.count == 1)
        #expect(toolCalls?.first?["id"] as? String == "call_abc")

        let function = toolCalls?.first?["function"] as? [String: Any]
        #expect(function?["name"] as? String == "list_files")
        let argsStr = function?["arguments"] as? String ?? ""
        let args = try JSONSerialization.jsonObject(with: Data(argsStr.utf8)) as? [String: Any]
        #expect(args?["path"] as? String == "/src")
    }

    @Test("OpenAI chat message with tool_call_id decodes from JSON")
    func openAIToolRoleMessage() throws {
        let json = """
        {
            "role": "tool",
            "content": "file1.swift\\nfile2.swift",
            "tool_call_id": "call_abc",
            "name": "list_files"
        }
        """
        let data = json.data(using: .utf8)!
        let msg = try JSONDecoder().decode(OpenAIChatMessage.self, from: data)
        #expect(msg.role == "tool")
        #expect(msg.toolCallId == "call_abc")
        #expect(msg.name == "list_files")
    }

    @Test("OpenAI stream chunk with tool_calls delta")
    func openAIStreamChunkToolCalls() throws {
        let chunk = OpenAIStreamChunk(
            id: "chatcmpl-1",
            model: "test",
            choices: [
                OpenAIStreamChoice(
                    index: 0,
                    delta: OpenAIDelta(
                        toolCalls: [
                            OpenAIToolCallDelta(
                                index: 0,
                                id: "call_xyz",
                                type: "function",
                                function: OpenAIFunctionCallDelta(
                                    name: "read_file",
                                    arguments: #"{"path":"test.swift"}"#
                                )
                            ),
                        ]
                    ),
                    finishReason: nil
                ),
            ]
        )
        let data = try JSONEncoder().encode(chunk)
        let decoded = try JSONDecoder().decode(OpenAIStreamChunk.self, from: data)
        let tc = decoded.choices.first?.delta.toolCalls?.first
        #expect(tc?.id == "call_xyz")
        #expect(tc?.function?.name == "read_file")
        let args = tc?.function?.arguments ?? ""
        let argsObj = try JSONSerialization.jsonObject(with: Data(args.utf8)) as? [String: Any]
        #expect(argsObj?["path"] as? String == "test.swift")
    }

    // MARK: - Anthropic API Types

    @Test("Anthropic request decodes tools from JSON")
    func anthropicRequestDecodesTools() throws {
        let json = """
        {
            "model": "test",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": "Search for files"}],
            "tools": [
                {
                    "name": "execute_command",
                    "description": "Execute a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command"},
                            "timeout": {"type": "integer", "description": "Timeout"}
                        },
                        "required": ["command"]
                    }
                }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let req = try JSONDecoder().decode(AnthropicRequest.self, from: data)

        #expect(req.tools?.count == 1)
        #expect(req.tools?.first?.name == "execute_command")
        #expect(req.tools?.first?.description == "Execute a shell command")

        // Verify input_schema preserves nested structure
        let schema = req.tools?.first?.inputSchema
        if case .dictionary(let props) = schema {
            let properties = props["properties"]
            if case .dictionary(let propDict) = properties {
                let cmd = propDict["command"]
                if case .dictionary(let cmdDict) = cmd {
                    #expect(cmdDict["type"] == .string("string"))
                }
            }
        }
    }

    @Test("Anthropic response with tool_use content block")
    func anthropicResponseToolUse() throws {
        let response = AnthropicResponse(
            id: "msg-1",
            model: "test",
            content: [
                AnthropicContentBlock(text: "I'll run that command."),
                AnthropicContentBlock(
                    type: "tool_use",
                    id: "call_123",
                    name: "execute_command",
                    input: .dictionary([
                        "command": .string("find /src -name '*.swift' | wc -l"),
                        "timeout": .int(30),
                    ])
                ),
            ],
            stopReason: "tool_use",
            usage: AnthropicUsage(inputTokens: 200, outputTokens: 50)
        )

        let data = try JSONEncoder().encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        #expect(json?["stop_reason"] as? String == "tool_use")

        let content = json?["content"] as? [[String: Any]]
        #expect(content?.count == 2)

        let toolBlock = content?.last
        #expect(toolBlock?["type"] as? String == "tool_use")
        #expect(toolBlock?["id"] as? String == "call_123")
        #expect(toolBlock?["name"] as? String == "execute_command")

        let input = toolBlock?["input"] as? [String: Any]
        #expect(input?["command"] as? String == "find /src -name '*.swift' | wc -l")
        #expect(input?["timeout"] as? Int == 30)
    }

    // MARK: - AnyCodable Recursive Conversion

    @Test("unwrapAnyCodable preserves nested tool schema structure")
    func unwrapAnyCodableNested() {
        let schema: AnyCodable = .dictionary([
            "type": .string("object"),
            "properties": .dictionary([
                "path": .dictionary([
                    "type": .string("string"),
                    "description": .string("File path"),
                ]),
                "recursive": .dictionary([
                    "type": .string("boolean"),
                ]),
            ]),
            "required": .array([.string("path")]),
        ])

        let result = unwrapAnyCodable(schema) as? [String: Any]
        #expect(result?["type"] as? String == "object")

        let props = result?["properties"] as? [String: Any]
        #expect(props?["path"] is [String: Any])

        let pathDef = props?["path"] as? [String: Any]
        #expect(pathDef?["type"] as? String == "string")

        let required = result?["required"] as? [Any]
        #expect(required?.count == 1)
    }

    @Test("unwrapAnyCodable handles full tool spec with nested function")
    func unwrapAnyCodableFullTool() {
        let tool: [String: AnyCodable] = [
            "type": .string("function"),
            "function": .dictionary([
                "name": .string("write_file"),
                "description": .string("Write to file"),
                "parameters": .dictionary([
                    "type": .string("object"),
                    "properties": .dictionary([
                        "path": .dictionary(["type": .string("string")]),
                        "content": .dictionary(["type": .string("string")]),
                    ]),
                    "required": .array([.string("path"), .string("content")]),
                ]),
            ]),
        ]

        let unwrapped = tool.mapValues { unwrapAnyCodable($0) }
        #expect(unwrapped["type"] as? String == "function")

        let function = unwrapped["function"] as? [String: Any]
        #expect(function?["name"] as? String == "write_file")

        let params = function?["parameters"] as? [String: Any]
        #expect(params?["type"] as? String == "object")

        let properties = params?["properties"] as? [String: Any]
        let pathProp = properties?["path"] as? [String: Any]
        #expect(pathProp?["type"] as? String == "string")
    }

    @Test("anyToAnyCodable round-trips nested dictionary")
    func anyToAnyCodableRoundTrip() {
        let original: [String: Any] = [
            "command": "grep -r 'pattern' /src",
            "timeout": 30,
            "options": ["verbose": true, "count": 5] as [String: Any],
        ]

        let codable = original.mapValues { anyToAnyCodable($0) }
        let back = codable.mapValues { unwrapAnyCodable($0) }

        #expect(back["command"] as? String == "grep -r 'pattern' /src")
        #expect(back["timeout"] as? Int == 30)

        let opts = back["options"] as? [String: Any]
        #expect(opts?["verbose"] as? Bool == true)
        #expect(opts?["count"] as? Int == 5)
    }

    // MARK: - Full Pipeline: API JSON → InferenceRequest

    @Test("Full pipeline: OpenAI JSON → InferenceRequest with tools")
    func fullPipelineOpenAI() throws {
        let json = """
        {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "List files in /src"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": "List directory contents",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Directory path"},
                                "recursive": {"type": "boolean"}
                            },
                            "required": ["path"]
                        }
                    }
                }
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }
        """
        let data = json.data(using: .utf8)!
        let openAIReq = try JSONDecoder().decode(OpenAIRequest.self, from: data)

        // Simulate what APIServer does
        let tools: [[String: Any]]? = openAIReq.tools?.map { tool in
            tool.mapValues { unwrapAnyCodable($0) }
        }

        let request = InferenceRequest(
            model: openAIReq.model,
            messages: openAIReq.messages.map { msg in
                ChatMessage(role: .user, content: msg.content?.textValue)
            },
            tools: tools,
            temperature: openAIReq.temperature,
            maxTokens: openAIReq.maxTokens
        )

        #expect(request.tools != nil)
        #expect(request.tools?.count == 1)
        #expect(request.model == "test-model")
        #expect(request.temperature == 0.7)

        // Verify nested structure survived
        guard let tool: [String: Any] = request.tools?.first else {
            Issue.record("No tool found"); return
        }
        #expect(tool["type"] as? String == "function")
        let funcDict = tool["function"] as? [String: Any]
        #expect(funcDict?["name"] as? String == "list_files")
        let params = funcDict?["parameters"] as? [String: Any]
        #expect(params?["type"] as? String == "object")
        let props = params?["properties"] as? [String: Any]
        #expect(props?["path"] is [String: Any])
    }

    @Test("Full pipeline: Anthropic JSON → InferenceRequest with tools")
    func fullPipelineAnthropic() throws {
        let json = """
        {
            "model": "test-model",
            "max_tokens": 400,
            "system": "You are a coding assistant.",
            "messages": [
                {"role": "user", "content": "Run grep"}
            ],
            "tools": [
                {
                    "name": "execute_command",
                    "description": "Execute a shell command",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"}
                        },
                        "required": ["command"]
                    }
                }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let anthropicReq = try JSONDecoder().decode(AnthropicRequest.self, from: data)

        // Simulate what APIServer does for Anthropic tools
        let tools: [[String: Any]]? = anthropicReq.tools?.map { tool in
            var dict: [String: Any] = ["name": tool.name]
            if let desc = tool.description { dict["description"] = desc }
            dict["type"] = "function"
            dict["function"] = [
                "name": tool.name,
                "description": tool.description ?? "",
                "parameters": unwrapAnyCodable(tool.inputSchema),
            ] as [String: Any]
            return dict
        }

        let request = InferenceRequest(
            model: anthropicReq.model,
            messages: anthropicReq.messages.map { msg in
                ChatMessage(role: .user, content: msg.textContent)
            },
            tools: tools,
            maxTokens: anthropicReq.maxTokens
        )

        #expect(request.tools != nil)
        #expect(request.tools?.count == 1)

        let tool = request.tools?.first
        let function = tool?["function"] as? [String: Any]
        #expect(function?["name"] as? String == "execute_command")

        let params = function?["parameters"] as? [String: Any]
        #expect(params?["type"] as? String == "object")
        let props = params?["properties"] as? [String: Any]
        let cmdProp = props?["command"] as? [String: Any]
        #expect(cmdProp?["type"] as? String == "string")
    }

    // MARK: - Multi-tool Pipeline

    @Test("Full pipeline: 4 real tools survive entire chain")
    func fullPipelineMultipleTools() throws {
        let json = """
        {
            "model": "test",
            "messages": [{"role": "user", "content": "list and read"}],
            "tools": [
                \(listFilesToolJSON),
                \(readFileToolJSON),
                \(writeToolJSON),
                \(execToolJSON)
            ],
            "max_tokens": 500
        }
        """
        let data = json.data(using: .utf8)!
        let openAIReq = try JSONDecoder().decode(OpenAIRequest.self, from: data)

        // APIServer conversion
        let apiTools = openAIReq.tools?.map { tool in
            tool.mapValues { unwrapAnyCodable($0) }
        }

        // InferenceService rebuild (settings override)
        let request = InferenceRequest(
            model: openAIReq.model,
            messages: openAIReq.messages.map { ChatMessage(role: .user, content: $0.content?.textValue) },
            tools: apiTools,
            maxTokens: openAIReq.maxTokens
        )
        let settings = ModelSettings()
        let overridden = settings.applySamplingOverrides(to: request)
        #expect(overridden.tools?.count == 4)

        // IPC serialization round-trip
        let codable = CodableInferenceRequest(from: overridden)
        #expect(codable.toolsJSON != nil)

        let decoded = codable.toInferenceRequest()
        #expect(decoded.tools?.count == 4)

        // Verify each tool name survived
        let names = decoded.tools?.compactMap { tool -> String? in
            (tool["function"] as? [String: Any])?["name"] as? String
        }
        #expect(names?.contains("list_files") == true)
        #expect(names?.contains("read_file") == true)
        #expect(names?.contains("write_file") == true)
        #expect(names?.contains("execute_command") == true)

        // Verify deep nesting survived for list_files tool
        guard let listTool: [String: Any] = decoded.tools?.first(where: {
            (($0["function"] as? [String: Any])?["name"] as? String) == "list_files"
        }) else {
            Issue.record("list_files tool not found"); return
        }
        let funcDict = listTool["function"] as? [String: Any]
        let params = funcDict?["parameters"] as? [String: Any]
        let props = params?["properties"] as? [String: Any]
        let pathProp = props?["path"] as? [String: Any]
        #expect(pathProp?["type"] as? String == "string")
    }

    @Test("Multi-turn: tool_calls in assistant message + tool result")
    func multiTurnToolConversation() throws {
        // Round 1 response
        let response = OpenAIResponse(
            id: "chatcmpl-1",
            model: "test",
            choices: [
                OpenAIChoice(
                    index: 0,
                    message: OpenAIChatMessage(
                        role: "assistant",
                        content: "I'll list the files.",
                        toolCalls: [
                            OpenAIToolCall(
                                id: "call_abc",
                                function: OpenAIFunctionCall(
                                    name: "list_files",
                                    arguments: #"{"path": "."}"#
                                )
                            ),
                        ]
                    ),
                    finishReason: "tool_calls"
                ),
            ],
            usage: OpenAIUsage(promptTokens: 100, completionTokens: 15)
        )

        // Build messages for round 2
        let round2Messages: [OpenAIChatMessage] = [
            OpenAIChatMessage(role: "user", content: "List files"),
            OpenAIChatMessage(
                role: "assistant",
                content: response.choices[0].message.content,
                toolCalls: response.choices[0].message.toolCalls
            ),
            OpenAIChatMessage(
                role: "tool",
                content: "file1.swift\nfile2.swift\nREADME.md",
                toolCallId: "call_abc",
                name: "list_files"
            ),
        ]

        // Encode round 2 messages
        let round2Data = try JSONEncoder().encode(round2Messages)
        let decoded = try JSONDecoder().decode([OpenAIChatMessage].self, from: round2Data)

        #expect(decoded.count == 3)
        #expect(decoded[1].role == "assistant")
        #expect(decoded[1].toolCalls?.count == 1)
        #expect(decoded[1].toolCalls?.first?.id == "call_abc")
        #expect(decoded[2].role == "tool")
        #expect(decoded[2].toolCallId == "call_abc")
        #expect(decoded[2].name == "list_files")
    }
}

// MARK: - Inline JSON helpers for multi-tool test

private let listFilesToolJSON = """
{"type":"function","function":{"name":"list_files","description":"List files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path"},"recursive":{"type":"boolean","description":"Recursive listing"}},"required":["path"]}}}
"""

private let readFileToolJSON = """
{"type":"function","function":{"name":"read_file","description":"Read file contents","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File path"},"offset":{"type":"integer","description":"Start line"},"limit":{"type":"integer","description":"Max lines"}},"required":["path"]}}}
"""
