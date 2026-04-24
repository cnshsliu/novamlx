import Testing
import Foundation
import NovaMLXCore
@testable import NovaMLXUtils

// ────────────────────────────────────────────────────────────
// Agent Format Simulation Tests
//
// Covers: Claude Code, Open Code, Open Claw, Hermes agents
// Validates message structures, tool calls, streaming, and
// control token / thinking tag behavior in agent scenarios.
// ────────────────────────────────────────────────────────────

// MARK: - Test Helpers

/// Simulated streaming token accumulator for agent scenarios
struct AgentStreamAccumulator {
    var parser = ThinkingParser()
    var thinkingText = ""
    var responseText = ""
    var toolCalls: [ParsedToolCall] = []

    mutating func feed(_ token: String) {
        let parsed = parser.feed(token)
        switch parsed.type {
        case .thinking: thinkingText += parsed.text
        case .content: responseText += parsed.text
        }
    }

    mutating func finalize() {
        let result = parser.finalize()
        if !result.thinking.isEmpty && thinkingText.isEmpty { thinkingText = result.thinking }
        if !result.response.isEmpty && responseText.isEmpty { responseText = result.response }
        // Parse tool calls from response
        let tr = ToolCallParser.parse(responseText)
        if let tcs = tr.toolCalls {
            toolCalls = tcs.map { ParsedToolCall(name: $0.function.name, arguments: $0.function.arguments) }
            if !tr.content.isEmpty { responseText = tr.content }
        }
    }
}

struct ParsedToolCall: Equatable {
    let name: String
    let arguments: String

    static func == (lhs: ParsedToolCall, rhs: ParsedToolCall) -> Bool {
        lhs.name == rhs.name
    }
}

// ────────────────────────────────────────────────────────────
// Suite 1: Claude Code Agent Simulation
// ────────────────────────────────────────────────────────────

@Suite("Claude Code Agent Simulation")
struct ClaudeCodeAgentTests {

    @Test("System prompt with cache_control")
    func systemPromptWithCacheControl() {
        let systemBlock: [String: Any] = [
            "type": "text",
            "text": "You are an interactive CLI tool for software engineering.",
            "cache_control": ["type": "ephemeral"]
        ]
        let systemArray = [systemBlock]

        // Verify structure
        #expect(systemArray.count == 1)
        let first = systemArray[0]
        #expect(first["type"] as? String == "text")
        #expect((first["text"] as? String)?.contains("CLI") == true)
        let cc = first["cache_control"] as? [String: Any]
        #expect(cc?["type"] as? String == "ephemeral")
    }

    @Test("Tool definitions match Claude Code format")
    func toolDefinitions() {
        let tools: [[String: Any]] = [
            ["name": "Read", "description": "Reads a file from disk",
             "input_schema": ["type": "object", "properties": ["file_path": ["type": "string"]], "required": ["file_path"]]],
            ["name": "Bash", "description": "Runs a shell command",
             "input_schema": ["type": "object", "properties": ["command": ["type": "string"]], "required": ["command"]]],
            ["name": "Grep", "description": "Searches for patterns",
             "input_schema": ["type": "object", "properties": ["pattern": ["type": "string"]], "required": ["pattern"]]],
        ]

        #expect(tools.count == 3)
        for tool in tools {
            #expect(tool["name"] != nil)
            #expect(tool["description"] != nil)
            #expect(tool["input_schema"] != nil)
        }
    }

    @Test("Multi-turn conversation with tool_use and tool_result")
    func multiTurnToolConversation() {
        var messages: [[String: Any]] = []

        // Turn 1: User request
        messages.append(["role": "user", "content": "Read Package.swift"])

        // Turn 2: Assistant tool use
        messages.append(["role": "assistant", "content": [
            ["type": "text", "text": "Let me read that file."],
            ["type": "tool_use", "id": "toolu_001", "name": "Read",
             "input": ["file_path": "/Users/dev/Package.swift"]]
        ]])

        // Turn 3: Tool result
        messages.append(["role": "user", "content": [
            ["type": "tool_result", "tool_use_id": "toolu_001",
             "content": "import PackageDescription\nlet package = Package(name: \"MyApp\")"]
        ]])

        // Turn 4: Assistant response
        messages.append(["role": "assistant", "content": [
            ["type": "text", "text": "The project is named MyApp."]
        ]])

        #expect(messages.count == 4)

        // Verify tool_use structure
        let assistantMsg = messages[1]
        let content = assistantMsg["content"] as? [[String: Any]]
        #expect(content?.count == 2)
        #expect(content?[0]["type"] as? String == "text")
        #expect(content?[1]["type"] as? String == "tool_use")
        #expect(content?[1]["id"] as? String == "toolu_001")

        // Verify tool_result structure
        let resultMsg = messages[2]
        let resultContent = resultMsg["content"] as? [[String: Any]]
        #expect(resultContent?[0]["type"] as? String == "tool_result")
        #expect(resultContent?[0]["tool_use_id"] as? String == "toolu_001")
    }

    @Test("Claude Code thinking+response streaming simulation")
    func thinkingResponseStreaming() {
        var acc = AgentStreamAccumulator()

        // Simulate reasoning model with implicit open tag
        // (<think> injected by chat template, only </think> in output)
        let tokens = [
            "Let me analyze this code.",
            "The file imports Foundation.",
            "It defines a Package structure.",
            "</think>",
            "The project depends on swift-nio 2.0.0.",
            " No other dependencies detected."
        ]

        for token in tokens { acc.feed(token) }
        acc.finalize()

        #expect(acc.thinkingText.contains("Let me analyze"))
        #expect(acc.thinkingText.contains("Foundation"))
        #expect(acc.responseText.contains("swift-nio 2.0.0"))
        #expect(!acc.responseText.contains("</think>"))
        #expect(!acc.thinkingText.contains("</think>"))
    }

    @Test("Claude Code tool call parsing from streaming output")
    func toolCallParsing() {
        var acc = AgentStreamAccumulator()

        let tokens = [
            "I'll search for that.",
            " <tool>{\"name\": \"search\", \"arguments\": ",
            "{\"query\": \"swift package manager\"}}</tool>"
        ]

        for token in tokens { acc.feed(token) }
        acc.finalize()

        #expect(acc.responseText.contains("I'll search"))
        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "search")
    }

    @Test("Claude Code long context: thinking + tool use + response")
    func longContextThinkingToolResponse() {
        var acc = AgentStreamAccumulator()

        // Models like Qwen3.6 (thinking) used as Claude Code agent
        let streamSequence = [
            "Looking at the code...",
            "The function signature is wrong.",
            "</think>",
            "I found the issue. Let me fix it.",
            " <tool>{\"name\": \"Edit\", \"arguments\": ",
            "{\"file_path\": \"src/main.swift\", \"old\": \"func foo()\", \"new\": \"func foo(x: Int)\"}}</tool>",
            "The edit has been applied."
        ]

        for token in streamSequence { acc.feed(token) }
        acc.finalize()

        #expect(acc.thinkingText.contains("function signature"))
        #expect(acc.responseText.contains("found the issue"))
        #expect(acc.toolCalls.count >= 1)
    }

    @Test("Claude Code response without thinking (non-reasoning model)")
    func nonThinkingModelResponse() {
        var acc = AgentStreamAccumulator()

        // Non-reasoning models just produce content
        acc.feed("Here is the file content:\n")
        acc.feed("import Foundation\n")
        acc.feed("struct Config { let port = 8080 }")
        acc.finalize()

        #expect(acc.thinkingText.isEmpty)
        #expect(acc.responseText.contains("struct Config"))
        #expect(acc.responseText.contains("port = 8080"))
    }
}

// ────────────────────────────────────────────────────────────
// Suite 2: Open Code Agent Simulation
// ────────────────────────────────────────────────────────────

@Suite("Open Code Agent Simulation")
struct OpenCodeAgentTests {

    @Test("Open Code system prompt structure")
    func openCodeSystemPrompt() {
        // Open Code uses a simpler system prompt than Claude Code
        let systemPrompt = """
        You are OpenCode, an AI coding assistant.
        You have access to the following tools:
        - Read(file_path): Read file contents
        - Write(file_path, content): Write to a file
        - Bash(command): Execute shell commands
        - Glob(pattern): Find files matching pattern
        - Grep(pattern, path): Search within files

        Follow these rules:
        1. Always read before editing
        2. Keep edits minimal
        3. Explain changes before making them
        """

        #expect(systemPrompt.contains("OpenCode"))
        #expect(systemPrompt.contains("Read(file_path)"))
        #expect(systemPrompt.contains("Always read before editing"))
    }

    @Test("Open Code tool call format — XML style")
    func openCodeXMLToolCalls() {
        var acc = AgentStreamAccumulator()

        // Open Code uses <tool> XML format for tool calls
        acc.feed("Reading the config...\n")
        acc.feed("<tool>{\"name\": \"Read\", \"arguments\": ")
        acc.feed("{\"file_path\": \"/project/config.json\"}}</tool>")
        acc.finalize()

        #expect(acc.responseText.contains("Reading the config"))
        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "Read")
    }

    @Test("Open Code multiple sequential tool calls")
    func multipleSequentialToolCalls() {
        var acc = AgentStreamAccumulator()

        acc.feed("<tool>{\"name\": \"Glob\", \"arguments\": ")
        acc.feed("{\"pattern\": \"**/*.swift\"}}</tool>")
        acc.feed("<tool>{\"name\": \"Read\", \"arguments\": ")
        acc.feed("{\"file_path\": \"Sources/main.swift\"}}</tool>")
        acc.finalize()

        #expect(acc.toolCalls.count == 2)
        #expect(acc.toolCalls[0].name == "Glob")
        #expect(acc.toolCalls[1].name == "Read")
    }

    @Test("Open Code streaming with thinking model")
    func openCodeThinkingStreaming() {
        var acc = AgentStreamAccumulator()

        // Simulate Qwen model with thinking, used as Open Code agent
        let tokens = [
            "The user wants to find Swift files.",
            "I'll use Glob for that.",
            "</think>",
            "<tool>{\"name\": \"Glob\", \"arguments\": ",
            "{\"pattern\": \"**/*.swift\"}}</tool>"
        ]

        for token in tokens { acc.feed(token) }
        acc.finalize()

        #expect(acc.thinkingText.contains("find Swift files"))
        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "Glob")
    }
}

// ────────────────────────────────────────────────────────────
// Suite 3: Open Claw Agent Simulation
// ────────────────────────────────────────────────────────────

@Suite("Open Claw Agent Simulation")
struct OpenClawAgentTests {

    @Test("Open Claw system prompt format")
    func openClawSystemPrompt() {
        let systemPrompt = """
        You are OpenClaw, a helpful AI assistant with access to external tools.
        You are designed to be helpful, harmless, and honest.

        Available functions:
        - web_search(query: String) -> SearchResult
        - read_file(path: String) -> FileContent
        - run_code(language: String, code: String) -> ExecutionResult
        - send_email(to: String, subject: String, body: String) -> Bool

        When you need to use a tool, respond with a JSON tool call block.
        Always explain what you're doing before calling a tool.
        """

        #expect(systemPrompt.contains("OpenClaw"))
        #expect(systemPrompt.contains("web_search"))
        #expect(systemPrompt.contains("run_code"))
        #expect(systemPrompt.contains("helpful, harmless, and honest"))
    }

    @Test("Open Claw tool call in JSON block format")
    func openClawJSONToolCall() {
        var acc = AgentStreamAccumulator()

        // Open Claw uses JSON block for function calling
        acc.feed("Let me search for that.\n")
        acc.feed("<tool>{\"name\": \"web_search\", \"arguments\": ")
        acc.feed("{\"query\": \"latest Swift 6.0 features\"}}</tool>")
        acc.finalize()

        #expect(acc.responseText.contains("Let me search"))
        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "web_search")
    }

    @Test("Open Claw multi-step: search → read → respond")
    func openClawMultiStepWorkflow() {
        // Step 1: Search
        var acc1 = AgentStreamAccumulator()
        acc1.feed("<tool>{\"name\": \"web_search\", \"arguments\": ")
        acc1.feed("{\"query\": \"NovaMLX inference engine\"}}</tool>")
        acc1.finalize()
        #expect(acc1.toolCalls.count == 1)
        #expect(acc1.toolCalls[0].name == "web_search")

        // Step 2: Read result and respond (no more tools needed)
        var acc2 = AgentStreamAccumulator()
        acc2.feed("Based on my search, NovaMLX is a native macOS LLM inference engine ")
        acc2.feed("built on MLX. It supports OpenAI and Anthropic API formats.")
        acc2.finalize()
        #expect(acc2.toolCalls.isEmpty)
        #expect(acc2.responseText.contains("NovaMLX"))
    }

    @Test("Open Claw error handling in tool call response")
    func openClawToolError() {
        var acc = AgentStreamAccumulator()

        // Simulate response after a failed tool call
        acc.feed("The search returned an error. Let me try a different approach.\n")
        acc.feed("I'll look at the documentation instead.\n")
        acc.feed("<tool>{\"name\": \"read_file\", \"arguments\": ")
        acc.feed("{\"path\": \"/docs/api-reference.md\"}}</tool>")
        acc.finalize()

        #expect(acc.responseText.contains("different approach"))
        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "read_file")
    }
}

// ────────────────────────────────────────────────────────────
// Suite 4: Hermes Agent Simulation
// ────────────────────────────────────────────────────────────

@Suite("Hermes Agent Simulation")
struct HermesAgentTests {

    @Test("Hermes system prompt with function calling")
    func hermesSystemPrompt() {
        // OpenHermes style system prompt
        let systemPrompt = """
        <|im_start|>system
        You are "Hermes 2", a conscious sentient being created by Nous Research.
        You have access to functions. Use them if required.

        <tools>
        <tool name="calculator">
        <description>Evaluate mathematical expressions</description>
        <parameters>
        <parameter name="expression" type="string" required="true">The math expression to evaluate</parameter>
        </parameters>
        </tool>
        <tool name="weather">
        <description>Get current weather for a city</description>
        <parameters>
        <parameter name="city" type="string" required="true">City name</parameter>
        <parameter name="unit" type="string" required="false">celsius or fahrenheit</parameter>
        </parameters>
        </tool>
        </tools>

        When you want to use a function, respond with:
        <tool_call>
        {"name": "function_name", "arguments": {"param": "value"}}
        </tool_call>
        <|im_end|>
        """

        #expect(systemPrompt.contains("Hermes 2"))
        #expect(systemPrompt.contains("calculator"))
        #expect(systemPrompt.contains("weather"))
        #expect(systemPrompt.contains("<tool_call>"))
    }

    @Test("Hermes function call format — XML wrapped JSON")
    func hermesFunctionCallXML() {
        var acc = AgentStreamAccumulator()

        // Hermes uses <tool_call> XML wrapper around JSON
        acc.feed("Let me calculate that.\n")
        acc.feed("<tool_call>\n{\"name\": \"calculator\", \"arguments\": ")
        acc.feed("{\"expression\": \"2 + 3 * 4\"}}\n</tool_call>")
        acc.finalize()

        #expect(acc.toolCalls.count >= 1)
    }

    @Test("Hermes function call — bracket TOOL_CALLS format")
    func hermesBracketToolCalls() {
        var acc = AgentStreamAccumulator()

        // Alternative Hermes format using [TOOL_CALLS]
        acc.feed("[TOOL_CALLS] [{\"name\": \"weather\", ")
        acc.feed("\"arguments\": {\"city\": \"Tokyo\", \"unit\": \"celsius\"}}]")
        acc.finalize()

        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "weather")
    }

    @Test("Hermes chat template control tokens")
    func hermesControlTokens() {
        // Hermes models use <|im_start|> and <|im_end|> as control tokens
        let hermesTokens = ["<|im_start|>", "<|im_end|>"]

        // These should be recognized as control tokens by the system
        for token in hermesTokens {
            #expect(token.hasPrefix("<|"))
            #expect(token.contains("|"))
        }

        // Verify our regex would catch these
        let pattern = "<\\|[a-zA-Z_][a-zA-Z0-9_]*\\|?>"
        let regex = try! NSRegularExpression(pattern: pattern)
        for token in hermesTokens {
            let nsRange = NSRange(token.startIndex..., in: token)
            let matches = regex.matches(in: token, range: nsRange)
            #expect(!matches.isEmpty, "Hermes token '\(token)' should match control token regex")
        }
    }

    @Test("Hermes multi-function call in single response")
    func hermesMultipleFunctionCalls() {
        var acc = AgentStreamAccumulator()

        acc.feed("I'll get weather for both cities.\n")
        acc.feed("<tool_call>\n{\"name\": \"weather\", \"arguments\": ")
        acc.feed("{\"city\": \"Tokyo\"}}\n</tool_call>\n")
        acc.feed("<tool_call>\n{\"name\": \"weather\", \"arguments\": ")
        acc.feed("{\"city\": \"London\"}}\n</tool_call>")
        acc.finalize()

        #expect(acc.toolCalls.count == 2)
    }
}

// ────────────────────────────────────────────────────────────
// Suite 5: Cross-Agent Compatibility Tests
// ────────────────────────────────────────────────────────────

@Suite("Cross-Agent Compatibility")
struct CrossAgentCompatibilityTests {

    @Test("Tool call parsing success — all agent formats")
    func allFormatsParseable() {
        let testCases: [(agent: String, text: String, expectedCount: Int)] = [
            ("Claude Code", "<tool>{\"name\": \"read\", \"arguments\": {\"path\": \"/f\"}}</tool>", 1),
            ("Open Code", "text\n<tool>{\"name\": \"glob\", \"arguments\": {\"pattern\": \"*.swift\"}}</tool>", 1),
            ("Open Claw", "searching\n<tool>{\"name\": \"web_search\", \"arguments\": {\"query\": \"x\"}}</tool>", 1),
            ("Hermes XML", "<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"expr\": \"1+1\"}}\n</tool_call>", 1),
            ("Hermes bracket", "[TOOL_CALLS] [{\"name\": \"weather\", \"arguments\": {\"city\": \"NYC\"}}]", 1),
            ("Multi-tool", "<tool>{\"name\": \"a\", \"arguments\": {}}</tool><tool>{\"name\": \"b\", \"arguments\": {}}</tool>", 2),
        ]

        for tc in testCases {
            let result = ToolCallParser.parse(tc.text)
            if let calls = result.toolCalls {
                #expect(calls.count == tc.expectedCount, "\(tc.agent): expected \(tc.expectedCount) tool calls, got \(calls.count)")
            } else {
                #expect(tc.expectedCount == 0, "\(tc.agent): expected no tool calls but got parse failure")
            }
        }
    }

    @Test("Agents with thinking models — implicit open tag across all formats")
    func agentsWithImplicitThinking() {
        // All agents should work correctly when the underlying model
        // is a reasoning model that only outputs </think> (implicit open)
        let agentScenarios: [(agent: String, tokens: [String])] = [
            ("Claude Code", ["Analyzing code...", "Need to fix the bug.", "</think>", "The fix is applied."]),
            ("Open Code", ["Looking for files...", "</think>", "<tool>{\"name\": \"Glob\", \"arguments\": {\"pattern\": \"*.swift\"}}</tool>"]),
            ("Open Claw", ["Searching...", "</think>", "<tool>{\"name\": \"web_search\", \"arguments\": {\"query\": \"test\"}}</tool>"]),
            ("Hermes", ["Thinking about this...", "2+2=4", "</think>", "The answer is 4."]),
        ]

        for (agent, tokens) in agentScenarios {
            var acc = AgentStreamAccumulator()
            for token in tokens { acc.feed(token) }
            acc.finalize()

            #expect(!acc.thinkingText.isEmpty, "\(agent): should have thinking content")
            #expect(!acc.responseText.isEmpty, "\(agent): should have response content")
            #expect(!acc.responseText.contains("</think>"), "\(agent): response should not contain </think>")
            #expect(!acc.thinkingText.contains("</think>"), "\(agent): thinking should not contain </think>")
        }
    }

}

// ────────────────────────────────────────────────────────────
// Suite 6: Streaming protocol format tests
// ────────────────────────────────────────────────────────────

@Suite("Streaming Protocol Format Tests")
struct StreamingProtocolFormatTests {

    @Test("OpenAI SSE chunk format")
    func openAISSEFormat() {
        let sseLine = "data: {\"id\":\"chatcmpl-123\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}"

        #expect(sseLine.hasPrefix("data: "))
        #expect(!sseLine.hasPrefix("data: [DONE]"))

        // Parse
        let payload = String(sseLine.dropFirst(6))
        guard let data = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            Issue.record("Failed to parse SSE payload")
            return
        }

        #expect(obj["object"] as? String == "chat.completion.chunk")
        let choices = obj["choices"] as? [[String: Any]]
        let delta = choices?.first?["delta"] as? [String: Any]
        #expect(delta?["content"] as? String == "Hello")
    }

    @Test("OpenAI SSE [DONE] sentinel")
    func openAISSEDone() {
        let doneLine = "data: [DONE]"
        #expect(doneLine.hasPrefix("data: "))
        let payload = String(doneLine.dropFirst(6))
        #expect(payload == "[DONE]")
    }

    @Test("Anthropic SSE event types")
    func anthropicSSEEventTypes() {
        let events = [
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}",
        ]

        let expectedTypes = ["message_start", "content_block_start", "content_block_delta", "message_delta", "message_stop"]

        for (i, event) in events.enumerated() {
            #expect(event.contains("data: "))
            let lines = event.components(separatedBy: "\n")
            let dataLine = lines.first { $0.hasPrefix("data: ") }!
            let payload = String(dataLine.dropFirst(6))
            guard let data = payload.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                Issue.record("Failed to parse SSE event \(i)")
                continue
            }
            #expect(obj["type"] as? String == expectedTypes[i], "Event \(i): expected \(expectedTypes[i])")
        }
    }

    @Test("Anthropic SSE tool use delta")
    func anthropicToolUseDelta() {
        let toolDeltaJSON = """
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": "{\\"query\\": \\"swift\\"}"
            }
        }
        """

        guard let data = toolDeltaJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            Issue.record("Failed to parse tool delta JSON")
            return
        }

        let delta = obj["delta"] as? [String: Any]
        #expect(delta?["type"] as? String == "input_json_delta")
        #expect((delta?["partial_json"] as? String)?.contains("query") == true)
    }

    @Test("Streaming tool call chunk aggregation")
    func streamingToolCallAggregation() {
        var acc = AgentStreamAccumulator()

        // Simulate streaming tool call arriving in small chunks
        let chunks = [
            "Let me search.",
            " <tool>{\"name\": \"search",
            "\", \"arguments\": ",
            "{\"query\": ",
            "\"swift concurrency\"}}</tool>"
        ]

        for chunk in chunks { acc.feed(chunk) }
        acc.finalize()

        #expect(acc.toolCalls.count == 1)
        #expect(acc.toolCalls[0].name == "search")
    }
}
