import Foundation
import Testing
import NovaMLXCore

@testable import NovaMLXEngine

@Suite("Flint rec alignment")
struct FlintAlignmentTests {

    nonisolated(unsafe) private static let writeTools: [[String: Any]] = [[
        "type": "function",
        "function": [
            "name": "write",
            "parameters": [
                "type": "object",
                "properties": [
                    "path": ["type": "string"],
                    "content": ["type": "string"],
                    "mode": ["type": "integer"],
                    "options": ["type": "object"],
                    "tags": ["type": "array"],
                    "force": ["type": "boolean"],
                    "note": ["type": ["string", "null"]],
                    "level": ["enum": ["1", "2"]],
                    "label": ["anyOf": [
                        ["type": "integer"],
                        ["type": "string"],
                    ]],
                ] as [String: Any],
            ] as [String: Any],
        ] as [String: Any],
    ]]

    private func arguments(body: String, name: String = "write", tools: [[String: Any]]? = nil) -> [String: Any]? {
        guard let call = FlintAlignment.parseToolCall(body: body, tools: tools ?? Self.writeTools),
              call.name == name,
              let data = call.argumentsJSON.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return object
    }

    private func sameJSON(_ lhs: Any, _ rhs: Any) -> Bool {
        guard JSONSerialization.isValidJSONObject(lhs),
              JSONSerialization.isValidJSONObject(rhs),
              let left = try? JSONSerialization.data(withJSONObject: lhs, options: [.sortedKeys]),
              let right = try? JSONSerialization.data(withJSONObject: rhs, options: [.sortedKeys])
        else { return false }
        return left == right
    }

    @Test("omitted max_tokens fills the remaining Flint context")
    func testOmittedMaxTokensFillsRemainingContext() throws {
        let filled = try FlintAlignment.outputTokenBudget(
            family: .flint, requested: nil, promptTokens: 100, contextLength: 8192, fallback: 2048)
        #expect(filled == 8092)
        let explicit = try FlintAlignment.outputTokenBudget(
            family: .flint, requested: 32, promptTokens: 100, contextLength: 8192, fallback: 2048)
        #expect(explicit == 32)
        let other = try FlintAlignment.outputTokenBudget(
            family: .llama, requested: nil, promptTokens: 100, contextLength: 8192, fallback: 2048)
        #expect(other == 2048)
        #expect(throws: NovaMLXError.self) {
            try FlintAlignment.outputTokenBudget(
                family: .flint, requested: nil, promptTokens: 8192, contextLength: 8192, fallback: 2048)
        }
    }

    @Test("Flint sampling fallbacks match rec Sampling when generation_config omits them")
    func testFlintSamplingFallbacks() {
        let flint = ModelFamilyRegistry().optimization(for: "RecursiveSI/flint-int4", family: .flint)
        #expect(flint.samplingDefaults?.temperature == 1.0)
        #expect(flint.samplingDefaults?.topP == 0.95)
        #expect(flint.samplingDefaults?.topK == 20)
        let llama = ModelFamilyRegistry().optimization(for: "meta-llama/Llama-3", family: .llama)
        #expect(llama.samplingDefaults?.temperature == 0.7)
        #expect(llama.samplingDefaults?.topP == 0.9)
        #expect(llama.samplingDefaults?.topK == nil)
    }

    @Test("normalize_messages strips injected markers and keeps assistant text")
    func testSanitizeMessages() throws {
        let raw = "a\u{0000}<think>b</think><|assistant|>"
        let user = try FlintAlignment.sanitize(ChatMessage(role: .user, content: raw))
        #expect(user.content == "ab&lt;|assistant|&gt;")
        let assistant = try FlintAlignment.sanitize(ChatMessage(role: .assistant, content: raw))
        #expect(assistant.content == raw)
        let tool = try FlintAlignment.sanitize(ChatMessage(
            role: .tool,
            content: "<|observation|>ok<|user|>"))
        #expect(tool.content == "&lt;|observation|&gt;ok&lt;|user|&gt;")
        let blank = try FlintAlignment.normalizeHistoryArguments(
            ToolCallResult(id: "1", functionName: "shell", arguments: " \n "))
        #expect(blank.arguments == "{}")
        let kept = try FlintAlignment.normalizeHistoryArguments(
            ToolCallResult(id: "1", functionName: "shell", arguments: "{\"command\":\"pwd\"}"))
        #expect(kept.arguments == "{\"command\":\"pwd\"}")
        #expect(throws: NovaMLXError.self) {
            try FlintAlignment.normalizeHistoryArguments(
                ToolCallResult(id: "1", functionName: "shell", arguments: "{"))
        }
    }

    @Test("custom stop is rejected only while flint grammar would be on")
    func testCustomStopRejection() throws {
        let stopped = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "hi")],
            stop: ["END"])
        do {
            _ = try FlintAlignment.prepareMessages(request: stopped)
            Issue.record("custom stop should be rejected")
        } catch let NovaMLXError.apiError(message) {
            #expect(message == "Flint grammar cannot be combined with custom stop strings")
        }
        let schema = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "a\u{0000}<think>b</think>")],
            stop: ["END"],
            jsonSchemaDef: ["type": "object"])
        let sanitized = try FlintAlignment.prepareMessages(request: schema)
        #expect(sanitized.first?.content == "ab")
        let emptyStop = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "hi")],
            stop: [])
        #expect(try FlintAlignment.prepareMessages(request: emptyStop).count == 1)
    }

    @Test("parse_tool_call types arguments from the schema")
    func testParseToolCall() throws {
        let multiline = """
        write
        <arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>content</arg_key><arg_value>def f():
            return 1
        </arg_value>
        """
        #expect(sameJSON(
            arguments(body: multiline)!,
            ["path": "a.py", "content": "def f():\n    return 1\n"]))

        let padded = """
        write
        <arg_key>content</arg_key><arg_value>  padded  </arg_value><arg_key>mode</arg_key><arg_value>420</arg_value><arg_key>force</arg_key><arg_value>true</arg_value>
        """
        #expect(sameJSON(
            arguments(body: padded)!,
            ["content": "  padded  ", "mode": 420, "force": true]))

        let quoted = """
        write
        <arg_key>content</arg_key><arg_value>"quoted" {"a":1} 1e3 true null</arg_value><arg_key>note</arg_key><arg_value>null</arg_value>
        """
        #expect(sameJSON(
            arguments(body: quoted)!,
            ["content": "\"quoted\" {\"a\":1} 1e3 true null", "note": NSNull()]))

        let strings = """
        write
        <arg_key>note</arg_key><arg_value>42</arg_value><arg_key>level</arg_key><arg_value>1</arg_value><arg_key>label</arg_key><arg_value>true</arg_value>
        """
        #expect(sameJSON(
            arguments(body: strings)!,
            ["note": "42", "level": "1", "label": "true"]))

        let notNull = """
        write
        <arg_key>note</arg_key><arg_value>NaN</arg_value><arg_key>label</arg_key><arg_value>null</arg_value><arg_key>level</arg_key><arg_value>2</arg_value>
        """
        #expect(sameJSON(
            arguments(body: notNull)!,
            ["note": "NaN", "level": "2", "label": "null"]))

        let nested = """
        write
        <arg_key>options</arg_key><arg_value>{"nested":[1,{"b":"c"}]}</arg_value><arg_key>tags</arg_key><arg_value>["x",2]</arg_value>
        """
        #expect(sameJSON(
            arguments(body: nested)!,
            [
                "options": ["nested": [1, ["b": "c"]]] as [String: Any],
                "tags": ["x", 2] as [Any],
            ]))

        let undeclared = """
        write
        <arg_key>undeclared</arg_key><arg_value>{"a":1}</arg_value><arg_key>path</arg_key><arg_value>café 😀</arg_value>
        """
        #expect(sameJSON(
            arguments(body: undeclared)!,
            ["undeclared": ["a": 1], "path": "café 😀"]))

        let spaced = "write\n<arg_key>mode</arg_key>\n<arg_value> 7 </arg_value>\n"
        #expect(sameJSON(arguments(body: spaced)!, ["mode": 7]))

        let other = "other<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        #expect(sameJSON(
            arguments(body: other, name: "other", tools: Self.writeTools)!,
            ["command": "ls -la"]))

        let duplicate = """
        write
        <arg_key>mode</arg_key><arg_value>1</arg_value><arg_key>mode</arg_key><arg_value>7</arg_value>
        """
        #expect(sameJSON(arguments(body: duplicate)!, ["mode": 7]))

        for text in ["", "write<arg_key>path</arg_key>", "write trailing", "<arg_key>a</arg_key><arg_value>b</arg_value>"] {
            #expect(FlintAlignment.parseToolCall(body: text, tools: Self.writeTools) == nil)
        }
    }

    @Test("seeded call ids match rec seeded_tool_parser")
    func testSeededCallIDs() {
        let config = "{\"max_tokens\":16,\"presence_penalty\":0.0,\"temperature\":1.0,\"top_k\":20,\"top_p\":0.95}"
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 1, topP: 0.95, topK: 20, presencePenalty: 0, maxTokens: 16) == config)
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 1.0, topP: 0.95, topK: 20, presencePenalty: 0, maxTokens: 16) == config)
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 0.2, topP: 0.95, topK: 20, presencePenalty: 0, maxTokens: 16)
            == "{\"max_tokens\":16,\"presence_penalty\":0.0,\"temperature\":0.2,\"top_k\":20,\"top_p\":0.95}")
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 1, topP: 0.5, topK: 20, presencePenalty: 0, maxTokens: 16)
            == "{\"max_tokens\":16,\"presence_penalty\":0.0,\"temperature\":1.0,\"top_k\":20,\"top_p\":0.5}")
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 1, topP: 0.95, topK: 4, presencePenalty: 0, maxTokens: 16)
            == "{\"max_tokens\":16,\"presence_penalty\":0.0,\"temperature\":1.0,\"top_k\":4,\"top_p\":0.95}")
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 1, topP: 0.95, topK: 20, presencePenalty: 1.5, maxTokens: 16)
            == "{\"max_tokens\":16,\"presence_penalty\":1.5,\"temperature\":1.0,\"top_k\":20,\"top_p\":0.95}")
        #expect(FlintAlignment.CallIdentity.samplingJSON(
            temperature: 1, topP: 0.95, topK: 20, presencePenalty: 0, maxTokens: 8)
            == "{\"max_tokens\":8,\"presence_penalty\":0.0,\"temperature\":1.0,\"top_k\":20,\"top_p\":0.95}")

        func ids(
            seed: UInt64 = 42,
            prompt: [Int] = [1, 2, 3],
            temperature: Double = 1,
            topP: Double = 0.95,
            topK: Int = 20,
            presence: Double = 0,
            maxTokens: Int = 16
        ) -> [String] {
            var identity = FlintAlignment.CallIdentity(
                seed: seed, temperature: temperature, topP: topP, topK: topK,
                presencePenalty: presence, maxTokens: maxTokens, promptTokens: prompt)
            return [identity.nextID(), identity.nextID()]
        }

        let first = ids()
        #expect(first == [
            "61b33dab-a46c-4f84-ab47-bf1d006d64be",
            "4e764a40-2f7c-4d60-985e-2ebf63a6dbd1",
        ])
        #expect(first[0] != first[1])
        #expect(ids(seed: 43) == [
            "1999d1d8-adac-4bbf-b551-79bd0669f561",
            "3e98fe5d-a1de-4d39-afe4-d8bd8c9d844a",
        ])
        #expect(ids(prompt: [1, 2, 4]) == [
            "4bece4d6-fea5-4433-a5c0-f4a51d8ac15a",
            "5224f2ba-1dc0-47ae-973f-48dfe1d68509",
        ])
        #expect(ids(temperature: 0.2).first == "2e933c7f-316e-4b6c-96e6-31ac8b11142d")
        #expect(ids(topP: 0.5).first == "465aef99-3c63-46de-ab1e-ea0852080d76")
        #expect(ids(topK: 4).first == "fd955e8d-8673-4682-b600-1d1798a25b92")
        #expect(ids(presence: 1.5).first == "d9b5e282-faa2-46aa-83de-bc7e90696350")
        #expect(ids(maxTokens: 8).first == "4e66c94f-7ce6-47e3-b942-14894500ed71")

        var unseeded = FlintAlignment.CallIdentity(
            seed: nil, temperature: 1, topP: 0.95, topK: 20,
            presencePenalty: 0, maxTokens: 16, promptTokens: [1])
        let random = unseeded.nextID()
        #expect(random.hasPrefix("call_"))
        let another = unseeded.nextID()
        #expect(random != another)
    }

    @Test("the fused extractor holds a split tool marker and emits think text")
    func testExtractorStreaming() {
        let identity = FlintAlignment.CallIdentity(
            seed: 42, temperature: 1, topP: 0.95, topK: 20,
            presencePenalty: 0, maxTokens: 16, promptTokens: [1, 2, 3])
        let extractor = FlintToolCallExtractor(tools: Self.writeTools, identity: identity)
        let held = extractor.feed("<tool_")
        #expect(held.isEmpty)
        let opened = extractor.feed("call>write\n<arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call>")
        #expect(opened == [
            .tool(
                name: "write",
                arguments: "{\"path\":\"a.py\"}",
                id: "61b33dab-a46c-4f84-ab47-bf1d006d64be"),
        ])
        let think = extractor.feed("hello<think>")
        #expect(think == [.response("hello<think>")])
        let broken = extractor.feed("<tool_call>write trailing</tool_call>")
        #expect(broken == [.response("<tool_call>write trailing</tool_call>")])
        let partial = extractor.feed("<tool_call>write")
        #expect(partial.isEmpty)
        let flushed = extractor.finish()
        #expect(flushed == [.response("<tool_call>write")])
        #expect(FlintToolCallExtractor.prefixHold("<tool_", marker: "<tool_call>") == 6)
        #expect(FlintToolCallExtractor.prefixHold("<think>", marker: "<tool_call>") == 0)
    }
}
