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
        #expect(ids(seed: 0) == [
            "da52baae-f952-46cc-8743-b38cf6beff9c",
            "772fcc4f-f3e7-46c3-a865-cf1e8b107cd7",
        ])
        #expect(FlintAlignment.resolvedSeed(requestSeed: nil, modelSeed: nil) == 0)
        #expect(FlintAlignment.resolvedSeed(requestSeed: nil, modelSeed: 9) == 9)
        #expect(FlintAlignment.resolvedSeed(requestSeed: 7, modelSeed: 9) == 7)

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

    @Test("omitted Flint sampling and API fields match rec validate_request")
    func testValidateRequest() throws {
        let config = ModelConfig(
            identifier: ModelIdentifier(id: "RecursiveSI/flint-int4", family: .flint))
        func message(_ request: InferenceRequest) -> String? {
            do {
                try FlintAlignment.validate(request: request, config: config)
                return nil
            } catch let NovaMLXError.apiError(text) {
                return text
            } catch {
                return String(describing: error)
            }
        }
        func request(
            frequency: Float? = nil,
            minP: Float? = nil,
            repetition: Float? = nil,
            stream: Bool = false,
            logprobs: Bool = false,
            toolChoice: String? = nil,
            parallel: Bool? = nil,
            store: Bool? = nil,
            grammar: String? = nil,
            stop: [String]? = nil,
            role: String? = nil
        ) -> InferenceRequest {
            InferenceRequest(
                model: "flint",
                messages: [ChatMessage(role: .user, content: "hi", wireRole: role)],
                frequencyPenalty: frequency,
                minP: minP,
                repetitionPenalty: repetition,
                stream: stream,
                stop: stop,
                includeLogprobs: logprobs,
                toolChoice: toolChoice,
                parallelToolCalls: parallel,
                storeCompletions: store,
                flintGrammar: grammar)
        }
        #expect(message(request(frequency: 0.1))
            == "Nonzero frequency_penalty is not supported by this decoder")
        #expect(message(request(minP: 0.05))
            == "Nonzero min_p is not supported by this decoder")
        #expect(message(request(repetition: 1.5))
            == "Only neutral repetition_penalty=1 is supported")
        #expect(message(request(repetition: 0)) == nil)
        #expect(message(request(repetition: 1)) == nil)
        #expect(message(request(stream: true, logprobs: true))
            == "Token logprobs require stream=false")
        #expect(message(request(logprobs: true)) == nil)
        #expect(message(request(toolChoice: "required"))
            == "tool_choice currently supports auto and none; forced calls are not implemented")
        #expect(message(request(parallel: false))
            == "parallel_tool_calls=false is not supported")
        #expect(message(request(store: true))
            == "Stored-completion retrieval is not supported; use store=false")
        #expect(message(request(store: false)) == nil)
        #expect(message(request(grammar: "json")) == "grammar must be none or flint")
        #expect(message(request(role: "function")) == "Unsupported message role")
        #expect(message(request()) == nil)

        let stopped = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "hi")],
            stop: ["END"],
            flintGrammar: "none")
        #expect(try FlintAlignment.prepareMessages(request: stopped).count == 1)
        #expect(message(request(stop: ["END"], grammar: "none")) == nil)

        let original = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "hi")],
            toolChoice: "none",
            parallelToolCalls: true,
            storeCompletions: false,
            flintGrammar: "none",
            toolsJSON: #"[{"type":"function"}]"#)
        var settings = ModelSettings()
        settings.temperature = 0.2
        let overridden = settings.applySamplingOverrides(to: original)
        #expect(overridden.toolChoice == "none")
        #expect(overridden.flintGrammar == "none")
        #expect(overridden.toolsJSON == original.toolsJSON)
        let restored = CodableInferenceRequest(from: overridden).toInferenceRequest()
        #expect(restored.toolChoice == "none")
        #expect(restored.parallelToolCalls == true)
        #expect(restored.storeCompletions == false)
        #expect(restored.flintGrammar == "none")
        #expect(restored.toolsJSON == original.toolsJSON)
    }

    @Test("the bundled Flint template is the rec package file")
    func testPinnedChatTemplate() throws {
        #expect(FlintRecChatTemplate.bundledSHA256() == FlintRecChatTemplate.sha256)
        let tool = try #require(FlintOrderedJSON.parse(
            #"{"type":"function","function":{"name":"shell","parameters":{"type":"object"},"strict":false}}"#))
        let text = FlintRecChatTemplate.render(
            messages: [
                .init(role: "system", content: "instructions"),
                .init(role: "system", content: "additional instructions"),
                .init(role: "developer", content: "developer instruction"),
                .init(role: "user", content: "ab&lt;|assistant|&gt;"),
                .init(role: "assistant", content: "answer", reasoning: "reasoning"),
                .init(role: "assistant", toolCalls: [
                    .init(name: "shell", arguments: [("command", .string("pwd"))]),
                ]),
                .init(role: "tool", content: "/work"),
            ],
            tools: [tool])
        #expect(text.hasPrefix("<|system|>instructions<|system|>additional instructions<|system|>\n# Tools\n"))
        #expect(text.contains(#""strict": false"#))
        #expect(text.contains("<think>reasoning</think>answer"))
        #expect(text.contains("<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"))
        #expect(text.hasSuffix("<|observation|><tool_response>/work</tool_response><|assistant|>"))
        #expect(FlintRecChatTemplate.sha256Hex(Data(text.utf8))
            == "032f373234cb07781f4ca3f90a69852afdd16ac6179daa6f2074eda0cb9fce0a")
        let ids = FlintRecChatTemplate.fixtureTokenIDs(text)
        #expect(ids.count == 489)
        #expect(ids.first == 3)
        #expect(ids.last == 6)
        let joined = ids.map(String.init).joined(separator: ",")
        #expect(FlintRecChatTemplate.sha256Hex(Data(joined.utf8))
            == "cc43c498d4271f96774850c0e9b7589b51842dea8a72072264908474d2dacdf2")

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-template-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = dir.appendingPathComponent("tokenizer_config.json")
        try Data(#"{"chat_template":"checkpoint"}"#.utf8).write(to: config)
        let jinja = dir.appendingPathComponent("chat_template.jinja")
        try Data("other".utf8).write(to: jinja)
        #expect(FlintRecChatTemplate.install(modelDir: dir, modelId: "flint") == FlintRecChatTemplate.sha256)
        let written = try Data(contentsOf: config)
        let object = try #require(JSONSerialization.jsonObject(with: written) as? [String: Any])
        #expect(object["chat_template"] as? String == FlintRecChatTemplate.bundledSource())
        #expect(!FileManager.default.fileExists(atPath: jinja.path))
        #expect(FileManager.default.fileExists(atPath: jinja.appendingPathExtension("disabled-by-novamlx").path))
    }
}
