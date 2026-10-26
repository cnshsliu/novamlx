import Foundation
import Testing
import NovaMLXCore

@testable import NovaMLXEngine

@Suite("Flint enforce_tool_schemas")
struct FlintToolSchemaTests {

    private func shellTool(strict: Bool?) -> [String: Any] {
        var function: [String: Any] = [
            "name": "shell",
            "parameters": [
                "type": "object",
                "additionalProperties": false,
                "required": ["command"],
                "properties": [
                    "command": ["type": "string", "pattern": "^[a-z]+$"],
                    "timeout": ["type": "integer"],
                    "verbose": ["type": "boolean"],
                    "meta": ["type": "object"],
                    "mode": ["type": "string", "enum": ["pwd", "ls"]] as [String: Any],
                    "label": ["anyOf": [
                        ["type": "integer"],
                        ["type": "string"],
                    ]] as [String: Any],
                ] as [String: Any],
            ] as [String: Any],
        ]
        if let strict { function["strict"] = strict }
        return ["type": "function", "function": function]
    }

    private func makeGrammar(strict: Bool?, flag: Bool, tools: [[String: Any]]? = nil) throws -> FlintStructuralGrammar {
        let list = tools ?? [shellTool(strict: strict)]
        return try #require(FlintChatDecode.structuralGrammar(
            tools: list, stop: nil, hasRequestConstraint: false, enforceToolSchemas: flag))
    }

    /// Feed GLM markup. Marker tags use their dedicated vocab ids. Other text is one token.
    private func play(_ grammar: inout FlintStructuralGrammar, _ markup: String) -> Bool {
        let markers: [(String, Int)] = [
            ("</arg_value>", FlintStructuralGrammar.argValueCloseId),
            ("</arg_key>", FlintStructuralGrammar.argKeyCloseId),
            ("</tool_call>", FlintStructuralGrammar.toolCloseId),
            ("<arg_value>", FlintStructuralGrammar.argValueOpenId),
            ("<arg_key>", FlintStructuralGrammar.argKeyOpenId),
            ("<tool_call>", FlintStructuralGrammar.toolOpenId),
            ("</think>", FlintStructuralGrammar.thinkCloseId),
            ("<think>", FlintStructuralGrammar.thinkOpenId),
        ]
        var rest = markup[markup.startIndex...]
        while !rest.isEmpty {
            var best: (range: Range<String.Index>, text: String, id: Int)?
            for (text, id) in markers {
                guard let range = rest.range(of: text) else { continue }
                if let current = best {
                    if range.lowerBound > current.range.lowerBound { continue }
                    if range.lowerBound == current.range.lowerBound, text.count <= current.text.count {
                        continue
                    }
                }
                best = (range, text, id)
            }
            if let best, best.range.lowerBound == rest.startIndex {
                let accepted = grammar.consume(tokenId: best.id, text: best.text)
                if !accepted { return false }
                rest = rest[best.range.upperBound...]
                continue
            }
            let end = best?.range.lowerBound ?? rest.endIndex
            let piece = String(rest[..<end])
            let accepted = grammar.consume(tokenId: 1, text: piece)
            if !accepted { return false }
            rest = rest[end...]
        }
        return true
    }

    @Test("omitted strict enforces; strict:false without the flag constrains names only")
    func testFlagOffStaysLoose() throws {
        let omitted = try makeGrammar(strict: nil, flag: false)
        #expect(omitted.toolSchema != nil)
        var omittedGrammar = omitted
        #expect(!play(&omittedGrammar, "<tool_call>nope"))
        var omittedMissing = try makeGrammar(strict: nil, flag: false)
        #expect(!play(&omittedMissing, "<tool_call>shell</tool_call>"))

        let loose = try makeGrammar(strict: false, flag: false)
        #expect(loose.toolSchema != nil)
        var grammar = loose
        let unknown = play(&grammar, "<tool_call>nope</tool_call>")
        #expect(!unknown)

        var violation = try makeGrammar(strict: false, flag: false)
        let missing = play(&violation, "<tool_call>shell</tool_call>")
        #expect(missing)
        var typed = try makeGrammar(strict: false, flag: false)
        let soon = play(
            &typed,
            "<tool_call>shell<arg_key>timeout</arg_key><arg_value>soon</arg_value></tool_call>")
        #expect(soon)
    }

    @Test("strict:true is enforced without the flag, and the flag forces strict:false")
    func testEnforcementGate() throws {
        #expect(try makeGrammar(strict: true, flag: false).toolSchema != nil)
        #expect(try makeGrammar(strict: false, flag: true).toolSchema != nil)
        #expect(try makeGrammar(strict: nil, flag: true).toolSchema != nil)

        let valid = "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
        let cases: [(Bool?, Bool)] = [(true, false), (false, true), (nil, true), (nil, false)]
        for (strict, flag) in cases {
            var grammar = try makeGrammar(strict: strict, flag: flag)
            let accepted = play(&grammar, valid)
            #expect(accepted)
            #expect(grammar.phase == .answer)
        }

        var enforced = try makeGrammar(strict: false, flag: true)
        let missing = play(&enforced, "<tool_call>shell</tool_call>")
        #expect(!missing)
        #expect(enforced.phase == .inTool)
        #expect(enforced.maskedTokenIds().contains(FlintStructuralGrammar.toolCloseId))

        var unknown = try makeGrammar(strict: true, flag: false)
        let rejectedName = play(&unknown, "<tool_call>nope")
        #expect(!rejectedName)
    }

    @Test("schema violations match the shell fixture, including schema key order")
    func testSchemaShapes() throws {
        var soon = try makeGrammar(strict: false, flag: true)
        let badTimeout = play(
            &soon,
            "<tool_call>shell<arg_key>timeout</arg_key><arg_value>soon</arg_value>")
        #expect(!badTimeout)

        var extra = try makeGrammar(strict: false, flag: true)
        let unexpected = play(&extra, "<tool_call>shell<arg_key>extra</arg_key>")
        #expect(!unexpected)

        var ordered = try makeGrammar(strict: false, flag: true)
        let swapped = play(
            &ordered,
            "<tool_call>shell<arg_key>timeout</arg_key><arg_value>12</arg_value><arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>")
        #expect(!swapped)

        var leadZero = try makeGrammar(strict: false, flag: true)
        let zero = play(
            &leadZero,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>timeout</arg_key><arg_value>01</arg_value>")
        #expect(!zero)

        var flagOn = try makeGrammar(strict: false, flag: true)
        let boolOk = play(
            &flagOn,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>verbose</arg_key><arg_value>true</arg_value></tool_call>")
        #expect(boolOk)

        var badBool = try makeGrammar(strict: false, flag: true)
        let yes = play(
            &badBool,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>verbose</arg_key><arg_value>yes</arg_value>")
        #expect(!yes)

        var object = try makeGrammar(strict: false, flag: true)
        let meta = play(
            &object,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>meta</arg_key><arg_value>{\"a\":1}</arg_value></tool_call>")
        #expect(meta)

        var badObject = try makeGrammar(strict: false, flag: true)
        let metaText = play(
            &badObject,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>meta</arg_key><arg_value>soon</arg_value>")
        #expect(!metaText)

        var enumerated = try makeGrammar(strict: false, flag: true)
        let mode = play(
            &enumerated,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>mode</arg_key><arg_value>ls</arg_value></tool_call>")
        #expect(mode)
        var badMode = try makeGrammar(strict: false, flag: true)
        let rm = play(
            &badMode,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>mode</arg_key><arg_value>rm</arg_value>")
        #expect(!rm)

        var union = try makeGrammar(strict: false, flag: true)
        let label = play(
            &union,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>label</arg_key><arg_value>soon</arg_value></tool_call>")
        #expect(label)
    }

    @Test("tool close stays masked until required arguments are present")
    func testRequiredMasksClose() throws {
        var grammar = try makeGrammar(strict: true, flag: false)
        let timeoutFirst = play(&grammar, "<tool_call>shell<arg_key>timeout")
        #expect(!timeoutFirst)
        grammar = try makeGrammar(strict: true, flag: false)
        let opened = play(&grammar, "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value>")
        #expect(opened)
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.toolCloseId))
        let later = play(&grammar, "<arg_key>timeout</arg_key><arg_value>12</arg_value>")
        #expect(later)
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.toolCloseId))
        let closed = play(&grammar, "</tool_call>")
        #expect(closed)
        #expect(grammar.phase == .answer)
    }

    @Test("integer values and patterned strings are vocabulary-masked")
    func testVocabMask() throws {
        var integer = try makeGrammar(strict: false, flag: true)
        let entered = play(
            &integer,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>timeout</arg_key><arg_value>")
        #expect(entered)
        integer.vocabTexts = FlintVocabTextCache(preset: [21: "soon", 22: "12"])
        let masked = integer.maskedTokenIds(vocabSize: 30)
        #expect(masked.contains(21))
        #expect(!masked.contains(22))
        #expect(masked.contains(FlintStructuralGrammar.argValueCloseId))
        integer.advancePhase(tokenId: 21)
        #expect(integer.maskedTokenIds().contains(FlintStructuralGrammar.argValueCloseId))
        integer.advancePhase(tokenId: 22)
        #expect(!integer.maskedTokenIds().contains(FlintStructuralGrammar.argValueCloseId))

        var raw = try makeGrammar(strict: false, flag: true)
        let command = play(&raw, "<tool_call>shell<arg_key>command</arg_key><arg_value>")
        #expect(command)
        raw.vocabTexts = FlintVocabTextCache(preset: [21: "soon", 22: "12"])
        let rawMask = raw.maskedTokenIds(vocabSize: 30)
        #expect(!rawMask.contains(21))
        #expect(rawMask.contains(22))
        #expect(rawMask.contains(FlintStructuralGrammar.argValueCloseId))
    }

    @Test("a mixed request constrains names, then loosens a strict:false body")
    func testMixedTools() throws {
        let echo: [String: Any] = [
            "type": "function",
            "function": [
                "name": "echo",
                "strict": true,
                "parameters": [
                    "type": "object",
                    "properties": ["n": ["type": "integer"]],
                    "required": ["n"],
                    "additionalProperties": false,
                ] as [String: Any],
            ] as [String: Any],
        ]
        var looseBody = try makeGrammar(strict: false, flag: false, tools: [shellTool(strict: false), echo])
        let soon = play(
            &looseBody,
            "<tool_call>shell<arg_key>timeout</arg_key><arg_value>soon</arg_value><arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>")
        #expect(soon)
        var unknown = try makeGrammar(strict: false, flag: false, tools: [shellTool(strict: false), echo])
        let nope = play(&unknown, "<tool_call>nope")
        #expect(!nope)
    }

    @Test("compile does not mutate the client tool list")
    func testToolsUnchanged() throws {
        var tools = [shellTool(strict: false)]
        let before = try JSONSerialization.data(withJSONObject: tools, options: [.sortedKeys])
        _ = FlintToolSchemaSession.compile(tools: tools, enforceToolSchemas: true)
        let after = try JSONSerialization.data(withJSONObject: tools, options: [.sortedKeys])
        #expect(before == after)
        let function = try #require(tools[0]["function"] as? [String: Any])
        #expect((function["strict"] as? Bool) == false)
    }

    @Test("only enforced tools must be named object schemas")
    func testRejection() throws {
        let nameless: [[String: Any]] = [["type": "function"]]
        #expect(
            FlintToolSchemaSession.rejection(tools: nameless, enforceToolSchemas: false)
                == FlintToolSchemaSession.functionNameMessage)
        #expect(
            FlintToolSchemaSession.rejection(tools: nameless, enforceToolSchemas: true)
                == FlintToolSchemaSession.functionNameMessage)
        let notObject: [[String: Any]] = [[
            "type": "function",
            "function": [
                "name": "shell",
                "strict": true,
                "parameters": ["type": "string"],
            ] as [String: Any],
        ]]
        #expect(
            FlintToolSchemaSession.rejection(tools: notObject, enforceToolSchemas: false)
                == FlintToolSchemaSession.objectSchemaMessage)
        let request = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "hi")],
            tools: notObject,
            enforceToolSchemas: false)
        do {
            _ = try FlintAlignment.prepareMessages(request: request)
            Issue.record("object schema should be rejected")
        } catch let NovaMLXError.apiError(message) {
            #expect(message == FlintToolSchemaSession.objectSchemaMessage)
        }
        let constrained = InferenceRequest(
            model: "flint",
            messages: [ChatMessage(role: .user, content: "hi")],
            tools: notObject,
            jsonSchemaDef: ["type": "object"],
            enforceToolSchemas: true)
        #expect(try FlintAlignment.prepareMessages(request: constrained).count == 1)

        let refused = try makeGrammar(strict: nil, flag: true, tools: notObject)
        var failing = refused
        let opened = failing.consume(tokenId: FlintStructuralGrammar.toolOpenId, text: "<tool_call>")
        #expect(opened)
        let body = failing.consume(tokenId: 1, text: "shell")
        #expect(!body)
    }

    @Test("other families ignore the flag")
    func testNonFlint() {
        let request = InferenceRequest(
            model: "qwen",
            messages: [],
            tools: [shellTool(strict: false)],
            enforceToolSchemas: true)
        #expect(FlintChatDecode.chatGrammar(for: request, family: .qwen) == nil)
        #expect(FlintChatDecode.chatGrammar(for: request, family: .flint)?.toolSchema != nil)
    }

    @Test("request flag wins over the environment and the server config")
    func testResolver() {
        #expect(FlintAlignment.resolvedEnforceToolSchemas(
            requestFlag: false, serverConfig: true, environment: "1") == false)
        #expect(FlintAlignment.resolvedEnforceToolSchemas(
            requestFlag: true, serverConfig: false, environment: "no") == true)
        #expect(FlintAlignment.resolvedEnforceToolSchemas(
            requestFlag: nil, serverConfig: false, environment: "yes") == true)
        #expect(FlintAlignment.resolvedEnforceToolSchemas(
            requestFlag: nil, serverConfig: false, environment: "true") == true)
        #expect(FlintAlignment.resolvedEnforceToolSchemas(
            requestFlag: nil, serverConfig: true, environment: "no") == true)
        #expect(FlintAlignment.resolvedEnforceToolSchemas(
            requestFlag: nil, serverConfig: false, environment: nil) == false)
    }

    @Test("pattern, format, bounds, nested objects, and JSON key order are masked")
    func testSchemaConstraints() throws {
        let json = """
        [{"type":"function","function":{"name":"shell","strict":false,"parameters":{"type":"object","additionalProperties":false,"required":["command"],"properties":{"timeout":{"type":"integer","minimum":1,"maximum":10},"command":{"type":"string","pattern":"^[a-z]+$","maxLength":8},"when":{"type":"string","format":"date"},"meta":{"type":"object","additionalProperties":false,"required":["path"],"properties":{"path":{"type":"string"},"n":{"type":"integer"}}}}}}}]
        """
        func grammar() throws -> FlintStructuralGrammar {
            try #require(FlintChatDecode.structuralGrammar(
                tools: nil, stop: nil, hasRequestConstraint: false,
                enforceToolSchemas: true, toolsJSON: json))
        }
        // JSON order is timeout, command, when, meta. 11 is above maximum 10.
        var early = try grammar()
        #expect(!play(
            &early,
            "<tool_call>shell<arg_key>timeout</arg_key><arg_value>11</arg_value>"))
        var inRange = try grammar()
        #expect(play(
            &inRange,
            "<tool_call>shell<arg_key>timeout</arg_key><arg_value>9</arg_value><arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"))
        var pattern = try grammar()
        #expect(!play(
            &pattern,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>12</arg_value>"))
        var dated = try grammar()
        #expect(!play(
            &dated,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>when</arg_key><arg_value>soon</arg_value>"))
        var goodDate = try grammar()
        #expect(play(
            &goodDate,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>when</arg_key><arg_value>2020-01-02</arg_value></tool_call>"))
        var nested = try grammar()
        #expect(!play(
            &nested,
            #"<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>meta</arg_key><arg_value>{"n":1}</arg_value>"#))
        var nestedOK = try grammar()
        #expect(play(
            &nestedOK,
            #"<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>meta</arg_key><arg_value>{"path":"/tmp"}</arg_value></tool_call>"#))
        var wrongOrder = try grammar()
        #expect(!play(
            &wrongOrder,
            "<tool_call>shell<arg_key>command</arg_key><arg_value>pwd</arg_value><arg_key>timeout</arg_key>"))
    }
}
