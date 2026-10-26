import Foundation
import MLX
import Testing
import NovaMLXCore

@testable import NovaMLXEngine

@Suite("Flint chat decode policy")
struct FlintChatDecodeTests {

    private func policy(
        temperature: Float = 0,
        topP: Float = 1,
        topK: Int = 0,
        minP: Float = 0,
        frequencyPenalty: Float = 0,
        repetitionPenalty: Float = 0,
        presencePenalty: Float = 0,
        wantsLogprobs: Bool = false,
        batchWidth: Int = 1,
        usesGrammar: Bool = false,
        usesStructuralGrammar: Bool = false
    ) -> FlintChatDecode.Policy {
        FlintChatDecode.Policy(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            frequencyPenalty: frequencyPenalty,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            wantsLogprobs: wantsLogprobs,
            batchWidth: batchWidth,
            usesGrammar: usesGrammar,
            usesStructuralGrammar: usesStructuralGrammar)
    }

    @Test("resident packet is greedy, single-sequence, and logit-neutral")
    func testAllowsResidentPacket() {
        #expect(FlintChatDecode.allowsResidentPacket(policy()))
        #expect(FlintChatDecode.allowsResidentPacket(policy(repetitionPenalty: 1)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(temperature: 1)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(frequencyPenalty: 0.1)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(repetitionPenalty: 1.1)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(presencePenalty: 0.2)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(wantsLogprobs: true)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(batchWidth: 2)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(usesGrammar: true)))
    }

    @Test("repetition penalty of 0 or 1 does not rewrite logits")
    func testPenaltyChangesLogits() {
        #expect(!FlintChatDecode.penaltyChangesLogits(0))
        #expect(!FlintChatDecode.penaltyChangesLogits(1))
        #expect(FlintChatDecode.penaltyChangesLogits(1.2))
        #expect(FlintChatDecode.penaltyChangesLogits(-1))
    }

    @Test("packet stop ids stay within the kernel cap of four")
    func testPacketStopIds() {
        #expect(FlintChatDecode.packetStopIds(tokenizerEOS: nil) == FlintChatDecode.canonicalStopIds)
        #expect(FlintChatDecode.packetStopIds(tokenizerEOS: 150001) == [
            150001, 150003, 150005, 150018,
        ])
        let capped = FlintChatDecode.packetStopIds(tokenizerEOS: 2, extra: [9, 8, 7])
        #expect(capped == [2, 150001, 150003, 150005])
        #expect(capped.count == 4)
    }

    @Test("kept prefix includes the stop token and reports the suffix to trim")
    func testKeptPrefix() {
        let clean = FlintChatDecode.keptPrefix(tokenIds: [4, 5, 6], stopIds: [150001])
        #expect(clean.kept == [4, 5, 6])
        #expect(clean.unused == 0)

        let split = FlintChatDecode.keptPrefix(tokenIds: [1, 2, 150001, 3, 4], stopIds: [150001])
        #expect(split.kept == [1, 2, 150001])
        #expect(split.unused == 2)

        let tail = FlintChatDecode.keptPrefix(tokenIds: [9, 150018], stopIds: Set(FlintChatDecode.canonicalStopIds))
        #expect(tail.kept == [9, 150018])
        #expect(tail.unused == 0)
    }

    @Test("the rate line is decode tokens over wall seconds")
    func testLine() {
        #expect(FlintChatDecode.line(
            decodeTokens: 0, seconds: 1, packetTokens: 0, singleTokens: 0, launches: 0) == nil)
        #expect(FlintChatDecode.line(
            decodeTokens: 10, seconds: 0, packetTokens: 10, singleTokens: 0, launches: 1) == nil)

        let line = FlintChatDecode.line(
            decodeTokens: 128, seconds: 0.5, packetTokens: 120, singleTokens: 8, launches: 16)
        #expect(line?.contains("FLINT-CHAT: 256.0 tok/s") == true)
        #expect(line?.contains("128 decode tokens / 0.500s wall") == true)
        #expect(line?.contains("prefill excluded") == true)
        #expect(line?.contains("packetTokens=120") == true)
        #expect(line?.contains("singleTokens=8") == true)
        #expect(line?.contains("launches=16") == true)
        #expect(line?.contains("width=\(FlintRecResident.width)") == true)
        #expect(line?.contains("Formula: decodeTokens / wallDecodeSeconds") == true)
        #expect(line?.contains("LOOP-RESULT") == true)
    }

    @Test("a no-op processor still allows the packet")
    func testProcessorAllowsPacket() {
        #expect(FlintChatDecode.processorAllowsPacket(nil))
        let turn = TurnStopProcessor(
            stopPatterns: ["<|user|>"], eosTokenIds: [150001], decode: { _ in "" })
        #expect(!turn.isConstraining)
        #expect(FlintChatDecode.processorAllowsPacket(turn))

        let budget = ThinkingBudgetProcessor(budget: 32, closeTokenIds: [150001])
        #expect(!budget.isConstraining)
        #expect(FlintChatDecode.processorAllowsPacket(budget))

        let idle = ComposedLogitProcessor(
            grammarProcessor: nil,
            penaltyProcessor: nil,
            turnStopProcessor: turn,
            thinkingBudgetProcessor: nil)
        #expect(FlintChatDecode.processorAllowsPacket(idle))

        let penalized = ComposedLogitProcessor(
            grammarProcessor: nil,
            penaltyProcessor: turn,
            turnStopProcessor: nil,
            thinkingBudgetProcessor: nil)
        #expect(!FlintChatDecode.processorAllowsPacket(penalized))
    }

    private func stops(primaryEOS: Int = 150001) -> Set<Int> {
        Set(FlintChatDecode.packetStopIds(tokenizerEOS: primaryEOS))
    }

    /// Same order as the fused decode step: a stop id ends the turn before
    /// its text is kept; otherwise the think guard sees the decoded piece.
    private func reduce(
        _ tokens: [(id: Int, text: String)],
        primaryEOS: Int = 150001,
        maxTokens: Int = 32
    ) -> (text: String, finish: FinishReason, steps: Int) {
        var state = FlintChatDecode.ThinkReentryGuard()
        let stopIds = stops(primaryEOS: primaryEOS)
        var text = ""
        var counted = 0
        for (index, token) in tokens.enumerated() {
            let step = FlintChatDecode.tokenDecision(
                tokenId: token.id, decoded: token.text,
                primaryEOS: primaryEOS, stopIds: stopIds, state: &state)
            if step.isEOS {
                return (text, .stop, index + 1)
            }
            text += step.yield
            counted += 1
            if step.thinkReentry {
                return (text, .stop, index + 1)
            }
            if counted >= maxTokens {
                return (text, .length, index + 1)
            }
        }
        return (text, .length, tokens.count)
    }

    @Test("a flint stop id other than the primary eos finishes the turn")
    func testSecondaryStopIdFinishes() {
        let stopIds = stops()
        #expect(FlintChatDecode.isStopToken(150003, primaryEOS: 150001, stopIds: stopIds))
        #expect(FlintChatDecode.isStopToken(150005, primaryEOS: 150001, stopIds: stopIds))
        #expect(FlintChatDecode.isStopToken(150018, primaryEOS: 150001, stopIds: stopIds))
        #expect(FlintChatDecode.isStopToken(150001, primaryEOS: 150001, stopIds: stopIds))
        #expect(!FlintChatDecode.isStopToken(
            FlintChatDecode.thinkOpenTokenId, primaryEOS: 150001, stopIds: stopIds))
        // Other families pass an empty stop set: only the primary eos matches.
        #expect(FlintChatDecode.isStopToken(2, primaryEOS: 2, stopIds: []))
        #expect(!FlintChatDecode.isStopToken(150003, primaryEOS: 2, stopIds: []))

        var state = FlintChatDecode.ThinkReentryGuard()
        let user = FlintChatDecode.tokenDecision(
            tokenId: 150003, decoded: "<|user|>",
            primaryEOS: 150001, stopIds: stopIds, state: &state)
        #expect(user.isEOS)
        #expect(user.yield.isEmpty)
        #expect(!user.thinkReentry)

        let developer = FlintChatDecode.tokenDecision(
            tokenId: 150018, decoded: "<|developer|>",
            primaryEOS: 150001, stopIds: stopIds, state: &state)
        #expect(developer.isEOS)
        #expect(developer.yield.isEmpty)

        let walked = reduce([
            (12, "Hello"),
            (150003, "<|user|>"),
            (13, "should not appear"),
        ])
        #expect(walked.finish == .stop)
        #expect(walked.text == "Hello")
        #expect(walked.steps == 2)
    }

    @Test("reopening think after a visible answer stops inside a small bound")
    func testThinkReentryAfterAnswer() {
        let open = FlintChatDecode.thinkOpenTokenId
        let close = FlintChatDecode.thinkCloseTokenId
        var script: [(id: Int, text: String)] = [
            (open, "<think>"),
            (11, "We need to respond to the user."),
            (close, "</think>"),
            (12, "Hello! How can I help you today?"),
            (open, "<think>"),
        ]
        script += Array(repeating: (99, "Hello again!"), count: 40)

        let walked = reduce(script, maxTokens: 64)
        #expect(walked.finish == .stop)
        #expect(walked.steps == 5)
        #expect(walked.steps < 16)
        #expect(walked.text.contains("<think>"))
        #expect(walked.text.contains("</think>"))
        #expect(walked.text.contains("Hello! How can I help you today?"))
        #expect(!walked.text.contains("Hello again"))
        #expect(walked.text.components(separatedBy: "<think>").count - 1 == 1)
    }

    @Test("a greeting with no think block stops when think opens")
    func testThinkReentryWithoutPriorThink() {
        let open = FlintChatDecode.thinkOpenTokenId
        let walked = reduce([
            (12, "Hello! How can I help you today?"),
            (open, "<think>"),
            (11, "We need to respond to the user."),
        ] + Array(repeating: (99, "loop"), count: 30), maxTokens: 64)
        #expect(walked.finish == .stop)
        #expect(walked.steps == 2)
        #expect(walked.text == "Hello! How can I help you today?")
        #expect(!walked.text.contains("<think>"))
    }

    @Test("whitespace between think blocks is not a visible answer")
    func testThinkReentryIgnoresWhitespace() {
        let open = FlintChatDecode.thinkOpenTokenId
        let close = FlintChatDecode.thinkCloseTokenId
        let walked = reduce([
            (open, "<think>"),
            (11, "reason"),
            (close, "</think>"),
            (2, "\n\n"),
            (open, "<think>"),
            (13, "more"),
            (close, "</think>"),
            (14, "Hi"),
        ])
        #expect(walked.steps == 8)
        #expect(walked.text == "<think>reason</think>\n\n<think>more</think>Hi")
    }

    @Test("a split think tag after an answer stops at the boundary")
    func testSplitThinkTagStops() {
        let walked = reduce([
            (12, "Hello"),
            (3, "<"),
            (4, "think>"),
            (99, "loop"),
        ])
        #expect(walked.finish == .stop)
        #expect(walked.text == "Hello")
        #expect(walked.steps == 3)
    }

    @Test("GLM4 tool calls still pass through, including a think token inside")
    func testToolCallSurvivesThinkToken() {
        let open = FlintChatDecode.thinkOpenTokenId
        let walked = reduce([
            (12, "Hello"),
            (FlintChatDecode.toolCallOpenTokenId, "<tool_call>"),
            (21, "search"),
            (open, "<think>"),
            (22, "q"),
            (FlintChatDecode.toolCallCloseTokenId, "</tool_call>"),
            (23, "done"),
        ])
        #expect(walked.steps == 7)
        #expect(walked.text == "Hello<tool_call>search<think>q</tool_call>done")
    }

    @Test("a think open after a finished tool call stops the turn")
    func testThinkAfterToolCallStops() {
        let walked = reduce([
            (FlintChatDecode.toolCallOpenTokenId, "<tool_call>"),
            (21, "search"),
            (FlintChatDecode.toolCallCloseTokenId, "</tool_call>"),
            (FlintChatDecode.thinkOpenTokenId, "<think>"),
            (99, "loop"),
        ])
        #expect(walked.finish == .stop)
        #expect(walked.text == "<tool_call>search</tool_call>")
        #expect(walked.steps == 4)
    }

    @Test("a fast prose packet arms the think-reentry guard")
    func testFastPacketThenThinkStops() {
        var state = FlintChatDecode.ThinkReentryGuard()
        let armed = state.observeFastPacket("Hello! How can I help you today?")
        #expect(armed)
        let step = FlintChatDecode.tokenDecision(
            tokenId: FlintChatDecode.thinkOpenTokenId, decoded: "<think>",
            primaryEOS: 150001, stopIds: stops(), state: &state)
        #expect(step.thinkReentry)
        #expect(step.yield.isEmpty)
        #expect(!step.isEOS)

        var untouched = FlintChatDecode.ThinkReentryGuard()
        let heldPrefix = untouched.observeFastPacket("Hello<")
        #expect(!heldPrefix)
        #expect(!untouched.hasVisibleContent)
    }

    @Test("stop ids are the checkpoint eos set union the canonical four")
    func testFinishStopIdsFromCheckpoint() throws {
        #expect(FlintChatDecode.parseEosTokenIds(150001) == [150001])
        #expect(FlintChatDecode.finishStopIds(configEOS: [150001]) == Set(FlintChatDecode.canonicalStopIds))
        let both = FlintChatDecode.finishStopIds(
            configJSON: ["eos_token_id": [150001, 150003, 150005, 150018] as [Any]],
            generationJSON: ["eos_token_id": 150001])
        #expect(both == Set([150001, 150003, 150005, 150018]))

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("flint-eos-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{\"eos_token_id\":[150001,150003]}".utf8).write(
            to: dir.appendingPathComponent("config.json"))
        try Data("{\"eos_token_id\":[42]}".utf8).write(
            to: dir.appendingPathComponent("generation_config.json"))
        let loaded = FlintChatDecode.loadFinishStopIds(modelDirectory: dir)
        #expect(loaded == Set([42, 150001, 150003, 150005, 150018]))
        #expect(loaded.contains(150018))
        #expect(!loaded.contains(FlintChatDecode.thinkOpenTokenId))
        #expect(FlintChatDecode.hostFinishStopIds(checkpoint: loaded, tokenizerEOS: 7) == loaded.union([7]))
        #expect(
            FlintChatDecode.hostFinishStopIds(checkpoint: [], tokenizerEOS: nil)
                == Set(FlintChatDecode.canonicalStopIds))
    }

    @Test("custom stop disables grammar as a backstop; the request itself is rejected")
    func testStructuralGrammarOptOut() {
        #expect(
            FlintChatDecode.customStopConflict(stop: ["END"], hasRequestConstraint: false)
                == "Flint grammar cannot be combined with custom stop strings")
        #expect(FlintChatDecode.customStopConflict(stop: [], hasRequestConstraint: false) == nil)
        #expect(FlintChatDecode.customStopConflict(stop: ["END"], hasRequestConstraint: true) == nil)
        #expect(FlintChatDecode.structuralGrammar(
            tools: nil, stop: ["END"], hasRequestConstraint: false) == nil)
        #expect(FlintChatDecode.structuralGrammar(
            tools: nil, stop: [], hasRequestConstraint: false) != nil)
        #expect(FlintChatDecode.structuralGrammar(
            tools: nil, stop: nil, hasRequestConstraint: true) == nil)
        let withTools = FlintChatDecode.structuralGrammar(
            tools: [["type": "function"] as [String: Any]], stop: nil, hasRequestConstraint: false)
        #expect(withTools?.allowsTools == true)
    }

    @Test("after a closed think and a visible answer a second think is masked")
    func testGrammarMasksSecondThink() {
        var grammar = FlintStructuralGrammar(allowsTools: false)
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkOpenId))
        let openedThink = grammar.consume(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>")
        #expect(openedThink)
        #expect(grammar.isAllowed(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>"))
        #expect(!grammar.isAllowed(tokenId: FlintStructuralGrammar.endOfTextId, text: "<|end_of_text|>"))
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.endOfTextId))
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkCloseId))
        let reasoned = grammar.consume(tokenId: 11, text: "reason")
        #expect(reasoned)
        let closedThink = grammar.consume(tokenId: FlintStructuralGrammar.thinkCloseId, text: "</think>")
        #expect(closedThink)
        // Official AnyText excludes `<think>` as soon as the block closes,
        // including when the only text so far is whitespace.
        #expect(grammar.phase == .answer)
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkOpenId))
        let whitespace = grammar.consume(tokenId: 2, text: "\n\n")
        #expect(whitespace)
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkOpenId))
        let hello = grammar.consume(tokenId: 12, text: "Hello")
        #expect(hello)

        #expect(!grammar.isAllowed(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>"))
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkOpenId))
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkCloseId))
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.developerId))
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.endOfTextId))
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.userId))
        #expect(grammar.isAllowed(tokenId: FlintStructuralGrammar.endOfTextId, text: "<|end_of_text|>"))
        #expect(grammar.isAllowed(tokenId: 12, text: "Hello"))
        #expect(!grammar.isAllowed(tokenId: FlintStructuralGrammar.toolOpenId, text: "<tool_call>"))

        #expect(grammar.isAllowed(tokenId: 3, text: "<"))
        let heldOpen = grammar.consume(tokenId: 3, text: "<")
        #expect(heldOpen)
        #expect(!grammar.isAllowed(tokenId: 4, text: "think>"))
        let spelledThink = grammar.consume(tokenId: 4, text: "think>")
        #expect(!spelledThink)
        #expect(grammar.phase == .answer)
    }

    @Test("tool calls stay available and still exclude think tags")
    func testGrammarAllowsOneToolRegion() {
        var grammar = FlintStructuralGrammar(allowsTools: true)
        let hello = grammar.consume(tokenId: 12, text: "Hello")
        #expect(hello)
        #expect(grammar.isAllowed(tokenId: FlintStructuralGrammar.toolOpenId, text: "<tool_call>"))
        #expect(!grammar.isAllowed(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>"))
        let openedTool = grammar.consume(tokenId: FlintStructuralGrammar.toolOpenId, text: "<tool_call>")
        #expect(openedTool)
        #expect(grammar.phase == .inTool)
        #expect(!grammar.isAllowed(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>"))
        #expect(grammar.isAllowed(tokenId: FlintStructuralGrammar.argKeyOpenId, text: "<arg_key>"))
        let argument = grammar.consume(tokenId: 21, text: "search")
        #expect(argument)
        let closedTool = grammar.consume(tokenId: FlintStructuralGrammar.toolCloseId, text: "</tool_call>")
        #expect(closedTool)
        #expect(grammar.phase == .answer)
        #expect(grammar.isAllowed(tokenId: FlintStructuralGrammar.observationId, text: "<|observation|>"))
        #expect(!grammar.maskedTokenIds().contains(FlintStructuralGrammar.observationId))
        #expect(grammar.maskedTokenIds().contains(FlintStructuralGrammar.thinkOpenId))
    }

    @Test("the logits mask drops a second think-open below every answer token")
    func testSuppressThinkOpenLogit() {
        var grammar = FlintStructuralGrammar(allowsTools: false)
        let openedThink = grammar.consume(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>")
        #expect(openedThink)
        let closedThink = grammar.consume(tokenId: FlintStructuralGrammar.thinkCloseId, text: "</think>")
        #expect(closedThink)
        let hello = grammar.consume(tokenId: 12, text: "Hello")
        #expect(hello)

        let vocab = FlintStructuralGrammar.developerId + 1
        var values = [Float](repeating: 0, count: vocab)
        values[FlintStructuralGrammar.thinkOpenId] = 50
        values[12] = 1
        let logits = MLXArray(values, [1, vocab])
        let masked = FlintStructuralGrammar.suppress(grammar.maskedTokenIds(), in: logits)
        let flat = masked.asArray(Float.self)
        #expect(flat[FlintStructuralGrammar.thinkOpenId].isInfinite)
        #expect(flat[FlintStructuralGrammar.thinkOpenId] < 0)
        #expect(flat[12] == 1)
        #expect(flat[FlintStructuralGrammar.endOfTextId] == 0)
        #expect(flat[FlintStructuralGrammar.userId] == 0)
    }

    @Test("grammar rejects a second think; the guard still does when grammar is off")
    func testOpeningDecisionGrammarAndGuard() {
        var grammar: FlintStructuralGrammar? = FlintStructuralGrammar(allowsTools: false)
        var guardState = FlintChatDecode.ThinkReentryGuard()
        let stops = Set(FlintChatDecode.canonicalStopIds)
        let open = FlintChatDecode.thinkOpenTokenId
        let close = FlintChatDecode.thinkCloseTokenId
        func step(_ id: Int, _ text: String) -> FlintChatDecode.TokenDecision {
            FlintChatDecode.openingDecision(
                tokenId: id, yieldText: text, grammarText: text,
                primaryEOS: 150001, stopIds: stops,
                grammar: &grammar, guardState: &guardState)
        }
        #expect(!step(open, "<think>").thinkReentry)
        #expect(!step(11, "reason").thinkReentry)
        #expect(!step(close, "</think>").thinkReentry)
        #expect(step(12, "Hello").yield == "Hello")
        let again = step(open, "<think>")
        #expect(again.thinkReentry)
        #expect(again.yield.isEmpty)

        grammar = nil
        guardState = FlintChatDecode.ThinkReentryGuard()
        #expect(!step(12, "Hello").thinkReentry)
        let guarded = step(open, "<think>")
        #expect(guarded.thinkReentry)
        #expect(guarded.yield.isEmpty)
    }

    @Test("greedy with no grammar stays on the device packet at width 16")
    func testGreedyDevicePacketUnchanged() {
        let greedy = policy()
        #expect(FlintChatDecode.residentRoute(greedy) == .greedyDevice)
        #expect(FlintChatDecode.allowsResidentPacket(greedy))
        #expect(FlintChatDecode.allowsResidentPacket(policy(repetitionPenalty: 1)))
        #expect(FlintRecResident.width == 16)
        #expect(FlintChatDecode.packetStopIds(tokenizerEOS: nil) == FlintChatDecode.canonicalStopIds)
        #expect(!FlintChatDecode.allowsResidentPacket(policy(temperature: 1)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(usesGrammar: true)))
        #expect(!FlintChatDecode.allowsResidentPacket(policy(usesStructuralGrammar: true)))
    }

    @Test("grammar and recommended sampling use the filtered packet")
    func testFilteredPacketRoute() {
        let recommended = policy(
            temperature: 1, topP: 0.95, topK: 20, usesStructuralGrammar: true)
        #expect(FlintChatDecode.residentRoute(recommended) == .filtered)
        #expect(FlintChatDecode.residentRoute(policy(usesStructuralGrammar: true)) == .filtered)
        #expect(FlintChatDecode.residentRoute(policy(temperature: 1, topP: 0.95, topK: 20)) == .filtered)
        #expect(FlintChatDecode.residentRoute(policy(presencePenalty: 0.2)) == .filtered)
        // Official rejects these. The packet stays off; frequency still runs
        // on the single-step fused sampler.
        #expect(FlintChatDecode.residentRoute(
            policy(temperature: 1, topP: 0.95, topK: 20, frequencyPenalty: 0.1, usesStructuralGrammar: true)
        ) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(repetitionPenalty: 1.1)) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(temperature: 1, minP: 0.05)) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(temperature: 1, wantsLogprobs: true)) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(temperature: 1, batchWidth: 2)) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(temperature: 0, usesGrammar: true)) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(temperature: 1, topP: 1.1)) == .singleStep)
        #expect(FlintChatDecode.residentRoute(policy(presencePenalty: 3)) == .singleStep)
    }

    @Test("filtered packet rejects a second think before acceptance")
    func testFilteredPacketRejectsSecondThink() {
        var grammar: FlintStructuralGrammar? = FlintStructuralGrammar(allowsTools: false)
        let open = FlintStructuralGrammar.thinkOpenId
        let close = FlintStructuralGrammar.thinkCloseId
        let script = [open, 11, close, 12, open, 99]
        var index = 0
        let walked = FlintChatDecode.acceptFilteredPacket(
            grammar: &grammar, stopIds: Set(FlintChatDecode.canonicalStopIds)
        ) { _ in
            let id = script[index]
            index += 1
            return id
        }
        #expect(walked.tokens == [open, 11, close, 12])
        #expect(walked.rejectedMaskedId == open)
        #expect(!walked.tokens.contains(99))
        #expect(walked.tokens.filter { $0 == open }.count == 1)
        #expect(grammar?.phase == .answer)
        #expect(grammar?.maskedTokenIds().contains(open) == true)
    }

    @Test("filtered packet still finishes on each grammar-legal eos")
    func testFilteredPacketMultiEOS() {
        let end = FlintStructuralGrammar.endOfTextId
        let user = FlintStructuralGrammar.userId
        let observation = FlintStructuralGrammar.observationId
        let developer = FlintStructuralGrammar.developerId
        let stops = Set(FlintChatDecode.canonicalStopIds)
        #expect(stops.contains(end))
        #expect(stops.contains(user))
        #expect(stops.contains(observation))
        #expect(stops.contains(developer))

        func finish(on id: Int, tools: Bool, prelude: [Int]) -> FlintChatDecode.FilteredPacketResult {
            var grammar: FlintStructuralGrammar? = FlintStructuralGrammar(allowsTools: tools)
            let script = prelude + [id, 99]
            var index = 0
            return FlintChatDecode.acceptFilteredPacket(grammar: &grammar, stopIds: stops) { _ in
                let next = script[index]
                index += 1
                return next
            }
        }

        let endTurn = finish(on: end, tools: false, prelude: [12])
        #expect(endTurn.stoppedOnEOS)
        #expect(endTurn.tokens == [12, end])
        #expect(endTurn.rejectedMaskedId == nil)

        let userTurn = finish(on: user, tools: false, prelude: [
            FlintStructuralGrammar.thinkOpenId, 11, FlintStructuralGrammar.thinkCloseId, 12,
        ])
        #expect(userTurn.stoppedOnEOS)
        #expect(userTurn.tokens.last == user)
        #expect(!userTurn.tokens.contains(99))

        let afterTool = finish(on: observation, tools: true, prelude: [
            12, FlintStructuralGrammar.toolOpenId, 21, FlintStructuralGrammar.toolCloseId,
        ])
        #expect(afterTool.stoppedOnEOS)
        #expect(afterTool.tokens.last == observation)

        // Developer stays in the host stop set, and the answer-phase grammar
        // masks it, so the packet does not accept it as a sampled token.
        let maskedDeveloper = finish(on: developer, tools: false, prelude: [12])
        #expect(maskedDeveloper.rejectedMaskedId == developer)
        #expect(maskedDeveloper.tokens == [12])
        #expect(!maskedDeveloper.stoppedOnEOS)
    }

    @Test("grammar mask drops a second think below the argmax")
    func testFilteredArgmaxSkipsSecondThink() {
        var grammar = FlintStructuralGrammar(allowsTools: false)
        let openedThink = grammar.consume(tokenId: FlintStructuralGrammar.thinkOpenId, text: "<think>")
        #expect(openedThink)
        let closedThink = grammar.consume(tokenId: FlintStructuralGrammar.thinkCloseId, text: "</think>")
        #expect(closedThink)
        let hello = grammar.consume(tokenId: 12, text: "Hello")
        #expect(hello)

        let vocab = FlintStructuralGrammar.developerId + 1
        var values = [Float](repeating: 0, count: vocab)
        values[FlintStructuralGrammar.thinkOpenId] = 50
        values[12] = 1
        let logits = MLXArray(values, [1, vocab])
        let filtered = FlintChatDecode.filterLogits(
            logits,
            grammar: grammar,
            presenceSeen: [],
            presencePenalty: 0,
            temperature: 0,
            topK: 20)
        let flat = filtered.asArray(Float.self)
        // Temperature 0 takes argmax after this mask. The think-open logit
        // is no longer the maximum, so the packet cannot accept a second `<think>`.
        #expect(flat[FlintStructuralGrammar.thinkOpenId].isInfinite)
        #expect(flat[FlintStructuralGrammar.thinkOpenId] < 0)
        #expect(flat[12] == 1)
        #expect(flat[12] > flat[FlintStructuralGrammar.thinkOpenId])
    }

    @Test("top-k keeps the largest logit and presence subtracts once")
    func testTopKAndPresenceMasks() {
        let row = MLXArray([Float(1), 3, 2], [1, 3])
        let top = FlintChatDecode.applyTopK(row, k: 1).asArray(Float.self)
        #expect(top[1] == 3)
        #expect(top[0].isInfinite && top[0] < 0)
        #expect(top[2].isInfinite && top[2] < 0)

        let present = FlintChatDecode.applyPresence(
            MLXArray([Float(1), 1, 1], [3]), seen: [1], penalty: 0.5
        ).asArray(Float.self)
        #expect(present[0] == 1)
        #expect(present[1] == 0.5)
        #expect(present[2] == 1)
        #expect(FlintChatDecode.applyTopK(row, k: 0).shape == row.shape)
    }
}
