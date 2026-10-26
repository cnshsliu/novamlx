import Foundation
import Testing
import NovaMLXCore

@testable import NovaMLXEngine

@Suite("Flint chat decode policy")
struct FlintChatDecodeTests {

    private func policy(
        temperature: Float = 0,
        frequencyPenalty: Float = 0,
        repetitionPenalty: Float = 0,
        presencePenalty: Float = 0,
        wantsLogprobs: Bool = false,
        batchWidth: Int = 1,
        usesGrammar: Bool = false
    ) -> FlintChatDecode.Policy {
        FlintChatDecode.Policy(
            temperature: temperature,
            frequencyPenalty: frequencyPenalty,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            wantsLogprobs: wantsLogprobs,
            batchWidth: batchWidth,
            usesGrammar: usesGrammar)
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
        #expect(state.observeFastPacket("Hello! How can I help you today?"))
        let step = FlintChatDecode.tokenDecision(
            tokenId: FlintChatDecode.thinkOpenTokenId, decoded: "<think>",
            primaryEOS: 150001, stopIds: stops(), state: &state)
        #expect(step.thinkReentry)
        #expect(step.yield.isEmpty)
        #expect(!step.isEOS)

        var untouched = FlintChatDecode.ThinkReentryGuard()
        #expect(!untouched.observeFastPacket("Hello<"))
        #expect(!untouched.hasVisibleContent)
    }
}
