import Foundation
import Testing

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
}
