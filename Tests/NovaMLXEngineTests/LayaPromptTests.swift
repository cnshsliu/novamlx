import Testing
@testable import NovaMLXEngine

private struct FakeTok: LayaTokenizing {
    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }
    let clsTokenId = 1
    let sepTokenId = 2
    let padTokenId = 0
    let maskTokenId = 3
    let maskToken = "[MASK]"
}

@Suite("Laya prompts")
struct LayaPromptTests {
    @Test("choice sequence places a marker on each option")
    func choiceMarkers() {
        let q = LayaQuestionSpec(
            kind: .choice,
            instructions: "Which team?",
            choiceLabels: ["billing", "technical"]
        )
        let seq = LayaPrompt.buildSequence(
            tokenizer: FakeTok(), state: "refund please", question: q, maxLen: 512, headMaxLen: 192
        )
        #expect(seq.markers.count == 2)
        #expect(seq.ids.first == 1)
        #expect(seq.qtype == 0)
        #expect(seq.ids.contains(3))
    }

    @Test("noul options are false then true")
    func noulOrder() {
        let q = LayaQuestionSpec(kind: .noul, instructions: "Is this a refund?")
        let options = LayaPrompt.renderOptions(q)
        #expect(options[0].hasPrefix("false:"))
        #expect(options[1].hasPrefix("true:"))
    }

    @Test("score levels stay ordered")
    func scoreLevels() {
        let q = LayaQuestionSpec(
            kind: .score, instructions: "How urgent?", scoreLevels: ["low", "high"]
        )
        #expect(LayaPrompt.renderOptions(q) == ["level 0: low", "level 1: high"])
    }

    @Test("confidence is 1 for a one-hot distribution")
    func confidence() {
        let c = LayaPrompt.confidence(from: [1, 0])
        #expect(abs(c - 1) < 0.0001)
        #expect(LayaPrompt.tempBucket(qtype: 0, optionCount: 3) == "choice:3-5")
    }

    @Test("question JSON accepts a label list")
    func parseQuestions() throws {
        let raw: [String: Any] = [
            "department": [
                "type": "choice",
                "instructions": "Who?",
                "criteria": ["billing", "technical"],
            ],
            "refund": [
                "type": "noul",
                "instructions": "Money back?",
            ],
        ]
        let qs = try DecisionService.questions(from: raw)
        #expect(qs["department"]?.choiceLabels == ["billing", "technical"])
        #expect(qs["refund"]?.kind == .noul)
    }
}
