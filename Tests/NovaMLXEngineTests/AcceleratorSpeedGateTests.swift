import Foundation
import MLX
import Testing
@testable import NovaMLXEngine

@Suite("Accelerator speed gate")
struct AcceleratorSpeedGateTests {
    @Test("Known-faster DSpark is auto_on and unknown companions are ask_user")
    func autoOnVersusAsk() {
        let eight = AcceleratorSpeedGate.decision(
            mainModelId: "mlx-community/Qwen3-8B-8bit",
            acceleratorId: "deepseek-ai/dspark_qwen3_8b_block7",
            kind: .dspark,
            scenario: .chatShort
        )
        #expect(eight.decision == .autoOn)

        let redhat = AcceleratorSpeedGate.decision(
            mainModelId: "mlx-community/Qwen3.8-27B-8bit",
            acceleratorId: "RedHatAI/Qwen3.8-27B-speculator.dspark",
            kind: .dspark,
            scenario: .unspecified
        )
        #expect(redhat.decision == .askUser)

        let missing = AcceleratorSpeedGate.decision(
            mainModelId: "org/never-profiled",
            acceleratorId: "org/never-profiled-draft",
            kind: .smallDraft,
            scenario: .unspecified
        )
        #expect(missing.decision == .askUser)
    }

    @Test("Short Qwen3.8 DFlash2 chat is auto_off; load time asks")
    func dflashShortIsOff() {
        let short = AcceleratorSpeedGate.decision(
            mainModelId: "mlx-community/Qwen3.8-27B-4bit",
            acceleratorId: "incoai/Qwen3.8-27B-DFlash2",
            kind: .dflash2,
            scenario: .chatShort
        )
        #expect(short.decision == .autoOff)

        let load = AcceleratorSpeedGate.decision(
            mainModelId: "orcarouter/Qwen3.8-27B-Uncensored-MLX",
            acceleratorId: "incoai/Qwen3.8-27B-DFlash2",
            kind: .dflash2,
            scenario: .unspecified
        )
        #expect(load.decision == .askUser)

        let long = AcceleratorSpeedGate.decision(
            mainModelId: "mlx-community/Qwen3.8-27B-8bit",
            acceleratorId: "incoai/Qwen3.8-27B-DFlash2",
            kind: .dflash2,
            scenario: .chatLong
        )
        #expect(long.decision == .askUser)
    }

    @Test("A saved yes does not override auto_off, and a saved no skips the prompt")
    func memoryRespectsOff() {
        let dflash = AcceleratorCandidate(
            kind: .dflash2,
            acceleratorId: "incoai/Qwen3.8-27B-DFlash2",
            displayName: "DFlash2",
            estimatedSizeMB: 1500,
            family: .qwen
        )
        let forced = AcceleratorSpeedGate.plan(
            mainModelId: "mlx-community/Qwen3.8-27B-8bit",
            scenario: .chatShort,
            candidates: [dflash],
            remembered: ["incoai/Qwen3.8-27B-DFlash2": true]
        )
        #expect(forced.attach.isEmpty)
        #expect(forced.ask.isEmpty)

        let declined = AcceleratorSpeedGate.plan(
            mainModelId: "mlx-community/Qwen3.8-27B-8bit",
            scenario: .unspecified,
            candidates: [dflash],
            remembered: ["incoai/Qwen3.8-27B-DFlash2": false]
        )
        #expect(declined.attach.isEmpty)
        #expect(declined.ask.isEmpty)
        #expect(declined.skipped.count == 1)
    }

    @Test("Unknown boosters are asked, and a higher auto_on companion hides lower asks")
    func askVersusAutoPlan() {
        let registry = DraftModelRegistry.shared
        let eight = registry.acceleratorCandidates(
            forMainId: "mlx-community/Qwen3-8B-8bit",
            family: .qwen,
            isHybrid: false,
            modelType: .llm
        )
        let plan = AcceleratorSpeedGate.plan(
            mainModelId: "mlx-community/Qwen3-8B-8bit",
            candidates: eight
        )
        #expect(plan.attach.map(\.acceleratorId) == ["deepseek-ai/dspark_qwen3_8b_block7"])
        #expect(plan.ask.isEmpty)

        let qwen38 = registry.acceleratorCandidates(
            forMainId: "mlx-community/Qwen3.8-27B-8bit",
            family: .qwen,
            isHybrid: false,
            modelType: .llm
        )
        let asked = AcceleratorSpeedGate.plan(
            mainModelId: "mlx-community/Qwen3.8-27B-8bit",
            candidates: qwen38
        )
        #expect(asked.attach.isEmpty)
        let askedIds = Set(asked.ask.map(\.candidate.acceleratorId))
        #expect(askedIds.contains("incoai/Qwen3.8-27B-DFlash2"))
        #expect(askedIds.contains("RedHatAI/Qwen3.8-27B-speculator.dspark"))
        #expect(askedIds.contains("mlx-community/Qwen3.8-27B-MTP-4bit"))
    }

    @Test("Enabled companions keep DFlash2, then MTP, then DSpark, then a small draft")
    func preferenceOrder() {
        let ids = [
            ("small", AcceleratorKind.smallDraft, "mlx-community/Qwen3-0.6B-4bit"),
            ("dspark", AcceleratorKind.dspark, "RedHatAI/Qwen3.8-27B-speculator.dspark"),
            ("mtp", AcceleratorKind.mtp, "mlx-community/Qwen3.8-27B-MTP-4bit"),
            ("dflash", AcceleratorKind.dflash2, "incoai/Qwen3.8-27B-DFlash2"),
        ]
        let candidates = ids.map {
            AcceleratorCandidate(
                kind: $0.1, acceleratorId: $0.2, displayName: $0.0,
                estimatedSizeMB: 1, family: .qwen
            )
        }
        let remembered = Dictionary(uniqueKeysWithValues: candidates.map { ($0.acceleratorId, true) })
        let plan = AcceleratorSpeedGate.plan(
            mainModelId: "mlx-community/Qwen3.8-27B-8bit",
            scenario: .unspecified,
            candidates: candidates,
            remembered: remembered
        )
        #expect(plan.attach.map(\.kind) == [.dflash2, .mtp, .dspark, .smallDraft])
        #expect(plan.ask.isEmpty)
    }

    @Test("A fresh DFlash draft has no speculative weights until the path is installed")
    func weightsGateTheIterator() throws {
        let json = """
            {"hidden_size":8,"vocab_size":32,"num_hidden_layers":1,"num_key_value_heads":1,
             "dflash_config":{"block_size":4,"selector_rank":4,"selector_top_k":2,"target_layer_ids":[0]}}
            """.data(using: .utf8)!
        let model = DFlashDraftModel(try DFlashConfig.fromJSON(json))
        #expect(!model.hasSpeculativeWeights)
        model.installSpeculativePath(
            selector: DFlashCandidateSelector(hiddenSize: 8, vocabSize: 32, rank: 4, topK: 2),
            conv: DFlashGroupedConv(hiddenSize: 8, taps: 2, groupSize: 4)
        )
        #expect(model.hasSpeculativeWeights)
        #expect(model.pathSelector != nil)
        #expect(model.localConv != nil)
    }

    @Test("Greedy selector walk follows the chain")
    func walkGreedyFollowsChain() {
        let sel = DFlashCandidateSelector(hiddenSize: 8, vocabSize: 32, rank: 4, topK: 3)
        let cand = MLXArray([Int32(10), 11, 12, 20, 21, 22, 30, 31, 32], [3, 3])
        let scores = MLXArray(
            [
                Float(0), 1, 5, 0, 1, 5, 0, 1, 5,
                0, 9, 0, 0, 0, 0, 7, 1, 0,
                0, 4, 1, 9, 0, 0, 0, 0, 9,
            ], [3, 3, 3])
        let picks = sel.walkGreedy(scores: scores, candidateIds: cand)
        eval(picks)
        #expect(picks.asArray(Int.self) == [12, 20, 31])
    }

    @Test("Grouped conv prepare keeps the documented shapes")
    func groupedConvShape() {
        let conv = DFlashGroupedConv(hiddenSize: 64, taps: 2, groupSize: 16)
        let x = MLXArray.zeros([1, 8, 64])
        let (convolved, outCoeff) = conv.prepare(x)
        eval(convolved, outCoeff)
        #expect(convolved.shape == [1, 8, 64])
        #expect(outCoeff.shape == [1, 8, 2, 4])
    }

    @Test("Block check matches the one-wait contract")
    func blockCheck() {
        let ok = DFlashBlockCheck.decide(draftCount: 4, logitLength: 5, fusedLength: 5, accepted: 2)
        #expect(ok.acceptNormally)
        #expect(ok.specWidth == 5)
        #expect(ok.nRejected == 2)
        let bad = DFlashBlockCheck.decide(draftCount: 4, logitLength: 4, fusedLength: 5, accepted: 2)
        #expect(!bad.acceptNormally)
        #expect(bad.nRejected == 4)
    }
}
