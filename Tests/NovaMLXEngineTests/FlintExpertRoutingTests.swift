import Foundation
import MLX
import Testing

@testable import NovaMLXEngine

@Suite("Flint expert routing")
struct FlintExpertRoutingTests {

    private func sigmoid(_ x: Float) -> Float {
        let tail = 1 / (1 + exp(abs(x)))
        return x < 0 ? tail : 1 - tail
    }

    @Test("rank routing prefers the lower index on a tie and renorms with eps")
    func testReferenceTies() {
        let logits = [Float](repeating: 0, count: 8)
        let tied = FlintExpertRouting.reference(
            logits: logits, bias: [Float](repeating: 0, count: 8), topK: 3)
        #expect(tied.map(\.index) == [0, 1, 2])
        let sum = tied.reduce(Float(0)) { $0 + $1.weight }
        #expect(abs(sum - 1) < Float(1e-5))

        var bias = [Float](repeating: 0, count: 8)
        bias[5] = 10
        bias[2] = 10
        let picked = FlintExpertRouting.reference(logits: logits, bias: bias, topK: 2)
        #expect(picked.map(\.index) == [2, 5])
    }

    @Test("expert bias changes who is selected and stays out of the mixture weight")
    func testBiasIsSelectionOnly() {
        let logits: [Float] = [0, 2, -2, 4, 1, -1]
        let plain = FlintExpertRouting.reference(
            logits: logits, bias: [Float](repeating: 0, count: 6), topK: 2)
        var bias = [Float](repeating: 0, count: 6)
        bias[5] = 50
        let biased = FlintExpertRouting.reference(logits: logits, bias: bias, topK: 2)
        #expect(biased[0].index == 5)
        #expect(!plain.map(\.index).contains(5))
        let scores = biased.map { sigmoid(logits[$0.index]) }
        let denom = scores.reduce(Float(0), +) + FlintExpertRouting.scoreEps
        for (choice, score) in zip(biased, scores) {
            #expect(abs(choice.weight - score / denom) < Float(1e-6))
        }
    }

    /// Mac Metal. The kernel is mlx-lm `expert_routing` with rec's `1e-20` renorm.
    @Test("fused kernel matches the rank reference")
    func testKernelMatchesReference() {
        let rows = [
            (0 ..< 16).map { Float($0) - 8 },
            [Float](repeating: 0.25, count: 16),
        ]
        var bias = [Float](repeating: 0, count: 16)
        bias[1] = 3
        bias[14] = -2
        let flat = rows.flatMap { $0 }
        let logits = MLXArray(flat, [2, 1, 16])
        let (inds, weights) = FlintExpertRouting.route(
            logits: logits, bias: MLXArray(bias, [16]), topK: 4)
        eval(inds, weights)
        let gotI = inds.asArray(Int32.self)
        let gotW = weights.asArray(Float.self)
        #expect(gotI.count == 8)
        for row in 0 ..< 2 {
            let ref = FlintExpertRouting.reference(logits: rows[row], bias: bias, topK: 4)
            for slot in 0 ..< 4 {
                let at = row * 4 + slot
                #expect(Int(gotI[at]) == ref[slot].index)
                #expect(abs(gotW[at] - ref[slot].weight) < Float(1e-4))
            }
        }
    }
}
