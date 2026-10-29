import Foundation

/// Host numeric spec for one DASH decode query. Output length is the constant
/// `topK * block + local` window (512 on the released config).
///
/// The GPU plan is `UnlimitedOCRDashMath.decodePlan` (shaped compile, not
/// shapeless). A one-thread Metal copy of this spec measured the same decode
/// speed on MacMaxPro and is not launched.
enum UnlimitedOCRDashPlan {
    struct Spec: Equatable, Sendable {
        var block: Int
        var localBlocks: Int
        var local: Int
        var topM: Int
        var topK: Int
        var indexDim: Int
        var indexScale: Float
        var floor: Float
        var sigma: Float

        var attended: Int { topK * block + local }

        static func make(_ params: UnlimitedOCRDashParams, indexScale: Float, floor: Float) -> Spec {
            Spec(
                block: params.blockSize,
                localBlocks: params.localBlocks,
                local: params.localWindow,
                topM: params.topM,
                topK: params.topK,
                indexDim: params.indexDim,
                indexScale: indexScale,
                floor: floor,
                sigma: params.sigma)
        }
    }

    struct Result: Equatable {
        var indices: [Int32]
        var mask: [Float]
    }

    static let scoreClamp: Float = 1e4
    static let entmaxFill: Float = -1e4

    /// Host spec. `candidates[i] != 0` means the block is eligible.
    static func reference(
        query: [Float],
        summaries: [[Float]],
        candidates: [Int],
        position: Int,
        spec: Spec
    ) -> Result {
        let cap = summaries.count
        let mCount = min(spec.topM, cap)
        let inv = 1 / Float(spec.indexDim).squareRoot()
        var scores = [Float](repeating: 0, count: cap)
        var remote = [Bool](repeating: false, count: cap)
        let remoteMax = position / spec.block - spec.localBlocks
        for i in 0..<cap {
            var dot: Float = 0
            let row = summaries[i]
            for d in 0..<spec.indexDim {
                dot += query[d] * row[d]
            }
            scores[i] = min(max(dot * inv, -scoreClamp), scoreClamp)
            remote[i] = i <= remoteMax
        }
        var selected = remote
        for i in 0..<cap {
            selected[i] = remote[i] && candidates[i] != 0
        }
        if !selected.contains(true) {
            selected = remote
        }
        var masked = [Float](repeating: entmaxFill, count: cap)
        for i in 0..<cap where selected[i] {
            masked[i] = scores[i] * spec.indexScale
        }
        var order = Array(0..<cap)
        order.sort { a, b in
            if masked[a] != masked[b] { return masked[a] > masked[b] }
            return a < b
        }
        let chosenOrder = Array(order.prefix(mCount))
        var taken = [Float](repeating: 0, count: mCount)
        for j in 0..<mCount {
            taken[j] = masked[chosenOrder[j]]
        }
        let prior = UnlimitedOCRDashMath.entmax15(taken)
        let k = min(spec.topK, mCount)
        var valid = [Bool](repeating: false, count: k)
        var logp = [Float](repeating: 0, count: k)
        var bias = [Float](repeating: 0, count: k)
        var topi = [Int](repeating: 0, count: k)
        for j in 0..<k {
            topi[j] = chosenOrder[j]
            valid[j] = selected[topi[j]]
            logp[j] = log(max(prior[j], spec.floor))
        }
        var count: Float = 0
        var sumLog: Float = 0
        for j in 0..<k where valid[j] {
            count += 1
            sumLog += logp[j]
        }
        if count < 1 { count = 1 }
        let mu = sumLog / count
        let invSigma = 1 / spec.sigma
        for j in 0..<k {
            bias[j] = valid[j] ? (logp[j] - mu) * invSigma : 0
        }
        var indices = [Int32](repeating: 0, count: spec.attended)
        var mask = [Float](repeating: -.infinity, count: spec.attended)
        let limit = position - spec.local
        for j in 0..<k {
            for t in 0..<spec.block {
                let blockIndex = topi[j] * spec.block + t
                let slot = j * spec.block + t
                let ok = valid[j] && blockIndex <= limit
                mask[slot] = ok ? bias[j] : -.infinity
                indices[slot] = Int32(min(blockIndex, position))
            }
        }
        let base = k * spec.block
        for t in 0..<spec.local {
            let localIndex = position - spec.local + 1 + t
            mask[base + t] = localIndex >= 0 ? 0 : -.infinity
            indices[base + t] = Int32(max(localIndex, 0))
        }
        return Result(indices: indices, mask: mask)
    }
}
