import Foundation
import MLX
import MLXNN

// MARK: - Block check

/// One-wait length check. A mismatch drops the drafted block and keeps the anchor.
public struct DFlashBlockCheck: Sendable, Equatable {
    public let acceptNormally: Bool
    public let specWidth: Int
    public let nRejected: Int

    public init(acceptNormally: Bool, specWidth: Int, nRejected: Int) {
        self.acceptNormally = acceptNormally
        self.specWidth = specWidth
        self.nRejected = nRejected
    }

    public static func decide(
        draftCount: Int, logitLength: Int, fusedLength: Int, accepted: Int
    ) -> DFlashBlockCheck {
        let width = draftCount + 1
        let lengthsAgree = logitLength == fusedLength && fusedLength == width
        let acceptedInRange = (0...draftCount).contains(accepted)
        if lengthsAgree && acceptedInRange {
            return DFlashBlockCheck(
                acceptNormally: true, specWidth: width, nRejected: draftCount - accepted)
        }
        return DFlashBlockCheck(acceptNormally: false, specWidth: width, nRejected: draftCount)
    }
}

// MARK: - Cap

/// Shrinks the draft width when acceptance is poor, and parks at zero until a strong probe.
public struct DFlashCapPolicy: Sendable {
    public let maxCap: Int
    public let windowSize: Int
    public let parkBelow: Int
    public let minRoundsBeforePark: Int
    public let probeEvery: Int
    public private(set) var cap: Int

    private var window: [Int] = []
    private var rounds = 0
    private var parked = false
    private var parkedRounds = 0

    public init(
        maxCap: Int,
        windowSize: Int = 4,
        parkBelow: Int = 3,
        minRoundsBeforePark: Int = 3,
        probeEvery: Int = 4
    ) {
        self.maxCap = maxCap
        self.windowSize = max(windowSize, 1)
        self.parkBelow = parkBelow
        self.minRoundsBeforePark = minRoundsBeforePark
        self.probeEvery = max(probeEvery, 1)
        self.cap = maxCap
    }

    /// True on a parked round that should try `maxCap` once. `cap` stays 0 until that probe accepts.
    public var shouldProbe: Bool {
        parked && parkedRounds > 0 && parkedRounds.isMultiple(of: probeEvery)
    }

    public mutating func observe(accepted: Int, capUsed: Int) {
        _ = capUsed
        rounds += 1
        window.append(accepted)
        if window.count > windowSize { window.removeFirst() }

        if parked {
            parkedRounds += 1
            if accepted >= parkBelow {
                parked = false
                parkedRounds = 0
                cap = maxCap
            } else {
                cap = 0
            }
            return
        }

        guard rounds >= minRoundsBeforePark else {
            cap = maxCap
            return
        }
        let mean = Double(window.reduce(0, +)) / Double(window.count)
        if mean < Double(parkBelow) {
            parked = true
            parkedRounds = 0
            cap = 0
            return
        }
        let peak = window.max() ?? 0
        cap = min(maxCap, peak + 1)
    }
}

// MARK: - Path selector

/// Low-rank candidate selector. `walkGreedy` follows the predecessor chain on the GPU.
public final class DFlashCandidateSelector: Module {
    public let hiddenSize: Int
    public let vocabSize: Int
    public let rank: Int
    public let topK: Int
    public var predecessorCodebook: MLXArray
    public var successorCodebook: MLXArray
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    public init(hiddenSize: Int, vocabSize: Int, rank: Int, topK: Int) {
        self.hiddenSize = hiddenSize
        self.vocabSize = max(vocabSize, 1)
        self.rank = max(rank, 1)
        self.topK = max(topK, 1)
        self.predecessorCodebook = MLXArray.zeros([self.vocabSize, self.rank])
        self.successorCodebook = MLXArray.zeros([self.vocabSize, self.rank])
        _hiddenProjection.wrappedValue = Linear(hiddenSize, self.rank, bias: false)
        super.init()
    }

    /// Greedy path through `scores[slot, predecessor, candidate]`.
    /// Slot 0 reads row 0. Each later slot reads the row of the previous winner.
    /// The index stays an `MLXArray` until the caller evaluates the returned ids.
    public func walkGreedy(scores: MLXArray, candidateIds: MLXArray) -> MLXArray {
        let slots = scores.dim(0)
        var index = argMax(scores[0, 0], axis: -1)
        var picks: [MLXArray] = [take(candidateIds[0], index, axis: 0)]
        if slots <= 1 {
            return MLX.stacked(picks, axis: 0)
        }
        for slot in 1..<slots {
            let row = take(scores[slot], index, axis: 0)
            index = argMax(row, axis: -1)
            picks.append(take(candidateIds[slot], index, axis: 0))
        }
        return MLX.stacked(picks, axis: 0)
    }

    /// One greedy block from the anchor logits and the (already convolved) hidden state.
    /// Candidates are the anchor row's top-k. Transitions use the codebooks.
    public func propose(
        anchorLogits: MLXArray, hidden: MLXArray, cap: Int
    ) -> MLXArray {
        let width = max(cap, 1)
        let row = anchorLogits.ndim >= 2 ? anchorLogits[0..., -1, 0...] : anchorLogits
        let flat = row.ndim > 1 ? row.reshaped(-1) : row
        let vocab = flat.dim(0)
        let k = min(topK, max(vocab, 1))
        let ids = argPartition(-flat, kth: k - 1, axis: -1)[0..<k]
        let unary = take(flat, ids, axis: 0)
        let hiddenRow: MLXArray
        if hidden.ndim >= 2 {
            hiddenRow = hidden[0..., -1, 0...].reshaped(-1)
        } else {
            hiddenRow = hidden.reshaped(-1)
        }
        let projected = hiddenProjection(hiddenRow.reshaped(1, -1)).reshaped(-1)
        let pred = take(predecessorCodebook, ids, axis: 0)
        let succ = take(successorCodebook, ids, axis: 0)
        var transition = pred.matmul(succ.swappedAxes(-1, -2))
        transition = transition + projected.mean()
        let unaryRow = unary.reshaped([1, k])
        let slot0 = repeated(unaryRow, count: k, axis: 0)
        let later = transition + unaryRow
        var blocks: [MLXArray] = [slot0.reshaped([1, k, k])]
        if width > 1 {
            let rest = repeated(later.reshaped([1, k, k]), count: width - 1, axis: 0)
            blocks.append(rest)
        }
        let scores = concatenated(blocks, axis: 0)
        let candidates = repeated(ids.reshaped([1, k]), count: width, axis: 0)
        return walkGreedy(scores: scores, candidateIds: candidates)
    }
}

// MARK: - Local grouped conv

/// Grouped dynamic depthwise convolution across one DFlash block.
/// `prepare` returns the convolved input and the output-side coefficients `[B, L, taps, groups]`.
public final class DFlashGroupedConv: Module {
    public let hiddenSize: Int
    public let taps: Int
    public let groupSize: Int
    public let numGroups: Int
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear
    /// `[taps, hidden]`. Tap 0 starts as identity.
    public var baseKernel: MLXArray

    public init(hiddenSize: Int, taps: Int, groupSize: Int) {
        precondition(groupSize > 0 && hiddenSize % groupSize == 0, "conv group must divide hidden")
        self.hiddenSize = hiddenSize
        self.taps = max(taps, 1)
        self.groupSize = groupSize
        self.numGroups = hiddenSize / groupSize
        let out = 2 * self.taps * self.numGroups
        _kernelProjection.wrappedValue = Linear(hiddenSize, out, bias: false)
        let identity = MLXArray.ones([1, hiddenSize])
        if self.taps == 1 {
            self.baseKernel = identity
        } else {
            let rest = MLXArray.zeros([self.taps - 1, hiddenSize])
            self.baseKernel = concatenated([identity, rest], axis: 0)
        }
        super.init()
    }

    public func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let projected = kernelProjection(hidden)
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let coeff = projected.reshaped([batch, length, 2, taps, numGroups])
        let inputDelta = coeff[0..., 0..., 0]
        let outCoeff = coeff[0..., 0..., 1]
        return (convolve(hidden, delta: inputDelta), outCoeff)
    }

    /// Causal grouped mix. `delta` is `[B, L, taps, groups]`.
    public func convolve(_ hidden: MLXArray, delta: MLXArray) -> MLXArray {
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let hiddenWidth = hidden.dim(2)
        var mixed = MLXArray.zeros([batch, length, hiddenWidth])
        for tap in 0..<taps {
            let source = shiftCausal(hidden, by: tap)
            let grouped = source.reshaped([batch, length, numGroups, groupSize])
            let scale = delta[0..., 0..., tap, 0...].reshaped([batch, length, numGroups, 1])
            let base = baseKernel[tap].reshaped([1, 1, numGroups, groupSize])
            let local = (grouped * (base + scale)).reshaped([batch, length, hiddenWidth])
            mixed = mixed + local
        }
        return mixed
    }

    private func shiftCausal(_ hidden: MLXArray, by tap: Int) -> MLXArray {
        if tap == 0 { return hidden }
        let batch = hidden.dim(0)
        let length = hidden.dim(1)
        let width = hidden.dim(2)
        if tap >= length { return MLXArray.zeros([batch, length, width]) }
        let pad = MLXArray.zeros([batch, tap, width])
        let tail = hidden[0..., ..<(length - tap), 0...]
        return concatenated([pad, tail], axis: 1)
    }
}
