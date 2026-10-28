import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NovaMLXUtils

/// Seq=1 decode GEMM fuse for Unlimited OCR.
///
/// One token was a long chain of tiny matrix multiplies: Q, K, and V; the
/// DASH router Q and K; expert gate and up; and the dense / shared gate and up.
/// Each pair or triple shares K, so stacking the output rows and calling
/// `quantizedMatmul` or `gatherQuantizedMM` once is the same product as the
/// separate calls. Affine rows stay byte-identical: quantization is per output
/// row, and the stacked weight is those rows concatenated.
///
/// This is not an `mlx.compile` of the MoE, and it is not the one-thread DASH
/// plan kernel. The 512-key gather, SDPA, block plan, expert down projection,
/// attention output projection, and `lm_head` stay on their existing kernels.
/// `NOVA_UOCR_DECODE_FUSE=0` keeps every one of those separate calls.
enum UnlimitedOCRDecodeFuse {
    struct GemmBudget: Equatable {
        var qkv: Int
        var attentionOut: Int
        var router: Int
        var experts: Int
        var shared: Int
        var dense: Int
        var routerScore: Int
        var lmHead: Int

        var total: Int {
            qkv + attentionOut + router + experts + shared + dense + routerScore + lmHead
        }
    }

    /// GEMM launches for one decode token. `fused` stacks QKV, router QK, and
    /// each gate+up. Output projection, expert down, the score linear, and
    /// `lm_head` stay one launch each.
    static func gemmBudget(layers: Int, denseLayers: Int, fused: Bool) -> GemmBudget {
        let moe = max(0, layers - denseLayers)
        let qkv = fused ? layers : layers * 3
        let router = fused ? layers : layers * 2
        let experts = fused ? moe * 2 : moe * 3
        let shared = fused ? moe * 2 : moe * 3
        let dense = fused ? denseLayers * 2 : denseLayers * 3
        return GemmBudget(
            qkv: qkv,
            attentionOut: layers,
            router: router,
            experts: experts,
            shared: shared,
            dense: dense,
            routerScore: moe,
            lmHead: 1)
    }

    static func allowsFuse(_ raw: String?) -> Bool {
        guard let raw else { return true }
        switch raw.lowercased() {
        case "0", "off", "false", "no":
            return false
        default:
            return true
        }
    }

    static let enabled = allowsFuse(ProcessInfo.processInfo.environment["NOVA_UOCR_DECODE_FUSE"])

    /// Bits and group size for a uint32 row of `packed` columns and `groups`
    /// scale columns. `nil` when it is not affine 4-bit or 8-bit with group
    /// 32, 64, or 128.
    static func affineBits(packed: Int, groups: Int) -> (bits: Int, group: Int)? {
        guard packed > 0, groups > 0 else { return nil }
        var matches: [(bits: Int, group: Int)] = []
        for bits in [4, 8] {
            let k = packed * (32 / bits)
            guard k % groups == 0 else { continue }
            let group = k / groups
            guard group == 32 || group == 64 || group == 128 else { continue }
            matches.append((bits, group))
        }
        // `quantize` uses group 64. Prefer that when 4-bit and 8-bit both fit.
        if let grouped = matches.first(where: { $0.group == 64 }) { return grouped }
        return matches.first
    }

    /// Dots of `x` with each row. Stacking rows is the identity the GPU fuse uses.
    static func referenceConcatDots(_ x: [Float], rows: [[Float]]) -> [Float] {
        rows.map { row in
            var sum: Float = 0
            let n = min(x.count, row.count)
            for i in 0..<n {
                sum += x[i] * row[i]
            }
            return sum
        }
    }

    /// Shape of one expert assignment after `SwitchGLU`'s expand, gather, and
    /// squeeze of the singleton M axis. `x` is `[batch, seq, hidden]`.
    static func expertRowShape(x: [Int], indices: [Int], hidden: Int) -> [Int]? {
        guard x.count >= 2, hidden > 0 else { return nil }
        let expanded = expand(x)
        let leading = Array(expanded.dropLast(2))
        let broadcast = broadcastLeading(leading, indices)
        var gathered = broadcast
        gathered.append(expanded[expanded.count - 2])
        gathered.append(hidden)
        guard gathered.count >= 2 else { return nil }
        gathered.remove(at: gathered.count - 2)
        return gathered
    }

    static func project(_ q: Linear, _ k: Linear, _ v: Linear, _ x: MLXArray) -> (
        MLXArray, MLXArray, MLXArray
    )? {
        guard enabled, isSingleToken(x) else { return nil }
        guard let packed = Cache.shared.triple(q, k, v) else { return nil }
        let y = apply(packed, x)
        let parts = split(y, widths: [packed.widths[0], packed.widths[1], packed.widths[2]])
        guard parts.count == 3 else { return nil }
        note()
        return (parts[0], parts[1], parts[2])
    }

    static func project(_ a: Linear, _ b: Linear, _ x: MLXArray) -> (MLXArray, MLXArray)? {
        guard enabled, isSingleToken(x) else { return nil }
        guard let packed = Cache.shared.pair(a, b) else { return nil }
        let y = apply(packed, x)
        let parts = split(y, widths: [packed.widths[0], packed.widths[1]])
        guard parts.count == 2 else { return nil }
        note()
        return (parts[0], parts[1])
    }

    /// Selected experts, gate and up in one `gatherQuantizedMM` (or `gatherMM`
    /// when the stack is still dense). Down stays its own gather. `nil` sends
    /// the caller back through `SwitchGLU`, including any sort of 64 or more
    /// assignments.
    static func projectExperts(
        _ glu: SwitchGLU, _ x: MLXArray, _ indices: MLXArray, assignments: Int
    ) -> MLXArray? {
        // One token picks top-6. `SwitchGLU` sorts only at 64 assignments or
        // more; this path does not, and it does not read `indices.size`.
        guard enabled, isSingleToken(x), assignments > 0, assignments < 64 else { return nil }
        guard let pair = Cache.shared.experts(glu) else { return nil }
        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let both = gather(expanded, pair.gateUp, indices)
        let hidden = pair.gateUp.widths[0]
        let gate = both[.ellipsis, ..<hidden]
        let up = both[.ellipsis, hidden...]
        let activated = compiledSiluProduct(gate, up)
        let down = gather(activated, pair.down, indices)
        note()
        return MLX.squeezed(down, axis: -2)
    }

    // MARK: - Host shape

    private static func expand(_ shape: [Int]) -> [Int] {
        let outRank = shape.count + 2
        let inserted = [-2, -3].map { $0 + outRank }.sorted()
        var result = [Int]()
        var source = 0
        for index in 0..<outRank {
            if inserted.contains(index) {
                result.append(1)
            } else {
                result.append(shape[source])
                source += 1
            }
        }
        return result
    }

    private static func broadcastLeading(_ lhs: [Int], _ rhs: [Int]) -> [Int] {
        let n = max(lhs.count, rhs.count)
        var out = [Int]()
        for index in 0..<n {
            let a = index < n - lhs.count ? 1 : lhs[index - (n - lhs.count)]
            let b = index < n - rhs.count ? 1 : rhs[index - (n - rhs.count)]
            out.append(max(a, b))
        }
        return out
    }

    private static func isSingleToken(_ x: MLXArray) -> Bool {
        x.ndim == 3 && x.dim(0) == 1 && x.dim(1) == 1
    }

    private static func note() {
        Cache.shared.logOnce()
    }

    // MARK: - Apply

    private struct Packed {
        var weight: MLXArray
        var scales: MLXArray?
        var biases: MLXArray?
        var bits: Int
        var groupSize: Int
        /// Output rows of each stacked matrix, in order.
        var widths: [Int]
        /// Reduction length. Affine and dense planes that share a multiply must match.
        var k: Int
    }

    private static func apply(_ packed: Packed, _ x: MLXArray) -> MLXArray {
        if packed.bits == 0 {
            return matmul(x, packed.weight.swappedAxes(-1, -2))
        }
        return MLX.quantizedMatmul(
            x,
            packed.weight,
            scales: packed.scales!,
            biases: packed.biases!,
            transpose: true,
            groupSize: packed.groupSize,
            bits: packed.bits,
            mode: .affine)
    }

    private static func gather(_ x: MLXArray, _ packed: Packed, _ indices: MLXArray) -> MLXArray {
        if packed.bits == 0 {
            return MLX.gatherMM(
                x,
                packed.weight.swappedAxes(-1, -2),
                rhsIndices: indices,
                sortedIndices: false)
        }
        return MLX.gatherQuantizedMM(
            x,
            packed.weight,
            scales: packed.scales!,
            biases: packed.biases!,
            rhsIndices: indices,
            transpose: true,
            groupSize: packed.groupSize,
            bits: packed.bits,
            mode: .affine,
            sortedIndices: false)
    }

    private static func split(_ y: MLXArray, widths: [Int]) -> [MLXArray] {
        var parts: [MLXArray] = []
        var start = 0
        for (index, width) in widths.enumerated() {
            if index == widths.count - 1 {
                parts.append(y[.ellipsis, start...])
            } else {
                parts.append(y[.ellipsis, start..<(start + width)])
            }
            start += width
        }
        return parts
    }

    // MARK: - Cache

    private struct PairKey: Hashable {
        var a: ObjectIdentifier
        var b: ObjectIdentifier
    }

    private struct TripleKey: Hashable {
        var a: ObjectIdentifier
        var b: ObjectIdentifier
        var c: ObjectIdentifier
    }

    private struct ExpertPair {
        var gateUp: Packed
        var down: Packed
    }

    private enum Slot<T> {
        case ready(T)
        case skip
    }

    private final class Cache: @unchecked Sendable {
        static let shared = Cache()
        let lock = NSLock()
        var pairs: [PairKey: Slot<Packed>] = [:]
        var triples: [TripleKey: Slot<Packed>] = [:]
        var expert: [ObjectIdentifier: Slot<ExpertPair>] = [:]
        var didLog = false

        func pair(_ a: Linear, _ b: Linear) -> Packed? {
            let key = PairKey(a: ObjectIdentifier(a), b: ObjectIdentifier(b))
            switch hit(pairs, key) {
            case .ready(let packed): return packed
            case .skip: return nil
            case .miss: break
            }
            let built: Slot<Packed>
            if let left = plane(a), let right = plane(b), let joined = join([left, right]) {
                built = .ready(joined)
            } else {
                built = .skip
            }
            return store(&pairs, key, built)
        }

        func triple(_ a: Linear, _ b: Linear, _ c: Linear) -> Packed? {
            let key = TripleKey(
                a: ObjectIdentifier(a), b: ObjectIdentifier(b), c: ObjectIdentifier(c))
            switch hit(triples, key) {
            case .ready(let packed): return packed
            case .skip: return nil
            case .miss: break
            }
            let built: Slot<Packed>
            if let qa = plane(a), let qb = plane(b), let qc = plane(c),
                let joined = join([qa, qb, qc])
            {
                built = .ready(joined)
            } else {
                built = .skip
            }
            return store(&triples, key, built)
        }

        func experts(_ glu: SwitchGLU) -> ExpertPair? {
            let key = ObjectIdentifier(glu)
            switch hit(expert, key) {
            case .ready(let packed): return packed
            case .skip: return nil
            case .miss: break
            }
            let flat = dictionary(glu.parameters().flattened())
            let built: Slot<ExpertPair>
            if let gate = plane(flat, prefix: "gate_proj"),
                let up = plane(flat, prefix: "up_proj"),
                let down = plane(flat, prefix: "down_proj"),
                gate.widths == up.widths,
                let hidden = gate.widths.first,
                down.k == hidden,
                let gateUp = join([gate, up])
            {
                built = .ready(ExpertPair(gateUp: gateUp, down: down))
            } else {
                built = .skip
            }
            return store(&expert, key, built)
        }

        func logOnce() {
            lock.lock()
            let first = !didLog
            didLog = true
            lock.unlock()
            guard first else { return }
            let budget = UnlimitedOCRDecodeFuse.gemmBudget(layers: 12, denseLayers: 1, fused: true)
            let separate = UnlimitedOCRDecodeFuse.gemmBudget(layers: 12, denseLayers: 1, fused: false)
            NovaMLXLog.info(
                "[UnlimitedOCR] seq=1 decode fuse stacks QKV, router QK, and gate+up (\(budget.total) GEMMs, separate path \(separate.total)). DASH gather stays 512 keys."
            )
        }

        private enum Hit<T> {
            case miss
            case skip
            case ready(T)
        }

        private func hit<Key: Hashable, T>(_ map: [Key: Slot<T>], _ key: Key) -> Hit<T> {
            lock.lock()
            let value = map[key]
            lock.unlock()
            switch value {
            case .ready(let packed):
                return .ready(packed)
            case .skip:
                return .skip
            case nil:
                return .miss
            }
        }

        private func store<Key: Hashable, T>(_ map: inout [Key: Slot<T>], _ key: Key, _ built: Slot<T>) -> T? {
            lock.lock()
            if map[key] == nil { map[key] = built }
            let stored = map[key]
            lock.unlock()
            if case .ready(let value) = stored { return value }
            return nil
        }

        private func plane(_ linear: Linear) -> Packed? {
            plane(dictionary(linear.parameters().flattened()), prefix: "")
        }

        private func plane(_ flat: [String: MLXArray], prefix: String) -> Packed? {
            let name = prefix.isEmpty ? "" : prefix + "."
            if flat[name + "bias"] != nil { return nil }
            guard let weight = flat[name + "weight"] else { return nil }
            if let scales = flat[name + "scales"], let biases = flat[name + "biases"] {
                guard let decoded = affine(weight, scales, biases) else { return nil }
                let packed = Packed(
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    bits: decoded.bits,
                    groupSize: decoded.group,
                    widths: [decoded.n],
                    k: decoded.k)
                MLX.eval(weight, scales, biases)
                return packed
            }
            guard weight.dtype == .float16 || weight.dtype == .bfloat16 || weight.dtype == .float32
            else { return nil }
            let n: Int
            let k: Int
            if weight.ndim == 2 {
                n = weight.dim(0)
                k = weight.dim(1)
            } else if weight.ndim == 3 {
                n = weight.dim(1)
                k = weight.dim(2)
            } else {
                return nil
            }
            guard n > 0, k > 0 else { return nil }
            MLX.eval(weight)
            return Packed(
                weight: weight, scales: nil, biases: nil, bits: 0, groupSize: 0, widths: [n], k: k)
        }

        private func affine(_ weight: MLXArray, _ scales: MLXArray, _ biases: MLXArray) -> (
            bits: Int, group: Int, n: Int, k: Int
        )? {
            guard weight.dtype == .uint32, weight.ndim == scales.ndim, biases.shape == scales.shape
            else { return nil }
            let nAxis = weight.ndim - 2
            guard nAxis >= 0, scales.dim(nAxis) == weight.dim(nAxis) else { return nil }
            if weight.ndim == 3, scales.dim(0) != weight.dim(0) { return nil }
            let packed = weight.dim(weight.ndim - 1)
            let groups = scales.dim(scales.ndim - 1)
            guard let bits = UnlimitedOCRDecodeFuse.affineBits(packed: packed, groups: groups) else {
                return nil
            }
            return (bits.bits, bits.group, weight.dim(nAxis), groups * bits.group)
        }

        private func join(_ parts: [Packed]) -> Packed? {
            guard let first = parts.first, parts.count >= 2 else { return nil }
            guard parts.allSatisfy({
                $0.bits == first.bits && $0.groupSize == first.groupSize && $0.k == first.k
                    && $0.weight.ndim == first.weight.ndim
                    && $0.weight.dim(-1) == first.weight.dim(-1)
                    && (first.weight.ndim < 3 || $0.weight.dim(0) == first.weight.dim(0))
            }) else { return nil }
            let axis = first.weight.ndim - 2
            let weight = concatenated(parts.map(\.weight), axis: axis)
            let scales = first.scales.map { _ in concatenated(parts.compactMap(\.scales), axis: axis) }
            let biases = first.biases.map { _ in concatenated(parts.compactMap(\.biases), axis: axis) }
            if let scales, let biases {
                MLX.eval(weight, scales, biases)
            } else {
                MLX.eval(weight)
            }
            return Packed(
                weight: weight,
                scales: scales,
                biases: biases,
                bits: first.bits,
                groupSize: first.groupSize,
                widths: parts.flatMap(\.widths),
                k: first.k)
        }

        private func dictionary(_ pairs: [(String, MLXArray)]) -> [String: MLXArray] {
            var flat: [String: MLXArray] = [:]
            for (key, value) in pairs {
                flat[key] = value
            }
            return flat
        }
    }
}
