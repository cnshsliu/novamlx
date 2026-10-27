import Foundation
import MLX
import MLXFast

// MARK: - Fused expert routing (Flint only)

/// Sigmoid top-k after the router matmul, in one Metal kernel.
///
/// This is mlx-lm's `expert_routing` (`mlx_lm/kernels/expert_routing.py`),
/// which `mlx_lm/models/flint.py` runs on Metal. The previous Swift block
/// was that file's training fallback: separate sigmoid, argpartition, and
/// renorm dispatches.
///
/// The constants match Recursive `rec` `model.py` `SparseMoeBlock` and the
/// decode megakernel, not mlx-lm's `score_eps=0` call. Scores are sigmoid.
/// `expert_bias` is added before the top-k and is not part of the weight.
/// Weights are the selected scores divided by their sum plus `1e-20`.
/// Ties keep the lower expert index, the same rule as `w3_route_own_rank`.
///
/// Official `RecursiveSI/flint-int4` sets `router_expert_bias` false, so the
/// bias the checkpoint omits is zeros. A present `gate.expert_bias` is
/// renamed onto this block in `sanitize` and packed into megakernel slot 9.
enum FlintExpertRouting {
    /// `rec` model.py and `w3_sum += 1.0e-20f`.
    static let scoreEps: Float = 1e-20
    static let multiplier: Float = 1
    static let routeScale: Float = 1

    struct Choice: Equatable {
        var index: Int
        var weight: Float
    }

    /// One token. Same selection and renorm as the Metal kernel.
    static func reference(logits: [Float], bias: [Float], topK: Int) -> [Choice] {
        let experts = logits.count
        guard experts > 0, bias.count == experts, topK > 0, topK <= experts else { return [] }
        var scores = [Float](repeating: 0, count: experts)
        var biased = [Float](repeating: -.infinity, count: experts)
        for index in 0 ..< experts {
            let x = logits[index] * multiplier
            let tail = 1 / (1 + exp(abs(x)))
            let score: Float = x < 0 ? tail : 1 - tail
            scores[index] = score
            biased[index] = score + bias[index]
        }
        var chosen: [(Int, Float)] = []
        chosen.reserveCapacity(topK)
        for _ in 0 ..< topK {
            var best = -Float.infinity
            var winner = experts
            for index in 0 ..< experts {
                let value = biased[index]
                if value > best || (value == best && index < winner) {
                    best = value
                    winner = index
                }
            }
            chosen.append((winner, scores[winner]))
            biased[winner] = -.infinity
        }
        let total = chosen.reduce(Float(0)) { $0 + $1.1 }
        let denom = total + scoreEps
        return chosen.map { pair in
            Choice(index: pair.0, weight: pair.1 / denom * routeScale)
        }
    }

    /// Indices `[..., topK]` uint32 and float32 weights with the same shape.
    static func route(logits: MLXArray, bias: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        let experts = logits.dim(-1)
        let tokens = experts > 0 ? logits.size / experts : 0
        guard experts > 0, topK > 0, topK <= experts, bias.size == experts, tokens > 0,
            logits.size == tokens * experts
        else {
            return fallback(logits: logits, bias: bias, topK: max(topK, 1))
        }
        let kernel = kernel(experts: experts, topK: topK)
        var shape = Array(logits.shape)
        shape[shape.count - 1] = topK
        let threads = tokens * 32
        let group = 256
        let grid = max(group, ((threads + group - 1) / group) * group)
        let out = kernel(
            [
                logits,
                bias.asType(.float32),
                MLXArray([UInt32(tokens)], [1]),
                MLXArray([multiplier], [1]),
                MLXArray([scoreEps], [1]),
                MLXArray([routeScale], [1]),
            ],
            grid: (grid, 1, 1),
            threadGroup: (group, 1, 1),
            outputShapes: [shape, shape],
            outputDTypes: [.uint32, .float32]
        )
        // argpartition on this stack is int32. 0..<E fits, and SwitchGLU
        // already gathers those indices.
        return (out[0].asType(.int32), out[1])
    }

    /// `rec` eager formula when the fused kernel's shape contract is not met.
    static func fallback(logits: MLXArray, bias: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        let scores0 = sigmoid(logits)
        let choice = scores0 + bias.asType(scores0.dtype)
        let inds = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let taken = takeAlong(scores0, inds, axis: -1)
        let weights = taken / (taken.sum(axis: -1, keepDims: true) + scoreEps)
        return (inds, weights)
    }

    private static let lock = NSLock()
    private nonisolated(unsafe) static var cache: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(experts: Int, topK: Int) -> MLXFast.MLXFastKernel {
        let key = "\(experts)_\(topK)"
        lock.lock()
        defer { lock.unlock() }
        if let existing = cache[key] { return existing }
        let source = """
            constexpr int E = \(experts);
            constexpr int K = \(topK);
            constexpr int R = (E + 31) / 32;
            uint token = thread_position_in_grid.x / 32;
            if (token >= tokens[0]) return;
            uint lane = thread_index_in_simdgroup;
            auto row = logits + size_t(token) * E;

            float score[R];
            float biased[R];
            for (int i = 0; i < R; i++) {
                int e = lane + 32 * i;
                score[i] = 0.0f;
                biased[i] = -INFINITY;
                if (e < E) {
                    float x = row[e] * multiplier[0];
                    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    score[i] = x < 0 ? y : 1.0f - y;
                    biased[i] = score[i] + bias[e];
                }
            }

            float w[K];
            for (int k = 0; k < K; k++) {
                float best = -INFINITY;
                int best_e = E;
                for (int i = 0; i < R; i++) {
                    int e = lane + 32 * i;
                    if (biased[i] > best || (biased[i] == best && e < best_e)) {
                        best = biased[i];
                        best_e = e;
                    }
                }
                float top = simd_max(best);
                int winner = simd_min(best == top ? best_e : E);
                float s = 0.0f;
                for (int i = 0; i < R; i++) {
                    if (lane + 32 * i == winner) {
                        s = score[i];
                        biased[i] = -INFINITY;
                    }
                }
                w[k] = simd_sum(s);
                if (lane == 0) {
                    inds[size_t(token) * K + k] = winner;
                }
            }

            float total = 0.0f;
            for (int k = 0; k < K; k++) {
                total = w[k] + total;
            }
            if (lane == 0) {
                for (int k = 0; k < K; k++) {
                    weights[size_t(token) * K + k] = w[k] / (total + eps[0]) * scale[0];
                }
            }
            """
        let kernel = MLXFast.metalKernel(
            name: "flint_expert_routing_\(experts)_\(topK)",
            inputNames: ["logits", "bias", "tokens", "multiplier", "eps", "scale"],
            outputNames: ["inds", "weights"],
            source: source,
            header: "",
            ensureRowContiguous: true
        )
        cache[key] = kernel
        return kernel
    }
}
