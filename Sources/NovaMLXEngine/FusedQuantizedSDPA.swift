import Foundation
import MLX
import MLXLMCommon

enum FusedQuantizedSDPA {

    private static func makeKernelSource(D: Int, groupSize: Int, bits: Int) -> String {
        let weightStride = D * bits / 32
        let scaleStride = D / groupSize
        let qkPerThread = D / 32
        let vPerThread = D / 32

        let numScaleGroups = D / groupSize
        let valuesPerWord = 32 / bits
        let mask = (1 << bits) - 1

        let src = """
        {
            uint3 tid = thread_position_in_grid;
            uint3 tpg = threadgroups_per_grid;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;

            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int D = \(D);
            constexpr int group_size = \(groupSize);
            constexpr int bits_val = \(bits);
            constexpr int qk_per_thread = \(qkPerThread);
            constexpr int v_per_thread = \(vPerThread);
            constexpr int num_scale_groups = \(numScaleGroups);
            constexpr int values_per_word = \(valuesPerWord);
            constexpr int unpack_mask = \(mask);
            typedef float U;

            thread U q[qk_per_thread];
            thread U o[v_per_thread];

            threadgroup U tg_outputs[BN * BD];
            threadgroup U tg_max[BN];
            threadgroup U tg_sum[BN];

            int q_batch_head_idx = tid.x;
            int q_seq_idx = tid.y;
            int kv_head_idx = q_batch_head_idx / gqa_factor;
            int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;

            queries += o_offset * D + simd_lid * qk_per_thread;

            k_weights += kv_head_idx * N * \(weightStride) + simd_gid * \(weightStride);
            k_scales  += kv_head_idx * N * \(scaleStride)  + simd_gid * \(scaleStride);
            k_biases  += kv_head_idx * N * \(scaleStride)  + simd_gid * \(scaleStride);
            v_weights += kv_head_idx * N * \(weightStride) + simd_gid * \(weightStride);
            v_scales  += kv_head_idx * N * \(scaleStride)  + simd_gid * \(scaleStride);
            v_biases  += kv_head_idx * N * \(scaleStride)  + simd_gid * \(scaleStride);

            out += o_offset * D + simd_gid * v_per_thread;

            for (int i = 0; i < qk_per_thread; i++) {
                q[i] = static_cast<U>(scale) * queries[i];
            }
            for (int i = 0; i < v_per_thread; i++) {
                o[i] = 0;
            }

            U max_score = -1e30f;
            U sum_exp_score = 0;

            for (int i = simd_gid; i < N; i += BN) {
                bool use_key = (i <= (N - int(tpg.y) + q_seq_idx));
                if (use_key) {
                    U score = 0;
                    int thread_base = simd_lid * qk_per_thread;
                    for (int j = 0; j < qk_per_thread; j++) {
                        int d = thread_base + j;
                        int gi = d / group_size;
                        U s_val = static_cast<U>(k_scales[gi]);
                        U b_val = static_cast<U>(k_biases[gi]);
                        int word = d / values_per_word;
                        int shift = (d % values_per_word) * bits_val;
                        U kv = static_cast<U>((k_weights[word] >> shift) & unpack_mask) * s_val + b_val;
                        score += q[j] * kv;
                    }
                    score = simd_sum(score);

                    U new_max = max(max_score, score);
                    U factor = fast::exp(max_score - new_max);
                    U exp_score = fast::exp(score - new_max);
                    max_score = new_max;
                    sum_exp_score = sum_exp_score * factor + exp_score;

                    for (int j = 0; j < v_per_thread; j++) {
                        int d = thread_base + j;
                        int gi = d / group_size;
                        U s_val = static_cast<U>(v_scales[gi]);
                        U b_val = static_cast<U>(v_biases[gi]);
                        int word = d / values_per_word;
                        int shift = (d % values_per_word) * bits_val;
                        U dv = static_cast<U>((v_weights[word] >> shift) & unpack_mask) * s_val + b_val;
                        o[j] = o[j] * factor + exp_score * dv;
                    }
                }
                k_weights += BN * \(weightStride);
                k_scales  += BN * \(scaleStride);
                k_biases  += BN * \(scaleStride);
                v_weights += BN * \(weightStride);
                v_scales  += BN * \(scaleStride);
                v_biases  += BN * \(scaleStride);
            }

            if (simd_lid == 0) {
                tg_max[simd_gid] = max_score;
                tg_sum[simd_gid] = sum_exp_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score = tg_max[simd_lid];
            U new_max = simd_max(max_score);
            U factor = fast::exp(max_score - new_max);
            sum_exp_score = simd_sum(tg_sum[simd_lid] * factor);

            for (int i = 0; i < v_per_thread; i++) {
                tg_outputs[simd_lid * BD + simd_gid] = o[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[i] = simd_sum(tg_outputs[simd_gid * BD + simd_lid] * factor);
                o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (simd_lid == 0) {
                for (int i = 0; i < v_per_thread; i++) {
                    out[i] = static_cast<half>(o[i]);
                }
            }
        }
        """

        return src
    }

    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernelCache: [String: MLXFast.MLXFastKernel] = [:]

    private static func getKernel(D: Int, groupSize: Int, bits: Int) -> MLXFast.MLXFastKernel {
        let key = "sdpa_qv_\(D)_\(groupSize)_\(bits)"
        lock.lock()
        defer { lock.unlock() }
        if let existing = kernelCache[key] {
            return existing
        }

        let source = makeKernelSource(D: D, groupSize: groupSize, bits: bits)
        let kernel = MLXFast.metalKernel(
            name: "sdpa_quantized_vector_\(D)_\(groupSize)_\(bits)",
            inputNames: [
                "queries", "k_weights", "k_scales", "k_biases",
                "v_weights", "v_scales", "v_biases",
                "gqa_factor", "N", "scale",
            ],
            outputNames: ["out"],
            source: source,
            header: "",
            ensureRowContiguous: true
        )
        kernelCache[key] = kernel
        return kernel
    }

    static func attention(
        queries: MLXArray,
        quantizedKeys: (MLXArray, MLXArray, MLXArray?),
        quantizedValues: (MLXArray, MLXArray, MLXArray?),
        scale: Float,
        groupSize: Int,
        bits: Int
    ) -> MLXArray? {
        let L = queries.dim(2)
        let D = queries.dim(3)

        guard L == 1 else { return nil }
        guard [64, 96, 128, 256].contains(D) else { return nil }
        guard D % 32 == 0 else { return nil }
        guard [4, 8].contains(bits) else { return nil }
        guard D % groupSize == 0 else { return nil }

        let kBiases = quantizedKeys.2
        let vBiases = quantizedValues.2

        guard kBiases != nil, vBiases != nil else { return nil }

        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let nKVHeads = quantizedKeys.0.dim(-3)
        let N = quantizedKeys.0.dim(-2)
        let nRepeats = nQHeads / nKVHeads

        var adjustedQueries = queries
        var qKeys = quantizedKeys
        var qValues = quantizedValues

        if nRepeats > 1 {
            adjustedQueries = queries.reshaped([B, nKVHeads, nRepeats, L, D])
            qKeys = (
                expandedDimensions(qKeys.0, axis: -4),
                expandedDimensions(qKeys.1, axis: -4),
                qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -4)
            )
            qValues = (
                expandedDimensions(qValues.0, axis: -4),
                expandedDimensions(qValues.1, axis: -4),
                qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -4)
            )
        }

        let qFlat = adjustedQueries.reshaped([B * nQHeads, L, D])
        let kWeightsFlat = qKeys.0.reshaped([B * nKVHeads, N, qKeys.0.dim(-1)])
        let kScalesFlat = qKeys.1.reshaped([B * nKVHeads, N, qKeys.1.dim(-1)])
        let kBiasesFlat = kBiases!.reshaped([B * nKVHeads, N, kBiases!.dim(-1)])
        let vWeightsFlat = qValues.0.reshaped([B * nKVHeads, N, qValues.0.dim(-1)])
        let vScalesFlat = qValues.1.reshaped([B * nKVHeads, N, qValues.1.dim(-1)])
        let vBiasesFlat = vBiases!.reshaped([B * nKVHeads, N, vBiases!.dim(-1)])

        let totalHeads = B * nQHeads
        let kernel = getKernel(D: D, groupSize: groupSize, bits: bits)

        let gqaFactorArr = MLXArray(nRepeats)
        let nArr = MLXArray(N)
        let scaleArr = MLXArray(scale)

        let results = kernel(
            [
                qFlat,
                kWeightsFlat,
                kScalesFlat,
                kBiasesFlat,
                vWeightsFlat,
                vScalesFlat,
                vBiasesFlat,
                gqaFactorArr,
                nArr,
                scaleArr,
            ],
            grid: (totalHeads, L, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[totalHeads, L, D]],
            outputDTypes: [queries.dtype]
        )

        let output = results[0].reshaped([B, nQHeads, L, D])
        return output
    }
}
