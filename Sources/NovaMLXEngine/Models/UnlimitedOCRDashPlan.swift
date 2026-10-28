import Foundation
import MLX
import MLXFast

/// One-thread DASH decode plan. Replaces the per-layer chain of sort,
/// 1.5-entmax, and elementwise kernels. Output length is the constant
/// `topK * block + local` window (512 on the released config).
///
/// `reference` is the numeric spec. The Metal kernel is the same loops,
/// in the same order, in float32. Shapeless compile is not used.
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

    static func supports(_ params: UnlimitedOCRDashParams, cap: Int) -> Bool {
        let spec = Spec.make(params, indexScale: 1, floor: 1)
        return cap >= spec.topM && cap <= 1024
            && spec.indexDim >= 1 && spec.indexDim <= 256
            && spec.topM >= 1 && spec.topM <= 32
            && spec.topK >= 1 && spec.topK <= spec.topM
            && spec.block >= 1 && spec.block <= 64
            && spec.local >= 1 && spec.local <= 512
            && spec.sigma != 0
    }

    /// Host spec for the kernel. `candidates[i] != 0` means the block is eligible.
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
        let prior = entmax15(taken)
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

    static func evaluate(
        query: MLXArray,
        summaries: MLXArray,
        candidates: MLXArray,
        position: MLXArray,
        indexScale: Float,
        floor: Float,
        params: UnlimitedOCRDashParams,
        cap: Int
    ) -> (MLXArray, MLXArray) {
        let spec = Spec.make(params, indexScale: indexScale, floor: floor)
        let kernel = kernel(spec)
        let coeffs = MLXArray([
            indexScale, floor, 1 / spec.sigma, 1 / Float(spec.indexDim).squareRoot(),
        ])
        let info = MLXArray([UInt32(cap)])
        let out = kernel(
            [query, summaries, candidates, position.reshaped([1]).asType(.uint32), info, coeffs],
            grid: (32, 1, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[spec.attended], [spec.attended]],
            outputDTypes: [.int32, .float32]
        )
        return (out[0], out[1])
    }

    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(_ spec: Spec) -> MLXFast.MLXFastKernel {
        let name = "uocr_dash_plan_\(spec.indexDim)_\(spec.block)_\(spec.local)_\(spec.localBlocks)_\(spec.topM)_\(spec.topK)"
        lock.lock()
        defer { lock.unlock() }
        if let cached = kernels[name] { return cached }
        let source = metalSource(spec)
        let kernel = MLXFast.metalKernel(
            name: name,
            inputNames: ["query", "summaries", "candidates", "position", "info", "coeffs"],
            outputNames: ["idx", "mask"],
            source: source,
            header: "",
            ensureRowContiguous: true
        )
        kernels[name] = kernel
        return kernel
    }

    /// Same control flow as `reference`. One thread, so the float order matches.
    private static func metalSource(_ spec: Spec) -> String {
        let d = spec.indexDim
        let block = spec.block
        let local = spec.local
        let localBlocks = spec.localBlocks
        let topM = spec.topM
        let topK = spec.topK
        let attended = spec.attended
        return """
            uint t = thread_position_in_threadgroup.x;
            threadgroup float score[1024];
            threadgroup uint sel[1024];
            if (t != 0) return;
            const int D = \(d);
            const int BLOCK = \(block);
            const int LOCAL = \(local);
            const int LOCAL_BLOCKS = \(localBlocks);
            const int TOPM = \(topM);
            const int TOPK = \(topK);
            const int pos = int(position[0]);
            const int cap = int(info[0]);
            const float indexScale = coeffs[0];
            const float floorv = coeffs[1];
            const float invSigma = coeffs[2];
            const float invDim = coeffs[3];
            const int remoteMax = pos / BLOCK - LOCAL_BLOCKS;

            int order[32];
            float taken[32];
            float prior[32];
            float shifted[32];
            float tau[32];

            bool any = false;
            for (int i = 0; i < cap; ++i) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += query[d] * summaries[i * D + d];
                }
                float s = dot * invDim;
                if (s < -1.0e4f) s = -1.0e4f;
                if (s > 1.0e4f) s = 1.0e4f;
                score[i] = s;
                uint remote = (i <= remoteMax) ? 1u : 0u;
                sel[i] = (remote != 0u && candidates[i] != 0u) ? 1u : 0u;
                if (sel[i] != 0u) any = true;
            }
            if (!any) {
                for (int i = 0; i < cap; ++i) {
                    sel[i] = (i <= remoteMax) ? 1u : 0u;
                }
            }

            const int mCount = cap < TOPM ? cap : TOPM;
            for (int slot = 0; slot < mCount; ++slot) order[slot] = -1;
            for (int i = 0; i < cap; ++i) {
                float value = (sel[i] != 0u) ? score[i] * indexScale : -1.0e4f;
                int place = mCount;
                for (int slot = 0; slot < mCount; ++slot) {
                    if (order[slot] < 0) { place = slot; break; }
                    float other = (sel[order[slot]] != 0u) ? score[order[slot]] * indexScale : -1.0e4f;
                    if (value > other || (value == other && i < order[slot])) {
                        place = slot;
                        break;
                    }
                }
                if (place < mCount) {
                    for (int slot = mCount - 1; slot > place; --slot) order[slot] = order[slot - 1];
                    order[place] = i;
                }
            }

            for (int j = 0; j < mCount; ++j) {
                int id = order[j];
                taken[j] = (sel[id] != 0u) ? score[id] * indexScale : -1.0e4f;
            }
            const float origin = taken[0];
            for (int j = 0; j < mCount; ++j) shifted[j] = (taken[j] - origin) * 0.5f;
            float csum = 0.0f;
            float csumSq = 0.0f;
            for (int j = 0; j < mCount; ++j) {
                float rho = float(j + 1);
                csum += shifted[j];
                csumSq += shifted[j] * shifted[j];
                float mean = csum / rho;
                float meanSq = csumSq / rho;
                float ss = rho * (meanSq - mean * mean);
                float delta = (1.0f - ss) / rho;
                if (delta < 0.0f) delta = 0.0f;
                tau[j] = mean - sqrt(delta);
            }
            int support = 0;
            for (int j = 0; j < mCount; ++j) {
                if (tau[j] <= shifted[j]) support += 1;
            }
            if (support < 1) support = 1;
            float tauStar = tau[support - 1];
            for (int j = 0; j < mCount; ++j) {
                float gap = shifted[j] - tauStar;
                if (gap < 0.0f) gap = 0.0f;
                prior[j] = gap * gap;
            }

            const int k = mCount < TOPK ? mCount : TOPK;
            float count = 0.0f;
            float sumLog = 0.0f;
            float logp[32];
            uint valid[32];
            for (int j = 0; j < k; ++j) {
                valid[j] = sel[order[j]];
                float chosen = prior[j] > floorv ? prior[j] : floorv;
                logp[j] = log(chosen);
                if (valid[j] != 0u) {
                    count += 1.0f;
                    sumLog += logp[j];
                }
            }
            if (count < 1.0f) count = 1.0f;
            float mu = sumLog / count;
            const int limit = pos - LOCAL;
            int slot = 0;
            for (int j = 0; j < k; ++j) {
                float b = 0.0f;
                if (valid[j] != 0u) b = (logp[j] - mu) * invSigma;
                for (int u = 0; u < BLOCK; ++u) {
                    int blockIndex = order[j] * BLOCK + u;
                    bool ok = valid[j] != 0u && blockIndex <= limit;
                    mask[slot] = ok ? b : -INFINITY;
                    idx[slot] = blockIndex < pos ? blockIndex : pos;
                    slot += 1;
                }
            }
            for (int u = 0; u < LOCAL; ++u) {
                int localIndex = pos - LOCAL + 1 + u;
                mask[slot] = localIndex >= 0 ? 0.0f : -INFINITY;
                idx[slot] = localIndex > 0 ? localIndex : 0;
                slot += 1;
            }
            for (; slot < \(attended); ++slot) {
                mask[slot] = -INFINITY;
                idx[slot] = 0;
            }
            """
    }
}
