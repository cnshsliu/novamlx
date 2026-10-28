import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import NovaMLXUtils

/// One Metal launch per decode layer. The 1280-d residual stays on device
/// between int4 QKV, RoPE, 512-key SDPA, the output projection, and SwiGLU.
/// Matvecs use MLX's `qdot` / `load_vector` (the qmv path), not the scalar
/// per-row kernel.
///
/// `MLXFast` has no grid barrier, so the launch is 32 resident threadgroups
/// and a ticket barrier. That is small enough for M4 Max to keep them all
/// scheduled. The flag is `NOVA_UOCR_MEGA=1` and defaults off. The row-stack
/// fuse stays the product path.
///
/// Metal gives a kernel 31 buffer arguments (indices 0...30). Binding each
/// quant plane on its own used 46 inputs and 6 outputs, and the JIT rejected
/// `buffer(46...51)`. The uint32 weights, the bf16 scales (post-norm weight
/// included), and the bf16 biases are each one buffer. `info` holds the
/// element offset of every plane. The copy is evaluated once per block and
/// reused. The 32-group launch stays.
enum UnlimitedOCRLayerKernel {
    static let groups = 32
    static let threads = 64

    /// Metal function buffer arguments are indices 0 ... `metalBufferLimit - 1`.
    static let metalBufferLimit = 31

    static let inputNames = [
        "x", "xn", "cosv", "sinv",
        "weights", "scales", "biases",
        "idx", "attnBias", "position", "info", "coeff", "sync",
        "cacheK", "cacheV",
    ]
    static let outputNames = ["y", "kNew", "vNew", "work", "wts", "ids"]

    static var boundBufferCount: Int { inputNames.count + outputNames.count }

    /// `info` layout: 10 scalars, then weight offsets, scale offsets (post last), bias offsets.
    static let infoScalars = 10
    static let weightPlaneNames = ["q", "k", "v", "o", "score", "eg", "eu", "ed", "sg", "su", "sd"]
    static let scalePlaneNames = weightPlaneNames + ["post"]
    static let biasPlaneNames = weightPlaneNames
    static let weightPointers = ["qw", "kw", "vw", "ow", "scw", "egw", "euw", "edw", "sgw", "suw", "sdw"]
    static let scalePointers = [
        "qs", "ks", "vs", "os", "scs", "egs", "eus", "eds", "sgs", "sus", "sds", "postw",
    ]
    static let biasPointers = ["qb", "kb", "vb", "ob", "scb", "egb", "eub", "edb", "sgb", "sub", "sdb"]

    /// Prefix sums. `counts[i]` elements of plane `i` start at the returned offset.
    static func planeOffsets(_ counts: [Int]) -> [Int] {
        var offsets: [Int] = []
        offsets.reserveCapacity(counts.count)
        var cursor = 0
        for count in counts {
            offsets.append(cursor)
            cursor += count
        }
        return offsets
    }

    /// Element of expert `id` inside a packed plane whose plane starts at `base`.
    /// Matches `pointer + id * rows * (K / pack)`.
    static func expertElement(base: Int, id: Int, rows: Int, packed: Int) -> Int {
        base + id * rows * packed
    }

    /// True when the grid barrier uses MLX's device fence, not the one-argument
    /// fence or a lambda that drops the device address space.
    static func metalBarrierCompilesOnMLX() -> Bool {
        let fence =
            "atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)"
        let barrier = qmvHeader
        let body = layerSource(bits: 4)
        guard barrier.contains(fence),
            body.contains("uocr_barrier(counter, epoch, \(groups), lid)"),
            !barrier.contains("atomic_thread_fence(mem_flags::mem_device);"),
            !barrier.contains("memory_order_acquire"),
            !barrier.contains("memory_order_acq_rel"),
            !barrier.contains("memory_order_release"),
            !body.contains("auto bar")
        else { return false }
        return true
    }

    /// Host check for the launch shape. True when every binding is in 0...30 and
    /// the alias block covers the last `info` slot.
    static func bindingFitsMetalLimit() -> Bool {
        guard boundBufferCount <= metalBufferLimit,
            weightPointers.count == weightPlaneNames.count,
            scalePointers.count == scalePlaneNames.count,
            biasPointers.count == biasPlaneNames.count
        else { return false }
        let infoCount = infoScalars + weightPointers.count + scalePointers.count + biasPointers.count
        let lines = packedAliases().split(separator: "\n")
        guard lines.count == infoCount - infoScalars else { return false }
        return lines.contains { $0.contains("info[\(infoCount - 1)]") }
    }

    /// Device pointers the kernel takes from the three packed buffers.
    static func packedAliases() -> String {
        var lines: [String] = []
        for (offset, name) in weightPointers.enumerated() {
            lines.append(
                "const device uint32_t* \(name) = weights + size_t(info[\(infoScalars + offset)]);")
        }
        let scaleBase = infoScalars + weightPointers.count
        for (offset, name) in scalePointers.enumerated() {
            lines.append(
                "const device bfloat16_t* \(name) = scales + size_t(info[\(scaleBase + offset)]);")
        }
        let biasBase = scaleBase + scalePointers.count
        for (offset, name) in biasPointers.enumerated() {
            lines.append(
                "const device bfloat16_t* \(name) = biases + size_t(info[\(biasBase + offset)]);")
        }
        return lines.joined(separator: "\n")
    }

    /// Holds the packed planes for one block. Not an `MLXArray`, so `parameters()` skips it.
    /// `planeLock` guards `plane`.
    final class PlaneCache: @unchecked Sendable {
        fileprivate var plane: PackedPlanes?
        init() {}
    }

    static func allows(_ raw: String?) -> Bool {
        switch raw?.lowercased() {
        case "1", "on", "true", "yes":
            return true
        default:
            return false
        }
    }

    static let enabled = allows(ProcessInfo.processInfo.environment["NOVA_UOCR_MEGA"])

    static var launchThreads: Int { groups * threads }

    /// Half-split RoPE. `rot[d] = -x[d+half]` for the first half and `x[d-half]` after.
    static func referenceRope(_ x: [Float], cos: [Float], sin: [Float]) -> [Float] {
        let half = x.count / 2
        return (0..<x.count).map { d in
            let rot = d < half ? -x[d + half] : x[d - half]
            return x[d] * cos[d] + rot * sin[d]
        }
    }

    static func step(
        x: MLXArray,
        normIn: RMSNorm,
        attention: UnlimitedOCRDashAttention,
        normOut: RMSNorm,
        mlp: Module,
        ctx: UnlimitedOCRDashContext,
        cache: UnlimitedOCRDashCache,
        planes: PlaneCache
    ) -> MLXArray? {
        guard enabled, x.dtype == .bfloat16, ctx.decode, ctx.cos.size == attention.headDim,
            ctx.sin.size == attention.headDim
        else { return nil }
        let xn = normIn(x)
        guard let cacheK = cache.keys, let cacheV = cache.values else { return nil }
        guard let (idx, mask) = attention.gatherPlan(xn, ctx, cache) else { return nil }
        guard let weights = Weights.load(attention: attention, normOut: normOut, mlp: mlp) else {
            return nil
        }
        guard let kernel = kernel(weights), let plane = packed(planes, weights) else { return nil }
        let hidden = weights.hidden
        let heads = attention.heads
        let dim = attention.headDim
        let width = attention.params.attendedKeys
        let header = weights.info(
            heads: heads, dim: dim, width: width, slab: cache.keys?.dim(2) ?? 0)
        guard header.count == infoScalars else { return nil }
        var words: [UInt32] = []
        words.reserveCapacity(
            infoScalars + weightPointers.count + scalePointers.count + biasPointers.count)
        for value in header + plane.weightOffsets + plane.scaleOffsets + plane.biasOffsets {
            guard let word = UInt32(exactly: value) else { return nil }
            words.append(word)
        }
        let info = MLXArray(words)
        let coeff = MLXArray([attention.scale, normOutEps])
        let position = ctx.position ?? MLXArray(Int32(0))
        let sync = MLXArray.zeros([2], dtype: .uint32)
        let siluRows = max(weights.expertInter * max(weights.assignments, 1), weights.sharedInter)
        let scratch = hidden * 3 + siluRows
        let y = kernel(
            [
                x.reshaped([hidden]), xn.reshaped([hidden]),
                ctx.cos.reshaped([dim]), ctx.sin.reshaped([dim]),
                plane.weights, plane.scales, plane.biases,
                idx.asType(.int32), mask.asType(.float32),
                position.reshaped([1]).asType(.int32),
                info, coeff, sync,
                cacheK, cacheV,
            ],
            grid: (launchThreads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[1, 1, hidden], [heads, dim], [heads, dim], [scratch], [8], [8]],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .bfloat16, .float32, .uint32])
        let hiddenOut = y[0]
        let kNew = y[1].reshaped([1, heads, 1, dim])
        let vNew = y[2].reshaped([1, heads, 1, dim])
        _ = cache.update(keys: kNew, values: vNew)
        note()
        return hiddenOut
    }

    private static let normOutEps: Float = 1e-6

    // MARK: weights

    private struct Quant {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
        var rows: Int
        var k: Int
        var experts: Int
        var bits: Int
    }

    fileprivate struct PackedPlanes {
        var weights: MLXArray
        var scales: MLXArray
        var biases: MLXArray
        var weightOffsets: [Int]
        var scaleOffsets: [Int]
        var biasOffsets: [Int]
    }

    /// `Weights` is private, so this stays a private method of the enclosing type.
    /// A `fileprivate` method on `PackedPlanes` cannot take `Weights`.
    private static func make(_ weights: Weights) -> PackedPlanes? {
        let quants = [
            weights.q, weights.k, weights.v, weights.o, weights.score,
            weights.eg, weights.eu, weights.ed, weights.sg, weights.su, weights.sd,
        ]
        guard quants.count == weightPlaneNames.count else { return nil }
        let weightFlat = quants.map { $0.weight.reshaped([$0.weight.size]) }
        let scaleFlat =
            quants.map { $0.scales.reshaped([$0.scales.size]) }
            + [weights.post.reshaped([weights.post.size])]
        let biasFlat = quants.map { $0.biases.reshaped([$0.biases.size]) }
        guard scaleFlat.count == scalePlaneNames.count, biasFlat.count == biasPlaneNames.count
        else { return nil }
        let weightOffsets = planeOffsets(weightFlat.map(\.size))
        let scaleOffsets = planeOffsets(scaleFlat.map(\.size))
        let biasOffsets = planeOffsets(biasFlat.map(\.size))
        let w = concatenated(weightFlat, axis: 0)
        let s = concatenated(scaleFlat, axis: 0)
        let b = concatenated(biasFlat, axis: 0)
        // Realized once. A lazy concat would copy every token.
        MLX.eval(w, s, b)
        return PackedPlanes(
            weights: w, scales: s, biases: b,
            weightOffsets: weightOffsets, scaleOffsets: scaleOffsets, biasOffsets: biasOffsets)
    }

    private static let planeLock = NSLock()

    private static func packed(_ cache: PlaneCache, _ weights: Weights) -> PackedPlanes? {
        planeLock.lock()
        if let plane = cache.plane {
            planeLock.unlock()
            return plane
        }
        planeLock.unlock()
        guard let built = make(weights) else { return nil }
        planeLock.lock()
        if cache.plane == nil { cache.plane = built }
        let stored = cache.plane
        planeLock.unlock()
        return stored
    }

    private struct Weights {
        var q, k, v, o, score: Quant
        var eg, eu, ed, sg, su, sd: Quant
        var post: MLXArray
        var hidden: Int
        var expertInter: Int
        var sharedInter: Int
        var assignments: Int
        var bits: Int

        func info(heads: Int, dim: Int, width: Int, slab: Int) -> [Int] {
            [hidden, heads, dim, width, slab, assignments, expertInter, sharedInter, bits, groups]
        }

        static func load(
            attention: UnlimitedOCRDashAttention, normOut: RMSNorm, mlp: Module
        ) -> Weights? {
            guard let q = quant(attention.qProj), let k = quant(attention.kProj),
                let v = quant(attention.vProj), let o = quant(attention.oProj),
                q.bits == k.bits, q.bits == v.bits, q.bits == o.bits,
                q.k == k.k, q.k == v.k, o.k == q.rows, q.rows == attention.heads * attention.headDim
            else { return nil }
            let postFlat = dictionary(normOut.parameters().flattened())
            guard let post = postFlat["weight"], post.dim(0) == q.k else { return nil }
            if let dense = mlp as? UnlimitedOCRDenseMLP {
                guard let g = quant(dense.gate), let u = quant(dense.up), let d = quant(dense.down),
                    g.bits == q.bits, g.rows == u.rows, d.k == g.rows, d.rows == q.k
                else { return nil }
                let dummy = g
                return Weights(
                    q: q, k: k, v: v, o: o, score: dummy,
                    eg: g, eu: u, ed: d, sg: g, su: u, sd: d,
                    post: post, hidden: q.k, expertInter: g.rows, sharedInter: g.rows,
                    assignments: 0, bits: q.bits)
            }
            guard let moe = mlp as? UnlimitedOCRMoE else { return nil }
            guard let score = quant(moe.gate), score.rows == moe.expertsCount, score.k == q.k else {
                return nil
            }
            let flat = dictionary(moe.experts.parameters().flattened())
            let shared = dictionary(moe.shared.parameters().flattened())
            guard let eg = quant(flat, "gate_proj"), let eu = quant(flat, "up_proj"),
                let ed = quant(flat, "down_proj"), let sg = quant(shared, "gate_proj"),
                let su = quant(shared, "up_proj"), let sd = quant(shared, "down_proj"),
                eg.bits == q.bits, eg.rows == eu.rows, ed.k == eg.rows, ed.rows == q.k,
                sg.rows == su.rows, sd.k == sg.rows, sd.rows == q.k
            else { return nil }
            return Weights(
                q: q, k: k, v: v, o: o, score: score,
                eg: eg, eu: eu, ed: ed, sg: sg, su: su, sd: sd,
                post: post, hidden: q.k, expertInter: eg.rows, sharedInter: sg.rows,
                assignments: moe.topK, bits: q.bits)
        }
    }

    private static func quant(_ linear: Linear) -> Quant? {
        quant(dictionary(linear.parameters().flattened()), "")
    }

    private static func quant(_ flat: [String: MLXArray], _ prefix: String) -> Quant? {
        let name = prefix.isEmpty ? "" : prefix + "."
        guard flat[name + "bias"] == nil, let weight = flat[name + "weight"],
            let scales = flat[name + "scales"], let biases = flat[name + "biases"],
            weight.dtype == .uint32, scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let experts = weight.ndim == 3 ? weight.dim(0) : 1
        let rows = weight.dim(weight.ndim - 2)
        let groups = scales.dim(-1)
        guard let bits = UnlimitedOCRDecodeFuse.affineBits(packed: weight.dim(-1), groups: groups),
            bits.group == 64, rows % 8 == 0
        else { return nil }
        return Quant(
            weight: weight, scales: scales, biases: biases,
            rows: rows, k: groups * 64, experts: experts, bits: bits.bits)
    }

    private static func dictionary(_ pairs: [(String, MLXArray)]) -> [String: MLXArray] {
        var flat: [String: MLXArray] = [:]
        for (key, value) in pairs { flat[key] = value }
        return flat
    }

    // MARK: kernel

    private static let lock = NSLock()
    private nonisolated(unsafe) static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private nonisolated(unsafe) static var didLog = false

    private static func kernel(_ weights: Weights) -> MLXFast.MLXFastKernel? {
        lock.lock()
        if let cached = kernels[weights.bits] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        precondition(
            boundBufferCount <= metalBufferLimit
                && weightPointers.count == weightPlaneNames.count
                && scalePointers.count == scalePlaneNames.count
                && biasPointers.count == biasPlaneNames.count,
            "Unlimited OCR layer kernel bound \(boundBufferCount) buffers; Metal allows \(metalBufferLimit)."
        )
        let built = MLXFast.metalKernel(
            name: "uocr_layer_\(weights.bits)",
            inputNames: inputNames,
            outputNames: outputNames,
            source: layerSource(bits: weights.bits),
            header: FlintM4MaxPrefillSteel.header + qmvHeader,
            ensureRowContiguous: true)
        lock.lock()
        if kernels[weights.bits] == nil { kernels[weights.bits] = built }
        let stored = kernels[weights.bits]
        lock.unlock()
        return stored
    }

    private static func note() {
        lock.lock()
        let first = !didLog
        didLog = true
        lock.unlock()
        guard first else { return }
        NovaMLXLog.info(
            "[UnlimitedOCR] one-launch layer kernel (qdot qmv, \(groups) groups, \(boundBufferCount) buffers). Fuse stays the default."
        )
    }

    private static let qmvHeader = """

        template <typename T, int bits>
        METAL_FUNC void uocr_qmv_all(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            device T* y,
            int K,
            int N,
            uint gid,
            uint simd_gid,
            uint simd_lid) {
            constexpr int pack_factor = get_pack_factor<bits, 32>();
            constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
            constexpr int values_per_thread = pack_factor;
            constexpr int block_size = values_per_thread * 32;
            constexpr int rps = 4;
            constexpr int tile = 8;
            const int ngrp = \(groups);
            const int in_w = K * bytes_per_pack / pack_factor;
            const int in_g = K / 64;
            for (int base = int(gid) * tile; base < N; base += ngrp * tile) {
                int out_row = simd_gid * rps;
                const device uint8_t* ws = ((const device uint8_t*)w) +
                    (size_t)(base + out_row) * in_w + simd_lid * bytes_per_pack;
                const device T* sc = scales + (size_t)(base + out_row) * in_g + simd_lid / (64 / values_per_thread);
                const device T* bc = biases + (size_t)(base + out_row) * in_g + simd_lid / (64 / values_per_thread);
                const device T* xv = x + simd_lid * values_per_thread;
                float result[rps] = {0, 0, 0, 0};
                int k = 0;
                for (; k + block_size <= K; k += block_size) {
                    float x_thread[values_per_thread];
                    float sum = load_vector<T, float, values_per_thread, bits>(xv, x_thread);
                    for (int row = 0; row < rps; ++row) {
                        auto wl = (const device uint8_t*)(ws + row * in_w);
                        float s = float(sc[row * in_g]);
                        float b = float(bc[row * in_g]);
                        result[row] += qdot<float, values_per_thread, bits>(wl, x_thread, s, b, sum);
                    }
                    ws += block_size * bytes_per_pack / pack_factor;
                    sc += block_size / 64;
                    bc += block_size / 64;
                    xv += block_size;
                }
                if (k < K) {
                    int remaining = clamp(K - k - int(simd_lid) * values_per_thread, 0, values_per_thread);
                    float x_thread[values_per_thread];
                    float sum = 0.0f;
                    if (remaining > 0) {
                        sum = load_vector_safe<T, float, values_per_thread, bits>(xv, x_thread, remaining);
                    }
                    for (int row = 0; row < rps; ++row) {
                        auto wl = (const device uint8_t*)(ws + row * in_w);
                        float s = float(sc[row * in_g]);
                        float b = float(bc[row * in_g]);
                        result[row] += qdot_safe<float, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
                    }
                }
                for (int row = 0; row < rps; ++row) {
                    result[row] = simd_sum(result[row]);
                    if (simd_lid == 0) {
                        y[base + out_row + row] = static_cast<T>(result[row]);
                    }
                }
            }
        }

        template <typename T, int bits>
        METAL_FUNC void uocr_swiglu_all(
            const device uint32_t* gw,
            const device T* gs,
            const device T* gb,
            const device uint32_t* uw,
            const device T* us,
            const device T* ub,
            const device T* x,
            device T* y,
            int K,
            int N,
            uint gid,
            uint simd_gid,
            uint simd_lid) {
            constexpr int pack_factor = get_pack_factor<bits, 32>();
            constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
            constexpr int values_per_thread = pack_factor;
            constexpr int block_size = values_per_thread * 32;
            constexpr int rps = 4;
            const int ngrp = \(groups);
            const int tile = 8;
            const int in_w = K * bytes_per_pack / pack_factor;
            const int in_g = K / 64;
            for (int base = int(gid) * tile; base < N; base += ngrp * tile) {
                int out_row = simd_gid * rps;
                const device uint8_t* gws = ((const device uint8_t*)gw) + (size_t)(base + out_row) * in_w + simd_lid * bytes_per_pack;
                const device uint8_t* uws = ((const device uint8_t*)uw) + (size_t)(base + out_row) * in_w + simd_lid * bytes_per_pack;
                const device T* gsc = gs + (size_t)(base + out_row) * in_g + simd_lid / (64 / values_per_thread);
                const device T* gbc = gb + (size_t)(base + out_row) * in_g + simd_lid / (64 / values_per_thread);
                const device T* usc = us + (size_t)(base + out_row) * in_g + simd_lid / (64 / values_per_thread);
                const device T* ubc = ub + (size_t)(base + out_row) * in_g + simd_lid / (64 / values_per_thread);
                const device T* xv = x + simd_lid * values_per_thread;
                float gate[rps] = {0, 0, 0, 0};
                float up[rps] = {0, 0, 0, 0};
                int k = 0;
                for (; k + block_size <= K; k += block_size) {
                    float x_thread[values_per_thread];
                    float sum = load_vector<T, float, values_per_thread, bits>(xv, x_thread);
                    for (int row = 0; row < rps; ++row) {
                        float s = float(gsc[row * in_g]);
                        float b = float(gbc[row * in_g]);
                        gate[row] += qdot<float, values_per_thread, bits>((const device uint8_t*)(gws + row * in_w), x_thread, s, b, sum);
                        s = float(usc[row * in_g]);
                        b = float(ubc[row * in_g]);
                        up[row] += qdot<float, values_per_thread, bits>((const device uint8_t*)(uws + row * in_w), x_thread, s, b, sum);
                    }
                    gws += block_size * bytes_per_pack / pack_factor;
                    uws += block_size * bytes_per_pack / pack_factor;
                    gsc += block_size / 64;
                    gbc += block_size / 64;
                    usc += block_size / 64;
                    ubc += block_size / 64;
                    xv += block_size;
                }
                if (k < K) {
                    int remaining = clamp(K - k - int(simd_lid) * values_per_thread, 0, values_per_thread);
                    float x_thread[values_per_thread];
                    float sum = 0.0f;
                    if (remaining > 0) {
                        sum = load_vector_safe<T, float, values_per_thread, bits>(xv, x_thread, remaining);
                    }
                    for (int row = 0; row < rps; ++row) {
                        float s = float(gsc[row * in_g]);
                        float b = float(gbc[row * in_g]);
                        gate[row] += qdot_safe<float, values_per_thread, bits>((const device uint8_t*)(gws + row * in_w), x_thread, s, b, sum, remaining);
                        s = float(usc[row * in_g]);
                        b = float(ubc[row * in_g]);
                        up[row] += qdot_safe<float, values_per_thread, bits>((const device uint8_t*)(uws + row * in_w), x_thread, s, b, sum, remaining);
                    }
                }
                for (int row = 0; row < rps; ++row) {
                    gate[row] = simd_sum(gate[row]);
                    up[row] = simd_sum(up[row]);
                    if (simd_lid == 0) {
                        float g = gate[row];
                        float clipped = g < -80.0f ? -80.0f : (g > 80.0f ? 80.0f : g);
                        float prod = (g / (1.0f + exp(-clipped))) * up[row];
                        y[base + out_row + row] = static_cast<T>(prod);
                    }
                }
            }
        }

        // MLX prepends mlx/backend/metal/kernels/utils.h. The one-argument
        // atomic_thread_fence and acquire/release orders do not bind a device
        // address space there. This is the same device fence as Flint sync_all.
        METAL_FUNC void uocr_barrier(device atomic_uint* counter, device atomic_uint* epoch,
            uint groups, uint lid) {
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            threadgroup_barrier(mem_flags::mem_device);
            if (lid == 0) {
                uint seen = atomic_load_explicit(epoch, memory_order_relaxed);
                uint ticket = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
                if (ticket + 1u == groups) {
                    atomic_store_explicit(counter, 0u, memory_order_relaxed);
                    atomic_store_explicit(epoch, seen + 1u, memory_order_relaxed);
                } else {
                    while (atomic_load_explicit(epoch, memory_order_relaxed) == seen) {
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        }
        """

    private static func layerSource(bits: Int) -> String {
        packedAliases()
            + """

            constexpr int BITS = \(bits);
            const uint H = info[0];
            const uint HEADS = info[1];
            const uint DIM = info[2];
            const uint WIDTH = info[3];
            const uint SLAB = info[4];
            const uint TOPK = info[5];
            const uint EINTER = info[6];
            const uint SINTER = info[7];
            const int POS = position[0];
            const float SCALE = coeff[0];
            const float EPS = coeff[1];
            const uint PACK = BITS == 4 ? 8u : 4u;
            uint gid = threadgroup_position_in_grid.x;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;
            uint lid = thread_position_in_threadgroup.x;
            device atomic_uint* counter = (device atomic_uint*)sync;
            device atomic_uint* epoch = counter + 1;

            uocr_qmv_all<bfloat16_t, BITS>(qw, qs, qb, xn, work, (int)H, (int)H, gid, simd_gid, simd_lid);
            uocr_qmv_all<bfloat16_t, BITS>(kw, ks, kb, xn, kNew, (int)H, (int)H, gid, simd_gid, simd_lid);
            uocr_qmv_all<bfloat16_t, BITS>(vw, vs, vb, xn, vNew, (int)H, (int)H, gid, simd_gid, simd_lid);
            uocr_barrier(counter, epoch, \(groups), lid);
            if (gid == 0 && lid == 0) {
                for (uint i = 0; i < H; ++i) {
                    uint d = i % DIM;
                    float q0 = float(work[i]);
                    float k0 = float(kNew[i]);
                    float c = float(cosv[d]);
                    float s = float(sinv[d]);
                    float qRot = d < DIM / 2u ? -float(work[i + DIM / 2u]) : float(work[i - DIM / 2u]);
                    float kRot = d < DIM / 2u ? -float(kNew[i + DIM / 2u]) : float(kNew[i - DIM / 2u]);
                    work[i] = static_cast<bfloat16_t>(q0 * c + qRot * s);
                    kNew[i] = static_cast<bfloat16_t>(k0 * c + kRot * s);
                }
            }
            uocr_barrier(counter, epoch, \(groups), lid);
            if (gid < HEADS) {
                threadgroup float scores[512];
                threadgroup float probs[512];
                for (uint i = lid; i < WIDTH && i < 512u; i += 64u) {
                    int src = idx[i];
                    float dot = 0.0f;
                    for (uint d = 0; d < DIM; ++d) {
                        float key = (src == POS) ? float(kNew[gid * DIM + d])
                            : float(cacheK[(gid * SLAB + (uint)max(src, 0)) * DIM + d]);
                        dot += float(work[gid * DIM + d]) * key;
                    }
                    scores[i] = dot * SCALE + attnBias[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid == 0) {
                    float m = scores[0];
                    for (uint i = 1; i < WIDTH && i < 512u; ++i) m = max(m, scores[i]);
                    float sum = 0.0f;
                    for (uint i = 0; i < WIDTH && i < 512u; ++i) {
                        probs[i] = exp(scores[i] - m);
                        sum += probs[i];
                    }
                    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
                    for (uint i = 0; i < WIDTH && i < 512u; ++i) probs[i] *= inv;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint d = lid; d < DIM; d += 64u) {
                    float acc = 0.0f;
                    for (uint i = 0; i < WIDTH && i < 512u; ++i) {
                        int src = idx[i];
                        float val = (src == POS) ? float(vNew[gid * DIM + d])
                            : float(cacheV[(gid * SLAB + (uint)max(src, 0)) * DIM + d]);
                        acc += probs[i] * val;
                    }
                    work[H + gid * DIM + d] = static_cast<bfloat16_t>(acc);
                }
            }
            uocr_barrier(counter, epoch, \(groups), lid);
            // Attention output. Q in work[0:H] is finished; O reads the heads at work+H.
            uocr_qmv_all<bfloat16_t, BITS>(ow, os, ob, work + H, y, (int)H, (int)H, gid, simd_gid, simd_lid);
            uocr_barrier(counter, epoch, \(groups), lid);
            if (gid == 0 && lid == 0) {
                float sq = 0.0f;
                for (uint i = 0; i < H; ++i) {
                    float h = float(x[i]) + float(y[i]);
                    y[i] = static_cast<bfloat16_t>(h);
                    sq += h * h;
                }
                float inv = rsqrt(sq / float(H) + EPS);
                for (uint i = 0; i < H; ++i) {
                    work[i] = static_cast<bfloat16_t>(float(y[i]) * inv * float(postw[i]));
                }
            }
            uocr_barrier(counter, epoch, \(groups), lid);

            device bfloat16_t* mix = work + 2 * H + TOPK * EINTER;
            if (TOPK > 0) {
                uocr_qmv_all<bfloat16_t, BITS>(scw, scs, scb, work, work + H, (int)H, 64, gid, simd_gid, simd_lid);
                uocr_barrier(counter, epoch, \(groups), lid);
                if (gid == 0 && lid == 0) {
                    float prob[64];
                    float m = float(work[H]);
                    for (int i = 1; i < 64; ++i) m = max(m, float(work[H + i]));
                    float sum = 0.0f;
                    for (int i = 0; i < 64; ++i) {
                        prob[i] = exp(float(work[H + i]) - m);
                        sum += prob[i];
                    }
                    float inv = sum == 0.0f ? 0.0f : 1.0f / sum;
                    for (int i = 0; i < 64; ++i) prob[i] *= inv;
                    for (uint e = 0; e < TOPK; ++e) {
                        int best = 0;
                        float bv = -1.0f;
                        for (int i = 0; i < 64; ++i) {
                            bool used = false;
                            for (uint p = 0; p < e; ++p) if ((int)ids[p] == i) used = true;
                            if (!used && prob[i] > bv) { bv = prob[i]; best = i; }
                        }
                        ids[e] = (uint)best;
                        wts[e] = bv;
                    }
                }
                uocr_barrier(counter, epoch, \(groups), lid);
                for (uint e = 0; e < TOPK; ++e) {
                    uint id = ids[e];
                    device bfloat16_t* prod = work + 2 * H + e * EINTER;
                    uocr_swiglu_all<bfloat16_t, BITS>(
                        egw + id * (size_t)EINTER * (H / PACK),
                        egs + id * (size_t)EINTER * (H / 64u),
                        egb + id * (size_t)EINTER * (H / 64u),
                        euw + id * (size_t)EINTER * (H / PACK),
                        eus + id * (size_t)EINTER * (H / 64u),
                        eub + id * (size_t)EINTER * (H / 64u),
                        work, prod, (int)H, (int)EINTER, gid, simd_gid, simd_lid);
                    uocr_barrier(counter, epoch, \(groups), lid);
                    device bfloat16_t* down = work + H;
                    uocr_qmv_all<bfloat16_t, BITS>(
                        edw + id * (size_t)H * (EINTER / PACK),
                        eds + id * (size_t)H * (EINTER / 64u),
                        edb + id * (size_t)H * (EINTER / 64u),
                        prod, down, (int)EINTER, (int)H, gid, simd_gid, simd_lid);
                    uocr_barrier(counter, epoch, \(groups), lid);
                    if (gid == 0 && lid == 0) {
                        float wt = wts[e];
                        for (uint i = 0; i < H; ++i) {
                            float acc = (e == 0u ? 0.0f : float(mix[i])) + wt * float(down[i]);
                            mix[i] = static_cast<bfloat16_t>(acc);
                        }
                    }
                    uocr_barrier(counter, epoch, \(groups), lid);
                }
            }
            if (TOPK > 0 && gid == 0 && lid == 0) {
                for (uint i = 0; i < H; ++i) {
                    y[i] = static_cast<bfloat16_t>(float(y[i]) + float(mix[i]));
                }
            }
            uocr_barrier(counter, epoch, \(groups), lid);
            device bfloat16_t* sprod = work + 2 * H;
            uocr_swiglu_all<bfloat16_t, BITS>(sgw, sgs, sgb, suw, sus, sub, work, sprod, (int)H, (int)SINTER, gid, simd_gid, simd_lid);
            uocr_barrier(counter, epoch, \(groups), lid);
            device bfloat16_t* sdown = work + H;
            uocr_qmv_all<bfloat16_t, BITS>(sdw, sds, sdb, sprod, sdown, (int)SINTER, (int)H, gid, simd_gid, simd_lid);
            uocr_barrier(counter, epoch, \(groups), lid);
            if (gid == 0 && lid == 0) {
                for (uint i = 0; i < H; ++i) {
                    y[i] = static_cast<bfloat16_t>(float(y[i]) + float(sdown[i]));
                }
            }
            uocr_barrier(counter, epoch, \(groups), lid);
            """
    }
}
