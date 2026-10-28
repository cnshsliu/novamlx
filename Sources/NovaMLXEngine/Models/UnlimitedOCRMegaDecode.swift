import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import NovaMLXUtils

/// Seq=1 Metal kernels for the two ops row-stacking cannot express.
///
/// `swiglu` is one threadgroup per token (or per selected expert): int4/int8
/// gate and up, `silu(gate) * up` in threadgroup memory, then the down
/// projection. `attendO` gathers the 512 DASH keys, runs SDPA, and applies
/// the output projection. QKV stays the stacked `quantizedMatmul` — that
/// GEMV is already one MLX launch, and RoPE plus the KV write sit between it
/// and SDPA.
///
/// Off unless `NOVA_UOCR_MEGA=1`. The measured row-stack fuse stays the
/// default. Ineligible weights fall back to that fuse. No shapeless compile
/// and no DASH plan kernel.
enum UnlimitedOCRMegaDecode {
    struct Budget: Equatable {
        var mlxGemm: Int
        var swigluKernels: Int
        var attendKernels: Int
    }

    /// Launches once the megakernel is on, for the released 12-layer stack.
    /// QKV, router QK, the score linear, and `lm_head` stay MLX GEMMs.
    static func budget(layers: Int, denseLayers: Int) -> Budget {
        let moe = max(0, layers - denseLayers)
        return Budget(
            mlxGemm: layers + layers + moe + 1,
            swigluKernels: moe + moe + denseLayers,
            attendKernels: layers)
    }

    /// Every store into the bf16 output uses `uocr_bf` (`static_cast<bfloat16_t>`).
    static func metalOutputsUseExplicitCasts() -> Bool {
        let swiglu = swigluSource(inter: 64, hidden: 64, k: 64, bits: 4, threads: 64)
        let attend = attendSource(heads: 1, dim: 64, width: 128, bits: 4, k: 64)
        return explicitBFloatStores(swiglu) && explicitBFloatStores(attend)
            && metalHeader.contains("static_cast<bfloat16_t>")
    }

    private static func explicitBFloatStores(_ source: String) -> Bool {
        source.split(separator: "\n").allSatisfy { line in
            let text = line.trimmingCharacters(in: .whitespaces)
            guard text.contains("y[") && text.contains("=") else { return true }
            return text.contains("uocr_bf(")
        }
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

    static func nibble4(_ word: UInt32, _ lane: Int) -> Int {
        let shift = (lane / 2) * 8 + (lane % 2) * 4
        return Int((word >> shift) & 0xf)
    }

    static func byte8(_ word: UInt32, _ lane: Int) -> Int {
        Int((word >> (lane * 8)) & 0xff)
    }

    static func silu(_ value: Float) -> Float {
        let clipped = min(max(value, -80), 80)
        return value / (1 + exp(-clipped))
    }

    /// Affine int4/int8 dot. `words` holds the packed K of one output row.
    /// Group 64. This is the CPU spec the Metal loop follows.
    static func affineDot(
        x: [Float], words: [UInt32], scales: [Float], biases: [Float], bits: Int
    ) -> Float {
        let groups = scales.count
        var acc: Float = 0
        if bits == 4 {
            for group in 0..<groups {
                let scale = scales[group]
                let bias = biases[group]
                for wordIndex in 0..<8 {
                    let word = words[group * 8 + wordIndex]
                    let base = (group * 8 + wordIndex) * 8
                    for lane in 0..<8 {
                        let q = Float(nibble4(word, lane))
                        acc += x[base + lane] * (scale * q + bias)
                    }
                }
            }
        } else if bits == 8 {
            for group in 0..<groups {
                let scale = scales[group]
                let bias = biases[group]
                for wordIndex in 0..<16 {
                    let word = words[group * 16 + wordIndex]
                    let base = (group * 16 + wordIndex) * 4
                    for lane in 0..<4 {
                        let q = Float(byte8(word, lane))
                        acc += x[base + lane] * (scale * q + bias)
                    }
                }
            }
        }
        return acc
    }

    static func referenceSwiGLU(
        x: [Float],
        gate: Row, up: Row, down: Row, bits: Int
    ) -> [Float] {
        let hidden = gate.scales.count == 0 ? [] : (0..<gate.rows).map { row in
            silu(affineDot(x: x, words: gate.row(row), scales: gate.scales[row], biases: gate.biases[row], bits: bits))
                * affineDot(x: x, words: up.row(row), scales: up.scales[row], biases: up.biases[row], bits: bits)
        }
        return (0..<down.rows).map { row in
            affineDot(
                x: hidden, words: down.row(row), scales: down.scales[row], biases: down.biases[row], bits: bits)
        }
    }

    struct Row {
        var words: [UInt32]
        var scales: [[Float]]
        var biases: [[Float]]
        var packed: Int
        var rows: Int { scales.count }
        func row(_ index: Int) -> [UInt32] {
            Array(words[(index * packed)..<((index + 1) * packed)])
        }
    }

    /// One head. `keys[i]` is the gathered key. Matches the Metal softmax.
    static func referenceAttend(
        q: [Float], keys: [[Float]], values: [[Float]], mask: [Float], scale: Float
    ) -> [Float] {
        var scores = [Float]()
        for index in 0..<keys.count {
            var dot: Float = 0
            for dim in 0..<q.count {
                dot += q[dim] * keys[index][dim]
            }
            scores.append(dot * scale + mask[index])
        }
        let maxScore = scores.max() ?? 0
        var weights = scores.map { exp($0 - maxScore) }
        let sum = weights.reduce(0, +)
        if sum == 0 { return [Float](repeating: 0, count: values.first?.count ?? 0) }
        for index in weights.indices { weights[index] /= sum }
        var out = [Float](repeating: 0, count: values[0].count)
        for index in 0..<values.count {
            for dim in 0..<out.count {
                out[dim] += weights[index] * values[index][dim]
            }
        }
        return out
    }

    static func experts(
        _ glu: SwitchGLU, _ x: MLXArray, _ indices: MLXArray, _ weights: MLXArray, scale: Float
    ) -> MLXArray? {
        guard enabled, x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1, x.dtype == .bfloat16 else { return nil }
        guard let planes = Cache.shared.glu(glu), let kernel = swigluKernel(planes) else { return nil }
        let assignments = weights.dim(-1)
        guard assignments > 0, assignments < 64 else { return nil }
        let hidden = planes.down.rows
        let out = kernel.kernel(
            [
                x.reshaped([planes.gate.k]),
                planes.gate.weight, planes.gate.scales, planes.gate.biases,
                planes.up.weight, planes.up.scales, planes.up.biases,
                planes.down.weight, planes.down.scales, planes.down.biases,
                indices.reshaped([assignments]).asType(.int32),
            ],
            grid: (assignments, 1, 1),
            threadGroup: (kernel.threads, 1, 1),
            outputShapes: [[assignments, hidden]],
            outputDTypes: [.bfloat16])[0]
        note()
        let shaped = out.reshaped([1, 1, assignments, hidden])
        return weightedExpertSum(shaped, (weights * scale).asType(out.dtype))
    }

    static func mlp(_ mlp: UnlimitedOCRDenseMLP, _ x: MLXArray) -> MLXArray? {
        guard enabled, x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1, x.dtype == .bfloat16 else { return nil }
        guard let planes = Cache.shared.mlp(mlp), let kernel = swigluKernel(planes) else { return nil }
        let hidden = planes.down.rows
        let out = kernel.kernel(
            [
                x.reshaped([planes.gate.k]),
                planes.gate.weight, planes.gate.scales, planes.gate.biases,
                planes.up.weight, planes.up.scales, planes.up.biases,
                planes.down.weight, planes.down.scales, planes.down.biases,
                MLXArray([Int32(0)]),
            ],
            grid: (1, 1, 1),
            threadGroup: (kernel.threads, 1, 1),
            outputShapes: [[1, hidden]],
            outputDTypes: [.bfloat16])[0]
        note()
        return out.reshaped([1, 1, hidden])
    }

    static func attend(
        q: MLXArray, keys: MLXArray, values: MLXArray, index: MLXArray, mask: MLXArray,
        scale: Float, output: Linear, heads: Int, dim: Int, width: Int
    ) -> MLXArray? {
        guard enabled, q.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16 else {
            return nil
        }
        guard width > 0, width <= 1024, width % 128 == 0, heads > 0, dim > 0 else { return nil }
        guard let plane = Cache.shared.linear(output), plane.bits == 4 || plane.bits == 8,
            plane.rows == heads * dim, plane.k == heads * dim, plane.experts == 1
        else { return nil }
        guard let kernel = attendKernel(heads: heads, dim: dim, width: width, bits: plane.bits, k: plane.k)
        else { return nil }
        let slab = keys.dim(2)
        let y = kernel(
            [
                q, keys, values, index.asType(.int32), mask.asType(.float32),
                plane.weight, plane.scales, plane.biases,
                MLXArray([UInt32(slab)]),
                MLXArray([scale]),
            ],
            grid: (1, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[1, 1, heads * dim]],
            outputDTypes: [.bfloat16])[0]
        note()
        return y
    }

    // MARK: - Kernels

    private struct Spec {
        var kernel: MLXFast.MLXFastKernel
        var threads: Int
    }

    private struct Quant {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
        var experts: Int
        var rows: Int
        var k: Int
        var bits: Int
    }

    private struct Planes {
        var gate: Quant
        var up: Quant
        var down: Quant
    }

    private static func swigluKernel(_ planes: Planes) -> Spec? {
        guard planes.gate.bits == planes.up.bits, planes.gate.bits == planes.down.bits,
            planes.gate.k == planes.up.k, planes.down.k == planes.gate.rows,
            planes.gate.rows == planes.up.rows, planes.gate.experts == planes.up.experts,
            planes.down.experts == planes.gate.experts
        else { return nil }
        let threads = threadCount(planes.gate.rows, planes.down.rows)
        guard threads > 0 else { return nil }
        let key = "s\(planes.gate.bits)_\(planes.gate.rows)_\(planes.down.rows)_\(planes.gate.k)_\(threads)"
        return Cache.shared.kernel(key: key, threads: threads) {
            MLXFast.metalKernel(
                name: "uocr_swiglu_\(key)",
                inputNames: ["x", "gw", "gs", "gb", "uw", "us", "ub", "dw", "ds", "db", "indices"],
                outputNames: ["y"],
                source: swigluSource(
                    inter: planes.gate.rows, hidden: planes.down.rows, k: planes.gate.k,
                    bits: planes.gate.bits, threads: threads),
                header: metalHeader,
                ensureRowContiguous: true)
        }
    }

    private static func attendKernel(heads: Int, dim: Int, width: Int, bits: Int, k: Int) -> MLXFast.MLXFastKernel? {
        let key = "a\(heads)_\(dim)_\(width)_\(bits)_\(k)"
        return Cache.shared.kernel(key: key, threads: 128) {
            MLXFast.metalKernel(
                name: "uocr_attno_\(key)",
                inputNames: ["q", "kc", "vc", "idx", "bias", "ow", "os", "ob", "info", "coeff"],
                outputNames: ["y"],
                source: attendSource(heads: heads, dim: dim, width: width, bits: bits, k: k),
                header: metalHeader,
                ensureRowContiguous: true)
        }?.kernel
    }

    private static func threadCount(_ inter: Int, _ hidden: Int) -> Int {
        for threads in [128, 64, 32] where inter % threads == 0 && hidden % threads == 0 {
            return threads
        }
        return 0
    }

    private static let metalHeader = """
        #include <metal_stdlib>
        using namespace metal;

        inline float uocr_dot(const device uint* row, const device bfloat16_t* scales,
            const device bfloat16_t* biases, const device bfloat16_t* x,
            uint groups, uint bits) {
            float acc = 0.0f;
            if (bits == 4) {
                for (uint grp = 0; grp < groups; ++grp) {
                    float s = float(scales[grp]);
                    float b = float(biases[grp]);
                    for (uint w = 0; w < 8u; ++w) {
                        uint word = row[grp * 8u + w];
                        uint base = (grp * 8u + w) * 8u;
                        for (uint lane = 0; lane < 8u; ++lane) {
                            uint shift = (lane / 2u) * 8u + (lane % 2u) * 4u;
                            float q = float((word >> shift) & 0xfu);
                            acc += float(x[base + lane]) * (s * q + b);
                        }
                    }
                }
            } else {
                for (uint grp = 0; grp < groups; ++grp) {
                    float s = float(scales[grp]);
                    float b = float(biases[grp]);
                    for (uint w = 0; w < 16u; ++w) {
                        uint word = row[grp * 16u + w];
                        uint base = (grp * 16u + w) * 4u;
                        for (uint lane = 0; lane < 4u; ++lane) {
                            float q = float((word >> (lane * 8u)) & 0xffu);
                            acc += float(x[base + lane]) * (s * q + b);
                        }
                    }
                }
            }
            return acc;
        }

        inline bfloat16_t uocr_bf(float v) { return static_cast<bfloat16_t>(v); }

        inline float uocr_dot_f(const device uint* row, const device bfloat16_t* scales,
            const device bfloat16_t* biases, threadgroup float* x,
            uint groups, uint bits) {
            float acc = 0.0f;
            if (bits == 4) {
                for (uint grp = 0; grp < groups; ++grp) {
                    float s = float(scales[grp]);
                    float b = float(biases[grp]);
                    for (uint w = 0; w < 8u; ++w) {
                        uint word = row[grp * 8u + w];
                        uint base = (grp * 8u + w) * 8u;
                        for (uint lane = 0; lane < 8u; ++lane) {
                            uint shift = (lane / 2u) * 8u + (lane % 2u) * 4u;
                            float q = float((word >> shift) & 0xfu);
                            acc += x[base + lane] * (s * q + b);
                        }
                    }
                }
            } else {
                for (uint grp = 0; grp < groups; ++grp) {
                    float s = float(scales[grp]);
                    float b = float(biases[grp]);
                    for (uint w = 0; w < 16u; ++w) {
                        uint word = row[grp * 16u + w];
                        uint base = (grp * 16u + w) * 4u;
                        for (uint lane = 0; lane < 4u; ++lane) {
                            float q = float((word >> (lane * 8u)) & 0xffu);
                            acc += x[base + lane] * (s * q + b);
                        }
                    }
                }
            }
            return acc;
        }
        """

    private static func swigluSource(inter: Int, hidden: Int, k: Int, bits: Int, threads: Int) -> String {
        let groups = k / 64
        let downGroups = inter / 64
        let gateStride = groups * (bits == 4 ? 8 : 16)
        let downStride = downGroups * (bits == 4 ? 8 : 16)
        return """
            const uint INTER = \(inter);
            const uint HIDDEN = \(hidden);
            const uint BITS = \(bits);
            const uint THREADS = \(threads);
            const uint GK = \(groups);
            const uint DK = \(downGroups);
            const uint GSTRIDE = \(gateStride);
            const uint DSTRIDE = \(downStride);
            uint t = thread_position_in_threadgroup.x;
            uint slot = threadgroup_position_in_grid.x;
            int expert = indices[slot];
            threadgroup float act[\(inter)];
            if (expert < 0) {
                for (uint n = t; n < HIDDEN; n += THREADS) y[slot * HIDDEN + n] = uocr_bf(0.0f);
                return;
            }
            for (uint n = t; n < INTER; n += THREADS) {
                size_t row = size_t(expert) * INTER + n;
                float g = uocr_dot(gw + row * GSTRIDE, gs + row * GK, gb + row * GK, x, GK, BITS);
                float u = uocr_dot(uw + row * GSTRIDE, us + row * GK, ub + row * GK, x, GK, BITS);
                float clipped = g < -80.0f ? -80.0f : (g > 80.0f ? 80.0f : g);
                act[n] = (g / (1.0f + exp(-clipped))) * u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint n = t; n < HIDDEN; n += THREADS) {
                size_t row = size_t(expert) * HIDDEN + n;
                y[slot * HIDDEN + n] = uocr_bf(uocr_dot_f(dw + row * DSTRIDE, ds + row * DK, db + row * DK, act, DK, BITS));
            }
            """
    }

    private static func attendSource(heads: Int, dim: Int, width: Int, bits: Int, k: Int) -> String {
        let groups = k / 64
        return """
            const uint HEADS = \(heads);
            const uint DIM = \(dim);
            const uint WIDTH = \(width);
            const uint BITS = \(bits);
            const uint GK = \(groups);
            const float scale = coeff[0];
            const uint slab = info[0];
            uint t = thread_position_in_threadgroup.x;
            threadgroup float scores[\(width)];
            threadgroup float probs[\(width)];
            threadgroup float attn[\(heads * dim)];
            threadgroup float sharedMax;
            threadgroup float sharedSum;
            for (uint h = 0; h < HEADS; ++h) {
                for (uint i = t; i < WIDTH; i += 128u) {
                    int src = idx[i];
                    if (src < 0 || uint(src) >= slab) src = 0;
                    float dot = 0.0f;
                    for (uint d = 0; d < DIM; ++d) {
                        dot += float(q[h * DIM + d]) * float(kc[(h * slab + uint(src)) * DIM + d]);
                    }
                    scores[i] = dot * scale + bias[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (t == 0) {
                    float m = scores[0];
                    for (uint i = 1; i < WIDTH; ++i) m = max(m, scores[i]);
                    sharedMax = m;
                    float sum = 0.0f;
                    for (uint i = 0; i < WIDTH; ++i) {
                        float e = exp(scores[i] - m);
                        probs[i] = e;
                        sum += e;
                    }
                    sharedSum = sum == 0.0f ? 1.0f : sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (t == 0) {
                    for (uint i = 0; i < WIDTH; ++i) probs[i] = probs[i] / sharedSum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint d = t; d < DIM; d += 128u) {
                    float acc = 0.0f;
                    for (uint i = 0; i < WIDTH; ++i) {
                        int src = idx[i];
                        if (src < 0 || uint(src) >= slab) src = 0;
                        acc += probs[i] * float(vc[(h * slab + uint(src)) * DIM + d]);
                    }
                    attn[h * DIM + d] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (uint n = t; n < HEADS * DIM; n += 128u) {
                y[n] = uocr_bf(uocr_dot_f(ow + n * (GK * (BITS == 4u ? 8u : 16u)), os + n * GK, ob + n * GK, attn, GK, BITS));
            }
            """
    }

    private static func note() {
        Cache.shared.logOnce()
    }

    private final class Cache: @unchecked Sendable {
        static let shared = Cache()
        let lock = NSLock()
        final class Box<T> {
            var value: T?
            init(_ value: T?) { self.value = value }
        }

        var gluPlanes: [ObjectIdentifier: Box<Planes>] = [:]
        var mlpPlanes: [ObjectIdentifier: Box<Planes>] = [:]
        var linears: [ObjectIdentifier: Box<Quant>] = [:]
        var kernels: [String: Spec] = [:]
        var didLog = false

        func glu(_ module: SwitchGLU) -> Planes? {
            let key = ObjectIdentifier(module)
            lock.lock()
            if let cached = gluPlanes[key] {
                lock.unlock()
                return cached.value
            }
            lock.unlock()
            let loaded = loadGLU(module)
            lock.lock()
            if gluPlanes[key] == nil { gluPlanes[key] = Box(loaded) }
            let stored = gluPlanes[key]?.value
            lock.unlock()
            return stored
        }

        func mlp(_ module: UnlimitedOCRDenseMLP) -> Planes? {
            let key = ObjectIdentifier(module)
            lock.lock()
            if let cached = mlpPlanes[key] {
                lock.unlock()
                return cached.value
            }
            lock.unlock()
            let flat = dictionary(module.parameters().flattened())
            let loaded = planes(flat, gate: "gate_proj", up: "up_proj", down: "down_proj")
            lock.lock()
            if mlpPlanes[key] == nil { mlpPlanes[key] = Box(loaded) }
            let stored = mlpPlanes[key]?.value
            lock.unlock()
            return stored
        }

        func linear(_ module: Linear) -> Quant? {
            let key = ObjectIdentifier(module)
            lock.lock()
            if let cached = linears[key] {
                lock.unlock()
                return cached.value
            }
            lock.unlock()
            let loaded = quant(dictionary(module.parameters().flattened()), prefix: "")
            lock.lock()
            if linears[key] == nil { linears[key] = Box(loaded) }
            let stored = linears[key]?.value
            lock.unlock()
            return stored
        }

        func kernel(key: String, threads: Int, make: () -> MLXFast.MLXFastKernel) -> Spec? {
            lock.lock()
            if let cached = kernels[key] {
                lock.unlock()
                return cached
            }
            lock.unlock()
            let built = Spec(kernel: make(), threads: threads)
            lock.lock()
            if kernels[key] == nil { kernels[key] = built }
            let stored = kernels[key]
            lock.unlock()
            return stored
        }

        func logOnce() {
            lock.lock()
            let first = !didLog
            didLog = true
            lock.unlock()
            guard first else { return }
            let budget = UnlimitedOCRMegaDecode.budget(layers: 12, denseLayers: 1)
            NovaMLXLog.info(
                "[UnlimitedOCR] seq=1 megakernel on: SwiGLU silu→down ×\(budget.swigluKernels), 512-key SDPA+O ×\(budget.attendKernels). QKV stays stacked (\(budget.mlxGemm) MLX GEMMs)."
            )
        }

        private func loadGLU(_ module: SwitchGLU) -> Planes? {
            planes(dictionary(module.parameters().flattened()), gate: "gate_proj", up: "up_proj", down: "down_proj")
        }

        private func planes(_ flat: [String: MLXArray], gate: String, up: String, down: String) -> Planes? {
            guard let g = quant(flat, prefix: gate), let u = quant(flat, prefix: up), let d = quant(flat, prefix: down)
            else { return nil }
            guard g.bits == u.bits, g.bits == d.bits, g.k == u.k, d.k == g.rows, g.rows == u.rows,
                g.experts == u.experts, d.experts == g.experts, g.k % 64 == 0, g.rows % 64 == 0
            else { return nil }
            return Planes(gate: g, up: u, down: d)
        }

        private func quant(_ flat: [String: MLXArray], prefix: String) -> Quant? {
            let name = prefix.isEmpty ? "" : prefix + "."
            if flat[name + "bias"] != nil { return nil }
            guard let weight = flat[name + "weight"], let scales = flat[name + "scales"],
                let biases = flat[name + "biases"], weight.dtype == .uint32,
                scales.dtype == .bfloat16, biases.dtype == .bfloat16,
                weight.ndim == scales.ndim, biases.shape == scales.shape
            else { return nil }
            let experts = weight.ndim == 3 ? weight.dim(0) : 1
            let rows = weight.dim(weight.ndim - 2)
            let groups = scales.dim(scales.ndim - 1)
            guard let bits = UnlimitedOCRDecodeFuse.affineBits(packed: weight.dim(-1), groups: groups),
                bits.group == 64
            else { return nil }
            if weight.ndim == 3, scales.dim(0) != experts || scales.dim(1) != rows { return nil }
            return Quant(
                weight: weight, scales: scales, biases: biases,
                experts: experts, rows: rows, k: groups * 64, bits: bits.bits)
        }

        private func dictionary(_ pairs: [(String, MLXArray)]) -> [String: MLXArray] {
            var flat: [String: MLXArray] = [:]
            for (key, value) in pairs { flat[key] = value }
            return flat
        }
    }
}
