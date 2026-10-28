import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import NovaMLXUtils

/// Seq-1 qmv epilogue. One MLX-style threadgroup owns 8 output rows and writes
/// `silu(gate) * up` before those rows leave the group, so the decode step does
/// not store gate and up and read them back in `compiledSiluProduct`.
///
/// The launch is `(rows / 8) * 64` threads. There is no ticket barrier and no
/// cross-group rendezvous. `NOVA_UOCR_EPILOGUE` defaults off. The row-stack
/// fuse stays the product path until a Mac invoice matches and STREAM is at
/// least the fuse.
enum UnlimitedOCRQMVEpilogue {
    static let rowsPerGroup = 8
    static let threads = 64

    static func allows(_ raw: String?) -> Bool {
        switch raw?.lowercased() {
        case "1", "on", "true", "yes":
            return true
        default:
            return false
        }
    }

    static let enabled = allows(ProcessInfo.processInfo.environment["NOVA_UOCR_EPILOGUE"])

    static func threadgroups(rows: Int) -> Int {
        guard rows > 0, rows.isMultiple(of: rowsPerGroup) else { return 0 }
        return rows / rowsPerGroup
    }

    static func totalThreads(rows: Int) -> Int {
        threadgroups(rows: rows) * threads
    }

    /// `silu(g) * up` with the same clamp the passing layer kernel uses.
    static func referenceSiluProduct(_ gate: Float, _ up: Float) -> Float {
        let clipped = min(80, max(-80, gate))
        return (gate / (1 + exp(-clipped))) * up
    }

    /// Store the dot, then add. That is the same rounding as a qmv followed by a residual add.
    static func referenceResidual(_ dot: Float, _ skip: Float) -> Float {
        dot + skip
    }

    static func kernelWritesProductInGroup() -> Bool {
        let source = swigluSource(bits: 4)
        return source.contains("threadgroup_position_in_grid")
            && source.contains("int(gid) * 8")
            && source.contains("(g / (1.0f + exp(-clipped))) * up[row]")
            && !source.contains("uocr_barrier")
            && !source.contains("uint half")
    }

    /// Gate and up for one token. `nil` keeps `compiledSiluProduct`.
    static func mlp(_ x: MLXArray, gate: Linear, up: Linear, down: Linear) -> MLXArray? {
        guard enabled, singleToken(x), x.dtype == .bfloat16 else { return nil }
        guard let g = load(gate), let u = load(up), pair(g, u) else { return nil }
        guard let activated = swiglu(x, g, u, indices: nil) else { return nil }
        return down(activated.reshaped([1, 1, g.rows]))
    }

    /// Selected experts. The result matches `compiledSiluProduct` of the gathered
    /// gate and up: `indices.shape + [1, intermediate]`.
    static func expertActivation(
        _ glu: SwitchGLU, _ x: MLXArray, _ indices: MLXArray
    ) -> MLXArray? {
        guard enabled, singleToken(x), x.dtype == .bfloat16 else { return nil }
        guard let g = load(glu.gateProj), let u = load(glu.upProj), pair(g, u), g.experts > 1
        else { return nil }
        let count = indices.size
        guard count > 0, count < 64 else { return nil }
        guard let activated = swiglu(x, g, u, indices: indices.asType(.int32)) else { return nil }
        var shape = indices.shape
        shape.append(1)
        shape.append(g.rows)
        return activated.reshaped(shape)
    }

    /// `residual + o_proj(projected)` inside the 8-row group. `nil` keeps the separate add.
    static func residual(_ projection: Linear, _ projected: MLXArray, skip: MLXArray) -> MLXArray? {
        guard enabled, singleToken(projected), singleToken(skip), projected.dtype == .bfloat16,
            skip.dtype == .bfloat16
        else { return nil }
        guard let plane = load(projection), plane.rows == plane.k, plane.experts == 1 else { return nil }
        guard let y = projectAdd(projected, plane, skip: skip) else { return nil }
        return y.reshaped([1, 1, plane.rows])
    }

    // MARK: launch

    private struct Plane {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
        var rows: Int
        var k: Int
        var bits: Int
        var experts: Int
    }

    private static func pair(_ gate: Plane, _ up: Plane) -> Bool {
        gate.bits == up.bits && gate.rows == up.rows && gate.k == up.k && gate.experts == up.experts
            && gate.rows.isMultiple(of: rowsPerGroup)
    }

    private static func singleToken(_ x: MLXArray) -> Bool {
        x.ndim == 3 && x.dim(0) == 1 && x.dim(1) == 1
    }

    private static func swiglu(
        _ x: MLXArray, _ gate: Plane, _ up: Plane, indices: MLXArray?
    ) -> MLXArray? {
        let assignments = indices?.size ?? 1
        guard assignments > 0, let kernel = swigluKernel(bits: gate.bits) else { return nil }
        let ids = (indices ?? MLXArray([Int32(0)])).reshaped([assignments]).asType(.int32)
        let info = MLXArray([UInt32(gate.k), UInt32(gate.rows), UInt32(assignments)])
        let tiles = threadgroups(rows: gate.rows)
        guard tiles > 0 else { return nil }
        let y = kernel(
            [
                x.reshaped([gate.k]),
                gate.weight, gate.scales, gate.biases,
                up.weight, up.scales, up.biases,
                ids, info,
            ],
            grid: (totalThreads(rows: gate.rows), 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[assignments, gate.rows]],
            outputDTypes: [.bfloat16])
        note()
        return y[0]
    }

    private static func projectAdd(_ projected: MLXArray, _ plane: Plane, skip: MLXArray) -> MLXArray? {
        guard let kernel = residualKernel(bits: plane.bits) else { return nil }
        let tiles = threadgroups(rows: plane.rows)
        guard tiles > 0 else { return nil }
        let info = MLXArray([UInt32(plane.k), UInt32(plane.rows)])
        let y = kernel(
            [
                projected.reshaped([plane.k]),
                plane.weight, plane.scales, plane.biases,
                skip.reshaped([plane.rows]),
                info,
            ],
            grid: (totalThreads(rows: plane.rows), 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[plane.rows]],
            outputDTypes: [.bfloat16])
        note()
        return y[0]
    }

    private static func load(_ linear: Linear) -> Plane? {
        load(dictionary(linear.parameters().flattened()))
    }

    private static func load(_ linear: SwitchLinear) -> Plane? {
        load(dictionary(linear.parameters().flattened()))
    }

    private static func load(_ flat: [String: MLXArray]) -> Plane? {
        guard flat["bias"] == nil, let weight = flat["weight"],
            let scales = flat["scales"], let biases = flat["biases"],
            weight.dtype == .uint32, scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let experts = weight.ndim == 3 ? weight.dim(0) : 1
        let rows = weight.dim(weight.ndim - 2)
        let groups = scales.dim(-1)
        guard let bits = UnlimitedOCRDecodeFuse.affineBits(packed: weight.dim(-1), groups: groups),
            bits.group == 64, rows.isMultiple(of: rowsPerGroup), rows > 0
        else { return nil }
        return Plane(
            weight: weight, scales: scales, biases: biases,
            rows: rows, k: groups * 64, bits: bits.bits, experts: experts)
    }

    private static func dictionary(_ pairs: [(String, MLXArray)]) -> [String: MLXArray] {
        var flat: [String: MLXArray] = [:]
        for (key, value) in pairs { flat[key] = value }
        return flat
    }

    // MARK: kernels

    private static let lock = NSLock()
    private nonisolated(unsafe) static var swigluKernels: [Int: MLXFast.MLXFastKernel] = [:]
    private nonisolated(unsafe) static var residualKernels: [Int: MLXFast.MLXFastKernel] = [:]
    private nonisolated(unsafe) static var didLog = false

    private static func swigluKernel(bits: Int) -> MLXFast.MLXFastKernel? {
        cached(&swigluKernels, bits: bits, name: "uocr_swiglu_epilogue_\(bits)") {
            swigluSource(bits: bits)
        }
    }

    private static func residualKernel(bits: Int) -> MLXFast.MLXFastKernel? {
        cached(&residualKernels, bits: bits, name: "uocr_residual_epilogue_\(bits)") {
            residualSource(bits: bits)
        }
    }

    private static func cached(
        _ table: inout [Int: MLXFast.MLXFastKernel], bits: Int, name: String,
        source: () -> String
    ) -> MLXFast.MLXFastKernel? {
        lock.lock()
        if let hit = table[bits] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        let built = MLXFast.metalKernel(
            name: name,
            inputNames: name.contains("swiglu")
                ? ["x", "gw", "gs", "gb", "uw", "us", "ub", "expertIndex", "info"]
                : ["x", "w", "s", "b", "skip", "info"],
            outputNames: ["y"],
            source: source(),
            header: FlintM4MaxPrefillSteel.header,
            ensureRowContiguous: true)
        lock.lock()
        if table[bits] == nil { table[bits] = built }
        let stored = table[bits]
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
            "[UnlimitedOCR] qmv epilogue writes silu(gate)*up on 8-row threadgroups. No grid barrier. Flag defaults off."
        )
    }

    static func swigluSource(bits: Int) -> String {
        """
            constexpr int BITS = \(bits);
            constexpr int pack_factor = get_pack_factor<BITS, 32>();
            constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
            constexpr int values_per_thread = pack_factor;
            constexpr int block_size = values_per_thread * 32;
            constexpr int rps = 4;
            const int K = int(info[0]);
            const int N = int(info[1]);
            const uint assignments = info[2];
            const int in_w = K * bytes_per_pack / pack_factor;
            const int in_g = K / 64;
            uint gid = threadgroup_position_in_grid.x;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;
            const int base = int(gid) * 8;
            if (base >= N) return;
            const int out_row = int(simd_gid) * rps;
            for (uint e = 0; e < assignments; ++e) {
                uint id = (uint)expertIndex[e];
                size_t row0 = (size_t)id * (size_t)N + (size_t)(base + out_row);
                const device uint8_t* gws = ((const device uint8_t*)gw) + row0 * (size_t)in_w + simd_lid * bytes_per_pack;
                const device uint8_t* uws = ((const device uint8_t*)uw) + row0 * (size_t)in_w + simd_lid * bytes_per_pack;
                const device bfloat16_t* gsc = gs + row0 * (size_t)in_g + simd_lid / (64 / values_per_thread);
                const device bfloat16_t* gbc = gb + row0 * (size_t)in_g + simd_lid / (64 / values_per_thread);
                const device bfloat16_t* usc = us + row0 * (size_t)in_g + simd_lid / (64 / values_per_thread);
                const device bfloat16_t* ubc = ub + row0 * (size_t)in_g + simd_lid / (64 / values_per_thread);
                const device bfloat16_t* xv = x + simd_lid * values_per_thread;
                float gatev[rps] = {0, 0, 0, 0};
                float up[rps] = {0, 0, 0, 0};
                int k = 0;
                for (; k + block_size <= K; k += block_size) {
                    float x_thread[values_per_thread];
                    float sum = load_vector<bfloat16_t, float, values_per_thread, BITS>(xv, x_thread);
                    for (int row = 0; row < rps; ++row) {
                        float s = float(gsc[row * in_g]);
                        float b = float(gbc[row * in_g]);
                        gatev[row] += qdot<float, values_per_thread, BITS>((const device uint8_t*)(gws + row * in_w), x_thread, s, b, sum);
                        s = float(usc[row * in_g]);
                        b = float(ubc[row * in_g]);
                        up[row] += qdot<float, values_per_thread, BITS>((const device uint8_t*)(uws + row * in_w), x_thread, s, b, sum);
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
                        sum = load_vector_safe<bfloat16_t, float, values_per_thread, BITS>(xv, x_thread, remaining);
                    }
                    for (int row = 0; row < rps; ++row) {
                        float s = float(gsc[row * in_g]);
                        float b = float(gbc[row * in_g]);
                        gatev[row] += qdot_safe<float, values_per_thread, BITS>((const device uint8_t*)(gws + row * in_w), x_thread, s, b, sum, remaining);
                        s = float(usc[row * in_g]);
                        b = float(ubc[row * in_g]);
                        up[row] += qdot_safe<float, values_per_thread, BITS>((const device uint8_t*)(uws + row * in_w), x_thread, s, b, sum, remaining);
                    }
                }
                for (int row = 0; row < rps; ++row) {
                    gatev[row] = simd_sum(gatev[row]);
                    up[row] = simd_sum(up[row]);
                    if (simd_lid == 0) {
                        float g = gatev[row];
                        float clipped = g < -80.0f ? -80.0f : (g > 80.0f ? 80.0f : g);
                        float prod = (g / (1.0f + exp(-clipped))) * up[row];
                        y[(size_t)e * (size_t)N + (size_t)(base + out_row + row)] = static_cast<bfloat16_t>(prod);
                    }
                }
            }
            """
    }

    static func residualSource(bits: Int) -> String {
        """
            constexpr int BITS = \(bits);
            constexpr int pack_factor = get_pack_factor<BITS, 32>();
            constexpr int bytes_per_pack = get_bytes_per_pack<BITS, 32>();
            constexpr int values_per_thread = pack_factor;
            constexpr int block_size = values_per_thread * 32;
            constexpr int rps = 4;
            const int K = int(info[0]);
            const int N = int(info[1]);
            const int in_w = K * bytes_per_pack / pack_factor;
            const int in_g = K / 64;
            uint gid = threadgroup_position_in_grid.x;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;
            const int base = int(gid) * 8;
            if (base >= N) return;
            const int out_row = int(simd_gid) * rps;
            const device uint8_t* ws = ((const device uint8_t*)w) + (size_t)(base + out_row) * (size_t)in_w + simd_lid * bytes_per_pack;
            const device bfloat16_t* sc = s + (size_t)(base + out_row) * (size_t)in_g + simd_lid / (64 / values_per_thread);
            const device bfloat16_t* bc = b + (size_t)(base + out_row) * (size_t)in_g + simd_lid / (64 / values_per_thread);
            const device bfloat16_t* xv = x + simd_lid * values_per_thread;
            float acc[rps] = {0, 0, 0, 0};
            int k = 0;
            for (; k + block_size <= K; k += block_size) {
                float x_thread[values_per_thread];
                float sum = load_vector<bfloat16_t, float, values_per_thread, BITS>(xv, x_thread);
                for (int row = 0; row < rps; ++row) {
                    float scale = float(sc[row * in_g]);
                    float bias = float(bc[row * in_g]);
                    acc[row] += qdot<float, values_per_thread, BITS>((const device uint8_t*)(ws + row * in_w), x_thread, scale, bias, sum);
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
                    sum = load_vector_safe<bfloat16_t, float, values_per_thread, BITS>(xv, x_thread, remaining);
                }
                for (int row = 0; row < rps; ++row) {
                    float scale = float(sc[row * in_g]);
                    float bias = float(bc[row * in_g]);
                    acc[row] += qdot_safe<float, values_per_thread, BITS>((const device uint8_t*)(ws + row * in_w), x_thread, scale, bias, sum, remaining);
                }
            }
            for (int row = 0; row < rps; ++row) {
                acc[row] = simd_sum(acc[row]);
                if (simd_lid == 0) {
                    int index = base + out_row + row;
                    bfloat16_t stored = static_cast<bfloat16_t>(acc[row]);
                    y[index] = static_cast<bfloat16_t>(float(stored) + float(skip[index]));
                }
            }
            """
    }
}
