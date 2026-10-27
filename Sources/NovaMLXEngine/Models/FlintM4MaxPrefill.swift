import Foundation
import Metal
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import NovaMLXUtils

// MARK: - M4 Max prefill fused gate+up

/// Official `rec` prefill gate+up (`m4_max_prefill.py` / `kernels/m4_max_prefill.py`).
///
/// `build_decoder` installs this only when the chip is M4 Max. `PrefillSwitchGLU`
/// still calls `SwitchGLU` when the projection is not 4-bit or when
/// `indices.size < num_experts * 4`. The kernel is mlx steel `BlockMMA` plus
/// `QuantizedBlockLoader` (tile 16×32×32, group 64, 4-bit), the same body as
/// `flint_prefill_gate_up`. Down projection stays `gatherQuantizedMM`, which
/// is `projection.down_proj` in rec. Activation is `silu(gate) * up`.
///
/// Decode does not use this. The resident shader, the 76-worker pool, `dot16`,
/// and bf16 KV are unchanged. A chunk under the row threshold, a non-M4-Max
/// device, or any weight that is not affine int4 group 64 with bf16 scales
/// stays on `SwitchGLU`.
enum FlintM4MaxPrefill {
    enum Route: Equatable, Sendable {
        /// M4 Max and `NOVA_FLINT_PREFILL_GATEUP` not disabled.
        case automatic
        /// Run the steel kernel when the weight contract matches. The speed
        /// test uses this so one process can time the fused path.
        case fused
        /// Always `SwitchGLU`. Shell equivalent: `NOVA_FLINT_PREFILL_GATEUP=0`.
        case switchGLU
    }

    /// `rec` `PrefillSwitchGLU`: `indices.size >= gate.weight.shape[0] * 4`.
    static let rowFactor = 4
    static let groupSize = 64
    static let bits = 4
    /// Tile from `kernels/m4_max_prefill.py` (`bm, bn, bk, wm, wn = 16, 32, 32, 1, 2`).
    static let bm = 16
    static let bn = 32
    static let bk = 32
    static let wm = 1
    static let wn = 2

    static let deviceName = MTLCreateSystemDefaultDevice()?.name ?? ""

    static let envAllows = envAllowsFused(
        ProcessInfo.processInfo.environment["NOVA_FLINT_PREFILL_GATEUP"])

    private static let lock = NSLock()
    private nonisolated(unsafe) static var routeStorage: Route = .automatic
    private nonisolated(unsafe) static var fusedCallsStorage = 0
    private nonisolated(unsafe) static var logged: Set<String> = []
    private nonisolated(unsafe) static var planeCache: [ObjectIdentifier: CacheEntry] = [:]

    static var route: Route {
        get {
            lock.lock()
            defer { lock.unlock() }
            return routeStorage
        }
        set {
            lock.lock()
            routeStorage = newValue
            lock.unlock()
        }
    }

    /// Swift calls that built the fused gate+up graph. One per MoE layer per
    /// forward, not per token.
    static var fusedCalls: Int {
        lock.lock()
        defer { lock.unlock() }
        return fusedCallsStorage
    }

    static func deviceIsM4Max(_ name: String) -> Bool {
        name.lowercased().contains("m4 max")
    }

    /// `0` / `off` / `false` / `no` keep `SwitchGLU`. Anything else, including
    /// an unset variable, leaves the chip check in charge.
    static func envAllowsFused(_ raw: String?) -> Bool {
        guard let raw else { return true }
        switch raw.lowercased() {
        case "0", "off", "false", "no":
            return false
        default:
            return true
        }
    }

    static func meetsRowThreshold(indexCount: Int, experts: Int) -> Bool {
        experts > 0 && indexCount >= experts * rowFactor
    }

    /// Fused gate+up, or nil when the caller should use `SwitchGLU`.
    /// `route` overrides the process-wide ``route`` for this call. Tests pass
    /// it so they do not race the prefill A/B, which sets the process route
    /// around a real `FlintSparseMoeBlock` forward.
    static func project(
        _ experts: SwitchGLU, x: MLXArray, indices: MLXArray, route override: Route? = nil
    ) -> MLXArray? {
        guard shouldFuse(override) else { return nil }
        guard let planes = planes(for: experts) else { return nil }
        guard meetsRowThreshold(indexCount: indices.size, experts: planes.gate.experts) else {
            return nil
        }
        guard x.dtype == .bfloat16, x.dim(-1) == planes.gate.k, indices.size > 0 else {
            logOnce(
                "activation",
                "[Flint] prefill gate+up stayed on SwitchGLU: activation \(x.dtype) x[-1]=\(x.dim(-1)) k=\(planes.gate.k)"
            )
            return nil
        }

        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let (sortedX, sortedIndices, inverse) = gatherSort(x: expanded, indices: indices)
        guard let (gate, up) = gateUp(
            sortedX: sortedX, gate: planes.gate, up: planes.up, sortedIndices: sortedIndices)
        else { return nil }

        // rec `SwiGLU`: `swiglu(gate, up)` is `silu(gate) * up`.
        let activated = MLXNN.silu(gate) * up
        let down = MLX.gatherQuantizedMM(
            activated,
            planes.down.weight,
            scales: planes.down.scales,
            biases: planes.down.biases,
            rhsIndices: sortedIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: true
        )
        let restored = scatterUnsort(x: down, invOrder: inverse, shape: Array(indices.shape))
        noteFusedCall(rows: indices.size, experts: planes.gate.experts)
        return MLX.squeezed(restored, axis: -2)
    }

    // MARK: kernel

    /// Sorted rows in, gate and up out, shaped like `sortedX` with the last
    /// axis replaced by the intermediate size. Nil when the tile contract fails.
    static func gateUp(
        sortedX: MLXArray,
        gate: Plane,
        up: Plane,
        sortedIndices: MLXArray
    ) -> (MLXArray, MLXArray)? {
        let k = gate.k
        guard gate.experts == up.experts, gate.n == up.n, gate.k == up.k,
            k > 0, sortedX.size % k == 0
        else { return nil }
        let m = sortedX.size / k
        let n = gate.n
        let experts = gate.experts
        guard n % bn == 0, k % bk == 0, m > 0, experts > 0 else { return nil }

        let flatIndices = sortedIndices.flattened()
        let starts = boundaryKernel(
            [flatIndices, MLXArray([UInt32(m), UInt32(experts)], [2])],
            grid: (experts + 1, 1, 1),
            threadGroup: (min(256, max(experts + 1, 1)), 1, 1),
            outputShapes: [[experts + 1]],
            outputDTypes: [.uint32]
        )[0]
        let counts = starts[1...] - starts[..<experts]
        // Integer division. `/` on MLX integer arrays can promote; floorDivide
        // is the uint32 tile count rec writes with `//`.
        let tileCounts = (counts + MLXArray(UInt32(bm - 1))).floorDivide(MLXArray(UInt32(bm)))
        let tiles = concatenated(
            [MLXArray.zeros([1], dtype: .uint32), cumsum(tileCounts, axis: 0)],
            axis: 0
        )
        var outShape = Array(sortedX.shape)
        outShape[outShape.count - 1] = n
        let threadsX = (n / bn) * 32 * wm * wn
        let tilesY = (m + bm - 1) / bm + experts
        let outputs = gateUpKernel(
            [
                sortedX,
                gate.weight,
                gate.scales,
                gate.biases,
                up.weight,
                up.scales,
                up.biases,
                starts,
                tiles,
                MLXArray([UInt32(m), UInt32(n), UInt32(k), UInt32(experts)], [4]),
            ],
            grid: (threadsX, tilesY, 1),
            threadGroup: (32 * wm * wn, 1, 1),
            outputShapes: [[m, n], [m, n]],
            outputDTypes: [sortedX.dtype, sortedX.dtype]
        )
        return (outputs[0].reshaped(outShape), outputs[1].reshaped(outShape))
    }

    /// Body of `kernels/m4_max_prefill.py` `_kernel()`. `MLXFast.metalKernel`
    /// has no compile-options argument, so this does not pass rec's
    /// `math_mode=safe`. The body is integer indexing plus bf16 MMA.
    static let kernelSource = """
        using T=bfloat16_t;
        constexpr int BM=\(bm),BN=\(bn),BK=\(bk),WM=\(wm),WN=\(wn);
        constexpr int PAD=BK+8;
        uint3 tid=threadgroup_position_in_grid;
        uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
        uint M=dimensions[0],N=dimensions[1],K=dimensions[2],E=dimensions[3];
        (void)M;
        if(tid.y>=tiles[E])return;
        uint lo=0,hi=E;
        while(lo<hi) {uint mid=(lo+hi)/2;if(tiles[mid+1]<=tid.y)lo=mid+1;else hi=mid;}
        if(lo==E)return;
        uint expert=lo,row=starts[expert]+(tid.y-tiles[expert])*BM,col=tid.x*BN;
        short valid=short(min(uint(BM),starts[expert+1]-row));
        using mma_t=mlx::steel::BlockMMA<T,T,BM,BN,BK,WM,WN,false,true,PAD,PAD>;
        using loader_x_t=mlx::steel::BlockLoader<T,BM,BK,PAD,1,WM*WN*32>;
        using loader_w_t=QuantizedBlockLoader<T,BN,BK,PAD,true,WM*WN*32,64,4>;
        threadgroup T Xs[BM*PAD],Gs[BN*PAD],Us[BN*PAD];
        thread mma_t mg(sg,lane),mu(sg,lane);
        thread loader_x_t lx(x+size_t(row)*K,K,Xs,sg,lane);
        size_t weight_offset=(size_t(expert)*N+col)*(K/2);
        size_t scale_offset=(size_t(expert)*N+col)*(K/64);
        thread loader_w_t lg(((const device uchar*)gw)+weight_offset,gs+scale_offset,gb+scale_offset,K,Gs,sg,lane);
        thread loader_w_t lu(((const device uchar*)uw)+weight_offset,us+scale_offset,ub+scale_offset,K,Us,sg,lane);
        for(uint at=0;at<K;at+=BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(valid==BM)lx.load_unsafe();else lx.load_safe(short2(BK,valid));
            lg.load_unsafe();lu.load_unsafe();
            threadgroup_barrier(mem_flags::mem_threadgroup);
            mg.mma(Xs,Gs);mu.mma(Xs,Us);
            lx.next();lg.next();lu.next();
        }
        if(valid==BM) {mg.store_result(gate+size_t(row)*N+col,N);mu.store_result(up+size_t(row)*N+col,N);}
        else {mg.store_result_slice(gate+size_t(row)*N+col,N,short2(0,0),short2(BN,valid));mu.store_result_slice(up+size_t(row)*N+col,N,short2(0,0),short2(BN,valid));}
        """

    static let boundarySource = """
        uint expert=thread_position_in_grid.x;
        if(expert<=dimensions[1]) {
            uint lo=0,hi=dimensions[0];
            while(lo<hi) {uint mid=(lo+hi)/2;if(indices[mid]<expert)lo=mid+1;else hi=mid;}
            starts[expert]=lo;
        }
        """

    // MARK: weight contract

    struct Plane {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
        var experts: Int
        var n: Int
        var k: Int
    }

    private struct Planes {
        var gate: Plane
        var up: Plane
        var down: Plane
    }

    private enum CacheEntry {
        case planes(Planes)
        case ineligible(String)
    }

    private static func shouldFuse(_ override: Route?) -> Bool {
        switch override ?? route {
        case .switchGLU:
            return false
        case .fused:
            return true
        case .automatic:
            return envAllows && deviceIsM4Max(deviceName)
        }
    }

    private static func planes(for experts: SwitchGLU) -> Planes? {
        let id = ObjectIdentifier(experts)
        lock.lock()
        let cached = planeCache[id]
        lock.unlock()
        if let cached {
            if case .planes(let planes) = cached { return planes }
            return nil
        }
        let loaded = loadPlanes(experts)
        lock.lock()
        if planeCache[id] == nil {
            switch loaded {
            case .planes(let planes):
                planeCache[id] = .planes(planes)
            case .ineligible(let reason):
                planeCache[id] = .ineligible(reason)
            }
        }
        let stored = planeCache[id]
        lock.unlock()
        switch stored {
        case .planes(let planes):
            return planes
        case .ineligible(let reason):
            logOnce("planes", "[Flint] prefill gate+up stayed on SwitchGLU: \(reason)")
            return nil
        case nil:
            return nil
        }
    }

    private static func loadPlanes(_ experts: SwitchGLU) -> CacheEntry {
        var flat: [String: MLXArray] = [:]
        for (key, value) in experts.parameters().flattened() {
            flat[key] = value
        }
        if flat["gate_proj.bias"] != nil || flat["up_proj.bias"] != nil
            || flat["down_proj.bias"] != nil
        {
            return .ineligible("expert projection has an additive bias")
        }
        guard let gate = plane(flat, "gate_proj"),
            let up = plane(flat, "up_proj"),
            let down = plane(flat, "down_proj")
        else {
            let dtype = flat["gate_proj.scales"]?.dtype
            let shape = flat["gate_proj.weight"]?.shape ?? []
            return .ineligible(
                "need affine int4 group \(groupSize) bf16 scales, gate weight \(shape) scales \(String(describing: dtype))"
            )
        }
        guard gate.experts == up.experts, gate.n == up.n, gate.k == up.k,
            down.experts == gate.experts, down.k == gate.n
        else {
            return .ineligible(
                "gate/up/down shapes disagree gate \(gate.experts)x\(gate.n)x\(gate.k) down k=\(down.k)"
            )
        }
        guard gate.n % bn == 0, gate.k % bk == 0 else {
            return .ineligible("intermediate \(gate.n) or hidden \(gate.k) is not a tile multiple")
        }
        return .planes(Planes(gate: gate, up: up, down: down))
    }

    /// Affine 4-bit, group 64, bf16 scales and biases. Accepts the uint32
    /// packing MLX stores (`K/8` uints per row) and a raw uint8 nibble row
    /// (`K/2` bytes). Both are the byte layout `QuantizedBlockLoader` reads.
    private static func plane(_ flat: [String: MLXArray], _ name: String) -> Plane? {
        guard let weight = flat["\(name).weight"],
            let scales = flat["\(name).scales"],
            let biases = flat["\(name).biases"],
            weight.ndim == 3, scales.ndim == 3, biases.shape == scales.shape,
            scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let experts = weight.dim(0)
        let n = weight.dim(1)
        let groups = scales.dim(2)
        guard experts > 0, n > 0, groups > 0,
            scales.dim(0) == experts, scales.dim(1) == n
        else { return nil }
        let k = groups * groupSize
        switch weight.dtype {
        case .uint32:
            guard weight.dim(2) * 8 == k else { return nil }
        case .uint8:
            guard weight.dim(2) * 2 == k else { return nil }
        default:
            return nil
        }
        return Plane(
            weight: weight, scales: scales, biases: biases,
            experts: experts, n: n, k: k)
    }

    private static func noteFusedCall(rows: Int, experts: Int) {
        lock.lock()
        fusedCallsStorage += 1
        let first = !logged.contains("fused")
        if first { logged.insert("fused") }
        lock.unlock()
        if first {
            NovaMLXLog.info(
                "[Flint] M4 Max prefill fused gate+up (steel BlockMMA, rows=\(rows) experts=\(experts) threshold=\(experts * rowFactor))"
            )
        }
    }

    private static func logOnce(_ key: String, _ message: String) {
        lock.lock()
        let first = !logged.contains(key)
        if first { logged.insert(key) }
        lock.unlock()
        if first { NovaMLXLog.info(message) }
    }

    private static let boundaryKernel = MLXFast.metalKernel(
        name: "flint_prefill_expert_boundaries",
        inputNames: ["indices", "dimensions"],
        outputNames: ["starts"],
        source: boundarySource,
        header: "",
        ensureRowContiguous: true
    )

    private static let gateUpKernel = MLXFast.metalKernel(
        name: "flint_prefill_gate_up",
        inputNames: ["x", "gw", "gs", "gb", "uw", "us", "ub", "starts", "tiles", "dimensions"],
        outputNames: ["gate", "up"],
        source: kernelSource,
        header: FlintM4MaxPrefillSteel.header,
        ensureRowContiguous: true
    )
}
