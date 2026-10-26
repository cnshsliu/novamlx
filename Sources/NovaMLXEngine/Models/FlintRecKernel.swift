import Foundation
import Metal
import MLX
import MLXFast
import NovaMLXCore

// MARK: - Launch profile

/// Resident worker pool baked into the Metal shader. M4 Pro / M5 Pro stay on
/// the 36/32/32 shader. M4 Max / M5 Max use 76/64/64, the pool rec measures
/// on those chips. `NOVA_FLINT_REC_WORKERS=36|76` overrides the device name.
/// 76 threadgroups must stay resident; on a 20-core GPU the spin wait fails
/// the diagnostic and decode falls back to eager.
struct FlintRecLaunch: Equatable, Sendable {
    let workers: Int
    let mlpWorkers: Int
    let headWorkers: Int

    static let pro = FlintRecLaunch(workers: 36, mlpWorkers: 32, headWorkers: 32)
    static let max = FlintRecLaunch(workers: 76, mlpWorkers: 64, headWorkers: 64)

    var label: String { "\(workers)/\(mlpWorkers)/\(headWorkers)" }

    static func resolve(deviceName: String, override: String? = nil) -> FlintRecLaunch {
        switch override?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "76", "max", "m4max", "m5max":
            return .max
        case "36", "pro", "m4pro", "m5pro":
            return .pro
        default:
            break
        }
        let name = deviceName.lowercased()
        if name.contains("m4 max") || name.contains("m5 max") {
            return .max
        }
        return .pro
    }

    static func current() -> FlintRecLaunch {
        let device = MTLCreateSystemDefaultDevice()?.name ?? ""
        return resolve(
            deviceName: device,
            override: ProcessInfo.processInfo.environment["NOVA_FLINT_REC_WORKERS"])
    }
}

// MARK: - Flint rec-engine decode kernel (port of recursive-oss/rec)

/// Swift port of Recursive's `rec` fused decode megakernel
/// (engine/src/rec_mlx/kernels/megakernel.py, Apache-2.0). One Metal kernel
/// covers the whole 49-layer decode forward — RMSNorm → RoPE → sink
/// attention → sigmoid routing → top-6 MoE → shared expert → residual —
/// against 256-byte-aligned packed weight banks and block-structured KV
/// storage. Flint-only: no other model family touches this path.
///
/// Decode-only, matching `QATSession`: prefill stays on the eager model,
/// which fills MLX caches; those caches are copied into this KV state and
/// `initial` is frozen at the prompt length. `forward` consumes one token
/// and returns logits. `forwardGreedy` is rec's resident packet: up to
/// `FlintRecResident.width` greedy tokens per launch, argmax on device,
/// which is the path behind the published ~197 tok/s INT4 number.
///
/// The shader is rendered for two resident pools (see `FlintRecKernelSources`).
/// Large scratch (workspace, attention partials, statistics, routing, trace)
/// lives on the session and is passed back in. Logits stay a kernel output
/// on both paths. MLX marks inputs const, and a const-cast is not a reliable
/// publish for the in-kernel argmax: the packet then repeats one token id.
/// The sync diagnostic stays an output so a failed resident spin is visible.
/// Single-token `forward` uploads a fresh zero control buffer every launch —
/// a dirty atomic counter deadlocks the device spin. The greedy packet
/// instead reuses the session buffer and `release_control` zeros it on
/// device after the launch. Set `NOVA_FLINT_REC_FRESH_CONTROL=1` to force
/// the upload on that path too.
public final class FlintRecKernel: @unchecked Sendable {
    private static let decodeThreads = 256
    private static let kvBlocks = 4
    private static let blockTokens = 65536
    /// Vocab the shader hardcodes (`word<150144u`, logits length).
    static let vocabulary = 150144
    /// Escape hatch when a reused control buffer is not visible to the next
    /// launch (symptom: GPU sync failure on the second greedy packet).
    static let uploadsFreshControl =
        ProcessInfo.processInfo.environment["NOVA_FLINT_REC_FRESH_CONTROL"] == "1"

    let launch: FlintRecLaunch
    private let kernel: MLXFast.MLXFastKernel
    /// Nil when the resident-packet source did not match this shader.
    /// Single-token `forward` does not need it.
    private let greedyKernel: MLXFast.MLXFastKernel?
    private let writeKernel: MLXFast.MLXFastKernel
    /// Bound for KV blocks this session has not grown into. The shader only
    /// dereferences block 0 below 65536 tokens.
    private let kvPlaceholder: MLXArray

    /// 13 packed weight banks (uint8 views).
    private(set) var banks: [MLXArray] = []
    /// [50, 64] uint32 layer metadata.
    private(set) var metadata: MLXArray
    private let siluTable: MLXArray
    private let configuration: MLXArray

    /// Buffers the megakernel overwrites every token. Official rec allocates
    /// these as kernel outputs (about 8.6 MB, most of it attention partials).
    /// Holding them on the session removes that alloc from the decode loop.
    final class DecodeScratch {
        let trace: MLXArray
        let workspace: MLXArray
        let partials: MLXArray
        let statistics: MLXArray
        let routing: MLXArray
        /// Greedy-packet spin counters. `release_control` restores zeros.
        /// Replaced when a launch fails before that epilogue runs.
        var control: MLXArray

        init() {
            // trace [50, 2048] bf16, workspace 78848 bf16,
            // partials 32 * 1024 * 128 / 2 f32, statistics 32 * 1024 * 2 f32,
            // routing 1280 u32. Same shapes as rec's `decode_scratch`.
            // control is 1024 u32. Logits are a kernel output, not scratch:
            // see `FlintRecKernel` (const inputs are not a safe argmax source).
            trace = MLXArray.zeros([50 * 2048], dtype: .bfloat16)
            workspace = MLXArray.zeros([78848], dtype: .bfloat16)
            partials = MLXArray.zeros([32 * 1024 * 128 / 2], dtype: .float32)
            statistics = MLXArray.zeros([32 * 1024 * 2], dtype: .float32)
            routing = MLXArray.zeros([1280], dtype: .uint32)
            control = MLXArray.zeros([1024], dtype: .uint32)
            eval(trace, workspace, partials, statistics, routing, control)
        }
    }

    /// Per-session KV storage (block-structured, bf16).
    final class KVState {
        var banks: [MLXArray]
        var metadata: MLXArray
        var capacity: Int
        let globalElements: Int
        let localElements: Int
        var position: Int = 0
        /// Prompt length captured when prefill handed off. Sliding-window
        /// insert is `(position - initial) % cap` once `initial >= cap`.
        var initial: Int = 0
        let scratch: DecodeScratch
        init(banks: [MLXArray], metadata: MLXArray, capacity: Int, globalElements: Int, localElements: Int) {
            self.banks = banks
            self.metadata = metadata
            self.capacity = capacity
            self.globalElements = globalElements
            self.localElements = localElements
            self.scratch = DecodeScratch()
        }
    }

    // MARK: init

    init(banks: [MLXArray], metadata: MLXArray, silu: MLXArray, configuration: MLXArray) {
        let launch = FlintRecLaunch.current()
        let sources = launch == .max
            ? (FlintRecKernelSources.headerMax, FlintRecKernelSources.bodyMax)
            : (FlintRecKernelSources.headerPro, FlintRecKernelSources.bodyPro)
        self.launch = launch
        self.banks = banks
        self.metadata = metadata
        self.siluTable = silu
        self.configuration = configuration
        let pad = MLXArray.zeros([1, 1024], dtype: .bfloat16)
        eval(pad)
        self.kvPlaceholder = pad
        self.kernel = MLXFast.metalKernel(
            name: "flint_decode_swift_\(launch.workers)",
            inputNames: (0..<13).map { "bank\($0)" } + [
                "metadata", "kv0", "kv1", "kv2", "kv3",
                "parameters", "silu_table", "private_control", "configuration",
                "trace", "workspace", "partials", "statistics", "routing",
            ],
            outputNames: ["logits", "diagnostic", "profile"],
            source: sources.1,
            header: sources.0,
            ensureRowContiguous: true
        )
        if let resident = FlintRecResident.body(from: sources.1, workers: launch.workers) {
            self.greedyKernel = MLXFast.metalKernel(
                name: "flint_int4_greedy_resident\(FlintRecResident.width)_\(launch.workers)",
                inputNames: (0..<13).map { "bank\($0)" } + [
                    "metadata", "kv0", "kv1", "kv2", "kv3",
                    "parameters", "silu_table", "private_control", "sampling",
                    "trace", "workspace", "partials", "statistics", "routing",
                ],
                outputNames: ["logits", "diagnostic", "profile"],
                source: resident,
                header: sources.0,
                ensureRowContiguous: true
            )
        } else {
            self.greedyKernel = nil
        }
        self.writeKernel = MLXFast.metalKernel(
            name: "flint_write_kv_swift",
            inputNames: ["storage", "keys", "values", "layout"],
            outputNames: ["done"],
            source: Self.writeSource,
            header: Self.writeHeader,
            ensureRowContiguous: true
        )
    }

    /// Build from the Swift-native packer output (production path).
    convenience init(packed: FlintRecPacker.Packed) {
        self.init(
            banks: packed.banks,
            metadata: MLXArray(packed.metadata.flatMap { $0 }, [50, 64]),
            silu: packed.siluTable,
            configuration: packed.configuration)
    }

    /// Load a packing produced by the reference packer dump.
    static func load(dumped dir: URL) throws -> FlintRecKernel {
        var banks: [MLXArray] = []
        for i in 0..<13 {
            let dict = try MLX.loadArrays(url: dir.appendingPathComponent("bank\(i).safetensors"))
            banks.append(dict["bank\(i)"]!)
        }
        let meta = try MLX.loadArrays(url: dir.appendingPathComponent("meta.safetensors"))["metadata"]!
        let misc = try MLX.loadArrays(url: dir.appendingPathComponent("misc.safetensors"))
        return FlintRecKernel(
            banks: banks, metadata: meta,
            silu: misc["silu"]!, configuration: misc["configuration"]!)
    }

    // MARK: KV state

    /// Build an empty KV state sized for `capacity` tokens (mirrors
    /// KVStorage.__init__ / grow).
    func makeKVState(capacity: Int) -> KVState {
        let rows = metadata.asArray(Int.self)  // [50*64]
        var globalIndex = 0, localIndex = 0
        let globalLayers = rows.prefix(49 * 64).enumerated().filter { $0.offset % 64 == 1 && $0.element != 0 }.count
        let globalElements = globalLayers * Self.blockTokens * 1024
        // Recompute per-layer offsets on the metadata copy (indices 61,62).
        var metaRows = rows
        for l in 0..<49 {
            if rows[l * 64 + 1] != 0 {
                metaRows[l * 64 + 61] = globalIndex * Self.blockTokens * 1024
                metaRows[l * 64 + 62] = Self.blockTokens
                globalIndex += 1
            } else {
                metaRows[l * 64 + 61] = globalElements + localIndex * 513 * 1024
                metaRows[l * 64 + 62] = 513
                localIndex += 1
            }
        }
        let localElements = localIndex * 513 * 1024
        let metaArray = MLXArray(metaRows.map { UInt32($0) }, [50, 64])
        eval(metaArray)

        let count = (capacity + Self.blockTokens - 1) / Self.blockTokens
        var banks: [MLXArray] = []
        for index in 0..<count {
            let size = globalElements + (index == 0 ? localElements : 0)
            let block = MLXArray.zeros([size / 1024, 1024], dtype: .bfloat16)
            eval(block)
            banks.append(block)
        }
        return KVState(
            banks: banks, metadata: metaArray,
            capacity: count * Self.blockTokens,
            globalElements: globalElements, localElements: localElements)
    }

    // MARK: decode step

    /// Grow global KV blocks up to the kernel's 4 × 65536-token ceiling.
    func ensureCapacity(_ state: KVState, tokens: Int) throws {
        let needed = max(tokens, 1)
        if needed <= state.capacity, !state.banks.isEmpty { return }
        let count = (needed + Self.blockTokens - 1) / Self.blockTokens
        guard count <= Self.kvBlocks else {
            throw NovaMLXError.inferenceFailed(
                "rec KV needs \(needed) tokens, past \(Self.kvBlocks * Self.blockTokens)")
        }
        while state.banks.count < count {
            let index = state.banks.count
            let size = state.globalElements + (index == 0 ? state.localElements : 0)
            let block = MLXArray.zeros([size / 1024, 1024], dtype: .bfloat16)
            eval(block)
            state.banks.append(block)
        }
        state.capacity = state.banks.count * Self.blockTokens
    }

    /// Copy prefill K/V into the block layout the decode kernel reads.
    /// `keys`/`values` are `[1, 4, S, 128]` or `[4, S, 128]`, temporal order
    /// (oldest at the front). Global layers append at `tokenOffset`; sliding
    /// layers are a single ring written at offset 0.
    func writeKV(
        state: KVState, layer: Int, tokenOffset: Int, keys: MLXArray, values: MLXArray
    ) throws {
        var key = try Self.kvMatrix(keys)
        var value = try Self.kvMatrix(values)
        guard key.dim(1) == value.dim(1) else {
            throw NovaMLXError.inferenceFailed("rec KV key/value lengths differ")
        }
        let rows = state.metadata.asArray(UInt32.self)
        let base = layer * 64
        guard base + 62 < rows.count else {
            throw NovaMLXError.inferenceFailed("rec KV metadata missing layer \(layer)")
        }
        let start = Int(rows[base + 61])
        let cap = Int(rows[base + 62])
        let global = rows[base + 1] != 0
        guard cap > 0 else {
            throw NovaMLXError.inferenceFailed("rec KV layer \(layer) has no capacity")
        }
        if !global, key.dim(1) > cap {
            let drop = key.dim(1) - cap
            key = key[.ellipsis, drop..., 0...]
            value = value[.ellipsis, drop..., 0...]
        }
        let seq = key.dim(1)
        if seq == 0 { return }
        let limit = global ? state.capacity : cap
        guard tokenOffset >= 0, tokenOffset + seq <= limit else {
            throw NovaMLXError.inferenceFailed(
                "rec KV write layer \(layer) offset \(tokenOffset)+\(seq) past \(limit)")
        }
        var begin = 0
        while begin < seq {
            let absolute = tokenOffset + begin
            let block = absolute / cap
            let position = absolute % cap
            let size = min(seq - begin, cap - position)
            guard block < state.banks.count else {
                throw NovaMLXError.inferenceFailed("rec KV block \(block) is not allocated")
            }
            let layout = MLXArray(
                [UInt32(start), UInt32(cap), UInt32(position), UInt32(size)], [4])
            let done = writeKernel(
                [
                    state.banks[block],
                    key[.ellipsis, begin ..< (begin + size), 0...],
                    value[.ellipsis, begin ..< (begin + size), 0...],
                    layout,
                ],
                grid: (size * 512, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[1]],
                outputDTypes: [.uint32]
            )
            eval(done)
            begin += size
        }
    }

    private static func kvMatrix(_ array: MLXArray) throws -> MLXArray {
        var a = array
        if a.ndim == 4 {
            guard a.dim(0) == 1 else {
                throw NovaMLXError.inferenceFailed("rec KV batch \(a.dim(0)) is not 1")
            }
            a = a.reshaped(a.dim(1), a.dim(2), a.dim(3))
        }
        guard a.ndim == 3, a.dim(0) == 4, a.dim(2) == 128 else {
            throw NovaMLXError.inferenceFailed("rec KV shape \(array.shape) is not [1,4,S,128]")
        }
        return a.asType(.bfloat16)
    }

    private func kvBank(_ state: KVState, _ index: Int) -> MLXArray {
        index < state.banks.count ? state.banks[index] : kvPlaceholder
    }

    /// One fused decode step. Returns logits [vocab] (bf16).
    @discardableResult
    func forward(state: KVState, token: Int, layerCount: Int = 49) throws -> MLXArray {
        try ensureCapacity(state, tokens: state.position + 1)
        // Fresh zeros every launch. rec does not reuse this buffer: a dirty
        // atomic counter deadlocks the resident spin.
        let control = MLXArray.zeros([1024], dtype: .uint32)
        let scratch = state.scratch
        // [token, position, initial, layer_count]. `initial` is the prefill
        // handoff, not the live position — see KVState.initial.
        let p = MLXArray(
            [UInt32(token), UInt32(state.position), UInt32(state.initial), UInt32(layerCount)],
            [4])

        let out = kernel(
            [
                banks[0], banks[1], banks[2], banks[3], banks[4], banks[5], banks[6],
                banks[7], banks[8], banks[9], banks[10], banks[11], banks[12],
                state.metadata,
                kvBank(state, 0), kvBank(state, 1), kvBank(state, 2), kvBank(state, 3),
                p,
                siluTable,
                control,
                configuration,
                scratch.trace,
                scratch.workspace,
                scratch.partials,
                scratch.statistics,
                scratch.routing,
            ],
            grid: (Self.decodeThreads * launch.workers, 1, 1),
            threadGroup: (Self.decodeThreads, 1, 1),
            outputShapes: [[Self.vocabulary], [launch.workers], [1]],
            outputDTypes: [.bfloat16, .uint32, .uint32]
        )
        // Logits and the diagnostic are the only new buffers. Scratch writes
        // land in the session arrays; eval waits for both before the next token.
        eval(out)
        let diagnostic = out[1].asArray(Int.self)
        if diagnostic.contains(where: { $0 != 0 }) {
            throw NovaMLXError.inferenceFailed("rec kernel GPU sync failure: \(diagnostic)")
        }
        state.position += 1
        return out[0]
    }

    /// Greedy resident packet. One launch runs `count` tokens
    /// (1...`FlintRecResident.width`) and returns the sampled ids.
    /// `eosTokenIds` (at most 4) stops the packet early, matching rec's
    /// sampler; an empty list runs the full count. Logits are a writable
    /// kernel output so the in-kernel argmax observes the vocabulary head.
    /// The host does not read them. The control buffer is reused unless
    /// `NOVA_FLINT_REC_FRESH_CONTROL=1`.
    func forwardGreedy(
        state: KVState, token: Int, count: Int, eosTokenIds: [Int] = [], layerCount: Int = 49
    ) throws -> [Int] {
        guard let greedyKernel else {
            throw NovaMLXError.inferenceFailed("rec greedy packet shader is not available")
        }
        guard (1...FlintRecResident.width).contains(count) else {
            throw NovaMLXError.inferenceFailed(
                "rec greedy packet count \(count) is not 1...\(FlintRecResident.width)")
        }
        guard eosTokenIds.count <= 4 else {
            throw NovaMLXError.inferenceFailed("rec greedy packet accepts at most 4 eos ids")
        }
        try ensureCapacity(state, tokens: state.position + count)
        let scratch = state.scratch
        let control = Self.uploadsFreshControl
            ? MLXArray.zeros([1024], dtype: .uint32) : scratch.control
        // float32 holds token ids through the vocab (150144 < 2^24).
        var sample = [Float](repeating: 0, count: 6)
        sample[0] = Float(count)
        sample[1] = Float(eosTokenIds.count)
        for (index, id) in eosTokenIds.enumerated() {
            sample[2 + index] = Float(id)
        }
        let sampling = MLXArray(sample)
        let p = MLXArray(
            [UInt32(token), UInt32(state.position), UInt32(state.initial), UInt32(layerCount)],
            [4])
        let out = greedyKernel(
            [
                banks[0], banks[1], banks[2], banks[3], banks[4], banks[5], banks[6],
                banks[7], banks[8], banks[9], banks[10], banks[11], banks[12],
                state.metadata,
                kvBank(state, 0), kvBank(state, 1), kvBank(state, 2), kvBank(state, 3),
                p,
                siluTable,
                control,
                sampling,
                scratch.trace,
                scratch.workspace,
                scratch.partials,
                scratch.statistics,
                scratch.routing,
            ],
            grid: (Self.decodeThreads * launch.workers, 1, 1),
            threadGroup: (Self.decodeThreads, 1, 1),
            outputShapes: [[Self.vocabulary], [launch.workers], [FlintRecResident.profileWords]],
            outputDTypes: [.bfloat16, .uint32, .uint32]
        )
        eval(out)
        let diagnostic = out[1].asArray(Int.self)
        if diagnostic.contains(where: { $0 != 0 }) {
            // The epilogue did not run, so the reused counters are dirty.
            if !Self.uploadsFreshControl {
                scratch.control = MLXArray.zeros([1024], dtype: .uint32)
                eval(scratch.control)
            }
            throw NovaMLXError.inferenceFailed("rec greedy packet GPU sync failure: \(diagnostic)")
        }
        let packet = out[2].asArray(Int.self)
        guard packet.count >= FlintRecResident.profileWords else {
            throw NovaMLXError.inferenceFailed(
                "rec greedy packet profile \(packet.count) < \(FlintRecResident.profileWords)")
        }
        let produced = packet[FlintRecResident.producedSlot]
        guard (1...count).contains(produced) else {
            throw NovaMLXError.inferenceFailed("rec greedy packet produced \(produced) for count \(count)")
        }
        var ids: [Int] = []
        ids.reserveCapacity(produced)
        var previous = token
        for step in 0..<produced {
            let input = packet[step * 2]
            let output = packet[step * 2 + 1]
            guard input == previous, output >= 0 else {
                throw NovaMLXError.inferenceFailed(
                    "rec greedy chain broke at step \(step): \(input) -> \(output), expected input \(previous)")
            }
            ids.append(output)
            previous = output
        }
        state.position += produced
        return ids
    }

    // MARK: KV write kernel

    static let writeHeader: String = "#include <metal_stdlib>\nusing namespace metal;"

    static let writeSource: String = """
        uint i=thread_position_in_grid.x, count=layout[3];
        if(i<count*512u) {
            uint head=i/(count*128u), pos=(i/128u)%count, dim=i%128u;
            uint offset=layout[0]+(head*layout[1]+layout[2]+pos)*128u+dim;
            device bfloat16_t* target=(device bfloat16_t*)storage;
            target[offset]=keys[i];
            target[offset+layout[1]*512u]=values[i];
        }
        if(i==0)done[0]=0;
    """
}

// MARK: - Resident greedy packet

/// Source transform from rec `resident_block.py` plus the greedy edits in
/// `sampling.py`. The loop is the single-token body, `width` times, with the
/// argmax inside the launch so the host does not read logits between tokens.
///
/// Official names the per-step dumps `block_logits` and `block_trace`. This
/// port keeps one logits output and the session trace: the argmax finishes
/// before the next step overwrites logits, and trace is write-only. That
/// avoids the `#define trace trace_rw` alias colliding with a shadow pointer.
///
/// Logits are a kernel output, not a const input. Width 8 matched single-token
/// argmax with that binding. Binding the same buffer as a const input made
/// every step of the packet emit one id (the head stores never reached the
/// argmax). The loop bound is a runtime load and the argmax stride is the
/// worker grid, so width 8 and 16 compile the same body.
enum FlintRecResident {
    /// Tokens per greedy launch. The loop is not unrolled, so this does not
    /// change the resident register footprint. 16 amortizes the host launch
    /// versus the previous packet of 8. Counters stay inside uint32 well
    /// past this width.
    static let width = 16
    /// `profile[producedSlot]` is how many tokens the packet actually ran.
    static var producedSlot: Int { width * 2 }
    static var profileWords: Int { producedSlot + 1 }

    private static let argmax = """
            // All original vocabulary-head writes precede the complete argmax.
            SYNC();
            float best=-as_type<float>(0x7f800000u);
            uint selected=0xffffffffu;
            for(uint word=index;word<150144u;word+=10240u) {
                float value=float(logits[word]);
                if(value>best || (value==best && word<selected)) {
                    best=value;selected=word;
                }
            }
            float peak=simd_max(best);
            uint first=simd_min(best==peak?selected:0xffffffffu);
            if(lane==0) {
                reduction[simdgroup_index_in_threadgroup]=peak;
                layer_descriptor[simdgroup_index_in_threadgroup]=first;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(simdgroup_index_in_threadgroup==0) {
                float value=lane<8?reduction[lane]:-as_type<float>(0x7f800000u);
                uint word=lane<8?layer_descriptor[lane]:0xffffffffu;
                float maximum=simd_max(value);
                uint choice=simd_min(value==maximum?word:0xffffffffu);
                if(lane==0) {
                    statistics[group*2]=maximum;
                    ((device uint*)statistics)[group*2+1]=choice;
                }
            }
            // Publish every group's maximum before any group chooses the next input.
            SYNC();
            if(simdgroup_index_in_threadgroup==0) {
                float value=-as_type<float>(0x7f800000u);
                uint word=0xffffffffu;
                for(uint peer=lane;peer<40;peer+=32) {
                    float other=statistics[peer*2];
                    uint other_word=((device uint*)statistics)[peer*2+1];
                    if(other>value || (other==value && other_word<word)) {
                        value=other;word=other_word;
                    }
                }
                float maximum=simd_max(value);
                uint choice=simd_min(value==maximum?word:0xffffffffu);
                if(lane==0)layer_descriptor[0]=choice;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            block_input_token=layer_descriptor[0];
            if(index==0)profile[block_step*2+1]=block_input_token;
    """

    /// Nil if `body` is not the rendered single-token shader this transform expects.
    static func body(from body: String, workers: Int) -> String? {
        guard workers == 36 || workers == 76 else { return nil }
        guard let embed = body.range(of: "Matrix embedding="),
              let loop = body.range(
                of: "    for(uint j=0;j<8;++j) {",
                range: embed.upperBound..<body.endIndex),
              let diagnostic = body.range(
                of: "    if(tid==0)diagnostic[group]=0;",
                range: loop.lowerBound..<body.endIndex)
        else { return nil }

        let suffix = body[diagnostic.lowerBound...]
        guard suffix.components(separatedBy: "diagnostic[group]=0;").count - 1 == 1,
              suffix.contains("0 && tid==0")
        else { return nil }

        var core = String(body[loop.lowerBound..<diagnostic.lowerBound])
        guard !core.contains("16384.0f"),
              core.components(separatedBy: "size_t row=uint(token);").count - 1 == 1,
              core.components(separatedBy: "uint(position)").count - 1 == 6,
              core.components(separatedBy: "expert_team_handoff(routing,control,tid,group,layer,phase,ready)").count - 1 == 1
        else { return nil }

        core = core.replacingOccurrences(of: "size_t row=uint(token);", with: "size_t row=block_input_token;")
        core = core.replacingOccurrences(of: "uint(position)", with: "uint(block_position)")
        core = core.replacingOccurrences(
            of: "expert_team_handoff(routing,control,tid,group,layer,phase,ready)",
            with: "expert_team_handoff(routing,control,tid,group,block_step*49u+layer,phase,ready)")

        let stride = workers * 256
        let reduced = Self.argmax
            .replacingOccurrences(of: "10240u", with: "\(stride)u")
            .replacingOccurrences(of: "peer<40", with: "peer<\(workers)")
        let header = """
            uint block_input_token=uint(token);
            #pragma clang loop unroll(disable)
            for(uint block_step=0;block_step<uint(sampling[0]);++block_step) {
                uint block_position=uint(position)+block_step;
                if(index==0)profile[block_step*2]=block_input_token;

        """
        let stop = """
                if(index==0)profile[\(Self.producedSlot)]=block_step+1u;
                bool stop=false;
                for(uint j=0;j<uint(sampling[1]);++j)stop=stop||block_input_token==uint(sampling[2+j]);
                if(stop)break;
            }
            SYNC();
            release_control(control,tid);
            if(tid==0)diagnostic[group]=0;

        """
        return String(body[..<loop.lowerBound]) + header + core + reduced + stop
    }
}
