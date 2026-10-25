import Foundation
import MLX
import MLXLMCommon
import MLXNN
import NovaMLXCore
import NovaMLXUtils

// MARK: - Swift-native rec weight packer (port of QATModel + FullModel packing)

/// Packs a loaded official-INT4 FlintModel into the rec megakernel's
/// bank layout — the Swift replica of recursive-oss/rec's `QATModel`
/// (quantization traversal) + `FullModel.__init__` (256-byte-aligned
/// banks, [50×64] metadata, silu table). No Python dependency.
///
/// The kernel consumes uint32-packed affine weights (group 64): the
/// attention/router projections 4-bit, the embedding 8-bit, and the
/// routed experts at the checkpoint's exported width. The official
/// `RecursiveSI/flint-int4` checkpoint already ships in that layout, so
/// packing is mostly re-layout (offset bookkeeping + bank assembly)
/// rather than requantization.
public enum FlintRecPacker {

    public struct Packed {
        public var banks: [MLXArray]
        public var metadata: [[UInt32]]
        public var siluTable: MLXArray
        public var configuration: MLXArray

        public init(banks: [MLXArray], metadata: [[UInt32]], siluTable: MLXArray, configuration: MLXArray) {
            self.banks = banks
            self.metadata = metadata
            self.siluTable = siluTable
            self.configuration = configuration
        }
    }

    static let weightAlignment = 256
    /// decode_threads from the rec HOST_LAYOUT.
    static let siluTableSize = 65536

    // MARK: entry

    /// Pack from an already-loaded official-INT4 `FlintModel`.
    /// Mirrors FullModel(model) with the M4-class schedule baked in.
    public static func pack(_ flint: FlintModel) throws -> Packed {
        let model = flint.model
        let layers = model.layers
        precondition(layers.count == 49)

        var metadata = Array(repeating: Array(repeating: UInt32(0), count: 64), count: 50)
        var banks: [MLXArray] = []

        // silu table: swiglu(gates, ones) where gates = (arange<<16).view(fp32).astype(bf16)
        let gates = MLXArray.arange(0, siluTableSize, dtype: .uint32)
        let shifted = (gates * 65536)  // << 16
        let asFloat = shifted.view(dtype: .float32)
        let gatesBF = asFloat.asType(.bfloat16)
        let ones = MLXArray.ones([siluTableSize], dtype: .bfloat16)
        let silu = MLXNN.silu(gatesBF) * ones
        eval(silu)

        let configuration = MLXArray([1], [1])
        eval(configuration)

        for first in stride(from: 0, to: 49, by: 4) {
            var pieces: [MLXArray] = []
            var size = 0
            let bankId = UInt32(banks.count)

            /// Append a raw tensor view to the current bank, returning its offset.
            func add(_ array: MLXArray) -> UInt32 {
                let pad = (256 - (size % 256)) % 256
                if pad > 0 {
                    pieces.append(MLXArray.zeros([pad], dtype: .uint8))
                }
                let offset = size + pad
                let flat = array.reshaped(-1).view(dtype: .uint8)
                pieces.append(flat)
                size += pad + flat.nbytes
                return UInt32(offset)
            }

            /// Append a packed quantized tensor triple + write the 5-slot
            /// descriptor (offsets + rows + group span).
            func descriptor(_ d: inout [UInt32], base: Int, weight: MLXArray, scales: MLXArray, biases: MLXArray?) {
                let wOff = add(weight)
                let sOff = add(scales)
                let bOff = add(biases ?? MLXArray.zeros([0]))
                d[base] = wOff
                d[base + 1] = sOff
                d[base + 2] = bOff
                d[base + 3] = UInt32(weight.dim(-2))
                d[base + 4] = UInt32(scales.dim(-1) * 64)
            }

            func linear(_ d: inout [UInt32], base: Int, q: QuantizedLinear) {
                descriptor(&d, base: base, weight: q.weight, scales: q.scales, biases: q.biases)
            }

            func quantizedSwitch(_ d: inout [UInt32], base: Int, q: QuantizedSwitchLinear) {
                descriptor(&d, base: base, weight: q.weight, scales: q.scales, biases: q.biases)
            }

            for i in first ..< min(first + 4, 49) {
                let layer = layers[i]
                let attn = layer.selfAttn
                let isSparse = !(layer.mlp is FlintMLP)
                let moe = layer.mlp as? FlintSparseMoeBlock
                var d = metadata[i]
                d[0] = bankId
                d[1] = attn.rope == nil ? 1 : 0  // is_global
                d[2] = isSparse ? 1 : 0
                d[3] = add(layer.inputNorm.weight)
                d[4] = add(layer.postNorm.weight)
                d[5] = add(attn.qNorm.weight)
                d[6] = add(attn.kNorm.weight)
                d[7] = add(attn.softmaxOffset.asType(.bfloat16))

                if isSparse, let moe {
                    // Router weight (fp32) + zero expert bias + bf16 copy at 60.
                    let routerW = moe.gate.weight.asType(.float32)
                    d[8] = add(routerW)
                    d[9] = add(MLXArray.zeros([256], dtype: .float32))
                    d[60] = add(routerW.asType(.bfloat16))
                }

                // Reference add order: q, k, v, o (10/15/20/25), then
                // gate/up/down (30/35/40 — routed experts when sparse, dense
                // MLP otherwise), then shared expert (45/50/55).
                linear(&d, base: 10, q: quantized(attn.qProj))
                linear(&d, base: 15, q: quantized(attn.kProj))
                linear(&d, base: 20, q: quantized(attn.vProj))
                linear(&d, base: 25, q: quantized(attn.oProj))
                if let moe {
                    quantizedSwitch(&d, base: 30, q: moe.switchMLP.gateProj as! QuantizedSwitchLinear)
                    quantizedSwitch(&d, base: 35, q: moe.switchMLP.upProj as! QuantizedSwitchLinear)
                    quantizedSwitch(&d, base: 40, q: moe.switchMLP.downProj as! QuantizedSwitchLinear)
                    linear(&d, base: 45, q: quantized(moe.sharedExpert.gate))
                    linear(&d, base: 50, q: quantized(moe.sharedExpert.up))
                    linear(&d, base: 55, q: quantized(moe.sharedExpert.down))
                } else if let mlp = layer.mlp as? FlintMLP {
                    linear(&d, base: 30, q: quantized(mlp.gate))
                    linear(&d, base: 35, q: quantized(mlp.up))
                    linear(&d, base: 40, q: quantized(mlp.down))
                }
                metadata[i] = d
            }

            // Final bank (49): model norm + embedding + lm head.
            if first == 48 {
                var d = metadata[49]
                d[0] = bankId
                d[3] = add(model.norm.weight)
                // Embedding (8-bit quantized) at base 10, head at base 30.
                if let emb = model.embedTokens as? QuantizedEmbedding {
                    let wOff = add(emb.weight)
                    let sOff = add(emb.scales)
                    let bOff = add(emb.biases ?? MLXArray.zeros([0]))
                    d[10] = wOff; d[11] = sOff; d[12] = bOff
                    d[13] = UInt32(emb.weight.dim(-2))
                    d[14] = UInt32(emb.scales.dim(-1) * 64)
                }
                if let lm = flint.lmHead {
                    linear(&d, base: 30, q: lm as! QuantizedLinear)
                }
                metadata[49] = d
            }

            let bank = concatenated(pieces, axis: 0)
            eval(bank)
            banks.append(bank)
            NovaMLXLog.info("[FlintRecPacker] bank \(banks.count - 1): layers \(first)..\(min(first + 4, 49)), \(size / 1_000_000) MB")
        }

        return Packed(
            banks: banks,
            metadata: metadata,
            siluTable: silu,
            configuration: configuration)
    }

    /// The packer runs against the quantized sublayers produced by loading
    /// the official INT4 checkpoint; this helper hides the cast dance.
    private static func quantized(_ linear: Linear) -> QuantizedLinear {
        linear as! QuantizedLinear
    }
}
