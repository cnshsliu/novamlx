import Foundation
import MLX
import MLXNN
import Testing

/// Bandwidth microbench for the quantized matmul shapes the 27B oQ8e model
/// hits at decode. Compares achieved GB/s of MLX's 8-bit affine group-64
/// GEMV against a same-shape fp16 matmul (the dequantized upper bound) and
/// the theoretical copy floor. If the 8-bit kernel lands far below the fp16
/// number, that kernel is the MTP round-cost gap.
@Suite("Quantized GEMV bandwidth")
struct QuantizedGEMVBandwidthTests {
    @Test func eightBitGEMVBandwidth() throws {
        // gate_proj shape from the 27B: [17408, 5120] at 8-bit affine g64.
        let outDim = 17408
        let inDim = 5120
        let groupSize = 64

        let q = MLXArray.zeros([1, inDim], dtype: .float16)
        let x = MLXArray.zeros([1, inDim], dtype: .float16)

        // Quantized linear built the same way mlx-community checkpoints do.
        let qlinear = QuantizedLinear(
            inDim, outDim, bias: false, groupSize: groupSize, bits: 8)
        let fpLinear = Linear(inDim, outDim, bias: false)

        func bench(_ op: () -> MLXArray, rounds: Int = 200) -> Double {
            eval(op())
            let t0 = DispatchTime.now()
            for _ in 0..<rounds { eval(op()) }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            return ms / Double(rounds)
        }

        // Pipelined variant: queue `chain` distinct ops then eval once — the
        // GPU streams back-to-back kernels without per-call sync, which is
        // what the real model's forward achieves.
        func benchPipelined(_ op: (Int) -> MLXArray, chain: Int = 32, rounds: Int = 8) -> Double {
            var acc: MLXArray = MLXArray.zeros([1, 1], dtype: .float16)
            eval(op(0))
            let t0 = DispatchTime.now()
            for _ in 0..<rounds {
                for i in 0..<chain {
                    acc = acc + op(i).sum()
                }
                eval(acc)
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6
            return ms / Double(rounds * chain)
        }

        let qMs = bench { qlinear(q) }
        let fpMs = bench { fpLinear(x) }
        // Pipelined: same op queued 32-deep. The linear layers are distinct
        // instances so results can't fold.
        let qliners = (0..<4).map { _ in QuantizedLinear(
            inDim, outDim, bias: false, groupSize: groupSize, bits: 8) }
        let fpliners = (0..<4).map { _ in Linear(inDim, outDim, bias: false) }
        for l in qliners + fpliners { eval(l.parameters()) }
        let qMsP = benchPipelined { i in qliners[i % 4](q) }
        let fpMsP = benchPipelined { i in fpliners[i % 4](x) }

        // bytes read per call: quantized = weights + scales + biases; fp16 = weights
        let qBytes = Double(outDim * inDim) / 8.0 * 8.0 * 0.5
            + Double(outDim * inDim / groupSize) * 4.0  // w(8bit→1B/elem→half of fp16? no: 1 byte each)
        // Recompute cleanly below instead:
        let qWeightBytes = Double(outDim * inDim)              // 1 byte per element
        let qScaleBytes = Double(outDim * (inDim / groupSize)) * 4.0  // fp32 scales+biases
        let qTotal = qWeightBytes + qScaleBytes
        let fpBytes = Double(outDim * inDim * 2)
        _ = qBytes

        print(String(
            format: "GEMVBENCH [1,%d]x[%d,%d]: 8bit=%.2fms (%.0f GB/s)  fp16=%.2fms (%.0f GB/s)",
            inDim, outDim, inDim,
            qMs, qTotal / qMs / 1e6, fpMs, fpBytes / fpMs / 1e6))
        print(String(
            format: "GEMVBENCH-PIPELINED: 8bit=%.3fms (%.0f GB/s)  fp16=%.3fms (%.0f GB/s)",
            qMsP, qTotal / qMsP / 1e6, fpMsP, fpBytes / fpMsP / 1e6))
    }
}
