import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import NovaMLXEngine

@Suite("Flint M4 Max prefill gate+up")
struct FlintM4MaxPrefillTests {

    @Test("row threshold matches rec PrefillSwitchGLU")
    func testRowThreshold() {
        // indices.size >= num_experts * 4. Flint is 256 experts, top-6.
        #expect(!FlintM4MaxPrefill.meetsRowThreshold(indexCount: 1023, experts: 256))
        #expect(FlintM4MaxPrefill.meetsRowThreshold(indexCount: 1024, experts: 256))
        #expect(!FlintM4MaxPrefill.meetsRowThreshold(indexCount: 6, experts: 256))
        #expect(FlintM4MaxPrefill.meetsRowThreshold(indexCount: 512 * 6, experts: 256))
        #expect(FlintM4MaxPrefill.meetsRowThreshold(indexCount: 171 * 6, experts: 256))
        #expect(!FlintM4MaxPrefill.meetsRowThreshold(indexCount: 170 * 6, experts: 256))
        #expect(FlintM4MaxPrefill.deviceIsM4Max("Apple M4 Max"))
        #expect(!FlintM4MaxPrefill.deviceIsM4Max("Apple M5 Max"))
        #expect(!FlintM4MaxPrefill.deviceIsM4Max("Apple M4 Pro"))
        #expect(!FlintM4MaxPrefill.deviceIsM4Max(""))
        #expect(!FlintM4MaxPrefill.envAllowsFused("0"))
        #expect(!FlintM4MaxPrefill.envAllowsFused("off"))
        #expect(FlintM4MaxPrefill.envAllowsFused(nil))
        #expect(FlintM4MaxPrefill.envAllowsFused("1"))
        #expect(FlintM4MaxPrefill.bm == 16)
        #expect(FlintM4MaxPrefill.bn == 32)
        #expect(FlintM4MaxPrefill.bk == 32)
        #expect(FlintM4MaxPrefill.kernelSource.contains("BlockMMA"))
        #expect(FlintM4MaxPrefill.kernelSource.contains("QuantizedBlockLoader"))
        #expect(FlintM4MaxPrefill.kernelSource.contains("BM=16"))
        #expect(FlintM4MaxPrefillSteel.header.contains("struct QuantizedBlockLoader"))
        #expect(FlintM4MaxPrefillSteel.header.contains("struct BlockMMA"))
        #expect(FlintM4MaxPrefillSteel.header.contains("struct BlockLoader"))
    }

    @Test("switch route never builds the fused graph")
    func testSwitchRouteSkipsKernel() {
        let glu = SwitchGLU(inputDims: 64, hiddenDims: 32, numExperts: 4)
        let x = MLXArray.zeros([16, 64], dtype: .bfloat16)
        let indices = MLXArray([Int32](repeating: 0, count: 64), [16, 4])
        let before = FlintM4MaxPrefill.fusedCalls
        #expect(FlintM4MaxPrefill.project(glu, x: x, indices: indices, route: .switchGLU) == nil)
        #expect(FlintM4MaxPrefill.fusedCalls == before)
    }

    /// Mac Metal. Tiny affine int4 SwitchGLU, above the row threshold.
    /// Input and hidden are multiples of the group (64) and the tile (32);
    /// a 32-wide hidden would not quantize at group 64. The reference is
    /// SwitchGLU itself (gather qmm + silu). The steel tile is not
    /// bit-identical to every gather-qmm schedule; a transpose or a missed
    /// dequant is far outside this band. Scales are forced to bf16, which is
    /// the flint-int4 contract the kernel reads.
    @Test("fused gate+up matches SwitchGLU above the row threshold")
    func testFusedMatchesSwitchGLU() throws {
        let glu = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 4)
        quantize(model: glu, groupSize: 64, bits: 4, mode: .affine) { layer, group, bits, mode in
            (layer as? SwitchLinear)?.toQuantized(groupSize: group, bits: bits, mode: mode)
        }
        var flat = glu.parameters().flattened()
        let scaleKeys = flat.keys.filter { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") }
        for key in scaleKeys {
            flat[key] = flat[key]?.asType(.bfloat16)
        }
        try glu.update(parameters: ModuleParameters.unflattened(flat), verify: .none)

        let x = (MLXRandom.normal([16, 64]) * 0.05).asType(.bfloat16)
        var ids = [Int32](repeating: 0, count: 64)
        for i in 0 ..< ids.count { ids[i] = Int32(i % 4) }
        let indices = MLXArray(ids, [16, 4])
        // 64 rows, 4 experts, threshold 16. Under the threshold must decline.
        let small = MLXArray.zeros([1, 64], dtype: .bfloat16)
        let smallIndex = MLXArray([Int32(0), Int32(1)], [1, 2])
        #expect(FlintM4MaxPrefill.project(glu, x: small, indices: smallIndex, route: .fused) == nil)

        let before = FlintM4MaxPrefill.fusedCalls
        let fused = FlintM4MaxPrefill.project(glu, x: x, indices: indices, route: .fused)
        if fused == nil {
            for (key, value) in glu.parameters().flattened() {
                print("PREFILL-PLANE \(key) shape=\(value.shape) dtype=\(value.dtype)")
            }
        }
        let reference = glu(x, indices)
        let got = try #require(fused)
        eval(got, reference)
        #expect(FlintM4MaxPrefill.fusedCalls == before + 1)
        #expect(got.shape == reference.shape)
        let diff = abs(got.asType(.float32) - reference.asType(.float32)).max().item(Float.self)
        let scale = abs(reference.asType(.float32)).max().item(Float.self)
        print(String(format: "PREFILL-PARITY maxAbs=%.5f scale=%.5f", diff, scale))
        #expect(diff <= 0.05 * max(scale, Float(1)))
    }
}
