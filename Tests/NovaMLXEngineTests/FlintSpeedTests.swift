import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Testing

@testable import NovaMLXEngine

/// Manual speed probe: bypasses NovaMLX's schedulers entirely and times the
/// raw model decode loop, matching Python mlx-lm's generate() shape. Runs
/// only when the weights are on disk (skips otherwise).
@Suite("Flint raw decode speed")
struct FlintSpeedTests {

    @Test("raw model decode ~128 tokens")
    func testRawDecodeSpeed() async throws {
        let dir = URL(fileURLWithPath: "/Volumes/Samsung768/Models/RecursiveSI/flint-int4")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue
        else {
            print("SPEED-SKIP: weights not present")
            return
        }

        await CustomModelRegistration.ensureRegistered()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: dir, using: MLXEngine.tokenizerLoader)
        let tokenizer = await container.tokenizer

        let promptIds = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "Write a detailed 400-word essay about the history of computing."]])

        // Dtype probe — official INT4: bf16 activations, 4-bit packed weights.
        let dtypeInfo = await container.perform { context -> String in
            let m = context.model as! FlintModel
            let l = m.model.layers[2]
            let q = l.selfAttn.qProj.weight.dtype
            return "q_proj=\(q) input_norm=\(l.inputNorm.weight.dtype) embed=\(m.model.embedTokens.weight.dtype) softmax_offset=\(l.selfAttn.softmaxOffset.dtype)"
        }
        print("SPEED-DTYPE:", dtypeInfo)

        // Raw decode, entirely inside one perform block (no per-token hops).
        let n = 128
        let dtBox = try await container.perform { context in
            let caches = try context.model.newCache(parameters: nil)
            let full = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)
            var logits = context.model(full, cache: caches)
            eval(logits)
            var lastId = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            logits = context.model(MLXArray(Int32(lastId)).reshaped(1, 1), cache: caches)
            eval(logits)
            lastId = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            // Warmup extra: force all compile traces before timing.
            for _ in 0 ..< 40 {
                logits = context.model(MLXArray(Int32(lastId)).reshaped(1, 1), cache: caches)
                eval(logits)
                lastId = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            }
            var segTimes: [Double] = []
            let per = n / 4
            for seg in 0 ..< 4 {
                let t0 = Date()
                for _ in 0 ..< per {
                    logits = context.model(MLXArray(Int32(lastId)).reshaped(1, 1), cache: caches)
                    eval(logits)
                    lastId = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
                }
                segTimes.append(Date().timeIntervalSince(t0) / Double(per) * 1000)
            }
            print("SPEED-SEGS ms/token:", segTimes.map { String(format: "%.1f", $0) }.joined(separator: " "))
            return SendableBox(segTimes.reduce(0, +) / 4 / 1000 * Double(n))
        }
        let dt = dtBox.value
        print(String(format: "SPEED-RESULT: raw model decode %.2f s for %d tokens = %.1f tok/s", dt, n, Double(n) / dt))

        // Per-block microbench (no cache), mirroring the Python probe.
        let micro = await container.perform { context -> String in
            let m = context.model as! FlintModel
            let layer = m.model.layers[10]
            let x = (MLXRandom.normal([1, 1, 2048]) * 0.01)
            func attnOnce() {
                let r = layer.selfAttn(layer.inputNorm(x), mask: .none, cache: nil)
                eval(r)
            }
            func mlpOnce() {
                let r = layer.mlp
                let out = (r as! FlintSparseMoeBlock)(layer.postNorm(x))
                eval(out)
            }
            for _ in 0 ..< 20 { attnOnce(); mlpOnce() }
            let N = 300
            let t0 = Date()
            for _ in 0 ..< N { attnOnce() }
            let tAttn = Date().timeIntervalSince(t0) / Double(N) * 1000
            let t1 = Date()
            for _ in 0 ..< N { mlpOnce() }
            let tMlp = Date().timeIntervalSince(t1) / Double(N) * 1000
            return String(format: "attn=%.2fms mlp=%.2fms total=%.2fms -> est %.1f tok/s (49 layers)", tAttn, tMlp, tAttn + tMlp, 1000.0 / ((tAttn + tMlp) * 49.0 / 1000.0))
        }
        print("SPEED-MICRO:", micro)

        // Layer 10 with a real RotatingKVCache — mirrors the Python probe.
        let rot = await container.perform { context -> String in
            let m = context.model as! FlintModel
            let layer = m.model.layers[10]
            let cache = RotatingKVCache(maxSize: 513)
            let x = (MLXRandom.normal([1, 1, 2048]) * 0.01)
            func step() {
                let r = layer.selfAttn(layer.inputNorm(x), mask: .none, cache: cache)
                let out = (layer.mlp as! FlintSparseMoeBlock)(layer.postNorm(x + r))
                eval(out, cache.state)
            }
            for _ in 0 ..< 30 { step() }
            let N = 500
            let t0 = Date()
            for _ in 0 ..< N { step() }
            return String(format: "%.3f ms", Date().timeIntervalSince(t0) / Double(N) * 1000)
        }
        print("SPEED-ROT:", rot)
        // RotatingKVCache.update alone, mirroring the Python probe.
        let rotUpd = await container.perform { context -> String in
            let c = RotatingKVCache(maxSize: 513)
            let k = MLXRandom.normal([1, 1, 4, 128]).asType(.bfloat16)
            let v = MLXRandom.normal([1, 1, 4, 128]).asType(.bfloat16)
            for _ in 0 ..< 600 { let _ = c.update(keys: k, values: v); eval(c.state) }
            let N = 2000
            let t0 = Date()
            for _ in 0 ..< N { let _ = c.update(keys: k, values: v); eval(c.state) }
            return String(format: "%.3f ms", Date().timeIntervalSince(t0) / Double(N) * 1000)
        }
        print("SPEED-ROTUPD:", rotUpd)
        // Mega-decode probe: single compiled call per token.
        let mega = try await container.perform { context -> [String] in
            let m = context.model as! FlintModel
            let md = FlintMegaDecode(model: m.model, args: m.model.config)
            var st = md.initialState()
            // Prefill token-by-token (eager functional), capturing logits head.
            var lastId = 0
            for (i, t) in promptIds.enumerated() {
                let hidden = md.prefillToken(tokenId: t, state: &st)
                if i == promptIds.count - 1 {
                    let logits = m.lmHead != nil ? m.lmHead!(hidden) : m.model.embedTokens.asLinear(hidden)
                    eval(logits)
                    lastId = Int(argMax(logits[0, 0]).item(Int.self))
                }
            }
            func step(_ id: Int) -> Int {
                let hidden = md.decodeHidden(tokenId: id, state: &st)
                let logits = m.lmHead != nil ? m.lmHead!(hidden) : m.model.embedTokens.asLinear(hidden)
                eval(logits)
                return Int(argMax(logits[0, 0]).item(Int.self))
            }
            // Warmup (compile trace)
            for _ in 0 ..< 10 { lastId = step(lastId) }
            // Bucketed timing: detect per-call retrace vs steady state.
            var buckets: [String] = []
            var t0 = Date()
            var count = 0
            for i in 0 ..< 400 {
                lastId = step(lastId)
                count += 1
                if i % 50 == 49 {
                    let dt = Date().timeIntervalSince(t0)
                    buckets.append(String(format: "%d-%d: %.1f tok/s", i - 49, i, Double(count) / dt))
                    t0 = Date()
                    count = 0
                }
            }
            return ["MEGA-BUCKETS: " + buckets.joined(separator: " | ")]
        }
        for line in mega { print(line) }

        // rec megakernel probe: reference-packed banks + Swift kernel invocation.
        let rec = try await container.perform { context -> [String] in
            let fm = try FlintRecKernel.load(dumped: URL(fileURLWithPath: "/tmp/flint_rec_packed"))
            let state = fm.makeKVState(capacity: 65536)
            var lastId = 150000  // <|begin_of_text|>
            // Warmup: trace + shader compile.
            for _ in 0 ..< 10 {
                let logits = try fm.forward(state: state, token: lastId)
                eval(logits)
                lastId = Int(argMax(logits).item(Int.self))
            }
            let N = 128
            let t0 = Date()
            for _ in 0 ..< N {
                let logits = try fm.forward(state: state, token: lastId)
                eval(logits)
                lastId = Int(argMax(logits).item(Int.self))
            }
            let dt = Date().timeIntervalSince(t0)
            return [String(format: "REC-RESULT: %.2f s for %d tokens = %.1f tok/s", dt, N, Double(N) / dt)]
        }
        for line in rec { print(line) }

        // Native packer probe: pack from the loaded model, compare to reference dump.
        let packed = try await container.perform { context -> [String] in
            let flint = context.model as! FlintModel
            let p = try FlintRecPacker.pack(flint)
            // Compare against reference dump banks.
            var results: [String] = []
            for i in 0 ..< p.banks.count {
                let ref = try MLX.loadArrays(url: URL(fileURLWithPath: "/tmp/flint_rec_packed/bank\(i).safetensors"))["bank\(i)"]!
                let mine = p.banks[i]
                let diff = abs(mine.asType(.float32) - ref.asType(.float32)).max()
                let match = mine.shape == ref.shape && (diff.item(Float.self)) < 1e-6
                results.append("bank\(i): shape \(mine.shape) vs \(ref.shape) match=\(match)")
            }
            // Metadata compare
            let refMeta = try MLX.loadArrays(url: URL(fileURLWithPath: "/tmp/flint_rec_packed/meta.safetensors"))["metadata"]!
            let refRows = refMeta.asArray(UInt32.self)
            let mineFlat = p.metadata.flatMap { $0 }
            let metaMatch = zip(mineFlat, refRows).allSatisfy { $0 == $1 }
            var mismatches = 0
            for (idx, (a, b)) in zip(mineFlat, refRows).enumerated() where a != b {
                if mismatches < 8 { results.append("meta mismatch @\(idx): \(a) vs \(b) (row \(idx/64) slot \(idx%64))") }
                mismatches += 1
            }
            results.append("metadata: \(mismatches) mismatches of 3200")
            return results
        }
        for line in packed { print("PACK:", line) }

        // Milestone 4 probe: load gate → native pack → decode closed loop.
        let m4 = try await container.perform { context -> [String] in
            let flint = context.model as! FlintModel
            guard let rec = flint.recKernelIfAvailable() else {
                return ["M4: gate declined (not official INT4?)"]
            }
            let state = rec.makeKVState(capacity: 65536)
            var lastId = 150000
            for _ in 0 ..< 10 {
                let logits = try rec.forward(state: state, token: lastId)
                eval(logits)
                lastId = Int(argMax(logits).item(Int.self))
            }
            let N = 128
            let t0 = Date()
            for _ in 0 ..< N {
                let logits = try rec.forward(state: state, token: lastId)
                eval(logits)
                lastId = Int(argMax(logits).item(Int.self))
            }
            let dt = Date().timeIntervalSince(t0)
            return [String(format: "M4-RESULT: load-gate → native-pack → decode = %.1f tok/s", Double(N) / dt)]
        }
        for line in m4 { print(line) }

        print("SPEED-QMM-START")
        // Single QuantizedLinear probe (2048 -> 4096, 4-bit, group 64), matching Python.
        // Standalone module — no container access needed.
        do {
            let ql = QuantizedLinear(
                weight: MLXRandom.normal([4096, 2048]) * 0.02,
                bias: nil, groupSize: 64, bits: 4, mode: .affine)
            let x = (MLXRandom.normal([1, 1, 2048]) * 0.01)
            for _ in 0 ..< 20 { eval(ql(x)) }
            let N = 500
            let t0 = Date()
            for _ in 0 ..< N { eval(ql(x)) }
            let dt = Date().timeIntervalSince(t0) / Double(N) * 1000
            print(String(format: "SPEED-QMM: %.3f ms", dt))
        }
    }
}
