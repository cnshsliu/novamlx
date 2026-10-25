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
