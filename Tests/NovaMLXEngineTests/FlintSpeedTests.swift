import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon
import Testing

@testable import NovaMLXEngine

/// Manual speed probe. Official INT4 decode is timed two ways. `SINGLE-RESULT`
/// is one `model()` call plus a host argmax per token. `LOOP-RESULT` is rec's
/// greedy resident packet (`FlintRecResident.width` tokens per launch, argmax
/// on device). The published baseline on MacMaxPro M4 Max at packet 8 is
/// 214.7 tok/s (`e8e7b27`). Re-run on that machine with
/// `./build.sh test --filter FlintSpeedTests` and read `PARITY`,
/// `SINGLE-RESULT`, `LOOP-RESULT`. `NOVA_FLINT_REC_FRESH_CONTROL=1` restores
/// the per-launch zero upload. Runs only when the weights are on disk.
/// Stops after `LOOP-RESULT`. `NOVA_FLINT_MEGA_DECODE=1` also runs the
/// obsolete MegaDecode probe.
@Suite("Flint raw decode speed")
struct FlintSpeedTests {

    private struct LoopStats: Sendable {
        var packetSeconds: Double
        var packetSteps: Int
        var singleSeconds: Double
        var singleSteps: Int
        var official: Bool
        var launch: String
        var parity: Bool
    }

    /// Official INT4 snapshot. Local disk first, then the removable volume
    /// (same roots as `BiRefNetMattingTests`). Shards are `model-0000N`, so
    /// the sentinel is `config.json`.
    private static func officialInt4Directory() -> URL? {
        for root in ["/Users/lucas/Models", "/Volumes/Samsung768/Models"] {
            let url = URL(fileURLWithPath: root + "/RecursiveSI/flint-int4")
            let config = url.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: config.path) { return url }
        }
        return nil
    }

    @Test("raw model decode ~128 tokens")
    func testRawDecodeSpeed() async throws {
        guard let dir = Self.officialInt4Directory() else {
            print("SPEED-SKIP: weights not present")
            return
        }
        print("SPEED-WEIGHTS:", dir.path)

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

        // Two closed loops. Both prefill the same prompt, take the eager
        // one-token tail, then one rec step so the session is live.
        // Parity compares the next 16 greedy tokens: host argmax versus the
        // resident packet. Timing starts after that, so shader compile is
        // outside the window.
        let n = 128
        let dtBox = try await container.perform { context in
            let flint = context.model as! FlintModel
            let prompt = MLXArray(promptIds.map { Int32($0) }).reshaped(1, -1)

            func arm(_ caches: [KVCache]) -> Int {
                var logits = context.model(prompt, cache: caches)
                eval(logits)
                var last = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
                // Eager tail, then the first rec step.
                for _ in 0 ..< 2 {
                    logits = context.model(MLXArray(Int32(last)).reshaped(1, 1), cache: caches)
                    eval(logits)
                    last = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
                }
                return last
            }
            func singleStep(_ caches: [KVCache], _ last: Int) -> Int {
                let logits = context.model(MLXArray(Int32(last)).reshaped(1, 1), cache: caches)
                eval(logits)
                return Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            }

            let singleCaches = try context.model.newCache(parameters: nil)
            var singleLast = arm(singleCaches)
            var reference: [Int] = []
            reference.reserveCapacity(16)
            for _ in 0 ..< 16 {
                singleLast = singleStep(singleCaches, singleLast)
                reference.append(singleLast)
            }

            // Time single-token while this cache still owns the rec session.
            // Arming the packet cache replaces that session.
            var singleSegs: [Double] = []
            let per = n / 4
            let singleBefore = flint.recDecodeSteps
            for _ in 0 ..< 4 {
                let t0 = Date()
                for _ in 0 ..< per {
                    singleLast = singleStep(singleCaches, singleLast)
                }
                singleSegs.append(Date().timeIntervalSince(t0) / Double(per) * 1000)
            }
            let singleSteps = flint.recDecodeSteps - singleBefore
            print("SINGLE-SEGS ms/token:", singleSegs.map { String(format: "%.1f", $0) }.joined(separator: " "))

            let packetCaches = try context.model.newCache(parameters: nil)
            var packetLast = arm(packetCaches)
            var resident: [Int] = []
            resident.reserveCapacity(16)
            let packetWidth = FlintRecResident.width
            while resident.count < 16 {
                let take = min(packetWidth, 16 - resident.count)
                let ids = try flint.recGreedyPacket(cache: packetCaches, token: packetLast, count: take)
                resident.append(contentsOf: ids)
                packetLast = ids[ids.count - 1]
            }
            let parity = reference == resident
            print("PARITY:", parity ? "match 16" : "MISMATCH single=\(reference) packet=\(resident)")

            // Resident packets. The parity packets already compiled this shader.
            var packetSegs: [Double] = []
            let packetBefore = flint.recDecodeSteps
            for _ in 0 ..< 4 {
                let t0 = Date()
                var left = per
                while left > 0 {
                    let take = min(FlintRecResident.width, left)
                    let ids = try flint.recGreedyPacket(cache: packetCaches, token: packetLast, count: take)
                    packetLast = ids[ids.count - 1]
                    left -= ids.count
                }
                packetSegs.append(Date().timeIntervalSince(t0) / Double(per) * 1000)
            }
            let packetSteps = flint.recDecodeSteps - packetBefore
            print("PACKET-SEGS ms/token:", packetSegs.map { String(format: "%.1f", $0) }.joined(separator: " "))

            func seconds(_ segs: [Double]) -> Double {
                segs.reduce(0, +) / 4 / 1000 * Double(n)
            }
            return SendableBox(LoopStats(
                packetSeconds: seconds(packetSegs), packetSteps: packetSteps,
                singleSeconds: seconds(singleSegs), singleSteps: singleSteps,
                official: flint.isOfficialInt4,
                launch: flint.recKernel?.launch.label ?? "none",
                parity: parity))
        }
        let stats = dtBox.value
        let singleRate = Double(n) / stats.singleSeconds
        let packetRate = Double(n) / stats.packetSeconds
        print(String(format: "SINGLE-RESULT: one launch + host argmax %.1f tok/s (%d rec steps)", singleRate, stats.singleSteps))
        print(String(
            format: "LOOP-RESULT: load→pack→decode %.1f tok/s (%d rec steps, official=%@, launch=%@, packet=%d)",
            packetRate, stats.packetSteps, stats.official ? "yes" : "no", stats.launch, FlintRecResident.width))
        // The first single-token forward stays eager (it may be the prefill
        // tail). Both timed windows are rec steps after that handoff.
        #expect(stats.official)
        #expect(stats.parity)
        #expect(stats.singleSteps == n)
        #expect(stats.packetSteps == n)

        // MegaDecode is an obsolete compile experiment. After a successful rec
        // loop it still launches and GPU-timeouts, which fails this suite.
        // The microbench / dump probes below it are the same optional A/B.
        guard ProcessInfo.processInfo.environment["NOVA_FLINT_MEGA_DECODE"] == "1" else {
            print("MEGA-SKIP: obsolete MegaDecode probe gated off (set NOVA_FLINT_MEGA_DECODE=1)")
            return
        }

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

        // Reference-dump probes. The closed loop above does not need them.
        let dump = URL(fileURLWithPath: "/tmp/flint_rec_packed")
        var dumpIsDir: ObjCBool = false
        let haveDump = FileManager.default.fileExists(atPath: dump.path, isDirectory: &dumpIsDir)
            && dumpIsDir.boolValue
        if !haveDump {
            print("REC-SKIP: reference dump not present; closed loop is the primary probe")
        }

        // rec megakernel probe: reference-packed banks + Swift kernel invocation.
        let rec: [String]
        if haveDump {
            rec = try await container.perform { context -> [String] in
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
        } else {
            rec = ["REC-SKIP"]
        }
        for line in rec { print(line) }

        // Native packer probe: pack from the loaded model, compare to reference dump.
        let packed: [String]
        if haveDump {
            packed = try await container.perform { context -> [String] in
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
            results.append("metadata: \(mismatches) mismatches of 3200 match=\(metaMatch)")
            return results
            }
        } else {
            packed = ["PACK-SKIP"]
        }
        for line in packed { print("PACK:", line) }

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
