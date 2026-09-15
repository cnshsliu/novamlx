import Foundation
import MLX
import MLXNN
import MLXLMCommon
import NovaMLXUtils

/// On-demand expert slots matching oMLX `ExpertOffloadPlan` (5% residency).
public final class DeepseekV41ExpertBank: @unchecked Sendable {
    struct ProjSpec {
        let file: URL
        let weightKey: String
        let scaleKey: String
        let dtype: String
        let weightShape: [Int]
    }

    private let lock = NSLock()
    private var files: [String: SafetensorMMap] = [:]
    private var specs: [String: [String: ProjSpec]] = [:]
    private var lru: [Int: [Int]] = [:]
    private var slotOf: [Int: [Int: Int]] = [:]
    let capacity: Int
    let expertCount: Int
    let layers: Int

    init(capacity: Int, expertCount: Int, layers: Int) {
        self.capacity = capacity
        self.expertCount = expertCount
        self.layers = layers
    }

    func register(layer: Int, proj: String, spec: ProjSpec) {
        specs["\(layer).\(proj)", default: [:]][proj] = spec
        _ = spec
        var bag = specs["L\(layer)"] ?? [:]
        bag[proj] = spec
        specs["L\(layer)"] = bag
    }

    func attach(to model: DeepseekV4Model) {
        // Official V4.1 streams experts itself. TIE's SwitchLinear hook would
        // treat already-local 0..<k indices as global ids and restack garbage.
        TierHooks.switchLinearSyncHook = nil
        for layer in model.model.layers {
            layer.ffn.expertBank = self
        }
    }

    /// Packed MXFP4 `(weight, scales)` for `layer.expert.proj`, LRU-capped.
    private var packed: [String: (MLXArray, MLXArray)] = [:]
    /// Disk fetch — set by the official loader. Returns nil on missing tensors.
    var fetchPacked: ((Int, Int, String) -> (MLXArray, MLXArray)?)?

    func ensure(layer: Int, indices: MLXArray, glu: DeepseekV4SwitchGLU) -> MLXArray {
        let needed = uniqueInts(indices).filter { $0 >= 0 && $0 < expertCount }
        lock.lock()
        defer { lock.unlock() }
        var map = slotOf[layer] ?? [:]
        var order = lru[layer] ?? []
        for e in needed {
            if let idx = order.firstIndex(of: e) { order.remove(at: idx) }
            order.append(e)
            if map[e] == nil {
                let slot: Int
                if map.count < capacity {
                    slot = map.count
                } else {
                    let evict = order.first { !needed.contains($0) } ?? order[0]
                    slot = map[evict]!
                    map.removeValue(forKey: evict)
                    order.removeAll { $0 == evict }
                    for p in ["gate_proj", "up_proj", "down_proj"] {
                        packed.removeValue(forKey: packKey(layer, evict, p))
                    }
                }
                writeSlot(glu, layer: layer, expert: e, slot: slot)
                map[e] = slot
            }
        }
        slotOf[layer] = map
        lru[layer] = order
        return remap(indices, map: map)
    }

    private func packKey(_ layer: Int, _ expert: Int, _ proj: String) -> String {
        "\(layer).\(expert).\(proj)"
    }

    /// oMLX: `rw[slot] = fetch(...)` into a stable capacity-sized table. Do not
    /// restack k experts — that remaps 0..<k every token and collapses decode.
    private func writeSlot(
        _ glu: DeepseekV4SwitchGLU, layer: Int, expert: Int, slot: Int
    ) {
        func apply(_ lin: SwitchLinear, _ proj: String) {
            let key = packKey(layer, expert, proj)
            if packed[key] == nil, let t = fetchPacked?(layer, expert, proj) {
                MLX.eval(t.0, t.1)
                packed[key] = t
            }
            guard let t = packed[key] else { return }
            lin.weight[slot] = t.0
            if let q = lin as? QuantizedSwitchLinear {
                q.scales[slot] = t.1
            }
        }
        apply(glu.gateProj, "gate_proj")
        apply(glu.upProj, "up_proj")
        apply(glu.downProj, "down_proj")
    }

    private func uniqueInts(_ indices: MLXArray) -> [Int] {
        let flat = indices.asType(.int32).flattened()
        MLX.eval(flat)
        var seen = Set<Int>()
        var out: [Int] = []
        for i in 0..<flat.size {
            let v = Int(flat[i].item(Int32.self))
            if seen.insert(v).inserted { out.append(v) }
        }
        return out
    }

    private func remap(_ indices: MLXArray, map: [Int: Int]) -> MLXArray {
        let flat = indices.asType(.int32).flattened()
        MLX.eval(flat)
        var mapped = [Int32](repeating: 0, count: flat.size)
        for i in 0..<flat.size {
            let e = Int(flat[i].item(Int32.self))
            mapped[i] = Int32(map[e] ?? 0)
        }
        return MLXArray(mapped).reshaped(indices.shape)
    }
}

struct SafetensorEntry {
    let dtype: String
    let shape: [Int]
    let begin: Int
    let end: Int
}

final class SafetensorMMap {
    let url: URL
    let header: [String: SafetensorEntry]
    let dataStart: Int
    private let handle: FileHandle

    func handleForSeek() -> FileHandle { handle }

    init(url: URL) throws {
        self.url = url
        let fh = try FileHandle(forReadingFrom: url)
        guard let lenBytes = try fh.read(upToCount: 8), lenBytes.count == 8 else {
            throw NSError(domain: "v41", code: 1)
        }
        let headerLen = lenBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard let headerData = try fh.read(upToCount: Int(headerLen)),
            headerData.count == Int(headerLen),
            let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
        else { throw NSError(domain: "v41", code: 2) }
        var header: [String: SafetensorEntry] = [:]
        for (k, v) in json where k != "__metadata__" {
            guard let e = v as? [String: Any],
                let dtype = e["dtype"] as? String,
                let shapeAny = e["shape"] as? [Any],
                let offAny = e["data_offsets"] as? [Any], offAny.count == 2
            else { continue }
            let shape = shapeAny.compactMap { Self.jsonInt($0) }
            let begin = Self.jsonInt(offAny[0]) ?? 0
            let end = Self.jsonInt(offAny[1]) ?? 0
            header[k] = SafetensorEntry(dtype: dtype, shape: shape, begin: begin, end: end)
        }
        self.header = header
        self.dataStart = 8 + Int(headerLen)
        self.handle = fh
    }

    private static func jsonInt(_ any: Any) -> Int? {
        if let i = any as? Int { return i }
        if let n = any as? NSNumber { return n.intValue }
        return nil
    }

    func bytes(_ key: String) throws -> (Data, SafetensorEntry) {
        guard let e = header[key] else {
            throw NSError(domain: "v41", code: 3, userInfo: [NSLocalizedDescriptionKey: key])
        }
        try handle.seek(toOffset: UInt64(dataStart + e.begin))
        let n = e.end - e.begin
        guard let chunk = try handle.read(upToCount: n), chunk.count == n else {
            throw NSError(domain: "v41", code: 6, userInfo: [NSLocalizedDescriptionKey: "short read \(key)"])
        }
        return (chunk, e)
    }
}

enum DeepseekV41OfficialLoader {
    static func isOfficialCheckpoint(at dir: URL) -> Bool {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        if json["omlx_deepseek_v41"] != nil { return false }
        let q = json["quantization_config"] as? [String: Any]
        return (json["model_type"] as? String) == "deepseek_v41"
            && (q?["quant_method"] as? String) == "fp8"
    }

    static func flattenConfig(_ data: Data) throws -> Data {
        guard var obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return data
        }
        if let text = obj["text_config"] as? [String: Any] {
            for (k, v) in text where obj[k] == nil { obj[k] = v }
        }
        if isOfficialJSON(obj) {
            let n = intVal(obj["n_routed_experts"]) ?? 384
            let k = intVal(obj["num_experts_per_tok"]) ?? 6
            obj["expert_resident_capacity"] = max(k, Int((Double(n) * 0.05).rounded()))
        }
        return try JSONSerialization.data(withJSONObject: obj)
    }

    private static func intVal(_ any: Any?) -> Int? {
        if let i = any as? Int { return i }
        if let n = any as? NSNumber { return n.intValue }
        return nil
    }

    private static func isOfficialJSON(_ obj: [String: Any]) -> Bool {
        obj["omlx_deepseek_v41"] == nil
            && (obj["model_type"] as? String) == "deepseek_v41"
            && ((obj["quantization_config"] as? [String: Any])?["quant_method"] as? String) == "fp8"
    }

    static func loadIfOfficial(dir: URL, model: Module) throws -> Bool {
        guard isOfficialCheckpoint(at: dir), let v41 = model as? DeepseekV4Model else {
            return false
        }
        try load(dir: dir, model: v41)
        return true
    }

    static func load(dir: URL, model: DeepseekV4Model) throws {
        NovaMLXLog.info("[V41] official load start capacity=\(model.args.expertResidentCapacity ?? -1)")
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: Data(contentsOf: indexURL))
        var currentName: String?
        var currentFile: SafetensorMMap?
        func file(_ name: String) throws -> SafetensorMMap {
            if currentName == name, let opened = currentFile { return opened }
            let opened = try SafetensorMMap(url: dir.appendingPathComponent(name))
            currentFile = opened
            currentName = name
            return opened
        }

        var weights = [String: MLXArray]()
        var qspec = [String: (Int, Int, QuantizationMode)]()
        let capacity = model.args.expertResidentCapacity ?? model.args.nRoutedExperts
        let bank = DeepseekV41ExpertBank(
            capacity: capacity, expertCount: model.args.nRoutedExperts,
            layers: model.args.numHiddenLayers)

        for (key, filename) in index.weightMap {
            if key.contains(".engram.") { continue }
            if key.contains(".ffn.experts.") { continue }
            if key.contains("bias_vl") { continue }
            if key.hasPrefix("mtp.") { continue }
            if key.hasPrefix("vision.") || key.hasPrefix("aligner.") || key.hasPrefix("image_") {
                continue
            }
            do {
                let mmapFile = try file(filename)
                let (raw, entry) = try mmapFile.bytes(key)
                let mapped = mapKey(key)
                if key.hasSuffix(".scale") { continue }
                if weights.count % 25 == 0 {
                    NovaMLXLog.info("[V41] loading \(weights.count) \(key) \(entry.dtype) \(entry.shape)")
                }
                if key.hasSuffix(".weight"), let scaleName = index.weightMap[scaleKey(key)] {
                    let (scaleRaw, scaleEntry) = try file(scaleName).bytes(scaleKey(key))
                    let (w, s, bits, mode) = try repack(
                        weight: raw, wEntry: entry, scale: scaleRaw, sEntry: scaleEntry)
                    // oMLX convert force-dequantizes wo_a so grouped einsum is dense.
                    if key.contains("wo_a.weight") {
                        weights[mapped] = dequantized(
                            w, scales: s, biases: nil, groupSize: 32, bits: bits, mode: mode
                        ).asType(.bfloat16)
                    } else {
                        weights[mapped] = w
                        weights[mapped.replacingOccurrences(of: ".weight", with: ".scales")] = s
                        let path = String(mapped.dropLast(".weight".count))
                        qspec[path] = (32, bits, mode)
                    }
                } else {
                    weights[mapped] = try decodeDense(raw, entry)
                }
            } catch {
                NovaMLXLog.error("[V41] skip \(key): \(error)")
            }
        }

        NovaMLXLog.info("[V41] read \(weights.count) dense tensors")
        var flag = false
        weights = DeepseekV4Sanitizer.remap(weights, config: model.args, nativeMtp: &flag)
        model.nativeMtpAvailable = false
        NovaMLXLog.info("[V41] sanitized \(weights.count) tensors, quantizing")

        quantize(model: model) { path, module in
            if module is QuantizedSwitchLinear { return nil }
            if let spec = qspec[path] { return spec }
            if weights["\(path).scales"] != nil {
                return (32, 8, .mxfp8)
            }
            return nil
        }

        let parameters = ModuleParameters.unflattened(weights)
        model.update(parameters: parameters)

        bank.fetchPacked = { layer, expert, proj in
            let src: String
            switch proj {
            case "gate_proj": src = "w1"
            case "up_proj": src = "w3"
            default: src = "w2"
            }
            let wKey = "layers.\(layer).ffn.experts.\(expert).\(src).weight"
            let sKey = "layers.\(layer).ffn.experts.\(expert).\(src).scale"
            guard let wFile = index.weightMap[wKey], let sFile = index.weightMap[sKey] else {
                return nil
            }
            guard let wm = try? file(wFile).bytes(wKey),
                let sm = try? file(sFile).bytes(sKey),
                let packed = try? repack(weight: wm.0, wEntry: wm.1, scale: sm.0, sEntry: sm.1)
            else { return nil }
            if layer == 0, expert == 0, proj == "gate_proj" {
                NovaMLXLog.info(
                    "[V41] expert L0 E\(expert) w1 packed=\(packed.0.shape) scales=\(packed.1.shape)"
                )
            }
            return (packed.0, packed.1)
        }
        bank.attach(to: model)
        try attachEngram(dir: dir, index: index, file: file, model: model)
        NovaMLXLog.info(
            "[V41] official loader: dense tensors=\(weights.count) expertSlots=\(capacity)/\(model.args.nRoutedExperts)"
        )
    }

    private static func attachEngram(
        dir: URL, index: SafetensorsIndex, file: (String) throws -> SafetensorMMap,
        model: DeepseekV4Model
    ) throws {
        let hashURL = dir.appendingPathComponent("engram_hash.json")
        guard FileManager.default.fileExists(atPath: hashURL.path) else {
            NovaMLXLog.info("[V41] no engram_hash.json, skipping Engram")
            return
        }
        let meta = try JSONDecoder().decode(
            DeepseekV41EngramHash.self, from: Data(contentsOf: hashURL))
        model.model.engramMeta = meta
        for (ix, layerId) in model.args.engramLayerIds.enumerated() {
            let wKey = "layers.\(layerId).engram.embed.weight"
            let sKey = "layers.\(layerId).engram.embed.scale"
            guard let wf = index.weightMap[wKey], let sf = index.weightMap[sKey] else { continue }
            let table = try DeepseekV41EngramTable(
                dir: dir, weightKey: wKey, scaleKey: sKey, weightFile: wf, scaleFile: sf)
            let eng = DeepseekV41Engram(config: model.args, tableIndex: ix, table: table)
            if let qf = index.weightMap["layers.\(layerId).engram.q_weight"],
                let kf = index.weightMap["layers.\(layerId).engram.k_weight"]
            {
                let q = try file(qf).bytes("layers.\(layerId).engram.q_weight")
                let k = try file(kf).bytes("layers.\(layerId).engram.k_weight")
                try eng.update(
                    parameters: ModuleParameters.unflattened([
                        "q_weight": decodeDense(q.0, q.1),
                        "k_weight": decodeDense(k.0, k.1),
                    ]))
            }
            let wkvW = "layers.\(layerId).engram.wkv.weight"
            let wkvS = "layers.\(layerId).engram.wkv.scale"
            if let wf2 = index.weightMap[wkvW], let sf2 = index.weightMap[wkvS] {
                let w = try file(wf2).bytes(wkvW)
                let s = try file(sf2).bytes(wkvS)
                let packed = try repack(weight: w.0, wEntry: w.1, scale: s.0, sEntry: s.1)
                eng.replaceWkv(
                    QuantizedLinear(
                        weight: packed.0, bias: nil, scales: packed.1, biases: nil,
                        groupSize: 32, bits: packed.2, mode: packed.3))
            }
            model.model.layers[layerId].engram = eng
            NovaMLXLog.info("[V41] attached Engram layer \(layerId)")
        }
    }

    private static func scaleKey(_ weightKey: String) -> String {
        String(weightKey.dropLast(".weight".count)) + ".scale"
    }

    private static func mapKey(_ key: String) -> String {
        if key.hasPrefix("language_model.") {
            return "model." + key.dropFirst("language_model.".count)
        }
        if key == "head.weight" { return "lm_head.weight" }
        if key == "embed.weight" { return "model.embed_tokens.weight" }
        if key == "norm.weight" { return "model.norm.weight" }
        if key.hasPrefix("layers.") || key.hasPrefix("mtp.") {
            return "model." + key
        }
        return key
    }

    private static func decodeDense(_ raw: Data, _ entry: SafetensorEntry) throws -> MLXArray {
        switch entry.dtype {
        case "BF16":
            return arrayFromBytes(raw, shape: entry.shape, dtype: .bfloat16)
        case "F32":
            return arrayFromBytes(raw, shape: entry.shape, dtype: .float32)
        case "F16":
            return arrayFromBytes(raw, shape: entry.shape, dtype: .float16)
        case "I32":
            return arrayFromBytes(raw, shape: entry.shape, dtype: .int32)
        default:
            throw NSError(
                domain: "v41", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "dense dtype \(entry.dtype)"])
        }
    }

    static func arrayFromBytes(_ data: Data, shape: [Int], dtype: DType) -> MLXArray {
        let expected = shape.reduce(1, *) * dtype.size
        precondition(
            data.count == expected,
            "v41 byte count \(data.count) != \(expected) shape=\(shape) dtype=\(dtype)")
        let copy = UnsafeMutableRawPointer.allocate(byteCount: data.count, alignment: 64)
        data.withUnsafeBytes { src in
            copy.copyMemory(from: src.baseAddress!, byteCount: data.count)
        }
        return MLXArray(rawPointer: copy, shape, dtype: dtype) {
            copy.deallocate()
        }
    }

    static func repack(
        weight: Data, wEntry: SafetensorEntry, scale: Data, sEntry: SafetensorEntry
    ) throws -> (MLXArray, MLXArray, Int, QuantizationMode) {
        let bits: Int
        let mode: QuantizationMode
        switch wEntry.dtype {
        case "I8", "U8":
            bits = 4
            mode = .mxfp4
        case "F8_E4M3", "F8_E4M3FN":
            bits = 8
            mode = .mxfp8
        default:
            throw NSError(
                domain: "v41", code: 5,
                userInfo: [NSLocalizedDescriptionKey: "quant dtype \(wEntry.dtype)"])
        }
        let u8w = arrayFromBytes(weight, shape: wEntry.shape, dtype: .uint8)
        let packedCount = u8w.size / 4
        let packedShape = [wEntry.shape[0], wEntry.shape[1] * 8 / bits / 8]
        // last dim in uint32 units: (logical_in * bits / 32)
        let logicalIn = wEntry.shape[1] * (8 / bits)
        let packedIn = logicalIn * bits / 32
        let w32 = arrayFromBytes(weight, shape: [wEntry.shape[0], packedIn], dtype: .uint32)
        _ = packedCount
        _ = packedShape

        var scales = arrayFromBytes(scale, shape: sEntry.shape, dtype: .uint8)
        if bits == 8, sEntry.shape[0] * 32 == wEntry.shape[0] {
            scales = MLX.repeated(scales.expandedDimensions(axis: 1), count: 32, axis: 1)
                .reshaped([wEntry.shape[0], sEntry.shape[1]])
        }
        return (w32, scales, bits, mode)
    }
}

private struct SafetensorsIndex: Codable {
    let weightMap: [String: String]
    enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
}
