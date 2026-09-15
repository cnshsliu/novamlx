import Foundation
import MLX
import NovaMLXCore
import NovaMLXUtils

/// Pure-Swift TIE layout converter. Splits a downloaded MLX checkpoint into
/// `tier0.safetensors` + `tie-shards/` so the worker can stream experts from SSD.
///
/// In-place: original `model-*.safetensors` move to `source-shards/` after success
/// so a later Remove TIE can restore them.
public enum ExpertShardConverter {
    public static let manifestName = "tier-manifest.json"
    public static let tier0Name = "tier0.safetensors"
    public static let shardsDirName = "tie-shards"
    public static let sourceDirName = "source-shards"
    public static let lockName = "tie-convert.lock"
    public static let workDirName = ".tie-work"

    public struct Progress: Sendable {
        public let message: String
        public let done: Int
        public let total: Int
        public var fraction: Double {
            total > 0 ? Double(done) / Double(total) : 0
        }
    }

    // MARK: - Inspect / delete

    public static func inspect(at modelDir: URL, converting: TieStatusInfo? = nil) -> TieStatusInfo {
        if let converting, converting.status == .converting { return converting }
        let fm = FileManager.default
        let lock = modelDir.appendingPathComponent(lockName)
        let manifestURL = modelDir.appendingPathComponent(manifestName)
        if fm.fileExists(atPath: lock.path) {
            return TieStatusInfo(
                status: .incomplete,
                message: "Conversion was interrupted. Remove the TIE layout and convert again."
            )
        }
        if fm.fileExists(atPath: manifestURL.path) {
            if let reason = validationError(at: modelDir) {
                return TieStatusInfo(status: .incomplete, message: reason)
            }
            return TieStatusInfo(status: .ready, message: "TIE (SSD streaming)")
        }
        if canConvert(at: modelDir) {
            return TieStatusInfo(
                status: .convertible,
                message: "Can convert to TIE for SSD streaming"
            )
        }
        return TieStatusInfo(status: .none)
    }

    public static func canConvert(at modelDir: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: modelDir.appendingPathComponent("config.json").path) else {
            return false
        }
        if fm.fileExists(atPath: modelDir.appendingPathComponent(manifestName).path) {
            return false
        }
        guard !topLevelSafetensors(in: modelDir).isEmpty else { return false }
        guard let data = try? Data(contentsOf: modelDir.appendingPathComponent("config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return false }
        let experts = (json["n_routed_experts"] as? Int)
            ?? (json["num_experts"] as? Int)
            ?? (json["num_local_experts"] as? Int)
            ?? 0
        let layers = (json["num_hidden_layers"] as? Int) ?? 0
        return experts > 0 || layers > 0
    }

    public static func validationError(at modelDir: URL) -> String? {
        let fm = FileManager.default
        if fm.fileExists(atPath: modelDir.appendingPathComponent(lockName).path) {
            return "Conversion lock is still present"
        }
        let manifestURL = modelDir.appendingPathComponent(manifestName)
        guard fm.fileExists(atPath: manifestURL.path) else { return nil }
        guard let manifest = try? TierManifestLoader.loadIfPresent(modelDir: modelDir) else {
            return "Unreadable tier-manifest.json"
        }
        let tier0 = modelDir.appendingPathComponent(manifest.tier0File)
        if !fm.fileExists(atPath: tier0.path) {
            return "Missing \(manifest.tier0File)"
        }
        let shards = modelDir.appendingPathComponent(shardsDirName)
        for e in manifest.experts {
            let a = shards.appendingPathComponent(e.file)
            let b = modelDir.appendingPathComponent(e.file)
            if !fm.fileExists(atPath: a.path) && !fm.fileExists(atPath: b.path) {
                return "Missing expert shard \(e.file)"
            }
        }
        for l in manifest.layers ?? [] {
            let a = shards.appendingPathComponent(l.file)
            let b = modelDir.appendingPathComponent(l.file)
            if !fm.fileExists(atPath: a.path) && !fm.fileExists(atPath: b.path) {
                return "Missing layer shard \(l.file)"
            }
        }
        if manifest.expertCount == 0 && (manifest.layers ?? []).isEmpty {
            return "Manifest has no expert or layer shards"
        }
        return nil
    }

    public static func removeLayout(at modelDir: URL) throws {
        let fm = FileManager.default
        for name in [manifestName, tier0Name, lockName] {
            let url = modelDir.appendingPathComponent(name)
            if fm.fileExists(atPath: url.path) {
                try fm.removeItem(at: url)
            }
        }
        for name in [shardsDirName, workDirName] {
            let url = modelDir.appendingPathComponent(name)
            if fm.fileExists(atPath: url.path) {
                try fm.removeItem(at: url)
            }
        }
        let source = modelDir.appendingPathComponent(sourceDirName)
        if fm.fileExists(atPath: source.path) {
            let items = try fm.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
            for item in items {
                let dest = modelDir.appendingPathComponent(item.lastPathComponent)
                if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
                try fm.moveItem(at: item, to: dest)
            }
            try? fm.removeItem(at: source)
        }
    }

    // MARK: - Convert

    public static func convert(
        at modelDir: URL,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) throws {
        let fm = FileManager.default
        let lock = modelDir.appendingPathComponent(lockName)
        try "converting".write(to: lock, atomically: true, encoding: .utf8)
        defer { try? fm.removeItem(at: lock) }

        let work = modelDir.appendingPathComponent(workDirName)
        if fm.fileExists(atPath: work.path) { try fm.removeItem(at: work) }
        try fm.createDirectory(at: work, withIntermediateDirectories: true)
        let workShards = work.appendingPathComponent(shardsDirName)
        try fm.createDirectory(at: workShards, withIntermediateDirectories: true)

        let files = topLevelSafetensors(in: modelDir)
        guard !files.isEmpty else {
            throw NovaMLXError.tieConversionFailed(modelDir.lastPathComponent, "No safetensors files")
        }

        report(progress, "Reading tensor index…", 0, files.count)
        var fileFor: [String: URL] = [:]
        var nbytes: [String: Int] = [:]
        for (i, file) in files.enumerated() {
            let listed = try SafetensorsHeader.list(in: file)
            for (name, size) in listed {
                fileFor[name] = file
                nbytes[name] = size
            }
            report(progress, "Indexed \(file.lastPathComponent)", i + 1, files.count)
        }

        var classic: [ExpertID: [String]] = [:]
        var stacked: [StackedKey: String] = [:]
        var tier0: [String] = []
        for name in fileFor.keys {
            switch classify(name, nbytes: nbytes[name] ?? 0) {
            case .skip:
                continue
            case .classic(let id, _):
                classic[id, default: []].append(name)
            case .stacked(let key):
                stacked[key] = name
            case .other:
                tier0.append(name)
            }
        }

        let layerBuckets = splitLayers(&tier0)
        var nameToLayer: [String: Int] = [:]
        for (L, names) in layerBuckets {
            for n in names { nameToLayer[n] = L }
        }
        let tier0Set = Set(tier0)
        let expectedClassic = classic.mapValues(\.count)
        let totalSteps = max(1, files.count + 3)
        var step = 0

        var pendingClassic: [ExpertID: [String: MLXArray]] = [:]
        var pendingLayers: [Int: [String: MLXArray]] = [:]
        var pendingTier0: [String: MLXArray] = [:]
        var pendingStacked: [StackedKey: MLXArray] = [:]
        var expertEntries: [TierManifest.ExpertEntry] = []

        for file in files {
            step += 1
            report(progress, "Reading \(file.lastPathComponent)…", step, totalSteps + expectedClassic.count)
            let arrays = try MLX.loadArrays(url: file)
            for (name, tensor) in arrays {
                switch classify(name, nbytes: nbytes[name] ?? 0) {
                case .skip:
                    continue
                case .classic(let id, _):
                    pendingClassic[id, default: [:]][
                        classicExpertSwitchName(name, layer: id.layer, expert: id.expert)
                    ] = tensor
                    if pendingClassic[id]?.count == expectedClassic[id] {
                        let bucket = pendingClassic[id] ?? [:]
                        pendingClassic[id] = nil
                        let fname = String(format: "expert.L%02d.E%03d.safetensors", id.layer, id.expert)
                        let out = workShards.appendingPathComponent(fname)
                        try MLX.save(arrays: bucket, url: out)
                        let size = (try? fm.attributesOfItem(atPath: out.path)[.size] as? NSNumber)?.int64Value ?? 0
                        expertEntries.append(
                            .init(
                                layer: id.layer, expert: id.expert, file: fname, bytes: size,
                                tensors: Array(bucket.keys)
                            )
                        )
                    }
                case .stacked(let key):
                    pendingStacked[key] = tensor
                case .other:
                    if let L = nameToLayer[name] {
                        pendingLayers[L, default: [:]][swiftTensorName(name)] = tensor
                    } else if tier0Set.contains(name) {
                        pendingTier0[swiftTensorName(name)] = tensor
                    }
                }
            }
        }

        if !stacked.isEmpty {
            var dummy = step
            expertEntries = try writeStackedFromPending(
                pendingStacked, dest: workShards, progress: progress, step: &dummy, total: totalSteps
            )
        }

        var layerEntries: [TierManifest.LayerEntry] = []
        for L in pendingLayers.keys.sorted() {
            let bucket = pendingLayers[L] ?? [:]
            guard !bucket.isEmpty else { continue }
            let fname = String(format: "layer.L%02d.safetensors", L)
            let out = workShards.appendingPathComponent(fname)
            try MLX.save(arrays: bucket, url: out)
            let size = (try? fm.attributesOfItem(atPath: out.path)[.size] as? NSNumber)?.int64Value ?? 0
            layerEntries.append(
                .init(layer: L, file: fname, bytes: size, tensors: Array(bucket.keys))
            )
        }

        report(progress, "Writing shared (tier 0) weights…", totalSteps - 1, totalSteps)
        let tier0Bucket = pendingTier0
        let tier0URL = work.appendingPathComponent(tier0Name)
        if !tier0Bucket.isEmpty {
            try MLX.save(arrays: tier0Bucket, url: tier0URL)
        } else {
            throw NovaMLXError.tieConversionFailed(
                modelDir.lastPathComponent, "No shared (tier 0) tensors found"
            )
        }
        let tier0Bytes = (try? fm.attributesOfItem(atPath: tier0URL.path)[.size] as? NSNumber)?.int64Value ?? 0

        let strategy: TierStrategy
        if !expertEntries.isEmpty && !layerEntries.isEmpty {
            strategy = .mixed
        } else if !expertEntries.isEmpty {
            strategy = .expert
        } else if !layerEntries.isEmpty {
            strategy = .layer
        } else {
            strategy = .none
        }

        let cfg = (try? Data(contentsOf: modelDir.appendingPathComponent("config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let arch = (cfg?["model_type"] as? String) ?? "unknown"

        let manifest = TierManifest(
            version: 1,
            converter: "NovaMLX-ExpertShardConverter",
            sourceModel: modelDir.path,
            architecture: arch,
            layout: stacked.isEmpty ? "classic" : "stacked",
            strategy: strategy,
            tier0File: tier0Name,
            tier0TensorCount: tier0Bucket.count,
            tier0Bytes: tier0Bytes,
            expertCount: expertEntries.count,
            experts: expertEntries,
            layers: layerEntries
        )
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try enc.encode(manifest).write(to: work.appendingPathComponent(manifestName))

        report(progress, "Installing TIE layout…", totalSteps - 1, totalSteps)
        try installWork(work: work, modelDir: modelDir)
        report(progress, "TIE conversion complete", totalSteps, totalSteps)
    }

    public static func shouldAutoConvertOnLoad(at modelDir: URL, estimatedBytes: UInt64, gpuBudgetBytes: UInt64) -> Bool {
        let status = inspect(at: modelDir)
        guard status.status == .convertible else { return false }
        let needed = estimatedBytes + MemoryFeasibility.evaluateSafetyMargin(estimatedBytes: estimatedBytes)
        return gpuBudgetBytes > 0 && needed > gpuBudgetBytes
    }

    // MARK: - Internals

    private enum Kind {
        case skip
        case classic(ExpertID, String)
        case stacked(StackedKey)
        case other
    }

    private struct ExpertID: Hashable, Comparable {
        let layer: Int
        let expert: Int
        static func < (lhs: Self, rhs: Self) -> Bool {
            if lhs.layer != rhs.layer { return lhs.layer < rhs.layer }
            return lhs.expert < rhs.expert
        }
    }

    private struct StackedKey: Hashable {
        let layer: Int
        let proj: String
        let suffix: String
    }

    private static func classify(_ name: String, nbytes: Int = 0) -> Kind {
        let lower = name.lowercased()
        if lower.hasPrefix("vision.") || lower.hasPrefix("aligner.") || lower.hasPrefix("image_") {
            return .skip
        }
        if lower.contains("engram") || lower.contains("confidence_head")
            || lower.contains("markov_head") || lower.contains(".main_proj")
            || lower.contains(".main_norm") || lower.hasSuffix(".ffn.gate.bias_vl")
            || lower.contains("rotary_emb.inv_freq")
        {
            return .skip
        }
        if nbytes > 2_000_000_000 { return .skip }
        if name.hasSuffix(".norm.weight")
            && !name.contains("attn_norm") && !name.contains("ffn_norm")
            && !name.contains("kv_norm") && !name.contains("q_norm")
            && name != "norm.weight" && !name.hasSuffix("model.norm.weight")
            && name != "language_model.norm.weight"
        {
            return .skip
        }

        if let m = match(name, #"^mtp\.(\d+)\.ffn\.experts\.(\d+)\."#) {
            return .classic(ExpertID(layer: 10_000 + m[0], expert: m[1]), name)
        }
        if let m = match(name, #"layers\.(\d+)(?:\.(?:mlp|ffn))?\.experts\.(\d+)\."#) {
            return .classic(ExpertID(layer: m[0], expert: m[1]), name)
        }
        if let sm = matchString(name, #"layers\.(\d+)(?:\.\w+)?\.switch_mlp\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.([\w.]+)$"#) {
            return .stacked(StackedKey(layer: Int(sm[0]) ?? 0, proj: sm[1], suffix: sm[2]))
        }
        return .other
    }

    private static func splitLayers(_ tier0: inout [String]) -> [Int: [String]] {
        var byLayer: [Int: [String]] = [:]
        var keep: [String] = []
        for n in tier0 {
            let lower = n.lowercased()
            if lower.contains("norm") || lower.contains("hc_attn") || lower.contains("hc_ffn")
                || lower.contains("attn_hc") || lower.contains("ffn_hc") || lower.contains("hc_head")
                || lower.contains(".gate.") || lower.contains("attn_sink") || lower.contains("tid2eid")
            {
                keep.append(n)
                continue
            }
            if let m = match(n, #"^(?:language_model|model)\.layers\.(\d+)\."#) {
                byLayer[m[0], default: []].append(n)
            } else if let m = match(n, #"^mtp\.(\d+)\."#) {
                byLayer[10_000 + m[0], default: []].append(n)
            } else {
                keep.append(n)
            }
        }
        tier0 = keep
        return byLayer
    }

    private static func swiftTensorName(_ name: String) -> String {
        var k = name
        if k.hasPrefix("language_model.head.") {
            k = "lm_head." + k.dropFirst("language_model.head.".count)
        } else if k.hasPrefix("language_model.") {
            k = "model." + k.dropFirst("language_model.".count)
        } else if k == "norm.weight" || k.hasPrefix("norm.") {
            k = "model." + k
        } else if k.hasPrefix("mtp.") {
            let rest = k.dropFirst("mtp.".count)
            if let dot = rest.firstIndex(of: ".") {
                let idx = rest[..<dot]
                let tail = rest[rest.index(after: dot)...]
                k = "model.mtpLayers.\(idx).\(tail)"
            }
        }
        k = k.replacingOccurrences(of: ".hc_attn_fn", with: ".attn_hc.fn")
        k = k.replacingOccurrences(of: ".hc_attn_base", with: ".attn_hc.base")
        k = k.replacingOccurrences(of: ".hc_attn_scale", with: ".attn_hc.scale")
        k = k.replacingOccurrences(of: ".hc_ffn_fn", with: ".ffn_hc.fn")
        k = k.replacingOccurrences(of: ".hc_ffn_base", with: ".ffn_hc.base")
        k = k.replacingOccurrences(of: ".hc_ffn_scale", with: ".ffn_hc.scale")
        k = k.replacingOccurrences(of: ".ffn.gate.bias", with: ".ffn.gate.e_score_correction_bias")
        if k.contains(".shared_experts") {
            k = k.replacingOccurrences(of: ".shared_experts.w1.", with: ".shared_experts.gate_proj.")
            k = k.replacingOccurrences(of: ".shared_experts.w3.", with: ".shared_experts.up_proj.")
            k = k.replacingOccurrences(of: ".shared_experts.w2.", with: ".shared_experts.down_proj.")
        }
        return k
    }

    private static func classicExpertSwitchName(_ name: String, layer: Int, expert: Int) -> String {
        let marker = ".experts.\(expert)."
        guard let range = name.range(of: marker) else { return swiftTensorName(name) }
        let projSuffix = String(name[range.upperBound...])
        let parts = projSuffix.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        let projRaw = String(parts.first ?? "")
        let suffix = parts.count > 1 ? String(parts[1]) : "weight"
        let proj = ["w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"][projRaw] ?? projRaw
        if layer >= 10_000 {
            return "model.mtpLayers.\(layer - 10_000).ffn.switch_mlp.\(proj).\(suffix)"
        }
        return "model.layers.\(layer).ffn.switch_mlp.\(proj).\(suffix)"
    }

    private static func writeStackedFromPending(
        _ tensors: [StackedKey: MLXArray],
        dest: URL,
        progress: (@Sendable (Progress) -> Void)?,
        step: inout Int,
        total: Int
    ) throws -> [TierManifest.ExpertEntry] {
        var byLayer: [Int: [StackedKey: MLXArray]] = [:]
        for (key, t) in tensors {
            byLayer[key.layer, default: [:]][key] = t
        }
        var entries: [TierManifest.ExpertEntry] = []
        let fm = FileManager.default
        for L in byLayer.keys.sorted() {
            let layerTensors = byLayer[L] ?? [:]
            guard let first = layerTensors.values.first else { continue }
            let numExperts = first.dim(0)
            for E in 0..<numExperts {
                var bucket: [String: MLXArray] = [:]
                for (key, t) in layerTensors {
                    let sliced: MLXArray = t.ndim >= 2 ? t[E, 0...] : t[E]
                    let name = "model.layers.\(L).switch_mlp.\(key.proj).\(key.suffix)"
                    bucket[name] = sliced
                }
                let fname = String(format: "expert.L%02d.E%03d.safetensors", L, E)
                let out = dest.appendingPathComponent(fname)
                try MLX.save(arrays: bucket, url: out)
                let size = (try? fm.attributesOfItem(atPath: out.path)[.size] as? NSNumber)?.int64Value ?? 0
                entries.append(
                    .init(
                        layer: L, expert: E, file: fname, bytes: size,
                        tensors: Array(bucket.keys), stackedSource: true
                    )
                )
            }
            step += 1
            report(progress, "Sliced layer \(L) experts", step, total)
        }
        return entries
    }

    private static func installWork(work: URL, modelDir: URL) throws {
        let fm = FileManager.default
        let shards = modelDir.appendingPathComponent(shardsDirName)
        if fm.fileExists(atPath: shards.path) { try fm.removeItem(at: shards) }
        try fm.moveItem(at: work.appendingPathComponent(shardsDirName), to: shards)
        let tier0Src = work.appendingPathComponent(tier0Name)
        let tier0Dst = modelDir.appendingPathComponent(tier0Name)
        if fm.fileExists(atPath: tier0Dst.path) { try fm.removeItem(at: tier0Dst) }
        try fm.moveItem(at: tier0Src, to: tier0Dst)
        let manSrc = work.appendingPathComponent(manifestName)
        let manDst = modelDir.appendingPathComponent(manifestName)
        if fm.fileExists(atPath: manDst.path) { try fm.removeItem(at: manDst) }
        try fm.moveItem(at: manSrc, to: manDst)
        try? fm.removeItem(at: work)

        let source = modelDir.appendingPathComponent(sourceDirName)
        try fm.createDirectory(at: source, withIntermediateDirectories: true)
        let contents = try fm.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        for item in contents where item.pathExtension == "safetensors" {
            let name = item.lastPathComponent
            if name == tier0Name { continue }
            if name.hasPrefix("layer.") || name.hasPrefix("expert.") { continue }
            let dest = source.appendingPathComponent(name)
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: item, to: dest)
        }
        let index = modelDir.appendingPathComponent("model.safetensors.index.json")
        if fm.fileExists(atPath: index.path) {
            let dest = source.appendingPathComponent(index.lastPathComponent)
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: index, to: dest)
        }
    }

    private static func topLevelSafetensors(in dir: URL) -> [URL] {
        let contents =
            (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return contents
            .filter { $0.pathExtension == "safetensors" }
            .filter {
                let n = $0.lastPathComponent
                return n != tier0Name && !n.hasPrefix("expert.") && !n.hasPrefix("layer.")
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func allTensorNames(in dir: URL) -> [String] {
        var names: [String] = []
        for file in topLevelSafetensors(in: dir) {
            if let listed = try? SafetensorsHeader.list(in: file) {
                names.append(contentsOf: listed.map(\.0))
            }
        }
        return names
    }

    private static func report(
        _ progress: (@Sendable (Progress) -> Void)?,
        _ message: String, _ done: Int, _ total: Int
    ) {
        progress?(Progress(message: message, done: done, total: total))
    }

    private static func match(_ text: String, _ pattern: String) -> [Int]? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = regex.firstMatch(in: text, range: range) else { return nil }
        var out: [Int] = []
        for i in 1..<m.numberOfRanges {
            guard let r = Range(m.range(at: i), in: text), let n = Int(text[r]) else { return nil }
            out.append(n)
        }
        return out
    }

    private static func matchString(_ text: String, _ pattern: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let m = regex.firstMatch(in: text, range: range) else { return nil }
        var out: [String] = []
        for i in 1..<m.numberOfRanges {
            guard let r = Range(m.range(at: i), in: text) else { return nil }
            out.append(String(text[r]))
        }
        return out
    }
}

enum SafetensorsHeader {
    static func list(in url: URL) throws -> [(String, Int)] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let sizeBytes = try handle.read(upToCount: 8), sizeBytes.count == 8 else {
            return []
        }
        let headerSize = sizeBytes.withUnsafeBytes { raw -> UInt64 in
            raw.load(as: UInt64.self).littleEndian
        }
        guard headerSize > 0, headerSize < 64 * 1024 * 1024 else { return [] }
        guard let header = try handle.read(upToCount: Int(headerSize)),
              let json = try JSONSerialization.jsonObject(with: header) as? [String: Any]
        else { return [] }
        var out: [(String, Int)] = []
        for (name, value) in json where name != "__metadata__" {
            guard let dict = value as? [String: Any],
                  let offsets = dict["data_offsets"] as? [Any],
                  offsets.count >= 2
            else {
                out.append((name, 0))
                continue
            }
            let start = (offsets[0] as? NSNumber)?.intValue ?? Int(offsets[0] as? Int ?? 0)
            let end = (offsets[1] as? NSNumber)?.intValue ?? Int(offsets[1] as? Int ?? 0)
            out.append((name, max(0, end - start)))
        }
        return out
    }
}

/// Host-side in-flight TIE conversion (progress for the model list).
public actor TieConversionBroker {
    public static let shared = TieConversionBroker()

    private var inFlight: [String: Task<Void, Error>] = [:]
    private var progress: [String: TieStatusInfo] = [:]

    public func snapshot(modelId: String) -> TieStatusInfo? {
        progress[modelId]
    }

    public func convert(modelId: String, at url: URL) async throws {
        if let existing = inFlight[modelId] {
            try await existing.value
            return
        }
        progress[modelId] = TieStatusInfo(
            status: .converting, message: "Starting TIE conversion…", fraction: 0, done: 0, total: 1
        )
        let task = Task.detached(priority: .userInitiated) {
            try ExpertShardConverter.convert(at: url) { p in
                Task {
                    await TieConversionBroker.shared.setProgress(
                        modelId: modelId,
                        TieStatusInfo(
                            status: .converting,
                            message: p.message,
                            fraction: p.fraction,
                            done: p.done,
                            total: p.total
                        )
                    )
                }
            }
        }
        inFlight[modelId] = task
        defer {
            inFlight[modelId] = nil
            progress[modelId] = nil
        }
        do {
            try await task.value
        } catch {
            progress[modelId] = TieStatusInfo(
                status: .incomplete,
                message: error.localizedDescription
            )
            throw error
        }
    }

    public func setProgress(modelId: String, _ info: TieStatusInfo) {
        progress[modelId] = info
    }
}
