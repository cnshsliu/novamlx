import Foundation
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

// MARK: - Draft Model Candidate

public struct DraftModelCandidate: Codable, Sendable {
    public let draftModelId: String
    public let displayName: String
    public let expectedVocabSize: Int
    public let family: ModelFamily
    public let downloadRepo: String
    public let estimatedSizeMB: Int
}

// MARK: - Spec Boost Status

public enum SpecBoostStatus: Sendable {
    case ineligible(reason: String)
    case eligible(candidate: DraftModelCandidate)
    case active(draftModelId: String)
}

// MARK: - DraftModelRegistry

public final class DraftModelRegistry: Sendable {

    public static let shared = DraftModelRegistry()

    private let candidates: [DraftModelCandidate]

    private init() {
        candidates = [
            // Qwen3 family — all share vocab_size=151936
            DraftModelCandidate(
                draftModelId: "mlx-community/Qwen3-0.6B-4bit",
                displayName: "Qwen3 0.6B",
                expectedVocabSize: 151936,
                family: .qwen,
                downloadRepo: "mlx-community/Qwen3-0.6B-4bit",
                estimatedSizeMB: 350
            ),
            // Llama 3.x family — all share vocab_size=128256
            DraftModelCandidate(
                draftModelId: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                displayName: "Llama 3.2 1B",
                expectedVocabSize: 128256,
                family: .llama,
                downloadRepo: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                estimatedSizeMB: 800
            ),
            // Gemma 2 family — all share vocab_size=256000
            DraftModelCandidate(
                draftModelId: "mlx-community/gemma-2-2b-it-4bit",
                displayName: "Gemma 2 2B",
                expectedVocabSize: 256000,
                family: .gemma,
                downloadRepo: "mlx-community/gemma-2-2b-it-4bit",
                estimatedSizeMB: 1500
            ),
        ]
    }

    /// Get the recommended draft model for a given model family.
    /// Returns nil for hybrid models (MambaCache), unknown families, or non-LLM types.
    /// MTP pairing is ``mtpCandidate(forMainId:)`` and is allowed on hybrid backbones.
    public func recommendation(family: ModelFamily, isHybrid: Bool) -> DraftModelCandidate? {
        if isHybrid { return nil }
        // DeepSeek V4 uses MLA + MoE with no compatible small draft model on macOS.
        // Qwen3-0.6B shares vocab but verification hit-rate is too low to be useful.
        if family == .deepseek { return nil }
        return candidates.first { $0.family == family }
    }

    /// DeepSeek DSpark drafter (V4.1 Flash). Preferred over a second LM draft.
    public func dsparkCandidate(forMainId mainId: String) -> DraftModelCandidate? {
        for id in NovaMLXEngine.dsparkDraftCandidates(forMainId: mainId) {
            let dir = NovaMLXPaths.directory(forModelId: id)
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            guard NovaMLXEngine.isDSparkDraftConfig(at: dir) || NovaMLXEngine.isMtpDraftConfig(at: dir) else { continue }
            return DraftModelCandidate(
                draftModelId: id,
                displayName: id.split(separator: "/").last.map(String.init) ?? id,
                expectedVocabSize: 129280,
                family: .deepseek,
                downloadRepo: id,
                estimatedSizeMB: 4500
            )
        }
        return nil
    }

    /// DSpark drafter ids for a backbone — no disk check, so this stays testable.
    /// They do not enter `dsparkCandidate(forMainId:)` (that stays the DeepSeek
    /// on-disk head). `AcceleratorSpeedGate` decides auto-load versus ask.
    public func dsparkOptInCandidateIds(forMainId mainId: String) -> [String] {
        let name = (mainId.split(separator: "/").last.map(String.init) ?? mainId).lowercased()
        if name.contains("dspark") || name.contains("dflash") || name.contains("-mtp") {
            return []
        }
        if name.contains("qwen3.8-27b") || name.contains("qwen3_8-27b") {
            return ["RedHatAI/Qwen3.8-27B-speculator.dspark"]
        }
        // DeepSpec-native heads for the dense Qwen3 line (block-7, bidirectional).
        if name.contains("qwen3-14b") || name.contains("qwen3_14b") {
            return ["deepseek-ai/dspark_qwen3_14b_block7"]
        }
        if name.contains("qwen3-8b") || name.contains("qwen3_8b") {
            return ["deepseek-ai/dspark_qwen3_8b_block7"]
        }
        return []
    }

    /// On-disk DSpark drafter for this backbone. The speed gate, not this lookup,
    /// decides whether load attaches it.
    public func dsparkOptInCandidate(forMainId mainId: String) -> DraftModelCandidate? {
        for id in dsparkOptInCandidateIds(forMainId: mainId) {
            let dir = NovaMLXPaths.directory(forModelId: id)
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            guard DSparkConfig.isDSparkDraftConfig(at: dir) else { continue }
            let sizeMB =
                id.contains("speculator.dspark") ? 3800
                : id.contains("14b") ? 6300
                : 4600
            return DraftModelCandidate(
                draftModelId: id,
                displayName: id.split(separator: "/").last.map(String.init) ?? id,
                expectedVocabSize: 0,
                family: .qwen,
                downloadRepo: id,
                estimatedSizeMB: sizeMB
            )
        }
        return nil
    }

    /// DFlash2 block-diffusion drafter for this backbone (preferred over MTP).
    public func dflashCandidate(forMainId mainId: String) -> DraftModelCandidate? {
        for id in NovaMLXEngine.dflashDraftCandidates(forMainId: mainId) {
            let dir = NovaMLXPaths.directory(forModelId:id)
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            guard NovaMLXEngine.isDFlashDraftConfig(at: dir) else { continue }
            return DraftModelCandidate(
                draftModelId: id,
                displayName: id.split(separator: "/").last.map(String.init) ?? id,
                expectedVocabSize: 0,
                family: .qwen,
                downloadRepo: id,
                estimatedSizeMB: 1500
            )
        }
        return nil
    }

    /// Native MTP head on disk for this backbone (same vocab, `qwen3_5_mtp` or `mtp.*` weights).
    ///
    /// Candidates come from two sources: the hard-coded known companions
    /// (`mtpDraftCandidates`) and a generic sibling guess — `<org>/<Name>-<quant>`
    /// pairs with `<org>/<Name>-MTP-{4,8}bit`. The guess is safe here (unlike
    /// in `acceleratorCandidates`, which drives download suggestions) because
    /// every candidate is validated against its on-disk config and vocab below.
    public func mtpCandidate(forMainId mainId: String) -> DraftModelCandidate? {
        let mainDir = NovaMLXPaths.directory(forModelId:mainId)
        let vocab = Self.readVocabSize(from: mainDir)
        var ids = NovaMLXEngine.mtpDraftCandidates(forMainId: mainId)
        if let siblingBase = Self.siblingMtpBase(forMainId: mainId) {
            for quant in ["4bit", "8bit"] {
                let id = "\(siblingBase)-MTP-\(quant)"
                if !ids.contains(id) { ids.append(id) }
            }
        }
        for id in ids {
            let dir = NovaMLXPaths.directory(forModelId:id)
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            guard NovaMLXEngine.isMtpDraftConfig(at: dir) || NovaMLXEngine.checkpointHasMtpWeights(at: dir) else { continue }
            if let vocab, let draftVocab = Self.readVocabSize(from: dir), draftVocab != vocab {
                continue
            }
            return DraftModelCandidate(
                draftModelId: id,
                displayName: id.split(separator: "/").last.map(String.init) ?? id,
                expectedVocabSize: vocab ?? 0,
                family: .qwen,
                downloadRepo: id,
                estimatedSizeMB: 256
            )
        }
        return nil
    }

    /// Check boost status for a model, given the engine pool state.
    public func boostStatus(
        family: ModelFamily,
        isHybrid: Bool,
        modelType: ModelType,
        modelId: String? = nil,
        nativeMtp: Bool = false,
        mtpEnabled: Bool = true,
        draftModelLoaded: (String) -> Bool,
        draftModelOnDisk: (String) -> Bool
    ) -> SpecBoostStatus {
        if nativeMtp && mtpEnabled {
            return .active(draftModelId: modelId ?? "native-mtp")
        }
        if let modelId, let dspark = dsparkCandidate(forMainId: modelId) {
            if draftModelLoaded(dspark.draftModelId) {
                return .active(draftModelId: dspark.draftModelId)
            }
            return .eligible(candidate: dspark)
        }
        if let modelId, let dflash = dflashCandidate(forMainId: modelId) {
            if mtpEnabled, draftModelLoaded(dflash.draftModelId) {
                return .active(draftModelId: dflash.draftModelId)
            }
            return .eligible(candidate: dflash)
        }
        if let modelId, let mtp = mtpCandidate(forMainId: modelId) {
            if mtpEnabled, draftModelLoaded(mtp.draftModelId) {
                return .active(draftModelId: mtp.draftModelId)
            }
            return .eligible(candidate: mtp)
        }
        if isHybrid {
            return .ineligible(reason: "Hybrid model (MambaCache)")
        }
        guard modelType == .llm else {
            return .ineligible(reason: "Not a text model")
        }
        guard let candidate = recommendation(family: family, isHybrid: false) else {
            return .ineligible(reason: "No compatible draft model known")
        }
        if draftModelLoaded(candidate.draftModelId) {
            return .active(draftModelId: candidate.draftModelId)
        }
        return .eligible(candidate: candidate)
    }

    /// Read vocab_size from a model's config.json on disk.
    public static func readVocabSize(from modelDir: URL) -> Int? {
        let cfgURL = modelDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfgURL) else { return nil }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        // VLM models store vocab_size in text_config
        if let tc = json["text_config"] as? [String: Any], let vs = tc["vocab_size"] as? Int {
            return vs
        }
        return json["vocab_size"] as? Int
    }

    /// Every companion the speed gate may attach. No disk check.
    /// Order here is not the run order; `AcceleratorPreference` sorts that.
    public func acceleratorCandidates(
        forMainId mainId: String,
        family: ModelFamily,
        isHybrid: Bool,
        modelType: ModelType
    ) -> [AcceleratorCandidate] {
        guard modelType == .llm || modelType == .vlm else { return [] }
        var seen = Set<String>()
        var result: [AcceleratorCandidate] = []
        func add(_ candidate: AcceleratorCandidate) {
            guard seen.insert(candidate.acceleratorId).inserted else { return }
            result.append(candidate)
        }
        for id in NovaMLXEngine.dflashDraftCandidates(forMainId: mainId) {
            add(AcceleratorCandidate(
                kind: .dflash2, acceleratorId: id, displayName: Self.leaf(id),
                estimatedSizeMB: 1500, family: .qwen
            ))
        }
        for id in NovaMLXEngine.mtpDraftCandidates(forMainId: mainId) {
            add(AcceleratorCandidate(
                kind: .mtp, acceleratorId: id, displayName: Self.leaf(id),
                estimatedSizeMB: 256, family: .qwen
            ))
        }
        for id in NovaMLXEngine.dsparkDraftCandidates(forMainId: mainId) {
            add(AcceleratorCandidate(
                kind: .dspark, acceleratorId: id, displayName: Self.leaf(id),
                estimatedSizeMB: 4500, family: .deepseek
            ))
        }
        for id in dsparkOptInCandidateIds(forMainId: mainId) {
            let sizeMB =
                id.contains("speculator.dspark") ? 3800
                : id.contains("14b") ? 6300
                : 4600
            add(AcceleratorCandidate(
                kind: .dspark, acceleratorId: id, displayName: Self.leaf(id),
                estimatedSizeMB: sizeMB, family: .qwen
            ))
        }
        if let small = recommendation(family: family, isHybrid: isHybrid) {
            add(AcceleratorCandidate(
                kind: .smallDraft,
                acceleratorId: small.draftModelId,
                displayName: small.displayName,
                estimatedSizeMB: small.estimatedSizeMB,
                family: small.family
            ))
        }
        return result
    }

    private static func leaf(_ id: String) -> String {
        id.split(separator: "/").last.map(String.init) ?? id
    }

    /// `<org>/<Name>` for a `<org>/<Name>-<quant>` backbone, so `mtpCandidate`
    /// can guess sibling companion ids (`<org>/<Name>-MTP-<quant>`). Returns
    /// nil for ids without an org prefix or without a recognisable quant
    /// suffix (`-8bit`, `-4bit`, `-bf16`, …) — no suffix, no naming convention
    /// to guess from.
    private static func siblingMtpBase(forMainId mainId: String) -> String? {
        let parts = mainId.split(separator: "/")
        guard parts.count == 2 else { return nil }
        let leaf = String(parts[1])
        let lower = leaf.lowercased()
        guard !lower.contains("-mtp"), !lower.contains("dflash"), !lower.contains("dspark") else {
            return nil
        }
        guard let range = leaf.range(
            of: #"-(?:\d+bit|bf16|fp16|fp8)$"#,
            options: [.regularExpression, .caseInsensitive]
        ) else { return nil }
        return "\(parts[0])/\(leaf[..<range.lowerBound])"
    }
}
