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

    /// DFlash2 block-diffusion drafter for this backbone (preferred over MTP).
    public func dflashCandidate(forMainId mainId: String) -> DraftModelCandidate? {
        for id in dflashDraftCandidates(forMainId: mainId) {
            let dir = NovaMLXPaths.modelsDir.appendingPathComponent(id)
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            guard isDFlashDraftConfig(at: dir) else { continue }
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
    public func mtpCandidate(forMainId mainId: String) -> DraftModelCandidate? {
        let mainDir = NovaMLXPaths.modelsDir.appendingPathComponent(mainId)
        let vocab = Self.readVocabSize(from: mainDir)
        for id in mtpDraftCandidates(forMainId: mainId) {
            let dir = NovaMLXPaths.modelsDir.appendingPathComponent(id)
            let cfg = dir.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: cfg.path) else { continue }
            guard isMtpDraftConfig(at: dir) || checkpointHasMtpWeights(at: dir) else { continue }
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
        draftModelLoaded: (String) -> Bool,
        draftModelOnDisk: (String) -> Bool
    ) -> SpecBoostStatus {
        if nativeMtp {
            return .active(draftModelId: modelId ?? "native-mtp")
        }
        if let modelId, let dflash = dflashCandidate(forMainId: modelId) {
            if draftModelLoaded(dflash.draftModelId) {
                return .active(draftModelId: dflash.draftModelId)
            }
            return .eligible(candidate: dflash)
        }
        if let modelId, let mtp = mtpCandidate(forMainId: modelId) {
            if draftModelLoaded(mtp.draftModelId) {
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
}
