import Foundation
import Testing
@testable import NovaMLXEngine

@Suite("SpecBoost")
struct SpecBoostTests {
    @Test("Native MTP is reported active even for hybrid backbones")
    func nativeMtpActiveOnHybrid() {
        let status = DraftModelRegistry.shared.boostStatus(
            family: .qwen,
            isHybrid: true,
            modelType: .llm,
            modelId: "mlx-community/Qwen3.8-27B-8bit",
            nativeMtp: true,
            draftModelLoaded: { _ in false },
            draftModelOnDisk: { _ in false }
        )
        guard case .active(let id) = status else {
            Issue.record("expected active native MTP, got \(status)")
            return
        }
        #expect(id == "mlx-community/Qwen3.8-27B-8bit")
    }

    @Test("Hybrid without MTP companion stays ineligible")
    func hybridWithoutMtpIneligible() {
        let status = DraftModelRegistry.shared.boostStatus(
            family: .qwen,
            isHybrid: true,
            modelType: .llm,
            modelId: "org/no-such-model",
            nativeMtp: false,
            draftModelLoaded: { _ in false },
            draftModelOnDisk: { _ in false }
        )
        guard case .ineligible = status else {
            Issue.record("expected ineligible, got \(status)")
            return
        }
    }

    @Test("Qwen dense still recommends a small draft")
    func qwenDraftEligible() {
        let status = DraftModelRegistry.shared.boostStatus(
            family: .qwen,
            isHybrid: false,
            modelType: .llm,
            modelId: "mlx-community/Qwen3-8B-4bit",
            nativeMtp: false,
            draftModelLoaded: { _ in false },
            draftModelOnDisk: { _ in false }
        )
        guard case .eligible(let candidate) = status else {
            Issue.record("expected eligible Qwen draft, got \(status)")
            return
        }
        #expect(candidate.draftModelId.contains("Qwen3-0.6B"))
    }
}
