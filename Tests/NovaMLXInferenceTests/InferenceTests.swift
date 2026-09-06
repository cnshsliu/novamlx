import Testing
import Foundation
@testable import NovaMLXInference

@Suite("InferenceService Tests")
struct InferenceServiceTests {
    @Test("Inference stats initial")
    func inferenceStatsInitial() {
        let stats = InferenceStats()
        #expect(stats.loadedModels == 0)
        #expect(stats.activeRequests == 0)
        #expect(stats.gpuMemoryUsed == 0)
    }

    @Test("Inference stats with values")
    func inferenceStatsWithValues() {
        let stats = InferenceStats(loadedModels: 2, activeRequests: 5, gpuMemoryUsed: 1024)
        #expect(stats.loadedModels == 2)
        #expect(stats.activeRequests == 5)
        #expect(stats.gpuMemoryUsed == 1024)
    }

    @Test("Exclusive keep set always includes the backbone id")
    func exclusiveKeepIncludesBackbone() {
        let keep = InferenceService.exclusiveKeepIds(for: "org/foo")
        #expect(keep.contains("org/foo"))
    }

    @Test("Exclusive keep for a Qwen3.8 backbone includes that id")
    func exclusiveKeepQwen38OptiQ() {
        let id = "mlx-community/Qwen3.8-27B-OptiQ-4bit"
        let keep = InferenceService.exclusiveKeepIds(for: id)
        #expect(keep.contains(id))
        #expect(!keep.contains("mlx-community/Qwen3.8-27B-8bit"))
    }

    @Test("MTP companion is not loaded when DFlash is available")
    func skipMtpWhenDFlashPresent() {
        #expect(InferenceService.shouldLoadMtpCompanion(hasDFlash: true) == false)
        #expect(InferenceService.shouldLoadMtpCompanion(hasDFlash: false) == true)
    }

    @Test("Exclusive keep prefers DFlash over MTP")
    func exclusiveKeepPrefersDFlash() {
        let keep = InferenceService.companionKeepIds(
            backboneId: "mlx-community/Qwen3.8-27B-8bit",
            dflashId: "incoai/Qwen3.8-27B-DFlash2",
            mtpId: "mlx-community/Qwen3.8-27B-MTP-4bit"
        )
        #expect(keep.contains("mlx-community/Qwen3.8-27B-8bit"))
        #expect(keep.contains("incoai/Qwen3.8-27B-DFlash2"))
        #expect(!keep.contains("mlx-community/Qwen3.8-27B-MTP-4bit"))
    }

    @Test("Exclusive keep includes MTP only when DFlash is absent")
    func exclusiveKeepMtpWithoutDFlash() {
        let keep = InferenceService.companionKeepIds(
            backboneId: "org/foo",
            dflashId: nil,
            mtpId: "org/foo-MTP-4bit"
        )
        #expect(keep == ["org/foo", "org/foo-MTP-4bit"])
    }
}
