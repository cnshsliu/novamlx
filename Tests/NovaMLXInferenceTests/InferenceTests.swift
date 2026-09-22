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

    @Test("Exclusive eviction never auto-unloads TTS/ASR/image")
    func exclusiveSkipsSideModels() {
        let keep: Set<String> = ["orcarouter/Qwen3.8-27B-Uncensored-MLX"]
        let side: Set<String> = [
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
            "mlx-community/Qwen3-ASR-1.7B-8bit",
        ]
        #expect(
            InferenceService.shouldExclusiveEvict(
                id: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
                keep: keep,
                sideLoaded: side
            ) == false
        )
        #expect(
            InferenceService.shouldExclusiveEvict(
                id: "some-other-chat-llm",
                keep: keep,
                sideLoaded: side
            ) == true
        )
        #expect(
            InferenceService.shouldExclusiveEvict(
                id: "orcarouter/Qwen3.8-27B-Uncensored-MLX",
                keep: keep,
                sideLoaded: side
            ) == false
        )
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

    @Test("Exclusive keep includes DSpark with the V4.1 backbone")
    func exclusiveKeepDSpark() {
        let keep = InferenceService.companionKeepIds(
            backboneId: "mlx-community/DeepSeek-V4.1-Flash-MLX-2bit",
            dflashId: nil,
            mtpId: nil,
            dsparkId: "mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"
        )
        #expect(keep.contains("mlx-community/DeepSeek-V4.1-Flash-MLX-2bit"))
        #expect(keep.contains("mlx-community/DeepSeek-V4.1-Flash-DSpark-drafter"))
    }
}
