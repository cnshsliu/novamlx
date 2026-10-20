import Foundation
import Testing
import NovaMLXCore

@Suite("MOSS-Image support")
struct MossImageSupportTests {
    @Test("Official and mirror ids match")
    func matchesIds() {
        #expect(MossImageSupport.matches(id: "OpenMOSS-Team/MOSS-Image-SFT"))
        #expect(MossImageSupport.matches(id: "OpenMOSS-Team/MOSS-Image-Base"))
        #expect(MossImageSupport.matches(id: "mlx-community/MOSS-Image-SFT-bf16"))
        #expect(MossImageSupport.matches(id: "org/MOSS_Image_SFT"))
        #expect(MossImageSupport.matches(id: "lab/mossimage-sft"))
    }

    @Test("Other image models stay off the MOSS path")
    func skipsOtherImages() {
        #expect(!MossImageSupport.matches(id: "Qwen/Qwen-Image"))
        #expect(!MossImageSupport.matches(id: "Qwen/Qwen-Image-2.1"))
        #expect(!MossImageSupport.matches(id: "black-forest-labs/FLUX.2-klein-4B"))
        #expect(!MossImageSupport.matches(id: "mzbac/flux1.dev.4bit.mlx"))
    }

    @Test("Quant tier: environment, then id, then memory")
    func quantTier() {
        let large = MossImageSupport.bf16MemoryFloorBytes
        let small = large - 1
        let sft = MossImageSupport.sftId

        #expect(MossImageSupport.quantTier(id: sft, physicalBytes: large, environment: [:]) == .bf16)
        #expect(MossImageSupport.quantTier(id: sft, physicalBytes: small, environment: [:]) == .int8)
        #expect(MossImageSupport.quantTier(id: sft, physicalBytes: 0, environment: [:]) == .bf16)
        #expect(
            MossImageSupport.quantTier(
                id: sft, physicalBytes: large, environment: ["NOVA_MOSS_IMAGE_QUANT": "int8"]
            ) == .int8
        )
        #expect(
            MossImageSupport.quantTier(
                id: sft, physicalBytes: small, environment: ["NOVA_MOSS_IMAGE_QUANT": "bf16"]
            ) == .bf16
        )
        #expect(
            MossImageSupport.quantTier(
                id: "org/MOSS-Image-SFT-int8", physicalBytes: large, environment: [:]
            ) == .int8
        )
        #expect(
            MossImageSupport.quantTier(
                id: "org/MOSS-Image-SFT-8bit",
                physicalBytes: small,
                environment: ["NOVA_MOSS_IMAGE_QUANT": "bf16"]
            ) == .bf16
        )
        #expect(
            MossImageSupport.quantTier(
                id: sft, physicalBytes: small, environment: ["NOVA_MOSS_IMAGE_QUANT": "nope"]
            ) == .int8
        )
    }

    @Test("Output size floors to multiples of 16 and rejects the envelope")
    func outputSize() throws {
        let square = try #require(MossImageSupport.outputSize(width: 1024, height: 1024))
        #expect(square.width == 1024)
        #expect(square.height == 1024)
        let widescreen = try #require(MossImageSupport.outputSize(width: 1920, height: 1080))
        #expect(widescreen.width == 1920)
        #expect(widescreen.height == 1072)
        let small = try #require(MossImageSupport.outputSize(width: 256, height: 256))
        #expect(small.width == 256)
        #expect(small.height == 256)
        #expect(MossImageSupport.outputSize(width: 200, height: 1024) == nil)
        #expect(MossImageSupport.outputSize(width: 2048, height: 2048) == nil)
        #expect(MossImageSupport.outputSize(width: 0, height: 1024) == nil)
    }
}
