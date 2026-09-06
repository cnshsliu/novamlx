import Testing
import Foundation
import NovaMLXCore

@Suite("ServerConfig allowUnlistedDownloads")
struct ServerConfigAllowUnlistedTests {
    @Test("Defaults to false")
    func defaultOff() {
        #expect(ServerConfig().allowUnlistedDownloads == false)
    }

    @Test("Legacy JSON without the key decodes as false")
    func legacyJSON() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591 }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.allowUnlistedDownloads == false)
    }

    @Test("Decodes true")
    func decodesTrue() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591, "allowUnlistedDownloads": true }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.allowUnlistedDownloads == true)
    }
}

@Suite("ServerConfig maxGpuMemory")
struct ServerConfigMaxGpuMemoryTests {
    @Test("Defaults to auto")
    func defaultAuto() {
        #expect(ServerConfig().maxGpuMemory == "auto")
        #expect(ServerConfig().maxProcessMemory == "auto")
    }

    @Test("Legacy JSON without the key decodes as auto")
    func legacyJSON() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591 }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.maxGpuMemory == "auto")
    }

    @Test("Decodes a fixed GPU ceiling")
    func decodesFixed() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591, "maxGpuMemory": "48GB" }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.maxGpuMemory == "48GB")
    }

    @Test("Legacy performanceMode JSON still decodes")
    func ignoresLegacyPerformanceMode() throws {
        let json = Data(#"{ "host": "127.0.0.1", "port": 6590, "adminPort": 6591, "performanceMode": "exclusive" }"#.utf8)
        let cfg = try JSONDecoder().decode(ServerConfig.self, from: json)
        #expect(cfg.maxGpuMemory == "auto")
    }
}

@Suite("ResourceLimits")
struct ResourceLimitsTests {
    @Test("Auto-exclusive when zero or one chat model is loaded")
    func autoExclusiveCounts() {
        #expect(ResourceLimits.isAutoExclusive(chatModelCount: 0))
        #expect(ResourceLimits.isAutoExclusive(chatModelCount: 1))
        #expect(!ResourceLimits.isAutoExclusive(chatModelCount: 2))
    }

    @Test("MTP companion ids are not chat models")
    func mtpIds() {
        #expect(ResourceLimits.isMtpModelId("mlx-community/Qwen3.8-27B-MTP-8bit"))
        #expect(ResourceLimits.isMtpModelId("org/Foo-MTP"))
        #expect(!ResourceLimits.isMtpModelId("mlx-community/Qwen3.8-27B-8bit"))
    }

    @Test("DFlash companion ids are not chat models")
    func dflashIds() {
        #expect(ResourceLimits.isDFlashModelId("incoai/Qwen3.8-27B-DFlash2"))
        #expect(ResourceLimits.isDFlashModelId("org/Foo-dflash"))
        #expect(!ResourceLimits.isDFlashModelId("mlx-community/Qwen3.8-27B-8bit"))
        #expect(!ResourceLimits.isDFlashModelId("mlx-community/Qwen3.8-27B-MTP-8bit"))
    }

    @Test("Companion drafts cannot be opened in Playground")
    func companionDraftIds() {
        #expect(ResourceLimits.isCompanionDraftModelId("incoai/Qwen3.8-27B-DFlash2"))
        #expect(ResourceLimits.isCompanionDraftModelId("mlx-community/Qwen3.8-27B-MTP-8bit"))
        #expect(!ResourceLimits.isCompanionDraftModelId("mlx-community/Qwen3.8-27B-8bit"))
    }

    @Test("GPU is clamped to RAM")
    func clampGpuToRam() {
        let gpu = 50 * ResourceLimits.bytesPerGB
        let ram = 40 * ResourceLimits.bytesPerGB
        #expect(ResourceLimits.clampedGpuBytes(gpu: gpu, ram: ram) == ram)
    }

    @Test("Resolved percent and GB strings")
    func resolveStrings() {
        let phys: UInt64 = 64 * ResourceLimits.bytesPerGB
        let pct = ResourceLimits.resolvedBytes(raw: "50%", physicalRAM: phys)
        #expect(pct == 32 * ResourceLimits.bytesPerGB)
        let fixed = ResourceLimits.resolvedBytes(raw: "24GB", physicalRAM: phys)
        #expect(fixed == 24 * ResourceLimits.bytesPerGB)
    }

    @Test("Exclusive cache is larger than shared cache")
    func cacheExclusiveVsShared() {
        let gpu = 48 * ResourceLimits.bytesPerGB
        let exclusive = ResourceLimits.cacheLimitBytes(gpuLimit: gpu, autoExclusive: true)
        let shared = ResourceLimits.cacheLimitBytes(gpuLimit: gpu, autoExclusive: false)
        #expect(exclusive > shared)
    }

    @Test("Safety cap is well above the old 4-slot default")
    func safetyCap() {
        #expect(ResourceLimits.safetyConcurrentCap > 4)
    }

    @Test("Hybrid GDN prefill uses a single large chunk")
    func hybridPrefillStepSize() {
        #expect(ResourceLimits.prefillStepSize(hasLinearAttention: true, familyDefault: 512) == 32768)
        #expect(ResourceLimits.prefillStepSize(hasLinearAttention: false, familyDefault: 512) == 512)
        #expect(ResourceLimits.prefillStepSize(hasLinearAttention: true, familyDefault: 4096) == 32768)
        #expect(ResourceLimits.hybridQuantizedKVStart == 32768)
    }

    @Test("DFlash and linear-attention models serialize decode")
    func decodeConcurrencyCapSerializesUnsafePaths() {
        #expect(ResourceLimits.decodeConcurrencyCap(hasDraftModel: true, hasLinearAttention: false) == 1)
        #expect(ResourceLimits.decodeConcurrencyCap(hasDraftModel: false, hasLinearAttention: true) == 1)
        #expect(ResourceLimits.decodeConcurrencyCap(hasDraftModel: true, hasLinearAttention: true) == 1)
        #expect(ResourceLimits.decodeConcurrencyCap(hasDraftModel: false, hasLinearAttention: false) == ResourceLimits.safetyConcurrentCap)
    }
}
