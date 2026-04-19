import Testing
import Foundation
import NovaMLXCore
import MLXLMCommon
@testable import NovaMLXEngine

@Suite("MLXEngine Tests")
struct MLXEngineTests {
    @Test("Engine initial state")
    func engineInitialState() {
        let engine = MLXEngine()
        #expect(engine.isReady == false)
    }

    @Test("Engine with custom config")
    func engineCustomConfig() {
        let engine = MLXEngine(kvCacheCapacity: 512, maxMemoryMB: 1024, maxConcurrent: 4)
        #expect(engine.isReady == false)
    }

    @Test("Get container for unloaded model returns nil")
    func getContainerNil() {
        let engine = MLXEngine()
        let container = engine.getContainer(for: "nonexistent")
        #expect(container == nil)
    }

    @Test("Model container initial state")
    func modelContainerInitialState() {
        let id = ModelIdentifier(id: "test", family: .llama)
        let config = ModelConfig(identifier: id)
        let container = ModelContainer(identifier: id, config: config)
        #expect(container.isLoaded == false)
        #expect(container.identifier.id == "test")
        #expect(container.kvMemoryOverride == nil)
    }

    @Test("Model container kvMemoryOverride can be set")
    func modelContainerKvOverride() {
        let id = ModelIdentifier(id: "test", family: .llama)
        let config = ModelConfig(identifier: id)
        let container = ModelContainer(identifier: id, config: config)
        container.kvMemoryOverride = 131072
        #expect(container.kvMemoryOverride == 131072)
    }

    @Test("ModelConfig kvMemoryBytesPerToken defaults to nil")
    func modelConfigKvMemoryDefault() {
        let id = ModelIdentifier(id: "test", family: .llama)
        let config = ModelConfig(identifier: id)
        #expect(config.kvMemoryBytesPerToken == nil)
    }

    @Test("ModelConfig kvMemoryBytesPerToken can be set")
    func modelConfigKvMemorySet() {
        let id = ModelIdentifier(id: "test", family: .llama)
        let config = ModelConfig(identifier: id, kvMemoryBytesPerToken: 65536)
        #expect(config.kvMemoryBytesPerToken == 65536)
    }

    @Test("Preflight check rejects when model not loaded (no-op)")
    func preflightNoLoadedModel() async {
        let engine = MLXEngine()
        do {
            try await engine.preflightCheck(modelId: "nonexistent", promptTokens: 100, maxTokens: 100)
        } catch {
            // Expected: no crash, model not found = no-op
        }
    }

    @Test("Preflight check uses container config without crash")
    func preflightWithContainer() {
        let engine = MLXEngine()
        let id = ModelIdentifier(id: "test-model", family: .llama)
        let config = ModelConfig(identifier: id, kvMemoryBytesPerToken: 65536)
        let container = ModelContainer(identifier: id, config: config)
        engine.pool.add(container)
        #expect(engine.getContainer(for: "test-model") != nil)
        #expect(engine.getContainer(for: "test-model")?.config.kvMemoryBytesPerToken == 65536)
    }

    @Test("BatchScheduler metrics initial state")
    func batchSchedulerMetricsInitial() {
        let engine = MLXEngine()
        let metrics = engine.batchScheduler.metrics
        #expect(metrics.pendingCount == 0)
        #expect(metrics.activeSequences == 0)
        #expect(metrics.totalFusedSteps == 0)
        #expect(metrics.totalTokensViaFused == 0)
        #expect(metrics.peakBatchWidth == 0)
        #expect(metrics.totalBatches == 0)
    }
}

@Suite("TurboQuantService Tests")
struct TurboQuantServiceTests {
    @Test("TurboQuant service initial state")
    func turboQuantInitialState() {
        let service = TurboQuantService()
        #expect(service.allConfigs().isEmpty)
        #expect(service.getConfig(forModel: "test") == nil)
        #expect(service.getStats(modelId: "test") == nil)
    }

    @Test("TurboQuant set and get config")
    func turboQuantSetGetConfig() {
        let service = TurboQuantService()
        let config = TurboQuantService.Config(bits: 4, groupSize: 64)
        service.setConfig(config, forModel: "test-model")
        let retrieved = service.getConfig(forModel: "test-model")
        #expect(retrieved != nil)
        #expect(retrieved!.bits == 4)
        #expect(retrieved!.groupSize == 64)
    }

    @Test("TurboQuant remove config")
    func turboQuantRemoveConfig() {
        let service = TurboQuantService()
        let config = TurboQuantService.Config(bits: 2, groupSize: 32)
        service.setConfig(config, forModel: "test-model")
        service.removeConfig(forModel: "test-model")
        #expect(service.getConfig(forModel: "test-model") == nil)
    }

    @Test("TurboQuant compression ratio")
    func turboQuantCompressionRatio() {
        #expect(TurboQuantService.Config(bits: 2, groupSize: 64).estimatedCompressionRatio == 8.0)
        #expect(TurboQuantService.Config(bits: 4, groupSize: 64).estimatedCompressionRatio == 4.0)
        #expect(TurboQuantService.Config(bits: 8, groupSize: 64).estimatedCompressionRatio == 2.0)
    }

    @Test("TurboQuant recommended bits")
    func turboQuantRecommendedBits() {
        #expect(TurboQuantService.recommendedBits(for: 7.0, contextLength: 4096, availableMemoryGB: 8.0) == 2)
        #expect(TurboQuantService.recommendedBits(for: 2.0, contextLength: 4096, availableMemoryGB: 8.0) == 4)
        #expect(TurboQuantService.recommendedBits(for: 5.0, contextLength: 4096, availableMemoryGB: 8.0) == 3)
    }

    @Test("TurboQuant auto-configure")
    func turboQuantAutoConfigure() {
        let service = TurboQuantService()
        let config = service.autoConfigure(modelId: "test", modelSizeGB: 3.0, contextLength: 8192, availableMemoryGB: 8.0)
        #expect(config.bits > 0)
        let retrieved = service.getConfig(forModel: "test")
        #expect(retrieved != nil)
        #expect(retrieved!.bits == config.bits)
    }

    @Test("TurboQuant stats")
    func turboQuantStats() {
        let service = TurboQuantService()
        let config = TurboQuantService.Config(bits: 4, groupSize: 64)
        service.setConfig(config, forModel: "test-model")
        let stats = service.getStats(modelId: "test-model")
        #expect(stats != nil)
        #expect(stats!.modelId == "test-model")
        #expect(stats!.bits == 4)
        #expect(stats!.groupSize == 64)
        #expect(stats!.compressionRatio == 4.0)
    }

    @Test("TurboQuant applyToGenerateParameters with existing config")
    func turboQuantApplyExisting() {
        let service = TurboQuantService()
        let config = TurboQuantService.Config(bits: 2, groupSize: 32)
        service.setConfig(config, forModel: "test-model")
        var params = GenerateParameters()
        let settings = ModelSettings()
        service.applyToGenerateParameters(&params, modelId: "test-model", settings: settings)
        #expect(params.kvBits == 2)
        #expect(params.kvGroupSize == 32)
    }

    @Test("TurboQuant applyToGenerateParameters from settings")
    func turboQuantApplyFromSettings() {
        let service = TurboQuantService()
        var params = GenerateParameters()
        let settings = ModelSettings(kvBits: 4, kvGroupSize: 128)
        service.applyToGenerateParameters(&params, modelId: "test-model", settings: settings)
        #expect(params.kvBits == 4)
        #expect(params.kvGroupSize == 128)
    }

    @Test("TurboQuant applyToGenerateParameters no-op when no config or settings bits")
    func turboQuantApplyNoOp() {
        let service = TurboQuantService()
        var params = GenerateParameters()
        let settings = ModelSettings()
        service.applyToGenerateParameters(&params, modelId: "test-model", settings: settings)
        #expect(params.kvBits == nil)
    }

    @Test("TurboQuant settings provider wired in engine")
    func turboQuantSettingsProvider() {
        let engine = MLXEngine()
        engine.settingsProvider = { modelId in
            ModelSettings(kvBits: 3, kvGroupSize: 64)
        }
        let id = ModelIdentifier(id: "test-model", family: .llama)
        let config = ModelConfig(identifier: id)
        let container = ModelContainer(identifier: id, config: config)
        engine.pool.add(container)
        let request = InferenceRequest(model: "test-model", messages: [ChatMessage(role: .user, content: "hello")])
        let params = engine.buildGenerateParameters(request: request, config: config, settings: engine.settingsProvider?("test-model"))
        #expect(params.kvBits == 3)
        #expect(params.kvGroupSize == 64)
    }
}

@Suite("ModelFamilyRegistry Tests")
struct ModelFamilyRegistryTests {
    @Test("Registry returns default for unknown family")
    func registryDefaultForUnknown() {
        let registry = ModelFamilyRegistry()
        let opt = registry.optimization(for: "test", family: .other)
        #expect(opt.prefillStepSize == 512)
        #expect(opt.defaultKVBits == nil)
    }

    @Test("Registry returns family-specific defaults for llama")
    func registryLlamaDefaults() {
        let registry = ModelFamilyRegistry()
        let opt = registry.optimization(for: "test", family: .llama)
        #expect(opt.defaultKVBits == 4)
        #expect(opt.prefillStepSize == 512)
        #expect(opt.recommendedContextLength == 8192)
    }

    @Test("Registry returns family-specific defaults for phi")
    func registryPhiDefaults() {
        let registry = ModelFamilyRegistry()
        let opt = registry.optimization(for: "test", family: .phi)
        #expect(opt.prefillStepSize == 256)
        #expect(opt.repeatLastN == 32)
    }

    @Test("Registry architecture mapping")
    func registryArchMapping() {
        let registry = ModelFamilyRegistry()
        let opt = registry.optimization(for: "test", family: .other, architectures: ["LlamaForCausalLM"])
        #expect(opt.defaultKVBits == 4)
    }

    @Test("Registry override takes precedence")
    func registryOverride() {
        let registry = ModelFamilyRegistry()
        let custom = ModelFamilyOptimization(defaultKVBits: 2, prefillStepSize: 128)
        registry.setOverride(custom, forModel: "my-model")
        let opt = registry.optimization(for: "my-model", family: .llama)
        #expect(opt.defaultKVBits == 2)
        #expect(opt.prefillStepSize == 128)
        registry.removeOverride(forModel: "my-model")
        let after = registry.optimization(for: "my-model", family: .llama)
        #expect(after.defaultKVBits == 4)
    }

    @Test("Registry all overrides")
    func registryAllOverrides() {
        let registry = ModelFamilyRegistry()
        registry.setOverride(ModelFamilyOptimization(defaultKVBits: 3), forModel: "m1")
        registry.setOverride(ModelFamilyOptimization(defaultKVBits: 2), forModel: "m2")
        let all = registry.allOverrides()
        #expect(all.count == 2)
        #expect(all["m1"]?.defaultKVBits == 3)
        registry.removeOverride(forModel: "m1")
        registry.removeOverride(forModel: "m2")
    }
}

@Suite("ContinuousBatcher Tests")
struct ContinuousBatcherTests {
    @Test("Batcher initial state")
    func batcherInitialState() {
        let engine = MLXEngine()
        let batcher = ContinuousBatcher(engine: engine, maxBatchSize: 4)
        #expect(batcher.activeRequests == 0)
    }
}

@Suite("ChatSessionManager Tests")
struct ChatSessionManagerTests {
    @Test("Session manager initial state")
    func sessionManagerInitialState() {
        let manager = ChatSessionManager()
        #expect(manager.sessionCount == 0)
        #expect(manager.maxSessions == 64)
    }

    @Test("Session manager custom config")
    func sessionManagerCustomConfig() {
        let manager = ChatSessionManager(maxSessions: 32, sessionTTL: 600)
        #expect(manager.maxSessions == 32)
        #expect(manager.sessionTTL == 600)
    }

    @Test("Session manager list empty")
    func sessionManagerListEmpty() {
        let manager = ChatSessionManager()
        let sessions = manager.listSessions()
        #expect(sessions.isEmpty)
    }

    @Test("Session manager remove nonexistent")
    func sessionManagerRemoveNonexistent() {
        let manager = ChatSessionManager()
        manager.remove("nonexistent")
        #expect(manager.sessionCount == 0)
    }

    @Test("Session info is codable")
    func sessionInfoCodable() throws {
        let info = SessionInfo(
            sessionId: "test-session",
            modelId: "test-model",
            createdAt: Date(),
            lastAccessed: Date(),
            messageCount: 5
        )
        let data = try JSONEncoder().encode(info)
        let decoded = try JSONDecoder().decode(SessionInfo.self, from: data)
        #expect(decoded.sessionId == "test-session")
        #expect(decoded.modelId == "test-model")
        #expect(decoded.messageCount == 5)
    }
}
