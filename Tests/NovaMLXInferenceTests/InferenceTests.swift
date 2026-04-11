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
}
