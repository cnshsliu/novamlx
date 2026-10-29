import Testing
@testable import NovaMLXEngine

@Suite("MLX GPU executor thread")
struct MLXGPUExecutorTests {
    @Test("sync stays on one pthread and re-enters")
    func syncStaysOnOnePthread() {
        let executor = MLXGPUExecutor.shared
        let outer = executor.sync { MLXGPUExecutor.pthreadLabel() }
        let inner = executor.sync {
            executor.sync { MLXGPUExecutor.pthreadLabel() }
        }
        #expect(outer == inner)
        #expect(outer != MLXGPUExecutor.pthreadLabel())
    }
}
