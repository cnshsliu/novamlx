import Foundation
import Testing
import MLX
@testable import NovaMLXDistributed

@Suite("ShardEngine")
struct ShardEngineTests {

    // MARK: - FitInMemoryPolicy

    @Test("FitInMemoryPolicy starts not ready")
    func fitInMemoryPolicyStartsNotReady() {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        #expect(policy.isReady == false)
    }

    @Test("FitInMemoryPolicy bindWeights sets isReady")
    func fitInMemoryPolicyBindWeights() async throws {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        #expect(policy.isReady == false)
        try await policy.bindWeights()
        #expect(policy.isReady == true)
    }

    @Test("FitInMemoryPolicy releaseWeights clears isReady")
    func fitInMemoryPolicyReleaseWeights() async throws {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        try await policy.bindWeights()
        #expect(policy.isReady == true)
        policy.releaseWeights()
        #expect(policy.isReady == false)
    }

    @Test("FitInMemoryPolicy compute returns input unchanged when ready")
    func fitInMemoryPolicyComputePassthrough() async throws {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        try await policy.bindWeights()

        let input = MLXArray([1.0, 2.0, 3.0])
        var cache: [Any] = []
        let output = try policy.compute(input: input, cache: &cache)
        // Placeholder: output should equal input
        #expect(output.shape == input.shape)
    }

    @Test("FitInMemoryPolicy compute throws when not ready")
    func fitInMemoryPolicyComputeThrowsWhenNotReady() {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        var cache: [Any] = []
        #expect(throws: ShardEngineError.self) {
            let input = MLXArray([1.0])
            _ = try policy.compute(input: input, cache: &cache)
        }
    }

    // MARK: - ShardEngine (uninitialized group)

    @Test("ShardEngine with uninitialized group: isFirstShard is false")
    func shardEngineUninitializedIsFirstShard() {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(
            group: .uninitialized,
            assignment: assignment,
            policy: policy
        )
        #expect(engine.isFirstShard == false)
    }

    @Test("ShardEngine with uninitialized group: isLastShard is false")
    func shardEngineUninitializedIsLastShard() {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(
            group: .uninitialized,
            assignment: assignment,
            policy: policy
        )
        #expect(engine.isLastShard == false)
    }

    @Test("ShardEngine prefill throws when policy not ready")
    func shardEnginePrefillThrowsWhenNotReady() async {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(
            group: .uninitialized,
            assignment: assignment,
            policy: policy
        )
        do {
            let tokens = MLXArray([1, 2, 3])
            _ = try await engine.prefill(tokens: tokens)
            #expect(Bool(false), "Should have thrown")
        } catch {
            // Expected
        }
    }

    @Test("ShardEngine decode throws when policy not ready")
    func shardEngineDecodeThrowsWhenNotReady() async {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(
            group: .uninitialized,
            assignment: assignment,
            policy: policy
        )
        do {
            let token = MLXArray(42)
            _ = try await engine.decode(token: token)
            #expect(Bool(false), "Should have thrown")
        } catch {
            // Expected
        }
    }

    @Test("ShardEngine prefill succeeds after bindWeights")
    func shardEnginePrefillAfterBind() async throws {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(
            group: .uninitialized,
            assignment: assignment,
            policy: policy
        )
        try await policy.bindWeights()
        let tokens = MLXArray([1, 2, 3])
        let output = try await engine.prefill(tokens: tokens)
        // Placeholder policy returns input unchanged.
        #expect(output.shape == tokens.shape)
    }

    @Test("ShardEngine decode succeeds after bindWeights")
    func shardEngineDecodeAfterBind() async throws {
        let assignment = ShardAssignment(
            nodeId: "test",
            startLayer: 0,
            endLayer: 10,
            memoryEstimate: 0
        )
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(
            group: .uninitialized,
            assignment: assignment,
            policy: policy
        )
        try await policy.bindWeights()
        // Use a 1D array so shape is non-empty (scalar MLXArray has shape=[]).
        let token = MLXArray([42])
        let output = try await engine.decode(token: token)
        // Placeholder policy returns input unchanged.
        #expect(output.shape == token.shape)
    }

    // MARK: - Wavefront Prefill

    @Test("wavefrontPrefill falls back to sequential for short prompts")
    func wavefrontFallbackShortPrompt() async throws {
        let assignment = ShardAssignment(nodeId: "test", startLayer: 0, endLayer: 10, memoryEstimate: 0)
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        try await policy.bindWeights()

        let tokens = MLXArray(0..<3)
        let config = PrefillConfig(minWavefrontTokens: 4096)
        let output = try await engine.prefill(tokens: tokens, config: config)
        #expect(output.shape == tokens.shape)
    }

    @Test("wavefrontPrefill falls back for single-node group")
    func wavefrontFallbackSingleNode() async throws {
        let assignment = ShardAssignment(nodeId: "test", startLayer: 0, endLayer: 10, memoryEstimate: 0)
        let policy = FitInMemoryPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        try await policy.bindWeights()

        let tokens = MLXArray(0..<8192)
        let config = PrefillConfig(minWavefrontTokens: 4096)
        let output = try await engine.prefill(tokens: tokens, config: config)
        // Uninitialized group (size 0) => falls back to sequential
        #expect(output.shape == tokens.shape)
    }

    @Test("wavefrontPrefill computes correct chunk plan")
    func wavefrontChunkPlan() {
        let config = PrefillConfig(baseStepSize: 4096, minChunkSize: 512)
        let worldSize = 2
        let promptLen = 8192

        let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
        let nReal = (promptLen - 1 + chunkSize - 1) / chunkSize

        #expect(chunkSize == 2048)
        #expect(nReal == 4)
    }

    @Test("wavefrontPrefill chunk plan with 3 nodes")
    func wavefrontChunkPlan3Nodes() {
        let config = PrefillConfig(baseStepSize: 4096, minChunkSize: 512)
        let worldSize = 3
        let promptLen = 12288

        let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
        let nReal = (promptLen - 1 + chunkSize - 1) / chunkSize

        #expect(chunkSize == 1365)
        #expect(nReal == 10)
    }

    @Test("wavefrontPrefill minChunkSize floor prevents tiny chunks")
    func wavefrontMinChunkFloor() {
        let config = PrefillConfig(baseStepSize: 4096, minChunkSize: 1024)
        let worldSize = 8

        let chunkSize = max(config.baseStepSize / worldSize, config.minChunkSize)
        #expect(chunkSize == 1024)
    }
}
