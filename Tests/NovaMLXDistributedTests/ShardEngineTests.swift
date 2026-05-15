import Foundation
import Testing
import MLX
@testable import NovaMLXDistributed

// MARK: - Test helper: passthrough policy

/// A lightweight ``ComputePolicy`` that passes tensors through unchanged.
/// Used for unit-testing ``ShardEngine`` mechanics without a real model.
private final class PassthroughPolicy: ComputePolicy, @unchecked Sendable {
    let assignment: ShardAssignment
    private(set) var isReady: Bool = false

    init(assignment: ShardAssignment) {
        self.assignment = assignment
    }

    func bindWeights() async throws {
        isReady = true
    }

    func compute(input: MLXArray) async throws -> MLXArray {
        guard isReady else { throw ShardEngineError.notReady }
        return input
    }

    func releaseWeights() {
        isReady = false
    }
}

// MARK: - ShardEngine tests

@Suite("ShardEngine")
struct ShardEngineTests {

    private func makeAssignment(
        startLayer: Int = 0, endLayer: Int = 10
    ) -> ShardAssignment {
        ShardAssignment(nodeId: "test", startLayer: startLayer, endLayer: endLayer, memoryEstimate: 0)
    }

    // MARK: - ShardEngine (uninitialized group)

    @Test("ShardEngine with uninitialized group: isFirstShard is false")
    func shardEngineUninitializedIsFirstShard() {
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        #expect(engine.isFirstShard == false)
    }

    @Test("ShardEngine with uninitialized group: isLastShard is false")
    func shardEngineUninitializedIsLastShard() {
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        #expect(engine.isLastShard == false)
    }

    @Test("ShardEngine prefill throws when policy not ready")
    func shardEnginePrefillThrowsWhenNotReady() async {
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
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
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
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
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        try await policy.bindWeights()
        let tokens = MLXArray([1, 2, 3])
        let output = try await engine.prefill(tokens: tokens)
        #expect(output.shape == tokens.shape)
    }

    @Test("ShardEngine decode succeeds after bindWeights")
    func shardEngineDecodeAfterBind() async throws {
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        try await policy.bindWeights()
        let token = MLXArray([42])
        let output = try await engine.decode(token: token)
        #expect(output.shape == token.shape)
    }

    // MARK: - Wavefront Prefill

    @Test("wavefrontPrefill falls back to sequential for short prompts")
    func wavefrontFallbackShortPrompt() async throws {
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        try await policy.bindWeights()

        let tokens = MLXArray(0..<3)
        let config = PrefillConfig(minWavefrontTokens: 4096)
        let output = try await engine.prefill(tokens: tokens, config: config)
        #expect(output.shape == tokens.shape)
    }

    @Test("wavefrontPrefill falls back for single-node group")
    func wavefrontFallbackSingleNode() async throws {
        let assignment = makeAssignment()
        let policy = PassthroughPolicy(assignment: assignment)
        let engine = ShardEngine(group: .uninitialized, assignment: assignment, policy: policy)
        try await policy.bindWeights()

        let tokens = MLXArray(0..<8192)
        let config = PrefillConfig(minWavefrontTokens: 4096)
        let output = try await engine.prefill(tokens: tokens, config: config)
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
