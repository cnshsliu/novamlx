import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Tracks GPU memory budget across all active inference sequences.
/// Provides memory-aware admission control: canAdmit() checks whether
/// a new sequence's estimated KV cache fits within the remaining budget.
public actor MemoryBudgetTracker {

    // MARK: - Types

    public struct Metrics: Sendable {
        public let gpuLimitBytes: UInt64
        public let weightsMemoryBytes: UInt64
        public let committedKVBytes: UInt64
        public let availableBudgetBytes: UInt64
        public let activeSequenceCount: Int
        public let modelCount: Int
    }

    private struct SequenceEntry {
        let modelId: String
        let estimatedBytes: UInt64
    }

    private struct ModelBudget {
        var weightsBytes: UInt64
        var sequences: [UUID: SequenceEntry]
    }

    // MARK: - State

    private let gpuLimitBytes: UInt64
    private var weightsMemoryBytes: UInt64 = 0
    private var committedKVBytes: UInt64 = 0
    private var models: [String: ModelBudget] = [:]
    private let headroomPercent: Int  // e.g. 10 means keep 10% free

    // MARK: - Init

    public init(gpuLimitBytes: UInt64, headroomPercent: Int = 10) {
        self.gpuLimitBytes = gpuLimitBytes
        self.headroomPercent = headroomPercent
    }

    // MARK: - Queries

    /// Available bytes for new KV cache allocations (after weights and committed KV).
    public var availableKVBudget: UInt64 {
        gpuLimitBytes >= (weightsMemoryBytes + committedKVBytes)
            ? gpuLimitBytes - weightsMemoryBytes - committedKVBytes
            : 0
    }

    /// Check whether a new sequence can be admitted without exceeding budget.
    /// Factors in the headroom percentage (default 10%).
    public func canAdmit(
        modelId: String,
        estimatedTokens: Int,
        bytesPerToken: Int
    ) -> Bool {
        let needed = UInt64(estimatedTokens) * UInt64(bytesPerToken)
        let usableBudget = availableKVBudget * UInt64(100 - headroomPercent) / 100
        return needed <= usableBudget
    }

    /// Current metrics snapshot.
    public var metrics: Metrics {
        Metrics(
            gpuLimitBytes: gpuLimitBytes,
            weightsMemoryBytes: weightsMemoryBytes,
            committedKVBytes: committedKVBytes,
            availableBudgetBytes: availableKVBudget,
            activeSequenceCount: models.values.reduce(0) { $0 + $1.sequences.count },
            modelCount: models.count
        )
    }

    // MARK: - Mutation

    /// Reserve memory for a new sequence. Called at admission time.
    public func reserve(
        modelId: String,
        sequenceId: UUID,
        weightsBytes: UInt64,
        estimatedTokens: Int,
        bytesPerToken: Int
    ) {
        let kvBytes = UInt64(estimatedTokens) * UInt64(bytesPerToken)

        if models[modelId] == nil {
            models[modelId] = ModelBudget(weightsBytes: weightsBytes, sequences: [:])
            self.weightsMemoryBytes += weightsBytes
            NovaMLXLog.info("MemoryBudget: registered model \(modelId), weights=\(weightsBytes / 1024 / 1024)MB")
        }

        models[modelId]?.sequences[sequenceId] = SequenceEntry(
            modelId: modelId,
            estimatedBytes: kvBytes
        )
        committedKVBytes += kvBytes

        NovaMLXLog.info("MemoryBudget: reserved \(kvBytes / 1024 / 1024)MB for sequence \(sequenceId.uuidString.prefix(8)), committed=\(committedKVBytes / 1024 / 1024)MB, available=\(availableKVBudget / 1024 / 1024)MB")
    }

    /// Release memory when a sequence completes or is preempted.
    public func release(sequenceId: UUID) {
        var foundModelId: String?
        var foundEntry: SequenceEntry?

        for (modelId, budget) in models {
            if let entry = budget.sequences[sequenceId] {
                foundModelId = modelId
                foundEntry = entry
                break
            }
        }

        guard let modelId = foundModelId, let entry = foundEntry else {
            NovaMLXLog.warning("MemoryBudget: release called for unknown sequence \(sequenceId.uuidString.prefix(8))")
            return
        }

        models[modelId]?.sequences.removeValue(forKey: sequenceId)
        committedKVBytes = committedKVBytes >= entry.estimatedBytes
            ? committedKVBytes - entry.estimatedBytes
            : 0
        NovaMLXLog.info("MemoryBudget: released \(entry.estimatedBytes / 1024 / 1024)MB for sequence \(sequenceId.uuidString.prefix(8)), committed=\(committedKVBytes / 1024 / 1024)MB, available=\(availableKVBudget / 1024 / 1024)MB")
    }

    /// Update the actual memory usage for a sequence (feedback loop).
    /// If actual differs from estimate, adjust the committed total.
    public func updateActual(modelId: String, sequenceId: UUID, actualBytes: UInt64) {
        guard var budget = models[modelId],
              let entry = budget.sequences[sequenceId] else { return }
        let delta = Int64(actualBytes) - Int64(entry.estimatedBytes)
        if delta != 0 {
            committedKVBytes = UInt64(Int64(committedKVBytes) + delta)
            budget.sequences[sequenceId] = SequenceEntry(
                modelId: modelId,
                estimatedBytes: actualBytes
            )
            models[modelId] = budget
            NovaMLXLog.info("MemoryBudget: adjusted sequence \(sequenceId.uuidString.prefix(8)) by \(delta / 1024)KB")
        }
    }
}
