import Foundation
import MLX

// Memo for TIE sync hooks. SwitchGLU calls 3 SwitchLinears with the same
// indices tensor; decode often repeats the same expert set. Remembering either
// fact avoids GPU→CPU asArray() and restack.

public final class TierHotPathCache: @unchecked Sendable {
    public static let shared = TierHotPathCache()

    public struct ForwardHit {
        public let expertIDs: [Int]
        public let localIndices: MLXArray
    }

    private let lock = NSLock()
    private var lastLayer: Int = -1
    private var lastIndicesId: ObjectIdentifier?
    private var lastExpertIDs: [Int] = []
    private var lastLocalIndices: MLXArray?
    private var lastStack: [ObjectIdentifier: [Int]] = [:]

    public init() {}

    public func reuseForward(layer: Int, indicesId: ObjectIdentifier) -> ForwardHit? {
        lock.lock(); defer { lock.unlock() }
        guard lastLayer == layer, lastIndicesId == indicesId, let local = lastLocalIndices else {
            return nil
        }
        return ForwardHit(expertIDs: lastExpertIDs, localIndices: local)
    }

    public func rememberForward(
        layer: Int,
        indicesId: ObjectIdentifier,
        expertIDs: [Int],
        localIndices: MLXArray
    ) {
        lock.lock(); defer { lock.unlock() }
        lastLayer = layer
        lastIndicesId = indicesId
        lastExpertIDs = expertIDs
        lastLocalIndices = localIndices
    }

    public func stackedMatches(id: ObjectIdentifier, uniqueSorted: [Int]) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return lastStack[id] == uniqueSorted
    }

    public func rememberStacked(id: ObjectIdentifier, uniqueSorted: [Int]) {
        lock.lock(); defer { lock.unlock() }
        lastStack[id] = uniqueSorted
    }

    public func reset() {
        lock.lock(); defer { lock.unlock() }
        lastLayer = -1
        lastIndicesId = nil
        lastExpertIDs = []
        lastLocalIndices = nil
        lastStack.removeAll()
    }
}
