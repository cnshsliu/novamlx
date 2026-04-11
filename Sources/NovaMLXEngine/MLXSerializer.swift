import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import NovaMLXCore
import NovaMLXUtils

public final class MLXSerializer: @unchecked Sendable {
    public static let shared = MLXSerializer()
    private let lock = NovaMLXLock()

    private init() {}

    public func perform<T>(_ operation: () throws -> T) rethrows -> T {
        try lock.withLock { try operation() }
    }
}
