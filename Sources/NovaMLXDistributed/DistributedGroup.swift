import Foundation
import MLX
import Cmlx

// MARK: - DistributedGroup

/// Wrapper around an MLX distributed communication group.
///
/// Use ``uninitialized`` as a sentinel value when no distributed backend is available.
/// A valid group is obtained via ``MLXDistributedWrapper/initialize(strict:backend:hostfile:rank:)``.
///
/// In single-process mode (no distributed backend compiled in), all operations return
/// sensible identity values: rank is `-1`, size is `0`, collective operations are no-ops.
public final class DistributedGroup: @unchecked Sendable, Equatable {

    /// The underlying C distributed group handle.
    fileprivate var cGroup: mlx_distributed_group

    /// Sentinel group representing an uninitialized / unavailable distributed group.
    public static let uninitialized = DistributedGroup(cGroup: .init())

    init(cGroup: mlx_distributed_group) {
        self.cGroup = cGroup
    }

    /// Rank of this process in the group. `-1` if the group is not valid.
    public var rank: Int {
        guard cGroup.ctx != nil else { return -1 }
        return Int(mlx_distributed_group_rank(cGroup))
    }

    /// Size (number of processes) in the group. `0` if the group is not valid.
    public var size: Int {
        guard cGroup.ctx != nil else { return 0 }
        return Int(mlx_distributed_group_size(cGroup))
    }

    /// Whether this group represents a valid, initialized distributed group.
    public var isValid: Bool {
        cGroup.ctx != nil
    }

    // -- Equatable (compare by identity) --

    public static func == (_ lhs: DistributedGroup, _ rhs: DistributedGroup) -> Bool {
        lhs.cGroup.ctx == rhs.cGroup.ctx
    }
}

// MARK: - MLXDistributedWrapper

/// Direct Swift wrappers around MLX distributed collective C operations.
///
/// Uses the Cmlx C API directly instead of dlsym — the Ring backend (TCP-based)
/// is always compiled in and provides proper send/recv/all_reduce operations.
public enum MLXDistributedWrapper {

    // MARK: - Backend queries

    /// Whether the distributed C backend symbols are available at runtime.
    public static var isCBBackendAvailable: Bool {
        // Ring backend is always compiled in, so distributed is available
        true
    }

    /// Check whether a distributed backend is available.
    public static func isBackendAvailable(_ backend: String) -> Bool {
        return backend.withCString { ptr in
            mlx_distributed_is_available(ptr)
        }
    }

    /// Return the best available distributed backend name.
    ///
    /// Tries `"jaccl"` first (RDMA over Thunderbolt), then `"ring"` (TCP).
    public static func bestAvailableBackend() -> String {
        if isBackendAvailable("jaccl") { return "jaccl" }
        if isBackendAvailable("ring") { return "ring" }
        return "ring"
    }

    // MARK: - Group lifecycle

    /// Initialize a new distributed group.
    ///
    /// - Parameters:
    ///   - strict: If `true`, fail when the requested backend is not available.
    ///   - backend: Backend name or `nil` to let MLX pick the default.
    /// - Returns: A ``DistributedGroup``. Will be ``DistributedGroup/uninitialized``
    ///   if no backend is available and `strict` is `false`.
    public static func initialize(strict: Bool = false, backend: String? = nil) -> DistributedGroup {
        let cGroup: mlx_distributed_group
        if let bk = backend {
            cGroup = bk.withCString { ptr in
                mlx_distributed_init(strict, ptr)
            }
        } else {
            cGroup = mlx_distributed_init(strict, nil)
        }

        guard cGroup.ctx != nil else {
            return .uninitialized
        }

        return DistributedGroup(cGroup: cGroup)
    }

    // MARK: - Point-to-point

    /// Send an array to rank `dst`.
    public static func send(
        _ array: MLXArray,
        to dst: Int,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard group.isValid else { return array }

        var result = mlx_array_new()
        let rc = mlx_distributed_send(&result, array.ctx, Int32(dst), group.cGroup, stream.ctx)

        if rc == 0 {
            return MLXArray(result)
        }
        mlx_array_free(result)
        return array
    }

    /// Receive an array from rank `src`.
    public static func recv(
        shape: [Int],
        dtype: DType = .float32,
        from src: Int,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard group.isValid else {
            return MLXArray.zeros(shape, dtype: dtype)
        }

        let cShape = shape.map { Int32($0) }
        var result = mlx_array_new()

        let rc = cShape.withUnsafeBufferPointer { shapePtr in
            mlx_distributed_recv(
                &result,
                shapePtr.baseAddress,
                cShape.count,
                dtype.cmlxDtype,
                Int32(src),
                group.cGroup,
                stream.ctx
            )
        }

        if rc == 0 {
            return MLXArray(result)
        }
        mlx_array_free(result)
        return MLXArray.zeros(shape, dtype: dtype)
    }

    /// Receive an array with the same shape and dtype as `reference`.
    public static func recvLike(
        _ reference: MLXArray,
        from src: Int,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard group.isValid else {
            return MLXArray.zeros(reference.shape)
        }

        var result = mlx_array_new()
        let rc = mlx_distributed_recv_like(&result, reference.ctx, Int32(src), group.cGroup, stream.ctx)

        if rc == 0 {
            return MLXArray(result)
        }
        mlx_array_free(result)
        return MLXArray.zeros(reference.shape)
    }

    // MARK: - Collectives

    /// Gather arrays from all ranks into a single array along axis 0.
    public static func allGather(
        _ array: MLXArray,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard group.isValid else { return array }

        var result = mlx_array_new()
        let rc = mlx_distributed_all_gather(&result, array.ctx, group.cGroup, stream.ctx)

        if rc == 0 {
            return MLXArray(result)
        }
        mlx_array_free(result)
        return array
    }

    /// Element-wise sum across all ranks.
    public static func allSum(
        _ array: MLXArray,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard group.isValid else { return array }

        var result = mlx_array_new()
        let rc = mlx_distributed_all_sum(&result, array.ctx, group.cGroup, stream.ctx)

        if rc == 0 {
            return MLXArray(result)
        }
        mlx_array_free(result)
        return array
    }
}

// MARK: - DType → mlx_dtype conversion

extension DType {
    var cmlxDtype: mlx_dtype {
        switch self {
        case .bool: return MLX_BOOL
        case .uint8: return MLX_UINT8
        case .uint16: return MLX_UINT16
        case .uint32: return MLX_UINT32
        case .uint64: return MLX_UINT64
        case .int8: return MLX_INT8
        case .int16: return MLX_INT16
        case .int32: return MLX_INT32
        case .int64: return MLX_INT64
        case .float16: return MLX_FLOAT16
        case .float32: return MLX_FLOAT32
        case .float64: return MLX_FLOAT64
        case .bfloat16: return MLX_BFLOAT16
        case .complex64: return MLX_COMPLEX64
        }
    }
}
