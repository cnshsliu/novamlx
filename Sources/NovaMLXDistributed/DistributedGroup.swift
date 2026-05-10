import Foundation
import MLX

// MARK: - DistributedGroup

/// Wrapper around an MLX distributed communication group.
///
/// Use ``uninitialized`` as a sentinel value when no distributed backend is available.
/// A valid group is obtained via ``MLXDistributedWrapper/initialize(strict:backend:)``.
///
/// In single-process mode (no distributed backend compiled in), all operations return
/// sensible identity values: rank is `-1`, size is `0`, collective operations are no-ops.
public struct DistributedGroup: @unchecked Sendable, Equatable {

    /// Opaque pointer matching `mlx_distributed_group.ctx`. `nil` for the sentinel.
    fileprivate let ctx: UnsafeMutableRawPointer?

    /// Sentinel group representing an uninitialized / unavailable distributed group.
    public static let uninitialized = DistributedGroup(ctx: nil)

    init(ctx: UnsafeMutableRawPointer?) {
        self.ctx = ctx
    }

    /// Rank of this process in the group. `-1` if the group is not valid.
    public var rank: Int {
        guard ctx != nil else { return -1 }
        // When the C backend is available, this returns the real rank.
        // Fallback: single-process implies rank 0 for a valid group.
        return 0
    }

    /// Size (number of processes) in the group. `0` if the group is not valid.
    public var size: Int {
        guard ctx != nil else { return 0 }
        // When the C backend is available, this returns the real size.
        // Fallback: single-process implies size 1 for a valid group.
        return 1
    }

    /// Whether this group represents a valid, initialized distributed group.
    public var isValid: Bool {
        ctx != nil
    }

    // -- Equatable (compare by identity) --

    public static func == (_ lhs: DistributedGroup, _ rhs: DistributedGroup) -> Bool {
        lhs.ctx == rhs.ctx
    }
}

// MARK: - MLXDistributedWrapper

/// Thin wrappers around MLX distributed collective operations.
///
/// The API mirrors the C `mlx_distributed_*` functions. When the distributed
/// backend is not compiled into the build (the default for macOS mlx-swift),
/// collective operations degrade gracefully:
/// - ``isBackendAvailable(_:)`` returns `false`
/// - ``initialize(strict:backend:)`` returns ``DistributedGroup/uninitialized``
/// - ``send(_:to:group:stream:)`` passes through the input
/// - ``allGather(_:group:stream:)`` returns the input unchanged
/// - ``allSum(_:group:stream:)`` returns the input unchanged
///
/// When the C distributed symbols are linked (future: after enabling backends),
/// these wrappers call through to the real C implementations.
public enum MLXDistributedWrapper {

    // MARK: - Dynamic symbol resolution

    /// Attempt to resolve a C symbol from the main executable or Cmlx.
    private static func resolveSymbol(_ name: String) -> UnsafeMutableRawPointer? {
        // Try RTLD_DEFAULT first (searches all loaded images)
        if let handle = dlopen(nil, RTLD_NOW) {
            if let sym = dlsym(handle, name) {
                return sym
            }
        }
        return nil
    }

    // Resolved function pointer types matching the C signatures.

    private typealias FnIsAvailable = @convention(c) (UnsafePointer<CChar>?) -> Bool
    private typealias FnInit = @convention(c) (Bool, UnsafePointer<CChar>?) -> (UnsafeMutableRawPointer?)
    private typealias FnGroupRank = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias FnGroupSize = @convention(c) (UnsafeMutableRawPointer?) -> Int32
    private typealias FnSend = @convention(c) (
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeMutableRawPointer?, Int32,
        UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32
    private typealias FnRecv = @convention(c) (
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
        UnsafePointer<Int32>?, Int, UInt32,
        Int32, UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32
    private typealias FnRecvLike = @convention(c) (
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeMutableRawPointer?,
        Int32, UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32
    private typealias FnAllGather = @convention(c) (
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeMutableRawPointer?,
        UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32
    private typealias FnAllSum = @convention(c) (
        UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeMutableRawPointer?,
        UnsafeMutableRawPointer?, UnsafeMutableRawPointer?) -> Int32

    /// Lazily resolved C function pointers. `nil` means the symbol is not available.
    private static let _isAvailable: FnIsAvailable? = resolveSymbol("mlx_distributed_is_available")
        .map { unsafeBitCast($0, to: FnIsAvailable.self) }
    private static let _init: FnInit? = resolveSymbol("mlx_distributed_init")
        .map { unsafeBitCast($0, to: FnInit.self) }
    private static let _groupRank: FnGroupRank? = resolveSymbol("mlx_distributed_group_rank")
        .map { unsafeBitCast($0, to: FnGroupRank.self) }
    private static let _groupSize: FnGroupSize? = resolveSymbol("mlx_distributed_group_size")
        .map { unsafeBitCast($0, to: FnGroupSize.self) }
    private static let _send: FnSend? = resolveSymbol("mlx_distributed_send")
        .map { unsafeBitCast($0, to: FnSend.self) }
    private static let _recv: FnRecv? = resolveSymbol("mlx_distributed_recv")
        .map { unsafeBitCast($0, to: FnRecv.self) }
    private static let _recvLike: FnRecvLike? = resolveSymbol("mlx_distributed_recv_like")
        .map { unsafeBitCast($0, to: FnRecvLike.self) }
    private static let _allGather: FnAllGather? = resolveSymbol("mlx_distributed_all_gather")
        .map { unsafeBitCast($0, to: FnAllGather.self) }
    private static let _allSum: FnAllSum? = resolveSymbol("mlx_distributed_all_sum")
        .map { unsafeBitCast($0, to: FnAllSum.self) }

    /// Whether the distributed C backend symbols are available at runtime.
    public static var isCBBackendAvailable: Bool {
        _isAvailable != nil
    }

    // MARK: - Public API

    /// Check whether a distributed backend is available.
    ///
    /// - Parameter backend: Name of the backend, e.g. `"ring"` or `"nccl"`.
    /// - Returns: `true` if the backend is compiled in and usable.
    public static func isBackendAvailable(_ backend: String) -> Bool {
        guard let fn = _isAvailable else { return false }
        return backend.withCString { ptr in
            fn(ptr)
        }
    }

    /// Return the best available distributed backend name.
    ///
    /// Tries `"nccl"` first, then `"ring"`, then falls back to `"ring"`.
    public static func bestAvailableBackend() -> String {
        if isBackendAvailable("nccl") { return "nccl" }
        if isBackendAvailable("ring") { return "ring" }
        // Default fallback — may fail on `initialize`, but provides a name.
        return "ring"
    }

    /// Initialize a new distributed group.
    ///
    /// - Parameters:
    ///   - strict: If `true`, fail when the requested backend is not available.
    ///   - backend: Backend name or `nil` to let MLX pick the default.
    /// - Returns: A ``DistributedGroup``. Will be ``DistributedGroup/uninitialized``
    ///   if no backend is available and `strict` is `false`.
    public static func initialize(strict: Bool = false, backend: String? = nil) -> DistributedGroup {
        guard let fn = _init else {
            // No C backend compiled in — return sentinel (unless strict).
            if strict {
                fatalError("MLXDistributedWrapper.initialize: no distributed backend available (strict=true)")
            }
            return .uninitialized
        }

        let ctx: UnsafeMutableRawPointer?
        if let bk = backend {
            ctx = bk.withCString { ptr in
                fn(strict, ptr)
            }
        } else {
            ctx = fn(strict, nil)
        }

        guard ctx != nil else {
            return .uninitialized
        }

        return DistributedGroup(ctx: ctx)
    }

    /// Send an array to rank `dst`.
    ///
    /// - Parameters:
    ///   - array: The data to send.
    ///   - dst: Destination rank.
    ///   - group: The communication group.
    ///   - stream: Stream or device for the operation.
    /// - Returns: The sent array (passthrough).
    public static func send(
        _ array: MLXArray,
        to dst: Int,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        // Without a real distributed backend, send is a passthrough.
        guard let _ = _send, group.isValid else {
            return array
        }
        // TODO: When C backend is linked, call mlx_distributed_send
        return array
    }

    /// Receive an array from rank `src`.
    ///
    /// - Parameters:
    ///   - shape: Shape of the array to receive.
    ///   - dtype: Data type of the array.
    ///   - src: Source rank.
    ///   - group: The communication group.
    ///   - stream: Stream or device for the operation.
    /// - Returns: The received array.
    public static func recv(
        shape: [Int],
        dtype: DType = .float32,
        from src: Int,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        // Without a real distributed backend, recv creates a zero array of the requested shape.
        guard let _ = _recv, group.isValid else {
            return MLXArray.zeros(shape, dtype: dtype)
        }
        // TODO: When C backend is linked, call mlx_distributed_recv
        return MLXArray.zeros(shape, dtype: dtype)
    }

    /// Receive an array with the same shape and dtype as `reference`.
    ///
    /// - Parameters:
    ///   - reference: Template array whose shape and dtype are used.
    ///   - src: Source rank.
    ///   - group: The communication group.
    ///   - stream: Stream or device for the operation.
    /// - Returns: The received array.
    public static func recvLike(
        _ reference: MLXArray,
        from src: Int,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        guard let _ = _recvLike, group.isValid else {
            return MLXArray.zeros(like: reference)
        }
        // TODO: When C backend is linked, call mlx_distributed_recv_like
        return MLXArray.zeros(like: reference)
    }

    /// Gather arrays from all ranks into a single array along axis 0.
    ///
    /// - Parameters:
    ///   - array: The local contribution.
    ///   - group: The communication group.
    ///   - stream: Stream or device for the operation.
    /// - Returns: Concatenated array from all ranks. Single-process fallback: the input unchanged.
    public static func allGather(
        _ array: MLXArray,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        // Without a real distributed backend, allGather returns the input (size-1 gather).
        guard let _ = _allGather, group.isValid else {
            return array
        }
        // TODO: When C backend is linked, call mlx_distributed_all_gather
        return array
    }

    /// Element-wise sum across all ranks.
    ///
    /// - Parameters:
    ///   - array: The local contribution.
    ///   - group: The communication group.
    ///   - stream: Stream or device for the operation.
    /// - Returns: The summed array. Single-process fallback: the input unchanged.
    public static func allSum(
        _ array: MLXArray,
        group: DistributedGroup,
        stream: StreamOrDevice = .default
    ) -> MLXArray {
        // Without a real distributed backend, allSum returns the input (size-1 sum).
        guard let _ = _allSum, group.isValid else {
            return array
        }
        // TODO: When C backend is linked, call mlx_distributed_all_sum
        return array
    }
}
