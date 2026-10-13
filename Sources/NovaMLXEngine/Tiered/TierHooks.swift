import MLX

/// Callbacks the tiered loader installs around Linear and SwitchLinear calls.
///
/// Current mlx-swift does not define this type or invoke it from those layers.
/// NovaMLX still assigns the hooks: DeepSeek V4 calls `linearSyncHook` directly,
/// and the installer records expert ids when a model invokes the switch hook.
public enum TierHooks {
    public nonisolated(unsafe) static var linearHook: ((AnyObject) -> Void)?
    public nonisolated(unsafe) static var switchLinearHook: ((AnyObject) -> Void)?
    public nonisolated(unsafe) static var linearSyncHook: ((AnyObject) -> Void)?
    public nonisolated(unsafe) static var switchLinearSyncHook: ((AnyObject, MLXArray) -> Void)?
}
