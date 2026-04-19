import MLX
import MLXLMCommon

// NOTE: _novamlxFusedSDPA was removed from mlx-swift 0.31.3.
// Fused SDPA registration is no longer needed — MLX handles this internally.

public enum FusedSDPARegistration {
    public static func register() {
        // No-op: fused SDPA is handled internally by MLX
    }

    public static func disableFused() {
        // No-op
    }
}
