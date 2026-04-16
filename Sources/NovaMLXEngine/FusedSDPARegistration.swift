import MLX
import MLXLMCommon

public enum FusedSDPARegistration {
    public static func register() {
        _novamlxFusedSDPA = { queries, qKeys, qValues, scale, groupSize, bits in
            FusedQuantizedSDPA.attention(
                queries: queries,
                quantizedKeys: qKeys,
                quantizedValues: qValues,
                scale: scale,
                groupSize: groupSize,
                bits: bits
            )
        }
    }

    public static func disableFused() {
        _novamlxFusedSDPA = nil
    }
}
