import Foundation

/// Which verify path Engine is allowed to run for a multi-token-prediction request.
///
/// Qwen greedy uses MLXLMCommon `MTPSpeculativeTokenIterator` (real draft +
/// target argmax). The Nova `MtpTarget` / `MtpDrafter` iterator is a disabled
/// stub: it used to call ordinary decode while logging MTP. DeepSeek's
/// in-graph head still conforms to those protocols; wiring it to
/// `MTPDrafterModel` is a follow-up, so that route is serial decode.
public enum MtpVerifyRoute: String, Sendable, Equatable {
    /// Temperature is 0. Load the Qwen MTP drafter and verify with MLXLMCommon.
    case qwenMLXLMCommon
    /// Qwen MTP acceptance is argmax. Above temperature 0, do not speculate.
    case qwenSerialTemperature
    /// `MtpTokenIterator` must not run. Serial decode, no MTP stats.
    case nativeStubDisabled
}

public enum MtpVerifyRouting {
    /// `draftIsQwenMtpConfig` wins over a native head on the same request.
    /// A nil route is an ordinary draft or plain decode, not MTP.
    public static func route(
        draftIsQwenMtpConfig: Bool,
        temperature: Float,
        nativeHeadMatched: Bool
    ) -> MtpVerifyRoute? {
        if draftIsQwenMtpConfig {
            return temperature <= 0 ? .qwenMLXLMCommon : .qwenSerialTemperature
        }
        if nativeHeadMatched {
            return .nativeStubDisabled
        }
        return nil
    }
}
