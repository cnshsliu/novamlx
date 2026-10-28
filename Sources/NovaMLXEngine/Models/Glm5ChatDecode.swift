import Foundation
import NovaMLXCore
import NovaMLXUtils

/// `GLM-CHAT` tok/s = `decodeTokens / wallDecodeSeconds`.
///
/// Prefill is outside the window. The Mac check is `Glm5SpeedTests`
/// (`./build.sh test --filter Glm5SpeedTests`) or a short `nova chat` after
/// load. Both print this line. Z.ai's GLM-5.3-Flash post does not publish an
/// Apple Silicon tok/s. The community ceiling to chase is oMLX
/// GLM-5.3-Flash-oQ2e-mtp at about 38.5 TG tok/s at 4K on M4 Max 128GB.
/// That run uses fused decode kernels and MTP. This line is plain NovaMLX
/// decode on whatever prompt was just run.
enum Glm5ChatDecode {
    /// One-token forwards call `MLX.eval` on the hidden state every this many
    /// layers. oMLX evaluates on a stride so a resident checkpoint does not
    /// keep the whole decode graph lazy, and it does not sync inside every
    /// KDA step.
    static let decodeEvalStride = 1

    /// oMLX public benchmark `0g84suc9`. Not a Z.ai figure.
    static let communityCeilingTokPerSecond = 38.5
    static let communityBenchmarkURL = "https://omlx.ai/benchmarks/performance/0g84suc9"
    static let zaiBlogURL = "https://z.ai/blog/glm-5.3-flash"

    enum MtpWeights: String, Sendable {
        /// Checkpoint had `mtp.*` or a next-n layer and sanitize dropped it.
        case ignored
        /// Sanitize saw no next-n keys.
        case absent
        /// The loaded module was not `Glm5NextModel`.
        case unknown
    }

    /// Nil when there is no decode window (prefill-only, or a zero clock).
    static func line(
        decodeTokens: Int,
        seconds: Double,
        promptTokens: Int,
        mtp: MtpWeights
    ) -> String? {
        guard decodeTokens > 0, seconds > 0, seconds.isFinite else { return nil }
        let rate = Double(decodeTokens) / seconds
        let posix = Locale(identifier: "en_US_POSIX")
        let head = String(
            format: "GLM-CHAT: %.1f tok/s (%d decode tokens / %.3fs wall, prefill excluded, promptTokens=%d, mtp=%@). Formula: decodeTokens / wallDecodeSeconds. ",
            locale: posix,
            rate,
            decodeTokens,
            seconds,
            promptTokens,
            mtp.rawValue as NSString)
        let tail = String(
            format: "Z.ai does not publish an Apple Silicon tok/s (%@). Community ceiling to chase: %.1f TG tok/s at 4K on M4 Max 128GB for oMLX GLM-5.3-Flash-oQ2e-mtp (%@). This line is NovaMLX plain decode on the prompt above, not that run.",
            locale: posix,
            zaiBlogURL as NSString,
            communityCeilingTokPerSecond,
            communityBenchmarkURL as NSString)
        return head + tail
    }

    static func mtpWeights(of model: Any) -> MtpWeights {
        guard let glm = model as? Glm5NextModel else { return .unknown }
        return glm.droppedMtpKeys > 0 ? .ignored : .absent
    }
}

func logGlmChatDecode(
    container: ModelContainer,
    model: Any,
    promptTokens: Int,
    generationTokens: Int,
    tokensPerSecond: Double
) {
    guard container.identifier.family == .glm else { return }
    let seconds = tokensPerSecond > 0
        ? Double(generationTokens) / tokensPerSecond
        : 0
    guard let line = Glm5ChatDecode.line(
        decodeTokens: generationTokens,
        seconds: seconds,
        promptTokens: promptTokens,
        mtp: Glm5ChatDecode.mtpWeights(of: model)
    ) else { return }
    NovaMLXLog.info(line)
}
