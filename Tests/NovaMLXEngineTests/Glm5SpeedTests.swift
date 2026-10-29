import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import NovaMLXEngine

/// Manual speed probe for text GLM-5.3-Flash. Prints `GLM-CHAT` after a
/// two-token warmup so shader compile sits outside the window. The published
/// comparison is the oMLX community ceiling (38.5 TG tok/s at 4K on M4 Max
/// 128GB), not a Z.ai number. This VM has no Apple Silicon and no weights;
/// the test returns when `config.json` is missing.
///
/// ```bash
/// NOVA_GLM53_MODEL=/Volumes/Samsung768/Models/TensorFold/GLM-5.3-Flash-MLX-2bit-MTP \
///   ./build.sh test --filter Glm5SpeedTests
/// ```
///
/// `NOVA_GLM53_PROMPT_TOKENS=4096` repeats the user text until the templated
/// prompt reaches that length (capped at the 32,768 text-path limit) so a
/// Mac can put a 4K prefill next to the community figure. The default is a
/// short chat. `NOVA_GLM53_DECODE_TOKENS` defaults to 32 and caps at 128.
@Suite("GLM-5.3-Flash decode speed")
struct Glm5SpeedTests {

    private static let repoIds = [
        "TensorFold/GLM-5.3-Flash-MLX-2bit-MTP",
        "mlx-community/GLM-5.3-Flash-4bit",
    ]

    private static let roots = [
        "/Volumes/Samsung768/Models",
        "/Users/lucas/Models",
        NSHomeDirectory() + "/Models",
    ]

    private static func weightDirectory() -> URL? {
        let fm = FileManager.default
        if let override = ProcessInfo.processInfo.environment["NOVA_GLM53_MODEL"],
           !override.isEmpty
        {
            let url = URL(fileURLWithPath: override)
            let config = url.appendingPathComponent("config.json")
            if fm.fileExists(atPath: config.path) { return url }
            print("GLM-SPEED-SKIP: NOVA_GLM53_MODEL has no config.json at \(url.path)")
            return nil
        }
        for root in roots {
            for repo in repoIds {
                let url = URL(fileURLWithPath: root).appendingPathComponent(repo)
                let config = url.appendingPathComponent("config.json")
                if fm.fileExists(atPath: config.path) { return url }
            }
        }
        return nil
    }

    private static func decodeCount() -> Int {
        let raw = ProcessInfo.processInfo.environment["NOVA_GLM53_DECODE_TOKENS"] ?? ""
        let requested = Int(raw) ?? 32
        return min(max(requested, 1), 128)
    }

    private static func promptTarget() -> Int {
        let raw = ProcessInfo.processInfo.environment["NOVA_GLM53_PROMPT_TOKENS"] ?? ""
        let requested = Int(raw) ?? 0
        return min(max(requested, 0), Glm5NextMemory.validatedContextTokens)
    }

    @Test("short chat prints GLM-CHAT tok/s when weights are on disk")
    func testShortChatDecodeSpeed() async throws {
        guard let dir = Self.weightDirectory() else {
            print(
                "GLM-SPEED-SKIP: weights not present. Set NOVA_GLM53_MODEL or place TensorFold/GLM-5.3-Flash-MLX-2bit-MTP under /Volumes/Samsung768/Models."
            )
            return
        }
        print("GLM-SPEED-WEIGHTS:", dir.path)

        await CustomModelRegistration.ensureRegistered()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: dir, using: MLXEngine.tokenizerLoader)
        let tokenizer = await container.tokenizer
        let base = "用三句话说明什么是混合专家模型。"
        var content = base
        var promptIds = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": content]])
        let target = Self.promptTarget()
        var grows = 0
        while target > promptIds.count, grows < 16 {
            content += "\n" + base
            promptIds = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": content]])
            grows += 1
        }
        if promptIds.count > Glm5NextMemory.validatedContextTokens {
            promptIds = Array(promptIds.prefix(Glm5NextMemory.validatedContextTokens))
        }
        let ids = promptIds
        let decodeTokens = Self.decodeCount()
        let measured = try await container.perform { context -> (Double, Int, Int) in
            let glm = context.model as? Glm5NextModel
            let prompt = MLXArray(ids.map { Int32($0) }).reshaped(1, -1)
            let caches = try context.model.newCache(parameters: nil)
            var logits = context.model(prompt, cache: caches)
            eval(logits)
            var last = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            for _ in 0 ..< 2 {
                logits = context.model(MLXArray(Int32(last)).reshaped(1, 1), cache: caches)
                eval(logits)
                last = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            }
            let start = Date()
            for _ in 0 ..< decodeTokens {
                logits = context.model(MLXArray(Int32(last)).reshaped(1, 1), cache: caches)
                eval(logits)
                last = Int(argMax(logits[0..., -1, 0...]).item(Int.self))
            }
            let seconds = Date().timeIntervalSince(start)
            return (seconds, ids.count, glm?.droppedMtpKeys ?? -1)
        }
        let mtp: Glm5ChatDecode.MtpWeights = measured.2 > 0 ? .ignored : (measured.2 == 0 ? .absent : .unknown)
        print("GLM-MTP:", mtp.rawValue, "droppedKeys=\(measured.2)")
        if let line = Glm5ChatDecode.line(
            decodeTokens: decodeTokens,
            seconds: measured.0,
            promptTokens: measured.1,
            mtp: mtp
        ) {
            print(line)
        }
    }
}
