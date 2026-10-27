import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import NovaMLXCore
import NovaMLXUtils

// Qwen companion MTP. The drafter is not a chat `LanguageModel` (`qwen3_5_mtp`
// / `qwen3_8_mtp` live on `MTPDrafterTypeRegistry`), so it is loaded here the
// same way DFlash2 and DSpark stay out of the model pool. Verify is
// `MTPSpeculativeTokenIterator` from MLXLMCommon. Nova's `MtpTokenIterator`
// is not used on this path.

enum QwenMtpRegistration {
    private static let lock = NovaMLXLock()
    private nonisolated(unsafe) static var done = false

    static func ensure() async {
        if lock.withLock({ done }) { return }
        await Qwen35TextMTPRegistration.register()
        await Qwen35VLMMTPRegistration.register()
        lock.withLock { done = true }
    }
}

/// On-disk Qwen MTP head with weights. Config alone is not enough.
public func qwenMtpCompanionReady(modelId: String) -> Bool {
    let dir = NovaMLXPaths.directory(forModelId: modelId)
    guard NovaMLXEngine.isMtpDraftConfig(at: dir) else { return false }
    if let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path),
       files.contains(where: { $0.hasSuffix(".safetensors") })
    {
        return true
    }
    return FileManager.default.fileExists(
        atPath: dir.appendingPathComponent("model.safetensors.index.json").path)
}

/// `true` when the target is the multimodal Qwen3.5/3.8 class. `false` for the
/// text classes. `nil` when this drafter cannot verify the target.
func qwenMtpUsesVisionDrafter(_ model: any LanguageModel) -> Bool? {
    if model is MLXVLM.Qwen35 { return true }
    if model is Qwen35Model || model is Qwen35TextModel { return false }
    return nil
}

private final class QwenMtpDrafterBox: @unchecked Sendable {
    let model: any MTPDrafterModel
    init(_ model: any MTPDrafterModel) { self.model = model }
}

/// Process-wide Qwen MTP drafters. State for a round stays on the iterator
/// (`StatefulMTPDrafterModel`); the loaded module is shared and read-only.
public final class QwenMtpDrafterStore: @unchecked Sendable {
    public static let shared = QwenMtpDrafterStore()

    private let lock = NovaMLXLock()
    private var containers: [String: MTPDrafterContainer] = [:]

    private init() {}

    func drafter(directory: URL, vision: Bool) async throws -> any MTPDrafterModel {
        let key = directory.path + (vision ? "#vlm" : "#text")
        if let cached = lock.withLock({ containers[key] }) {
            let box = await cached.perform { context in
                QwenMtpDrafterBox(context.model)
            }
            return box.model
        }
        await QwenMtpRegistration.ensure()
        // Text and VLM registrations both land on MTPDrafterTypeRegistry.shared.
        // This tree's factory is MTPDrafterModelFactory.shared; model_type
        // selects the head. `vision` only splits the cache key.
        let container = try await MTPDrafterModelFactory.shared.loadContainer(
            from: directory, using: LocalTokenizerLoader())
        lock.withLock { containers[key] = container }
        let box = await container.perform { context in
            QwenMtpDrafterBox(context.model)
        }
        return box.model
    }

    public func release(modelId: String) {
        let prefix = NovaMLXPaths.directory(forModelId: modelId).path + "#"
        lock.withLock {
            containers = containers.filter { !$0.key.hasPrefix(prefix) }
        }
    }
}

private final class MtpLogitProcessorCarrier: @unchecked Sendable {
    private let processor: any LogitProcessor

    init(_ processor: any LogitProcessor) {
        self.processor = processor
    }

    func make() -> any LogitProcessor { processor }
}

func logNativeMtpStubDisabled(tag: String) {
    NovaMLXLog.info(
        "[\(tag)] Nova MtpTarget/MtpDrafter verify stub is disabled (route=\(MtpVerifyRoute.nativeStubDisabled.rawValue)); serial decode"
    )
}

func logQwenMtpCompletion(_ info: GenerateCompletionInfo, tag: String) {
    guard let proposed = info.proposedDraftTokens else { return }
    let accepted = info.acceptedDraftTokens ?? 0
    let passthrough = info.passthroughReason.map { " passthrough=\($0)" } ?? ""
    NovaMLXLog.info(
        "[MTP:\(tag)] MLXLMCommon verify proposed=\(proposed) accepted=\(accepted)\(passthrough)"
    )
}

/// Qwen greedy MTP iterator, or nil when this request must stay on serial decode.
/// Temperature above 0 and a non-Qwen target return nil without touching the cache.
func openQwenMtpIterator(
    model: any LanguageModel,
    draftModelId: String,
    input: LMInput,
    cache: [KVCache]?,
    parameters: GenerateParameters,
    processor: LogitProcessor?,
    numDraftTokens: Int?,
    logTag: String
) async throws -> MTPSpeculativeTokenIterator? {
    let route = MtpVerifyRouting.route(
        draftIsQwenMtpConfig: true,
        temperature: parameters.temperature,
        nativeHeadMatched: false)
    switch route {
    case .qwenSerialTemperature:
        NovaMLXLog.info(
            "[\(logTag)] Qwen MTP '\(draftModelId)' idle — temperature \(parameters.temperature) > 0; serial decode"
        )
        return nil
    case .qwenMLXLMCommon:
        break
    case .nativeStubDisabled, nil:
        return nil
    }

    guard let vision = qwenMtpUsesVisionDrafter(model) else {
        NovaMLXLog.info(
            "[\(logTag)] Qwen MTP '\(draftModelId)' target is not Qwen3.5/3.8 text or VLM — serial decode"
        )
        return nil
    }

    let directory = NovaMLXPaths.directory(forModelId: draftModelId)
    let drafter = try await QwenMtpDrafterStore.shared.drafter(
        directory: directory, vision: vision)
    // MTPDrafterModel does not declare isCompatible. The target class was
    // already chosen above; the loaded head must be the matching concrete type.
    // A mismatch stays on serial decode. The VLM head traps on a text target
    // inside prepareDrafterState, so this check runs before the iterator.
    let drafterMatchesTarget = vision
        ? drafter is Qwen35VLMNextNDraftModel
        : drafter is Qwen35MTPDraftModel
    guard drafterMatchesTarget else {
        NovaMLXLog.info(
            "[\(logTag)] Qwen MTP '\(draftModelId)' loaded \(type(of: drafter)), which does not match \(type(of: model)) — serial decode"
        )
        return nil
    }

    // The caller already folded penalties into `processor`. Leaving them on
    // `parameters` would apply them a second time inside the MLXLMCommon iterator.
    var mtpParameters = parameters
    mtpParameters.repetitionPenalty = nil
    mtpParameters.presencePenalty = nil
    mtpParameters.frequencyPenalty = nil
    let components: GenerationComponents
    if let processor {
        let carrier = MtpLogitProcessorCarrier(processor)
        components = GenerationComponents(logitProcessorFactory: { carrier.make() })
    } else {
        components = GenerationComponents()
    }
    // Qwen's drafter publishes `maximumBlockSize` 2 (one draft + the bonus).
    // The iterator clamps to that. A larger request still verifies correctly.
    let blockSize = max(2, (numDraftTokens ?? 1) + 1)
    let iterator = try MTPSpeculativeTokenIterator(
        input: input,
        mainModel: model,
        drafter: drafter,
        mainCache: cache,
        parameters: mtpParameters,
        blockSize: blockSize,
        components: components)
    NovaMLXLog.info(
        "[\(logTag)] Qwen MTP MLXLMCommon verify draft=\(draftModelId) block=\(iterator.blockSize) greedy"
    )
    return iterator
}
