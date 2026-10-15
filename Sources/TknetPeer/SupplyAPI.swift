import Foundation

/// Which interface a local model serves. Same operations as tknet
/// `API_PRESETS` in `src/lib/supply/api-contract.ts`. Returns nil when the
/// model cannot be placed on a known interface — a wrong path is worse
/// than no path.
public enum SupplyAPI {
    /// Pools with no `api` only accept chat (legacy / pathless tunnel frames).
    public static func allows(_ api: APIContract?, gatewayPath: String) -> Bool {
        let ops = api?.operations ?? chat.operations
        return ops.contains { $0.path == gatewayPath }
    }

    /// `/v1/chat/completions` → `chat/completions` for a member base ending in `/v1`.
    public static func upstreamSuffix(gatewayPath: String) -> String? {
        let trimmed = gatewayPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/v1/") else { return nil }
        let rest = String(trimmed.dropFirst(4))
        return rest.isEmpty ? nil : rest
    }

    /// Presets operators can assign to a pool (same kinds as tknet).
    public static var allContracts: [APIContract] {
        [chat, imageGenerations, imageEdits, speech, transcription,
         embeddings, decision, backgroundRemovalContract]
    }

        public static func contract(kind: String) -> APIContract? {
        switch kind {
        case chat.kind: return chat
        case imageGenerations.kind: return imageGenerations
        case imageEdits.kind: return imageEdits
        case speech.kind: return speech
        case transcription.kind: return transcription
        case embeddings.kind: return embeddings
        case decision.kind: return decision
        case backgroundRemovalContract.kind: return backgroundRemovalContract
        default: return nil
        }
    }

    public static func contract(modelId: String, modelType: String, family: String) -> APIContract? {
        let type = modelType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let fam = family.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = leaf(modelId)
        if backgroundRemoval.contains(name) { return backgroundRemovalContract }
        switch type {
        case "image":
            return imageGenerations
        case "image_edit", "image-edit", "image_edits", "image-edits":
            return imageEdits
        case "embedding":
            return embeddings
        case "decision":
            return decision
        case "audio":
            return audio(name: name, family: fam)
        case "vlm":
            return vlm
        case "llm", "":
            return chat
        default:
            return nil
        }
    }

    private static func leaf(_ modelId: String) -> String {
        let trimmed = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        let part = trimmed.split(separator: "/").last.map(String.init) ?? trimmed
        return part.lowercased()
    }

    private static func audio(name: String, family: String) -> APIContract? {
        let speechFamilies: Set<String> = ["qwen3Tts", "dotsTts"]
        let transcriptFamilies: Set<String> = ["qwen3Asr", "whisper"]
        if speechFamilies.contains(family) || name.contains("tts") || name.contains("speech") {
            return speech
        }
        if transcriptFamilies.contains(family) || name.contains("asr") || name.contains("whisper") {
            return transcription
        }
        return nil
    }

    private static let backgroundRemoval: Set<String> = [
        "u2net", "u2netp", "u2net_human_seg", "isnet-general-use",
        "silueta", "birefnet-portrait", "bria-rmbg",
    ]

    public static let chat = APIContract(
        kind: "openai.chat",
        summary: "OpenAI chat completions.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/chat/completions",
                contentType: "application/json",
                summary: "JSON body: model, messages. stream is optional.")
        ])

    private static let vlm = APIContract(
        kind: "openai.chat",
        summary: "OpenAI chat completions. Put images in message content.",
        operations: chat.operations)

    /// Text-to-image only. Separate from edits (see peer-api-consistency spec).
    public static let imageGenerations = APIContract(
        kind: "openai.image.generations",
        summary: "OpenAI image generation.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/images/generations",
                contentType: "application/json",
                summary: "JSON body: model, prompt, size, n, response_format (b64_json or url)."),
        ])

    /// Image edit only. Not bundled with generations.
    public static let imageEdits = APIContract(
        kind: "openai.image.edits",
        summary: "OpenAI image edits.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/images/edits",
                contentType: "multipart/form-data",
                summary: "Multipart fields: image, prompt, model. Optional mask, size, n."),
        ])

    public static let speech = APIContract(
        kind: "openai.audio.speech",
        summary: "Speech synthesis.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/audio/speech",
                contentType: "application/json",
                summary: "JSON body: model, input, voice. The response is audio bytes.")
        ])

    public static let transcription = APIContract(
        kind: "openai.audio.transcription",
        summary: "Audio transcription.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/audio/transcriptions",
                contentType: "multipart/form-data",
                summary: "Multipart fields: file, model. A JSON body with a base64 file is also accepted.")
        ])

    public static let embeddings = APIContract(
        kind: "openai.embeddings",
        summary: "Embeddings.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/embeddings",
                contentType: "application/json",
                summary: "JSON body: model, input.")
        ])

    public static let decision = APIContract(
        kind: "novamlx.decision",
        summary: "Typed decisions.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/decisions",
                contentType: "application/json",
                summary: "JSON body: model, state, questions.")
        ])

    public static let backgroundRemovalContract = APIContract(
        kind: "novamlx.background-removal",
        summary: "Background removal.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/images/background-removals",
                contentType: "multipart/form-data",
                summary: "Multipart fields: image, model (u2net, isnet-general-use, birefnet-portrait, or bria-rmbg), response_format (b64_json or url). The response uses the image generation envelope.")
        ])
}
