import Foundation

/// Which interface a local model serves. Same operations as tknet
/// `API_PRESETS` in `src/lib/supply/api-contract.ts`. Returns nil when the
/// model cannot be placed on a known interface — a wrong path is worse
/// than no path.
public enum SupplyAPI {
    public static func contract(modelId: String, modelType: String, family: String) -> APIContract? {
        let type = modelType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let fam = family.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = leaf(modelId)
        if backgroundRemoval.contains(name) { return backgroundRemovalContract }
        switch type {
        case "image":
            return image
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

    private static let chat = APIContract(
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

    private static let image = APIContract(
        kind: "openai.image",
        summary: "OpenAI image generation.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/images/generations",
                contentType: "application/json",
                summary: "JSON body: model, prompt, size, n, response_format (b64_json or url)."),
            APIOperation(
                method: "POST", path: "/v1/images/edits",
                contentType: "multipart/form-data",
                summary: "Multipart fields: image, prompt, model. Optional mask, size, n."),
            APIOperation(
                method: "POST", path: "/v1/images/variations",
                contentType: "multipart/form-data",
                summary: "Multipart fields: image, model. Optional size, n."),
        ])

    private static let speech = APIContract(
        kind: "openai.audio.speech",
        summary: "Speech synthesis.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/audio/speech",
                contentType: "application/json",
                summary: "JSON body: model, input, voice. The response is audio bytes.")
        ])

    private static let transcription = APIContract(
        kind: "openai.audio.transcription",
        summary: "Audio transcription.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/audio/transcriptions",
                contentType: "multipart/form-data",
                summary: "Multipart fields: file, model. A JSON body with a base64 file is also accepted.")
        ])

    private static let embeddings = APIContract(
        kind: "openai.embeddings",
        summary: "Embeddings.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/embeddings",
                contentType: "application/json",
                summary: "JSON body: model, input.")
        ])

    private static let decision = APIContract(
        kind: "novamlx.decision",
        summary: "Typed decisions.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/decisions",
                contentType: "application/json",
                summary: "JSON body: model, state, questions.")
        ])

    private static let backgroundRemovalContract = APIContract(
        kind: "novamlx.background-removal",
        summary: "Background removal.",
        operations: [
            APIOperation(
                method: "POST", path: "/v1/images/background-removals",
                contentType: "multipart/form-data",
                summary: "Multipart fields: image, model (u2net, isnet-general-use, birefnet-portrait, or bria-rmbg), response_format (b64_json or url). The response uses the image generation envelope.")
        ])
}
