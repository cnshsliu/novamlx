import CryptoKit
import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Official Recursive `rec` chat template (`chat_template.jinja`).
///
/// `load_tokenizer` overwrites the checkpoint template with the package file.
/// Nova does the same on Flint load: this file wins, and a live
/// `chat_template.jinja` is quarantined because swift-transformers prefers it.
enum FlintRecChatTemplate {
    static let resourceName = "flint_rec"
    /// `hashlib.sha256(template.encode()).hexdigest()` of the package file.
    static let sha256 = "44fd1b0f94b78d9cf554c5a207945e48fef698b279d65863a33a77d787ca411b"

    struct TemplateMessage: Equatable, Sendable {
        var role: String
        var content: String
        var reasoning: String?
        var toolCalls: [TemplateCall]

        init(
            role: String,
            content: String = "",
            reasoning: String? = nil,
            toolCalls: [TemplateCall] = []
        ) {
            self.role = role
            self.content = content
            self.reasoning = reasoning
            self.toolCalls = toolCalls
        }
    }

    struct TemplateCall: Equatable, Sendable {
        var name: String
        var arguments: [(String, FlintOrderedJSON)]

        static func == (lhs: TemplateCall, rhs: TemplateCall) -> Bool {
            lhs.name == rhs.name && lhs.arguments.elementsEqual(rhs.arguments, by: { $0.0 == $1.0 && $0.1 == $1.1 })
        }
    }

    static func bundledData() -> Data {
        if let url = ResourceBundleLocator.url(
            forResource: resourceName,
            withExtension: "jinja",
            subdirectory: "ChatTemplates",
            inBundle: "NovaMLX_NovaMLXEngine"
        ), let data = try? Data(contentsOf: url), !data.isEmpty {
            return data
        }
        if let url = ResourceBundleLocator.url(
            forResource: resourceName,
            withExtension: "jinja",
            inBundle: "NovaMLX_NovaMLXEngine"
        ), let data = try? Data(contentsOf: url), !data.isEmpty {
            return data
        }
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("ChatTemplates/\(resourceName).jinja")
        if let data = try? Data(contentsOf: source), !data.isEmpty {
            return data
        }
        if let root = ProcessInfo.processInfo.environment["SRCROOT"] {
            let url = URL(fileURLWithPath: root)
                .appendingPathComponent("Sources/NovaMLXEngine/ChatTemplates/\(resourceName).jinja")
            if let data = try? Data(contentsOf: url), !data.isEmpty {
                return data
            }
        }
        return Data()
    }

    static func bundledSource() -> String {
        String(data: bundledData(), encoding: .utf8) ?? ""
    }

    /// Same algorithm as `rec` `hashlib.sha256(template.encode()).hexdigest()`.
    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func bundledSHA256() -> String {
        sha256Hex(bundledData())
    }

    /// Write the package template over `tokenizer_config.json` and quarantine
    /// a conflicting `chat_template.jinja`. Returns the sha256 that was logged.
    static func install(modelDir: URL, modelId: String) -> String {
        let data = bundledData()
        let digest = sha256Hex(data)
        let template = String(data: data, encoding: .utf8) ?? ""
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        var json: [String: Any] = [:]
        if let existing = try? Data(contentsOf: tcPath),
           let object = try? JSONSerialization.jsonObject(with: existing) as? [String: Any] {
            json = object
        }
        json["chat_template"] = template
        if let written = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) {
            try? written.write(to: tcPath, options: .atomic)
        }
        let jinjaPath = modelDir.appendingPathComponent("chat_template.jinja")
        quarantine(jinjaPath, modelId: modelId)
        NovaMLXLog.info("[Flint] chat_template_sha256=\(digest)")
        return digest
    }

    private static func quarantine(_ jinjaPath: URL, modelId: String) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: jinjaPath.path) else { return }
        let backup = jinjaPath.appendingPathExtension("disabled-by-novamlx")
        if fm.fileExists(atPath: backup.path) {
            try? fm.removeItem(at: jinjaPath)
            NovaMLXLog.info("[ChatTemplate] \(modelId): removed chat_template.jinja (existing backup at \(backup.lastPathComponent), reason=flint package template)")
            return
        }
        do {
            try fm.moveItem(at: jinjaPath, to: backup)
            NovaMLXLog.info("[ChatTemplate] \(modelId): quarantined chat_template.jinja → \(backup.lastPathComponent) (reason=flint package template)")
        } catch {
            NovaMLXLog.warning("[ChatTemplate] \(modelId): failed to quarantine chat_template.jinja: \(error)")
        }
    }

    /// Jinja render of `chat_template.jinja` (`tokenize=False`).
    /// The string is the joined segments, so a segment edit cannot drift from `render`.
    static func render(
        messages: [TemplateMessage],
        tools: [FlintOrderedJSON] = [],
        addGenerationPrompt: Bool = true
    ) -> String {
        promptSegments(messages: messages, tools: tools, addGenerationPrompt: addGenerationPrompt)
            .map(\.text)
            .joined()
    }

    /// Template specials are their own segments. Text the template copies
    /// from the message stays plain, including a special spelling.
    static func promptSegments(
        messages: [TemplateMessage],
        tools: [FlintOrderedJSON] = [],
        addGenerationPrompt: Bool = true
    ) -> [PromptSegment] {
        var segments: [PromptSegment] = []
        func plain(_ text: String) {
            guard !text.isEmpty else { return }
            if case .plain(let existing) = segments.last {
                segments[segments.count - 1] = .plain(existing + text)
            } else {
                segments.append(.plain(text))
            }
        }
        func special(_ token: String) {
            segments.append(.special(token))
        }
        var toolsPending = true
        for message in messages {
            if toolsPending && message.role != "system" {
                segments.append(contentsOf: toolPreambleSegments(tools))
                toolsPending = false
            }
            switch message.role {
            case "system":
                special("<|system|>")
                plain(message.content)
            case "developer":
                special("<|developer|>")
                plain(message.content)
            case "user":
                special("<|user|>")
                plain(message.content)
            case "assistant":
                special("<|assistant|>")
                if let reasoning = message.reasoning, !reasoning.isEmpty {
                    special("<think>")
                    plain(reasoning)
                    special("</think>")
                }
                plain(message.content)
                for call in message.toolCalls {
                    special("<tool_call>")
                    plain(call.name)
                    for (key, value) in call.arguments {
                        let shown: String
                        if case .string(let raw) = value {
                            shown = raw
                        } else {
                            shown = FlintJSONText.python(value)
                        }
                        special("<arg_key>")
                        plain(key)
                        special("</arg_key>")
                        special("<arg_value>")
                        plain(shown)
                        special("</arg_value>")
                    }
                    special("</tool_call>")
                }
            case "tool":
                special("<|observation|>")
                special("<tool_response>")
                plain(message.content)
                special("</tool_response>")
            default:
                break
            }
        }
        if toolsPending {
            segments.append(contentsOf: toolPreambleSegments(tools))
        }
        if addGenerationPrompt {
            special("<|assistant|>")
        }
        return segments
    }

    private static func toolPreambleSegments(_ tools: [FlintOrderedJSON]) -> [PromptSegment] {
        guard !tools.isEmpty else { return [] }
        // `{%-` strips the newline before each tag, so the package template is
        // `<tools>\n{json}\n</tools>` and one newline after the example call.
        // Adjacent plains are merged so a BPE encode of the region is one shot.
        var segments: [PromptSegment] = []
        func plain(_ text: String) {
            guard !text.isEmpty else { return }
            if case .plain(let existing) = segments.last {
                segments[segments.count - 1] = .plain(existing + text)
            } else {
                segments.append(.plain(text))
            }
        }
        func special(_ token: String) { segments.append(.special(token)) }
        special("<|system|>")
        plain("\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within ")
        special("<tools>")
        special("</tools>")
        plain(" XML tags:\n")
        special("<tools>")
        for tool in tools {
            let function = tool.field("function") ?? tool
            plain("\n" + FlintJSONText.python(function) + "\n")
        }
        special("</tools>")
        plain("\n\nFor each function call, output the function name and arguments within the following XML format:\n")
        special("<tool_call>")
        plain("{function-name}")
        special("<arg_key>")
        plain("{arg-key-1}")
        special("</arg_key>")
        special("<arg_value>")
        plain("{arg-value-1}")
        special("</arg_value>")
        plain("...")
        special("</tool_call>")
        plain("\n")
        return segments
    }

    static func fixtureSpecialID(_ token: String) -> Int? {
        guard let ordinal = fixtureSpecials.firstIndex(of: token) else { return nil }
        return ordinal + 1
    }

    /// Codepoint encoder used by `fixtureTokenIDs` for non-special text.
    static func fixturePlain(_ text: String) -> [Int] {
        text.unicodeScalars.map { 100 + Int($0.value) }
    }

    /// Fixture tokenizer from the alignment test: specials are 1-based in
    /// `specials` order, every other scalar is `100 + codepoint`. Longest
    /// special wins. This is the lock against `engine.prompt`, not the Flint BPE.
    static let fixtureSpecials = [
        "<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>",
        "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>",
        "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
        "<arg_value>", "</arg_value>", "<tools>", "</tools>",
        "<tool_response>", "</tool_response>",
    ]

    static func fixtureTokenIDs(_ text: String) -> [Int] {
        let specials = fixtureSpecials.sorted { $0.count > $1.count }
        var ids: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            var matched: String?
            for special in specials where text[index...].hasPrefix(special) {
                matched = special
                break
            }
            if let matched, let ordinal = fixtureSpecials.firstIndex(of: matched) {
                ids.append(ordinal + 1)
                index = text.index(index, offsetBy: matched.count)
            } else {
                let scalar = text[index].unicodeScalars.first!
                ids.append(100 + Int(scalar.value))
                index = text.index(after: index)
            }
        }
        return ids
    }
}
