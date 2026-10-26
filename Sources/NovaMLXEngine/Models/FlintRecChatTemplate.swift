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
    static func render(
        messages: [TemplateMessage],
        tools: [FlintOrderedJSON] = [],
        addGenerationPrompt: Bool = true
    ) -> String {
        var out = ""
        var toolsPending = true
        for message in messages {
            if toolsPending && message.role != "system" {
                out += toolPreamble(tools)
                toolsPending = false
            }
            switch message.role {
            case "system":
                out += "<|system|>" + message.content
            case "developer":
                out += "<|developer|>" + message.content
            case "user":
                out += "<|user|>" + message.content
            case "assistant":
                out += "<|assistant|>"
                if let reasoning = message.reasoning, !reasoning.isEmpty {
                    out += "<think>" + reasoning + "</think>"
                }
                if !message.content.isEmpty {
                    out += message.content
                }
                for call in message.toolCalls {
                    out += "<tool_call>" + call.name
                    for (key, value) in call.arguments {
                        let shown: String
                        if case .string(let raw) = value {
                            shown = raw
                        } else {
                            shown = FlintJSONText.python(value)
                        }
                        out += "<arg_key>\(key)</arg_key><arg_value>\(shown)</arg_value>"
                    }
                    out += "</tool_call>"
                }
            case "tool":
                out += "<|observation|><tool_response>" + message.content + "</tool_response>"
            default:
                break
            }
        }
        if toolsPending {
            out += toolPreamble(tools)
        }
        if addGenerationPrompt {
            out += "<|assistant|>"
        }
        return out
    }

    private static func toolPreamble(_ tools: [FlintOrderedJSON]) -> String {
        guard !tools.isEmpty else { return "" }
        // `{%-` strips the newline before each tag, so the package template is
        // `<tools>\n{json}\n</tools>` and one newline after the example call.
        var body = "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        for tool in tools {
            let function = tool.field("function") ?? tool
            body += "\n"
            body += FlintJSONText.python(function)
            body += "\n"
        }
        body += "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>...</tool_call>\n"
        return body
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
