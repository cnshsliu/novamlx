import Foundation
import NovaMLXCore
import NovaMLXUtils

/// Read-only diagnostics for chat-template processing for a specific model.
/// Used by the `nova chat-template diagnose <modelId>` CLI command and the
/// `/admin/api/chat-template/diagnose` HTTP endpoint to surface everything
/// that influences the model's prompt rendering and output processing.
public enum ChatTemplateDiagnostics {

    public struct Report: Codable, Sendable {
        public let modelId: String
        public let modelDir: String?
        public let family: String
        public let architectures: [String]
        public let modelType: String?

        // Template files on disk
        public let tokenizerConfigChatTemplateLength: Int
        public let tokenizerConfigChatTemplateExcerpt: String
        public let chatTemplateJinjaPresent: Bool
        public let chatTemplateJinjaLength: Int
        public let chatTemplateJinjaQuarantined: Bool
        public let chatTemplateJsonPresent: Bool
        public let templatesAgree: Bool

        // Format detection
        public let detectedFormats: [DetectionEntry]

        // Registry override
        public let registryOverrideName: String?

        // Family interpretation
        public let familyConfigLabel: String
        public let expectedEosTokens: [String]
        public let hallucinationPatterns: [String]
        public let leakageBlacklist: [String]
        public let thinkingMarkers: [String]
        public let preserveThinkingSupported: Bool

        // Health summary
        public let healthIssues: [String]
        public let healthy: Bool

        public struct DetectionEntry: Codable, Sendable {
            public let format: String
            public let confidence: Int
            public let evidence: [String]
        }
    }

    /// Build a diagnostic report for `modelId`. `modelsDir` is the parent
    /// directory under which model dirs live (typically NovaMLXPaths.modelsDir).
    public static func diagnose(modelId: String, modelsDir: URL, family: ModelFamily) -> Report {
        let fm = FileManager.default
        let modelDir = modelsDir.appendingPathComponent(modelId)
        let dirExists = fm.fileExists(atPath: modelDir.path)

        // Architecture & model_type from config.json
        var architectures: [String] = []
        var modelType: String? = nil
        let configPath = modelDir.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            architectures = (json["architectures"] as? [String]) ?? []
            modelType = json["model_type"] as? String
        }

        // tokenizer_config.json
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        var tcTemplate: String = ""
        if let data = try? Data(contentsOf: tcPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let t = json["chat_template"] as? String {
            tcTemplate = t
        }

        // chat_template.jinja
        let jinjaPath = modelDir.appendingPathComponent("chat_template.jinja")
        let quarantinePath = modelDir.appendingPathComponent("chat_template.jinja.disabled-by-novamlx")
        let jinjaPresent = fm.fileExists(atPath: jinjaPath.path)
        let jinjaContent: String = jinjaPresent
            ? ((try? String(contentsOf: jinjaPath, encoding: .utf8)) ?? "")
            : ""
        let jinjaQuarantined = fm.fileExists(atPath: quarantinePath.path)

        // chat_template.json (legacy)
        let jsonPath = modelDir.appendingPathComponent("chat_template.json")
        let chatTemplateJsonPresent = fm.fileExists(atPath: jsonPath.path)

        // Determine which template the rendering pipeline will actually use.
        // swift-transformers prefers .jinja over tokenizer_config when both
        // exist with content, so we report agreement state.
        let agree: Bool = {
            guard !tcTemplate.isEmpty, !jinjaContent.isEmpty else { return true }
            return tcTemplate == jinjaContent
        }()

        // Format detection — run on the canonical template (whatever swift-transformers will pick)
        let templateForDetection: String
        if !jinjaContent.isEmpty {
            templateForDetection = jinjaContent
        } else {
            templateForDetection = tcTemplate
        }
        let detected = ChatTemplateFormat.detectAll(from: templateForDetection)
        let detectedEntries = detected.map { d in
            Report.DetectionEntry(format: d.format.label, confidence: d.confidence, evidence: d.evidence)
        }

        // Registry override
        let regOverride = ChatTemplateRegistry.shared.templateOverride(
            modelId: modelId,
            family: family,
            architecture: architectures.first
        )

        // Family config
        let famCfg = ChatTemplateRegistry.shared.familyConfig(for: family)

        // Health analysis
        var issues: [String] = []
        if !dirExists {
            issues.append("model directory does not exist at \(modelDir.path)")
        }
        if tcTemplate.isEmpty && jinjaContent.isEmpty {
            issues.append("no chat template found (tokenizer_config.json AND chat_template.jinja both missing/empty)")
        }
        if !tcTemplate.isEmpty && !jinjaContent.isEmpty && tcTemplate != jinjaContent {
            issues.append("tokenizer_config.json.chat_template (\(tcTemplate.count) chars) and chat_template.jinja (\(jinjaContent.count) chars) DISAGREE — swift-transformers will use the .jinja, which may corrupt the prompt format")
        }
        if detected.count > 1 {
            let formats = detected.map { "\($0.format.label) (×\($0.confidence))" }.joined(separator: ", ")
            issues.append("multiple format markers detected in template: \(formats) — likely corruption (one family format expected)")
        }
        if detected.isEmpty && !templateForDetection.isEmpty {
            issues.append("template format unknown — no recognized markers (model may use a brand-new format)")
        }
        if templateForDetection.contains("{{") && templateForDetection.contains("}}") {
            // Sanity: every Jinja template has these. This is just a tautology check
            // unless someone wrote a template that's ALL static text with no rendering.
        }
        if !famCfg.expectedFormats.isEmpty && !detected.isEmpty {
            let actual = detected.first!.format.label
            if !famCfg.expectedFormats.contains(where: { actual.contains($0) || $0.contains(actual) }) {
                issues.append("detected format \(actual) not in family's expected list \(famCfg.expectedFormats) — possible mismatch")
            }
        }

        return Report(
            modelId: modelId,
            modelDir: dirExists ? modelDir.path : nil,
            family: family.rawValue,
            architectures: architectures,
            modelType: modelType,
            tokenizerConfigChatTemplateLength: tcTemplate.count,
            tokenizerConfigChatTemplateExcerpt: String(tcTemplate.prefix(400)),
            chatTemplateJinjaPresent: jinjaPresent,
            chatTemplateJinjaLength: jinjaContent.count,
            chatTemplateJinjaQuarantined: jinjaQuarantined,
            chatTemplateJsonPresent: chatTemplateJsonPresent,
            templatesAgree: agree,
            detectedFormats: detectedEntries,
            registryOverrideName: regOverride,
            familyConfigLabel: famCfg.label,
            expectedEosTokens: famCfg.expectedEosTokens,
            hallucinationPatterns: famCfg.hallucinationPatterns,
            leakageBlacklist: famCfg.leakageBlacklist,
            thinkingMarkers: famCfg.thinkingMarkers,
            preserveThinkingSupported: famCfg.preserveThinkingSupported,
            healthIssues: issues,
            healthy: issues.isEmpty
        )
    }
}
