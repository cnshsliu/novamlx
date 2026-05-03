import Foundation

// MARK: - Shared Control Token Logic
// Extracted from MLXEngine.swift — shared across all processors.

enum SharedControlTokenLogic {

    // Thinking markers excluded from scrubbing — ThinkingParser handles these.
    static let thinkingMarkerWords: Set<String> = [
        "begin_of_thought", "end_of_thought",
        "begin_of_think", "end_of_think",
        "think", "/think",
        "thinking", "/thinking",
    ]

    // Regex: matches <|xxx|>, <|xxx>, <|xxx\n, <|xxx\s patterns
    static let controlTokenRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "<\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>|\\n|\\s)")
    }()

    /// Regex-based scrub: strip control tokens from output text.
    /// Excludes thinking markers (handled by ThinkingParser).
    static func scrubControlTokens(_ text: String) -> String {
        let nsRange = NSRange(text.startIndex..., in: text)
        let matches = controlTokenRegex.matches(in: text, range: nsRange)
        guard !matches.isEmpty else { return text }
        var result = text
        for match in matches.reversed() {
            if let range = Range(match.range, in: result) {
                let matched = String(result[range])
                let isThinkingMarker = thinkingMarkerWords.contains { matched.contains($0) }
                if !isThinkingMarker {
                    result.removeSubrange(range)
                }
            }
        }
        return result
    }

    /// Truncate text at first control token pattern match.
    static func trimControlTokens(_ text: String, patterns: [String]) -> String {
        guard !patterns.isEmpty else { return text }
        var result = text
        for pattern in patterns {
            if let range = result.range(of: pattern) {
                result = String(result[..<range.lowerBound])
            }
        }
        return result
    }

    /// Incremental streaming filter. Returns (cleanText, shouldStop).
    static func filterControlInChunk(
        _ text: String,
        accumulated: inout String,
        yieldedCount: inout Int,
        patterns: [String]
    ) -> (String, Bool) {
        accumulated += text
        let totalLen = accumulated.count

        // 1. Full control token match — trim and stop
        for pattern in patterns {
            if let range = accumulated.range(of: pattern) {
                let safeEnd = accumulated.distance(from: accumulated.startIndex, to: range.lowerBound)
                let cleanText: String
                if safeEnd > yieldedCount {
                    cleanText = String(accumulated.dropFirst(yieldedCount).prefix(safeEnd - yieldedCount))
                } else {
                    cleanText = ""
                }
                yieldedCount = totalLen
                return (cleanText, true)
            }
        }

        // 2. Partial prefix check — buffer tail that could be start of a control token
        let maxLen = patterns.map(\.count).max() ?? 0
        let checkLen = min(totalLen, maxLen)
        var safeEnd = totalLen
        if checkLen > 0 {
            for i in 1...checkLen {
                let suffixStart = accumulated.index(accumulated.endIndex, offsetBy: -i)
                let suffix = accumulated[suffixStart...]
                for pattern in patterns {
                    if pattern.hasPrefix(suffix) {
                        safeEnd = min(safeEnd, totalLen - i)
                    }
                }
            }
        }

        if safeEnd > yieldedCount {
            let cleanText = String(accumulated.dropFirst(yieldedCount).prefix(safeEnd - yieldedCount))
            yieldedCount = safeEnd
            return (cleanText, false)
        }
        return ("", false)
    }

    /// Generate close-tag variants: <|turn|> → <|/turn|>, <|/turn>
    static func generateCloseVariants(_ tokens: Set<String>) -> Set<String> {
        var closeVariants = Set<String>()
        for token in tokens {
            guard token.hasPrefix("<|") && !token.contains("/") else { continue }
            let inner: String
            if token.hasSuffix("|>") {
                inner = String(token.dropFirst(2).dropLast(2))
            } else if token.hasSuffix(">") {
                inner = String(token.dropFirst(2).dropLast(1))
            } else { continue }
            closeVariants.insert("<|/\(inner)|>")
            closeVariants.insert("<|/\(inner)>")
        }
        return closeVariants
    }

    /// Extract <|xxx|> and <|xxx> patterns from chat template.
    static func extractTemplateTokens(from template: String) -> Set<String> {
        var tokens = Set<String>()
        if let regex = try? NSRegularExpression(pattern: "<\\|[a-zA-Z_/][a-zA-Z0-9_/]*\\|?>") {
            let nsRange = NSRange(template.startIndex..., in: template)
            for match in regex.matches(in: template, range: nsRange) {
                if let range = Range(match.range, in: template) {
                    tokens.insert(String(template[range]))
                }
            }
        }
        return tokens
    }

    /// Extract special tokens from tokenizer.json added_tokens.
    static func extractAddedTokens(from entries: [[String: Any]]) -> Set<String> {
        var tokens = Set<String>()
        for entry in entries {
            if entry["special"] as? Bool == true,
               let content = entry["content"] as? String,
               content.hasPrefix("<") {
                tokens.insert(content)
            }
            // Also pick up <|xxx|> pattern tokens marked non-special
            if entry["special"] as? Bool == false,
               let content = entry["content"] as? String,
               content.hasPrefix("<|") && (content.hasSuffix("|>") || content.hasSuffix(">")) {
                tokens.insert(content)
            }
        }
        return tokens
    }

    /// Semantic content tags — parsed by ThinkingParser, not stream delimiters.
    static let semanticTags: Set<String> = [
        "<think", "</think",
        "<thinking", "</thinking",
        "<|begin_of_thought|>", "<|end_of_thought|>",
        "<|begin_of_think|>", "<|end_of_think|>",
    ]
}
