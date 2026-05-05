import Foundation

public struct ThinkingContent: Sendable {
    public let thinking: String
    public let response: String

    public init(thinking: String, response: String) {
        self.thinking = thinking
        self.response = response
    }
}

public enum ParsedTokenType: Sendable {
    case thinking
    case content
}

public struct ParsedToken: Sendable {
    public let text: String
    public let type: ParsedTokenType

    public init(text: String, type: ParsedTokenType) {
        self.text = text
        self.type = type
    }
}

/// Streaming state machine that parses `<think>...</think>` blocks from token streams.
///
/// Handles two patterns:
/// - **Explicit**: `<think>` ... `</think>` — both tags in generated output
/// - **Implicit**: `</think>` only — `<think>` was injected by chat template prompt prefix
///
/// In the implicit case, all content before the first `</think>` is treated as thinking.
/// Content after a close tag is treated as response until a new open tag.
public final class ThinkingParser: @unchecked Sendable {
    private var buffer: String
    private var state: State
    private var thinkingContent: String
    private var responseContent: String
    private var preCloseContent: String
    private var implicitCloseTag: Bool
    private var closeWasSeen: Bool
    /// Sticky version of `closeWasSeen` — once a `</think>` has been observed,
    /// stays `true` for the lifetime of the parser. Used by the `finalize()`
    /// fallback to distinguish "model truncated mid-thinking, no close ever
    /// arrived" from "model emitted explicit close but no response content
    /// after it". The first case wants to promote thinking → response (so the
    /// user gets *some* answer); the second case wants to keep them separate
    /// so a properly-terminated thinking block stays in `reasoning_content`.
    private var sawExplicitCloseEver: Bool
    private var expectImplicitThinking: Bool

    private enum State: Sendable {
        case normal
        case potentialOpenTag
        case insideThinking
        case potentialCloseTag
    }

    private static let thinkOpen = "<think"
    private static let thinkClose = "</think"
    private static let bufferRetention = max(thinkOpen.count, thinkClose.count)
    private static let flushThreshold = 500

    public init(expectImplicitThinking: Bool = false) {
        self.buffer = ""
        self.state = .normal
        self.thinkingContent = ""
        self.responseContent = ""
        self.preCloseContent = ""
        self.implicitCloseTag = false
        self.closeWasSeen = false
        self.sawExplicitCloseEver = false
        self.expectImplicitThinking = expectImplicitThinking
    }

    public func reset() {
        buffer = ""
        state = .normal
        thinkingContent = ""
        responseContent = ""
        preCloseContent = ""
        implicitCloseTag = false
        closeWasSeen = false
        sawExplicitCloseEver = false
        // expectImplicitThinking is preserved across resets
    }

    /// Feed a streaming token into the parser.
    ///
    /// Content in `.normal` state is buffered until a tag is found or the buffer
    /// exceeds the flush threshold. This ensures correct typing for the implicit
    /// open tag case where `</think>` appears without a preceding `<think>`.
    public func feed(_ token: String) -> ParsedToken {
        buffer += token

        // Normalize alternative thinking markers (Qwen3.6+)
        buffer = buffer
            .replacingOccurrences(of: "<|begin_of_thought|>", with: "<think/>")
            .replacingOccurrences(of: "<|end_of_thought|>", with: "</think/>")

        var resultText = ""
        var resultType: ParsedTokenType = .content

        loop: while true {
            switch state {
            case .normal:
                // Check close tag first — handles implicit open tag case where
                // chat template injects <think> in prompt so only </think> is generated
                if let closeRange = buffer.range(of: Self.thinkClose) {
                    var beforeClose = String(buffer[buffer.startIndex..<closeRange.lowerBound])
                    // Explicit case: a single feed may contain the WHOLE <think>...</think>
                    // block (e.g. when an upstream scrubber normalized a non-standard
                    // marker into <think>...</think> in one shot). Strip the literal
                    // <think> open tag so it doesn't leak into reasoning_content.
                    // Implicit thinking models never emit the open tag, so this is a no-op.
                    if !expectImplicitThinking,
                       let openRange = beforeClose.range(of: Self.thinkOpen) {
                        let afterOpen = beforeClose[openRange.upperBound...]
                        if let gtIdx = afterOpen.firstIndex(of: ">") {
                            // Any text BEFORE the open tag is response content
                            let preOpen = String(beforeClose[..<openRange.lowerBound])
                            if !preOpen.isEmpty {
                                responseContent += preOpen
                            }
                            beforeClose = String(afterOpen[afterOpen.index(after: gtIdx)...])
                        }
                    }
                    let thinkingText = preCloseContent + beforeClose
                    if !thinkingText.isEmpty {
                        thinkingContent += thinkingText
                        resultText += thinkingText
                        resultType = .thinking
                    }
                    preCloseContent = ""
                    let afterClose = String(buffer[closeRange.upperBound...])
                    if let gt = afterClose.firstIndex(of: ">") {
                        buffer = String(afterClose[afterClose.index(gt, offsetBy: 1)...])
                        state = .normal
                    } else {
                        buffer = afterClose
                        state = .potentialCloseTag
                        implicitCloseTag = true
                    }
                    closeWasSeen = true
                    sawExplicitCloseEver = true
                    // If we just emitted thinking and the buffer still has content,
                    // break out so the caller gets a thinking-typed ParsedToken now.
                    // The remaining content is processed on the next feed() call (or
                    // flushed by finalize()) and emitted as a separate content-typed
                    // ParsedToken. Without this, thinking and content would be
                    // concatenated into one ParsedToken with the wrong (last) type.
                    if resultType == .thinking && !buffer.isEmpty {
                        break loop
                    }
                } else if let openRange = buffer.range(of: Self.thinkOpen) {
                    // Explicit open tag — flush preCloseContent as response
                    let beforeOpen = String(buffer[buffer.startIndex..<openRange.lowerBound])
                    if !preCloseContent.isEmpty || !beforeOpen.isEmpty {
                        let text = preCloseContent + beforeOpen
                        responseContent += text
                        resultText += text
                        resultType = .content
                    }
                    preCloseContent = ""
                    let afterOpen = String(buffer[openRange.upperBound...])
                    buffer = afterOpen
                    if let closeAngle = afterOpen.firstIndex(of: ">") {
                        buffer = String(afterOpen[afterOpen.index(closeAngle, offsetBy: 1)...])
                        state = .insideThinking
                    } else {
                        state = .potentialOpenTag
                    }
                    closeWasSeen = false
                } else if closeWasSeen {
                    // After a close tag we know type is .content — yield everything.
                    // No partial buffering needed since we aren't expecting another tag.
                    if !buffer.isEmpty {
                        responseContent += buffer
                        resultText += buffer
                        resultType = .content
                        buffer = ""
                    }
                    break loop
                } else {
                    let safeEnd = max(0, buffer.count - Self.bufferRetention)
                    if safeEnd > 0 {
                        if expectImplicitThinking && !closeWasSeen {
                            // Implicit thinking model: stream tokens as thinking immediately
                            // rather than buffering — enables real-time thinking visualization
                            let safe = String(buffer.prefix(safeEnd))
                            thinkingContent += safe
                            resultText += safe
                            resultType = .thinking
                            buffer = String(buffer.suffix(Self.bufferRetention))
                        } else {
                            // Buffer into preCloseContent — if </think arrives later,
                            // this content is reclassified as thinking. Only flush as
                            // response when exceeding threshold (safe to commit).
                            preCloseContent += String(buffer.prefix(safeEnd))
                            buffer = String(buffer.suffix(Self.bufferRetention))

                            // Flush preCloseContent as response when it exceeds threshold
                            // and no close tag has appeared — content is confirmed response
                            if preCloseContent.count > Self.flushThreshold {
                                responseContent += preCloseContent
                                resultText += preCloseContent
                                resultType = .content
                                preCloseContent = ""
                            }
                        }
                    }
                    break loop
                }

            case .potentialOpenTag:
                if let closeAngle = buffer.firstIndex(of: ">") {
                    buffer = String(buffer[buffer.index(closeAngle, offsetBy: 1)...])
                    state = .insideThinking
                    closeWasSeen = false
                } else if buffer.count > 100 {
                    // Not actually a think tag — flush as thinking content
                    thinkingContent += buffer
                    resultText += buffer
                    resultType = .thinking
                    buffer = ""
                    state = .insideThinking
                    break loop
                } else {
                    break loop
                }

            case .insideThinking:
                if let closeRange = buffer.range(of: Self.thinkClose) {
                    let before = String(buffer[buffer.startIndex..<closeRange.lowerBound])
                    if !before.isEmpty {
                        thinkingContent += before
                        resultText += before
                        resultType = .thinking
                    }
                    let afterClose = String(buffer[closeRange.upperBound...])
                    buffer = afterClose
                    if let closeAngle = afterClose.firstIndex(of: ">") {
                        buffer = String(afterClose[afterClose.index(closeAngle, offsetBy: 1)...])
                        state = .normal
                        closeWasSeen = true
                        sawExplicitCloseEver = true
                    } else {
                        state = .potentialCloseTag
                    }
                } else {
                    let safeEnd = max(0, buffer.count - Self.thinkClose.count)
                    if safeEnd > 0 {
                        let safe = String(buffer.prefix(safeEnd))
                        thinkingContent += safe
                        resultText += safe
                        resultType = .thinking
                        buffer = String(buffer.suffix(Self.thinkClose.count))
                    }
                    break loop
                }

            case .potentialCloseTag:
                if let closeAngle = buffer.firstIndex(of: ">") {
                    buffer = String(buffer[buffer.index(closeAngle, offsetBy: 1)...])
                    state = .normal
                    implicitCloseTag = false
                    closeWasSeen = true
                    sawExplicitCloseEver = true
                } else if buffer.count > 1 {
                    thinkingContent += Self.thinkClose + buffer
                    resultText += Self.thinkClose + buffer
                    resultType = .thinking
                    buffer = ""
                    state = implicitCloseTag ? .normal : .insideThinking
                    if implicitCloseTag { closeWasSeen = true }
                    implicitCloseTag = false
                } else {
                    break loop
                }
            }
        }

        return ParsedToken(text: resultText, type: resultType)
    }

    public func finalize() -> ThinkingContent {
        // Flush preCloseContent — if no tag was ever seen, it's response content
        if !preCloseContent.isEmpty {
            responseContent += preCloseContent
            preCloseContent = ""
        }

        switch state {
        case .insideThinking, .potentialOpenTag:
            thinkingContent += buffer
        case .potentialCloseTag:
            thinkingContent += Self.thinkClose + buffer
        case .normal:
            responseContent += buffer
        }
        buffer = ""
        state = .normal
        implicitCloseTag = false
        closeWasSeen = false

        // Fallback: if model opened thinking but never closed it and produced
        // no response content, the answer is embedded in thinking. Move it to
        // response so the user gets it in `content` instead of `reasoning_content`.
        // Only applies for explicit-thinking models (not implicit like DeepSeek-R1).
        //
        // **Sticky-close guard:** when an explicit `</think>` was observed at
        // any point, the model successfully terminated the thinking block —
        // empty response content after that means the model was cut off
        // mid-final-answer (e.g. max_tokens), not that thinking is the answer.
        // Promoting in that case would corrupt `reasoning_content` (e.g. the
        // gpt-oss / Harmony scenario where channel boundaries are normalized
        // to `<think>...</think>` upstream).
        if !expectImplicitThinking
            && responseContent.isEmpty
            && !thinkingContent.isEmpty
            && !sawExplicitCloseEver {
            responseContent = thinkingContent
            thinkingContent = ""
        }

        return ThinkingContent(thinking: thinkingContent, response: responseContent)
    }

    public var currentThinkingContent: String { thinkingContent }
    public var currentResponseContent: String { responseContent }
    public var isInThinkingBlock: Bool {
        state == .insideThinking || state == .potentialOpenTag || (implicitCloseTag && preCloseContent.isEmpty)
    }
}
