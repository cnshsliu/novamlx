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

public final class ThinkingParser: @unchecked Sendable {
    private var buffer: String
    private var state: State
    private var thinkingContent: String
    private var responseContent: String

    private enum State: Sendable {
        case normal
        case potentialOpenTag
        case insideThinking
        case potentialCloseTag
    }

    private static let thinkOpen = "<think"
    private static let thinkClose = "</think"

    public init() {
        self.buffer = ""
        self.state = .normal
        self.thinkingContent = ""
        self.responseContent = ""
    }

    public func reset() {
        buffer = ""
        state = .normal
        thinkingContent = ""
        responseContent = ""
    }

    public func feed(_ token: String) -> ParsedToken {
        buffer += token

        var resultText = ""
        var resultType: ParsedTokenType = .content

        loop: while true {
            switch state {
            case .normal:
                if let openRange = buffer.range(of: Self.thinkOpen) {
                    let before = String(buffer[buffer.startIndex..<openRange.lowerBound])
                    if !before.isEmpty {
                        responseContent += before
                        resultText += before
                        resultType = .content
                    }
                    let afterOpen = String(buffer[openRange.upperBound...])
                    buffer = afterOpen
                    if let closeAngle = afterOpen.firstIndex(of: ">") {
                        buffer = String(afterOpen[afterOpen.index(closeAngle, offsetBy: 1)...])
                        state = .insideThinking
                    } else {
                        state = .potentialOpenTag
                    }
                } else {
                    let safeEnd = max(0, buffer.count - Self.thinkOpen.count)
                    if safeEnd > 0 {
                        let safe = String(buffer.prefix(safeEnd))
                        responseContent += safe
                        resultText += safe
                        resultType = .content
                        buffer = String(buffer.suffix(Self.thinkOpen.count))
                    }
                    break loop
                }

            case .potentialOpenTag:
                if let closeAngle = buffer.firstIndex(of: ">") {
                    buffer = String(buffer[buffer.index(closeAngle, offsetBy: 1)...])
                    state = .insideThinking
                } else if buffer.count > 100 {
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
                } else if buffer.count > 1 {
                    thinkingContent += Self.thinkClose
                    resultText += Self.thinkClose
                    resultType = .thinking
                    thinkingContent += buffer
                    resultText += buffer
                    buffer = ""
                    state = .insideThinking
                    break loop
                } else {
                    break loop
                }
            }
        }

        return ParsedToken(text: resultText, type: resultType)
    }

    public func finalize() -> ThinkingContent {
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
        return ThinkingContent(thinking: thinkingContent, response: responseContent)
    }

    public var currentThinkingContent: String { thinkingContent }
    public var currentResponseContent: String { responseContent }
    public var isInThinkingBlock: Bool { state == .insideThinking || state == .potentialOpenTag }
}
