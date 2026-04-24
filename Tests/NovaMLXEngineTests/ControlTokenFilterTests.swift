import Testing
import Foundation
import MLX
import MLXLMCommon
import NovaMLXCore
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// filterControlInChunk — streaming control token filter
// ────────────────────────────────────────────────────────────

@Suite("filterControlInChunk Tests")
struct FilterControlInChunkTests {

    // MARK: - Full match & stop

    @Test("Full control token match triggers stop")
    func fullMatchStop() {
        let patterns = ["<|turn|>", "<|im_end|>", "<|endoftext|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "hello<|turn|>", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "hello")
        #expect(shouldStop == true)
    }

    @Test("Full match at start of first chunk")
    func fullMatchAtStart() {
        let patterns = ["<|im_end|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "<|im_end|>", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "")
        #expect(shouldStop == true)
    }

    @Test("Multiple patterns — matches first found")
    func multiplePatternsFirstMatch() {
        let patterns = ["<|endoftext|>", "<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "some text<|turn|>more", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "some text")
        #expect(shouldStop == true)
    }

    @Test("Pattern not in patterns list passes through")
    func patternNotInList() {
        let patterns = ["<|turn|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "hello <|something_else|> world", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text.contains("hello"))
        #expect(shouldStop == false)
    }

    // MARK: - Partial prefix buffering

    @Test("Partial token prefix is buffered, not yielded")
    func partialPrefixBuffered() {
        let patterns = ["<|turn|>"]
        var acc = ""
        var yielded = 0

        // Feed "<|tur" — should be buffered since it could start <|turn|>
        let (text1, stop1) = MLXEngine.filterControlInChunk(
            "hello <|tur", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text1 == "hello ")
        #expect(stop1 == false)

        // Feed "n|>" — completes the control token
        let (text2, stop2) = MLXEngine.filterControlInChunk(
            "n|>", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text2 == "")
        #expect(stop2 == true)
    }

    @Test("Partial prefix that turns out to be regular text")
    func partialPrefixBecomesRegular() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        // Feed "<|tu" — could be start of <|turn|>
        let (text1, _) = MLXEngine.filterControlInChunk(
            "text <|tu", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text1 == "text ")
        // "<|tu" is buffered as potential prefix

        // Feed "rbo" — now it's "<|turbo" which doesn't match any pattern prefix
        let (text2, _) = MLXEngine.filterControlInChunk(
            "rbo", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        // The prefix was 7 chars max. "<|turbo" no longer matches any pattern prefix
        // so "<|tu" should be released
        #expect(!text2.isEmpty)
    }

    // MARK: - Yield tracking

    @Test("Yield tracking prevents re-emitting buffered content")
    func yieldTrackingNoDuplicate() {
        let patterns = ["<|turn|>"]
        var acc = ""
        var yielded = 0

        // Feed "abc" — yields "abc", yieldedCount becomes 3
        let (text1, _) = MLXEngine.filterControlInChunk(
            "abc", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text1 == "abc")

        // Feed "def" — should only yield "def", not "abcdef"
        let (text2, _) = MLXEngine.filterControlInChunk(
            "def", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text2 == "def")
    }

    @Test("Yield after partial prefix release")
    func yieldAfterPrefixRelease() {
        let patterns = ["<|turn|>"]
        var acc = ""
        var yielded = 0

        // "hello" → yielded
        let (t1, _) = MLXEngine.filterControlInChunk(
            "hello", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(t1 == "hello")

        // "<|tur" → partial prefix, buffered
        let (t2, _) = MLXEngine.filterControlInChunk(
            "<|tur", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(t2 == "")

        // "n|>rest" → control token matched, yields nothing, stops
        let (t3, stop) = MLXEngine.filterControlInChunk(
            "n|>rest", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(t3 == "")
        #expect(stop == true)
    }

    // MARK: - Edge cases

    @Test("Empty input returns empty")
    func emptyInput() {
        let patterns = ["<|turn|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "")
        #expect(shouldStop == false)
    }

    @Test("Empty patterns list passes everything through")
    func emptyPatterns() {
        let patterns: [String] = []
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "hello <|turn|> world", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "hello <|turn|> world")
        #expect(shouldStop == false)
    }

    @Test("Content with angle brackets but no pipe passes through")
    func angleBracketsNoPipe() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "x < 5 and y > 3", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "x < 5 and y > 3")
        #expect(shouldStop == false)
    }

    @Test("Exact control token with no surrounding text")
    func exactControlToken() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        let (text, shouldStop) = MLXEngine.filterControlInChunk(
            "<|turn|>", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "")
        #expect(shouldStop == true)
    }

    @Test("Multiple chunks building control token character by character")
    func charByCharControlToken() {
        let patterns = ["<|turn|>"]
        var acc = ""
        var yielded = 0
        let token = "<|turn|>"
        var finalStop = false
        var allText = ""

        for ch in token {
            let (text, stop) = MLXEngine.filterControlInChunk(
                String(ch), accumulated: &acc, yieldedCount: &yielded, patterns: patterns
            )
            allText += text
            if stop { finalStop = true; break }
        }
        #expect(allText == "")
        #expect(finalStop == true)
    }

    @Test("Longest pattern wins partial prefix check")
    func longestPatternPrefixCheck() {
        // <|turn|> is 8 chars, <|im_end|> is 10 chars
        let patterns = ["<|turn|>", "<|im_end|>"]
        var acc = ""
        var yielded = 0

        // Feed "<|im_e" — this matches prefix of "<|im_end|>" (10 chars) but not "<|turn|>"
        let (text, _) = MLXEngine.filterControlInChunk(
            "prefix<|im_e", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "prefix")
        // "<|im_e" should be buffered since it matches "<|im_end|>" prefix
    }

    @Test("Chunk that starts safe then hits control token mid-stream")
    func safeThenControlMidStream() {
        let patterns = ["<|im_end|>"]
        var acc = ""
        var yielded = 0

        let (text, stop) = MLXEngine.filterControlInChunk(
            "Here is the answer. <|im_end|>", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "Here is the answer. ")
        #expect(stop == true)
    }

    @Test("Control token without closing pipe — regex scrub catches it")
    func controlTokenNoClosingPipe() {
        let patterns = ["<|turn", "<|im_end"]
        var acc = ""
        var yielded = 0

        let (text, stop) = MLXEngine.filterControlInChunk(
            "text<|turn", accumulated: &acc, yieldedCount: &yielded, patterns: patterns
        )
        #expect(text == "text")
        #expect(stop == true)
    }
}

// ────────────────────────────────────────────────────────────
// scrubControlTokens — regex-based <|xxx|> scrub
// ────────────────────────────────────────────────────────────

@Suite("scrubControlTokens Tests")
struct ScrubControlTokensTests {

    @Test("Single <|xxx|> pattern removed")
    func singlePipePattern() {
        let result = MLXEngine.scrubControlTokens("hello <|mask_start|> world")
        #expect(result == "hello  world")
    }

    @Test("Single <|xxx> pattern without closing pipe removed")
    func singleNoPipePattern() {
        // Regex matches <|xxx> (closing pipe optional, closing > required)
        let result = MLXEngine.scrubControlTokens("text <|turn> more text")
        #expect(result == "text  more text")
    }

    @Test("Multiple control tokens removed")
    func multipleTokens() {
        let result = MLXEngine.scrubControlTokens("<|start|> hello <|tool_call|> world <|end|>")
        #expect(result == " hello  world ")
    }

    @Test("Adjacent control tokens both removed")
    func adjacentTokens() {
        let result = MLXEngine.scrubControlTokens("<|start|><|tool_call|>text")
        #expect(result == "text")
    }

    @Test("No control tokens passes through unchanged")
    func noTokens() {
        let input = "Hello world! No special tokens here."
        let result = MLXEngine.scrubControlTokens(input)
        #expect(result == input)
    }

    @Test("Empty string passes through")
    func emptyString() {
        let result = MLXEngine.scrubControlTokens("")
        #expect(result == "")
    }

    @Test("Only control token, nothing else")
    func onlyControlToken() {
        let result = MLXEngine.scrubControlTokens("<|turn|>")
        #expect(result == "")
    }

    @Test("Numeric in token name")
    func numericToken() {
        let result = MLXEngine.scrubControlTokens("model <|im_end|> response")
        #expect(result == "model  response")
    }

    @Test("Underscore in token name")
    func underscoreToken() {
        let result = MLXEngine.scrubControlTokens("text <|tool_call_id|> more")
        #expect(result == "text  more")
    }

    @Test("Angle brackets without pipe not scrubbed")
    func angleBracketsNoPipeNotScrubbed() {
        let input = "if x < 10 then print('ok')"
        let result = MLXEngine.scrubControlTokens(input)
        #expect(result == input)
    }
}

// ────────────────────────────────────────────────────────────
// trimControlTokens — cut at first control token occurrence
// ────────────────────────────────────────────────────────────

@Suite("trimControlTokens Tests")
struct TrimControlTokensTests {

    @Test("Trim at first pattern occurrence")
    func trimAtFirstPattern() {
        let patterns = ["<|turn|>", "<|im_end|>"]
        let result = MLXEngine.trimControlTokens(
            "hello world<|turn|>garbage after", patterns: patterns
        )
        #expect(result == "hello world")
    }

    @Test("Trim at first of multiple patterns")
    func trimAtFirstOfMultiple() {
        let patterns = ["<|endoftext|>", "<|turn|>", "<|im_end|>"]
        let result = MLXEngine.trimControlTokens(
            "text <|turn|> more <|im_end|>", patterns: patterns
        )
        #expect(result == "text ")
    }

    @Test("No pattern match returns full text")
    func noMatchReturnsFullText() {
        let patterns = ["<|turn|>"]
        let result = MLXEngine.trimControlTokens("clean text here", patterns: patterns)
        #expect(result == "clean text here")
    }

    @Test("Empty patterns returns full text")
    func emptyPatternsReturnsFull() {
        let result = MLXEngine.trimControlTokens("anything <|turn|> here", patterns: [])
        #expect(result == "anything <|turn|> here")
    }

    @Test("Pattern at very start returns empty")
    func patternAtStart() {
        let patterns = ["<|turn|>"]
        let result = MLXEngine.trimControlTokens("<|turn|>rest", patterns: patterns)
        #expect(result == "")
    }

    @Test("Pattern at end trims correctly")
    func patternAtEnd() {
        let patterns = ["<|im_end|>"]
        let result = MLXEngine.trimControlTokens("complete text<|im_end|>", patterns: patterns)
        #expect(result == "complete text")
    }
}
