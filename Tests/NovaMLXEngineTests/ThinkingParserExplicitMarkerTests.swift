import Testing
import Foundation
import NovaMLXCore
import NovaMLXUtils
@testable import NovaMLXEngine

// ────────────────────────────────────────────────────────────
// Qwen3.6 (and any future model) explicit-thinking-marker
// regression suite — todo.markdown §2.11.
//
// Pins three pieces of behavior that §1's T1–T4 only validate
// end-to-end with a live model:
//
//   1. `ThinkingParser` (NovaMLXUtils/ThinkingParser.swift:98-101)
//      normalizes `<|begin_of_thought|>` → `<think/>` and
//      `<|end_of_thought|>` → `</think/>` and partitions the surrounding
//      text into thinking vs. response.
//
//   2. `ModelContainer.isImplicitThinkingModel(for:)`
//      (NovaMLXEngine/MLXEngine.swift:260-291) returns `false` for a
//      Qwen3.6-style fixture (chat template contains `<think` AND
//      tokenizer.json `added_tokens` declares `<|begin_of_thought|>`)
//      and `true` for a DeepSeek-R1-style fixture (template-only,
//      no explicit marker tokens).
//
//   3. `enable_thinking=false` request path in
//      `APIServer.swift:1738-1761` strips `<think>...</think>` markers
//      from the visible response.
//
// Without these pins, a regex tweak in `ThinkingParser.feed` or a
// detection-flag flip in `isImplicitThinkingModel` would only surface
// at API integration time on a real machine with the real model
// downloaded — which is exactly the situation §1 was investigating.
// ────────────────────────────────────────────────────────────

@Suite("ThinkingParser explicit-marker regression (Qwen3.6)")
struct ThinkingParserExplicitMarkerTests {

    // MARK: - 1. Single-shot

    @Test("Single-shot: <|begin_of_thought|>X<|end_of_thought|>Y → thinking=X, response=Y")
    func singleShotExplicit() {
        let parser = ThinkingParser(expectImplicitThinking: false)
        _ = parser.feed("<|begin_of_thought|>reasoning<|end_of_thought|>answer")
        let result = parser.finalize()
        #expect(result.thinking == "reasoning",
            "thinking should equal the text between begin/end markers; got '\(result.thinking)'")
        #expect(result.response == "answer",
            "response should equal the text after the close marker; got '\(result.response)'")
    }

    @Test("Single-shot: marker normalization is independent of surrounding whitespace")
    func singleShotWithWhitespace() {
        let parser = ThinkingParser(expectImplicitThinking: false)
        _ = parser.feed("<|begin_of_thought|>\nfirst line\nsecond line\n<|end_of_thought|>\n\nFinal.")
        let result = parser.finalize()
        #expect(result.thinking.contains("first line"))
        #expect(result.thinking.contains("second line"))
        #expect(result.response.contains("Final."))
        #expect(!result.thinking.contains("begin_of_thought"))
        #expect(!result.thinking.contains("end_of_thought"))
        #expect(!result.response.contains("begin_of_thought"))
        #expect(!result.response.contains("end_of_thought"))
    }

    // MARK: - 2. Streaming chunk-boundary fuzz
    //
    // **Realism note**: Qwen3.6's `<|begin_of_thought|>` and
    // `<|end_of_thought|>` are registered as `added_tokens` in
    // `tokenizer.json`, so the tokenizer emits each marker as a single
    // special-token decode — they never split across `StreamingDetokenizer`
    // chunks in production. The fuzz tests below therefore keep the
    // markers atomic and split only the surrounding plaintext at every
    // byte offset. Splitting *inside* a special-token marker is not a
    // contract `ThinkingParser` has to honor, and asserting it would
    // pin behavior that production never exercises.

    /// Splits the thinking-block plaintext at every byte offset across
    /// two feeds, with the explicit markers held atomic in their own
    /// chunks. Confirms final thinking/response output is identical
    /// regardless of plaintext chunk shape.
    @Test("Chunk-boundary fuzz: thinking-text 1-split with atomic markers yields identical output")
    func chunkBoundaryFuzzThinkingTextOneSplit() {
        let openMarker = "<|begin_of_thought|>"
        let closeMarker = "<|end_of_thought|>"
        let thinkingText = "analysis with several words"
        let responseText = "final answer"

        let thinkingChars = Array(thinkingText)
        for splitAt in 0...thinkingChars.count {
            let head = String(thinkingChars[0..<splitAt])
            let tail = String(thinkingChars[splitAt..<thinkingChars.count])

            let parser = ThinkingParser(expectImplicitThinking: false)
            _ = parser.feed(openMarker)
            _ = parser.feed(head)
            _ = parser.feed(tail)
            _ = parser.feed(closeMarker)
            _ = parser.feed(responseText)
            let result = parser.finalize()

            #expect(result.thinking == thinkingText,
                "split at \(splitAt): thinking='\(result.thinking)'")
            #expect(result.response == responseText,
                "split at \(splitAt): response='\(result.response)'")
        }
    }

    /// Splits the response-text plaintext at every byte offset across
    /// two feeds (markers atomic). Catches any regression where text
    /// after a normalized close marker is mis-classified due to chunking.
    @Test("Chunk-boundary fuzz: response-text 1-split with atomic markers yields identical output")
    func chunkBoundaryFuzzResponseTextOneSplit() {
        let openMarker = "<|begin_of_thought|>"
        let closeMarker = "<|end_of_thought|>"
        let thinkingText = "R"
        let responseText = "the final answer is forty-two"

        let respChars = Array(responseText)
        for splitAt in 0...respChars.count {
            let head = String(respChars[0..<splitAt])
            let tail = String(respChars[splitAt..<respChars.count])

            let parser = ThinkingParser(expectImplicitThinking: false)
            _ = parser.feed(openMarker)
            _ = parser.feed(thinkingText)
            _ = parser.feed(closeMarker)
            _ = parser.feed(head)
            _ = parser.feed(tail)
            let result = parser.finalize()

            #expect(result.thinking == thinkingText,
                "resp split at \(splitAt): thinking='\(result.thinking)'")
            #expect(result.response == responseText,
                "resp split at \(splitAt): response='\(result.response)'")
        }
    }

    /// Three-feed fuzz: thinking-text split AND response-text split,
    /// markers atomic. Covers the case where both partitions arrive
    /// piecewise across the streaming boundary.
    @Test("Chunk-boundary fuzz: combined 2-split (thinking + response) with atomic markers")
    func chunkBoundaryFuzzCombined() {
        let openMarker = "<|begin_of_thought|>"
        let closeMarker = "<|end_of_thought|>"
        let thinkingText = "analysis"
        let responseText = "answer"

        let tChars = Array(thinkingText)
        let rChars = Array(responseText)

        for ti in 0...tChars.count {
            for ri in 0...rChars.count {
                let tHead = String(tChars[0..<ti])
                let tTail = String(tChars[ti..<tChars.count])
                let rHead = String(rChars[0..<ri])
                let rTail = String(rChars[ri..<rChars.count])

                let parser = ThinkingParser(expectImplicitThinking: false)
                _ = parser.feed(openMarker)
                _ = parser.feed(tHead)
                _ = parser.feed(tTail)
                _ = parser.feed(closeMarker)
                _ = parser.feed(rHead)
                _ = parser.feed(rTail)
                let result = parser.finalize()

                #expect(result.thinking == thinkingText,
                    "ti=\(ti) ri=\(ri): thinking='\(result.thinking)'")
                #expect(result.response == responseText,
                    "ti=\(ti) ri=\(ri): response='\(result.response)'")
            }
        }
    }

    /// Char-by-char streaming for non-marker text, with markers held
    /// atomic (matches StreamingDetokenizer's per-token output where
    /// special tokens decode as one chunk and regular tokens as
    /// individual short chunks).
    @Test("Char-by-char non-marker text with atomic markers preserves both partitions")
    func charByCharNonMarkerText() {
        let parser = ThinkingParser(expectImplicitThinking: false)
        _ = parser.feed("<|begin_of_thought|>")
        for ch in "analysis content" {
            _ = parser.feed(String(ch))
        }
        _ = parser.feed("<|end_of_thought|>")
        for ch in "the answer" {
            _ = parser.feed(String(ch))
        }
        let result = parser.finalize()
        #expect(result.thinking == "analysis content",
            "char-by-char thinking text dropped or corrupted; got '\(result.thinking)'")
        #expect(result.response == "the answer",
            "char-by-char response text dropped or corrupted; got '\(result.response)'")
    }

    // MARK: - 3. Mixed markers

    /// A stream containing both `<think>` and `<|begin_of_thought|>` —
    /// each pair of open/close markers must round-trip through the
    /// parser intact. Feeds are split into per-chunk calls (matching
    /// real streaming where each special token / text fragment lands
    /// in its own `feed()` call); a single mega-feed exercises the
    /// parser's intra-loop early-break and leaves trailing buffer
    /// for `finalize()` to handle, which is a different code path
    /// from production streaming.
    @Test("Mixed markers: both <think>...</think> and <|begin_of_thought|>...<|end_of_thought|> normalize to thinking blocks")
    func mixedMarkers() {
        let parser = ThinkingParser(expectImplicitThinking: false)

        // Stream pattern: legacy <think> block, then plain text, then
        // a Qwen3.6+ explicit-marker block, then final response. Each
        // marker arrives as a self-contained chunk (matching how the
        // tokenizer special-token decode produces it).
        _ = parser.feed("<think>")
        _ = parser.feed("standard reasoning")
        _ = parser.feed("</think>")
        _ = parser.feed("middle response ")
        _ = parser.feed("<|begin_of_thought|>")
        _ = parser.feed("extra reasoning")
        _ = parser.feed("<|end_of_thought|>")
        _ = parser.feed("final answer")
        let result = parser.finalize()

        // Both reasoning blocks must contribute to thinking; the
        // intervening text plus the post-close text both reach response.
        // Crucially, no marker bytes leak into either partition.
        #expect(result.thinking.contains("standard reasoning"),
            "first thinking block missing; got thinking='\(result.thinking)'")
        #expect(result.thinking.contains("extra reasoning"),
            "second thinking block missing; got thinking='\(result.thinking)'")
        #expect(result.response.contains("middle response"),
            "intervening response text missing; got response='\(result.response)'")
        #expect(result.response.contains("final answer"),
            "post-close response text missing; got response='\(result.response)'")

        // No literal marker bytes in either partition.
        #expect(!result.thinking.contains("begin_of_thought"))
        #expect(!result.thinking.contains("end_of_thought"))
        #expect(!result.response.contains("begin_of_thought"))
        #expect(!result.response.contains("end_of_thought"))
        #expect(!result.thinking.contains("<think"))
        #expect(!result.response.contains("<think"))
    }

    // MARK: - 4. Implicit-detection regression guard

    /// Helper: writes a fake model directory under `NovaMLXPaths.modelsDir`
    /// with the minimum file shape that `isImplicitThinkingModel` reads
    /// (tokenizer.json `added_tokens`, plus optional chat_template).
    /// Returns the synthetic modelId for use in the assertion. Cleanup
    /// is the caller's responsibility (use `defer`).
    private func makeFixtureModel(
        nameSuffix: String,
        addedTokens: [[String: Any]],
        chatTemplate: String?
    ) throws -> String {
        // Unique modelId per test invocation so parallel runs / retained
        // failures don't collide with real downloaded models or each other.
        let modelId = "novamlx-test-\(nameSuffix)-\(UUID().uuidString.prefix(8))"
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let tokenizer: [String: Any] = ["added_tokens": addedTokens]
        try JSONSerialization.data(withJSONObject: tokenizer)
            .write(to: modelDir.appendingPathComponent("tokenizer.json"))

        if let template = chatTemplate {
            let tc: [String: Any] = ["chat_template": template]
            try JSONSerialization.data(withJSONObject: tc)
                .write(to: modelDir.appendingPathComponent("tokenizer_config.json"))
        }
        return modelId
    }

    private func cleanupFixture(_ modelId: String) {
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        try? FileManager.default.removeItem(at: modelDir)
    }

    @Test("isImplicitThinkingModel returns FALSE for Qwen3.6-style fixture (template + explicit thinking tokens)")
    func implicitDetectionQwenLikeReturnsFalse() throws {
        // Mirrors the real Qwen3.6-35B-A3B-4bit shape:
        //   - chat_template injects <think tag prefix
        //   - tokenizer added_tokens contains <|begin_of_thought|>
        // Per `isImplicitThinkingModel` contract: the explicit thinking
        // token wins, so the model is classified as NON-implicit.
        let modelId = try makeFixtureModel(
            nameSuffix: "qwen36-explicit",
            addedTokens: [
                ["content": "<|begin_of_thought|>", "special": true],
                ["content": "<|end_of_thought|>", "special": true],
                ["content": "<|im_start|>", "special": true],
                ["content": "<|im_end|>", "special": true],
            ],
            chatTemplate: """
            {%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}
            """
        )
        defer { cleanupFixture(modelId) }

        let isImplicit = ModelContainer.isImplicitThinkingModel(for: modelId)
        #expect(isImplicit == false,
            "Qwen3.6-style fixture (template <think + explicit <|begin_of_thought|> token) must be detected as NON-implicit so ThinkingParser is constructed with expectImplicitThinking=false")
    }

    @Test("isImplicitThinkingModel returns TRUE for DeepSeek-R1-style fixture (template-only, no explicit marker tokens)")
    func implicitDetectionDeepSeekR1LikeReturnsTrue() throws {
        // DeepSeek-R1-Distill style: chat template injects <think prefix
        // but tokenizer added_tokens has NO <|begin_of_thought|> /
        // </thinkgt-style explicit marker token. The model only emits
        // </think> as a special token mid-stream → implicit mode.
        let modelId = try makeFixtureModel(
            nameSuffix: "r1-implicit",
            addedTokens: [
                // Common DeepSeek-R1 distill added_tokens — none of these
                // are thinking markers; isImplicitThinkingModel must walk
                // through the list and conclude no explicit token exists.
                ["content": "<｜begin▁of▁sentence｜>", "special": true],
                ["content": "<｜end▁of▁sentence｜>", "special": true],
                ["content": "<｜User｜>", "special": true],
                ["content": "<｜Assistant｜>", "special": true],
            ],
            chatTemplate: """
            {{ bos_token }}{%- for message in messages %}{%- if message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}{%- elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' + message['content'] + '<｜end▁of▁sentence｜>' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{ '<｜Assistant｜><think>\n' }}{%- endif %}
            """
        )
        defer { cleanupFixture(modelId) }

        let isImplicit = ModelContainer.isImplicitThinkingModel(for: modelId)
        #expect(isImplicit == true,
            "DeepSeek-R1-style fixture (template <think but no explicit marker token) must be detected as implicit so ThinkingParser yields tokens as thinking immediately rather than buffering")
    }

    @Test("isImplicitThinkingModel returns FALSE when chat template has no <think tag")
    func implicitDetectionTemplateLacksThinkTag() throws {
        // Sanity check: a model whose chat template doesn't mention
        // <think at all must always be classified as non-implicit
        // regardless of added_tokens content (the gate-clause at line
        // 264 short-circuits before scanning tokens).
        let modelId = try makeFixtureModel(
            nameSuffix: "no-think-template",
            addedTokens: [
                ["content": "<|im_start|>", "special": true],
                ["content": "<|im_end|>", "special": true],
            ],
            chatTemplate: """
            {%- for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
            """
        )
        defer { cleanupFixture(modelId) }

        let isImplicit = ModelContainer.isImplicitThinkingModel(for: modelId)
        #expect(isImplicit == false,
            "no-<think-tag template must short-circuit to non-implicit")
    }

    // MARK: - 5. Negative case — enable_thinking=false APIServer scrub

    /// Replicates the regex used by `APIServer.swift:1742` on the
    /// `shouldParseThinking == false` branch and asserts what it
    /// actually scrubs from the model's raw output. This pins the
    /// current production behavior so any regression (or intentional
    /// fix) shows up here rather than at integration time.
    ///
    /// Current behavior (locked in):
    ///   - `<think>`, `</think>`, `<thinking>`, `</thinking>` markers
    ///     are stripped (markers only — content between them is preserved).
    ///   - Non-thinking control tokens like `<|im_end|>` are stripped.
    ///   - `<|begin_of_thought|>` / `<|end_of_thought|>` are NOT
    ///     stripped by this regex on its own — they rely on the
    ///     upstream `MLXEngine.scrubControlTokens` chain to handle them
    ///     before they reach this point. We document the gap here so
    ///     a future regex unification removes it.
    @Test("APIServer enable_thinking=false regex strips <think>/</think> markers, preserves text between")
    func enableThinkingFalseScrubsThinkMarkers() throws {
        let pattern = "<(?:\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)|/?think[^>]*)>"
        let regex = try NSRegularExpression(pattern: pattern)

        var text = "Before<think>reasoning here</think>middle<thinking>more</thinking>after"
        let nsRange = NSRange(text.startIndex..., in: text)
        let matches = regex.matches(in: text, range: nsRange)
        for match in matches.reversed() {
            if let r = Range(match.range, in: text) {
                text.removeSubrange(r)
            }
        }

        // Markers stripped — the text BETWEEN open/close survives.
        #expect(text == "Beforereasoning heremiddlemoreafter",
            "expected markers stripped, content preserved; got '\(text)'")
        #expect(!text.contains("<think"))
        #expect(!text.contains("</think"))
        #expect(!text.contains("<thinking"))
        #expect(!text.contains("</thinking"))
    }

    /// Contract pin: explicitly document that `<|begin_of_thought|>` and
    /// `<|im_end|>`-style `<|word|>` tokens are NOT stripped by the
    /// `enable_thinking=false` regex in isolation, because the
    /// `<\|word(?:\|>|>)>` outer pattern requires a trailing `>` *beyond*
    /// the inner `(?:\|>|>)` group — i.e. an effectively `<|word|>>` shape
    /// that real tokens never have. The control-token branch of this regex
    /// is therefore dead code for typical model output. Upstream
    /// `MLXEngine.scrubControlTokens` removes these markers before they
    /// reach `APIServer`, so the gap doesn't surface in production. If a
    /// future regex unification fixes the trailing-`>` quirk, this test
    /// fails and signals the contract update.
    @Test("APIServer enable_thinking=false regex does NOT strip <|word|>-shape control tokens (regex quirk pin)")
    func enableThinkingFalseDoesNotScrubBarPipeTokens() throws {
        let pattern = "<(?:\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>)|/?think[^>]*)>"
        let regex = try NSRegularExpression(pattern: pattern)

        // Each input is the realistic single-`>` shape that tokenizers emit;
        // none of them satisfy the regex's double-`>` requirement.
        let realisticControlTokens = [
            "<|begin_of_thought|>",
            "<|end_of_thought|>",
            "<|im_end|>",
            "<|im_start|>",
            "<|tool_call|>",
        ]
        for token in realisticControlTokens {
            let matches = regex.matches(in: token, range: NSRange(token.startIndex..., in: token))
            #expect(matches.isEmpty,
                "regex unexpectedly matched '\(token)' — the trailing-`>` quirk may have been fixed; update the contract")
        }
    }
}
