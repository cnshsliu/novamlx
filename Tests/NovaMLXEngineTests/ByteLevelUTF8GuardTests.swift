import Foundation
import Testing

@testable import NovaMLXEngine

/// ByteLevel split-character completion (`ByteLevelUTF8Guard`) and the
/// U+FFFD emission withhold (`FFFDSafePrefix`).
///
/// Token pieces below are the real Flint (RecursiveSI/flint-int4) ByteLevel
/// vocab entries verified against `tokenizer.json` in
/// `.scratch/flint-utf8-diag/REPORT.md`: 娟 = ids [19537, 253],
/// 苑 = [9188, 239], 纶 = [4682, 114], and `Ġ` (space) = 220.
@Suite("ByteLevel UTF-8 guard")
struct ByteLevelUTF8GuardTests {

    // MARK: - Fixtures

    /// Real ByteLevel pieces (byte-level-mapped strings from the flint vocab).
    private static let flintPieces: [Int: String] = [
        220: "Ġ",          // [0x20] space
        253: "Ł",          // [0x9F] continuation byte (娟 second half)
        239: "ĳ",          // [0x91] continuation byte (苑 second half)
        114: "¶",          // [0xB6] continuation byte (纶 second half)
        4682: "çº",        // [0xE7, 0xBA] 纶 first half
        9188: "èĭ",        // [0xE8, 0x8B] 苑 first half
        19537: "å¨",       // [0xE5, 0xA8] 娟 first half
        53808: "ä¼¦",      // [0xE4, 0xBC, 0xA6] 伦, whole-char token
        // Added (non-ByteLevel) token: contains scalars outside the alphabet.
        150002: "<|user|>",
    ]

    private static let vocabSize = 150003

    private static func makeTable() -> ByteLevelTokenBytes {
        ByteLevelTokenBytes(vocabSize: vocabSize) { id in
            flintPieces[id]
        }
    }

    /// The ByteLevel piece string for arbitrary raw bytes (alphabet inverse).
    private static func piece(for bytes: [UInt8]) -> String {
        var scalars = String.UnicodeScalarView()
        for byte in bytes {
            let scalar = GPT2ByteLevelAlphabet.scalarToByte.first { $0.value == byte }!.key
            scalars.append(scalar)
        }
        return String(scalars)
    }

    // MARK: - Alphabet

    @Test("GPT-2 byte alphabet round-trips the 256 bytes")
    func alphabetCoversAllBytes() {
        var seen = Set<UInt8>()
        for (_, byte) in GPT2ByteLevelAlphabet.scalarToByte {
            seen.insert(byte)
        }
        #expect(seen.count == 256)
        // Known mappings from the flint tokenizer.json.
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "å¨") == [0xE5, 0xA8])
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "Ł") == [0x9F])
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "ĳ") == [0x91])
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "¶") == [0xB6])
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "Ġ") == [0x20])
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "ä¼¦") == [0xE4, 0xBC, 0xA6])
        // Pure-ASCII added tokens map to their ASCII bytes — harmless: their
        // first byte is never a continuation byte, so the pending mask
        // forbids them anyway. Added tokens with scalars outside the
        // alphabet (e.g. full-width ｜) are rejected outright.
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "<|user|>") == Array("<|user|>".utf8))
        #expect(GPT2ByteLevelAlphabet.bytes(ofPiece: "<｜end▁of▁text｜>") == nil)
    }

    // MARK: - Guard state machine

    @Test("half of 娟 opens the constraint; its continuation closes it")
    func juanSplitLifecycle() {
        let table = Self.makeTable()
        var guardState = ByteLevelUTF8Guard(table: table)

        #expect(!guardState.hasPending)
        #expect(guardState.peekMask() == nil)

        guardState.consume(19537)  // 娟 bytes E5 A8 — dangling
        #expect(guardState.hasPending)
        #expect(guardState.carry == [0xE5, 0xA8])
        #expect(guardState.neededContinuations == 1)

        let mask = guardState.peekMask()
        #expect(mask != nil)
        #expect(mask!.count == Self.vocabSize)
        // The real second half is allowed.
        #expect(mask![253] == true)
        // Whole-character token (does not start with a continuation byte).
        #expect(mask![53808] == false)
        // Space must be forbidden: this is exactly the "� " leak.
        #expect(mask![220] == false)
        // Added tokens (nil bytes) are forbidden, including stops.
        #expect(mask![150002] == false)

        guardState.consume(253)  // 9F completes 娟
        #expect(!guardState.hasPending)
        #expect(guardState.peekMask() == nil)
    }

    @Test("苑 and 纶 split the same way")
    func yuanLunSplits() {
        let table = Self.makeTable()
        var yuan = ByteLevelUTF8Guard(table: table)
        yuan.consume(9188)  // E8 8B
        #expect(yuan.neededContinuations == 1)
        #expect(yuan.peekMask()![239] == true)
        yuan.consume(239)
        #expect(!yuan.hasPending)

        var lun = ByteLevelUTF8Guard(table: table)
        lun.consume(4682)  // E7 BA
        #expect(lun.neededContinuations == 1)
        #expect(lun.peekMask()![114] == true)
        lun.consume(114)
        #expect(!lun.hasPending)
    }

    @Test("a 3-byte lead dangling alone needs two continuations")
    func threeByteLeadNeedsTwo() {
        let table = Self.makeTable()
        var state = ByteLevelUTF8Guard(table: table)
        state.consume(19537)  // E5 A8 → needs one
        #expect(state.neededContinuations == 1)
        state.consume(9188)  // E8 8B — resolves E5 A8 + (E8 lead)… still dangling
        #expect(state.neededContinuations == 1)
        #expect(state.carry == [0xE8, 0x8B])
        state.consume(239)  // 91 closes 苑
        #expect(!state.hasPending)
    }

    @Test("4-byte emoji split needs three continuations; partial fill advances")
    func fourByteEmojiSplit() {
        let table = Self.makeTable()
        var pieces: [Int: String] = Self.flintPieces
        // 🀄 U+1F004 = F0 9F 80 84, split F0 9F | 80 | 84.
        let firstHalf = Self.piece(for: [0xF0, 0x9F])
        let secondByte = Self.piece(for: [0x80])
        pieces[1] = firstHalf
        pieces[2] = secondByte
        let table4 = ByteLevelTokenBytes(vocabSize: Self.vocabSize) { id in
            pieces[id]
        }
        var state = ByteLevelUTF8Guard(table: table4)
        state.consume(1)
        #expect(state.neededContinuations == 2)
        #expect(state.carry == [0xF0, 0x9F])
        let mask = state.peekMask()
        // A single continuation byte partially fills and is allowed.
        #expect(mask![2] == true)
        state.consume(2)
        #expect(state.neededContinuations == 1)
        #expect(state.carry == [0xF0, 0x9F, 0x80])
    }

    @Test("invalid and stray bytes resolve instead of wedging")
    func strayContinuationResolves() {
        #expect(ByteLevelUTF8Guard.scanTail([0x80, 0x80]) == ([], 0))
        #expect(ByteLevelUTF8Guard.scanTail([0xC0, 0x80]) == ([], 0))  // overlong
        #expect(ByteLevelUTF8Guard.scanTail([0x41]) == ([], 0))  // ASCII
        #expect(ByteLevelUTF8Guard.scanTail([0xE5, 0xA8]) == ([0xE5, 0xA8], 1))
        #expect(ByteLevelUTF8Guard.scanTail([0xE5, 0xA8, 0x9F]) == ([], 0))
        #expect(ByteLevelUTF8Guard.scanTail([0xF0]) == ([0xF0], 3))
        #expect(ByteLevelUTF8Guard.scanTail([0xF0, 0x9F]) == ([0xF0, 0x9F], 2))
    }

    @Test("hopeless constraint lifts after the cap")
    func constraintLiftsAfterCap() {
        // Vocab: one dangling first-half piece (E5 A8) and a space; nothing
        // starts with the needed continuation byte.
        let table = ByteLevelTokenBytes(vocabSize: 2) { id in
            id == 0 ? Self.piece(for: [0xE5, 0xA8]) : "Ġ"
        }
        var seeded = ByteLevelUTF8Guard(table: table)
        seeded.consume(0)
        #expect(seeded.hasPending)
        // No token in the table continues 9F → peekMask nil, and the step
        // accounting must lift the constraint rather than mask everything.
        #expect(seeded.peekMask() == nil)
        for _ in 0...ByteLevelUTF8Guard.maxConstrainedSteps {
            seeded.noteStillPending()
        }
        #expect(!seeded.hasPending)
    }

    // MARK: - FFFD safe prefix

    @Test("withhold holds at the replacement char and releases on completion")
    func safePrefixWithholdAndRelease() {
        // Stream: 麦美 + dangling half (decodes to �), then completed 娟.
        var state = FFFDSafePrefix.nextDelta(fullText: "麦美", emittedCount: 0)
        #expect(state.delta == "麦美")
        #expect(state.emittedCount == 2)

        state = FFFDSafePrefix.nextDelta(fullText: "麦美�", emittedCount: state.emittedCount)
        #expect(state.delta == "")
        #expect(state.emittedCount == 2)

        // Completing token replaces the replacement char; emit it plus the tail.
        state = FFFDSafePrefix.nextDelta(fullText: "麦美娟 宏福", emittedCount: state.emittedCount)
        #expect(state.delta == "娟 宏福")
        #expect(state.emittedCount == 6)
    }

    @Test("interior replacement char holds everything after it")
    func safePrefixInteriorHold() {
        var state = FFFDSafePrefix.nextDelta(fullText: "a�b", emittedCount: 0)
        #expect(state.delta == "a")
        #expect(state.emittedCount == 1)
        // Same text again — nothing new is safe.
        state = FFFDSafePrefix.nextDelta(fullText: "a�b", emittedCount: state.emittedCount)
        #expect(state.delta == "")
        #expect(state.emittedCount == 1)
        // Initial pointer stops at the first replacement char.
        #expect(FFFDSafePrefix.initialSafeCount(of: "a�b") == 1)
        #expect(FFFDSafePrefix.initialSafeCount(of: "干净") == 2)
    }

    @Test("seed from a dangling prefix-cache tail")
    func seedFromDanglingTail() {
        let table = Self.makeTable()
        // …麦美 [19537] cached with the half-character open.
        let seeded = ByteLevelUTF8Guard(table: table, seedTokens: [53808, 19537])
        #expect(seeded.hasPending)
        #expect(seeded.neededContinuations == 1)
        // A seed that closes cleanly leaves nothing pending.
        let closed = ByteLevelUTF8Guard(table: table, seedTokens: [53808, 19537, 253])
        #expect(!closed.hasPending)
    }

    // MARK: - Repetition loop

    @Test("paragraph repetition is detected, legitimate prose is not")
    func repetitionLoopDetection() {
        let paragraph = "31日，民政及青年事务局局长麦美娟就宏福苑火灾问题向受影响居民及社会道歉，又指当局早前有关修订建筑物管理条例的公众咨询收到500份意见。"
        // One copy is fine.
        #expect(!FlintChatDecode.detectRepetitionLoop(paragraph))
        // Two copies: the tail of the second matches the first.
        #expect(FlintChatDecode.detectRepetitionLoop(paragraph + paragraph))
        // A partial second copy trips once its tail (40 chars) overlaps.
        let partial = String(paragraph.suffix(60))
        #expect(FlintChatDecode.detectRepetitionLoop(paragraph + partial))
        // Legitimate prose: repeated names and numbers, no verbatim block.
        let prose = "麦美娟表示会谨守岗位。黄伟纶到访宏福苑，麦美娟强调会修例，黄伟纶支持修例，麦美娟与黄伟纶均认为条例要尽快通过，让法团有更清晰的指引与方向。"
        #expect(!FlintChatDecode.detectRepetitionLoop(prose))
        // Character-level loops trip too.
        #expect(FlintChatDecode.detectRepetitionLoop(String(repeating: "重复", count: 50)))
    }

    @Test("planning chatter markers trip, news prose does not")
    func planningChatterMarkers() {
        #expect(FlintChatDecode.containsCondensePlanningMarker("1.  **拆解请求：**"))
        #expect(FlintChatDecode.containsCondensePlanningMarker("4.  **起草摘要（遵循规则）：**"))
        #expect(FlintChatDecode.containsCondensePlanningMarker("初稿： “31日，财政司副司长……"))
        #expect(FlintChatDecode.containsCondensePlanningMarker("**尝试1（遵循所有规则）：**"))
        let news = "31日，民政及青年事务局局长麦美娟就宏福苑火灾问题向受影响居民及社会道歉，表示将做好修例工作，希望得到立法会支持。"
        #expect(!FlintChatDecode.containsCondensePlanningMarker(news))
    }

    @Test("recitation matching ignores reflowed whitespace and width")
    func recitationNormalization() {
        // Prompt text as written in NewPrompter.ts: full-width ＋ and ／.
        let promptRules = """
        *   固定骨架：时间＋主体＋做了什么＋结果／影响。
        *   一句一事，先结果后必要细节。
        """
        let normalized = FlintChatDecode.normalizedForRecitation(promptRules)
        #expect(!normalized.contains(" "))
        #expect(!normalized.contains("\n"))
        #expect(!normalized.contains("＋"))
        // The model's reflowed copy (newlines → spaces, ＋→+, ／→/) matches.
        let reflowed = FlintChatDecode.normalizedForRecitation(
            "* 固定骨架：时间 + 主体 + 做了什么 + 结果/影响。 * 一句一事，先结果后必要细节。")
        #expect(normalized.contains(reflowed))
        // A paraphrase does not.
        let paraphrase = FlintChatDecode.normalizedForRecitation(
            "每句只讲一件事，先给结论再补必要细节，不使用港媒式铺垫")
        #expect(!normalized.contains(paraphrase))
    }
}
