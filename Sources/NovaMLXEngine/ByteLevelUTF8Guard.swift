import Foundation
import NovaMLXCore
import NovaMLXUtils

/// ByteLevel BPE UTF-8 completion guard.
///
/// Flint's tokenizer (ByteLevel BPE) has no whole-vocab entry for 娟 / 苑 /
/// 纶 and similar hanzi: each char is emitted as two tokens — the first two
/// UTF-8 bytes, then the final continuation byte (e.g. 娟 = E5 A8 + 9F →
/// ids `[19537, 253]`). The int4 model frequently samples only the first
/// piece and moves on, and `ByteLevelDecoder`'s lossy
/// `String(decoding:as:UTF8.self)` renders the dangling lead bytes as
/// U+FFFD mid-text. Diagnostics: `.scratch/flint-utf8-diag/REPORT.md`.
///
/// Two defenses, both owned here:
///
/// 1. **Sampling constraint** (`ByteLevelUTF8Guard`). When the bytes sampled
///    so far end in an incomplete UTF-8 sequence, the next sample is masked
///    to tokens that supply the missing continuation bytes (0x80–0xBF).
///    EOS and special tokens are forbidden while pending — a turn may not
///    end on half a character. The constraint normally clears in one step,
///    so the cost is one constrained step per split character.
/// 2. **Emission withhold** (`FFFDSafePrefix`). Whatever the sampler does,
///    the streamed delta never crosses a U+FFFD until it resolves (a later
///    token completing the sequence replaces it with the real character) or
///    the stream finishes, at which point the raw tail is flushed.

/// The GPT-2 ByteLevel alphabet: 256 bytes ↔ 256 Unicode scalars.
///
/// Printable bytes map to themselves (0x21–0x7E, 0xA1–0xAC, 0xAE–0xFF);
/// the rest map to U+0100 + sequential index. swift-transformers keeps this
/// table internal (`ByteEncoder.swift`), so it is rebuilt here to turn
/// `convertIdToToken` strings (e.g. `"å¨"`, `"Ł"`) back into raw bytes.
enum GPT2ByteLevelAlphabet {
    /// Unicode scalar → byte. Scalars outside the alphabet cannot come from
    /// a ByteLevel-encoded piece.
    static let scalarToByte: [Unicode.Scalar: UInt8] = {
        var map: [Unicode.Scalar: UInt8] = [:]
        map.reserveCapacity(256)
        var next = 0
        for byte in UInt8(0)...UInt8(255) {
            let scalarValue: UInt32
            if (0x21...0x7E).contains(Int(byte))
                || (0xA1...0xAC).contains(Int(byte))
                || (0xAE...0xFF).contains(Int(byte))
            {
                scalarValue = UInt32(byte)
            } else {
                scalarValue = 0x100 + UInt32(next)
                next += 1
            }
            map[Unicode.Scalar(scalarValue)!] = byte
        }
        return map
    }()

    /// Reconstruct the raw bytes of a ByteLevel token piece, or nil when the
    /// piece contains a scalar outside the alphabet (non-ByteLevel / added
    /// token such as `<|assistant|>`).
    static func bytes(ofPiece piece: String) -> [UInt8]? {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(piece.utf8.count)
        for scalar in piece.unicodeScalars {
            guard let byte = scalarToByte[scalar] else { return nil }
            bytes.append(byte)
        }
        return bytes
    }
}

/// Per-model, shared across sequences: token-id → raw byte table plus the
/// cached "starts with N continuation bytes" masks for N = 1…3.
public final class ByteLevelTokenBytes: @unchecked Sendable {
    let vocabSize: Int
    private let table: [[UInt8]?]
    private let lock = NSLock()
    /// `prefixMasks[n - 1]` allows exactly the tokens whose first
    /// `min(n, count)` bytes are all 0x80–0xBF. nil = not built yet;
    /// built lazily because unconstrained models never pay for it.
    private var prefixMasks: [[Bool]?] = [nil, nil, nil]

    public init(vocabSize: Int, convertIdToToken: (Int) -> String?) {
        let size = max(0, vocabSize)
        var table = [[UInt8]?](repeating: nil, count: size)
        for id in 0..<size {
            if let piece = convertIdToToken(id) {
                table[id] = GPT2ByteLevelAlphabet.bytes(ofPiece: piece)
            }
        }
        self.vocabSize = size
        self.table = table
    }

    public func bytes(for tokenId: Int) -> [UInt8]? {
        guard tokenId >= 0, tokenId < table.count else { return nil }
        return table[tokenId]
    }

    /// Host-side allow-vector for "the next token must start with `count`
    /// continuation bytes" (count 1–3). nil when no vocab token qualifies —
    /// the caller must then lift the constraint rather than mask everything.
    public func prefixContinuationMask(needing count: Int) -> [Bool]? {
        precondition((1...3).contains(count), "continuation count must be 1–3")
        lock.lock()
        defer { lock.unlock() }
        if let cached = prefixMasks[count - 1] { return cached }
        var mask = [Bool](repeating: false, count: table.count)
        var anyAllowed = false
        for id in 0..<table.count {
            guard let bytes = table[id], !bytes.isEmpty else { continue }
            let check = min(count, bytes.count)
            var allowed = true
            for k in 0..<check where bytes[k] & 0xC0 != 0x80 {
                allowed = false
                break
            }
            if allowed {
                mask[id] = true
                anyAllowed = true
            }
        }
        guard anyAllowed else { return nil }
        prefixMasks[count - 1] = mask
        return mask
    }
}

/// Per-sequence sampler state. Track the trailing incomplete UTF-8 sequence
/// across sampled tokens; while one is pending, only tokens that continue it
/// may be sampled.
///
/// State transitions happen where accepted tokens land (the scheduler's
/// result-application loop), so the copies used to build a decode step only
/// read: `peekMask()` is non-mutating, and ``noteStillPending()`` does the
/// step accounting / cap.
struct ByteLevelUTF8Guard: Sendable {
    /// Give up after this many consecutive constrained steps (a vocab whose
    /// continuation bytes are unreachable should degrade to old behavior,
    /// not livelock).
    static let maxConstrainedSteps = 8

    private let table: ByteLevelTokenBytes
    /// Trailing bytes of the incomplete sequence (lead byte + whatever
    /// continuation bytes have arrived; at most 3).
    private(set) var carry: [UInt8] = []
    /// Continuation bytes still required to close the carried sequence.
    private(set) var neededContinuations = 0
    private var constrainedSteps = 0

    var hasPending: Bool { neededContinuations > 0 }

    init(table: ByteLevelTokenBytes) {
        self.table = table
    }

    /// Seed from tokens that were generated before this sequence existed
    /// (prefix-cache continuation). Only the tail matters; scanning the last
    /// 8 tokens is plenty (carry ≤ 3 bytes).
    init(table: ByteLevelTokenBytes, seedTokens: [Int]) {
        self.table = table
        for tokenId in seedTokens.suffix(8) {
            consume(tokenId)
        }
    }

    /// Fold one sampled token into the carry state.
    mutating func consume(_ tokenId: Int) {
        guard let bytes = table.bytes(for: tokenId) else {
            // Special / out-of-table token: contributes no continuation
            // bytes. `carry` stands; the mask forbids these while pending.
            return
        }
        if carry.isEmpty, bytes.isEmpty { return }
        var scan = carry
        scan.append(contentsOf: bytes)
        let tail = Self.scanTail(scan)
        carry = tail.carry
        neededContinuations = tail.needed
        if neededContinuations == 0 { constrainedSteps = 0 }
    }

    /// The `[Bool]` allow-vector for the next sample, or nil when no
    /// constraint applies or no vocab token can continue the sequence (the
    /// caller samples unmasked; ``noteStillPending()`` then decides whether
    /// to lift).
    func peekMask() -> [Bool]? {
        guard neededContinuations > 0 else { return nil }
        return table.prefixContinuationMask(needing: neededContinuations)
    }

    /// Account for one finished decode step. A pending sequence that stays
    /// pending across ``maxConstrainedSteps`` steps (unreachable continuation
    /// bytes) lifts the constraint instead of wedging the sequence.
    mutating func noteStillPending() {
        guard neededContinuations > 0 else {
            constrainedSteps = 0
            return
        }
        constrainedSteps += 1
        if constrainedSteps > Self.maxConstrainedSteps {
            NovaMLXLog.warning(
                "[UTF8Guard] constrained steps exceeded \(Self.maxConstrainedSteps); lifting constraint (carry=\(carry.map { String($0, radix: 16) }))")
            lift()
        }
    }

    mutating func lift() {
        carry = []
        neededContinuations = 0
        constrainedSteps = 0
    }

    /// Walk `bytes` from the end over continuation bytes to the nearest lead
    /// byte and report the still-incomplete tail. Defensive on anything that
    /// is not a well-formed prefix: such a tail resolves to empty (the mask
    /// then stops constraining rather than wedging on garbage).
    static func scanTail(_ bytes: [UInt8]) -> (carry: [UInt8], needed: Int) {
        var i = bytes.count - 1
        var continuationCount = 0
        while i >= 0, bytes[i] & 0xC0 == 0x80 {
            continuationCount += 1
            i -= 1
        }
        guard i >= 0 else { return ([], 0) }
        let lead = bytes[i]
        let expected: Int
        switch lead {
        case 0xC2...0xDF: expected = 2
        case 0xE0...0xEF: expected = 3
        case 0xF0...0xF4: expected = 4
        default: return ([], 0)  // ASCII, invalid, or stray lead: resolved
        }
        guard continuationCount + 1 < expected else { return ([], 0) }
        return (Array(bytes[i...]), expected - continuationCount - 1)
    }
}

/// Emission-side withhold: given the full re-decoded text and how many
/// characters were already streamed, produce the next safe delta. The
/// pointer never advances across a U+FFFD; a later token that completes the
/// sequence replaces the U+FFFD and the withheld text streams as one piece.
enum FFFDSafePrefix {
    static let replacement = "\u{fffd}"

    /// Safe initial pointer for a seed text (prefix-cache continuation).
    static func initialSafeCount(of text: String) -> Int {
        if let range = text.range(of: replacement) {
            return text.distance(from: text.startIndex, to: range.lowerBound)
        }
        return text.count
    }

    /// Next delta and updated pointer. `emittedCount` must be ≤ fullText.count.
    static func nextDelta(fullText: String, emittedCount: Int) -> (delta: String, emittedCount: Int) {
        let total = fullText.count
        var start = fullText.startIndex
        if emittedCount > 0 {
            start = fullText.index(start, offsetBy: min(emittedCount, total))
        }
        if start >= fullText.endIndex { return ("", total) }
        let remainder = fullText[start...]
        if let range = remainder.range(of: replacement) {
            let safe = remainder[remainder.startIndex..<range.lowerBound]
            let newCount = emittedCount + safe.count
            return (String(safe), newCount)
        }
        return (String(remainder), total)
    }
}

/// Process-wide cache of `ByteLevelTokenBytes` keyed by model id, mirroring
/// `TokenMaskBuilderCache`. Built once per model (a full vocabulary scan);
/// only ByteLevel-BPE families (currently Flint) ever ask for it.
public actor ByteLevelTokenBytesCache {
    private var entries: [String: ByteLevelTokenBytes] = [:]

    public init() {}

    /// Get-or-build. nil when the vocab size is unknown or the tokenizer
    /// does not expose `convertIdToToken` — callers then run without the
    /// UTF-8 constraint (previous behavior).
    public func builder(
        for modelId: String,
        tokenizer: NovaMLXEngine.Tokenizer,
        vocabSize: Int
    ) -> ByteLevelTokenBytes? {
        guard vocabSize > 0, let convertIdToToken = tokenizer.convertIdToToken else {
            return nil
        }
        if let existing = entries[modelId] { return existing }
        let started = Date()
        let fresh = ByteLevelTokenBytes(vocabSize: vocabSize, convertIdToToken: convertIdToToken)
        entries[modelId] = fresh
        NovaMLXLog.info(
            "[UTF8Guard] byte table built model=\(modelId) vocab=\(vocabSize) elapsedMs=\(String(format: "%.0f", Date().timeIntervalSince(started) * 1000))")
        return fresh
    }

    public func invalidate(modelId: String) {
        entries.removeValue(forKey: modelId)
    }

    public func invalidateAll() {
        entries.removeAll()
    }
}
