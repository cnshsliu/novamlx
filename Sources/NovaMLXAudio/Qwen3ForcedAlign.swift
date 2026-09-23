import Foundation
import MLX
import MLXNN
import Tokenizers

/// Text side of Qwen3-ForcedAligner. Chinese is one character per token.
/// English is whitespace-separated. Japanese and Korean are one script
/// character per token; the Python worker used nagisa/soynlp, which this
/// process does not ship.
public enum Qwen3ForceAlignText {
    public static func words(in text: String, language: String) -> [String] {
        switch language.lowercased() {
        case "japanese", "korean":
            return tokenizeAtomic(text)
        case "chinese":
            return tokenizeChineseMixed(text)
        default:
            return tokenizeSpace(text)
        }
    }

    public static func prompt(words: [String], audioTokens: Int) -> String {
        let stamps = words.joined(separator: "<timestamp><timestamp>") + "<timestamp><timestamp>"
        let pads = String(repeating: "<|audio_pad|>", count: max(audioTokens, 1))
        return "<|audio_start|>\(pads)<|audio_end|>" + stamps
    }

    /// Longest-increasing-subsequence repair from the MLX aligner.
    public static func fixTimestamps(_ data: [Int]) -> [Int] {
        let n = data.count
        if n == 0 { return [] }
        var dp = [Int](repeating: 1, count: n)
        var parent = [Int](repeating: -1, count: n)
        for i in 1..<n {
            for j in 0..<i where data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1
                parent[i] = j
            }
        }
        let maxLength = dp.max() ?? 1
        var maxIdx = dp.firstIndex(of: maxLength) ?? 0
        var lis: [Int] = []
        while maxIdx != -1 {
            lis.append(maxIdx)
            maxIdx = parent[maxIdx]
        }
        var normal = [Bool](repeating: false, count: n)
        for idx in lis { normal[idx] = true }

        var result = data.map(Double.init)
        var i = 0
        while i < n {
            if normal[i] {
                i += 1
                continue
            }
            var j = i
            while j < n && !normal[j] { j += 1 }
            let count = j - i
            let left = (0..<i).reversed().first { normal[$0] }.map { result[$0] }
            let right = (j..<n).first { normal[$0] }.map { result[$0] }
            if count <= 2 {
                for k in i..<j {
                    if let left, let right {
                        result[k] = (k - (i - 1)) <= (j - k) ? left : right
                    } else {
                        result[k] = left ?? right ?? result[k]
                    }
                }
            } else if let left, let right {
                let step = (right - left) / Double(count + 1)
                for k in i..<j {
                    result[k] = left + step * Double(k - i + 1)
                }
            } else if let left {
                for k in i..<j { result[k] = left }
            } else if let right {
                for k in i..<j { result[k] = right }
            }
            i = j
        }
        return result.map { Int($0) }
    }

    private static func tokenizeChineseMixed(_ text: String) -> [String] {
        var tokens: [String] = []
        var latin = ""
        func flush() {
            let cleaned = clean(latin)
            if !cleaned.isEmpty { tokens.append(cleaned) }
            latin = ""
        }
        for ch in text {
            if isCJK(ch) {
                flush()
                tokens.append(String(ch))
            } else if isKept(ch) {
                latin.append(ch)
            } else {
                flush()
            }
        }
        flush()
        return tokens
    }

    private static func tokenizeSpace(_ text: String) -> [String] {
        var tokens: [String] = []
        for seg in text.split(whereSeparator: \.isWhitespace) {
            let cleaned = clean(String(seg))
            guard !cleaned.isEmpty else { continue }
            tokens.append(contentsOf: splitChineseRuns(cleaned))
        }
        return tokens
    }

    private static func tokenizeAtomic(_ text: String) -> [String] {
        var tokens: [String] = []
        var latin = ""
        func flush() {
            let cleaned = clean(latin)
            if !cleaned.isEmpty { tokens.append(cleaned) }
            latin = ""
        }
        for ch in text {
            if isAtomic(ch) {
                flush()
                tokens.append(String(ch))
            } else if isKept(ch) {
                latin.append(ch)
            } else {
                flush()
            }
        }
        flush()
        return tokens
    }

    private static func splitChineseRuns(_ seg: String) -> [String] {
        var tokens: [String] = []
        var buf = ""
        func flush() {
            if !buf.isEmpty { tokens.append(buf); buf = "" }
        }
        for ch in seg {
            if isCJK(ch) {
                flush()
                tokens.append(String(ch))
            } else {
                buf.append(ch)
            }
        }
        flush()
        return tokens
    }

    private static func clean(_ token: String) -> String {
        String(token.filter(isKept))
    }

    private static func isKept(_ ch: Character) -> Bool {
        if ch == "'" { return true }
        return ch.unicodeScalars.allSatisfy { scalar in
            CharacterSet.letters.contains(scalar) || CharacterSet.decimalDigits.contains(scalar)
        }
    }

    private static func isCJK(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first, ch.unicodeScalars.count == 1 else { return false }
        let code = scalar.value
        return (0x4E00...0x9FFF).contains(code)
            || (0x3400...0x4DBF).contains(code)
            || (0x20000...0x2A6DF).contains(code)
            || (0x2A700...0x2B73F).contains(code)
            || (0x2B740...0x2B81F).contains(code)
            || (0x2B820...0x2CEAF).contains(code)
            || (0xF900...0xFAFF).contains(code)
    }

    private static func isAtomic(_ ch: Character) -> Bool {
        if isCJK(ch) { return true }
        guard let scalar = ch.unicodeScalars.first, ch.unicodeScalars.count == 1 else { return false }
        let code = scalar.value
        return (0x3040...0x30FF).contains(code)
            || (0xAC00...0xD7AF).contains(code)
            || (0x1100...0x11FF).contains(code)
    }
}

extension Qwen3ASRModel {
    /// Word timestamps. `audio` is mono 16 kHz. Times are seconds.
    public func align(audio: MLXArray, text: String, language: String) throws -> [(text: String, start: Double, end: Double)] {
        guard config.isForcedAligner else {
            throw NSError(
                domain: "Qwen3ForcedAlign", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Checkpoint is not a Qwen3 ForcedAligner"]
            )
        }
        guard let tokenizer else {
            throw NSError(
                domain: "Qwen3ForcedAlign", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "ForcedAligner tokenizer is not loaded"]
            )
        }
        let words = Qwen3ForceAlignText.words(in: text, language: language)
        guard !words.isEmpty else { return [] }

        let (features, featureMask, audioTokens) = preprocessAudio(audio)
        let prompt = Qwen3ForceAlignText.prompt(words: words, audioTokens: audioTokens)
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false)
        guard !ids.isEmpty else {
            throw NSError(
                domain: "Qwen3ForcedAlign", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "ForcedAligner tokenizer returned no tokens"]
            )
        }
        let inputIds = MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
        let logits = self(
            inputIds: inputIds,
            inputFeatures: features,
            featureAttentionMask: featureMask
        )
        eval(logits)
        let predicted = argMax(logits, axis: -1)
        let inputFlat = inputIds[0].asArray(Int32.self)
        let outputFlat = predicted[0].asArray(Int32.self)
        let stampId = Int32(config.timestampTokenId ?? 151705)
        let segment = Double(config.timestampSegmentTime ?? 80)
        var raw: [Int] = []
        raw.reserveCapacity(words.count * 2)
        for (index, token) in inputFlat.enumerated() where token == stampId && index < outputFlat.count {
            raw.append(Int((Double(outputFlat[index]) * segment).rounded(.towardZero)))
        }
        guard raw.count >= words.count * 2 else {
            throw NSError(
                domain: "Qwen3ForcedAlign", code: 4,
                userInfo: [NSLocalizedDescriptionKey: "ForcedAligner returned \(raw.count) timestamps for \(words.count) words"]
            )
        }
        let fixed = Qwen3ForceAlignText.fixTimestamps(Array(raw.prefix(words.count * 2)))
        return words.enumerated().map { i, word in
            (
                text: word,
                start: (Double(fixed[i * 2]) / 1000).rounded(toPlaces: 3),
                end: (Double(fixed[i * 2 + 1]) / 1000).rounded(toPlaces: 3)
            )
        }
    }
}

private extension Double {
    func rounded(toPlaces places: Int) -> Double {
        let scale = Foundation.pow(10.0, Double(places))
        return (self * scale).rounded() / scale
    }
}
