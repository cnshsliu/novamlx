import Foundation
import NovaMLXCore
import NovaMLXUtils

public struct TimedWord: Sendable, Equatable {
    public var text: String
    public var start: Double
    public var end: Double
    public init(text: String, start: Double, end: Double) {
        self.text = text
        self.start = start
        self.end = end
    }
}

public struct TimedSentence: Sendable, Equatable {
    public var start: Double
    public var end: Double
    public var text: String
    public var words: [TimedWord]
    public init(start: Double, end: Double, text: String, words: [TimedWord] = []) {
        self.start = start
        self.end = end
        self.text = text
        self.words = words
    }
}

public struct ClipHeadline: Sendable, Equatable {
    public var title: String
    public var point: String
    /// False when the model returned a transcript slice instead of a summary.
    public var acceptedProposal: Bool
    public init(title: String, point: String, acceptedProposal: Bool) {
        self.title = title
        self.point = point
        self.acceptedProposal = acceptedProposal
    }
}

public struct VideoSliceSpec: Sendable {
    public var start: Double
    public var end: Double
    public var title: String
    public var point: String
    public var words: [TimedWord]
    public init(start: Double, end: Double, title: String, point: String, words: [TimedWord]) {
        self.start = start
        self.end = end
        self.title = title
        self.point = point
        self.words = words
    }
}

public struct PlannedSlice: Sendable, Equatable {
    public var start: Double
    public var end: Double
    public var title: String
    public var point: String
    public var sentenceIds: [Int]
    public init(
        start: Double,
        end: Double,
        title: String,
        point: String,
        sentenceIds: [Int] = []
    ) {
        self.start = start
        self.end = end
        self.title = title
        self.point = point
        self.sentenceIds = sentenceIds
    }
}

public enum TitleOverlayStyle: String, CaseIterable, Identifiable, Sendable {
    case news, youtube, poster, minimal
    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .news: return "News"
        case .youtube: return "YouTube"
        case .poster: return "Poster"
        case .minimal: return "Minimal"
        }
    }
}

public enum PointOverlayStyle: String, CaseIterable, Identifiable, Sendable {
    case box, bar, outline, caption
    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .box: return "Box"
        case .bar: return "Accent bar"
        case .outline: return "Outline"
        case .caption: return "Caption"
        }
    }
}

public enum SliceOutputDir: Equatable, Sendable {
    case missingSetting
    case unreachable(path: String)
    case notDirectory(path: String)
    case notWritable(path: String)
    case ready(URL)

    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

public enum VideoSlicePipeline {
    public static let defaultFillers: Set<String> = [
        "嗯", "啊", "呃", "额", "唔", "哦", "噢", "呀", "哈", "嘿", "哎", "欸",
        "那个", "就是", "就是说", "然后", "然后呢", "这个", "的话", "其实",
        "怎么说", "你知道", "对吧", "对对", "是吧", "那么", "反正", "基本上",
        "uh", "um", "er", "ah", "like",
    ]

    public static let sentenceEnders = CharacterSet(charactersIn: "。！？!?")
    public static let clauseBreaks = CharacterSet(charactersIn: "，、；;,:")
    private static let danglingSuffixes = [
        "但是", "而且", "并且", "因为", "所以", "如果", "的话", "或者",
        "但", "而", "并", "且", "的", "呢", "啊", "吧", "把", "被", "就", "跟", "让", "或", "与", "和",
    ]

    public static func inspectOutputDirectory(_ savedPath: String) -> SliceOutputDir {
        let trimmed = savedPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .missingSetting }
        let url = URL(fileURLWithPath: trimmed, isDirectory: true)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            return .unreachable(path: url.path)
        }
        guard isDir.boolValue else {
            return .notDirectory(path: url.path)
        }
        guard FileManager.default.isWritableFile(atPath: url.path) else {
            return .notWritable(path: url.path)
        }
        return .ready(url)
    }

    public static func jobFolderName(videoName: String, now: Date = Date()) -> String {
        let stem = URL(fileURLWithPath: videoName).deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        let fmt = DateFormatter()
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = TimeZone.current
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        let base = stem.isEmpty ? "slice" : stem
        return "\(base)-\(fmt.string(from: now))"
    }

    public static func ffmpegPath() -> String {
        for p in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"] {
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return "ffmpeg"
    }

    public static func ffprobePath() -> String {
        for p in ["/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe", "/usr/bin/ffprobe"] {
            if FileManager.default.isExecutableFile(atPath: p) { return p }
        }
        return "ffprobe"
    }

    public static func pingFangFont() -> String {
        let candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) } ?? candidates[0]
    }

    public static func stripMarkup(_ raw: String) -> String {
        var s = raw
        s = s.replacingOccurrences(of: #"<[^>]+>"#, with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: #"```[\s\S]*?```"#, with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: #"[#*_`>\[\]\(\)]+"#, with: " ", options: .regularExpression)
        return s.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func stripThinking(_ raw: String) -> String {
        raw.replacingOccurrences(
            of: #"<think>[\s\S]*?</think>"#,
            with: " ",
            options: .regularExpression
        )
    }

    public static func stripCodeFence(_ raw: String) -> String {
        var s = stripThinking(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("```") {
            if let nl = s.firstIndex(of: "\n") {
                s = String(s[s.index(after: nl)...])
            }
            if let fence = s.range(of: "```", options: .backwards) {
                s = String(s[..<fence.lowerBound])
            }
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func extractJSONValue(_ raw: String) -> Any? {
        let trimmed = stripCodeFence(raw)
        if let data = trimmed.data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data)
        {
            return obj
        }
        if let start = trimmed.firstIndex(of: "{"),
           let end = trimmed.lastIndex(of: "}"),
           start < end,
           let data = String(trimmed[start...end]).data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data)
        {
            return obj
        }
        if let start = trimmed.firstIndex(of: "["),
           let end = trimmed.lastIndex(of: "]"),
           start < end,
           let data = String(trimmed[start...end]).data(using: .utf8),
           let obj = try? JSONSerialization.jsonObject(with: data)
        {
            return obj
        }
        return nil
    }

    public static func displayLen(_ text: String) -> Int {
        text.replacingOccurrences(of: "\\N", with: "").filter { !$0.isWhitespace }.count
    }

    public static func joinWords(_ words: [TimedWord]) -> String {
        guard !words.isEmpty else { return "" }
        var out = words[0].text
        for w in words.dropFirst() {
            let t = w.text
            if t.isEmpty { continue }
            if shouldInsertSpace(out, t) {
                out += " " + t
            } else {
                out += t
            }
        }
        return out
    }

    public static func isPlausibleCorrection(_ corrected: String, original: String) -> Bool {
        let a = original.filter { !$0.isWhitespace && !$0.isNewline }
        let b = corrected.filter { !$0.isWhitespace && !$0.isNewline }
        guard !b.isEmpty, !a.isEmpty else { return !b.isEmpty && a.isEmpty }
        let ratio = Double(b.count) / Double(a.count)
        return ratio >= 0.55 && ratio <= 1.85
    }

    public static func parseCorrectedChunks(_ raw: String, fallback: [String]) -> [String] {
        guard !fallback.isEmpty else { return [] }
        if let obj = extractJSONValue(raw) as? [String: Any] {
            if let chunks = obj["chunks"] as? [[String: Any]], !chunks.isEmpty {
                var mapped = fallback
                for row in chunks {
                    let text = (row["text"] as? String)?
                        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    guard !text.isEmpty else { continue }
                    let idx = (row["i"] as? Int)
                        ?? (row["i"] as? Double).map { Int($0) }
                        ?? (row["index"] as? Int)
                    if let idx, idx >= 0, idx < mapped.count {
                        if isPlausibleCorrection(text, original: fallback[idx]) {
                            mapped[idx] = text
                        }
                    }
                }
                return mapped
            }
            if let text = (obj["text"] as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
               !text.isEmpty
            {
                let joined = fallback.joined()
                if isPlausibleCorrection(text, original: joined) {
                    return splitCorrectedText(text, onto: fallback)
                }
            }
            return fallback
        }
        let plain = stripCodeFence(raw)
        if plain.hasPrefix("{") || plain.hasPrefix("[") { return fallback }
        if fallback.count == 1, isPlausibleCorrection(plain, original: fallback[0]) {
            return [plain]
        }
        return fallback
    }

    public static func splitCorrectedText(_ text: String, onto fallback: [String]) -> [String] {
        guard fallback.count > 1 else { return [text] }
        let total = max(1, fallback.map { displayLen($0) }.reduce(0, +))
        let chars = Array(text)
        var out: [String] = []
        var offset = 0
        for (i, raw) in fallback.enumerated() {
            if i == fallback.count - 1 {
                out.append(String(chars[offset...]).trimmingCharacters(in: .whitespacesAndNewlines))
                break
            }
            let share = max(1, Int((Double(displayLen(raw)) / Double(total) * Double(chars.count)).rounded()))
            var end = min(chars.count, offset + share)
            if end < chars.count {
                let window = min(chars.count, end + 18)
                if let punct = (end..<window).first(where: { idx in
                    guard let s = chars[idx].unicodeScalars.first else { return false }
                    return sentenceEnders.contains(s) || clauseBreaks.contains(s)
                }) {
                    end = punct + 1
                }
            }
            out.append(String(chars[offset..<end]).trimmingCharacters(in: .whitespacesAndNewlines))
            offset = end
        }
        return out.count == fallback.count ? out : fallback
    }

    public static func tidyTranscript(_ text: String) -> String {
        var s = text
        for p in ["哈", "嗯", "啊", "哎", "呃", "额", "唔", "欸"] {
            s = s.replacingOccurrences(of: p, with: "")
        }
        s = collapseCharRuns(s)
        s = collapseRepeatedPhrases(s)
        s = s.replacingOccurrences(of: "比方比如说", with: "比如说")
        s = s.replacingOccurrences(of: "也是还是要", with: "还是要")
        s = s.replacingOccurrences(of: "比打个比喻", with: "打个比喻")
        s = s.replacingOccurrences(of: #"([，。！？、])\1+"#, with: "$1", options: .regularExpression)
        s = s.replacingOccurrences(of: "，。", with: "。")
        s = collapseLooseSpaces(s)
        s = restoreProductNames(s)
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Hotword line for Qwen3-ASR's system context. Empty when nothing was supplied.
    public static func asrContext(glossary: String, reference: String = "") -> String {
        var terms = glossaryTerms(glossary)
        var seen = Set(terms.map { $0.lowercased() })
        if terms.count < 12 {
            for phrase in latinPhrases(in: reference) where terms.count < 12 {
                let key = phrase.lowercased()
                if seen.insert(key).inserted {
                    terms.append(phrase)
                }
            }
        }
        guard !terms.isEmpty else { return "" }
        let line = "Vocabulary: " + terms.joined(separator: ", ")
        if line.count <= 240 { return line }
        return String(line.prefix(240))
    }

    public static func glossaryTerms(_ raw: String) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        let parts = raw.split { ch in
            ch == "," || ch == "，" || ch == ";" || ch == "；" || ch == "、" || ch == "\n" || ch == "\r"
        }
        for part in parts {
            let term = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard term.count >= 2, term.count <= 40 else { continue }
            let key = term.lowercased()
            if seen.insert(key).inserted {
                out.append(term)
            }
            if out.count == 32 { break }
        }
        return out
    }

    /// Capitalized or numbered Latin phrases in a reference script, for ASR biasing.
    public static func latinPhrases(in text: String) -> [String] {
        guard !text.isEmpty,
              let re = try? NSRegularExpression(
                pattern: #"[A-Za-z][A-Za-z0-9.+#-]{1,}(?:[ ]+[A-Za-z0-9.+#-]{1,})?"#
              )
        else { return [] }
        let ns = text as NSString
        var out: [String] = []
        var seen = Set<String>()
        for match in re.matches(in: text, range: NSRange(location: 0, length: ns.length)) {
            let phrase = ns.substring(with: match.range(at: 0)).trimmingCharacters(in: .whitespaces)
            let key = phrase.lowercased()
            guard phrase.count <= 40, seen.insert(key).inserted else { continue }
            let hasCapital = phrase.contains { $0.isUppercase }
            let hasDigit = phrase.contains { $0.isNumber }
            if !hasCapital && !hasDigit { continue }
            out.append(phrase)
            if out.count == 12 { break }
        }
        return out
    }

    /// Rewrite the heard collocation for Anthropic's Fable. Bare「刷皮」(a reskin) is left alone.
    public static func restoreProductNames(_ text: String) -> String {
        var s = replaceDigits(
            pattern: "(?:刷皮|安索皮克|安斯罗皮克|安思罗皮克|安索罗皮克)的飞[豹宝]\\s*([0-9一二三四五六七八九两])",
            in: text
        ) { "Anthropic的Fable \($0)" }
        s = replaceDigits(
            pattern: "Anthropic的飞[豹宝]\\s*([0-9一二三四五六七八九两])",
            in: s
        ) { "Anthropic的Fable \($0)" }
        s = s.replacingOccurrences(
            of: #"(?:刷皮|安索皮克|安斯罗皮克|安思罗皮克|安索罗皮克)的Fable"#,
            with: "Anthropic的Fable",
            options: .regularExpression
        )
        s = replaceDigits(
            pattern: "Fable\\s*([一二三四五六七八九两])",
            in: s
        ) { "Fable \($0)" }
        s = s.replacingOccurrences(
            of: #"Fable ([0-9])呢(?=\p{Han})"#,
            with: "Fable $1",
            options: .regularExpression
        )
        return s
    }

    public static func sanitizeHeadline(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        for prefix in ["标题：", "标题:", "大标题：", "大标题:"] where s.hasPrefix(prefix) {
            s.removeFirst(prefix.count)
            break
        }
        s = s.trimmingCharacters(in: CharacterSet(charactersIn: "\"“”'‘’「」"))
        s = s.replacingOccurrences(of: "\n", with: "")
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        while let last = s.last, "。！？!?".contains(last) {
            s.removeLast()
        }
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public static func isDanglingTitle(_ title: String) -> Bool {
        let s = sanitizeHeadline(title)
        guard !s.isEmpty else { return true }
        for suffix in danglingSuffixes where s.hasSuffix(suffix) {
            return true
        }
        return false
    }

    /// A headline has to be its own sentence. A short prefix of the transcript is a cut, not a summary.
    public static func isAcceptableHeadline(_ title: String, source: String) -> Bool {
        let t = sanitizeHeadline(title)
        guard displayLen(t) >= 2, !isDanglingTitle(t) else { return false }
        let spoken = compactForCut(source)
        let line = compactForCut(t)
        if line.count >= 8, spoken.count > line.count + 4, spoken.hasPrefix(line), line.count <= 18 {
            return false
        }
        return true
    }

    public static func resolveHeadline(
        proposedTitle: String,
        proposedPoint: String,
        source: String
    ) -> ClipHeadline {
        let fallback = headlineFallback(source: source)
        let title = sanitizeHeadline(proposedTitle)
        let point = sanitizeHeadline(proposedPoint)
        let titleOK = isAcceptableHeadline(title, source: source)
        let pointOK = isAcceptableHeadline(point, source: source)
        return ClipHeadline(
            title: titleOK ? title : fallback.title,
            point: pointOK ? point : fallback.point,
            acceptedProposal: titleOK
        )
    }

    /// Complete corrected sentence, used when the model only sliced the opening.
    public static func headlineFallback(source: String) -> (title: String, point: String) {
        let parts = splitSentences(restoreProductNames(source))
        let title = completedLine(parts.first ?? source, maxLen: 36)
        let point: String
        if parts.count >= 2 {
            point = completedLine(parts[1], maxLen: 32)
        } else if let range = title.range(of: "但"),
                  title.distance(from: range.upperBound, to: title.endIndex) >= 2
        {
            point = String(title[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            point = title
        }
        let safeTitle = title.isEmpty ? "片段" : title
        let safePoint = point.isEmpty ? safeTitle : point
        return (safeTitle, safePoint)
    }

    public static func parseHeadlines(_ raw: String, count: Int) -> [(title: String, point: String)] {
        var out = Array(repeating: (title: "", point: ""), count: max(0, count))
        guard count > 0, let obj = extractJSONValue(raw) else { return out }
        let rows: [[String: Any]]
        if let dict = obj as? [String: Any] {
            rows = (dict["items"] as? [[String: Any]])
                ?? (dict["slices"] as? [[String: Any]])
                ?? (dict["clips"] as? [[String: Any]])
                ?? []
        } else if let arr = obj as? [[String: Any]] {
            rows = arr
        } else {
            return out
        }
        for (offset, row) in rows.enumerated() {
            let idx = (row["i"] as? Int)
                ?? (row["i"] as? Double).map { Int($0) }
                ?? (row["index"] as? Int)
                ?? offset
            guard idx >= 0, idx < out.count else { continue }
            let title = (row["title"] as? String) ?? (row["大标题"] as? String) ?? ""
            let point = (row["point"] as? String) ?? (row["核心观点"] as? String) ?? ""
            out[idx] = (title, point)
        }
        return out
    }

    public static func collapseLooseSpaces(_ text: String) -> String {
        var out = ""
        var pendingSpace = false
        for ch in text {
            if ch.isWhitespace {
                pendingSpace = !out.isEmpty
                continue
            }
            if pendingSpace {
                if let prev = out.last, isLatinOrDigit(prev), isLatinOrDigit(ch) {
                    out.append(" ")
                }
                pendingSpace = false
            }
            out.append(ch)
        }
        return out
    }

    public static func collapseCharRuns(_ text: String) -> String {
        let stutter: Set<Character> = [
            "我", "的", "了", "是", "就", "那", "这", "有", "很", "都", "用", "在", "把", "给", "也",
        ]
        var runs: [(Character, Int)] = []
        for ch in text {
            if let last = runs.last, last.0 == ch {
                runs[runs.count - 1].1 += 1
            } else {
                runs.append((ch, 1))
            }
        }
        var out = ""
        for (ch, n) in runs {
            if isCJK(ch), n >= 3 {
                out.append(ch)
            } else if stutter.contains(ch), n >= 2 {
                out.append(ch)
            } else {
                out.append(contentsOf: repeatElement(ch, count: n))
            }
        }
        return out
    }

    public static func collapseRepeatedPhrases(_ text: String) -> String {
        let chars = Array(text)
        var i = 0
        var out: [Character] = []
        while i < chars.count {
            var collapsed = false
            let maxN = min(6, (chars.count - i) / 2)
            if maxN >= 2 {
                for n in stride(from: maxN, through: 2, by: -1) {
                    let a = Array(chars[i..<(i + n)])
                    let b = Array(chars[(i + n)..<(i + 2 * n)])
                    guard a == b else { continue }
                    let allCJK = a.allSatisfy { isCJK($0) || $0 == "的" }
                    if allCJK {
                        out.append(contentsOf: a)
                        i += 2 * n
                        collapsed = true
                        break
                    }
                }
            }
            if !collapsed {
                out.append(chars[i])
                i += 1
            }
        }
        return String(out)
    }

    public static func stripFillers(_ words: [TimedWord], extra: Set<String> = []) -> [TimedWord] {
        let ban = defaultFillers.union(extra)
        var out: [TimedWord] = []
        var prev = ""
        for w in words {
            let t = w.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if t.isEmpty { continue }
            if ban.contains(t.lowercased()) || ban.contains(t) { continue }
            if t == prev { continue }
            out.append(TimedWord(text: t, start: w.start, end: w.end))
            prev = t
        }
        return out
    }

    public static func packSlices(words: [TimedWord], target: Double, maxDur: Double? = nil) -> [(start: Double, end: Double, words: [TimedWord])] {
        guard !words.isEmpty else { return [] }
        let cap = maxDur ?? (target * 1.35)
        var slices: [(Double, Double, [TimedWord])] = []
        var cur: [TimedWord] = []
        func flush() {
            guard let first = cur.first, let last = cur.last else { return }
            slices.append((first.start, last.end, cur))
            cur = []
        }
        for w in words {
            if cur.isEmpty {
                cur = [w]
                continue
            }
            let start = cur[0].start
            let dur = w.end - start
            let gap = w.start - (cur.last?.end ?? w.start)
            let punctBreak = "。！？!?".contains(where: { (cur.last?.text ?? "").contains($0) })
            if dur >= target && (gap >= 0.28 || punctBreak || dur >= cap) {
                flush()
                cur = [w]
            } else {
                cur.append(w)
            }
        }
        flush()
        if slices.count >= 2 {
            let last = slices[slices.count - 1]
            let prev = slices[slices.count - 2]
            if last.1 - last.0 < target * 0.45 {
                slices.removeLast()
                slices[slices.count - 1] = (prev.0, last.1, prev.2 + last.2)
            }
        }
        return slices
    }

    public static func sentences(from words: [TimedWord]) -> [TimedSentence] {
        guard !words.isEmpty else { return [] }
        var out: [TimedSentence] = []
        var cur: [TimedWord] = []
        func flush() {
            guard let first = cur.first, let last = cur.last else { return }
            let text = joinWords(cur).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty, !isPunctuationOnly(text) {
                out.append(TimedSentence(start: first.start, end: last.end, text: text, words: cur))
            }
            cur = []
        }
        for w in words {
            if cur.isEmpty {
                cur = [w]
                continue
            }
            let gap = w.start - (cur.last?.end ?? w.start)
            let ended = hasSentenceEnd(cur.last?.text ?? "")
            if ended || gap >= 0.45 {
                flush()
                cur = [w]
            } else {
                cur.append(w)
            }
        }
        flush()
        return out
    }

    public static func wrapCaption(_ text: String, lineChars: Int = 16, maxLines: Int = 3) -> String {
        let plain = text.replacingOccurrences(of: "\\N", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard displayLen(plain) > lineChars else { return plain }
        var lines: [String] = []
        var current = ""
        for ch in plain {
            if ch == "\n" {
                if !current.isEmpty { lines.append(current); current = "" }
                continue
            }
            current.append(ch)
            let scalar = ch.unicodeScalars.first
            let ended = scalar.map { sentenceEnders.contains($0) || clauseBreaks.contains($0) } ?? false
            if displayLen(current) >= lineChars && (ended || displayLen(current) >= lineChars + 6) {
                lines.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            }
        }
        if !current.isEmpty {
            lines.append(current.trimmingCharacters(in: .whitespaces))
        }
        if lines.count > maxLines {
            let head = Array(lines.prefix(maxLines - 1))
            let tail = lines.dropFirst(maxLines - 1).joined()
            lines = head + [tail]
        }
        return lines.filter { !$0.isEmpty }.joined(separator: "\\N")
    }

    public static func captionCues(from words: [TimedWord]) -> [TimedSentence] {
        let sents = sentences(from: words)
        var cues: [TimedSentence] = []
        for sent in sents {
            let longText = displayLen(sent.text) > 40
            let longTime = (sent.end - sent.start) > 8.0
            if longText || longTime {
                cues.append(contentsOf: splitClauses(sent))
            } else {
                var cue = sent
                cue.text = wrapCaption(sent.text)
                cues.append(cue)
            }
        }
        return clampCueEnds(cues)
    }

    public static func splitClauses(_ sent: TimedSentence, maxChars: Int = 22) -> [TimedSentence] {
        guard !sent.words.isEmpty else { return [sent] }
        var out: [TimedSentence] = []
        var cur: [TimedWord] = []
        func flush() {
            guard let first = cur.first, let last = cur.last else { return }
            let text = wrapCaption(joinWords(cur))
            if !text.isEmpty, !isPunctuationOnly(text.replacingOccurrences(of: "\\N", with: "")) {
                out.append(TimedSentence(start: first.start, end: last.end, text: text, words: cur))
            }
            cur = []
        }
        for w in sent.words {
            if cur.isEmpty {
                cur = [w]
                continue
            }
            let gap = w.start - (cur.last?.end ?? w.start)
            let clause = hasClauseBreak(cur.last?.text ?? "")
            let count = displayLen(joinWords(cur))
            if (clause && count >= 8) || gap >= 0.22 || count >= maxChars {
                flush()
                cur = [w]
            } else {
                cur.append(w)
            }
        }
        flush()
        return out.isEmpty ? [sent] : out
    }

    public static func clampCueEnds(_ cues: [TimedSentence], minDur: Double = 0.75) -> [TimedSentence] {
        guard !cues.isEmpty else { return [] }
        var out = cues
        for i in out.indices {
            let nextStart = i + 1 < out.count ? out[i + 1].start : out[i].end + 2
            var end = max(out[i].end, out[i].start + minDur)
            end = min(end, nextStart - 0.04)
            if end <= out[i].start {
                end = min(out[i].end, nextStart)
            }
            out[i].end = max(out[i].start + 0.04, end)
        }
        return out
    }

    public static func packTopics(
        sentences: [TimedSentence],
        maxDur: Double = 90,
        minDur: Double = 8,
        pauseBreak: Double = 1.05
    ) -> [VideoSliceSpec] {
        guard !sentences.isEmpty else { return [] }
        var groups: [[TimedSentence]] = []
        var cur: [TimedSentence] = []
        func flush() {
            guard !cur.isEmpty else { return }
            groups.append(cur)
            cur = []
        }
        for s in sentences {
            if cur.isEmpty {
                cur = [s]
                continue
            }
            let gap = s.start - (cur.last?.end ?? s.start)
            let dur = s.end - cur[0].start
            if gap >= pauseBreak || dur >= maxDur {
                flush()
                cur = [s]
            } else {
                cur.append(s)
            }
        }
        flush()
        if groups.count >= 2 {
            let last = groups[groups.count - 1]
            let lastDur = (last.last?.end ?? 0) - (last.first?.start ?? 0)
            if lastDur < minDur * 0.6 {
                groups[groups.count - 2].append(contentsOf: last)
                groups.removeLast()
            }
        }
        return groups.compactMap { spec(from: $0, title: nil, point: nil) }
    }

    public static func wordsInRange(_ words: [TimedWord], start: Double, end: Double) -> [TimedWord] {
        words.filter { $0.end > start - 0.02 && $0.start < end + 0.02 }
    }

    public static func applyPlan(_ plan: [PlannedSlice], words: [TimedWord]) -> [VideoSliceSpec] {
        let sents = sentences(from: words)
        if plan.contains(where: { !$0.sentenceIds.isEmpty }) {
            let fromSentences = applySentencePlan(plan, sentences: sents)
            if !fromSentences.isEmpty { return fromSentences }
        }
        return plan.compactMap { p in
            let ws = wordsInRange(words, start: p.start, end: p.end)
            guard let first = ws.first, let last = ws.last else { return nil }
            let text = joinWords(ws)
            let headline = resolveHeadline(proposedTitle: p.title, proposedPoint: p.point, source: text)
            return VideoSliceSpec(
                start: first.start,
                end: last.end,
                title: headline.title,
                point: headline.point,
                words: ws
            )
        }
    }

    public static func applySentencePlan(
        _ plan: [PlannedSlice],
        sentences: [TimedSentence]
    ) -> [VideoSliceSpec] {
        let n = sentences.count
        guard n > 0 else { return [] }
        var used = Set<Int>()
        var specs: [VideoSliceSpec] = []
        for p in plan {
            var ids = normalizeSentenceIds(p.sentenceIds, count: n)
            if ids.isEmpty {
                ids = sentences.indices.filter {
                    sentences[$0].end > p.start && sentences[$0].start < p.end
                }
            }
            ids = ids.filter { !used.contains($0) }.sorted()
            guard let first = ids.first, let last = ids.last else { continue }
            let range = Array(first...last)
            for i in range { used.insert(i) }
            if let spec = spec(
                from: range.map { sentences[$0] },
                title: p.title,
                point: p.point
            ) {
                specs.append(spec)
            }
        }
        var i = 0
        while i < n {
            if used.contains(i) {
                i += 1
                continue
            }
            var j = i
            while j < n && !used.contains(j) { j += 1 }
            if let spec = spec(
                from: Array(sentences[i..<j]),
                title: nil,
                point: nil
            ) {
                specs.append(spec)
            }
            i = j
        }
        return specs.sorted { $0.start < $1.start }
    }

    public static func normalizeSentenceIds(_ ids: [Int], count: Int) -> [Int] {
        let clipped = ids.filter { $0 >= 0 }
        guard !clipped.isEmpty, count > 0 else { return [] }
        let shifted = clipped.contains(0) ? clipped : clipped.map { $0 - 1 }
        return shifted.filter { $0 >= 0 && $0 < count }
    }

    public static func srt(from words: [TimedWord], origin: Double = 0) -> String {
        words.enumerated().map { i, w in
            let a = max(0, w.start - origin)
            let b = max(a + 0.04, w.end - origin)
            return """
            \(i + 1)
            \(srtTime(a)) --> \(srtTime(b))
            \(w.text)
            """
        }.joined(separator: "\n\n") + "\n"
    }

    public static func captionSRT(from words: [TimedWord], origin: Double = 0) -> String {
        let cues = captionCues(from: words)
        return srt(from: cues, origin: origin)
    }

    public static func srt(from cues: [TimedSentence], origin: Double = 0) -> String {
        cues.enumerated().map { i, c in
            let a = max(0, c.start - origin)
            let b = max(a + 0.04, c.end - origin)
            return """
            \(i + 1)
            \(srtTime(a)) --> \(srtTime(b))
            \(c.text)
            """
        }.joined(separator: "\n\n") + "\n"
    }

    public static func srtTime(_ seconds: Double) -> String {
        let ms = max(0, Int((seconds * 1000).rounded()))
        let h = ms / 3_600_000
        let m = (ms / 60_000) % 60
        let s = (ms / 1000) % 60
        let milli = ms % 1000
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, milli)
    }

    public static func run(_ launchPath: String, _ args: [String]) throws {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: launchPath)
        proc.arguments = args
        let err = Pipe()
        proc.standardError = err
        proc.standardOutput = Pipe()
        try proc.run()
        proc.waitUntilExit()
        if proc.terminationStatus != 0 {
            let msg = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            throw NovaMLXError.apiError("ffmpeg failed (\(proc.terminationStatus)): \(msg.suffix(400))")
        }
    }

    public static func probeDuration(_ url: URL) throws -> Double {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffprobePath())
        proc.arguments = ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", url.path]
        let out = Pipe()
        proc.standardOutput = out
        proc.standardError = Pipe()
        try proc.run()
        proc.waitUntilExit()
        let s = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "0"
        return Double(s) ?? 0
    }

    public static func extractAudio(from video: URL, to wav: URL) throws {
        try run(ffmpegPath(), [
            "-y", "-i", video.path, "-vn", "-ac", "1", "-ar", "16000", wav.path,
        ])
    }

    public static func extractAudioSegment(from wav: URL, start: Double, duration: Double, to dest: URL) throws {
        try run(ffmpegPath(), [
            "-y", "-ss", String(format: "%.3f", start), "-t", String(format: "%.3f", duration),
            "-i", wav.path, dest.path,
        ])
    }

    public static func cutOverlay(
        video: URL,
        start: Double,
        duration: Double,
        title: String,
        point: String,
        srtURL: URL,
        titleStyle: TitleOverlayStyle,
        pointStyle: PointOverlayStyle,
        output: URL
    ) throws {
        let font = pingFangFont().replacingOccurrences(of: ":", with: "\\:")
        let titleEsc = escapeDrawtext(title)
        let pointEsc = escapeDrawtext(point)
        let titleFilter = titleDrawtext(
            style: titleStyle,
            text: titleEsc,
            font: font,
            fontSize: titleFontSize(text: title, style: titleStyle)
        )
        let pointFilter = pointDrawtext(style: pointStyle, text: pointEsc, font: font)
        // Alignment 2 = bottom-center so full-sentence captions sit under title/point.
        let subs = "subtitles='\(escapeFilterPath(srtURL.path))':force_style='FontName=PingFang SC,FontSize=28,Outline=2,Shadow=1,Alignment=2,MarginL=70,MarginR=70,MarginV=110,WrapStyle=2'"
        let vf = "\(titleFilter),\(pointFilter),\(subs)"
        try run(ffmpegPath(), [
            "-y", "-ss", String(format: "%.3f", start), "-t", String(format: "%.3f", duration),
            "-i", video.path,
            "-vf", vf,
            "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart",
            output.path,
        ])
    }

    public static func parseLLMJSON(_ raw: String) -> (title: String, point: String) {
        if let obj = extractJSONValue(raw) as? [String: Any] {
            let t = (obj["title"] as? String) ?? (obj["大标题"] as? String) ?? ""
            let p = (obj["point"] as? String) ?? (obj["核心观点"] as? String) ?? ""
            if !t.isEmpty { return (t, p) }
        }
        let trimmed = stripCodeFence(raw)
        let line = trimmed.split(whereSeparator: \.isNewline).first.map(String.init) ?? "片段"
        return (String(line.prefix(18)), String(trimmed.prefix(40)))
    }

    public static func parseSlicePlan(_ raw: String) -> [PlannedSlice] {
        func intList(_ value: Any?) -> [Int] {
            if let arr = value as? [Int] { return arr }
            if let arr = value as? [Double] { return arr.map { Int($0) } }
            if let arr = value as? [Any] {
                return arr.compactMap { item in
                    (item as? Int) ?? (item as? Double).map { Int($0) }
                        ?? (item as? String).flatMap { Int($0) }
                }
            }
            return []
        }
        func slices(from obj: Any) -> [PlannedSlice] {
            let rows: [[String: Any]]
            if let dict = obj as? [String: Any] {
                rows = (dict["slices"] as? [[String: Any]])
                    ?? (dict["clips"] as? [[String: Any]])
                    ?? []
            } else if let arr = obj as? [[String: Any]] {
                rows = arr
            } else {
                return []
            }
            return rows.compactMap { row in
                let ids = intList(row["sentences"] ?? row["sentence_ids"] ?? row["ids"])
                let start = (row["start"] as? Double)
                    ?? (row["start"] as? Int).map(Double.init)
                    ?? (row["start_sec"] as? Double)
                    ?? 0
                let end = (row["end"] as? Double)
                    ?? (row["end"] as? Int).map(Double.init)
                    ?? (row["end_sec"] as? Double)
                    ?? 0
                if ids.isEmpty, end <= start { return nil }
                let title = (row["title"] as? String)
                    ?? (row["大标题"] as? String)
                    ?? ""
                let point = (row["point"] as? String)
                    ?? (row["核心观点"] as? String)
                    ?? ""
                return PlannedSlice(
                    start: start, end: end, title: title, point: point, sentenceIds: ids)
            }
        }
        if let obj = extractJSONValue(raw) {
            return slices(from: obj)
        }
        return []
    }

    private static func spec(
        from sents: [TimedSentence],
        title: String?,
        point: String?
    ) -> VideoSliceSpec? {
        guard let first = sents.first, let last = sents.last else { return nil }
        let words = sents.flatMap(\.words)
        let text = sents.map(\.text).joined()
        let headline = resolveHeadline(proposedTitle: title ?? "", proposedPoint: point ?? "", source: text)
        return VideoSliceSpec(
            start: first.start,
            end: last.end,
            title: headline.title,
            point: headline.point,
            words: words.isEmpty
                ? [TimedWord(text: text, start: first.start, end: last.end)]
                : words
        )
    }

    private static func hasSentenceEnd(_ text: String) -> Bool {
        text.unicodeScalars.contains { sentenceEnders.contains($0) }
    }

    private static func hasClauseBreak(_ text: String) -> Bool {
        text.unicodeScalars.contains { clauseBreaks.contains($0) || sentenceEnders.contains($0) }
    }

    private static func isPunctuationOnly(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        return trimmed.unicodeScalars.allSatisfy {
            CharacterSet.punctuationCharacters.contains($0)
                || CharacterSet.whitespacesAndNewlines.contains($0)
        }
    }

    private static func shouldInsertSpace(_ left: String, _ right: String) -> Bool {
        guard let a = left.unicodeScalars.last, let b = right.unicodeScalars.first else { return false }
        if CharacterSet.punctuationCharacters.contains(b) { return false }
        if CharacterSet.punctuationCharacters.contains(a) && isCJK(b) { return false }
        if isCJK(a) || isCJK(b) { return false }
        if a == " " || b == " " { return false }
        return true
    }

    private static func isCJK(_ ch: Character) -> Bool {
        ch.unicodeScalars.contains { isCJK($0) }
    }

    private static func isLatinOrDigit(_ ch: Character) -> Bool {
        ch.isASCII && (ch.isLetter || ch.isNumber)
    }

    private static func compactForCut(_ text: String) -> String {
        String(text.filter { ch in
            guard let scalar = ch.unicodeScalars.first else { return false }
            if CharacterSet.whitespacesAndNewlines.contains(scalar) { return false }
            if CharacterSet.punctuationCharacters.contains(scalar) { return false }
            return true
        })
    }

    private static func splitSentences(_ text: String) -> [String] {
        var out: [String] = []
        var cur = ""
        for ch in text {
            cur.append(ch)
            if let scalar = ch.unicodeScalars.first, sentenceEnders.contains(scalar) {
                let piece = sanitizeHeadline(cur)
                if !piece.isEmpty { out.append(piece) }
                cur = ""
            }
        }
        let tail = sanitizeHeadline(cur)
        if !tail.isEmpty { out.append(tail) }
        return out
    }

    private static func completedLine(_ text: String, maxLen: Int) -> String {
        var s = sanitizeHeadline(text)
        guard !s.isEmpty else { return "" }
        if displayLen(s) <= maxLen && !isDanglingTitle(s) { return s }
        var best = ""
        var current = ""
        for ch in s {
            current.append(ch)
            if displayLen(current) > maxLen { break }
            guard let scalar = ch.unicodeScalars.first else { continue }
            if sentenceEnders.contains(scalar) || clauseBreaks.contains(scalar) {
                let piece = sanitizeHeadline(current)
                if displayLen(piece) >= 4 && !isDanglingTitle(piece) {
                    best = piece
                }
            }
        }
        if !best.isEmpty { return best }
        var hard = ""
        for ch in s {
            hard.append(ch)
            if displayLen(hard) >= maxLen { break }
        }
        hard = sanitizeHeadline(hard)
        while isDanglingTitle(hard) && displayLen(hard) > 4 {
            hard.removeLast()
            hard = sanitizeHeadline(hard)
        }
        return hard
    }

    private static func arabicDigit(_ raw: String) -> String {
        guard let ch = raw.first else { return raw }
        switch ch {
        case "0", "1", "2", "3", "4", "5", "6", "7", "8", "9":
            return String(ch)
        case "一": return "1"
        case "二", "两": return "2"
        case "三": return "3"
        case "四": return "4"
        case "五": return "5"
        case "六": return "6"
        case "七": return "7"
        case "八": return "8"
        case "九": return "9"
        default: return String(ch)
        }
    }

    private static func replaceDigits(
        pattern: String,
        in text: String,
        rewrite: (String) -> String
    ) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return text }
        let ns = text as NSString
        let matches = re.matches(in: text, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return text }
        let mutable = NSMutableString(string: text)
        for match in matches.reversed() {
            guard match.numberOfRanges > 1 else { continue }
            let digit = mutable.substring(with: match.range(at: 1))
            mutable.replaceCharacters(in: match.range(at: 0), with: rewrite(arabicDigit(digit)))
        }
        return mutable as String
    }

    private static func titleFontSize(text: String, style: TitleOverlayStyle) -> Int {
        let base: Int
        switch style {
        case .news: base = 48
        case .youtube: base = 50
        case .poster: base = 52
        case .minimal: base = 44
        }
        let n = displayLen(text)
        if n <= 14 { return base }
        if n <= 20 { return max(34, base - 8) }
        if n <= 28 { return max(30, base - 14) }
        return max(26, base - 18)
    }

    private static func isCJK(_ s: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(s.value)
            || (0x3400...0x4DBF).contains(s.value)
            || (0x3040...0x30FF).contains(s.value)
    }

    private static func escapeDrawtext(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: ":", with: "\\:")
            .replacingOccurrences(of: "'", with: "’")
            .replacingOccurrences(of: "%", with: "%%")
            .replacingOccurrences(of: "\n", with: " ")
    }

    private static func escapeFilterPath(_ path: String) -> String {
        path.replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: ":", with: "\\:")
            .replacingOccurrences(of: "'", with: "\\'")
    }

    private static func titleDrawtext(
        style: TitleOverlayStyle,
        text: String,
        font: String,
        fontSize: Int
    ) -> String {
        let base = "drawtext=fontfile='\(font)':text='\(text)':x=(w-text_w)/2:y=h*0.04:fontsize=\(fontSize)"
        switch style {
        case .news:
            return "\(base):fontcolor=white:borderw=4:bordercolor=black"
        case .youtube:
            return "\(base):fontcolor=yellow:borderw=3:bordercolor=black:shadowx=2:shadowy=2"
        case .poster:
            return "\(base):fontcolor=white:box=1:boxcolor=red@0.85:boxborderw=16"
        case .minimal:
            return "\(base):fontcolor=white:borderw=2:bordercolor=black@0.6"
        }
    }

    private static func pointDrawtext(style: PointOverlayStyle, text: String, font: String) -> String {
        let base = "drawtext=fontfile='\(font)':text='\(text)':x=(w-text_w)/2:y=h*0.13"
        switch style {
        case .box:
            return "\(base):fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12"
        case .bar:
            return "\(base):fontsize=28:fontcolor=white:borderw=2:bordercolor=0x2EA043"
        case .outline:
            return "\(base):fontsize=28:fontcolor=white:borderw=3:bordercolor=black"
        case .caption:
            return "\(base):fontsize=26:fontcolor=0xF5F5F5:borderw=1:bordercolor=black@0.8"
        }
    }
}
