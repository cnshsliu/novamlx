import Foundation
import Testing
@testable import NovaMLXEngine

@Suite("Video slice pipeline")
struct VideoSlicePipelineTests {
    @Test("filler and stutter words are dropped")
    func stripFillers() {
        let words = [
            TimedWord(text: "嗯", start: 0, end: 0.1),
            TimedWord(text: "今天", start: 0.1, end: 0.3),
            TimedWord(text: "今天", start: 0.3, end: 0.4),
            TimedWord(text: "天气", start: 0.4, end: 0.6),
            TimedWord(text: "那个", start: 0.6, end: 0.7),
            TimedWord(text: "很好", start: 0.7, end: 0.9),
        ]
        let out = VideoSlicePipeline.stripFillers(words)
        #expect(out.map(\.text) == ["今天", "天气", "很好"])
        #expect(out[0].start == 0.1)
        #expect(out[2].end == 0.9)
    }

    @Test("slices pack near the target duration")
    func packSlices() {
        var words: [TimedWord] = []
        for i in 0..<40 {
            let t = Double(i) * 0.8
            words.append(TimedWord(text: "字", start: t, end: t + 0.5))
        }
        let slices = VideoSlicePipeline.packSlices(words: words, target: 8)
        #expect(!slices.isEmpty)
        for s in slices {
            #expect(s.end - s.start <= 12)
        }
    }

    @Test("LLM JSON parser accepts title/point")
    func parseLLM() {
        let r = VideoSlicePipeline.parseLLMJSON("{\"title\":\"标题\",\"point\":\"观点\"}")
        #expect(r.title == "标题")
        #expect(r.point == "观点")
    }

    @Test("markup strip keeps body text")
    func stripHTML() {
        let t = VideoSlicePipeline.stripMarkup("<p>今天<strong>天气</strong></p>")
        #expect(t.contains("今天"))
        #expect(!t.contains("<"))
    }

    @Test("SRT uses word timestamps relative to origin")
    func srtRelative() {
        let words = [
            TimedWord(text: "今", start: 10.0, end: 10.12),
            TimedWord(text: "天", start: 10.12, end: 10.28),
        ]
        let srt = VideoSlicePipeline.srt(from: words, origin: 10.0)
        #expect(srt.contains("00:00:00,000 --> 00:00:00,120"))
        #expect(srt.contains("今"))
        #expect(srt.contains("00:00:00,120 --> 00:00:00,280"))
    }

    @Test("sentence captions show a whole sentence in one cue")
    func sentenceCaptions() {
        let words = [
            TimedWord(text: "今", start: 0.0, end: 0.12),
            TimedWord(text: "天", start: 0.12, end: 0.24),
            TimedWord(text: "天", start: 0.24, end: 0.36),
            TimedWord(text: "气", start: 0.36, end: 0.48),
            TimedWord(text: "不", start: 0.48, end: 0.60),
            TimedWord(text: "错", start: 0.60, end: 0.72),
            TimedWord(text: "。", start: 0.72, end: 0.80),
            TimedWord(text: "我", start: 1.40, end: 1.52),
            TimedWord(text: "们", start: 1.52, end: 1.64),
            TimedWord(text: "出", start: 1.64, end: 1.76),
            TimedWord(text: "发", start: 1.76, end: 1.88),
            TimedWord(text: "。", start: 1.88, end: 1.96),
        ]
        let srt = VideoSlicePipeline.captionSRT(from: words)
        #expect(srt.contains("今天天气不错。"))
        #expect(srt.contains("我们出发。"))
        #expect(!srt.contains("00:00:00,000 --> 00:00:00,120\n今\n"))
        let cues = VideoSlicePipeline.captionCues(from: words)
        #expect(cues.count == 2)
        #expect(cues[0].text.contains("今天天气不错"))
        #expect(cues[1].text.contains("我们出发"))
    }

    @Test("wrapCaption breaks long lines without splitting the cue")
    func wrapCaption() {
        let t = VideoSlicePipeline.wrapCaption("这是一句比较长的中文口播需要折行显示出来", lineChars: 10)
        #expect(t.contains("\\N"))
        #expect(!t.contains("\n"))
    }

    @Test("LLM slice plan maps words into clips")
    func slicePlan() {
        let json = """
        {"slices":[{"start":0.0,"end":8.0,"title":"开场","point":"先讲背景"},{"start":8.0,"end":20.0,"title":"结论","point":"立刻行动"}]}
        """
        let plan = VideoSlicePipeline.parseSlicePlan(json)
        #expect(plan.count == 2)
        #expect(plan[0].title == "开场")
        var words: [TimedWord] = []
        for i in 0..<20 {
            let t = Double(i)
            words.append(TimedWord(text: "字", start: t, end: t + 0.8))
        }
        let specs = VideoSlicePipeline.applyPlan(plan, words: words)
        #expect(specs.count == 2)
        #expect(specs[0].title == "开场")
        #expect(specs[0].end <= 8.9)
    }

    @Test("theme plan uses sentence ids and covers leftovers")
    func themeSentenceIds() {
        let sents = [
            TimedSentence(start: 0, end: 5, text: "先讲背景。", words: [
                TimedWord(text: "先讲背景。", start: 0, end: 5),
            ]),
            TimedSentence(start: 5.2, end: 12, text: "再讲方法。", words: [
                TimedWord(text: "再讲方法。", start: 5.2, end: 12),
            ]),
            TimedSentence(start: 13, end: 20, text: "最后行动。", words: [
                TimedWord(text: "最后行动。", start: 13, end: 20),
            ]),
        ]
        let json = """
        {"slices":[{"sentences":[1,2],"title":"方法","point":"怎么做"},{"sentences":[3],"title":"行动","point":"马上做"}]}
        """
        let plan = VideoSlicePipeline.parseSlicePlan(json)
        #expect(plan[0].sentenceIds == [1, 2])
        let specs = VideoSlicePipeline.applySentencePlan(plan, sentences: sents)
        #expect(specs.count == 2)
        #expect(specs[0].title == "方法")
        #expect(specs[0].end == 12)
        #expect(specs[1].title == "行动")
        #expect(VideoSlicePipeline.joinWords(specs[0].words).contains("背景"))
    }

    @Test("theme leftover sentences become their own clip")
    func leftoverSentences() {
        let sents = [
            TimedSentence(start: 0, end: 4, text: "A。", words: [TimedWord(text: "A。", start: 0, end: 4)]),
            TimedSentence(start: 5, end: 9, text: "B。", words: [TimedWord(text: "B。", start: 5, end: 9)]),
            TimedSentence(start: 10, end: 14, text: "C。", words: [TimedWord(text: "C。", start: 10, end: 14)]),
        ]
        let plan = [
            PlannedSlice(start: 0, end: 0, title: "只切第一句", point: "A", sentenceIds: [1]),
        ]
        let specs = VideoSlicePipeline.applySentencePlan(plan, sentences: sents)
        #expect(specs.count == 2)
        #expect(specs[0].title == "只切第一句")
        #expect(specs[1].start == 5)
    }

    @Test("packTopics splits on pauses not a 25s clock")
    func packTopicsByPause() {
        let sents = [
            TimedSentence(start: 0, end: 8, text: "主题一上。", words: [TimedWord(text: "主题一上。", start: 0, end: 8)]),
            TimedSentence(start: 8.2, end: 16, text: "主题一下。", words: [TimedWord(text: "主题一下。", start: 8.2, end: 16)]),
            TimedSentence(start: 18.5, end: 28, text: "主题二。", words: [TimedWord(text: "主题二。", start: 18.5, end: 28)]),
        ]
        let specs = VideoSlicePipeline.packTopics(sentences: sents, maxDur: 90, pauseBreak: 1.05)
        #expect(specs.count == 2)
        #expect(specs[0].end == 16)
        #expect(specs[1].start == 18.5)
        #expect(specs[0].end - specs[0].start > 10)
    }

    @Test("proofread JSON maps onto ASR chunks")
    func parseCorrectedChunks() {
        let fallback = ["今天天汽不错", "我们出法"]
        let json = """
        {"chunks":[{"i":0,"text":"今天天气不错。"},{"i":1,"text":"我们出发。"}]}
        """
        let out = VideoSlicePipeline.parseCorrectedChunks(json, fallback: fallback)
        #expect(out == ["今天天气不错。", "我们出发。"])
    }

    @Test("implausible LLM summary is rejected")
    func rejectSummary() {
        let fallback = ["今天我们讨论了三个完全不同的产品策略并且每一个都需要单独展开说明"]
        let json = "{\"text\":\"总结：讲了产品\"}"
        let out = VideoSlicePipeline.parseCorrectedChunks(json, fallback: fallback)
        #expect(out == fallback)
    }

    @Test("tidyTranscript strips particles and stutters")
    func tidyTranscript() {
        let raw = "我不知道有多少小白哈，很多小白可能可能因为每个人的每个人的这个，哎，的的了解这个行业。比方比如说什么是投工投工工厂啊，那么呃，通用工厂。我先比比比打个比喻。天天用用电。我我先把电的逻辑跟大家。"
        let t = VideoSlicePipeline.tidyTranscript(raw)
        #expect(!t.contains("哈"))
        #expect(!t.contains("可能可能"))
        #expect(!t.contains("每个人的每个人的"))
        #expect(!t.contains("投工投工"))
        #expect(t.contains("投工工厂"))
        #expect(t.contains("通用工厂"))
        #expect(!t.contains("比比比"))
        #expect(t.contains("打个比喻"))
        #expect(!t.contains("比打个比喻"))
        #expect(t.contains("用电"))
        #expect(!t.contains("用用电"))
        #expect(!t.contains("我我"))
        #expect(t.contains("比如说"))
        #expect(t.contains("天天"))
    }

    @Test("tidy leaves product names for the LLM")
    func tidyDoesNotRewriteNames() {
        let heard = "刷皮的飞豹五呢确实是非常贵"
        let t = VideoSlicePipeline.tidyTranscript(heard)
        #expect(t.contains("飞豹五"))
        #expect(!t.contains("Fable"))
    }

    @Test("proper noun file merges with the page glossary")
    func properNounFile() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx-nouns-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("proper-nouns.json")
        try VideoSlicePipeline.ensureProperNounFile(at: url)
        #expect(VideoSlicePipeline.properNounTerms(at: url).isEmpty)
        let body = "{\"terms\":[\"Fable 5\",\"Methos\",\"Fable 5\"]}"
        try body.write(to: url, atomically: true, encoding: .utf8)
        let merged = VideoSlicePipeline.mergedGlossary(
            uiText: "DeepSeek V4 Pro",
            fileTerms: VideoSlicePipeline.properNounTerms(at: url)
        )
        #expect(merged == ["Fable 5", "Methos", "DeepSeek V4 Pro"])
    }

    @Test("glossary becomes an ASR vocabulary hint")
    func asrVocabulary() {
        let ctx = VideoSlicePipeline.asrContext(glossary: "Anthropic, Fable 5、Fable 5")
        #expect(ctx == "Vocabulary: Anthropic, Fable 5")
        let fromScript = VideoSlicePipeline.asrContext(
            glossary: "",
            reference: "今天聊 Anthropic 的 Fable 5，以及 the price。"
        )
        #expect(fromScript.contains("Anthropic"))
        #expect(fromScript.contains("Fable 5"))
        #expect(!fromScript.contains("the"))
    }

    @Test("a 14-character opening cut is not a headline")
    func titleIsNotPrefixCut() {
        let spoken = "这套方案确实是非常贵但确实是太好。后面再讲它贵在哪里。"
        let source = VideoSlicePipeline.tidyTranscript(spoken)
        let cut = String(source.prefix(14))
        #expect(!VideoSlicePipeline.isAcceptableHeadline(cut, source: source))
        let chopped = "这套方案确实是非常贵但"
        #expect(!VideoSlicePipeline.isAcceptableHeadline(chopped, source: source))
        let resolved = VideoSlicePipeline.resolveHeadline(
            proposedTitle: cut,
            proposedPoint: cut,
            source: source
        )
        #expect(!resolved.acceptedProposal)
        #expect(resolved.title.contains("确实是太好"))
        #expect(resolved.point.contains("贵在哪里"))
        let summary = "这套方案很贵但很好"
        let ok = VideoSlicePipeline.resolveHeadline(
            proposedTitle: summary,
            proposedPoint: "贵，但确实好用",
            source: source
        )
        #expect(ok.acceptedProposal)
        #expect(ok.title == summary)
        #expect(ok.point == "贵，但确实好用")
    }

    @Test("headline JSON maps onto clips")
    func parseHeadlines() {
        let raw = """
        {"items":[{"i":1,"title":"Fable 5很贵但很好","point":"贵，但值得"},{"i":0,"title":"先讲背景","point":"为什么要提"}]}
        """
        let rows = VideoSlicePipeline.parseHeadlines(raw, count: 2)
        #expect(rows[0].title == "先讲背景")
        #expect(rows[1].title == "Fable 5很贵但很好")
        #expect(rows[1].point == "贵，但值得")
    }

    @Test("joinWords does not space CJK")
    func joinCJK() {
        let words = [
            TimedWord(text: "今", start: 0, end: 0.1),
            TimedWord(text: "天", start: 0.1, end: 0.2),
            TimedWord(text: "。", start: 0.2, end: 0.3),
        ]
        #expect(VideoSlicePipeline.joinWords(words) == "今天。")
    }

    @Test("output directory setting is required and must still be openable")
    func inspectOutputDirectory() throws {
        #expect(VideoSlicePipeline.inspectOutputDirectory("") == .missingSetting)
        #expect(VideoSlicePipeline.inspectOutputDirectory("   ") == .missingSetting)
        let missing = "/tmp/novamlx-missing-output-\(UUID().uuidString)"
        if case .unreachable(let path) = VideoSlicePipeline.inspectOutputDirectory(missing) {
            #expect(path.contains("novamlx-missing-output-"))
        } else {
            Issue.record("expected unreachable")
        }
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx-out-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("not-a-folder.txt")
        try "x".write(to: file, atomically: true, encoding: .utf8)
        if case .notDirectory = VideoSlicePipeline.inspectOutputDirectory(file.path) {
            // ok
        } else {
            Issue.record("expected notDirectory")
        }
        if case .ready(let url) = VideoSlicePipeline.inspectOutputDirectory(dir.path) {
            #expect(url.path == dir.path)
        } else {
            Issue.record("expected ready")
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir.path)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir.path)
        }
        let locked = VideoSlicePipeline.inspectOutputDirectory(dir.path)
        #expect(locked == .notWritable(path: dir.path) || locked.isReady)
    }

    @Test("job folder name includes video stem and timestamp")
    func jobFolderName() {
        let now = Date(timeIntervalSince1970: 1_779_000_000)
        let name = VideoSlicePipeline.jobFolderName(videoName: "9月7日-03.mov", now: now)
        #expect(name.hasPrefix("9月7日-03-"))
        #expect(!name.contains("/"))
        #expect(!name.contains(":"))
    }

    @Test("ffmpeg burns title, point, and captions")
    func overlay() throws {
        let ffmpeg = VideoSlicePipeline.ffmpegPath()
        try #require(FileManager.default.isExecutableFile(atPath: ffmpeg))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx-slice-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("src.mp4")
        try VideoSlicePipeline.run(ffmpeg, [
            "-y", "-f", "lavfi", "-i", "color=c=black:s=640x360:d=2",
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", "2", "-shortest", "-c:v", "libx264", "-c:a", "aac", src.path,
        ])
        let srt = dir.appendingPathComponent("cap.srt")
        try "1\n00:00:00,000 --> 00:00:01,500\n今天天气不错。\n".write(to: srt, atomically: true, encoding: .utf8)
        let out = dir.appendingPathComponent("out.mp4")
        try VideoSlicePipeline.cutOverlay(
            video: src,
            start: 0,
            duration: 2,
            title: "大标题",
            point: "核心观点",
            srtURL: srt,
            titleStyle: .news,
            pointStyle: .box,
            output: out
        )
        #expect(FileManager.default.fileExists(atPath: out.path))
        let size = try FileManager.default.attributesOfItem(atPath: out.path)[.size] as? NSNumber
        #expect((size?.intValue ?? 0) > 1000)
    }

    @Test("sentence captions burn onto the 视频号 sample clip")
    func realVideoSentenceOverlay() throws {
        let src = URL(fileURLWithPath: "/Volumes/WD/Media/shipinhao/9月7日-03.mov")
        try #require(FileManager.default.fileExists(atPath: src.path))
        let ffmpeg = VideoSlicePipeline.ffmpegPath()
        try #require(FileManager.default.isExecutableFile(atPath: ffmpeg))
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx-slice-real-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let clip = dir.appendingPathComponent("clip.mp4")
        try VideoSlicePipeline.run(ffmpeg, [
            "-y", "-ss", "8", "-t", "6", "-i", src.path,
            "-c:v", "libx264", "-c:a", "aac", "-an", clip.path,
        ])
        let words = [
            TimedWord(text: "这", start: 0.0, end: 0.12),
            TimedWord(text: "是", start: 0.12, end: 0.24),
            TimedWord(text: "一", start: 0.24, end: 0.36),
            TimedWord(text: "句", start: 0.36, end: 0.48),
            TimedWord(text: "完", start: 0.48, end: 0.60),
            TimedWord(text: "整", start: 0.60, end: 0.72),
            TimedWord(text: "的", start: 0.72, end: 0.84),
            TimedWord(text: "话", start: 0.84, end: 0.96),
            TimedWord(text: "。", start: 0.96, end: 1.10),
            TimedWord(text: "下", start: 2.0, end: 2.12),
            TimedWord(text: "一", start: 2.12, end: 2.24),
            TimedWord(text: "句", start: 2.24, end: 2.36),
            TimedWord(text: "整", start: 2.36, end: 2.48),
            TimedWord(text: "体", start: 2.48, end: 2.60),
            TimedWord(text: "出", start: 2.60, end: 2.72),
            TimedWord(text: "现", start: 2.72, end: 2.84),
            TimedWord(text: "。", start: 2.84, end: 3.00),
        ]
        let srt = VideoSlicePipeline.captionSRT(from: words)
        #expect(srt.contains("这是一句完整的话。"))
        #expect(srt.contains("下一句整体出现。"))
        let srtURL = dir.appendingPathComponent("cap.srt")
        try srt.write(to: srtURL, atomically: true, encoding: .utf8)
        let out = dir.appendingPathComponent("out.mp4")
        try VideoSlicePipeline.cutOverlay(
            video: clip,
            start: 0,
            duration: 6,
            title: "完整主题",
            point: "整句字幕不是逐字跳",
            srtURL: srtURL,
            titleStyle: .news,
            pointStyle: .box,
            output: out
        )
        #expect(FileManager.default.fileExists(atPath: out.path))
        let size = try FileManager.default.attributesOfItem(atPath: out.path)[.size] as? NSNumber
        #expect((size?.intValue ?? 0) > 5000)
    }
}
