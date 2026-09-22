import SwiftUI
import AppKit
import UniformTypeIdentifiers
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

struct VideoSlicePageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n

    @AppStorage("novamlx.videoSlice.videoPath") private var videoPath = ""
    @State private var videoURL: URL?
    @State private var referenceURL: URL?
    @State private var referenceText = ""
    @State private var sliceSeconds: Double = 90
    @State private var removeFillers = true
    @State private var titleStyle: TitleOverlayStyle = .news
    @State private var pointStyle: PointOverlayStyle = .box
    @AppStorage("novamlx.videoSlice.outputDir") private var outputDirPath = ""
    @AppStorage("novamlx.videoSlice.glossary") private var glossary = ""
    @State private var logLines: [String] = []
    @State private var isRunning = false
    @State private var isLoadingStack = false
    @State private var produced: [URL] = []
    @State private var errorText: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    controls
                    if let errorText {
                        Text(errorText).font(.caption).foregroundColor(.red)
                    }
                    logView
                    if !produced.isEmpty {
                        Text(l10n.tr("videoSlice.outputs")).font(.headline)
                        ForEach(produced, id: \.path) { url in
                            Button(url.lastPathComponent) {
                                NSWorkspace.shared.activateFileViewerSelecting([url])
                            }
                            .buttonStyle(.plain)
                            .font(.system(size: 12, design: .monospaced))
                        }
                    }
                }
                .padding(16)
            }
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            refreshOutputError()
            restoreVideo()
        }
        .onChange(of: outputDirPath) { _, _ in refreshOutputError() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(l10n.tr("app.videoSlice")).font(.title2.bold())
            Text(l10n.tr("videoSlice.subtitle")).font(.caption).foregroundColor(.secondary)
            HStack {
                Button {
                    loadStack()
                } label: {
                    if isLoadingStack { ProgressView().controlSize(.small) }
                    Text(l10n.tr("videoSlice.loadStack"))
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLoadingStack || isRunning)
                Spacer()
            }
        }
        .padding(16)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button(l10n.tr("videoSlice.pickVideo"), action: pickVideo)
                Text(videoLabel)
                    .font(.caption)
                    .foregroundColor(videoURL == nil && !videoPath.isEmpty ? .red : .secondary)
                    .lineLimit(1)
                    .help(videoPath)
            }
            HStack {
                Button(l10n.tr("videoSlice.pickRef"), action: pickReference)
                Text(referenceURL?.lastPathComponent ?? l10n.tr("videoSlice.noRef"))
                    .font(.caption).foregroundColor(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(l10n.tr("videoSlice.glossary")).font(.caption)
                TextField(l10n.tr("videoSlice.glossaryHint"), text: $glossary)
                    .textFieldStyle(.roundedBorder)
            }
            HStack {
                Text(String(format: l10n.tr("videoSlice.duration"), Int(sliceSeconds)))
                Slider(value: $sliceSeconds, in: 40...150, step: 5)
            }
            Toggle(l10n.tr("videoSlice.removeFillers"), isOn: $removeFillers)
            Picker(l10n.tr("videoSlice.titleStyle"), selection: $titleStyle) {
                ForEach(TitleOverlayStyle.allCases) { s in
                    Text(l10n.tr("videoSlice.title.\(s.rawValue)")).tag(s)
                }
            }
            .pickerStyle(.segmented)
            Picker(l10n.tr("videoSlice.pointStyle"), selection: $pointStyle) {
                ForEach(PointOverlayStyle.allCases) { s in
                    Text(l10n.tr("videoSlice.point.\(s.rawValue)")).tag(s)
                }
            }
            .pickerStyle(.segmented)
            HStack {
                Button(l10n.tr("videoSlice.pickOut"), action: pickOutput)
                Text(outputDirLabel)
                    .font(.caption2)
                    .foregroundColor(outputDirColor)
                    .lineLimit(2)
                    .help(outputDirPath)
                if case .ready(let url) = VideoSlicePipeline.inspectOutputDirectory(outputDirPath) {
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: {
                        Image(systemName: "folder")
                    }
                    .buttonStyle(.borderless)
                    .help(url.path)
                }
            }
            Button {
                runPipeline()
            } label: {
                if isRunning { ProgressView().controlSize(.small) }
                Text(l10n.tr("videoSlice.run"))
            }
            .buttonStyle(.borderedProminent)
            .disabled(isRunning || videoURL == nil)
        }
    }

    private var logView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(logLines.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.system(size: 11, design: .monospaced))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minHeight: 120)
        .padding(8)
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func log(_ s: String) {
        logLines.append(s)
    }

    private var videoLabel: String {
        if let videoURL { return videoURL.lastPathComponent }
        if !videoPath.isEmpty {
            return URL(fileURLWithPath: videoPath).lastPathComponent
        }
        return l10n.tr("videoSlice.noVideo")
    }

    private func restoreVideo() {
        let trimmed = videoPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let url = URL(fileURLWithPath: trimmed)
        if FileManager.default.isReadableFile(atPath: url.path) {
            videoURL = url
            if isRememberedVideoError { errorText = nil }
        } else {
            videoURL = nil
            errorText = String(format: l10n.tr("videoSlice.videoMissing"), trimmed)
        }
    }

    private var isRememberedVideoError: Bool {
        guard let errorText else { return false }
        let marker = l10n.tr("videoSlice.videoMissing").replacingOccurrences(of: "%@", with: "")
        return !marker.isEmpty && errorText.contains(marker)
    }

    private func pickVideo() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [
            .movie, .mpeg4Movie, .quickTimeMovie, .avi,
            UTType(filenameExtension: "mkv") ?? .movie,
            UTType(filenameExtension: "webm") ?? .movie,
        ]
        p.allowsMultipleSelection = false
        p.canChooseDirectories = false
        let remembered = videoURL ?? (videoPath.isEmpty ? nil : URL(fileURLWithPath: videoPath))
        if let remembered {
            p.directoryURL = remembered.deletingLastPathComponent()
            p.nameFieldStringValue = remembered.lastPathComponent
        }
        guard p.runModal() == .OK, let url = p.url else { return }
        videoURL = url
        videoPath = url.path
        if isRememberedVideoError { errorText = nil }
    }

    private func pickReference() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.plainText, .text, .html, UTType(filenameExtension: "md") ?? .plainText]
        guard p.runModal() == .OK, let url = p.url else { return }
        referenceURL = url
        if let raw = try? String(contentsOf: url, encoding: .utf8) {
            referenceText = VideoSlicePipeline.stripMarkup(raw)
        }
    }

    private var outputDirLabel: String {
        switch VideoSlicePipeline.inspectOutputDirectory(outputDirPath) {
        case .missingSetting:
            return l10n.tr("videoSlice.noOutput")
        case .unreachable(let path), .notDirectory(let path), .notWritable(let path):
            return path
        case .ready(let url):
            return url.path
        }
    }

    private var outputDirColor: Color {
        switch VideoSlicePipeline.inspectOutputDirectory(outputDirPath) {
        case .ready:
            return .secondary
        case .missingSetting:
            return .secondary
        default:
            return .red
        }
    }

    private func outputErrorText(_ status: SliceOutputDir) -> String? {
        switch status {
        case .missingSetting:
            return l10n.tr("videoSlice.needOutput")
        case .unreachable(let path), .notDirectory(let path), .notWritable(let path):
            return String(format: l10n.tr("videoSlice.outputUnreachable"), path)
        case .ready:
            return nil
        }
    }

    private func refreshOutputError() {
        let status = VideoSlicePipeline.inspectOutputDirectory(outputDirPath)
        switch status {
        case .ready:
            if isOutputFolderError { errorText = nil }
        case .missingSetting:
            break
        default:
            errorText = outputErrorText(status)
        }
    }

    private var isOutputFolderError: Bool {
        guard let errorText else { return false }
        if errorText == l10n.tr("videoSlice.needOutput") { return true }
        let marker = l10n.tr("videoSlice.outputUnreachable").replacingOccurrences(of: "%@", with: "")
        return !marker.isEmpty && errorText.contains(marker)
    }

    private func pickOutput() {
        let p = NSOpenPanel()
        p.canChooseDirectories = true
        p.canChooseFiles = false
        p.canCreateDirectories = true
        if case .ready(let url) = VideoSlicePipeline.inspectOutputDirectory(outputDirPath) {
            p.directoryURL = url
        }
        guard p.runModal() == .OK, let url = p.url else { return }
        outputDirPath = url.path
        errorText = nil
    }

    private func loadStack() {
        isLoadingStack = true
        errorText = nil
        Task {
            do {
                if inferenceService.transcriptionService.listLoadedModels().isEmpty {
                    guard let rec = modelManager.downloadedModels().first(where: {
                        $0.family == .qwen3Asr || $0.family == .whisper
                    }) else { throw NovaMLXError.apiError(l10n.tr("videoSlice.needASR")) }
                    await MainActor.run { log("ASR \(rec.id)") }
                    try await inferenceService.loadModel(
                        at: rec.localURL,
                        config: ModelConfig(
                            identifier: ModelIdentifier(id: rec.id, family: rec.family),
                            modelType: .audio
                        )
                    )
                }
                await MainActor.run { log(l10n.tr("videoSlice.loadingAligner")) }
                try await ForcedAlignerService.ensureModel()
                await MainActor.run {
                    if pickLLM() == nil {
                        log(l10n.tr("videoSlice.needLLM"))
                    }
                    log(l10n.tr("videoSlice.stackReady"))
                    isLoadingStack = false
                }
            } catch {
                await MainActor.run {
                    errorText = error.localizedDescription
                    isLoadingStack = false
                }
            }
        }
    }

    private func runPipeline() {
        guard let videoURL else { return }
        let asrId = inferenceService.transcriptionService.listLoadedModels().first
        let llmId = pickLLM()
        guard let asrId else {
            errorText = l10n.tr("videoSlice.needASR")
            return
        }
        let destStatus = VideoSlicePipeline.inspectOutputDirectory(outputDirPath)
        guard case .ready(let outputRoot) = destStatus else {
            errorText = outputErrorText(destStatus)
            return
        }
        isRunning = true
        produced = []
        logLines = []
        errorText = nil
        let target = sliceSeconds
        let fillers = removeFillers
        let tStyle = titleStyle
        let pStyle = pointStyle
        let ref = referenceText
        let nameList = glossary
        let destRoot = outputRoot.appendingPathComponent(
            VideoSlicePipeline.jobFolderName(videoName: videoURL.lastPathComponent),
            isDirectory: true
        )
        Task {
            do {
                try FileManager.default.createDirectory(at: destRoot, withIntermediateDirectories: true)
                await MainActor.run { log("extract audio") }
                let wav = destRoot.appendingPathComponent("full.wav")
                try VideoSlicePipeline.extractAudio(from: videoURL, to: wav)
                let duration = try VideoSlicePipeline.probeDuration(wav)
                await MainActor.run { log(String(format: "audio %.1fs", duration)) }
                if llmId == nil {
                    await MainActor.run { log(l10n.tr("videoSlice.needLLM")) }
                }

                struct ASRChunk {
                    var start: Double
                    var duration: Double
                    var url: URL
                    var text: String
                }
                var chunks: [ASRChunk] = []
                let asrContext = VideoSlicePipeline.asrContext(glossary: nameList, reference: ref)
                let nameTerms = VideoSlicePipeline.glossaryTerms(nameList)
                let chunkLen: Double = duration <= 360 ? max(duration, 1) : 90
                var t: Double = 0
                var idx = 0
                while t < duration - 0.2 {
                    let dur = min(chunkLen, duration - t)
                    let piece = destRoot.appendingPathComponent("chunk-\(idx).wav")
                    try VideoSlicePipeline.extractAudioSegment(from: wav, start: t, duration: dur, to: piece)
                    let data = try Data(contentsOf: piece)
                    await MainActor.run { log(String(format: "ASR %.0f–%.0fs", t, t + dur)) }
                    let result = try await inferenceService.transcriptionService.transcribe(
                        modelId: asrId, audioData: data, language: "zh", context: asrContext)
                    chunks.append(ASRChunk(
                        start: t,
                        duration: dur,
                        url: piece,
                        text: result.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    ))
                    t += dur
                    idx += 1
                }

                var corrected = chunks.map { VideoSlicePipeline.tidyTranscript($0.text) }
                if let llmId {
                    await MainActor.run { log("LLM proofread") }
                    corrected = await proofreadChunks(
                        chunks: corrected, reference: ref, glossary: nameTerms, model: llmId)
                    corrected = corrected.map { VideoSlicePipeline.tidyTranscript($0) }
                }
                let transcript = corrected.joined(separator: "\n")
                try transcript.write(
                    to: destRoot.appendingPathComponent("transcript.txt"),
                    atomically: true, encoding: .utf8)

                var allWords: [TimedWord] = []
                for (i, chunk) in chunks.enumerated() {
                    let spoken = VideoSlicePipeline.tidyTranscript(
                        i < corrected.count ? corrected[i] : chunk.text)
                    guard !spoken.isEmpty else { continue }
                    await MainActor.run { log(String(format: "align %.0f–%.0fs", chunk.start, chunk.start + chunk.duration)) }
                    let aligned = try await ForcedAlignerService.align(
                        audioURL: chunk.url, text: spoken, language: "Chinese")
                    for w in aligned {
                        allWords.append(TimedWord(
                            text: w.text, start: w.start + chunk.start, end: w.end + chunk.start))
                    }
                }

                if fillers {
                    allWords = VideoSlicePipeline.stripFillers(allWords)
                }
                let captionSRT = VideoSlicePipeline.captionSRT(from: allWords)
                try captionSRT.write(
                    to: destRoot.appendingPathComponent("full.srt"), atomically: true, encoding: .utf8)
                try VideoSlicePipeline.srt(from: allWords).write(
                    to: destRoot.appendingPathComponent("full.words.srt"), atomically: true, encoding: .utf8)
                let sents = VideoSlicePipeline.sentences(from: allWords)
                await MainActor.run {
                    log("SRT sentences=\(sents.count) words=\(allWords.count)")
                }

                var specs: [VideoSliceSpec] = []
                if let llmId, !sents.isEmpty {
                    await MainActor.run { log("LLM theme plan") }
                    let planned = await planThemes(
                        sentences: sents, maxDur: target, model: llmId)
                    specs = VideoSlicePipeline.applySentencePlan(planned, sentences: sents)
                }
                if specs.isEmpty {
                    specs = VideoSlicePipeline.packTopics(sentences: sents, maxDur: target)
                }
                if let llmId, !specs.isEmpty {
                    await MainActor.run { log("LLM titles") }
                    specs = await summarizeSpecs(specs, model: llmId, glossary: nameTerms)
                }
                let themeDump = specs.enumerated().map { i, s in
                    "\(i + 1)\t\(String(format: "%.1f", s.start))-\(String(format: "%.1f", s.end))\t\(s.title)\t\(s.point)"
                }.joined(separator: "\n")
                try themeDump.write(
                    to: destRoot.appendingPathComponent("themes.txt"),
                    atomically: true, encoding: .utf8)

                var outs: [URL] = []
                for (i, pack) in specs.enumerated() {
                    let title = pack.title
                    let point = pack.point
                    let relSRT = VideoSlicePipeline.captionSRT(from: pack.words, origin: pack.start)
                    let srtURL = destRoot.appendingPathComponent(String(format: "slice-%02d.srt", i + 1))
                    try relSRT.write(to: srtURL, atomically: true, encoding: .utf8)
                    let out = destRoot.appendingPathComponent(String(format: "slice-%02d.mp4", i + 1))
                    await MainActor.run { log("slice \(i + 1)/\(specs.count) \(title)") }
                    try VideoSlicePipeline.cutOverlay(
                        video: videoURL,
                        start: pack.start,
                        duration: max(0.4, pack.end - pack.start),
                        title: title,
                        point: point,
                        srtURL: srtURL,
                        titleStyle: tStyle,
                        pointStyle: pStyle,
                        output: out
                    )
                    outs.append(out)
                }
                await MainActor.run {
                    produced = outs
                    log(String(format: l10n.tr("videoSlice.done"), outs.count, destRoot.path))
                    isRunning = false
                    NSWorkspace.shared.open(destRoot)
                }
            } catch {
                await MainActor.run {
                    errorText = error.localizedDescription
                    isRunning = false
                }
            }
        }
    }

    private func pickLLM() -> String? {
        inferenceService.listLoadedModels().first { id in
            let lower = id.lowercased()
            if lower.contains("dflash") || lower.contains("mtp") { return false }
            if lower.contains("asr") || lower.contains("whisper")
                || lower.contains("tts") || lower.contains("aligner")
            {
                return false
            }
            if let rec = modelManager.getRecord(id) {
                return rec.modelType == .llm || rec.modelType == .vlm
            }
            return true
        }
    }

    private func proofreadChunks(
        chunks: [String],
        reference: String,
        glossary: [String],
        model: String
    ) async -> [String] {
        guard chunks.contains(where: { !$0.isEmpty }) else { return chunks }
        let listing = chunks.enumerated().map { i, text in
            "[\(i)] \(text)"
        }.joined(separator: "\n")
        var prompt = """
        你是中文口播校对，必须改稿，禁止原样复制。
        规则：
        1. 纠正同音错字。后面出现的正确词用来改前面的错词。
        2. 补上。！？，让每一句都是完整意思。
        3. 不要发明没说的内容，不要总结，不要改写口吻。
        4. chunks 数量必须与输入一致。
        5. 公司、实验室、模型、产品写成官方名称，数字用阿拉伯数字。中文谐音不要留着。不确定就保持原文。
        只输出 JSON。
        示例：输入 [0] 今天天汽不错，我先讲讲一下
        输出 {"chunks":[{"i":0,"text":"今天天气不错。我先讲一下。"}]}

        【ASR分片】
        \(listing)
        """
        if !glossary.isEmpty {
            let listed = glossary.joined(separator: "\n")
            prompt += """


            【官方名称。口播里的谐音、错字改成这些写法，保留拉丁字母和阿拉伯数字】
            \(listed)
            """
        }
        let heard = chunks.joined()
        if Self.mentionsFable(heard, glossary: glossary) {
            prompt += """


            若上下文是在讲这款模型：「刷皮的飞豹五」写成「Anthropic的Fable 5」。只说游戏换皮肤时不要改「刷皮」。
            """
        }
        if !reference.isEmpty {
            prompt += """


            【口播原稿，用于纠正同音字，不要照抄未说出的段落】
            \(reference.prefix(8000))
            """
        }
        let req = InferenceRequest(
            model: model,
            messages: [ChatMessage(role: .user, content: prompt)],
            temperature: 0.2,
            maxTokens: min(8192, max(1024, chunks.joined().count + 512)),
            enableThinking: false
        )
        do {
            let r = try await inferenceService.generate(req)
            return VideoSlicePipeline.parseCorrectedChunks(r.text, fallback: chunks)
        } catch {
            return chunks
        }
    }

    private func planThemes(
        sentences: [TimedSentence],
        maxDur: Double,
        model: String
    ) async -> [PlannedSlice] {
        guard !sentences.isEmpty else { return [] }
        let listing = sentences.prefix(160).enumerated().map { i, s in
            String(format: "%d\t%.2f\t%.2f\t%@", i + 1, s.start, s.end, s.text)
        }.joined(separator: "\n")
        let prompt = """
        你是短视频选题剪辑。下面是已经校对并断句的口播句子表（序号从 1 开始）。
        按「完整主题」切片：
        - 同一个主题从头讲到讲完，必须放在同一个切片
        - 换主题才切开，只能切在句子边界
        - 不要按时长硬切，不要把不相关的内容拼在一起
        - 单个主题如果超过 \(Int(maxDur)) 秒，按子观点再拆，每个子切片仍须意思完整
        - 覆盖全部句子，不要漏句
        只输出 JSON，不要写标题：
        {"slices":[{"sentences":[1,2,3]}]}

        \(listing)
        """
        let req = InferenceRequest(
            model: model,
            messages: [ChatMessage(role: .user, content: prompt)],
            temperature: 0.2,
            maxTokens: 2048,
            enableThinking: false
        )
        do {
            let r = try await inferenceService.generate(req)
            return VideoSlicePipeline.parseSlicePlan(r.text)
        } catch {
            return []
        }
    }

    private static func mentionsFable(_ text: String, glossary: [String]) -> Bool {
        if text.contains("飞豹") || text.contains("飞宝") || text.contains("Fable") || text.contains("Anthropic") {
            return true
        }
        return glossary.contains {
            $0.localizedCaseInsensitiveContains("fable") || $0.localizedCaseInsensitiveContains("anthropic")
        }
    }

    private static func titleExample(texts: [String], glossary: [String]) -> String {
        if mentionsFable(texts.joined(), glossary: glossary) {
            return """
            差的标题：刷皮的飞豹五呢确实是非常贵但
            好的标题：Anthropic的Fable 5很贵但很好
            """
        }
        return """
        差的标题：今天我们来讲讲这个东西它其实
        好的标题：这套做法比旧方案更省事
        """
    }

    private func summarizeSpecs(
        _ specs: [VideoSliceSpec],
        model: String,
        glossary: [String]
    ) async -> [VideoSliceSpec] {
        var specs = specs
        let texts = specs.map { VideoSlicePipeline.joinWords($0.words) }
        let first = await headlineBatch(texts: texts, model: model, glossary: glossary, rejected: nil)
        var retry: [Int] = []
        for i in specs.indices {
            let proposed = i < first.count ? first[i] : (title: "", point: "")
            let resolved = VideoSlicePipeline.resolveHeadline(
                proposedTitle: proposed.title,
                proposedPoint: proposed.point,
                source: texts[i]
            )
            if resolved.acceptedProposal {
                specs[i].title = resolved.title
                specs[i].point = resolved.point
            } else {
                retry.append(i)
            }
        }
        for i in retry {
            let rejected = i < first.count ? first[i].title : ""
            let second = await headlineBatch(
                texts: [texts[i]],
                model: model,
                glossary: glossary,
                rejected: rejected
            )
            let proposed = second.first ?? (title: "", point: "")
            let resolved = VideoSlicePipeline.resolveHeadline(
                proposedTitle: proposed.title,
                proposedPoint: proposed.point,
                source: texts[i]
            )
            specs[i].title = resolved.title
            specs[i].point = resolved.point
            await MainActor.run {
                log("title \(i + 1) \(resolved.acceptedProposal ? "summary" : "fallback") \(resolved.title)")
            }
        }
        return specs
    }

    private func headlineBatch(
        texts: [String],
        model: String,
        glossary: [String],
        rejected: String?
    ) async -> [(title: String, point: String)] {
        let listing = texts.enumerated().map { i, text in
            let body = text.count > 1200 ? String(text.prefix(1200)) : text
            return "[\(i)] \(body)"
        }.joined(separator: "\n")
        var prompt = """
        你是短视频标题编辑。下面每一段都是一个已经切好的完整主题。
        为每一段写概括标题，不要把口播开头截下来当标题。
        - title：一句完整的话，说出这段的结论。中文大约 12 到 28 个字。
          公司、模型、产品用官方写法（拉丁字母 + 阿拉伯数字），不要写成中文谐音，也不要为了变短把名字截断。
          不要停在「但、而、因为、所以、如果、的」。
          不要照抄第一句，不要用第一句的前十几个字。
        - point：不超过 32 字的核心观点，同样不要截取原文开头。
        \(Self.titleExample(texts: texts, glossary: glossary))
        只输出 JSON：
        {"items":[{"i":0,"title":"...","point":"..."}]}

        \(listing)
        """
        if !glossary.isEmpty {
            prompt += "\n\n官方名称：\(glossary.joined(separator: "、"))"
        }
        if let rejected, !rejected.isEmpty {
            prompt += "\n\n不要再输出这个不合格标题：\(rejected)"
        }
        let req = InferenceRequest(
            model: model,
            messages: [ChatMessage(role: .user, content: prompt)],
            temperature: 0.3,
            maxTokens: min(4096, 240 * max(1, texts.count) + 200),
            enableThinking: false
        )
        do {
            let r = try await inferenceService.generate(req)
            return VideoSlicePipeline.parseHeadlines(r.text, count: texts.count)
        } catch {
            return Array(repeating: (title: "", point: ""), count: texts.count)
        }
    }
}
