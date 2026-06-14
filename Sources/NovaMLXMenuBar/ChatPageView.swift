import SwiftUI
import Foundation
import AVFoundation
import CoreMedia
import NovaMLXCore
import NovaMLXDB
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils
import NovaMLXEngine

private enum ChatDisplayMode: String, CaseIterable {
    case pretty = "Pretty"
    case rawJSON = "JSON"
    case rawStream = "Stream"
}

private enum PlaygroundMode {
    case llm, asr, tts, image
}

struct ChatPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @EnvironmentObject var l10n: L10n
    @State private var messages: [ChatMessageRow] = []
    @State private var inputText = ""
    @State private var selectedModel = ""
    @State private var selectedTag = ""
    @State private var isLoading = false
    @State private var displayMode: ChatDisplayMode = .pretty

    // Active inference task for cancellation support (playground stop button)
    @State private var currentInferenceTask: Task<Void, Never>? = nil
    @State private var currentRequestId: UUID? = nil

    // History navigation
    @State private var sentHistory: [String] = []
    @State private var historyIndex: Int? = nil
    @State private var savedDraft: String? = nil

    // Parameter controls
    @FocusState private var isInputFocused: Bool
    @State private var paramTemp: Double = 0.7
    @State private var paramMaxTokens: Double = 4096
    @State private var paramTopP: Double = 0.9
    @State private var paramTopK: Double = 0
    @State private var paramMinP: Double = 0
    @State private var paramRepeatPenalty: Double = 1.0

    // Copy buffers
    @State private var lastPayload: String?
    @State private var lastResponse: String?

    // ASR recording (chat mic input)
    @State private var isRecording = false
    @State private var chatRecorder: AVAudioRecorder?
    @State private var chatRecordingURL: URL?

    // TTS playback
    @State private var ttsPlayingMessageId: UUID?

    // MARK: - ASR Playground State
    @State private var selectedASRModel: String = ""
    @State private var isASRRecording = false
    @State private var asrRecorder: AVAudioRecorder?
    @State private var asrRecordingURL: URL?
    @State private var transcriptionText = ""
    @State private var isTranscribing = false
    @State private var isASRModelLoading = false
    @State private var loadingASRModelName: String?
    @State private var asrError: String?
    @State private var uploadedFileName: String?
    @State private var asrCodeTab: CodeTab = .curl

    // MARK: - TTS Playground State
    @State private var ttsText = ""
    @State private var isSynthesizing = false
    @State private var ttsError: String?
    @State private var ttsSuccess: String?
    @State private var isPlaying = false
    @State private var synthesizedAudioURL: URL?
    @State private var audioPlayer: AVAudioPlayer?
    @State private var ttsEngine: TTSEngine = .neural
    @State private var selectedSystemVoice: String = "Tingting"
    @State private var macOSVoices: [MacOSVoice] = []
    @State private var voiceProfiles: [VoiceProfile] = []
    @State private var selectedProfileId: UUID? = nil
    @State private var showCloneSheet = false
    @State private var ttsCodeTab: CodeTab = .curl

    // MARK: - Image Playground State
    @State private var imagePrompt = ""
    @State private var isGeneratingImage = false
    @State private var generatedImage: NSImage?
    @State private var imageError: String?
    @State private var imageSize = "1024x1024"
    @State private var imageSeed: String = ""
    @State private var imageSteps: Int = 4
    @State private var selectedImageModel: String = ""
    @State private var imageCodeTab: CodeTab = .curl
    @State private var isImageModelLoading = false
    @State private var loadingImageModelName: String?
    @State private var generatedImageData: Data?

    private let quickPrompts = [
        "2+2=? Please explain step by step",
        "Write a haiku about coding",
        "Explain quantum computing in simple terms",
        "Translate 'Hello, how are you?' to Japanese",
        "Debug this: why does my code return nil?"
    ]

    // MARK: - Model Type Detection

    @State private var playgroundMode: PlaygroundMode = .llm

    private func autoDetectMode(_ model: String) -> PlaygroundMode {
        if model.isEmpty || model == "tknet" { return .llm }
        if model.hasPrefix("tknet:") {
            let name = String(model.dropFirst("tknet:".count))
            return inferModeFromName(name)
        }
        if let record = modelManager.getRecord(model) {
            if record.family == .whisper || record.family == .qwen3Asr { return .asr }
            if record.family == .dotsTts || record.family == .qwen3Tts { return .tts }
            if record.family == .flux || record.family == .stableDiffusion { return .image }
            return .llm
        }
        return inferModeFromName(model)
    }

    private func inferModeFromName(_ name: String) -> PlaygroundMode {
        let lower = name.lowercased()
        if lower.contains("whisper") || lower.contains("asr")
            || lower.contains("speech-to-text") || lower.contains("transcrib") {
            return .asr
        }
        if lower.contains("tts") || lower.contains("speech-synthesis")
            || lower.contains("dots-tts") || lower.contains("voice-clone") {
            return .tts
        }
        if lower.contains("flux") || lower.contains("stable-diffusion")
            || lower.contains("sdxl") || lower.contains("dall-e")
            || lower.contains("imagen") || lower.contains("midjourney") {
            return .image
        }
        return .llm
    }

    var body: some View {
        VStack(spacing: 0) {
            chatToolbar
            Divider()
            switch playgroundMode {
            case .llm:
                llmContent
            case .asr:
                asrContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .tts:
                ttsContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            case .image:
                imageContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .onAppear {
            autoSelectASRModel()
            loadMacOSVoices()
            loadVoiceProfiles()
        }
        .onChange(of: appState.loadedModels) { _, _ in
            autoSelectASRModel()
        }
        .sheet(isPresented: $showCloneSheet) {
            VoiceCloneSheet(l10n: l10n) {
                loadVoiceProfiles()
                if let newest = voiceProfiles.first {
                    selectedProfileId = newest.id
                }
            }
        }
    }

    // MARK: - LLM Content

    private var llmContent: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                llmToolbarStrip
                Divider()
                messageList
                Divider()
                if inputText.isEmpty {
                    suggestionsBar
                        .padding(.horizontal, 16)
                        .padding(.vertical, 6)
                }
                inputBar
            }
            .frame(maxWidth: .infinity)

            Divider()
            rightParamsPanel
                .frame(width: 200)
        }
    }

    /// LLM-only toolbar strip: DisplayMode segmented picker on the left,
    /// Stop / Copy / Clear on the right. Lives inside `llmContent` (not the
    /// global `chatToolbar`) because every control here is meaningful only
    /// for LLM chat — keeps the top toolbar focused on Model + Mode.
    private var llmToolbarStrip: some View {
        HStack(spacing: 8) {
            Picker("", selection: $displayMode) {
                ForEach(ChatDisplayMode.allCases, id: \.self) { mode in
                    Text(mode.rawValue).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 200)

            Spacer()

            if isLoading {
                ProgressView()
                    .controlSize(.small)
                Button(action: { cancelCurrentInference() }) {
                    HStack(spacing: 3) {
                        Image(systemName: "stop.circle")
                            .font(.system(size: 10))
                        Text("Stop")
                            .font(.system(size: 10))
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .foregroundColor(.red)
            }

            Button {
                if let payload = lastPayload {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(payload, forType: .string)
                }
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 10))
                    Text("Copy Payload")
                        .font(.system(size: 11))
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(lastPayload == nil)

            Button {
                if let resp = lastResponse {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(resp, forType: .string)
                }
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 10))
                    Text("Copy Result")
                        .font(.system(size: 11))
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(lastResponse == nil)

            Button {
                if let payload = lastPayload, let resp = lastResponse {
                    let combined = "PAYLOAD:\n\(payload)\n\nRESULT:\n\(resp)"
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(combined, forType: .string)
                }
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: "doc.on.doc.on.clipboard")
                        .font(.system(size: 10))
                    Text("Copy Both")
                        .font(.system(size: 11))
                }
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .disabled(lastPayload == nil || lastResponse == nil)

            Button(l10n.tr("chat.clear")) {
                cancelCurrentInference()
                messages.removeAll()
                lastPayload = nil
                lastResponse = nil
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
    }

    // MARK: - Toolbar

    private var chatToolbar: some View {
        HStack(spacing: 12) {
            Picker(l10n.tr("chat.model"), selection: $selectedModel) {
                if appState.loadedModels.isEmpty && tokenhubModels.isEmpty && loadBalancerEntries.isEmpty {
                    Text(l10n.tr("chat.noModels")).tag("")
                }
                if !appState.loadedModels.isEmpty {
                    Section("LOCAL — DIRECT IN-PROCESS") {
                        ForEach(appState.loadedModels, id: \.self) { model in
                            HStack(spacing: 6) {
                                Image(systemName: modelTypeIcon(modelType(for: model)))
                                    .foregroundColor(modelTypeColor(modelType(for: model)))
                                    .font(.system(size: 10))
                                Text(shortModelName(model))
                            }
                            .tag(model)
                        }
                    }
                }
                if !tokenhubModels.isEmpty {
                    Section("TOKENHUB — HTTP ROUTING") {
                        ForEach(tokenhubModels, id: \.self) { model in
                            HStack {
                                Image(systemName: "server.rack")
                                    .font(.system(size: 10))
                                Text(model)
                            }
                            .tag("tknet:\(model)")
                        }
                    }
                }
                if !loadBalancerEntries.isEmpty {
                    Section("LOAD BALANCE — NAMED POOLS") {
                        ForEach(loadBalancerEntries, id: \.slug) { lb in
                            HStack {
                                Image(systemName: "scalemass")
                                    .font(.system(size: 10))
                                Text("\(lb.name)  ·  lb:\(lb.slug)")
                            }
                            .tag("lb:\(lb.slug)")
                        }
                    }
                }
            }
            .frame(width: 240)

            Picker("Mode", selection: $playgroundMode) {
                Text("LLM").tag(PlaygroundMode.llm)
                Text("ASR").tag(PlaygroundMode.asr)
                Text("TTS").tag(PlaygroundMode.tts)
                Text("Image").tag(PlaygroundMode.image)
            }
            .pickerStyle(.menu)
            .frame(width: 110)
            .help("Override playground type")

            Spacer()
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)
        .onAppear {
            if selectedModel.isEmpty {
                if let first = appState.loadedModels.first {
                    selectedModel = first
                    loadDefaultsFromModel(first)
                }
            }
            playgroundMode = autoDetectMode(selectedModel)
        }
        .onChange(of: appState.loadedModels) { _, newModels in
            let isTokenhub = selectedModel == "tknet" || selectedModel.hasPrefix("tknet:")
            if !newModels.contains(selectedModel) && !isTokenhub {
                if let first = newModels.first {
                    selectedModel = first
                    loadDefaultsFromModel(first)
                }
            }
        }
        .onChange(of: selectedModel) { _, newModel in
            if !newModel.isEmpty && newModel != "tknet" && !newModel.hasPrefix("tknet:") {
                loadDefaultsFromModel(newModel)
            }
            playgroundMode = autoDetectMode(newModel)
        }
    }

    // MARK: - Right Params Panel

    private var rightParamsPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Parameters")
                .font(.headline)
                .padding(.top, 8)

            ParamSlider(label: "Temperature", value: $paramTemp, min: 0, max: 2, step: 0.05)
            ParamSlider(label: "Top P", value: $paramTopP, min: 0, max: 1, step: 0.05)
            ParamSlider(label: "Top K", value: $paramTopK, min: 0, max: 200, step: 1)
            ParamSlider(label: "Min P", value: $paramMinP, min: 0, max: 1, step: 0.05)
            ParamSlider(label: "Max Tokens", value: $paramMaxTokens, min: 64, max: 32768, step: 64)
            ParamSlider(label: "Repeat Penalty", value: $paramRepeatPenalty, min: 1.0, max: 2.0, step: 0.05)

            Spacer()

            Button("Reset Defaults") {
                loadDefaultsFromModel(selectedModel)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .foregroundColor(NovaTheme.Colors.accent)

            if displayMode != .pretty {
                HStack(spacing: 4) {
                    Image(systemName: "terminal")
                        .font(.caption2)
                    Text(displayMode == .rawJSON ? "Raw JSON response" : "Raw SSE stream")
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
    }

    // MARK: - Message List

    private func loadDefaultsFromModel(_ modelId: String) {
        let container = inferenceService.engine.getContainer(for: modelId)
        let config = container?.config
        paramTemp = config?.temperature ?? 0.7
        paramMaxTokens = Double(min(config?.maxTokens ?? 4096, 8192))
        paramTopP = config?.topP ?? 0.9
        paramTopK = 0
        paramMinP = 0
        paramRepeatPenalty = Double(config?.repeatPenalty ?? 1.0)
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if messages.isEmpty {
                        VStack(spacing: 8) {
                            Image(systemName: "bubble.left.and.bubble.right")
                                .font(.system(size: 40))
                                .foregroundColor(.secondary.opacity(0.5))
                            Text(l10n.tr("chat.startConversation"))
                                .font(.title3)
                                .foregroundColor(.secondary)
                            Text(l10n.tr("chat.selectModel"))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 80)
                    }

                    ForEach(messages) { msg in
                        messageBubble(msg)
                            .id(msg.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: messages.count) { _, _ in
                if let last = messages.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private func messageBubble(_ msg: ChatMessageRow) -> some View {
        let isRaw = !msg.isUser && (displayMode == .rawJSON || displayMode == .rawStream)

        return HStack {
            if msg.isUser { Spacer(minLength: 60) }
            VStack(alignment: msg.isUser ? .trailing : .leading, spacing: 6) {
                Text(msg.isUser ? l10n.tr("chat.you") : l10n.tr("chat.assistant"))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                if !msg.content.isEmpty {
                    Group {
                        if isRaw {
                            ScrollView(.horizontal, showsIndicators: true) {
                                Text(msg.content)
                                    .font(.system(size: 11, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        } else {
                            Text(msg.content)
                                .font(.system(size: 13))
                                .textSelection(.enabled)
                        }
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: msg.isUser ? .trailing : .leading)
                    .background(
                        isRaw
                            ? Color.black.opacity(0.3)
                            : msg.isUser ? NovaTheme.Colors.accentDim : NovaTheme.Colors.cardBackground
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        isRaw
                            ? RoundedRectangle(cornerRadius: 10).stroke(Color.gray.opacity(0.3), lineWidth: 0.5)
                            : nil
                    )

                    // TTS button on assistant messages
                    if !msg.isUser && !isRaw {
                        Button {
                            speakMessage(msg)
                        } label: {
                            Image(systemName: ttsPlayingMessageId == msg.id ? "speaker.wave.2.fill" : "speaker.fill")
                                .font(.system(size: 10))
                                .foregroundColor(ttsPlayingMessageId == msg.id ? NovaTheme.Colors.accent : .secondary)
                        }
                        .buttonStyle(.plain)
                        .help("Read aloud")
                    }
                }
            }
            if !msg.isUser { Spacer(minLength: 60) }
        }
    }

    // MARK: - Suggestions

    private var suggestionsBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(quickPrompts, id: \.self) { prompt in
                    Button {
                        inputText = prompt
                        isInputFocused = true
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "sparkles")
                                .font(.caption2)
                            Text(prompt)
                                .font(.caption)
                                .lineLimit(1)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.8))
                        .clipShape(Capsule())
                        .overlay(
                            Capsule().stroke(Color(nsColor: .separatorColor), lineWidth: 0.5)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        HStack(spacing: 12) {
            ZStack(alignment: .topLeading) {
                TextEditor(text: $inputText)
                    .font(.system(size: NSFont.systemFontSize))
                    .scrollContentBackground(.hidden)
                    .frame(minHeight: 28, maxHeight: 120)
                    .focused($isInputFocused)
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.shift) {
                            sendMessage()
                            return .handled
                        }
                        return .ignored
                    }
                    .onKeyPress(.upArrow, phases: .down) { _ in
                        if !inputText.contains("\n") {
                            navigateHistory(.up)
                            return .handled
                        }
                        return .ignored
                    }
                    .onKeyPress(.downArrow, phases: .down) { _ in
                        if !inputText.contains("\n") {
                            navigateHistory(.down)
                            return .handled
                        }
                        return .ignored
                    }

                if inputText.isEmpty && !isInputFocused {
                    Text("Type a message...")
                        .foregroundColor(Color(NSColor.placeholderTextColor))
                        .font(.system(size: NSFont.systemFontSize))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 5)
                        .allowsHitTesting(false)
                }
            }

            if isLoading {
                Button(action: { cancelCurrentInference() }) {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundColor(.red)
                }
                .buttonStyle(.plain)
                .help("Stop inference")
            } else {
                // Mic button for ASR
                if hasLoadedASRModel {
                    Button {
                        toggleChatRecording()
                    } label: {
                        Image(systemName: isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.title2)
                            .foregroundColor(isRecording ? .red : .secondary)
                    }
                    .buttonStyle(.plain)
                    .help(isRecording ? "Stop recording" : "Voice input")
                }

                Button(action: { sendMessage() }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .buttonStyle(.plain)
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || selectedModel.isEmpty)
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)
    }

    // MARK: - History

    private enum HistoryDirection { case up, down }

    private func navigateHistory(_ direction: HistoryDirection) {
        guard !sentHistory.isEmpty else { return }

        switch direction {
        case .up:
            if historyIndex == nil {
                savedDraft = inputText
                historyIndex = sentHistory.count
            }
            guard let idx = historyIndex, idx > 0 else { return }
            historyIndex = idx - 1
            inputText = sentHistory[idx - 1]

        case .down:
            guard let idx = historyIndex else { return }
            if idx >= sentHistory.count - 1 {
                historyIndex = nil
                inputText = savedDraft ?? ""
                savedDraft = nil
            } else {
                historyIndex = idx + 1
                inputText = sentHistory[idx + 1]
            }
        }
    }

    // MARK: - Send Message

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !selectedModel.isEmpty else { return }

        if sentHistory.last != text { sentHistory.append(text) }
        historyIndex = nil
        savedDraft = nil
        inputText = ""
        messages.append(ChatMessageRow(content: text, isUser: true))

        let assistantMsg = ChatMessageRow(content: "", isUser: false)
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1
        isLoading = true
        let model = selectedModel
        let tag = (model == "tknet" && !selectedTag.isEmpty) ? selectedTag : nil

        NovaMLXLog.info("[sendMessage] model=\(model) displayMode=\(displayMode.rawValue) isTknet=\(model == "tknet" || model.hasPrefix("tknet:"))")
        switch displayMode {
        case .pretty:
            sendPretty(model: model, text: text, assistantIdx: assistantIdx, tag: tag)
        case .rawJSON:
            sendRawJSON(model: model, text: text, assistantIdx: assistantIdx, tag: tag)
        case .rawStream:
            sendRawStream(model: model, text: text, assistantIdx: assistantIdx, tag: tag)
        }
    }

    private func cancelCurrentInference() {
        currentInferenceTask?.cancel()
        if let id = currentRequestId {
            Task { await inferenceService.abort(requestId: id) }
        }
        currentInferenceTask = nil
        currentRequestId = nil
        isLoading = false
    }

    // MARK: Pretty mode (existing InferenceService streaming)

    private func sendPretty(model: String, text: String, assistantIdx: Int, tag: String? = nil) {
        // Resolve model ID: strip "tknet:Local " prefix for local models
        var resolvedModel = model
        if resolvedModel.hasPrefix("tknet:Local ") {
            resolvedModel = String(resolvedModel.dropFirst("tknet:Local ".count))
        }

        // Anything that needs the API server to route it (rather than the
        // in-process engine) goes through the HTTP path. That covers:
        //   - "tknet" / "tknet:<provider>"  → tokenhub passthrough
        //   - "lb:<slug>"                   → LBProxy picks a member
        // Local models fall through and use inferenceService.stream() below.
        if resolvedModel.hasPrefix("lb:")
            || resolvedModel == "tknet"
            || (resolvedModel.hasPrefix("tknet:") && !resolvedModel.hasPrefix("tknet:Local")) {
            sendPrettyTokenhub(model: resolvedModel, text: text, assistantIdx: assistantIdx, tag: tag)
            return
        }

        currentInferenceTask = Task {
            defer {
                isLoading = false
                currentInferenceTask = nil
                currentRequestId = nil
            }

            guard inferenceService.isModelLoaded(resolvedModel) else {
                messages[assistantIdx].content = l10n.tr("chat.error", "Model '\(resolvedModel.components(separatedBy: "/").last ?? resolvedModel)' is not loaded. Load it from the Models page first.")
                return
            }

            let payload: [String: Any] = [
                "model": resolvedModel,
                "messages": [["role": "user", "content": text]],
                "stream": true,
                "temperature": paramTemp,
                "max_tokens": Int(paramMaxTokens),
                "top_p": paramTopP,
                "top_k": Int(paramTopK),
                "min_p": paramMinP,
                "repetition_penalty": paramRepeatPenalty
            ]
            if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) {
                lastPayload = String(data: data, encoding: .utf8)
            }

            let request = InferenceRequest(
                model: resolvedModel,
                messages: [ChatMessage(role: .user, content: text)],
                temperature: paramTemp,
                maxTokens: Int(paramMaxTokens),
                topP: paramTopP,
                topK: Int(paramTopK),
                minP: Float(paramMinP),
                repetitionPenalty: Float(paramRepeatPenalty),
                stream: true
            )
            currentRequestId = request.id

            do {
                // Detect if this is a thinking model and use ThinkingParser
                let isImplicitModel = ModelContainer.isImplicitThinkingModel(for: resolvedModel)
                NovaMLXLog.info("[sendPretty] model=\(resolvedModel) isImplicit=\(isImplicitModel)")
                let thinkingParser = ThinkingParser(expectImplicitThinking: isImplicitModel)

                var tokenCount = 0
                var nonEmptyTokens = 0
                let tokenStream = inferenceService.stream(request)
                for try await token in tokenStream {
                    if Task.isCancelled { break }
                    tokenCount += 1
                    if !token.text.isEmpty { nonEmptyTokens += 1 }

                    // Parse thinking vs content for thinking models
                    let parsed = thinkingParser.feed(token.text)
                    if !parsed.text.isEmpty {
                        messages[assistantIdx].content += parsed.text
                    }
                }

                // Finalize thinking parser to flush any buffered content
                let finalResult = thinkingParser.finalize()
                NovaMLXLog.info("[sendPretty] tokens=\(tokenCount) nonEmpty=\(nonEmptyTokens) parserState=\(thinkingParser.isInThinkingBlock) finalResponse=\(finalResult.response.prefix(80)) finalThinking=\(finalResult.thinking.prefix(80))")
                if !finalResult.response.isEmpty {
                    messages[assistantIdx].content += finalResult.response
                }

                if messages[assistantIdx].content.isEmpty {
                    NovaMLXLog.warning("[sendPretty] EMPTY RESPONSE for model=\(resolvedModel)")
                    messages[assistantIdx].content = l10n.tr("chat.noResponse")
                }
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    if error is CancellationError || Task.isCancelled {
                        messages[assistantIdx].content = "(cancelled)"
                    } else {
                        messages[assistantIdx].content = l10n.tr("chat.error", error.localizedDescription)
                    }
                }
            }
            lastResponse = messages[assistantIdx].content
        }
    }

    // MARK: Pretty mode for Tokenhub (SSE via API server)

    private func sendPrettyTokenhub(model: String, text: String, assistantIdx: Int, tag: String? = nil) {
        currentInferenceTask = Task {
            defer {
                isLoading = false
                currentInferenceTask = nil
                currentRequestId = nil
            }
            do {
                guard let url = URL(string: "http://127.0.0.1:\(appState.serverPort)/v1/chat/completions") else {
                    messages[assistantIdx].content = "Invalid URL"
                    return
                }
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let key = appState.apiKey { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                var body: [String: Any] = [
                    "model": model,
                    "messages": [["role": "user", "content": text]],
                    "stream": true,
                    "temperature": paramTemp,
                    "max_tokens": Int(paramMaxTokens)
                ]
                if let tag { body["tag"] = tag }
                req.httpBody = try JSONSerialization.data(withJSONObject: body)

                let (bytes, _) = try await URLSession.shared.bytes(for: req)
                // Thinking models split output across `delta.reasoning_content`
                // and `delta.content`. A short max_tokens budget can exhaust
                // before the model finishes thinking and emits any `content`
                // — without this accumulation the message would render as
                // "(no response)" even though real reasoning happened.
                // Render as `<think>...</think>` + visible content so the
                // user sees what the model produced.
                var reasoning = ""
                var reasoningClosed = false
                var visible = ""
                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    guard line.hasPrefix("data: ") else { continue }
                    let payload = String(line.dropFirst(6))
                    if payload == "[DONE]" { break }
                    guard let data = payload.data(using: .utf8),
                          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let choices = json["choices"] as? [[String: Any]],
                          let delta = choices.first?["delta"] as? [String: Any]
                    else { continue }
                    if let r = delta["reasoning_content"] as? String, !r.isEmpty {
                        reasoning += r
                    }
                    if let c = delta["content"] as? String, !c.isEmpty {
                        if !reasoning.isEmpty && !reasoningClosed {
                            reasoningClosed = true
                        }
                        visible += c
                    }
                    // Render: open `<think>` if reasoning started; close it
                    // once content arrives (or stream ends); append visible.
                    var rendered = ""
                    if !reasoning.isEmpty {
                        rendered += "<think>" + reasoning + (reasoningClosed ? "</think>\n" : "")
                    }
                    rendered += visible
                    messages[assistantIdx].content = rendered
                }
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = "(no response)"
                } else if !reasoning.isEmpty && !reasoningClosed {
                    // Stream ended mid-thinking — close the tag so the UI
                    // doesn't show an unclosed `<think>`.
                    messages[assistantIdx].content = "<think>" + reasoning + "</think>"
                }
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    if error is CancellationError || Task.isCancelled {
                        messages[assistantIdx].content = "(cancelled)"
                    } else {
                        messages[assistantIdx].content = "Error: \(error.localizedDescription)"
                    }
                }
            }
            lastResponse = messages[assistantIdx].content
        }
    }

    // MARK: Raw JSON mode (non-streaming HTTP)

    private func sendRawJSON(model: String, text: String, assistantIdx: Int, tag: String? = nil) {
        currentInferenceTask = Task {
            defer {
                isLoading = false
                currentInferenceTask = nil
                currentRequestId = nil
            }
            do {
                guard let url = URL(string: "http://127.0.0.1:\(appState.serverPort)/v1/chat/completions") else {
                    messages[assistantIdx].content = "Invalid URL"
                    return
                }
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let key = appState.apiKey { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                var body: [String: Any] = [
                    "model": model,
                    "messages": [["role": "user", "content": text]],
                    "stream": false,
                    "temperature": paramTemp,
                    "max_tokens": Int(paramMaxTokens),
                    "top_p": paramTopP,
                    "top_k": Int(paramTopK),
                    "min_p": paramMinP,
                    "repetition_penalty": paramRepeatPenalty
                ]
                if let tag { body["tag"] = tag }
                req.httpBody = try JSONSerialization.data(withJSONObject: body)
                if let prettyBody = try? JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]) {
                    lastPayload = String(data: prettyBody, encoding: .utf8)
                }
                let (data, _) = try await URLSession.shared.data(for: req)
                let json = try JSONSerialization.jsonObject(with: data)
                let pretty = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
                messages[assistantIdx].content = String(data: pretty, encoding: .utf8) ?? "Invalid response"
            } catch {
                if error is CancellationError || Task.isCancelled {
                    messages[assistantIdx].content = "(cancelled)"
                } else {
                    messages[assistantIdx].content = "Error: \(error.localizedDescription)"
                }
            }
            lastResponse = messages[assistantIdx].content
        }
    }

    // MARK: Raw Stream mode (SSE)

    private func sendRawStream(model: String, text: String, assistantIdx: Int, tag: String? = nil) {
        currentInferenceTask = Task {
            defer {
                isLoading = false
                currentInferenceTask = nil
                currentRequestId = nil
            }
            do {
                guard let url = URL(string: "http://127.0.0.1:\(appState.serverPort)/v1/chat/completions") else {
                    messages[assistantIdx].content = "Invalid URL"
                    return
                }
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let key = appState.apiKey { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                var body: [String: Any] = [
                    "model": model,
                    "messages": [["role": "user", "content": text]],
                    "stream": true,
                    "temperature": paramTemp,
                    "max_tokens": Int(paramMaxTokens),
                    "top_p": paramTopP,
                    "top_k": Int(paramTopK),
                    "min_p": paramMinP,
                    "repetition_penalty": paramRepeatPenalty
                ]
                if let tag { body["tag"] = tag }
                req.httpBody = try JSONSerialization.data(withJSONObject: body)
                if let prettyBody = try? JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]) {
                    lastPayload = String(data: prettyBody, encoding: .utf8)
                }
                let (bytes, _) = try await URLSession.shared.bytes(for: req)
                var accumulated = ""
                for try await line in bytes.lines {
                    if Task.isCancelled { break }
                    guard line.hasPrefix("data: ") else { continue }
                    let payload = String(line.dropFirst(6))
                    if payload == "[DONE]" {
                        accumulated += "data: [DONE]\n\n"
                        break
                    }
                    if let data = payload.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data),
                       let pretty = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) {
                        accumulated += "data: \(String(data: pretty, encoding: .utf8) ?? payload)\n\n"
                    } else {
                        accumulated += "data: \(payload)\n\n"
                    }
                    messages[assistantIdx].content = accumulated
                }
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = "(no response)"
                }
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    if error is CancellationError || Task.isCancelled {
                        messages[assistantIdx].content = "(cancelled)"
                    } else {
                        messages[assistantIdx].content = "Error: \(error.localizedDescription)"
                    }
                }
            }
            lastResponse = messages[assistantIdx].content
        }
    }

    private func shortModelLabel(_ modelId: String) -> String {
        let shortName = shortModelName(modelId)
        let tag = modelTypeShort(modelType(for: modelId))
        return "[\(tag)] \(shortName)"
    }

    /// Last path component of a model id — e.g. "mlx-community/Qwen3-4B-4bit" → "Qwen3-4B-4bit".
    /// Used by the picker (where the type prefix is rendered as a separate icon)
    /// and by `shortModelLabel` (which still emits the textual tag for non-picker callers).
    private func shortModelName(_ modelId: String) -> String {
        modelId.components(separatedBy: "/").last ?? modelId
    }

    /// Live model type from the loaded container, defaulting to LLM if the
    /// container isn't built yet (shouldn't happen for items in `loadedModels`,
    /// but defensive against UI flashes during load).
    private func modelType(for modelId: String) -> ModelType {
        inferenceService.engine.getContainer(for: modelId)?.config.modelType ?? .llm
    }

    private func modelTypeShort(_ type: ModelType) -> String {
        switch type {
        case .llm: return "LLM"
        case .vlm: return "VLM"
        case .embedding: return "EMB"
        case .audio: return "ASR"
        case .image: return "IMG"
        }
    }

    /// SF Symbol per model type. Keeps the same semantic color mapping as
    /// `modelTypeColor` so icon + color reinforce the type at a glance.
    private func modelTypeIcon(_ type: ModelType) -> String {
        switch type {
        case .llm: return "text.bubble"
        case .vlm: return "eye"
        case .embedding: return "scope"
        case .audio: return "waveform"
        case .image: return "photo"
        }
    }

    private func modelTypeColor(_ type: ModelType) -> Color {
        switch type {
        case .llm: return .blue.opacity(0.8)
        case .vlm: return .purple.opacity(0.8)
        case .embedding: return .green.opacity(0.8)
        case .audio: return .orange.opacity(0.8)
        case .image: return .pink.opacity(0.8)
        }
    }

    private var tokenhubModels: [String] {
        TokenhubManager.shared.list().filter { $0.isEnabled }.map(\.name)
    }

    /// Enabled Load Balancers from SQLite. Reads the same `load_balancers` table
    /// `LoadBalancersPageView` shows, so the picker sees fresh state without the
    /// page being open. Disabled LBs are hidden — they can't accept requests.
    private var loadBalancerEntries: [LoadBalancer] {
        (try? NovaDB.shared.loadBalancerStore.listLBs())?
            .filter { $0.isEnabled } ?? []
    }

    // MARK: - Chat Audio

    private var hasLoadedASRModel: Bool {
        appState.loadedModels.contains { modelId in
            guard let record = modelManager.getRecord(modelId) else { return false }
            return record.family == .whisper || record.family == .qwen3Asr
        }
    }

    private var firstLoadedASRModel: String? {
        appState.loadedModels.first { modelId in
            guard let record = modelManager.getRecord(modelId) else { return false }
            return record.family == .whisper || record.family == .qwen3Asr
        }
    }

    private func toggleChatRecording() {
        if isRecording {
            chatRecorder?.stop()
            isRecording = false
            guard let url = chatRecordingURL else { return }
            transcribeChatRecording(url: url)
        } else {
            let tempDir = FileManager.default.temporaryDirectory
            let url = tempDir.appendingPathComponent("novamlx_chat_\(UUID().uuidString).wav")
            let settings: [String: Any] = [
                AVFormatIDKey: Int(kAudioFormatLinearPCM),
                AVSampleRateKey: 16000.0,
                AVNumberOfChannelsKey: 1,
                AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
            ]
            do {
                let recorder = try AVAudioRecorder(url: url, settings: settings)
                recorder.record()
                chatRecorder = recorder
                chatRecordingURL = url
                isRecording = true
            } catch {
                NovaMLXLog.error("Chat recording failed: \(error)")
            }
        }
    }

    private func transcribeChatRecording(url: URL) {
        guard let modelId = firstLoadedASRModel else { return }
        Task {
            do {
                let audioData = try Data(contentsOf: url)
                let result = try await inferenceService.transcriptionService.transcribe(
                    modelId: modelId,
                    audioData: audioData
                )
                await MainActor.run {
                    if !result.text.isEmpty {
                        inputText = result.text
                        isInputFocused = true
                    }
                }
            } catch {
                NovaMLXLog.error("Chat ASR failed: \(error)")
            }
            try? FileManager.default.removeItem(at: url)
        }
    }

    private func speakMessage(_ msg: ChatMessageRow) {
        if ttsPlayingMessageId == msg.id {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/killall")
            proc.arguments = ["say"]
            try? proc.run()
            ttsPlayingMessageId = nil
            return
        }
        ttsPlayingMessageId = msg.id
        let text = msg.content
        Task {
            do {
                let _ = try await inferenceService.ttsService.synthesize(text: text, voice: "Tingting")
                await MainActor.run {
                    ttsPlayingMessageId = nil
                }
            } catch {
                await MainActor.run {
                    ttsPlayingMessageId = nil
                }
            }
        }
    }

    private var availableTags: [String] {
        TokenhubManager.shared.allTags()
    }

    // MARK: - ASR Playground

    private var asrContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Input controls
                VStack(spacing: 12) {
                    HStack(spacing: 12) {
                        Button { toggleASRRecording() } label: {
                            HStack(spacing: 8) {
                                Image(systemName: isASRRecording ? "stop.circle.fill" : "mic.circle.fill")
                                    .font(.system(size: 20))
                                Text(isASRRecording ? l10n.tr("audio.asr.stopRecording") : l10n.tr("audio.asr.startRecording"))
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(isASRRecording ? Color.red.opacity(0.2) : NovaTheme.Colors.accent.opacity(0.15))
                            .foregroundColor(isASRRecording ? .red : NovaTheme.Colors.accent)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .buttonStyle(.plain)
                        .disabled(selectedASRModel.isEmpty || isTranscribing)

                        Button { uploadAudioFile() } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "doc.badge.plus")
                                    .font(.system(size: 20))
                                Text(l10n.tr("audio.asr.uploadFile"))
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(NovaTheme.Colors.cardBackground)
                            .foregroundColor(.secondary)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
                        }
                        .buttonStyle(.plain)
                        .disabled(selectedASRModel.isEmpty || isTranscribing)
                    }

                    if let name = uploadedFileName {
                        HStack(spacing: 4) {
                            Image(systemName: "doc.fill").font(.caption)
                            Text(name).font(.caption).lineLimit(1)
                            Spacer()
                            if asrRecordingURL != nil {
                                Button { downloadRecording() } label: {
                                    Image(systemName: "arrow.down.circle.fill").font(.caption)
                                        .foregroundColor(NovaTheme.Colors.accent)
                                }.buttonStyle(.plain)
                            }
                            Button { uploadedFileName = nil; asrRecordingURL = nil } label: {
                                Image(systemName: "xmark.circle.fill").font(.caption).foregroundColor(.secondary)
                            }.buttonStyle(.plain)
                        }.foregroundColor(.secondary)
                    }
                }

                if isTranscribing {
                    HStack(spacing: 8) {
                        ProgressView().scaleEffect(0.8)
                        Text(l10n.tr("audio.asr.transcribing")).font(.system(size: 12)).foregroundColor(.secondary)
                    }
                }

                if let error = asrError {
                    Text(error).font(.system(size: 12)).foregroundColor(.red).padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1)).clipShape(RoundedRectangle(cornerRadius: 6))
                }

                if !transcriptionText.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(l10n.tr("audio.asr.result")).font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(transcriptionText, forType: .string) } label: {
                                Image(systemName: "doc.on.doc").font(.system(size: 11)).foregroundColor(.secondary)
                            }.buttonStyle(.plain)
                        }
                        TextEditor(text: .constant(transcriptionText))
                            .font(.system(size: 13)).frame(minHeight: 120).padding(8)
                            .background(Color(nsColor: .textBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(RoundedRectangle(cornerRadius: 6).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))
                    }
                }

                if !selectedASRModel.isEmpty { asrApiExamples }

                Spacer(minLength: 0)
            }
            .padding(20)
        }
    }

    // MARK: - TTS Playground

    private var ttsContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Engine picker
                VStack(alignment: .leading, spacing: 6) {
                    Text(l10n.tr("audio.tts.model")).font(.system(size: 13, weight: .semibold))
                    HStack(spacing: 0) {
                        ForEach(TTSEngine.allCases, id: \.self) { engine in
                            Button {
                                ttsEngine = engine
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: engine == .neural ? "brain" : "speaker.wave.2.fill").font(.system(size: 11))
                                    Text(engine.rawValue).font(.system(size: 12, weight: ttsEngine == engine ? .semibold : .regular))
                                }
                                .frame(maxWidth: .infinity).padding(.vertical, 6)
                                .background(ttsEngine == engine ? NovaTheme.Colors.accent.opacity(0.15) : Color.clear)
                                .foregroundColor(ttsEngine == engine ? NovaTheme.Colors.accent : .secondary)
                                .clipShape(RoundedRectangle(cornerRadius: 6)).contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(3).background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                    if ttsEngine == .neural {
                        if let ttsName = loadedTTSModelName {
                            HStack(spacing: 6) {
                                Image(systemName: "checkmark.circle.fill").font(.system(size: 11)).foregroundColor(.green)
                                Text(ttsName).font(.system(size: 11)).foregroundColor(.secondary)
                            }.padding(6)
                        } else {
                            HStack(spacing: 6) {
                                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 11)).foregroundColor(.orange)
                                Text(l10n.tr("audio.tts.noModel")).font(.system(size: 11)).foregroundColor(.secondary)
                            }.padding(6)
                        }
                    }

                    if ttsEngine == .neural {
                        voiceProfileSection
                    } else {
                        if macOSVoices.isEmpty {
                            Text("Loading voices...").font(.system(size: 11)).foregroundColor(.secondary)
                        } else {
                            Picker("Voice", selection: $selectedSystemVoice) {
                                ForEach(macOSVoices) { voice in
                                    Text("\(voice.name) (\(voice.locale))").tag(voice.name)
                                }
                            }
                            .pickerStyle(.menu).frame(maxWidth: .infinity)
                        }
                    }
                }

                // Text input
                VStack(alignment: .leading, spacing: 6) {
                    Text(l10n.tr("audio.tts.input")).font(.system(size: 13, weight: .semibold))
                    TextEditor(text: $ttsText)
                        .font(.system(size: 13)).frame(minHeight: 120).padding(4)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))
                }

                // Synthesize + Play
                HStack(spacing: 12) {
                    Button { synthesizeSpeech() } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "speaker.wave.2.fill").font(.system(size: 14))
                            Text(l10n.tr("audio.tts.synthesize")).font(.system(size: 13, weight: .medium))
                        }
                        .frame(maxWidth: .infinity).padding(.vertical, 10)
                        .background(NovaTheme.Colors.accent).foregroundColor(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                    .disabled(ttsText.isEmpty || isSynthesizing || (ttsEngine == .neural && loadedTTSModelName == nil))

                    if synthesizedAudioURL != nil {
                        Button { togglePlayback() } label: {
                            HStack(spacing: 8) {
                                Image(systemName: isPlaying ? "pause.fill" : "play.fill").font(.system(size: 14))
                                Text(isPlaying ? l10n.tr("audio.tts.pause") : l10n.tr("audio.tts.play")).font(.system(size: 13, weight: .medium))
                            }
                            .frame(maxWidth: .infinity).padding(.vertical, 10)
                            .background(NovaTheme.Colors.accent.opacity(0.15))
                            .foregroundColor(NovaTheme.Colors.accent)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .buttonStyle(.plain)
                    }
                }

                if isSynthesizing {
                    HStack(spacing: 8) {
                        ProgressView().scaleEffect(0.8)
                        Text(l10n.tr("audio.tts.synthesizing")).font(.system(size: 12)).foregroundColor(.secondary)
                    }
                }

                if let error = ttsError {
                    Text(error).font(.system(size: 12)).foregroundColor(.red).padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1)).clipShape(RoundedRectangle(cornerRadius: 6))
                }
                if let success = ttsSuccess {
                    Text(success).font(.system(size: 12)).foregroundColor(.green).padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.1)).clipShape(RoundedRectangle(cornerRadius: 6))
                }

                ttsApiExamples
                Spacer(minLength: 0)
            }
            .padding(20)
        }
    }

    // MARK: - Voice Profile Section

    private var voiceProfileSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "person.wave.2").font(.system(size: 11)).foregroundColor(.secondary)
                Text(l10n.tr("audio.tts.voiceProfile")).font(.system(size: 12, weight: .semibold))
                Spacer()
                Button { showCloneSheet = true } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "plus.circle.fill").font(.system(size: 11))
                        Text(l10n.tr("audio.tts.cloneNew")).font(.system(size: 11, weight: .medium))
                    }
                    .foregroundColor(NovaTheme.Colors.accent)
                }
                .buttonStyle(.plain)
            }

            if voiceProfiles.isEmpty {
                Text(l10n.tr("audio.tts.noVoiceProfiles")).font(.system(size: 11)).foregroundColor(.secondary)
                    .padding(8).frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.06)).clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(voiceProfiles) { profile in
                            VoiceProfileCard(profile: profile, isSelected: selectedProfileId == profile.id, l10n: l10n) {
                                selectedProfileId = profile.id
                            } onDelete: {
                                VoiceProfileManager.shared.deleteProfile(profile.id)
                                if selectedProfileId == profile.id {
                                    selectedProfileId = voiceProfiles.first(where: { $0.id != profile.id })?.id
                                }
                                loadVoiceProfiles()
                            }
                        }
                    }.padding(.vertical, 2)
                }
            }
        }.padding(.horizontal, 6)
    }

    // MARK: - ASR/TTS Computed Properties

    private var loadedTTSModelName: String? {
        let ttsModels = appState.loadedModels.filter { modelId in
            guard let record = modelManager.getRecord(modelId) else { return false }
            return record.family == .dotsTts || record.family == .qwen3Tts
        }
        return ttsModels.first.map { shortModelLabel($0) }
    }

    private var loadedASRModels: [String] {
        appState.loadedModels.filter { modelId in
            guard let record = modelManager.getRecord(modelId) else { return false }
            return record.family == .whisper || record.family == .qwen3Asr
        }
    }

    private var downloadedASRModels: [(id: String, isLoaded: Bool)] {
        modelManager.downloadedModels()
            .filter { $0.family == .whisper || $0.family == .qwen3Asr }
            .map { (id: $0.id, isLoaded: appState.loadedModels.contains($0.id)) }
    }

    private var realApiKey: String { appState.apiKey ?? "YOUR_API_KEY" }

    // MARK: - ASR Actions

    private func autoSelectASRModel() {
        let asrModels = loadedASRModels
        if asrModels.count == 1, selectedASRModel.isEmpty || !asrModels.contains(selectedASRModel) {
            selectedASRModel = asrModels[0]
        } else if !asrModels.contains(selectedASRModel) {
            selectedASRModel = asrModels.first ?? ""
        }
    }

    private func loadASRModel(_ modelId: String) {
        isASRModelLoading = true
        loadingASRModelName = modelId.components(separatedBy: "/").last ?? modelId
        asrError = nil
        Task {
            do {
                guard let record = modelManager.getRecord(modelId) else {
                    throw NovaMLXError.modelNotFound(modelId)
                }
                let config = ModelConfig(
                    identifier: ModelIdentifier(id: modelId, family: record.family),
                    modelType: record.modelType
                )
                _ = try await inferenceService.transcriptionService.loadModel(from: record.localURL, config: config)
                selectedASRModel = modelId
                await MainActor.run { isASRModelLoading = false; loadingASRModelName = nil }
            } catch {
                await MainActor.run { asrError = error.localizedDescription; isASRModelLoading = false; loadingASRModelName = nil }
            }
        }
    }

    private func toggleASRRecording() {
        if isASRRecording {
            asrRecorder?.stop()
            isASRRecording = false
            uploadedFileName = "recording.wav"
            guard let url = asrRecordingURL else { return }
            transcribeAudio(url: url)
        } else {
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized: startASRRecording()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async { if granted { self.startASRRecording() } else { self.asrError = "Microphone access denied." } }
                }
            case .denied, .restricted: asrError = "Microphone access denied. Please allow in System Settings."
            @unknown default: startASRRecording()
            }
        }
    }

    private func startASRRecording() {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx_asr_\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM), AVSampleRateKey: 48000.0,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsFloatKey: false
        ]
        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.record()
            asrRecorder = recorder
            asrRecordingURL = url
            isASRRecording = true
            uploadedFileName = nil
            asrError = nil
        } catch {
            asrError = "Recording failed: \(error.localizedDescription)"
        }
    }

    private func uploadAudioFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        uploadedFileName = url.lastPathComponent
        asrRecordingURL = url
        transcribeAudio(url: url)
    }

    private func downloadRecording() {
        guard let srcURL = asrRecordingURL else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "recording.wav"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        try? FileManager.default.removeItem(at: dest)
        do { try FileManager.default.copyItem(at: srcURL, to: dest) }
        catch { asrError = "Download failed: \(error.localizedDescription)" }
    }

    private func transcribeAudio(url: URL) {
        guard !selectedASRModel.isEmpty else {
            asrError = l10n.tr("audio.asr.noModel"); return
        }
        let modelId = selectedASRModel
        isTranscribing = true
        transcriptionText = ""
        asrError = nil
        Task {
            do {
                let audioData = try Data(contentsOf: url)
                let result = try await inferenceService.transcriptionService.transcribe(modelId: modelId, audioData: audioData, language: nil)
                await MainActor.run { transcriptionText = result.text; isTranscribing = false }
            } catch {
                await MainActor.run { asrError = error.localizedDescription; isTranscribing = false }
            }
        }
    }

    // MARK: - TTS Actions

    private func loadMacOSVoices() {
        guard macOSVoices.isEmpty else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            let voices = TTSService.listMacOSVoices()
            DispatchQueue.main.async { macOSVoices = voices }
        }
    }

    private func loadVoiceProfiles() {
        voiceProfiles = VoiceProfileManager.shared.listProfiles()
    }

    private func synthesizeSpeech() {
        guard !ttsText.isEmpty else { return }
        isSynthesizing = true; ttsError = nil; ttsSuccess = nil
        let profile = voiceProfiles.first { $0.id == selectedProfileId }
        Task {
            do {
                let audioData = try await inferenceService.ttsService.synthesize(
                    text: ttsText, voice: selectedSystemVoice, engine: ttsEngine, voiceProfile: profile
                )
                let ext = ttsEngine == .system ? "aiff" : "wav"
                let url = FileManager.default.temporaryDirectory.appendingPathComponent("novamlx_tts_\(UUID().uuidString).\(ext)")
                try audioData.write(to: url)
                await MainActor.run { synthesizedAudioURL = url; isSynthesizing = false }
            } catch {
                await MainActor.run { ttsError = error.localizedDescription; isSynthesizing = false }
            }
        }
    }

    private func togglePlayback() {
        if isPlaying { audioPlayer?.stop(); isPlaying = false; return }
        guard let url = synthesizedAudioURL else { return }
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = nil
            audioPlayer = player
            player.play()
            isPlaying = true
            DispatchQueue.main.asyncAfter(deadline: .now() + player.duration) { isPlaying = false }
        } catch { ttsError = "Playback failed: \(error.localizedDescription)" }
    }

    // MARK: - API Example Helpers

    private enum CodeTab: String, CaseIterable { case curl, python, node }

    private func codeBlock(_ code: String) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code).font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(nsColor: .secondaryLabelColor)).textSelection(.enabled).padding(8)
            }
            Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(code, forType: .string) } label: {
                Image(systemName: "doc.on.doc").font(.system(size: 10)).foregroundColor(.secondary)
            }
            .buttonStyle(.plain).padding(6)
        }
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))
    }

    private func langTabs(_ selected: Binding<CodeTab>) -> some View {
        HStack(spacing: 0) {
            ForEach(CodeTab.allCases, id: \.self) { tab in
                Button { selected.wrappedValue = tab } label: {
                    Text(tab.rawValue.uppercased())
                        .font(.system(size: 10, weight: selected.wrappedValue == tab ? .semibold : .regular))
                        .frame(maxWidth: .infinity).padding(.vertical, 4)
                        .background(selected.wrappedValue == tab ? NovaTheme.Colors.accent.opacity(0.15) : Color.clear)
                        .foregroundColor(selected.wrappedValue == tab ? NovaTheme.Colors.accent : .secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }.buttonStyle(.plain)
            }
        }.padding(2).background(Color(nsColor: .controlBackgroundColor)).clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private var asrApiExamples: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "terminal").font(.system(size: 11))
                Text("API Examples").font(.system(size: 12, weight: .semibold))
                Spacer()
                langTabs($asrCodeTab)
            }.foregroundColor(.secondary)
            let model = selectedASRModel; let key = realApiKey
            switch asrCodeTab {
            case .curl:
                codeBlock("curl -X POST http://localhost:6590/v1/audio/transcriptions \\\n  -H \"Authorization: Bearer \(key)\" \\\n  -F \"file=@recording.wav\" \\\n  -F \"model=\(model)\"")
            case .python:
                codeBlock("import requests\n\nresp = requests.post(\n    \"http://localhost:6590/v1/audio/transcriptions\",\n    headers={\"Authorization\": \"Bearer \(key)\"},\n    files={\"file\": open(\"recording.wav\", \"rb\")},\n    data={\"model\": \"\(model)\"}\n)\nprint(resp.json())")
            case .node:
                codeBlock("const FormData = require(\"form-data\");\nconst fs = require(\"fs\");\nconst form = new FormData();\nform.append(\"file\", fs.createReadStream(\"recording.wav\"));\nform.append(\"model\", \"\(model)\");\nconst resp = await fetch(\"http://localhost:6590/v1/audio/transcriptions\", {\n  method: \"POST\",\n  headers: { Authorization: \"Bearer \(key)\", ...form.getHeaders() },\n  body: form\n});\nconsole.log(await resp.json());")
            }
        }.padding(10).background(Color(nsColor: .controlBackgroundColor).opacity(0.5)).clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var ttsApiExamples: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "terminal").font(.system(size: 11))
                Text("API Examples").font(.system(size: 12, weight: .semibold))
                Spacer()
                langTabs($ttsCodeTab)
            }.foregroundColor(.secondary)
            let key = realApiKey
            let profileName = voiceProfiles.first(where: { $0.id == selectedProfileId })?.name ?? "MyVoice"
            switch ttsCodeTab {
            case .curl:
                codeBlock("curl -X POST http://localhost:6590/v1/audio/speech \\\n  -H \"Authorization: Bearer \(key)\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"model\":\"tts-1\",\"input\":\"Hello world\",\"voice\":\"\(profileName)\"}' \\\n  --output speech.wav")
            case .python:
                codeBlock("import requests\n\nresp = requests.post(\n    \"http://localhost:6590/v1/audio/speech\",\n    headers={\"Authorization\": \"Bearer \(key)\"},\n    json={\"model\": \"tts-1\", \"input\": \"Hello world\", \"voice\": \"\(profileName)\"}\n)\nwith open(\"speech.wav\", \"wb\") as f:\n    f.write(resp.content)")
            case .node:
                codeBlock("const resp = await fetch(\"http://localhost:6590/v1/audio/speech\", {\n  method: \"POST\",\n  headers: { \"Authorization\": \"Bearer \(key)\", \"Content-Type\": \"application/json\" },\n  body: JSON.stringify({ model: \"tts-1\", input: \"Hello world\", voice: \"\(profileName)\" })\n});\nconst buf = Buffer.from(await resp.arrayBuffer());\nrequire(\"fs\").writeFileSync(\"speech.wav\", buf);")
            }
        }.padding(10).background(Color(nsColor: .controlBackgroundColor).opacity(0.5)).clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Image Playground

    private var downloadedImageModels: [ModelRecord] {
        modelManager.downloadedModels().filter { $0.modelType == .image }
    }

    private var loadedImageModels: [String] {
        inferenceService.imageGenerationService.listLoadedModels()
    }

    private var imageContent: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Model picker
                VStack(alignment: .leading, spacing: 6) {
                    Text("Model").font(.system(size: 11, weight: .semibold)).foregroundColor(.secondary)
                    Picker("Image Model", selection: $selectedImageModel) {
                        Text("Select model...").tag("")
                        ForEach(downloadedImageModels, id: \.id) { record in
                            HStack {
                                Text(record.id)
                                if loadedImageModels.contains(record.id) {
                                    Circle().fill(NovaTheme.Colors.statusOK).frame(width: 6, height: 6)
                                }
                            }.tag(record.id)
                        }
                    }
                    .onChange(of: appState.loadedModels) { _, _ in
                        if selectedImageModel.isEmpty, let first = downloadedImageModels.first {
                            selectedImageModel = first.id
                        }
                    }
                    .onAppear {
                        if selectedImageModel.isEmpty, let first = downloadedImageModels.first {
                            selectedImageModel = first.id
                        }
                    }
                }

                // Settings row
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Size").font(.system(size: 10, weight: .medium)).foregroundColor(.secondary)
                        Picker("Size", selection: $imageSize) {
                            Text("1024×1024").tag("1024x1024")
                            Text("768×1344").tag("768x1344")
                            Text("1344×768").tag("1344x768")
                            Text("864×1152").tag("864x1152")
                            Text("1152×864").tag("1152x864")
                        }.frame(width: 140)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Steps").font(.system(size: 10, weight: .medium)).foregroundColor(.secondary)
                        HStack(spacing: 4) {
                            TextField("Steps", value: $imageSteps, format: .number)
                                .textFieldStyle(.roundedBorder).frame(width: 50)
                                .font(.system(size: 11, design: .monospaced))
                            Text(isSchnellModel ? "(1-8)" : "(10-50)").font(.system(size: 9)).foregroundColor(Color(nsColor: .tertiaryLabelColor))
                        }
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Seed").font(.system(size: 10, weight: .medium)).foregroundColor(.secondary)
                        TextField("Auto", text: $imageSeed)
                            .textFieldStyle(.roundedBorder).frame(width: 80)
                            .font(.system(size: 11, design: .monospaced))
                    }
                }

                // Prompt
                VStack(alignment: .leading, spacing: 6) {
                    Text("Prompt").font(.system(size: 11, weight: .semibold)).foregroundColor(.secondary)
                    TextEditor(text: $imagePrompt)
                        .font(.system(size: 13))
                        .frame(minHeight: 60, maxHeight: 100)
                        .scrollContentBackground(.hidden)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))
                }

                // Actions
                HStack(spacing: 8) {
                    Button {
                        generateImage()
                    } label: {
                        HStack(spacing: 4) {
                            if isGeneratingImage { ProgressView().controlSize(.small) }
                            Image(systemName: "photo")
                            Text(isGeneratingImage ? "Generating..." : "Generate")
                        }
                    }
                    .disabled(imagePrompt.isEmpty || selectedImageModel.isEmpty || isGeneratingImage || isImageModelLoading)
                    .buttonStyle(.borderedProminent)

                    if generatedImage != nil {
                        Button { saveGeneratedImage() } label: {
                            HStack(spacing: 4) { Image(systemName: "square.and.arrow.down"); Text("Save") }
                        }.buttonStyle(.bordered)
                    }

                    if generatedImage != nil || imageError != nil {
                        Button {
                            generatedImage = nil; generatedImageData = nil; imageError = nil
                        } label: {
                            Text("Clear")
                        }.buttonStyle(.bordered)
                    }
                }

                // Loading state
                if isImageModelLoading, let name = loadingImageModelName {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Loading \(name)...").font(.system(size: 12)).foregroundColor(.secondary)
                    }.padding(8)
                }

                // Error
                if let error = imageError {
                    Text(error).font(.system(size: 11)).foregroundColor(.red).padding(8)
                        .background(Color.red.opacity(0.1)).clipShape(RoundedRectangle(cornerRadius: 6))
                }

                // Result
                if let image = generatedImage {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Result").font(.system(size: 11, weight: .semibold)).foregroundColor(.secondary)
                        Image(nsImage: image)
                            .resizable().aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 400)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .shadow(color: .black.opacity(0.1), radius: 4)
                    }
                }

                // API Examples
                if !selectedImageModel.isEmpty {
                    imageApiExamples
                }
            }
            .padding(16)
        }
    }

    private var isSchnellModel: Bool {
        guard let record = modelManager.getRecord(selectedImageModel) else { return true }
        return record.id.lowercased().contains("schnell")
    }

    private var imageApiExamples: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "terminal").font(.system(size: 11))
                Text("API Examples").font(.system(size: 12, weight: .semibold))
                Spacer()
                langTabs($imageCodeTab)
            }.foregroundColor(.secondary)
            let model = selectedImageModel; let key = realApiKey
            switch imageCodeTab {
            case .curl:
                codeBlock("curl -X POST http://localhost:6590/v1/images/generations \\\n  -H \"Authorization: Bearer \(key)\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"model\":\"\(model)\",\"prompt\":\"\(imagePrompt.isEmpty ? "A sunset over mountains" : imagePrompt)\",\"size\":\"\(imageSize)\",\"steps\":\(imageSteps)}'")
            case .python:
                codeBlock("import openai\nclient = openai.OpenAI(base_url=\"http://localhost:6590/v1\", api_key=\"\(key)\")\nresponse = client.images.generate(\n    model=\"\(model)\",\n    prompt=\"\(imagePrompt.isEmpty ? "A sunset over mountains" : imagePrompt)\",\n    size=\"\(imageSize)\"\n)\nprint(response.data[0].b64_json[:100])")
            case .node:
                codeBlock("import OpenAI from 'openai';\nconst client = new OpenAI({ baseURL: 'http://localhost:6590/v1', apiKey: '\(key)' });\nconst response = await client.images.generate({\n  model: '\(model)',\n  prompt: '\(imagePrompt.isEmpty ? "A sunset over mountains" : imagePrompt)',\n  size: '\(imageSize)'\n});\nconsole.log(response.data[0].b64_json?.slice(0, 100));")
            }
        }.padding(10).background(Color(nsColor: .controlBackgroundColor).opacity(0.5)).clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func generateImage() {
        guard !imagePrompt.isEmpty, !selectedImageModel.isEmpty else { return }
        imageError = nil
        isGeneratingImage = true

        Task {
            do {
                // Auto-load model if needed
                if !loadedImageModels.contains(selectedImageModel) {
                    guard let record = modelManager.getRecord(selectedImageModel) else {
                        throw NSError(domain: "NovaMLX", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not found"])
                    }
                    let config = ModelConfig(identifier: ModelIdentifier(id: selectedImageModel, family: record.family), modelType: .image)
                    isImageModelLoading = true
                    loadingImageModelName = record.id
                    _ = try await inferenceService.imageGenerationService.loadModel(
                        from: record.localURL, config: config)
                    isImageModelLoading = false
                    loadingImageModelName = nil
                }

                let dims = imageSize.components(separatedBy: "x")
                let w = Int(dims.first ?? "1024") ?? 1024
                let h = Int(dims.last ?? "1024") ?? 1024
                let seed = imageSeed.isEmpty ? nil : UInt64(imageSteps)

                let result = try await inferenceService.imageGenerationService.generate(
                    modelId: selectedImageModel,
                    prompt: imagePrompt,
                    width: w, height: h,
                    seed: seed,
                    steps: imageSteps
                )

                if let b64 = result.images.first,
                   let data = Data(base64Encoded: b64) {
                    generatedImageData = data
                    generatedImage = NSImage(data: data)
                }
            } catch {
                imageError = error.localizedDescription
                isImageModelLoading = false
                loadingImageModelName = nil
            }
            isGeneratingImage = false
        }
    }

    private func saveGeneratedImage() {
        guard let data = generatedImageData else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "flux_\(Int(Date().timeIntervalSince1970)).png"
        panel.begin { response in
            if response == .OK, let url = panel.url {
                try? data.write(to: url)
            }
        }
    }
}

private struct ChatMessageRow: Identifiable {
    let id = UUID()
    var content: String
    let isUser: Bool
}

// MARK: - Parameter Slider

private struct ParamSlider: View {
    let label: String
    @Binding var value: Double
    let min: Double
    let max: Double
    let step: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                Spacer()
                Text(step >= 1 ? "\(Int(value))" : String(format: "%.2f", value))
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundColor(NovaTheme.Colors.accent)
            }
            Slider(value: $value, in: min...max, step: step)
                .controlSize(.mini)
        }
    }
}
