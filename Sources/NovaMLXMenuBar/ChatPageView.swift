import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils
import NovaMLXEngine

struct ChatPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService

    @EnvironmentObject var l10n: L10n
    @State private var messages: [ChatMessageRow] = []
    @State private var inputText = ""
    @State private var selectedModel = ""
    @State private var isLoading = false

    // History navigation
    @State private var sentHistory: [String] = []
    @State private var historyIndex: Int? = nil
    @State private var savedDraft: String? = nil

    // Parameter controls
    @FocusState private var isInputFocused: Bool
    @State private var showParams = false
    @State private var paramTemp: Double = 0.7
    @State private var paramMaxTokens: Double = 4096
    @State private var paramTopP: Double = 0.9
    @State private var paramTopK: Double = 0
    @State private var paramMinP: Double = 0
    @State private var paramRepeatPenalty: Double = 1.0
    @State private var thinkingExpanded: Set<UUID> = []

    var body: some View {
        VStack(spacing: 0) {
            infoBanner
            chatToolbar
            if showParams { paramsPanel }
            Divider()
            messageList
            Divider()
            inputBar
        }
    }

    private var chatToolbar: some View {
        HStack(spacing: 12) {
            Picker(l10n.tr("chat.model"), selection: $selectedModel) {
                if appState.loadedModels.isEmpty && appState.cloudModels.isEmpty {
                    Text(l10n.tr("chat.noModels")).tag("")
                }
                ForEach(appState.loadedModels, id: \.self) { model in
                    Text(model.components(separatedBy: "/").last ?? model).tag(model)
                }
                if !appState.cloudModels.isEmpty {
                    Divider()
                    ForEach(appState.cloudModels, id: \.self) { model in
                        HStack {
                            Image(systemName: "cloud.fill")
                                .font(.system(size: 10))
                            Text(model.components(separatedBy: ":cloud").first ?? model)
                        }
                        .tag(model)
                    }
                }
            }
            .frame(width: 240)

            Spacer()

            if isLoading {
                ProgressView()
                    .controlSize(.small)
            }

            Button(action: { withAnimation(.easeInOut(duration: 0.2)) { showParams.toggle() } }) {
                Image(systemName: showParams ? "slider.horizontal.3" : "slider.horizontal.3")
                    .font(.system(size: 13))
                    .foregroundColor(showParams ? NovaTheme.Colors.accent : .secondary)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
            .help(l10n.tr("chat.toggleParams"))

            Button(l10n.tr("chat.clear")) {
                messages.removeAll()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)
        .onAppear {
            if selectedModel.isEmpty {
                if let first = appState.loadedModels.first {
                    selectedModel = first
                    loadDefaultsFromModel(first)
                } else if let first = appState.cloudModels.first {
                    selectedModel = first
                }
            }
        }
        .onChange(of: appState.loadedModels) { _, newModels in
            if !newModels.contains(selectedModel) && !appState.cloudModels.contains(selectedModel) {
                if let first = newModels.first {
                    selectedModel = first
                    loadDefaultsFromModel(first)
                } else if let first = appState.cloudModels.first {
                    selectedModel = first
                }
            }
        }
        .onChange(of: selectedModel) { _, newModel in
            if !newModel.isEmpty && !CloudBackend.isCloudModel(newModel) {
                loadDefaultsFromModel(newModel)
            }
        }
    }

    private var infoBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .foregroundColor(NovaTheme.Colors.accent)
                .font(.system(size: 12))

            Text(l10n.tr("chat.rawOutput"))
                .font(.system(size: 11))
                .foregroundColor(NovaTheme.Colors.textSecondary)

            Spacer()

            Button(l10n.tr("chat.webChat")) {
                NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(String(appState.serverPort))/chat")!)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(NovaTheme.Colors.accentDim)
    }

    private var paramsPanel: some View {
        VStack(spacing: 8) {
            HStack(spacing: 16) {
                ParamSlider(label: l10n.tr("chat.temperature"), value: $paramTemp, min: 0, max: 2, step: 0.05)
                ParamSlider(label: l10n.tr("chat.topP"), value: $paramTopP, min: 0, max: 1, step: 0.05)
                ParamSlider(label: l10n.tr("chat.topK"), value: $paramTopK, min: 0, max: 200, step: 1)
                ParamSlider(label: l10n.tr("chat.minP"), value: $paramMinP, min: 0, max: 1, step: 0.05)
            }
            HStack(spacing: 16) {
                ParamSlider(label: l10n.tr("chat.maxTokens"), value: $paramMaxTokens, min: 64, max: 32768, step: 64)
                ParamSlider(label: l10n.tr("chat.repeatPenalty"), value: $paramRepeatPenalty, min: 1.0, max: 2.0, step: 0.05)

                Spacer()

                Button(l10n.tr("chat.resetDefault")) {
                    loadDefaultsFromModel(selectedModel)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .foregroundColor(NovaTheme.Colors.accent)
            }
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
    }

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
        HStack {
            if msg.isUser { Spacer(minLength: 60) }
            VStack(alignment: msg.isUser ? .trailing : .leading, spacing: 6) {
                Text(msg.isUser ? l10n.tr("chat.you") : l10n.tr("chat.assistant"))
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // Thinking section — collapsible, only for assistant messages with thinking content
                if !msg.isUser && !msg.thinkingContent.isEmpty {
                    DisclosureGroup(
                        isExpanded: Binding(
                            get: { thinkingExpanded.contains(msg.id) },
                            set: { expanded in
                                if expanded {
                                    thinkingExpanded.insert(msg.id)
                                } else {
                                    thinkingExpanded.remove(msg.id)
                                }
                            }
                        )
                    ) {
                        Text(msg.thinkingContent)
                            .font(.system(size: 12))
                            .textSelection(.enabled)
                            .foregroundColor(NovaTheme.Colors.textSecondary)
                            .padding(.top, 4)
                    } label: {
                        Text(l10n.tr("chat.thinking") + " (\(msg.thinkingContent.count) chars)")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(NovaTheme.Colors.accent)
                    }
                    .padding(10)
                    .background(NovaTheme.Colors.accentDim.opacity(0.3))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                // Response content
                if !msg.content.isEmpty {
                    Text(msg.content)
                        .font(.system(size: 13))
                        .textSelection(.enabled)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: msg.isUser ? .trailing : .leading)
                        .background(
                            msg.isUser
                                ? NovaTheme.Colors.accentDim
                                : NovaTheme.Colors.cardBackground
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
            if !msg.isUser { Spacer(minLength: 60) }
        }
    }

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

            Button(action: { sendMessage() }) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || isLoading || selectedModel.isEmpty)
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)
    }

    private enum HistoryDirection { case up, down }

    private func navigateHistory(_ direction: HistoryDirection) {
        guard !sentHistory.isEmpty else { return }

        switch direction {
        case .up:
            // Save current draft before entering history
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
                // Past the end — restore draft
                historyIndex = nil
                inputText = savedDraft ?? ""
                savedDraft = nil
            } else {
                historyIndex = idx + 1
                inputText = sentHistory[idx + 1]
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !selectedModel.isEmpty else { return }

        // Push to history
        if sentHistory.last != text {
            sentHistory.append(text)
        }
        historyIndex = nil
        savedDraft = nil

        inputText = ""
        messages.append(ChatMessageRow(content: text, isUser: true))

        // Add empty assistant message — will be filled token by token
        let assistantMsg = ChatMessageRow(content: "", isUser: false)
        messages.append(assistantMsg)
        let assistantIdx = messages.count - 1

        isLoading = true
        let model = selectedModel

        Task {
            // Guard: model must be loaded before we attempt inference
            guard inferenceService.isModelLoaded(model) else {
                messages[assistantIdx].content = l10n.tr("chat.error", "Model '\(model.components(separatedBy: "/").last ?? model)' is not loaded. Load it from the Models page first.")
                isLoading = false
                return
            }

            let request = InferenceRequest(
                model: model,
                messages: [ChatMessage(role: .user, content: text)],
                temperature: paramTemp,
                maxTokens: Int(paramMaxTokens),
                topP: paramTopP,
                topK: Int(paramTopK),
                minP: Float(paramMinP),
                repetitionPenalty: Float(paramRepeatPenalty),
                stream: true
            )

            do {
                let tokenStream = inferenceService.stream(request)
                let thinkingParser = ThinkingParser()
                for try await token in tokenStream {
                    let parsed = thinkingParser.feed(token.text)
                    if !parsed.text.isEmpty {
                        if parsed.type == .thinking {
                            messages[assistantIdx].thinkingContent += parsed.text
                        } else {
                            messages[assistantIdx].content += parsed.text
                        }
                    }
                }
                // Finalize handles edge cases (implicit open tag cleanup)
                let finalized = thinkingParser.finalize()
                if !finalized.thinking.isEmpty {
                    messages[assistantIdx].thinkingContent = finalized.thinking
                    messages[assistantIdx].content = finalized.response
                }
                if messages[assistantIdx].content.isEmpty && messages[assistantIdx].thinkingContent.isEmpty {
                    messages[assistantIdx].content = l10n.tr("chat.noResponse")
                }
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = l10n.tr("chat.error", error.localizedDescription)
                }
            }
            isLoading = false
        }
    }
}

private struct ChatMessageRow: Identifiable {
    let id = UUID()
    var content: String
    var thinkingContent: String = ""
    let isUser: Bool
}

// MARK: - Parameter Slider Component

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
        .frame(width: 140)
    }
}
