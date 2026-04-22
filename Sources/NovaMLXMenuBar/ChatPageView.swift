import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils
import NovaMLXEngine

struct ChatPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService

    @State private var messages: [ChatMessageRow] = []
    @State private var inputText = ""
    @State private var selectedModel = ""
    @State private var isLoading = false

    // History navigation
    @State private var sentHistory: [String] = []
    @State private var historyIndex: Int? = nil
    @State private var savedDraft: String? = nil
    @FocusState private var isInputFocused: Bool

    // Parameter controls
    @State private var showParams = false
    @State private var paramTemp: Double = 0.7
    @State private var paramMaxTokens: Double = 4096
    @State private var paramTopP: Double = 0.9
    @State private var paramTopK: Double = 0
    @State private var paramMinP: Double = 0
    @State private var paramRepeatPenalty: Double = 1.0

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
            Picker("Model", selection: $selectedModel) {
                if appState.loadedModels.isEmpty {
                    Text("No models loaded").tag("")
                }
                ForEach(appState.loadedModels, id: \.self) { model in
                    Text(model.components(separatedBy: "/").last ?? model).tag(model)
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
            .help("Toggle parameter settings")

            Button("Clear") {
                messages.removeAll()
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(12)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .top)
        .onAppear {
            if selectedModel.isEmpty, let first = appState.loadedModels.first {
                selectedModel = first
                loadDefaultsFromModel(first)
            }
        }
        .onChange(of: appState.loadedModels) { _, newModels in
            if !newModels.contains(selectedModel), let first = newModels.first {
                selectedModel = first
                loadDefaultsFromModel(first)
            }
        }
        .onChange(of: selectedModel) { _, newModel in
            if !newModel.isEmpty { loadDefaultsFromModel(newModel) }
        }
    }

    private var infoBanner: some View {
        HStack(spacing: 8) {
            Image(systemName: "info.circle")
                .foregroundColor(NovaTheme.Colors.accent)
                .font(.system(size: 12))

            Text("Raw model output viewer — not designed for daily chat use.")
                .font(.system(size: 11))
                .foregroundColor(NovaTheme.Colors.textSecondary)

            Spacer()

            Button("Web Chat") {
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
                ParamSlider(label: "Temperature", value: $paramTemp, min: 0, max: 2, step: 0.05)
                ParamSlider(label: "Top P", value: $paramTopP, min: 0, max: 1, step: 0.05)
                ParamSlider(label: "Top K", value: $paramTopK, min: 0, max: 200, step: 1)
                ParamSlider(label: "Min P", value: $paramMinP, min: 0, max: 1, step: 0.05)
            }
            HStack(spacing: 16) {
                ParamSlider(label: "Max Tokens", value: $paramMaxTokens, min: 64, max: 32768, step: 64)
                ParamSlider(label: "Repeat Penalty", value: $paramRepeatPenalty, min: 1.0, max: 2.0, step: 0.05)

                Spacer()

                Button("Reset to Default") {
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
                            Text("Start a conversation")
                                .font(.title3)
                                .foregroundColor(.secondary)
                            Text("Select a model above and type a message below.")
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
            VStack(alignment: msg.isUser ? .trailing : .leading, spacing: 4) {
                Text(msg.isUser ? "You" : "Assistant")
                    .font(.caption2)
                    .foregroundColor(.secondary)

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
            if !msg.isUser { Spacer(minLength: 60) }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 12) {
            HistoryTextField(
                placeholder: "Type a message...",
                text: $inputText,
                isFocused: _isInputFocused,
                onNavigateUp: { navigateHistory(.up) },
                onNavigateDown: { navigateHistory(.down) },
                onSubmit: { sendMessage() }
            )

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
                for try await token in tokenStream {
                    messages[assistantIdx].content += token.text
                }
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = "(no response)"
                }
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = "Error: \(error.localizedDescription)"
                }
            }
            isLoading = false
        }
    }
}

private struct ChatMessageRow: Identifiable {
    let id = UUID()
    var content: String
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

// MARK: - Multiline TextField with arrow key history navigation

private struct HistoryTextField: NSViewRepresentable {
    let placeholder: String
    @Binding var text: String
    @FocusState var isFocused: Bool
    let onNavigateUp: () -> Void
    let onNavigateDown: () -> Void
    let onSubmit: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        guard let textView = scrollView.documentView as? NSTextView else {
            return scrollView
        }

        textView.delegate = context.coordinator
        textView.isRichText = false
        textView.allowsUndo = true
        textView.drawsBackground = false
        textView.isEditable = true
        textView.isSelectable = true
        textView.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        textView.insertionPointColor = NSColor.controlTextColor
        textView.textContainerInset = NSSize(width: 2, height: 4)

        // Placeholder via NSTextFieldCell overlay
        context.coordinator.placeholderText = placeholder
        if textView.string.isEmpty {
            textView.textColor = NSColor.placeholderTextColor
            textView.string = placeholder
            context.coordinator.showingPlaceholder = true
        }

        // Min/max height for 1-5 lines
        textView.minSize = NSSize(width: 0, height: 28)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.widthTracksTextView = true
        textView.sizeToFit()

        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = false
        scrollView.hasHorizontalScroller = false

        context.coordinator.textView = textView
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        guard let textView = scrollView.documentView as? NSTextView else { return }
        if context.coordinator.showingPlaceholder {
            // Don't sync while showing placeholder
            if !text.isEmpty {
                context.coordinator.showingPlaceholder = false
                textView.textColor = NSColor.controlTextColor
                textView.string = text
            }
        } else if textView.string != text {
            textView.string = text
        }
    }

    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: HistoryTextField
        weak var textView: NSTextView?
        var placeholderText: String = ""
        var showingPlaceholder: Bool = false

        init(_ parent: HistoryTextField) {
            self.parent = parent
        }

        func textDidBeginEditing(_ notification: Notification) {
            guard let tv = textView, showingPlaceholder else { return }
            showingPlaceholder = false
            tv.textColor = NSColor.controlTextColor
            tv.string = ""
        }

        func textDidChange(_ notification: Notification) {
            guard let tv = textView else { return }
            if showingPlaceholder {
                return
            }
            parent.text = tv.string
            // Show placeholder if emptied
            if tv.string.isEmpty {
                showingPlaceholder = true
                tv.textColor = NSColor.placeholderTextColor
                tv.string = placeholderText
            }
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            // Don't navigate history while showing placeholder
            guard !showingPlaceholder else { return false }

            // Up arrow — navigate history when cursor is on first line
            if commandSelector == #selector(NSResponder.moveUp(_:)) {
                let cursor = textView.selectedRanges.first?.rangeValue.location ?? 0
                let prefix = (textView.string as NSString).substring(to: min(cursor, textView.string.count))
                if !prefix.contains("\n") {
                    parent.onNavigateUp()
                    return true
                }
            }
            // Down arrow — navigate history when cursor is on last line
            if commandSelector == #selector(NSResponder.moveDown(_:)) {
                let cursor = textView.selectedRanges.first?.rangeValue.location ?? 0
                let suffix = (textView.string as NSString).substring(from: min(cursor, textView.string.count))
                if !suffix.contains("\n") {
                    parent.onNavigateDown()
                    return true
                }
            }
            // Enter — submit; Shift+Enter inserts newline
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                if NSEvent.modifierFlags.contains(.shift) {
                    return false
                }
                parent.onSubmit()
                return true
            }
            return false
        }
    }
}
