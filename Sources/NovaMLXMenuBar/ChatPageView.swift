import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils
import NovaMLXEngine

private enum ChatDisplayMode: String, CaseIterable {
    case pretty = "Pretty"
    case rawJSON = "JSON"
    case rawStream = "Stream"
}

struct ChatPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService

    @EnvironmentObject var l10n: L10n
    @State private var messages: [ChatMessageRow] = []
    @State private var inputText = ""
    @State private var selectedModel = ""
    @State private var isLoading = false
    @State private var displayMode: ChatDisplayMode = .pretty

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

    private let quickPrompts = [
        "2+2=? Please explain step by step",
        "Write a haiku about coding",
        "Explain quantum computing in simple terms",
        "Translate 'Hello, how are you?' to Japanese",
        "Debug this: why does my code return nil?"
    ]

    var body: some View {
        VStack(spacing: 0) {
            infoBanner
            chatToolbar
            Divider()
            HStack(spacing: 0) {
                // Left: chat area
                VStack(spacing: 0) {
                    messageList
                    Divider()
                    if inputText.isEmpty && messages.isEmpty {
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
    }

    // MARK: - Toolbar

    private var chatToolbar: some View {
        HStack(spacing: 12) {
            Picker(l10n.tr("chat.model"), selection: $selectedModel) {
                if appState.loadedModels.isEmpty && appState.cloudModels.isEmpty {
                    Text(l10n.tr("chat.noModels")).tag("")
                }
                ForEach(appState.loadedModels, id: \.self) { model in
                    Text(shortModelLabel(model)).tag(model)
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
            .frame(width: 200)

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
                messages.removeAll()
                lastPayload = nil
                lastResponse = nil
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

    // MARK: - Banner

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

        switch displayMode {
        case .pretty:
            sendPretty(model: model, text: text, assistantIdx: assistantIdx)
        case .rawJSON:
            sendRawJSON(model: model, text: text, assistantIdx: assistantIdx)
        case .rawStream:
            sendRawStream(model: model, text: text, assistantIdx: assistantIdx)
        }
    }

    // MARK: Pretty mode (existing InferenceService streaming)

    private func sendPretty(model: String, text: String, assistantIdx: Int) {
        Task {
            guard inferenceService.isModelLoaded(model) else {
                messages[assistantIdx].content = l10n.tr("chat.error", "Model '\(model.components(separatedBy: "/").last ?? model)' is not loaded. Load it from the Models page first.")
                isLoading = false
                return
            }

            let payload: [String: Any] = [
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
            if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys]) {
                lastPayload = String(data: data, encoding: .utf8)
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
                for try await token in tokenStream {
                    messages[assistantIdx].content += token.text
                }
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = l10n.tr("chat.noResponse")
                }
            } catch {
                if messages[assistantIdx].content.isEmpty {
                    messages[assistantIdx].content = l10n.tr("chat.error", error.localizedDescription)
                }
            }
            lastResponse = messages[assistantIdx].content
            isLoading = false
        }
    }

    // MARK: Raw JSON mode (non-streaming HTTP)

    private func sendRawJSON(model: String, text: String, assistantIdx: Int) {
        Task {
            do {
                guard let url = URL(string: "http://127.0.0.1:\(appState.serverPort)/v1/chat/completions") else {
                    messages[assistantIdx].content = "Invalid URL"
                    isLoading = false
                    return
                }
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let key = appState.apiKey { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                let body: [String: Any] = [
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
                req.httpBody = try JSONSerialization.data(withJSONObject: body)
                if let prettyBody = try? JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]) {
                    lastPayload = String(data: prettyBody, encoding: .utf8)
                }
                let (data, _) = try await URLSession.shared.data(for: req)
                let json = try JSONSerialization.jsonObject(with: data)
                let pretty = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
                messages[assistantIdx].content = String(data: pretty, encoding: .utf8) ?? "Invalid response"
            } catch {
                messages[assistantIdx].content = "Error: \(error.localizedDescription)"
            }
            lastResponse = messages[assistantIdx].content
            isLoading = false
        }
    }

    // MARK: Raw Stream mode (SSE)

    private func sendRawStream(model: String, text: String, assistantIdx: Int) {
        Task {
            do {
                guard let url = URL(string: "http://127.0.0.1:\(appState.serverPort)/v1/chat/completions") else {
                    messages[assistantIdx].content = "Invalid URL"
                    isLoading = false
                    return
                }
                var req = URLRequest(url: url)
                req.httpMethod = "POST"
                req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let key = appState.apiKey { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                let body: [String: Any] = [
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
                req.httpBody = try JSONSerialization.data(withJSONObject: body)
                if let prettyBody = try? JSONSerialization.data(withJSONObject: body, options: [.prettyPrinted, .sortedKeys]) {
                    lastPayload = String(data: prettyBody, encoding: .utf8)
                }
                let (bytes, _) = try await URLSession.shared.bytes(for: req)
                var accumulated = ""
                for try await line in bytes.lines {
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
                    messages[assistantIdx].content = "Error: \(error.localizedDescription)"
                }
            }
            lastResponse = messages[assistantIdx].content
            isLoading = false
        }
    }

    private func shortModelLabel(_ modelId: String) -> String {
        let shortName = modelId.components(separatedBy: "/").last ?? modelId
        let mType = inferenceService.engine.getContainer(for: modelId)?.config.modelType ?? .llm
        let tag = modelTypeShort(mType)
        return "[\(tag)] \(shortName)"
    }

    private func modelTypeShort(_ type: ModelType) -> String {
        switch type {
        case .llm: return "LLM"
        case .vlm: return "VLM"
        case .embedding: return "EMB"
        }
    }

    private func modelTypeColor(_ type: ModelType) -> Color {
        switch type {
        case .llm: return .blue.opacity(0.8)
        case .vlm: return .purple.opacity(0.8)
        case .embedding: return .green.opacity(0.8)
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
