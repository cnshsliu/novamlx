import SwiftUI
import NovaMLXCore
import NovaMLXInference
import NovaMLXUtils

struct ChatPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService

    @State private var messages: [ChatMessageRow] = []
    @State private var inputText = ""
    @State private var selectedModel = ""
    @State private var isLoading = false

    var body: some View {
        VStack(spacing: 0) {
            infoBanner
            chatToolbar
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
            }
        }
        .onChange(of: appState.loadedModels) { _, newModels in
            if !newModels.contains(selectedModel), let first = newModels.first {
                selectedModel = first
            }
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

            Button("For daily use, Open Web Chat") {
                NSWorkspace.shared.open(URL(string: "http://127.0.0.1:\(String(appState.serverPort))/chat")!)
            }
            .buttonStyle(.bordered)
            .controlSize(.small)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(NovaTheme.Colors.accentDim)
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
            TextField("Type a message...", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .onSubmit { sendMessage() }

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

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !selectedModel.isEmpty else { return }

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
                temperature: 0.7,
                maxTokens: 2048,
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
