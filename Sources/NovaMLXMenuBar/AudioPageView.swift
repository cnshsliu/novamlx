import SwiftUI
import AVFoundation
import CoreMedia
import NovaMLXCore
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXEngine

private enum AudioTab: String, CaseIterable {
    case asr = "ASR"
    case tts = "TTS"
}

struct AudioPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager

    @EnvironmentObject var l10n: L10n
    @State private var selectedTab: AudioTab = .asr

    // MARK: - ASR State
    @State private var selectedASRModel: String = ""
    @State private var isRecording = false
    @State private var transcriptionText = ""
    @State private var isTranscribing = false
    @State private var isASRModelLoading = false
    @State private var loadingASRModelName: String?
    @State private var asrError: String?
    @State private var asrCodeTab: CodeTab = .curl
    @State private var ttsCodeTab: CodeTab = .curl
    @State private var uploadedFileName: String?

    // Recording
    @State private var audioRecorder: AVAudioRecorder?
    @State private var recordingURL: URL?

    // MARK: - TTS State
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

    // Voice cloning
    @State private var voiceProfiles: [VoiceProfile] = []
    @State private var selectedProfileId: UUID? = nil
    @State private var showCloneSheet = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            tabPicker
            Divider()
            contentView
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            autoSelectModels()
            loadMacOSVoices()
            loadVoiceProfiles()
        }
        .onChange(of: appState.loadedModels) { _, _ in
            autoSelectModels()
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack {
            Text(l10n.tr("audio.title"))
                .font(.title2.bold())
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(Rectangle().fill(NovaTheme.Colors.cardBorder).frame(height: 1), alignment: .bottom)
    }

    // MARK: - Tab Picker

    private var tabPicker: some View {
        HStack(spacing: 0) {
            ForEach(AudioTab.allCases, id: \.self) { tab in
                Button {
                    selectedTab = tab
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: tab == .asr ? "mic.fill" : "speaker.wave.2.fill")
                            .font(.system(size: 12))
                        Text(tab.rawValue)
                            .font(.system(size: 13, weight: selectedTab == tab ? .semibold : .regular))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(selectedTab == tab ? NovaTheme.Colors.accentDim : Color.clear)
                    .foregroundColor(selectedTab == tab ? NovaTheme.Colors.accent : .secondary)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 4)
    }

    // MARK: - Content

    @ViewBuilder
    private var contentView: some View {
        switch selectedTab {
        case .asr:
            asrView
        case .tts:
            ttsView
        }
    }

    // MARK: - ASR View

    private var asrView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                asrModelSection

                // Input controls
                VStack(spacing: 12) {
                    HStack(spacing: 12) {
                        Button {
                            toggleRecording()
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: isRecording ? "stop.circle.fill" : "mic.circle.fill")
                                    .font(.system(size: 20))
                                Text(isRecording ? l10n.tr("audio.asr.stopRecording") : l10n.tr("audio.asr.startRecording"))
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(isRecording ? Color.red.opacity(0.2) : NovaTheme.Colors.accent.opacity(0.15))
                            .foregroundColor(isRecording ? .red : NovaTheme.Colors.accent)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .buttonStyle(.plain)
                        .disabled(selectedASRModel.isEmpty || isTranscribing)

                        Button {
                            uploadAudioFile()
                        } label: {
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
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .disabled(selectedASRModel.isEmpty || isTranscribing)
                    }

                    if let name = uploadedFileName {
                        HStack(spacing: 4) {
                            Image(systemName: "doc.fill")
                                .font(.caption)
                            Text(name)
                                .font(.caption)
                                .lineLimit(1)
                            Spacer()
                            if recordingURL != nil {
                                Button {
                                    downloadRecording()
                                } label: {
                                    Image(systemName: "arrow.down.circle.fill")
                                        .font(.caption)
                                        .foregroundColor(NovaTheme.Colors.accent)
                                }
                                .buttonStyle(.plain)
                            }
                            Button {
                                uploadedFileName = nil
                                recordingURL = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                        .foregroundColor(.secondary)
                    }
                }

                if isTranscribing {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text(l10n.tr("audio.asr.transcribing"))
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                }

                if let error = asrError {
                    Text(error)
                        .font(.system(size: 12))
                        .foregroundColor(.red)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                if !transcriptionText.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text(l10n.tr("audio.asr.result"))
                                .font(.system(size: 13, weight: .semibold))
                            Spacer()
                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(transcriptionText, forType: .string)
                            } label: {
                                Image(systemName: "doc.on.doc")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }
                            .buttonStyle(.plain)
                        }

                        TextEditor(text: .constant(transcriptionText))
                            .font(.system(size: 13))
                            .frame(minHeight: 120)
                            .padding(8)
                            .background(Color(nsColor: .textBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
                            )
                    }
                }

                // API Examples
                if !selectedASRModel.isEmpty {
                    asrApiExamples
                }

                Spacer(minLength: 0)
            }
            .padding(20)
        }
    }

    // MARK: - TTS View

    private var ttsView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Engine picker
                VStack(alignment: .leading, spacing: 6) {
                    Text(l10n.tr("audio.tts.model"))
                        .font(.system(size: 13, weight: .semibold))

                    HStack(spacing: 0) {
                        ForEach(TTSEngine.allCases, id: \.self) { engine in
                            Button {
                                ttsEngine = engine
                            } label: {
                                HStack(spacing: 4) {
                                    Image(systemName: engine == .neural ? "brain" : "speaker.wave.2.fill")
                                        .font(.system(size: 11))
                                    Text(engine.rawValue)
                                        .font(.system(size: 12, weight: ttsEngine == engine ? .semibold : .regular))
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 6)
                                .background(ttsEngine == engine ? NovaTheme.Colors.accent.opacity(0.15) : Color.clear)
                                .foregroundColor(ttsEngine == engine ? NovaTheme.Colors.accent : .secondary)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(3)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                    if ttsEngine == .neural {
                        if let ttsName = loadedTTSModelName {
                            HStack(spacing: 6) {
                                Image(systemName: "checkmark.circle.fill")
                                    .font(.system(size: 11))
                                    .foregroundColor(.green)
                                Text(ttsName)
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }
                            .padding(6)
                        } else {
                            HStack(spacing: 6) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .font(.system(size: 11))
                                    .foregroundColor(.orange)
                                Text(l10n.tr("audio.tts.noModel"))
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }
                            .padding(6)
                        }

                        // Voice profile picker
                        voiceProfileSection
                    } else {
                        if macOSVoices.isEmpty {
                            Text("Loading voices...")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        } else {
                            Picker("Voice", selection: $selectedSystemVoice) {
                                ForEach(macOSVoices) { voice in
                                    Text("\(voice.name) (\(voice.locale))")
                                        .tag(voice.name)
                                }
                            }
                            .pickerStyle(.menu)
                            .frame(maxWidth: .infinity)
                        }
                    }
                }

                // Text input
                VStack(alignment: .leading, spacing: 6) {
                    Text(l10n.tr("audio.tts.input"))
                        .font(.system(size: 13, weight: .semibold))

                    TextEditor(text: $ttsText)
                        .font(.system(size: 13))
                        .frame(minHeight: 120)
                        .padding(4)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
                        )
                }

                // Synthesize + Play
                HStack(spacing: 12) {
                    Button {
                        synthesizeSpeech()
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: "speaker.wave.2.fill")
                                .font(.system(size: 14))
                            Text(l10n.tr("audio.tts.synthesize"))
                                .font(.system(size: 13, weight: .medium))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(NovaTheme.Colors.accent)
                        .foregroundColor(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                    .buttonStyle(.plain)
                    .disabled(ttsText.isEmpty || isSynthesizing || (ttsEngine == .neural && loadedTTSModelName == nil))

                    if synthesizedAudioURL != nil {
                        Button {
                            togglePlayback()
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                                    .font(.system(size: 14))
                                Text(isPlaying ? l10n.tr("audio.tts.pause") : l10n.tr("audio.tts.play"))
                                    .font(.system(size: 13, weight: .medium))
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                            .background(NovaTheme.Colors.accent.opacity(0.15))
                            .foregroundColor(NovaTheme.Colors.accent)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .buttonStyle(.plain)
                    }
                }

                if isSynthesizing {
                    HStack(spacing: 8) {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text(l10n.tr("audio.tts.synthesizing"))
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                }

                if let error = ttsError {
                    Text(error)
                        .font(.system(size: 12))
                        .foregroundColor(.red)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                if let success = ttsSuccess {
                    Text(success)
                        .font(.system(size: 12))
                        .foregroundColor(.green)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.green.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                // API Examples
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
                Image(systemName: "person.wave.2")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                Text(l10n.tr("audio.tts.voiceProfile"))
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                Button {
                    showCloneSheet = true
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "plus.circle.fill")
                            .font(.system(size: 11))
                        Text(l10n.tr("audio.tts.cloneVoice"))
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundColor(NovaTheme.Colors.accent)
                }
                .buttonStyle(.plain)
                .sheet(isPresented: $showCloneSheet) {
                    VoiceCloneSheet(l10n: l10n) {
                        loadVoiceProfiles()
                        // Auto-select the newest profile
                        if let newest = voiceProfiles.first {
                            selectedProfileId = newest.id
                        }
                    }
                }
            }

            if voiceProfiles.isEmpty {
                Text(l10n.tr("audio.tts.noVoiceProfiles"))
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(voiceProfiles) { profile in
                            VoiceProfileCard(
                                profile: profile,
                                isSelected: selectedProfileId == profile.id,
                                l10n: l10n
                            ) {
                                selectedProfileId = profile.id
                            } onDelete: {
                                VoiceProfileManager.shared.deleteProfile(profile.id)
                                if selectedProfileId == profile.id {
                                    selectedProfileId = voiceProfiles.first(where: { $0.id != profile.id })?.id
                                }
                                loadVoiceProfiles()
                            }
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
        .padding(.horizontal, 6)
    }

    // MARK: - Computed Properties

    private var loadedTTSModelName: String? {
        let ttsModels = appState.loadedModels.filter { modelId in
            guard let record = modelManager.getRecord(modelId) else { return false }
            return record.family == .dotsTts || record.family == .qwen3Tts
        }
        return ttsModels.first.map { shortModelName($0) }
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

    // MARK: - API Example Helpers

    private var realApiKey: String {
        appState.apiKey ?? "YOUR_API_KEY"
    }

    private func codeBlock(_ code: String) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(nsColor: .secondaryLabelColor))
                    .textSelection(.enabled)
                    .padding(8)
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(code, forType: .string)
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .padding(6)
        }
        .background(Color(nsColor: .textBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
        )
    }

    private enum CodeTab: String, CaseIterable {
        case curl, python, node
    }

    private func langTabs(_ selected: Binding<CodeTab>) -> some View {
        HStack(spacing: 0) {
            ForEach(CodeTab.allCases, id: \.self) { tab in
                Button {
                    selected.wrappedValue = tab
                } label: {
                    Text(tab.rawValue.uppercased())
                        .font(.system(size: 10, weight: selected.wrappedValue == tab ? .semibold : .regular))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                        .background(selected.wrappedValue == tab ? NovaTheme.Colors.accent.opacity(0.15) : Color.clear)
                        .foregroundColor(selected.wrappedValue == tab ? NovaTheme.Colors.accent : .secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                }
                .buttonStyle(.plain)
            }
        }
        .padding(2)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    // MARK: - ASR API Examples

    private var asrApiExamples: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "terminal")
                    .font(.system(size: 11))
                Text("API Examples")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                langTabs($asrCodeTab)
            }
            .foregroundColor(.secondary)

            let model = selectedASRModel
            let key = realApiKey

            switch asrCodeTab {
            case .curl:
                codeBlock("curl -X POST http://localhost:6590/v1/audio/transcriptions \\\n  -H \"Authorization: Bearer \(key)\" \\\n  -F \"file=@recording.wav\" \\\n  -F \"model=\(model)\"")
            case .python:
                codeBlock("import requests\n\nresp = requests.post(\n    \"http://localhost:6590/v1/audio/transcriptions\",\n    headers={\"Authorization\": \"Bearer \(key)\"},\n    files={\"file\": open(\"recording.wav\", \"rb\")},\n    data={\"model\": \"\(model)\"}\n)\nprint(resp.json())")
            case .node:
                codeBlock("const FormData = require(\"form-data\");\nconst fs = require(\"fs\");\nconst form = new FormData();\nform.append(\"file\", fs.createReadStream(\"recording.wav\"));\nform.append(\"model\", \"\(model)\");\nconst resp = await fetch(\"http://localhost:6590/v1/audio/transcriptions\", {\n  method: \"POST\",\n  headers: { Authorization: \"Bearer \(key)\", ...form.getHeaders() },\n  body: form\n});\nconsole.log(await resp.json());")
            }
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - TTS API Examples

    private var ttsApiExamples: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "terminal")
                    .font(.system(size: 11))
                Text("API Examples")
                    .font(.system(size: 12, weight: .semibold))
                Spacer()
                langTabs($ttsCodeTab)
            }
            .foregroundColor(.secondary)

            let key = realApiKey
            let profileName = voiceProfiles.first(where: { $0.id == selectedProfileId })?.name ?? "MyVoice"

            switch ttsCodeTab {
            case .curl:
                VStack(alignment: .leading, spacing: 4) {
                    Text("Voice Cloning TTS")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                    codeBlock("curl -X POST http://localhost:6590/v1/audio/speech \\\n  -H \"Authorization: Bearer \(key)\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"model\":\"tts\",\"input\":\"Hello world\",\"voice\":\"\(profileName)\"}' \\\n  --output speech.wav")

                    Text("System TTS (no neural model)")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                        .padding(.top, 4)
                    codeBlock("curl -X POST http://localhost:6590/v1/audio/speech \\\n  -H \"Authorization: Bearer \(key)\" \\\n  -H \"Content-Type: application/json\" \\\n  -d '{\"model\":\"tts\",\"input\":\"Hello world\",\"voice\":\"Tingting\"}' \\\n  --output speech.wav")
                }
            case .python:
                VStack(alignment: .leading, spacing: 4) {
                    Text("Voice Cloning TTS")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                    codeBlock("import requests\n\nresp = requests.post(\n    \"http://localhost:6590/v1/audio/speech\",\n    headers={\"Authorization\": \"Bearer \(key)\"},\n    json={\"model\": \"tts\", \"input\": \"Hello world\", \"voice\": \"\(profileName)\"}\n)\nwith open(\"speech.wav\", \"wb\") as f:\n    f.write(resp.content)")

                    Text("System TTS (no neural model)")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                        .padding(.top, 4)
                    codeBlock("import requests\n\nresp = requests.post(\n    \"http://localhost:6590/v1/audio/speech\",\n    headers={\"Authorization\": \"Bearer \(key)\"},\n    json={\"model\": \"tts\", \"input\": \"Hello world\", \"voice\": \"Tingting\"}\n)\nwith open(\"speech.wav\", \"wb\") as f:\n    f.write(resp.content)")
                }
            case .node:
                VStack(alignment: .leading, spacing: 4) {
                    Text("Voice Cloning TTS")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                    codeBlock("const resp = await fetch(\"http://localhost:6590/v1/audio/speech\", {\n  method: \"POST\",\n  headers: { Authorization: \"Bearer \(key)\", \"Content-Type\": \"application/json\" },\n  body: JSON.stringify({ model: \"tts\", input: \"Hello world\", voice: \"\(profileName)\" })\n});\nconst buf = Buffer.from(await resp.arrayBuffer());\nrequire(\"fs\").writeFileSync(\"speech.wav\", buf);")

                    Text("System TTS (no neural model)")
                        .font(.system(size: 9, weight: .medium))
                        .foregroundColor(.secondary)
                        .padding(.top, 4)
                    codeBlock("const resp = await fetch(\"http://localhost:6590/v1/audio/speech\", {\n  method: \"POST\",\n  headers: { Authorization: \"Bearer \(key)\", \"Content-Type\": \"application/json\" },\n  body: JSON.stringify({ model: \"tts\", input: \"Hello world\", voice: \"Tingting\" })\n});\nconst buf = Buffer.from(await resp.arrayBuffer());\nrequire(\"fs\").writeFileSync(\"speech.wav\", buf);")
                }
            }
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - ASR Model Section

    private var asrModelSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(l10n.tr("audio.asr.model"))
                .font(.system(size: 13, weight: .semibold))

            let loaded = loadedASRModels
            if !loaded.isEmpty {
                Picker("", selection: $selectedASRModel) {
                    Text(l10n.tr("audio.selectModel")).tag("")
                    ForEach(loaded, id: \.self) { model in
                        Text(shortModelName(model)).tag(model)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
            } else {
                let downloaded = downloadedASRModels
                if downloaded.isEmpty {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text(l10n.tr("audio.asr.noModel"))
                            .font(.system(size: 12))
                            .foregroundColor(.secondary)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.08))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                } else {
                    ForEach(downloaded, id: \.id) { item in
                        HStack(spacing: 10) {
                            Image(systemName: "waveform")
                                .font(.system(size: 14))
                                .foregroundColor(NovaTheme.Colors.accent)
                            Text(shortModelName(item.id))
                                .font(.system(size: 12))
                                .lineLimit(1)
                            Spacer()
                            if item.isLoaded {
                                Text("✓")
                                    .font(.system(size: 12, weight: .bold))
                                    .foregroundColor(.green)
                            } else if isASRModelLoading {
                                HStack(spacing: 6) {
                                    ProgressView()
                                        .scaleEffect(0.7)
                                    if let name = loadingASRModelName {
                                        Text(name)
                                            .font(.system(size: 10))
                                            .foregroundColor(NovaTheme.Colors.accent)
                                            .lineLimit(1)
                                    }
                                }
                            } else {
                                Button {
                                    loadASRModel(item.id)
                                } label: {
                                    Text(l10n.tr("audio.asr.load"))
                                        .font(.system(size: 11, weight: .medium))
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 4)
                                        .background(NovaTheme.Colors.accent)
                                        .foregroundColor(.white)
                                        .clipShape(RoundedRectangle(cornerRadius: 4))
                                }
                                .buttonStyle(.plain)
                            }
                        }
                        .padding(8)
                        .background(Color(nsColor: .controlBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                }
            }

            if let error = asrError, loaded.isEmpty {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundColor(.red)
            }
        }
    }

    // MARK: - Helpers

    private func shortModelName(_ id: String) -> String {
        if let slash = id.lastIndex(of: "/") {
            return String(id[id.index(after: slash)...])
        }
        return id
    }

    private func autoSelectModels() {
        let asrModels = loadedASRModels
        if asrModels.count == 1, selectedASRModel.isEmpty || !asrModels.contains(selectedASRModel) {
            selectedASRModel = asrModels[0]
        } else if !asrModels.contains(selectedASRModel) {
            selectedASRModel = asrModels.first ?? ""
        }
    }

    private func loadMacOSVoices() {
        guard macOSVoices.isEmpty else { return }
        DispatchQueue.global(qos: .userInitiated).async {
            let voices = TTSService.listMacOSVoices()
            DispatchQueue.main.async {
                macOSVoices = voices
            }
        }
    }

    private func loadVoiceProfiles() {
        voiceProfiles = VoiceProfileManager.shared.listProfiles()
    }

    // MARK: - ASR Actions

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
                _ = try await inferenceService.transcriptionService.loadModel(
                    from: record.localURL, config: config
                )
                selectedASRModel = modelId
                await MainActor.run {
                    isASRModelLoading = false
                    loadingASRModelName = nil
                }
            } catch {
                await MainActor.run {
                    asrError = error.localizedDescription
                    isASRModelLoading = false
                    loadingASRModelName = nil
                }
            }
        }
    }

    private func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            // Check microphone permission before recording
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                startRecording()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async {
                        if granted {
                            self.startRecording()
                        } else {
                            self.asrError = "Microphone access denied. Please allow in System Settings > Privacy & Security > Microphone."
                        }
                    }
                }
            case .denied, .restricted:
                asrError = "Microphone access denied. Please allow in System Settings > Privacy & Security > Microphone."
            @unknown default:
                startRecording()
            }
        }
    }

    private func startRecording() {
        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("novamlx_recording_\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 48000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]

        do {
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder.record()
            audioRecorder = recorder
            recordingURL = url
            isRecording = true
            uploadedFileName = nil
            asrError = nil
        } catch {
            asrError = "Recording failed: \(error.localizedDescription)"
        }
    }

    private func stopRecording() {
        audioRecorder?.stop()
        isRecording = false
        uploadedFileName = "recording.wav"

        guard let url = recordingURL else { return }
        transcribeAudio(url: url)
    }

    private func uploadAudioFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false

        guard panel.runModal() == .OK, let url = panel.url else { return }
        uploadedFileName = url.lastPathComponent
        recordingURL = url
        transcribeAudio(url: url)
    }

    private func downloadRecording() {
        guard let srcURL = recordingURL else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "recording.wav"
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        try? FileManager.default.removeItem(at: dest)
        do {
            try FileManager.default.copyItem(at: srcURL, to: dest)
        } catch {
            asrError = "Download failed: \(error.localizedDescription)"
        }
    }

    private func transcribeAudio(url: URL) {
        guard !selectedASRModel.isEmpty else {
            asrError = l10n.tr("audio.asr.noModel")
            return
        }

        let modelId = selectedASRModel
        isTranscribing = true
        transcriptionText = ""
        asrError = nil

        Task {
            do {
                let audioData = try Data(contentsOf: url)
                let result = try await inferenceService.transcriptionService.transcribe(
                    modelId: modelId,
                    audioData: audioData,
                    language: nil
                )
                await MainActor.run {
                    transcriptionText = result.text
                    isTranscribing = false
                }
            } catch {
                await MainActor.run {
                    asrError = error.localizedDescription
                    isTranscribing = false
                }
            }
        }
    }

    // MARK: - TTS Actions

    private func synthesizeSpeech() {
        guard !ttsText.isEmpty else { return }

        isSynthesizing = true
        ttsError = nil
        ttsSuccess = nil

        let profile = voiceProfiles.first { $0.id == selectedProfileId }

        Task {
            do {
                let audioData = try await inferenceService.ttsService.synthesize(
                    text: ttsText,
                    voice: selectedSystemVoice,
                    engine: ttsEngine,
                    voiceProfile: profile
                )

                let tempDir = FileManager.default.temporaryDirectory
                let ext = ttsEngine == .system ? "aiff" : "wav"
                let url = tempDir.appendingPathComponent("novamlx_tts_\(UUID().uuidString).\(ext)")
                try audioData.write(to: url)

                await MainActor.run {
                    synthesizedAudioURL = url
                    isSynthesizing = false
                }
            } catch {
                await MainActor.run {
                    ttsError = error.localizedDescription
                    isSynthesizing = false
                }
            }
        }
    }

    private func togglePlayback() {
        if isPlaying {
            audioPlayer?.stop()
            isPlaying = false
            return
        }

        guard let url = synthesizedAudioURL else { return }

        do {
            let player = try AVAudioPlayer(contentsOf: url)
            player.delegate = nil
            audioPlayer = player
            player.play()
            isPlaying = true

            DispatchQueue.main.asyncAfter(deadline: .now() + player.duration) {
                isPlaying = false
            }
        } catch {
            ttsError = "Playback failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Voice Profile Card

struct VoiceProfileCard: View {
    let profile: VoiceProfile
    let isSelected: Bool
    @ObservedObject var l10n: L10n
    let onSelect: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Button(action: onSelect) {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.system(size: 10))
                        .foregroundColor(isSelected ? NovaTheme.Colors.accent : .secondary)
                    Text(profile.name)
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1)
                    Spacer()
                    Button {
                        onDelete()
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(size: 9))
                            .foregroundColor(.red.opacity(0.6))
                    }
                    .buttonStyle(.plain)
                }

                Text(profile.refTranscript.prefix(40) + "...")
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
            .padding(8)
            .background(isSelected ? NovaTheme.Colors.accent.opacity(0.1) : Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(isSelected ? NovaTheme.Colors.accent.opacity(0.3) : NovaTheme.Colors.cardBorder, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Voice Clone Sheet

struct VoiceCloneSheet: View {
    @ObservedObject var l10n: L10n
    let onSaved: () -> Void

    @Environment(\.dismiss) private var dismiss

    @State private var profileName = ""
    @State private var isRecording = false
    @State private var cloneRecorder: AVAudioRecorder?
    @State private var cloneRecordingURL: URL?
    @State private var isPreviewPlaying = false
    @State private var previewPlayer: AVAudioPlayer?
    @State private var cloneError: String?
    @State private var recordingDuration: TimeInterval = 0
    @State private var isSaving = false

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text(l10n.tr("audio.tts.cloneVoice"))
                    .font(.title3.bold())
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }

            // Instructions
            VStack(alignment: .leading, spacing: 8) {
                Text(l10n.tr("audio.tts.cloneInstructions"))
                    .font(.system(size: 12, weight: .medium))

                Text(l10n.tr("audio.tts.cloneText"))
                    .font(.system(size: 13))
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(nsColor: .textBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5)
                    )
            }

            // Record button
            HStack(spacing: 12) {
                Button {
                    toggleCloneRecording()
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: isRecording ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.system(size: 18))
                        Text(isRecording ? l10n.tr("audio.tts.cloneStop") : l10n.tr("audio.tts.cloneRecord"))
                            .font(.system(size: 13, weight: .medium))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(isRecording ? Color.red.opacity(0.2) : NovaTheme.Colors.accent.opacity(0.15))
                    .foregroundColor(isRecording ? .red : NovaTheme.Colors.accent)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)

                if cloneRecordingURL != nil && !isRecording {
                    Button {
                        togglePreview()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: isPreviewPlaying ? "pause.fill" : "play.fill")
                                .font(.system(size: 14))
                            Text(l10n.tr("audio.tts.clonePreview"))
                                .font(.system(size: 12, weight: .medium))
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 10)
                        .background(NovaTheme.Colors.cardBackground)
                        .foregroundColor(.secondary)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(NovaTheme.Colors.cardBorder, lineWidth: 1)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }

            if isRecording {
                Text("Recording... speak naturally")
                    .font(.system(size: 11))
                    .foregroundColor(.red)
            }

            if let error = cloneError {
                Text(error)
                    .font(.system(size: 11))
                    .foregroundColor(.red)
            }

            // Profile name + save
            HStack(spacing: 12) {
                TextField(l10n.tr("audio.tts.cloneName"), text: $profileName)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 12))

                Button {
                    saveVoiceProfile()
                } label: {
                    HStack(spacing: 6) {
                        if isSaving {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        Text(l10n.tr("audio.tts.cloneSave"))
                            .font(.system(size: 13, weight: .medium))
                    }
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
                    .background(canSave ? NovaTheme.Colors.accent : Color.gray.opacity(0.3))
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
                .disabled(!canSave || isSaving)
            }
        }
        .padding(24)
        .frame(width: 480)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var canSave: Bool {
        cloneRecordingURL != nil && !isRecording && !profileName.trimmingCharacters(in: .whitespaces).isEmpty
    }

    private func toggleCloneRecording() {
        if isRecording {
            cloneRecorder?.stop()
            isRecording = false

            // Check duration
            if let url = cloneRecordingURL,
               let asset = try? AVAudioPlayer(contentsOf: url) {
                recordingDuration = asset.duration
                if asset.duration < 3.0 {
                    cloneError = l10n.tr("audio.tts.cloneDurationWarning")
                    cloneRecordingURL = nil
                    return
                }
            }
            cloneError = nil
        } else {
            // Check microphone permission first
            let micStatus = CGPreflightScreenCaptureAccess()
            debugLog("Checking mic permission...")

            // Use AVCaptureDevice to check/request mic access on macOS
            switch AVCaptureDevice.authorizationStatus(for: .audio) {
            case .authorized:
                debugLog("Mic permission: authorized")
                startCloneRecording()
            case .notDetermined:
                debugLog("Mic permission: not determined, requesting...")
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async {
                        debugLog("Mic permission request result: \(granted)")
                        if granted {
                            self.startCloneRecording()
                        } else {
                            self.cloneError = "Microphone access denied. Please allow in System Settings > Privacy & Security > Microphone."
                        }
                    }
                }
            case .denied, .restricted:
                debugLog("Mic permission: denied/restricted")
                cloneError = "Microphone access denied. Please allow in System Settings > Privacy & Security > Microphone."
            @unknown default:
                debugLog("Mic permission: unknown, trying anyway")
                startCloneRecording()
            }
        }
    }

    private func startCloneRecording() {
        let tempDir = FileManager.default.temporaryDirectory
        let url = tempDir.appendingPathComponent("voice_clone_\(UUID().uuidString).wav")

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 48000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]

        do {
            debugLog("Creating AVAudioRecorder at \(url.path)")
            let recorder = try AVAudioRecorder(url: url, settings: settings)
            debugLog("AVAudioRecorder created, meteringEnabled=\(recorder.isMeteringEnabled), calling record()")
            let started = recorder.record()
            debugLog("recorder.record() -> \(started)")
            cloneRecorder = recorder
            cloneRecordingURL = url
            isRecording = true
            cloneError = nil
        } catch {
            debugLog("AVAudioRecorder error: \(error)")
            cloneError = "Recording failed: \(error.localizedDescription)"
        }
    }

    private func debugLog(_ msg: String) {
        let logPath = "/tmp/novamlx_voice_clone_debug.log"
        let ts = DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .medium)
        let line = "[\(ts)] \(msg)\n"
        if let handle = FileHandle(forWritingAtPath: logPath) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            handle.closeFile()
        } else {
            try? line.write(toFile: logPath, atomically: true, encoding: .utf8)
        }
    }

    private func togglePreview() {
        debugLog("togglePreview called, isPreviewPlaying=\(isPreviewPlaying), url=\(cloneRecordingURL?.path ?? "nil")")

        if isPreviewPlaying {
            previewPlayer?.stop()
            isPreviewPlaying = false
            debugLog("Stopped existing playback")
            return
        }

        guard let url = cloneRecordingURL else {
            debugLog("ERROR: cloneRecordingURL is nil")
            return
        }

        let exists = FileManager.default.fileExists(atPath: url.path)
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? -1
        debugLog("File: \(url.path), exists=\(exists), size=\(size)")

        // Use NSSound directly — most reliable in macOS menu bar apps
        guard let sound = NSSound(contentsOf: url, byReference: false) else {
            debugLog("NSSound init FAILED, trying AVAudioPlayer")
            do {
                let player = try AVAudioPlayer(contentsOf: url)
                previewPlayer = player
                player.prepareToPlay()
                let ok = player.play()
                debugLog("AVAudioPlayer.play() -> \(ok), duration=\(player.duration)")
                if ok {
                    isPreviewPlaying = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + player.duration) {
                        isPreviewPlaying = false
                    }
                } else {
                    cloneError = "Playback failed (AVAudioPlayer returned false)"
                }
            } catch {
                debugLog("AVAudioPlayer error: \(error)")
                cloneError = "Playback failed: \(error.localizedDescription)"
            }
            return
        }

        debugLog("NSSound created OK, duration=\(sound.duration), calling play()")
        let played = sound.play()
        debugLog("NSSound.play() -> \(played)")
        isPreviewPlaying = true
        DispatchQueue.main.asyncAfter(deadline: .now() + sound.duration) {
            isPreviewPlaying = false
            debugLog("Playback finished, isPreviewPlaying reset")
        }
    }

    private func saveVoiceProfile() {
        guard let url = cloneRecordingURL else { return }

        isSaving = true
        cloneError = nil

        let refText = l10n.tr("audio.tts.cloneText")

        do {
            let profile = try VoiceProfileManager.shared.saveProfile(
                name: profileName,
                refAudioURL: url,
                refTranscript: refText
            )
            onSaved()
            dismiss()
        } catch {
            cloneError = error.localizedDescription
            isSaving = false
        }
    }
}
