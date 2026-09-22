import SwiftUI
import AVFoundation
import AppKit
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXModelManager
import NovaMLXUtils

private struct CloneCue: Identifiable {
    let id = UUID()
    let text: String
    let start: Double
    let end: Double
}

private struct CloneEngineRow: Identifiable, Hashable {
    let id: String
    let family: ModelFamily
    let title: String
    let downloaded: Bool
}

struct VoiceClonePageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let modelManager: ModelManager
    @EnvironmentObject var l10n: L10n

    @State private var profiles: [VoiceProfile] = []
    @State private var selectedId: UUID?

    @State private var profileName = ""
    @State private var transcript = ""
    @State private var isRecording = false
    @State private var recorder: AVAudioRecorder?
    @State private var recordingURL: URL?
    @State private var isSaving = false
    @State private var isTranscribing = false

    @State private var speakText = ""
    @State private var isGenerating = false
    @State private var outputURL: URL?
    @State private var cues: [CloneCue] = []
    @State private var timestampsAreWordLevel = false
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var recordingPlayer: AVAudioPlayer?
    @State private var isPlayingRecording = false
    @State private var errorText: String?
    @State private var isLoadingStack = false
    @State private var loadProgress: String?
    @State private var renamingId: UUID?
    @State private var renameDraft = ""
    @AppStorage("novamlx.voiceClone.ttsModelId") private var selectedTTSId = Qwen3TTSCloneService.defaultModelId
    @AppStorage("novamlx.voiceClone.scriptLang") private var scriptLangRaw = VoiceCloneLanguage.chinese.rawValue

    private var scriptLang: VoiceCloneLanguage {
        VoiceCloneLanguage(rawValue: scriptLangRaw) ?? .chinese
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            HSplitView {
                profileList
                    .frame(minWidth: 200, idealWidth: 240, maxWidth: 280)
                    .frame(maxHeight: .infinity)
                inspectorColumn
                    .frame(minWidth: 480)
                    .frame(maxHeight: .infinity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color(nsColor: .windowBackgroundColor))
        .onAppear {
            reloadProfiles()
            if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                transcript = scriptLang.defaultTranscript
            }
            syncInspector(from: selectedId)
        }
        .onChange(of: selectedId) { oldId, newId in
            if let oldId {
                VoiceProfileManager.shared.updateTranscript(oldId, refTranscript: transcript)
            }
            if renamingId != nil, renamingId != newId { commitRename() }
            syncInspector(from: newId)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(l10n.tr("app.voiceClone"))
                .font(.title2.bold())
            Text(l10n.tr("voiceClone.subtitle"))
                .font(.caption)
                .foregroundColor(.secondary)
            HStack(spacing: 12) {
                Picker(l10n.tr("voiceClone.engine"), selection: $selectedTTSId) {
                    ForEach(cloneEngineRows) { row in
                        Text(String(
                            format: l10n.tr(row.downloaded ? "voiceClone.engineOnDisk" : "voiceClone.engineWillDownload"),
                            row.title
                        )).tag(row.id)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 360, alignment: .leading)
                .disabled(isLoadingStack)
                Label(
                    ttsModelId == selectedTTSId
                        ? String(format: l10n.tr("voiceClone.ttsLoaded"), selectedTTSId)
                        : l10n.tr("voiceClone.ttsNone"),
                    systemImage: ttsModelId == selectedTTSId ? "checkmark.circle" : "exclamationmark.triangle"
                )
                .font(.caption)
                .foregroundColor(ttsModelId == selectedTTSId ? .secondary : .orange)
                if asrModelId != nil {
                    Text(l10n.tr("voiceClone.asrHint"))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Button {
                    loadRequiredModels()
                } label: {
                    if isLoadingStack {
                        ProgressView().controlSize(.small)
                    }
                    Text(l10n.tr("voiceClone.loadStack"))
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLoadingStack)
            }
            if let loadProgress {
                Text(loadProgress)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(16)
    }

    private var profileList: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(l10n.tr("voiceClone.profiles"))
                .font(.headline)
                .padding(.horizontal, 12)
                .padding(.top, 12)
                .padding(.bottom, 8)
            List(selection: $selectedId) {
                ForEach(profiles) { profile in
                    profileRow(profile)
                        .tag(profile.id)
                        .contextMenu {
                            Button(l10n.tr("voiceClone.rename")) { startRename(profile) }
                            Button(l10n.tr("audio.tts.cloneDelete"), role: .destructive) {
                                removeProfile(profile.id)
                            }
                        }
                }
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)
            .frame(maxHeight: .infinity)
            .overlay {
                if profiles.isEmpty {
                    Text(l10n.tr("voiceClone.noProfiles"))
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(16)
                        .allowsHitTesting(false)
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                profileListFooter
            }
        }
        .frame(maxHeight: .infinity)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    private func profileRow(_ profile: VoiceProfile) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            if renamingId == profile.id {
                TextField("", text: $renameDraft)
                    .textFieldStyle(.roundedBorder)
                    .font(.system(size: 13, weight: .medium))
                    .onSubmit { commitRename() }
                    .onExitCommand { renamingId = nil }
            } else {
                Text(profile.name)
                    .font(.system(size: 13, weight: .medium))
                    .lineLimit(1)
            }
            Text(profileSubtitle(profile))
                .font(.caption2)
                .foregroundColor(.secondary)
                .lineLimit(2)
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) { startRename(profile) }
    }

    private func profileSubtitle(_ profile: VoiceProfile) -> String {
        if !VoiceProfileManager.shared.hasReferenceAudio(for: profile) {
            return l10n.tr("voiceClone.notRecorded")
        }
        let t = profile.refTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.isEmpty { return l10n.tr("voiceClone.hasAudio") }
        return String(t.prefix(48)) + (t.count > 48 ? "…" : "")
    }

    /// macOS source-list footer: template + / − stay pinned; list scrolls above.
    private var profileListFooter: some View {
        VStack(spacing: 0) {
            Divider()
            HStack(spacing: 0) {
                Button(action: beginNewClone) {
                    Image(nsImage: NSImage(named: NSImage.addTemplateName) ?? NSImage())
                        .frame(width: 24, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .help(l10n.tr("voiceClone.addProfile"))
                .accessibilityLabel(l10n.tr("voiceClone.addProfile"))

                Button(action: removeSelectedProfile) {
                    Image(nsImage: NSImage(named: NSImage.removeTemplateName) ?? NSImage())
                        .frame(width: 24, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.borderless)
                .disabled(selectedId == nil)
                .help(l10n.tr("voiceClone.removeProfile"))
                .accessibilityLabel(l10n.tr("voiceClone.removeProfile"))

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 4)
            .frame(height: 24)
            .background(Color(nsColor: .controlBackgroundColor))
        }
    }

    private var inspectorColumn: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                clonePanel
                if selectedProfile.map({ VoiceProfileManager.shared.hasReferenceAudio(for: $0) }) == true {
                    Divider()
                    speakPanel
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var inspectorTitle: String {
        selectedProfile?.name ?? l10n.tr("voiceClone.pickVoice")
    }

    private var clonePanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if selectedProfile == nil {
                Text(l10n.tr("voiceClone.pickVoice"))
                    .font(.headline)
                Text(l10n.tr("voiceClone.noProfiles"))
                    .font(.caption)
                    .foregroundColor(.secondary)
                Button(l10n.tr("voiceClone.addProfile"), action: beginNewClone)
                    .buttonStyle(.borderedProminent)
            } else {
                Text(inspectorTitle)
                    .font(.headline)
                TextField(l10n.tr("audio.tts.cloneName"), text: $profileName)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { commitInspectorName() }
                Picker(l10n.tr("voiceClone.lang"), selection: $scriptLangRaw) {
                    Text(l10n.tr("voiceClone.langChinese")).tag(VoiceCloneLanguage.chinese.rawValue)
                    Text(l10n.tr("voiceClone.langEnglish")).tag(VoiceCloneLanguage.english.rawValue)
                }
                .pickerStyle(.segmented)
                .onChange(of: scriptLangRaw) { oldRaw, newRaw in
                    let old = VoiceCloneLanguage(rawValue: oldRaw) ?? .chinese
                    if VoiceCloneLanguage.shouldReplaceTranscript(transcript, switchingFrom: old) {
                        transcript = (VoiceCloneLanguage(rawValue: newRaw) ?? .chinese).defaultTranscript
                        persistTranscript()
                    }
                }
                Text(l10n.tr("voiceClone.script"))
                    .font(.caption)
                    .foregroundColor(.secondary)
                TextEditor(text: $transcript)
                    .font(.system(size: 13))
                    .frame(minHeight: 88)
                    .padding(4)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))
                    .onChange(of: transcript) { _, _ in persistTranscript() }

                HStack(spacing: 8) {
                    Button(action: toggleRecord) {
                        Label(
                            isRecording ? l10n.tr("audio.tts.cloneStop") : l10n.tr("audio.tts.cloneRecord"),
                            systemImage: isRecording ? "stop.circle.fill" : "mic.circle.fill"
                        )
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(isRecording ? .red : NovaTheme.Colors.accent)

                    Button(action: togglePlayRecording) {
                        Label(
                            isPlayingRecording
                                ? l10n.tr("voiceClone.stopPlayback")
                                : l10n.tr("voiceClone.playRecording"),
                            systemImage: isPlayingRecording ? "stop.fill" : "play.fill"
                        )
                    }
                    .buttonStyle(.bordered)
                    .disabled(isRecording || previewableRecordingURL == nil)

                    Button(l10n.tr("voiceClone.import"), action: importAudio)
                        .buttonStyle(.bordered)

                    if asrModelId != nil {
                        Button(l10n.tr("voiceClone.autoTranscribe"), action: autoTranscribe)
                            .buttonStyle(.bordered)
                            .disabled(isTranscribing || (
                                recordingURL == nil
                                && !(selectedProfile.map { VoiceProfileManager.shared.hasReferenceAudio(for: $0) } ?? false)
                            ))
                    }
                }

                if let profile = selectedProfile, VoiceProfileManager.shared.hasReferenceAudio(for: profile) {
                    Text(l10n.tr("voiceClone.hasAudio"))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Button {
                    saveProfile()
                } label: {
                    if isSaving { ProgressView().controlSize(.small) }
                    Text(recordingURL == nil
                         ? l10n.tr("voiceClone.saveChanges")
                         : l10n.tr("audio.tts.cloneSave"))
                }
                .buttonStyle(.borderedProminent)
                .disabled(!canSave || isSaving)
            }

            if let errorText {
                Text(errorText)
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
        .padding(16)
    }

    private var speakPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(l10n.tr("voiceClone.speakTitle"))
                .font(.headline)
            TextEditor(text: $speakText)
                .font(.system(size: 14))
                .frame(minHeight: 80)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(NovaTheme.Colors.cardBorder, lineWidth: 0.5))

            HStack {
                Button {
                    generate()
                } label: {
                    if isGenerating { ProgressView().controlSize(.small) }
                    Text(l10n.tr("voiceClone.generate"))
                }
                .buttonStyle(.borderedProminent)
                .disabled(isGenerating || speakText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                Button {
                    togglePlay()
                } label: {
                    Label(isPlaying ? "Stop" : l10n.tr("audio.tts.clonePreview"),
                          systemImage: isPlaying ? "stop.fill" : "play.fill")
                }
                .disabled(outputURL == nil)

                Button(l10n.tr("voiceClone.exportFfmpeg"), action: exportForFFmpeg)
                    .disabled(outputURL == nil || cues.isEmpty)

                Button(l10n.tr("voiceClone.copySrt"), action: copySRT)
                    .disabled(cues.isEmpty)
            }

            if !cues.isEmpty {
                Text(timestampsAreWordLevel
                     ? l10n.tr("voiceClone.timestampsWord")
                     : l10n.tr("voiceClone.timestampsEstimated"))
                    .font(.subheadline.weight(.semibold))
                List(cues) { cue in
                    HStack {
                        Text(String(format: "%6.2f – %6.2f", cue.start, cue.end))
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.secondary)
                        Text(cue.text)
                            .font(.system(size: 12))
                    }
                }
                .frame(minHeight: 120)
            }
        }
        .padding(16)
    }

    private var ttsModelId: String? {
        inferenceService.ttsService.listLoadedModels().first
    }

    private var asrModelId: String? {
        inferenceService.transcriptionService.listLoadedModels().first
    }

    private var selectedProfile: VoiceProfile? {
        profiles.first { $0.id == selectedId }
    }

    private var previewableRecordingURL: URL? {
        recordingURL ?? selectedProfile.flatMap { VoiceProfileManager.shared.refAudioURL(for: $0) }
    }

    private var canSave: Bool {
        selectedId != nil && !isRecording
            && !profileName.trimmingCharacters(in: .whitespaces).isEmpty
            && !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func reloadProfiles() {
        let keep = selectedId
        profiles = VoiceProfileManager.shared.listProfiles()
        if let keep, profiles.contains(where: { $0.id == keep }) {
            selectedId = keep
        } else if selectedId == nil {
            selectedId = profiles.first?.id
        } else if !profiles.contains(where: { $0.id == selectedId }) {
            selectedId = profiles.first?.id
        }
    }

    private func syncInspector(from id: UUID?) {
        guard let id, let profile = profiles.first(where: { $0.id == id }) else { return }
        profileName = profile.name
        if !isRecording {
            transcript = profile.refTranscript.isEmpty ? scriptLang.defaultTranscript : profile.refTranscript
            recordingURL = nil
            stopRecordingPlayback()
        }
        errorText = nil
    }

    private func beginNewClone() {
        if isRecording {
            recorder?.stop()
            isRecording = false
        }
        stopRecordingPlayback()
        let base = l10n.tr("voiceClone.untitled")
        let name = VoiceProfileManager.uniqueName(base: base, existing: profiles.map(\.name))
        do {
            let profile = try VoiceProfileManager.shared.createDraft(
                name: name,
                refTranscript: scriptLang.defaultTranscript
            )
            recordingURL = nil
            errorText = nil
            profileName = profile.name
            transcript = profile.refTranscript
            reloadProfiles()
            selectedId = profile.id
            startRename(profile)
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func startRename(_ profile: VoiceProfile) {
        selectedId = profile.id
        renameDraft = profile.name
        renamingId = profile.id
        profileName = profile.name
    }

    private func commitRename() {
        guard let id = renamingId else { return }
        let name = renameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        if !name.isEmpty {
            VoiceProfileManager.shared.renameProfile(id, newName: name)
            profileName = name
        }
        renamingId = nil
        reloadProfiles()
    }

    private func commitInspectorName() {
        guard let id = selectedId else { return }
        let name = profileName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        VoiceProfileManager.shared.renameProfile(id, newName: name)
        reloadProfiles()
    }

    private func persistTranscript() {
        guard let id = selectedId else { return }
        VoiceProfileManager.shared.updateTranscript(id, refTranscript: transcript)
    }

    private func removeSelectedProfile() {
        guard let id = selectedId else { return }
        removeProfile(id)
    }

    private func removeProfile(_ id: UUID) {
        if renamingId == id { renamingId = nil }
        let ids = profiles.map(\.id)
        let next = ids.first { $0 != id }
        VoiceProfileManager.shared.deleteProfile(id)
        selectedId = next
        reloadProfiles()
        if selectedId == id { selectedId = profiles.first?.id }
    }

    private func toggleRecord() {
        if isRecording {
            recorder?.stop()
            isRecording = false
            if let url = recordingURL, let player = try? AVAudioPlayer(contentsOf: url), player.duration < 3 {
                errorText = l10n.tr("audio.tts.cloneDurationWarning")
                recordingURL = nil
                return
            }
            errorText = nil
            if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                transcript = scriptLang.defaultTranscript
            }
            attachRecordingIfNeeded()
            if asrModelId != nil { autoTranscribe() }
            return
        }
        if selectedId == nil { beginNewClone() }
        stopRecordingPlayback()
        stopGeneratedPlayback()
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized: startRecording()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                DispatchQueue.main.async {
                    if granted { startRecording() }
                    else { errorText = l10n.tr("voiceClone.needMic") }
                }
            }
        default:
            errorText = l10n.tr("voiceClone.needMic")
        }
    }

    private func startRecording() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("voice_clone_\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 48000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsFloatKey: false
        ]
        do {
            let rec = try AVAudioRecorder(url: url, settings: settings)
            rec.record()
            recorder = rec
            recordingURL = url
            isRecording = true
            errorText = nil
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func importAudio() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if selectedId == nil { beginNewClone() }
        recordingURL = url
        if transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            transcript = scriptLang.defaultTranscript
        }
        errorText = nil
        attachRecordingIfNeeded()
        if asrModelId != nil { autoTranscribe() }
    }

    private func autoTranscribe() {
        let url = recordingURL ?? selectedProfile.flatMap { VoiceProfileManager.shared.refAudioURL(for: $0) }
        guard let url, let model = asrModelId else { return }
        isTranscribing = true
        Task {
            do {
                let data = try Data(contentsOf: url)
                let result = try await inferenceService.transcriptionService.transcribe(
                    modelId: model, audioData: data, language: scriptLang.asrCode)
                await MainActor.run {
                    if !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        transcript = result.text
                        persistTranscript()
                    }
                    isTranscribing = false
                }
            } catch {
                await MainActor.run {
                    errorText = error.localizedDescription
                    isTranscribing = false
                }
            }
        }
    }

    private func attachRecordingIfNeeded() {
        guard let id = selectedId, let url = recordingURL else { return }
        do {
            try VoiceProfileManager.shared.setReferenceAudio(id: id, from: url)
            persistTranscript()
            commitInspectorName()
            reloadProfiles()
            errorText = nil
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func saveProfile() {
        guard selectedId != nil else {
            beginNewClone()
            return
        }
        isSaving = true
        commitInspectorName()
        persistTranscript()
        attachRecordingIfNeeded()
        recordingURL = nil
        isSaving = false
        reloadProfiles()
    }

    private func generate() {
        guard let profile = selectedProfile else {
            errorText = l10n.tr("voiceClone.needProfile")
            return
        }
        guard ttsModelId == selectedTTSId else {
            errorText = l10n.tr("voiceClone.needTTS")
            return
        }
        let text = speakText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        isGenerating = true
        errorText = nil
        Task {
            do {
                let data = try await inferenceService.ttsService.synthesize(
                    text: text,
                    engine: .neural,
                    voiceProfile: profile
                )
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("novamlx_clone_\(UUID().uuidString).wav")
                try data.write(to: url)
                let duration = (try? AVAudioPlayer(contentsOf: url))?.duration ?? 0
                var built = Self.estimateCues(text: text, duration: duration)
                var wordLevel = false
                do {
                    let words = try await ForcedAlignerService.align(
                        audioURL: url, text: text, language: scriptLang.alignerLanguage)
                    if !words.isEmpty {
                        built = words.map {
                            CloneCue(text: $0.text, start: $0.start, end: $0.end)
                        }
                        wordLevel = true
                    }
                } catch {
                    NovaMLXLog.warning("[VoiceClone] Forced aligner fallback: \(error.localizedDescription)")
                }
                await MainActor.run {
                    outputURL = url
                    cues = built
                    timestampsAreWordLevel = wordLevel
                    isGenerating = false
                    if let url = outputURL {
                        try? writeSidecars(beside: url)
                    }
                    togglePlay()
                }
            } catch {
                await MainActor.run {
                    errorText = error.localizedDescription
                    isGenerating = false
                }
            }
        }
    }

    private func togglePlay() {
        if isPlaying {
            stopGeneratedPlayback()
            return
        }
        guard let url = outputURL else { return }
        stopRecordingPlayback()
        do {
            let p = try AVAudioPlayer(contentsOf: url)
            player = p
            p.play()
            isPlaying = true
            DispatchQueue.main.asyncAfter(deadline: .now() + p.duration) {
                isPlaying = false
            }
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func togglePlayRecording() {
        if isPlayingRecording {
            stopRecordingPlayback()
            return
        }
        guard let url = previewableRecordingURL else { return }
        stopGeneratedPlayback()
        do {
            let p = try AVAudioPlayer(contentsOf: url)
            recordingPlayer = p
            p.play()
            isPlayingRecording = true
            DispatchQueue.main.asyncAfter(deadline: .now() + p.duration) {
                isPlayingRecording = false
            }
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func stopRecordingPlayback() {
        recordingPlayer?.stop()
        recordingPlayer = nil
        isPlayingRecording = false
    }

    private func stopGeneratedPlayback() {
        player?.stop()
        isPlaying = false
    }

    private func exportForFFmpeg() {
        guard let src = outputURL else { return }
        let panel = NSSavePanel()
        panel.title = l10n.tr("voiceClone.exportFfmpeg")
        panel.nameFieldStringValue = "clone.wav"
        panel.allowedContentTypes = [.wav]
        guard panel.runModal() == .OK, let dest = panel.url else { return }
        try? FileManager.default.removeItem(at: dest)
        do {
            try FileManager.default.copyItem(at: src, to: dest)
            try writeSidecars(beside: dest)
        } catch {
            errorText = error.localizedDescription
        }
    }

    private func writeSidecars(beside wavURL: URL) throws {
        let base = wavURL.deletingPathExtension()
        let srt = cues.enumerated().map { i, c in
            """
            \(i + 1)
            \(Self.srtTime(c.start)) --> \(Self.srtTime(c.end))
            \(c.text)
            """
        }.joined(separator: "\n\n") + "\n"
        try srt.write(to: base.appendingPathExtension("srt"), atomically: true, encoding: .utf8)

        let words: [[String: Any]] = cues.map {
            ["text": $0.text, "start": $0.start, "end": $0.end]
        }
        let jsonObj: [String: Any] = [
            "audio": wavURL.lastPathComponent,
            "format": timestampsAreWordLevel ? "word" : "estimated",
            "words": words,
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: jsonObj, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: base.appendingPathExtension("json"))

        let wavName = wavURL.lastPathComponent
        let srtName = base.appendingPathExtension("srt").lastPathComponent
        let cmd = """
        # ffmpeg: burn word-level subtitles onto a video, using this cloned WAV as audio.
        # Replace VIDEO.mp4 / OUTPUT.mp4. Run this in the same folder as \(wavName) and \(srtName).
        ffmpeg -y -i VIDEO.mp4 -i "\(wavName)" \\
          -vf "subtitles=\(srtName):force_style='FontName=PingFang SC,FontSize=22,Outline=1'" \\
          -map 0:v -map 1:a -c:v libx264 -c:a aac -shortest OUTPUT.mp4
        """
        try cmd.write(to: base.appendingPathExtension("ffmpeg.txt"), atomically: true, encoding: .utf8)
    }

    private var cloneEngineRows: [CloneEngineRow] {
        let pinned: [Qwen3TTSCloneService.EngineOption] = [
            .init(id: Qwen3TTSCloneService.defaultModelId, family: .qwen3Tts),
            .init(id: Qwen3TTSCloneService.dotsModelId, family: .dotsTts),
        ]
        var extra: [Qwen3TTSCloneService.EngineOption] = []
        for rec in modelManager.downloadedModels() {
            extra.append(.init(id: rec.id, family: rec.family))
        }
        for cat in modelManager.catalogModels(forCategory: .audio) {
            extra.append(.init(id: cat.id, family: cat.family))
        }
        return Qwen3TTSCloneService.mergeEngineList(pinned: pinned, extra: extra).map { opt in
            let title = modelManager.catalogModels(forCategory: .audio).first(where: { $0.id == opt.id })?.name
                ?? opt.id.split(separator: "/").last.map(String.init)
                ?? opt.id
            return CloneEngineRow(
                id: opt.id,
                family: opt.family,
                title: title,
                downloaded: modelManager.isDownloaded(opt.id)
            )
        }
    }

    private func loadRequiredModels() {
        errorText = nil
        isLoadingStack = true
        loadProgress = l10n.tr("voiceClone.loading")
        let engineId = selectedTTSId
        let failFmt = l10n.tr("voiceClone.downloadEngineFailed")
        Task {
            do {
                let rec = try await ensureSelectedEngine(id: engineId)
                try await runLoadStack(ttsRecord: rec)
                await MainActor.run {
                    loadProgress = l10n.tr("voiceClone.loadStackDone")
                    isLoadingStack = false
                }
            } catch {
                await MainActor.run {
                    errorText = String(format: failFmt, error.localizedDescription)
                    loadProgress = nil
                    isLoadingStack = false
                }
            }
        }
    }

    private func runLoadStack(ttsRecord: ModelRecord) async throws {
        let ttsFmt = l10n.tr("voiceClone.loadingTTS")
        let asrFmt = l10n.tr("voiceClone.loadingASR")
        if ttsModelId != ttsRecord.id {
            await MainActor.run { loadProgress = String(format: ttsFmt, ttsRecord.id) }
            let cfg = ModelConfig(
                identifier: ModelIdentifier(id: ttsRecord.id, family: ttsRecord.family),
                modelType: .audio
            )
            try await inferenceService.loadModel(at: ttsRecord.localURL, config: cfg)
        }
        if asrModelId == nil {
            if let rec = modelManager.downloadedModels().first(where: {
                $0.family == .qwen3Asr || $0.family == .whisper
            }) {
                await MainActor.run { loadProgress = String(format: asrFmt, rec.id) }
                let cfg = ModelConfig(
                    identifier: ModelIdentifier(id: rec.id, family: rec.family),
                    modelType: .audio
                )
                try await inferenceService.loadModel(at: rec.localURL, config: cfg)
            }
        }
        try await ensureAligner()
    }

    private func ensureAligner() async throws {
        if ForcedAlignerService.isModelOnDisk() {
            await MainActor.run { loadProgress = l10n.tr("voiceClone.loadingAligner") }
            return
        }
        let id = ForcedAlignerService.defaultModelId
        if modelManager.getRecord(id) == nil {
            modelManager.register(
                id: id,
                family: .qwen3Asr,
                modelType: .audio,
                remoteURL: "https://huggingface.co/\(id)",
                sizeBytes: 700_000_000
            )
        }
        if !modelManager.isDownloaded(id) {
            await MainActor.run {
                loadProgress = String(format: l10n.tr("voiceClone.downloadingAligner"), 0)
                appState.startDownload(repoId: id)
            }
            for _ in 0..<7_200 {
                if modelManager.isDownloaded(id) { break }
                let task = await MainActor.run { appState.downloadTasks[id] }
                if task?.status == .completed {
                    modelManager.discoverModels()
                    if modelManager.isDownloaded(id) { break }
                }
                if task?.status == .failed {
                    throw NovaMLXError.apiError(task?.errorMessage ?? "download failed")
                }
                let pct = Int(task?.progress ?? 0)
                await MainActor.run {
                    loadProgress = String(format: l10n.tr("voiceClone.downloadingAligner"), pct)
                }
                try await Task.sleep(nanoseconds: 1_000_000_000)
            }
            if !modelManager.isDownloaded(id) {
                throw NovaMLXError.apiError("ForcedAligner download timed out")
            }
        }
        await MainActor.run { loadProgress = l10n.tr("voiceClone.loadingAligner") }
        try await ForcedAlignerService.ensureModel(at: modelManager.getRecord(id)?.localURL)
    }

    private func ensureSelectedEngine(id: String) async throws -> ModelRecord {
        let family = cloneEngineRows.first(where: { $0.id == id })?.family
            ?? (Qwen3TTSCloneService.isBaseModelId(id) ? .qwen3Tts : .dotsTts)
        if modelManager.isDownloaded(id), let rec = modelManager.getRecord(id) {
            return rec
        }
        if modelManager.getRecord(id) == nil {
            let size = modelManager.catalogModels(forCategory: .audio).first(where: { $0.id == id })?.sizeBytes ?? 0
            modelManager.register(
                id: id,
                family: family,
                modelType: .audio,
                remoteURL: "https://huggingface.co/\(id)",
                sizeBytes: size
            )
        }
        await MainActor.run {
            loadProgress = String(format: l10n.tr("voiceClone.downloadingEngine"), id, 0)
            appState.startDownload(repoId: id)
        }
        for _ in 0..<7_200 {
            if modelManager.isDownloaded(id), let rec = modelManager.getRecord(id) {
                return rec
            }
            let task = await MainActor.run { appState.downloadTasks[id] }
            if task?.status == .completed {
                modelManager.discoverModels()
                if modelManager.isDownloaded(id), let rec = modelManager.getRecord(id) {
                    return rec
                }
            }
            if task?.status == .failed {
                throw NovaMLXError.apiError(task?.errorMessage ?? "download failed")
            }
            let pct = Int(task?.progress ?? 0)
            await MainActor.run {
                loadProgress = String(format: l10n.tr("voiceClone.downloadingEngine"), id, pct)
            }
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
        throw NovaMLXError.apiError("download timed out")
    }

    private func copySRT() {
        let srt = cues.enumerated().map { i, c in
            """
            \(i + 1)
            \(Self.srtTime(c.start)) --> \(Self.srtTime(c.end))
            \(c.text)
            """
        }.joined(separator: "\n\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(srt, forType: .string)
    }

    fileprivate static func estimateCues(text: String, duration: Double) -> [CloneCue] {
        let parts = text
            .split(whereSeparator: { "。！？.!?\n".contains($0) })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let chunks = parts.isEmpty ? [text] : parts
        let weights = chunks.map { Double(max($0.count, 1)) }
        let total = weights.reduce(0, +)
        guard total > 0, duration > 0 else { return [] }
        var t = 0.0
        return zip(chunks, weights).map { chunk, w in
            let len = duration * (w / total)
            let cue = CloneCue(text: chunk, start: t, end: t + len)
            t += len
            return cue
        }
    }

    fileprivate static func srtTime(_ seconds: Double) -> String {
        let ms = Int((seconds * 1000).rounded())
        let h = ms / 3_600_000
        let m = (ms / 60_000) % 60
        let s = (ms / 1000) % 60
        let milli = ms % 1000
        return String(format: "%02d:%02d:%02d,%03d", h, m, s, milli)
    }
}
