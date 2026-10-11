import AVKit
import SwiftUI
import NovaMLXCore
import NovaMLXEngine
import NovaMLXInference
import NovaMLXUtils

/// Digital-human (talking-head) page — pick a portrait + narration audio,
/// generate a lip-synced vertical video via the external LTX-2.5 a2v pipeline,
/// preview it and reveal it in Finder.
///
/// Unlike LLM/TTS/image pages there is no model to load here: the pipeline is
/// an external CLI that loads its weights per generation (and frees them
/// after), so the page shows pipeline readiness instead of a load button.
struct TalkingHeadPageView: View {
    @ObservedObject var appState: MenuBarAppState
    let inferenceService: InferenceService
    let memory: TalkingHeadPageMemory

    @EnvironmentObject var l10n: L10n

    @State private var imagePath: String = ""
    @State private var audioPath: String = ""
    @State private var prompt: String = ""
    @State private var resolutionIndex: Int = 0
    @State private var frameRate: Int = 24
    @State private var seedText: String = ""
    @State private var distilled: Bool = true
    @State private var isGenerating: Bool = false
    @State private var errorText: String?
    @State private var result: TalkingHeadResult?
    @State private var player: AVPlayer?
    @State private var progressTick: Date = Date()
    @State private var pollTimer: Timer?

    /// (width, height) presets — LTX two-stage needs multiples of 64.
    private static let presets: [(Int, Int)] = [(576, 1024), (768, 576), (704, 1280)]

    private var service: TalkingHeadService { inferenceService.talkingHeadService }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: NovaTheme.Spacing.lg) {
                header
                pipelineStatus
                inputSection
                optionsSection
                generateSection
                if let errorText {
                    Text(errorText)
                        .font(.system(size: 12))
                        .foregroundColor(NovaTheme.Colors.statusError)
                }
                resultSection
            }
            .padding(NovaTheme.Spacing.xxl)
        }
        .onAppear {
            imagePath = memory.imagePath
            audioPath = memory.audioPath
            prompt = memory.prompt
            resolutionIndex = memory.resolutionIndex
            frameRate = memory.frameRate
            distilled = memory.distilled
            startPolling()
        }
        .onDisappear {
            stopPolling()
            persist()
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: NovaTheme.Spacing.xs) {
            Text(l10n.tr("app.talkingHead")).font(.title2.bold())
            Text(l10n.tr("talkingHead.subtitle")).font(.caption).foregroundColor(.secondary)
        }
    }

    /// Pipeline readiness instead of a load button: the CLI loads weights
    /// per generation, so there is no resident-model lifecycle to drive.
    private var pipelineStatus: some View {
        HStack(spacing: NovaTheme.Spacing.sm) {
            Image(systemName: service.isReady ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                .foregroundColor(service.isReady ? NovaTheme.Colors.statusOK : NovaTheme.Colors.statusWarn)
            if service.isReady {
                Text(l10n.tr("talkingHead.statusReady")).font(.system(size: 12))
            } else {
                Text(String(format: l10n.tr("talkingHead.statusMissing"), service.config.cliPath, service.config.modelDir))
                    .font(.system(size: 12))
                    .foregroundColor(NovaTheme.Colors.statusWarn)
            }
            Spacer()
        }
        .padding(NovaTheme.Spacing.md)
        .background(NovaTheme.Colors.cardBackground)
        .overlay(RoundedRectangle(cornerRadius: NovaTheme.Radius.lg).stroke(NovaTheme.Colors.cardBorder, lineWidth: 1))
    }

    // MARK: - Inputs

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: NovaTheme.Spacing.md) {
            HStack {
                Button(l10n.tr("talkingHead.pickImage"), action: pickImage)
                Text(imagePath.isEmpty ? l10n.tr("talkingHead.noFile") : URL(fileURLWithPath: imagePath).lastPathComponent)
                    .font(.system(size: 12)).foregroundColor(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
            }
            HStack {
                Button(l10n.tr("talkingHead.pickAudio"), action: pickAudio)
                Text(audioPath.isEmpty ? l10n.tr("talkingHead.noFile") : URL(fileURLWithPath: audioPath).lastPathComponent)
                    .font(.system(size: 12)).foregroundColor(.secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
            }
            VStack(alignment: .leading, spacing: 4) {
                Text(l10n.tr("talkingHead.prompt")).font(.system(size: 12, weight: .semibold))
                TextField(l10n.tr("talkingHead.promptPlaceholder"), text: $prompt, axis: .vertical)
                    .lineLimit(2...4)
                    .textFieldStyle(.roundedBorder)
            }
        }
    }

    private var optionsSection: some View {
        HStack(spacing: NovaTheme.Spacing.lg) {
            Picker(l10n.tr("talkingHead.resolution"), selection: $resolutionIndex) {
                ForEach(Array(Self.presets.enumerated()), id: \.offset) { i, preset in
                    Text("\(preset.0)×\(preset.1)").tag(i)
                }
            }
            .frame(width: 160)
            Picker(l10n.tr("talkingHead.fps"), selection: $frameRate) {
                ForEach([24, 25, 30], id: \.self) { Text("\($0)").tag($0) }
            }
            .frame(width: 110)
            Toggle(l10n.tr("talkingHead.distilled"), isOn: $distilled)
            VStack(alignment: .leading, spacing: 2) {
                Text(l10n.tr("talkingHead.seed")).font(.system(size: 11)).foregroundColor(.secondary)
                TextField("—", text: $seedText)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
            }
            Spacer()
        }
    }

    // MARK: - Generate

    private var generateSection: some View {
        HStack {
            Button {
                startGeneration()
            } label: {
                if isGenerating {
                    ProgressView().controlSize(.small)
                } else {
                    Image(systemName: "person.wave.2")
                }
                Text(isGenerating ? l10n.tr("talkingHead.generating") : l10n.tr("talkingHead.generate"))
            }
            .disabled(isGenerating || imagePath.isEmpty || audioPath.isEmpty || !service.isReady)

            if isGenerating, let job = service.inFlight() {
                Text(String(
                    format: l10n.tr("talkingHead.progress"),
                    job.segment + 1, job.totalSegments,
                    String(format: "%.0fs", Date().timeIntervalSince(job.startedAt))
                ))
                .font(.system(size: 12)).foregroundColor(.secondary)
            }
            Spacer()
        }
    }

    // MARK: - Result

    @ViewBuilder
    private var resultSection: some View {
        if let result {
            VStack(alignment: .leading, spacing: NovaTheme.Spacing.md) {
                Text(String(
                    format: l10n.tr("talkingHead.done"),
                    String(format: "%.1f", result.audioSeconds),
                    String(format: "%.0f", result.elapsed)
                ))
                .font(.system(size: 12, weight: .semibold))
                if let player {
                    VideoPlayer(player: player)
                        .frame(height: 320)
                        .cornerRadius(8)
                }
                HStack {
                    Button(l10n.tr("talkingHead.reveal")) { NSWorkspace.shared.activateFileViewerSelecting([result.outputURL]) }
                    Text(result.outputURL.lastPathComponent)
                        .font(.system(size: 11)).foregroundColor(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                    Spacer()
                }
            }
        }
    }

    // MARK: - Actions

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        imagePath = url.path
        persist()
    }

    private func pickAudio() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        audioPath = url.path
        persist()
    }

    private func startGeneration() {
        persist()
        errorText = nil
        result = nil
        player = nil
        isGenerating = true
        let image = URL(fileURLWithPath: imagePath)
        let audio = URL(fileURLWithPath: audioPath)
        let preset = Self.presets[resolutionIndex]
        Task {
            do {
                let output = try await service.generate(
                    image: image,
                    audio: audio,
                    prompt: prompt.isEmpty ? nil : prompt,
                    width: preset.0,
                    height: preset.1,
                    frameRate: frameRate,
                    seed: UInt64(seedText),
                    distilled: distilled
                )
                await MainActor.run {
                    result = output
                    player = AVPlayer(url: output.outputURL)
                    isGenerating = false
                }
            } catch {
                await MainActor.run {
                    errorText = error.localizedDescription
                    isGenerating = false
                }
            }
        }
    }

    // MARK: - State plumbing

    /// Refresh the running-job line while a generation is in flight (the
    /// service mutates on its own queue).
    private func startPolling() {
        stopPolling()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @MainActor in progressTick = Date() }
        }
    }

    private func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    private func persist() {
        memory.imagePath = imagePath
        memory.audioPath = audioPath
        memory.prompt = prompt
        memory.resolutionIndex = resolutionIndex
        memory.frameRate = frameRate
        memory.distilled = distilled
    }
}

/// Draft state that survives page switches within one app run.
final class TalkingHeadPageMemory: ObservableObject {
    @Published var imagePath: String = ""
    @Published var audioPath: String = ""
    @Published var prompt: String = ""
    @Published var resolutionIndex: Int = 0
    @Published var frameRate: Int = 24
    @Published var distilled: Bool = true
}
