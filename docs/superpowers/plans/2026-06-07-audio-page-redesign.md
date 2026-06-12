# Audio Page Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix ASR recording, add recording download, and add TTS voice/backend selection to the Audio page.

**Architecture:** Three independent features. ASR fix is a 1-line audio session config. Recording download adds an NSSavePanel button. TTS voice selection adds a TTSEngine enum to TTSService and a segmented control + voice picker to AudioPageView.

**Tech Stack:** SwiftUI, AVFoundation (AVAudioSession, AVAudioRecorder, AVAudioPlayer), Process (`/usr/bin/say`)

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `Sources/NovaMLXEngine/TTSService.swift` | Modify | Add `TTSEngine` enum, `engine` param to `synthesize()`, `listMacOSVoices()` |
| `Sources/NovaMLXMenuBar/AudioPageView.swift` | Modify | Fix `startRecording()`, add download button, redesign TTS section with engine picker + voice dropdown |

---

### Task 1: Fix ASR Recording — Audio Session Config

**Files:**
- Modify: `Sources/NovaMLXMenuBar/AudioPageView.swift:506-530`

- [ ] **Step 1: Add audio session setup to `startRecording()`**

Replace the `startRecording()` function (lines 506-530) with:

```swift
private func startRecording() {
    let tempDir = FileManager.default.temporaryDirectory
    let url = tempDir.appendingPathComponent("novamlx_recording_\(UUID().uuidString).wav")

    let settings: [String: Any] = [
        AVFormatIDKey: Int(kAudioFormatLinearPCM),
        AVSampleRateKey: 16000.0,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsBigEndianKey: false,
        AVLinearPCMIsFloatKey: false
    ]

    do {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement)
        try session.setActive(true)

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
```

- [ ] **Step 2: Deactivate audio session in `stopRecording()`**

Replace the `stopRecording()` function (lines 532-539) with:

```swift
private func stopRecording() {
    audioRecorder?.stop()
    isRecording = false
    uploadedFileName = "recording.wav"

    try? AVAudioSession.sharedInstance().setActive(false)

    guard let url = recordingURL else { return }
    transcribeAudio(url: url)
}
```

- [ ] **Step 3: Build**

Run: `./build.sh`
Expected: Build complete

- [ ] **Step 4: Deploy and test**

Run: `killall NovaMLX; sleep 2; open dist/NovaMLX.app`

Manual test: Open Audio page → ASR tab → click Record → speak Chinese → stop → verify transcription is accurate (not "dram dram").

---

### Task 2: Add Recording Download Button

**Files:**
- Modify: `Sources/NovaMLXMenuBar/AudioPageView.swift:172-191`

- [ ] **Step 1: Replace the recording info row with download button**

Replace the recording info block (lines 172-191, the `if let name = uploadedFileName` section) with:

```swift
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
```

- [ ] **Step 2: Add the `downloadRecording()` function**

Add after `uploadAudioFile()` (after line 551):

```swift
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
```

- [ ] **Step 3: Build**

Run: `./build.sh`
Expected: Build complete

- [ ] **Step 4: Deploy and test**

Run: `killall NovaMLX; sleep 2; open dist/NovaMLX.app`

Manual test: Record audio → see download button (arrow down icon) → click → choose save location → verify file saved.

---

### Task 3: Add TTSEngine Enum and Voice List to TTSService

**Files:**
- Modify: `Sources/NovaMLXEngine/TTSService.swift`

- [ ] **Step 1: Add `TTSEngine` enum and `macOSVoice` struct**

Add before `class TTSService` (after line 9):

```swift
public enum TTSEngine: String, CaseIterable, Sendable {
    case neural = "Neural TTS"
    case system = "System TTS"
}

public struct MacOSVoice: Identifiable, Hashable, Sendable {
    public let name: String
    public let locale: String
    public var id: String { "\(name) (\(locale))" }
}
```

- [ ] **Step 2: Add `engine` parameter to `synthesize()`**

Replace the `synthesize()` function (lines 107-126) with:

```swift
public func synthesize(
    text: String,
    voice: String = "Tingting",
    rate: Int = 175,
    engine: TTSEngine? = nil
) async throws -> Data {
    let resolvedEngine: TTSEngine
    if let engine {
        resolvedEngine = engine
    } else {
        resolvedEngine = lock.withLock { ttsModel != nil } ? .neural : .system
    }

    switch resolvedEngine {
    case .neural:
        let (model, config, tokenizer) = lock.withLock { (ttsModel, ttsConfig, speechTokenizer) }
        guard let model, let config else {
            throw NovaMLXError.apiError("No neural TTS model loaded. Load a model or switch to System TTS.")
        }
        ttsLog.info("[TTS] Using Qwen3-TTS for synthesis")
        do {
            return try synthesizeWithQwen3TTS(model: model, config: config, tokenizer: tokenizer, text: text)
        } catch {
            ttsLog.error("[TTS] Qwen3-TTS synthesis failed: \(error)")
            throw error
        }
    case .system:
        ttsLog.info("[TTS] Using macOS system TTS (voice=\(voice))")
        return try synthesizeWithMacOS(text: text, voice: voice, rate: rate)
    }
}
```

- [ ] **Step 3: Add `listMacOSVoices()` static method**

Add after `unloadModel()` (after line 105):

```swift
public static func listMacOSVoices() -> [MacOSVoice] {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
    process.arguments = ["-v", "?"]

    let pipe = Pipe()
    process.standardOutput = pipe

    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        return output.split(separator: "\n").compactMap { line in
            let parts = line.split(separator: " ", omittingEmptySubsequences: true)
            guard parts.count >= 2 else { return nil }
            return MacOSVoice(name: String(parts[0]), locale: String(parts[1]))
        }
    } catch {
        return [MacOSVoice(name: "Tingting", locale: "zh_CN")]
    }
}
```

- [ ] **Step 4: Build**

Run: `./build.sh`
Expected: Build complete

---

### Task 4: Redesign TTS Section UI in AudioPageView

**Files:**
- Modify: `Sources/NovaMLXMenuBar/AudioPageView.swift`

- [ ] **Step 1: Add TTS state variables**

Replace the TTS State block (lines 34-40) with:

```swift
// MARK: - TTS State
@State private var ttsText = ""
@State private var isSynthesizing = false
@State private var ttsError: String?
@State private var isPlaying = false
@State private var synthesizedAudioURL: URL?
@State private var audioPlayer: AVAudioPlayer?

@State private var ttsEngine: TTSEngine = .neural
@State private var selectedSystemVoice: String = "Tingting"
@State private var macOSVoices: [MacOSVoice] = []
```

Remove the `loadedTTSModelName` computed property (lines 42-48) — it will be replaced.

- [ ] **Step 2: Add voice loading in `onAppear` and helper**

Replace the `onAppear` in `body` (line 60-66) with:

```swift
.onAppear {
    autoSelectModels()
    loadMacOSVoices()
}
```

Add helper function after `autoSelectModels()`:

```swift
private func loadMacOSVoices() {
    guard macOSVoices.isEmpty else { return }
    DispatchQueue.global(qos: .userInitiated).async {
        let voices = TTSService.listMacOSVoices()
        DispatchQueue.main.async {
            macOSVoices = voices
        }
    }
}
```

- [ ] **Step 3: Replace the entire `ttsView` (lines 252-357)**

Replace with:

```swift
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
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(3)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))

                // Engine info
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
                            Text("No neural TTS model loaded")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        .padding(6)
                    }
                } else {
                    // Voice picker
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

            Spacer(minLength: 0)
        }
        .padding(20)
    }
}

private var loadedTTSModelName: String? {
    let ttsModels = appState.loadedModels.filter { modelId in
        guard let record = modelManager.getRecord(modelId) else { return false }
        return record.family == .qwen3Tts
    }
    return ttsModels.first.map { shortModelName($0) }
}
```

- [ ] **Step 4: Update `synthesizeSpeech()` to pass engine and voice**

Replace `synthesizeSpeech()` (lines 587-615) with:

```swift
private func synthesizeSpeech() {
    guard !ttsText.isEmpty else { return }

    isSynthesizing = true
    ttsError = nil

    Task {
        do {
            let audioData = try await inferenceService.ttsService.synthesize(
                text: ttsText,
                voice: selectedSystemVoice,
                engine: ttsEngine
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
```

- [ ] **Step 5: Build**

Run: `./build.sh`
Expected: Build complete

- [ ] **Step 6: Deploy and test all features**

Run: `killall NovaMLX; sleep 2; open dist/NovaMLX.app`

Manual tests:
1. **ASR recording**: Record Chinese → verify accurate transcription
2. **Recording download**: Record → click download arrow → save → verify file
3. **Neural TTS**: Select "Neural TTS" → type text → synthesize → verify audio plays
4. **System TTS**: Select "System TTS" → pick a voice → synthesize → verify audio plays
5. **Engine switch**: Switch between Neural and System → verify UI updates correctly
6. **No model loaded**: Select Neural when no model loaded → verify synthesize button disabled

---

### Task 5: Final Cleanup — Remove Verbose Debug Logging

**Files:**
- Modify: `Sources/NovaMLXAudio/Qwen3TTSModel.swift`

- [ ] **Step 1: Remove verbose shape/dtype logging from `sample()` and `predictAll()`**

Replace the verbose `sample()` function with the clean version:

```swift
private func sample(logits: MLXArray, temperature: Float, topK: Int) -> Int32 {
    var l = logits.asType(.float32)
    if temperature > 0 {
        l = l / temperature
        if topK > 0 && topK < l.dim(-1) {
            let kth = topK - 1
            let maskIndices = argPartition(-l, kth: kth, axis: -1)[.ellipsis, topK...]
            l = putAlong(l, maskIndices, values: MLXArray(Float(-1e9)), axis: -1)
        }
        return MLXRandom.categorical(l).item(Int32.self)
    }
    return argMax(l, axis: -1).item(Int32.self)
}
```

Remove all `ttsModelLog.info("[predictAll]...")` and `ttsModelLog.info("[synthesize]...")` debug lines from `predictAll()` and `synthesize()`, keeping only the entry/exit logs:

In `predictAll`, keep only:
```swift
ttsModelLog.info("[predictAll] start, numCodeGroups=\(ncg)")
```
and:
```swift
ttsModelLog.info("[predictAll] done. \(codecIds.count) tokens")
```

In `synthesize`, keep only:
```swift
ttsModelLog.info("[synthesize] start")
```
and:
```swift
ttsModelLog.info("[synthesize] done. \(step) steps, \(allCodecIds.count) tokens")
```

- [ ] **Step 2: Build**

Run: `./build.sh`
Expected: Build complete

- [ ] **Step 3: Final deploy**

Run: `killall NovaMLX; sleep 2; open dist/NovaMLX.app`
