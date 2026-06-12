# Audio Page Redesign Spec

**Date:** 2026-06-07
**Scope:** AudioPageView.swift, TTSService.swift
**Dependencies:** Qwen3TTSModel (loaded), AVFoundation

## Overview

Three fixes for the Audio page: broken ASR recording, missing recording download, and TTS voice/backend selection.

---

## 1. ASR Recording Fix

### Problem
Microphone recording produces garbled transcription ("dram dram dram") while uploaded audio files transcribe correctly. Root cause: `AVAudioSession` is never configured for recording.

### Solution
Before creating `AVAudioRecorder`, set the audio session category:

```
AVAudioSession.sharedInstance()
  .setCategory(.playAndRecord, mode: .measurement)
  .setActive(true)
```

On stop recording, deactivate: `setActive(false)`.

Recording format settings (16kHz, mono, 16-bit PCM) are already correct for Whisper/Qwen3-ASR models.

### Files
- `Sources/NovaMLXMenuBar/AudioPageView.swift` — `startRecording()`, `stopRecording()`

---

## 2. Recording Download

### Problem
After recording, the user sees only a filename and a clear (X) button. Cannot save the recording to disk.

### Solution
Add a download button next to the recording info row. When tapped:
1. Open `NSSavePanel` with default name "recording.wav"
2. User picks save location
3. Copy temp recording file to chosen location via `FileManager.copyItem`

UI layout after recording:
```
[doc icon] recording.wav   [Download] [X clear]
```

### Files
- `Sources/NovaMLXMenuBar/AudioPageView.swift` — recording info section in `asrView`

---

## 3. TTS Voice Selection

### Problem
Two TTS backends exist (Qwen3-TTS neural model vs macOS `say`) but the UI has no way to:
- Switch between them
- Pick a macOS voice
- See which backend will be used

### Architecture

#### TTSEngine enum (in TTSService)
```swift
public enum TTSEngine: Sendable {
    case neural      // Qwen3-TTS MLX model (must be loaded)
    case system      // macOS `say` command (always available)
}
```

#### TTSService changes
- `synthesize()` gains optional `engine: TTSEngine?` parameter (default nil = auto)
- Auto logic: if nil, use `.neural` when model loaded, else `.system`
- When `.neural` requested but no model loaded → throw error
- When `.system` requested → always use macOS `say`, ignore loaded model
- `listMacOSVoices()` -> `[(name: String, locale: String)]` — parses `say -v '?'` output, cached

#### AudioPageView TTS section

**State:**
```swift
@State private var ttsEngine: TTSEngine = .neural  // or .system
@State private var selectedSystemVoice: String = "Tingting"
@State private var macOSVoices: [(name: String, locale: String)] = []
```

**UI layout:**
```
Model section:
  [Segmented: Neural TTS | System TTS]

  IF Neural selected:
    Show loaded model name OR "No model loaded" warning

  IF System selected:
    [Voice dropdown: Tingting (zh_CN) ▼]
    Filtered list of macOS voices, grouped or searchable

Input text area (unchanged)

[Synthesize] [Play/Pause]
```

**Voice list fetching:**
- On appear, run `Process()` with `/usr/bin/say` args `["-v", "?"]`
- Parse stdout: each line = `"VoiceName    Locale    SampleText"`
- Cache in `@State`, only fetch once
- Default selection: "Tingting" (zh_CN)

### Files
- `Sources/NovaMLXEngine/TTSService.swift` — add `TTSEngine` enum, voice list method, engine parameter
- `Sources/NovaMLXMenuBar/AudioPageView.swift` — TTS section UI redesign, voice fetching, engine picker

---

## Success Criteria
1. ASR recording produces accurate Chinese transcription (matching upload quality)
2. Recording can be downloaded to user-chosen location
3. User can switch between Neural TTS and System TTS
4. System TTS shows all macOS voices, user can pick any
5. Neural TTS shows loaded model name, disabled when no model loaded
6. Both engines produce playable audio

## Out of Scope
- Qwen3-TTS speaker/voice presets (Base models have none)
- Recording waveform visualization
- Audio format conversion (WAV/MP3 selection)
- Performance optimization of Qwen3-TTS synthesis speed
