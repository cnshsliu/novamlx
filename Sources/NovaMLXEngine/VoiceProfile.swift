import Foundation
import MLX
import NovaMLXCore
import NovaMLXAudio

public struct VoiceProfile: Codable, Identifiable, Sendable {
    public let id: UUID
    public var name: String
    public let refAudioFilename: String
    public let refTranscript: String
    public let createdAt: Date

    private enum CodingKeys: String, CodingKey {
        case id, name, refAudioFilename, refTranscript, createdAt
    }

    public init(id: UUID = UUID(), name: String, refAudioFilename: String, refTranscript: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.refAudioFilename = refAudioFilename
        self.refTranscript = refTranscript
        self.createdAt = createdAt
    }
}

public final class VoiceProfileManager: @unchecked Sendable {
    public static let shared = VoiceProfileManager()
    private let lock = NSLock()

    private init() {}

    // MARK: - List

    public func listProfiles() -> [VoiceProfile] {
        let fm = FileManager.default
        let voicesDir = NovaMLXPaths.voicesDir

        guard let contents = try? fm.contentsOfDirectory(at: voicesDir, includingPropertiesForKeys: [.isDirectoryKey]) else {
            return []
        }

        return contents.compactMap { dir in
            let profilePath = dir.appendingPathComponent("profile.json")
            guard let data = try? Data(contentsOf: profilePath),
                  let profile = try? JSONDecoder().decode(VoiceProfile.self, from: data)
            else { return nil }
            return profile
        }.sorted { $0.createdAt > $1.createdAt }
    }

    /// Next "Voice" / "Voice 2" style name that is not already used.
    public static func uniqueName(base: String, existing: [String]) -> String {
        let set = Set(existing)
        if !set.contains(base) { return base }
        var n = 2
        while set.contains("\(base) \(n)") { n += 1 }
        return "\(base) \(n)"
    }

    public func hasReferenceAudio(for profile: VoiceProfile) -> Bool {
        refAudioURL(for: profile) != nil
    }

    /// Create a list row immediately. Reference audio can be attached later.
    public func createDraft(name: String, refTranscript: String) throws -> VoiceProfile {
        let profileId = UUID()
        let profileDir = NovaMLXPaths.voicesDir.appendingPathComponent(profileId.uuidString)
        try FileManager.default.createDirectory(at: profileDir, withIntermediateDirectories: true)
        let profile = VoiceProfile(
            id: profileId,
            name: name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "Voice \(profileId.uuidString.prefix(8))" : name,
            refAudioFilename: "reference.wav",
            refTranscript: refTranscript
        )
        let profileData = try JSONEncoder().encode(profile)
        try profileData.write(to: profileDir.appendingPathComponent("profile.json"))
        return profile
    }

    public func updateTranscript(_ id: UUID, refTranscript: String) {
        guard let profile = loadProfile(id) else { return }
        writeProfile(VoiceProfile(
            id: profile.id,
            name: profile.name,
            refAudioFilename: profile.refAudioFilename,
            refTranscript: refTranscript,
            createdAt: profile.createdAt
        ))
    }

    public func setReferenceAudio(id: UUID, from url: URL) throws {
        guard let profile = loadProfile(id) else {
            throw NovaMLXError.apiError("Voice profile not found")
        }
        let dest = NovaMLXPaths.voicesDir
            .appendingPathComponent(id.uuidString)
            .appendingPathComponent(profile.refAudioFilename)
        if FileManager.default.fileExists(atPath: dest.path) {
            try FileManager.default.removeItem(at: dest)
        }
        try FileManager.default.copyItem(at: url, to: dest)
    }

    // MARK: - Save

    public func saveProfile(name: String, refAudioURL: URL, refTranscript: String) throws -> VoiceProfile {
        let profileId = UUID()
        let profileDir = NovaMLXPaths.voicesDir.appendingPathComponent(profileId.uuidString)

        try FileManager.default.createDirectory(at: profileDir, withIntermediateDirectories: true)

        let destAudio = profileDir.appendingPathComponent("reference.wav")
        try FileManager.default.copyItem(at: refAudioURL, to: destAudio)

        let profile = VoiceProfile(
            id: profileId,
            name: name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Voice \(profileId.uuidString.prefix(8))" : name,
            refAudioFilename: "reference.wav",
            refTranscript: refTranscript
        )

        let profileData = try JSONEncoder().encode(profile)
        try profileData.write(to: profileDir.appendingPathComponent("profile.json"))

        return profile
    }

    // MARK: - Delete

    public func deleteProfile(_ id: UUID) {
        let profileDir = NovaMLXPaths.voicesDir.appendingPathComponent(id.uuidString)
        try? FileManager.default.removeItem(at: profileDir)
    }

    // MARK: - Rename

    public func renameProfile(_ id: UUID, newName: String) {
        guard let profile = loadProfile(id) else { return }
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        writeProfile(VoiceProfile(
            id: profile.id,
            name: trimmed,
            refAudioFilename: profile.refAudioFilename,
            refTranscript: profile.refTranscript,
            createdAt: profile.createdAt
        ))
    }

    // MARK: - Load Audio

    public func refAudioURL(for profile: VoiceProfile) -> URL? {
        let audioPath = NovaMLXPaths.voicesDir
            .appendingPathComponent(profile.id.uuidString)
            .appendingPathComponent(profile.refAudioFilename)
        guard FileManager.default.fileExists(atPath: audioPath.path) else { return nil }
        return audioPath
    }

    public func loadRefAudio(for profile: VoiceProfile) -> MLXArray? {
        guard let audioPath = refAudioURL(for: profile) else { return nil }

        do {
            let (_, audioArray) = try loadAudioArray(from: audioPath, sampleRate: 48000)
            return audioArray.squeezed() // flat 1D array for DotsTTS
        } catch {
            return nil
        }
    }

    // MARK: - Private

    private func writeProfile(_ profile: VoiceProfile) {
        let profilePath = NovaMLXPaths.voicesDir
            .appendingPathComponent(profile.id.uuidString)
            .appendingPathComponent("profile.json")
        guard let data = try? JSONEncoder().encode(profile) else { return }
        try? data.write(to: profilePath)
    }

    private func loadProfile(_ id: UUID) -> VoiceProfile? {
        let profilePath = NovaMLXPaths.voicesDir
            .appendingPathComponent(id.uuidString)
            .appendingPathComponent("profile.json")

        guard let data = try? Data(contentsOf: profilePath),
              let profile = try? JSONDecoder().decode(VoiceProfile.self, from: data)
        else { return nil }
        return profile
    }
}
