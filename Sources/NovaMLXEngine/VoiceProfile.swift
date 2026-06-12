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
        let profileDir = NovaMLXPaths.voicesDir.appendingPathComponent(id.uuidString)
        let profilePath = profileDir.appendingPathComponent("profile.json")

        guard var profile = loadProfile(id),
              let data = try? JSONEncoder().encode(VoiceProfile(
                  id: profile.id,
                  name: newName,
                  refAudioFilename: profile.refAudioFilename,
                  refTranscript: profile.refTranscript,
                  createdAt: profile.createdAt
              ))
        else { return }
        try? data.write(to: profilePath)
    }

    // MARK: - Load Audio

    public func loadRefAudio(for profile: VoiceProfile) -> MLXArray? {
        let profileDir = NovaMLXPaths.voicesDir.appendingPathComponent(profile.id.uuidString)
        let audioPath = profileDir.appendingPathComponent(profile.refAudioFilename)

        guard FileManager.default.fileExists(atPath: audioPath.path) else { return nil }

        do {
            let (_, audioArray) = try loadAudioArray(from: audioPath, sampleRate: 48000)
            return audioArray.squeezed() // flat 1D array for DotsTTS
        } catch {
            return nil
        }
    }

    // MARK: - Private

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
