import Foundation
import NovaMLXCore

public struct UpdateInfo: Codable, Sendable {
    public let currentVersion: String
    public let latestVersion: String
    public let updateAvailable: Bool
    public let releaseUrl: String
    public let releaseNotes: String
    public let checkedAt: Date

    public init(
        currentVersion: String,
        latestVersion: String,
        updateAvailable: Bool,
        releaseUrl: String,
        releaseNotes: String,
        checkedAt: Date = Date()
    ) {
        self.currentVersion = currentVersion
        self.latestVersion = latestVersion
        self.updateAvailable = updateAvailable
        self.releaseUrl = releaseUrl
        self.releaseNotes = releaseNotes
        self.checkedAt = checkedAt
    }
}

public final class UpdateChecker: @unchecked Sendable {
    private let repository: String
    private let currentVersion: String
    private var cachedResult: UpdateInfo?
    private var lastCheckTime: Date?
    private let cacheInterval: TimeInterval
    private let lock = NovaMLXLock()

    public init(
        repository: String = "lucas/novamlx",
        currentVersion: String = NovaMLXCore.version,
        cacheInterval: TimeInterval = 3600
    ) {
        self.repository = repository
        self.currentVersion = currentVersion
        self.cacheInterval = cacheInterval
    }

    public func checkForUpdates() async throws -> UpdateInfo {
        if let cached = cachedResult, let lastCheck = lastCheckTime,
           Date().timeIntervalSince(lastCheck) < cacheInterval {
            return cached
        }

        let urlString = "https://api.github.com/repos/\(repository)/releases/latest"
        guard let url = URL(string: urlString) else {
            throw NovaMLXError.configurationError("Invalid update URL")
        }

        var request = URLRequest(url: url)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 15

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NovaMLXError.downloadFailed(urlString, underlying: NSError(domain: "UpdateChecker", code: -1))
        }

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let tagName = json["tag_name"] as? String else {
            throw NovaMLXError.apiError("Invalid GitHub API response")
        }

        let latestVersion = tagName.hasPrefix("v") ? String(tagName.dropFirst()) : tagName
        let htmlUrl = json["html_url"] as? String ?? ""
        let body = json["body"] as? String ?? ""

        let updateAvailable = compareVersions(latestVersion, currentVersion) == .orderedDescending

        let info = UpdateInfo(
            currentVersion: currentVersion,
            latestVersion: latestVersion,
            updateAvailable: updateAvailable,
            releaseUrl: htmlUrl,
            releaseNotes: String(body.prefix(1000))
        )

        lock.withLock {
            cachedResult = info
            lastCheckTime = Date()
        }

        return info
    }

    private func compareVersions(_ a: String, _ b: String) -> ComparisonResult {
        let aParts = a.split(separator: ".").compactMap { Int($0) }
        let bParts = b.split(separator: ".").compactMap { Int($0) }
        let maxLen = max(aParts.count, bParts.count)

        for i in 0..<maxLen {
            let aVal = i < aParts.count ? aParts[i] : 0
            let bVal = i < bParts.count ? bParts[i] : 0
            if aVal > bVal { return .orderedDescending }
            if aVal < bVal { return .orderedAscending }
        }
        return .orderedSame
    }
}
