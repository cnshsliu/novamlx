import Foundation
import NovaMLXUtils
import NovaMLXCore

/// Dedicated service for Alibaba ModelScope (魔搭社区).
/// This is the first per-source implementation as requested.
/// All ModelScope-specific logic (search, file listing, URLs, future LFS handling, etc.)
/// should live here instead of being scattered with if-branches in HuggingFaceService.
public final class ModelScopeService: Sendable {

    public let endpoint: String
    private let session: URLSession

    public init(endpoint: String = "https://www.modelscope.cn", session: URLSession = .shared) {
        self.endpoint = endpoint.hasSuffix("/") ? String(endpoint.dropLast()) : endpoint
        self.session = session
    }

    // MARK: - Search (already wired from previous fix)
    // The real search is currently still called from HuggingFaceService for now.
    // We can move the full search implementation here in a follow-up.

    // MARK: - File Listing

    /// Lists all downloadable files for a repo on ModelScope.
    /// Uses the live endpoint observed from the official website.
    public func listFiles(
        repoId: String,
        revision: String = "master"
    ) async throws -> [ModelScopeFile] {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        // Live site uses ?Revision=...&Root=  (Recursive=true also works but we match production)
        let url = URL(string: "\(endpoint)/api/v1/models/\(encoded)/repo/files?Revision=\(revision)&Root=")!

        NovaMLXLog.info("[ModelScope] Listing files: \(url)")

        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await session.data(for: req)

        if let http = response as? HTTPURLResponse {
            NovaMLXLog.info("[ModelScope] File list HTTP \(http.statusCode)")
        }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            let preview = String(data: data.prefix(600), encoding: .utf8) ?? "<binary>"
            NovaMLXLog.error("[ModelScope] File list response was not JSON. Preview: \(preview)")
            throw ModelScopeError.unexpectedFileListResponse
        }

        let dataObj = json["Data"] as? [String: Any] ?? json["data"] as? [String: Any] ?? [:]
        let filesArray = (dataObj["Files"] as? [[String: Any]])
                      ?? (dataObj["files"] as? [[String: Any]])
                      ?? (dataObj["RepoFiles"] as? [[String: Any]])

        guard let filesArray = filesArray else {
            NovaMLXLog.error("[ModelScope] File list JSON keys: top=\(json.keys), Data keys=\(dataObj.keys)")
            let preview = String(data: data.prefix(600), encoding: .utf8) ?? "<binary>"
            NovaMLXLog.error("[ModelScope] Could not find Files array. Preview: \(preview)")
            throw ModelScopeError.unexpectedFileListResponse
        }

        let files: [ModelScopeFile] = filesArray.compactMap { dict in
            guard let name = dict["Name"] as? String ?? dict["Path"] as? String,
                  let type = dict["Type"] as? String,
                  // "blob" = file, "tree" = directory
                  type == "blob" || type == "file" else {
                return nil
            }
            let size = dict["Size"] as? Int ?? 0
            let isLFS = dict["IsLFS"] as? Bool ?? false
            let sha256 = dict["Sha256"] as? String
            return ModelScopeFile(
                path: dict["Path"] as? String ?? name,
                name: name,
                size: size,
                isLFS: isLFS,
                sha256: sha256
            )
        }

        NovaMLXLog.info("[ModelScope] Found \(files.count) downloadable files for \(repoId)")
        return files
    }

    // MARK: - Download URLs

    /// Returns the direct download URL for a file.
    /// Current pattern used by the site for raw content / resolve.
    public func resolveURL(repoId: String, filename: String, revision: String = "master") -> URL {
        let encodedRepo = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        let encodedFile = filename.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? filename
        return URL(string: "\(endpoint)/models/\(encodedRepo)/resolve/\(revision)/\(encodedFile)")!
    }

    /// Optional: future LFS pointer handling, signed URLs, etc. can be added here.
    public func needsSpecialHandling(_ file: ModelScopeFile) -> Bool {
        file.isLFS
    }
}

// MARK: - Supporting Types

public struct ModelScopeFile: Sendable {
    public let path: String
    public let name: String
    public let size: Int
    public let isLFS: Bool
    public let sha256: String?

    public var rfilename: String { path } // compatibility with existing HFFile shape
}

public enum ModelScopeError: Error, LocalizedError {
    case unexpectedFileListResponse

    public var errorDescription: String? {
        switch self {
        case .unexpectedFileListResponse:
            return "ModelScope returned an unexpected file list format. Check logs for raw response."
        }
    }
}
