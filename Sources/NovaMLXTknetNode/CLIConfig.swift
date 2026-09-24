import Foundation

/// Path handling shared by the `tknet-node` CLI and the module's tests.
/// Lives in the module (not the executable target) because SwiftPM test
/// targets cannot import executable targets.
public enum CLIConfig {
    public static let defaultConfigPath = "~/.config/tknet-node/node.json"

    /// Expands a leading `~`/`~/` to the current user's home directory.
    /// `~user/...` forms belong to other accounts and stay literal. All other
    /// paths pass through untouched (absolute, relative, or empty).
    public static func expandTilde(_ path: String) -> String {
        guard path == "~" || path.hasPrefix("~/") else { return path }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return path == "~" ? home : home + path.dropFirst()
    }
}
