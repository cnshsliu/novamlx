import Foundation

/// Disk menu for the downloaded-model list: "all" plus each configured model root.
public enum ModelRootFilter {
    public struct Option: Equatable, Sendable, Identifiable {
        public let path: String
        public let label: String
        public var id: String { path }

        public init(path: String, label: String) {
            self.path = path
            self.label = label
        }
    }

    /// Empty selection means every root.
    public static let all = ""

    /// One option per root, in the given order. Labels prefer the external
    /// volume name (`/Volumes/Disk1/...` → `Disk1`); otherwise the root folder
    /// name. Shared labels gain the folder name in parentheses, then the full
    /// path if that still collides.
    public static func options(for roots: [URL]) -> [Option] {
        var seen = Set<String>()
        let unique: [URL] = roots.compactMap { root in
            let url = root.standardizedFileURL
            guard seen.insert(url.path).inserted else { return nil }
            return url
        }
        let base = unique.map { (path: $0.path, label: shortLabel($0)) }
        var counts: [String: Int] = [:]
        for item in base {
            counts[item.label, default: 0] += 1
        }
        let preliminary = base.map { item in
            let leaf = URL(fileURLWithPath: item.path).lastPathComponent
            let label: String
            if counts[item.label, default: 0] > 1, item.label != leaf, !leaf.isEmpty {
                label = "\(item.label) (\(leaf))"
            } else if counts[item.label, default: 0] > 1 {
                label = item.path
            } else {
                label = item.label
            }
            return Option(path: item.path, label: label)
        }
        var labelCounts: [String: Int] = [:]
        for option in preliminary {
            labelCounts[option.label, default: 0] += 1
        }
        return preliminary.map { option in
            guard labelCounts[option.label, default: 0] > 1 else { return option }
            return Option(path: option.path, label: option.path)
        }
    }

    /// Volume name for `/Volumes/<name>/...` (other than the boot volume).
    /// Otherwise the last path component.
    public static func shortLabel(_ root: URL) -> String {
        let url = root.standardizedFileURL
        let parts = url.pathComponents
        if parts.count > 2, parts[0] == "/", parts[1] == "Volumes" {
            let volume = parts[2]
            if !volume.isEmpty, !NovaMLXPaths.isBootVolumeName(volume) {
                return volume
            }
        }
        let leaf = url.lastPathComponent
        if leaf.isEmpty || leaf == "/" { return url.path }
        return leaf
    }

    /// True when `selectedRoot` is "all", or `modelURL` lives inside that root.
    /// `/Volumes/Disk1` does not match `/Volumes/Disk10`.
    public static func matches(modelURL: URL, selectedRoot: String) -> Bool {
        guard !selectedRoot.isEmpty else { return true }
        let model = modelURL.standardizedFileURL.path
        let root = URL(fileURLWithPath: selectedRoot).standardizedFileURL.path
        if model == root { return true }
        let prefix = root.hasSuffix("/") ? root : root + "/"
        return model.hasPrefix(prefix)
    }

    /// The configured root that contains `modelURL`. Nested roots keep the
    /// longest match so a model under `/Volumes/Disk/Models/extra` is not
    /// treated as living in `/Volumes/Disk/Models`.
    public static func containingRoot(of modelURL: URL, roots: [URL]) -> URL? {
        let model = modelURL.standardizedFileURL.path
        let matches = roots.map(\.standardizedFileURL).filter { root in
            let path = root.path
            if model == path { return true }
            let prefix = path.hasSuffix("/") ? path : path + "/"
            return model.hasPrefix(prefix)
        }
        return matches.max { $0.path.count < $1.path.count }
    }

    /// Other mounted, writable model roots a downloaded model can move to.
    /// The root that currently holds the files is excluded. An unknown
    /// location (not under any root) has nowhere to move from.
    public static func moveDestinations(
        modelURL: URL,
        statuses: [NovaMLXPaths.ModelDirStatus]
    ) -> [NovaMLXPaths.ModelDirStatus] {
        let roots = statuses.map { URL(fileURLWithPath: $0.path, isDirectory: true) }
        guard let source = containingRoot(of: modelURL, roots: roots) else { return [] }
        let sourcePath = source.standardizedFileURL.path
        return statuses.filter { status in
            guard status.mounted, status.writable else { return false }
            let path = URL(fileURLWithPath: status.path, isDirectory: true).standardizedFileURL.path
            return path != sourcePath
        }
    }
}
