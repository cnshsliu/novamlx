/// Versions reported in `hello`. `protocolVersion` is the WIRE CONTRACT —
/// the server closes the tunnel with 4005 upgrade-required when it is below
/// the server's minimum. `version` (app) is telemetry only.
public enum TknetPeer {
    public static let version = "0.2.0"

    /// Wire protocol this build speaks.
    public static let protocolVersion = 1

    /// Upgrade landing page — baked in because the server's error frame
    /// (min version + URL) is best-effort and can be lost before close.
    public static let downloadURL = "https://tknet.ai"
}
