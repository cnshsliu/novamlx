import Foundation

/// Global holder for the model ID currently being inferred.
/// Written by API handlers when inference starts and cleared when it ends.
/// Multiple concurrent requests overwrite each other — the UI shows whoever
/// wrote most recently.
public final class CurrentInferenceModel: @unchecked Sendable {
    public static let shared = CurrentInferenceModel()
    public var modelID: String?
    private init() {}
}
