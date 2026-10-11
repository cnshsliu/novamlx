import Foundation
import NovaMLXCore
import NovaMLXEngine
import NovaMLXTknetPeer

/// Window-owned page drafts. Plain classes on purpose: NovaAppView holds the
/// box in `@State` and does not observe it, so typing on one page does not
/// rebuild the sidebar or any other page.
final class PageMemoryBox {
    let models = ModelsPageMemory()
    let downloads = DownloadsPageMemory()
    let voice = VoiceClonePageMemory()
    let video = VideoSlicePageMemory()
    let talkingHead = TalkingHeadPageMemory()
    let requests = RequestLogPageMemory()
    let catalog = CatalogAdminPageMemory()
    let tknet = TknetPageMemory()
    let tokenhub = TokenhubPageMemory()
    let loadBalancers = LoadBalancerPageMemory()
    let apiKeys = APIKeysPageMemory()
    let cluster = ClusterPageMemory()
    let settings = SettingsPageMemory()
}

final class ModelsPageMemory {
    var typeFilter: ModelsPageView.ModelTypeFilter = .all
    var modelQuery = ""
    var scrollAnchor: String?
}

final class DownloadsPageMemory {
    var typeFilter: ModelsPageView.ModelTypeFilter = .all
    var searchText = ""
    var searchRegex = false
    var suffixMlxCommunity = false
    var searchMlxOnly = true
    var selectedMirrorOption = "official"
    var customMirrorURL = ""
    var mirrorTouched = false
    var newApiKey = ""
    var showApiKeyPrompt = false
    var scrollAnchor: String?
}

final class VoiceClonePageMemory {
    var touched = false
    var selectedId: UUID?
    var profileName = ""
    var transcript = ""
    var speakText = ""
    var renamingId: UUID?
    var renameDraft = ""
}

final class VideoSlicePageMemory {
    var referencePath = ""
    var referenceText = ""
    var sliceSeconds: Double = 90
    var removeFillers = true
    var titleStyle: TitleOverlayStyle = .news
    var pointStyle: PointOverlayStyle = .box
}

final class RequestLogPageMemory {
    var onlyErrors = false
    /// At most one request row is open. Nil means every row is collapsed.
    var expandedId: String?
    /// Empty means every model. Otherwise an exact `RequestLogEntry.model` from Recent Requests.
    var modelFilter = ""
    var scrollAnchor: String?
}

final class CatalogAdminPageMemory {
    var keep = false
    var query = ""
    var categoryFilter = ""
    var selectedId: String?
    var draft: CatalogDraft?
}

final class TknetPageMemory {
    var keep = false
    var selectedServer = TknetServers.official
    var peerNameText = ""
    var nameEdited = false
    var revealKey = false
    var loginEmail = ""
    var loginPassword = ""
}

final class TokenhubPageMemory {
    var keep = false
    var selectedId: String?
    var isCreatingNew = false
    var panel: TokenhubPageView.RightPanelMode = .empty
    var formName = ""
    var formEndpoint = ""
    var formApiKey = ""
    var formRemoteModel = ""
    var formEnabled = false
    var formIsFree = false
    var formSupportsResponses = false
    var formTags = ""
}

final class LoadBalancerPageMemory {
    var expandedId: UUID?
    var creating = false
    var editingId: UUID?
}

final class APIKeysPageMemory {
    var showCreateKeySheet = false
    var newKeyName = ""
    var newKeyRateLimit: Double?
    var newKeyRateBurst: Int?
    var newKeyMaxTokens: Int64?
    var newKeyMaxRequests: Int64?
    var newKeyResetPeriod: UsageResetPeriod = .daily
    var createdRawKey: String?
    var expandedKeyId: String?
}

final class ClusterPageMemory {
    var expandedNodeId: String?
    var workerSort: WorkerSortOrder = .joinTime
    var selectedActivationModel = ""
    var showCredentialDialog = false
    var credentialHost = ""
    var credentialUsername = ""
    var credentialPassword = ""
}

final class SettingsPageMemory {
    var captured = false
    var isEditing = false
    var cfgHost = "127.0.0.1"
    var cfgPort = 6590
    var cfgAdminPort = 6591
    var cfgApiKeys = ""
    var cfgMaxConcurrent = 16
    var cfgTimeout = 300
    var cfgMaxBodyMB = 100.0
    var cfgDefaultModel = ""
    var cfgMaxProcessMemory = "auto"
    var memoryLimitMode: MemoryLimitMode = .auto
    var memoryLimitPercent: Double = 70
    var memoryLimitValue: Double = 24
    var memoryLimitUnit = "GB"
    var cfgHasUnsavedChanges = false
    var clusterEnabled = false
    var clusterRole = "coordinator"
    var clusterHost = ""
    var clusterPort = 6591
    var clusterStrategy = "minNodes"
    var clusterMinLayers = 32
    var clusterTransport = "tcp"
    var tknetApiKey = ""
}
