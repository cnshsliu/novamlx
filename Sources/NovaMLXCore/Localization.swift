import Foundation
import SwiftUI

public enum AppLanguage: String, CaseIterable, Identifiable, Sendable {
    case en = "en"
    case zhHans = "zh-Hans"
    case zhHantHK = "zh-Hant-HK"
    case zhHantTW = "zh-Hant-TW"
    case ja = "ja"
    case ko = "ko"
    case fr = "fr"
    case de = "de"
    case ru = "ru"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .en: return "English"
        case .zhHans: return "简体中文"
        case .zhHantHK: return "繁體中文（香港）"
        case .zhHantTW: return "繁體中文（台灣）"
        case .ja: return "日本語"
        case .ko: return "한국어"
        case .fr: return "Français"
        case .de: return "Deutsch"
        case .ru: return "Русский"
        }
    }

    public static func detectFromSystem() -> AppLanguage {
        let preferred = Locale.preferredLanguages.first ?? "en"
        // Match to supported languages
        if preferred.hasPrefix("zh-Hans") { return .zhHans }
        if preferred.hasPrefix("zh-Hant") || preferred.hasPrefix("zh-HK") || preferred.hasPrefix("zh-MO") {
            return .zhHantHK
        }
        if preferred.hasPrefix("zh-TW") { return .zhHantTW }
        if preferred.hasPrefix("ja") { return .ja }
        if preferred.hasPrefix("ko") { return .ko }
        if preferred.hasPrefix("fr") { return .fr }
        if preferred.hasPrefix("de") { return .de }
        if preferred.hasPrefix("ru") { return .ru }
        return .en
    }
}

@MainActor
public final class L10n: ObservableObject {
    public static let shared = L10n()

    @Published public var currentLanguage: AppLanguage {
        didSet { persistLanguage() }
    }

    private let strings: [String: [String: String]] = LocalizationStrings.all

    private init() {
        // Priority: saved preference > OS language > English
        if let saved = Self.loadSavedLanguage(),
           let lang = AppLanguage(rawValue: saved) {
            currentLanguage = lang
        } else {
            currentLanguage = AppLanguage.detectFromSystem()
        }
    }

    public func tr(_ key: String) -> String {
        strings[currentLanguage.rawValue]?[key]
            ?? strings["en"]?[key]
            ?? key
    }

    public func tr(_ key: String, _ args: CVarArg...) -> String {
        let template = strings[currentLanguage.rawValue]?[key]
            ?? strings["en"]?[key]
            ?? key
        return String(format: template, arguments: args)
    }

    public func setLanguage(_ lang: AppLanguage) {
        currentLanguage = lang
    }

    // MARK: - Persistence

    private func persistLanguage() {
        let configURL = NovaMLXPaths.configFile

        guard var config = Self.readJSON(at: configURL) else { return }
        config["language"] = currentLanguage.rawValue

        if let data = try? JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: configURL, options: .atomic)
        }
    }

    private static func loadSavedLanguage() -> String? {
        let configURL = NovaMLXPaths.configFile
        guard let config = readJSON(at: configURL) else { return nil }
        return config["language"] as? String
    }

    private static func readJSON(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json
    }
}
