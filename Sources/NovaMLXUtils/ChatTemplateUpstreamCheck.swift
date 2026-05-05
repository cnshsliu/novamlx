import Foundation
import NovaMLXCore
import CryptoKit

/// Optional upstream-fingerprint sanity check.
///
/// Compares the local chat template (the one that swift-transformers will
/// actually use for prompt rendering) with the canonical copy fetched from
/// HuggingFace's raw file API. Drift indicates one of:
///   • the local copy was corrupted by an earlier NovaMLX bug or manual edit
///   • the upstream repo updated its template (legitimate, no action needed)
///   • a mlx-community quant stripped/replaced the original template
///
/// This catches the case our reconciliation logic CAN'T catch: when
/// `tokenizer_config.json.chat_template` itself is wrong (so there's nothing
/// to compare it against locally). Example: Gemma-4 and Ling-2.6 quants
/// where the broken Bailing-style template ended up inside tokenizer_config.json.
///
/// **Off by default** because:
///   • requires network on every model load
///   • upstream may legitimately diverge (community quants strip templates)
///
/// Enable per-instance by setting `NOVAMLX_TEMPLATE_UPSTREAM_CHECK=1` in the
/// environment of the NovaMLX server. Results cached for 24h to amortize cost.
public actor ChatTemplateUpstreamCheck {
    public static let shared = ChatTemplateUpstreamCheck()

    private struct CacheEntry: Codable {
        let modelId: String
        let localSha256: String
        let upstreamSha256: String?
        let matched: Bool
        let checkedAt: Date
        let upstreamFetchFailed: Bool
    }

    private var cache: [String: CacheEntry] = [:]
    private let ttl: TimeInterval = 24 * 60 * 60

    private init() {}

    /// True when the env var is set to "1" or "true" (case-insensitive).
    public static var isEnabled: Bool {
        let raw = ProcessInfo.processInfo.environment["NOVAMLX_TEMPLATE_UPSTREAM_CHECK"]?.lowercased() ?? ""
        return raw == "1" || raw == "true" || raw == "yes"
    }

    /// Run the check and log a warning if drift is detected.
    /// Idempotent within `ttl` (24h) per modelId.
    public func check(modelId: String, localTemplate: String) async {
        guard Self.isEnabled else { return }
        guard !localTemplate.isEmpty else { return }
        guard !modelId.isEmpty else { return }

        // TTL gate
        if let cached = cache[modelId],
           Date().timeIntervalSince(cached.checkedAt) < ttl,
           !cached.upstreamFetchFailed {
            return
        }

        let localHash = sha256(localTemplate)

        // Fetch upstream raw template. Try chat_template.jinja first, then the
        // chat_template field embedded in tokenizer_config.json.
        let urlBase = "https://huggingface.co/\(modelId)/raw/main"
        async let jinja = fetchString("\(urlBase)/chat_template.jinja")
        async let tcText = fetchString("\(urlBase)/tokenizer_config.json")

        var upstreamTemplate: String? = nil
        if let j = await jinja, !j.isEmpty {
            upstreamTemplate = j
        } else if let tc = await tcText,
                  let data = tc.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let t = json["chat_template"] as? String, !t.isEmpty {
            upstreamTemplate = t
        }

        guard let up = upstreamTemplate else {
            cache[modelId] = CacheEntry(
                modelId: modelId,
                localSha256: localHash,
                upstreamSha256: nil,
                matched: false,
                checkedAt: Date(),
                upstreamFetchFailed: true
            )
            NovaMLXLog.info("[UpstreamCheck] \(modelId): could not fetch upstream template (model is not on HF, private, or offline) — skipping comparison")
            return
        }

        let upHash = sha256(up)
        let matched = localHash == upHash
        cache[modelId] = CacheEntry(
            modelId: modelId,
            localSha256: localHash,
            upstreamSha256: upHash,
            matched: matched,
            checkedAt: Date(),
            upstreamFetchFailed: false
        )

        if matched {
            NovaMLXLog.info("[UpstreamCheck] \(modelId): local chat template matches upstream HF copy ✓")
        } else {
            NovaMLXLog.warning("[UpstreamCheck] \(modelId): local chat template DIFFERS from upstream HF (local sha=\(localHash.prefix(12))…, upstream sha=\(upHash.prefix(12))…). May be intentional (quant strip, legitimate update) or corruption. Run `nova chat-template diagnose \(modelId)` to inspect.")
        }
    }

    // MARK: - HTTP

    private func fetchString(_ urlString: String) async -> String? {
        guard let url = URL(string: urlString) else { return nil }
        var req = URLRequest(url: url, timeoutInterval: 8)
        req.httpMethod = "GET"
        req.setValue("NovaMLX/upstream-check", forHTTPHeaderField: "User-Agent")
        do {
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                return nil
            }
            return String(data: data, encoding: .utf8)
        } catch {
            return nil
        }
    }

    // MARK: - Hashing

    private func sha256(_ text: String) -> String {
        let bytes = Data(text.utf8)
        let digest = SHA256.hash(data: bytes)
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}
