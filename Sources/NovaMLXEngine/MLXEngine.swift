import Foundation
import CoreImage
import MLX
import MLXNN
import MLXRandom
import MLXLLM
import MLXLMCommon
import MLXVLM
import NovaMLXCore
import NovaMLXUtils
import NovaMLXPrefixCache

public final class ModelContainer: @unchecked Sendable {
    public let identifier: ModelIdentifier
    public var config: ModelConfig
    public private(set) var mlxContainer: MLXLMCommon.ModelContainer?
    public private(set) var tokenizer: Tokenizer?
    public private(set) var isLoaded: Bool
    public private(set) var wiredReservationTicket: WiredMemoryTicket?
    public var kvMemoryOverride: Int?
    /// Control tokens extracted from tokenizer.json + chat_template (e.g. "<|turn>", "<|im_end|>")
    public private(set) var controlTokens: [String] = []
    /// Whether this model uses thinking/reasoning tags (detected from chat_template)
    public private(set) var isThinkingModel: Bool = false
    /// Per-model-family chat template processor for control token handling
    public private(set) var chatTemplateProcessor: ChatTemplateProcessor?

    public init(identifier: ModelIdentifier, config: ModelConfig) {
        self.identifier = identifier
        self.config = config
        self.isLoaded = false
    }

    public func setLoaded(mlxContainer: MLXLMCommon.ModelContainer, tokenizer: Tokenizer) {
        self.mlxContainer = mlxContainer
        self.tokenizer = tokenizer
        self.isLoaded = true

        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(identifier.id)
        let chatTemplate = Self.loadChatTemplate(modelDir: modelDir)
        let addedTokens = Self.loadAddedTokens(modelDir: modelDir)

        // Select processor based on family + detected template format
        let processor = ChatTemplateProcessorRegistry.processor(
            family: identifier.family, chatTemplate: chatTemplate
        )
        self.chatTemplateProcessor = processor

        // Extract and refine control tokens via processor
        self.controlTokens = Self.extractControlTokensWithProcessor(
            processor: processor, chatTemplate: chatTemplate, addedTokens: addedTokens
        )
        self.isThinkingModel = processor.isThinkingModel(chatTemplate: chatTemplate, addedTokens: addedTokens)

        if !controlTokens.isEmpty {
            NovaMLXLog.info("[ControlTokens] \(identifier.displayName): \(controlTokens)")
        }
        NovaMLXLog.info("Model loaded: \(identifier.displayName)")
    }

    public func setWiredReservation(_ ticket: WiredMemoryTicket) {
        self.wiredReservationTicket = ticket
    }

    public func unload() {
        mlxContainer = nil
        tokenizer = nil
        isLoaded = false
        wiredReservationTicket = nil
        controlTokens = []
        NovaMLXLog.info("Model unloaded: \(identifier.displayName)")
    }

    // MARK: - Control Token Extraction

    /// Extract control/special tokens from tokenizer.json added_tokens + chat_template
    public static func extractControlTokens(for modelId: String) -> [String] {
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        var tokens = Set<String>()

        // 1. From tokenizer.json added_tokens where special=true
        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.json")
        if let data = try? Data(contentsOf: tokenizerPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let added = json["added_tokens"] as? [[String: Any]] {
            for entry in added {
                if entry["special"] as? Bool == true,
                   let content = entry["content"] as? String,
                   content.hasPrefix("<") {
                    tokens.insert(content)
                }
            }
            // Also pick up <|xxx|> pattern tokens marked non-special (e.g. Qwen3.6 tool_call)
            // These are protocol-level tokens the model shouldn't emit outside tool-use mode
            for entry in added {
                if entry["special"] as? Bool == false,
                   let content = entry["content"] as? String,
                   content.hasPrefix("<|") && (content.hasSuffix("|>") || content.hasSuffix(">")) {
                    tokens.insert(content)
                }
            }
        }

        // 2. From chat_template: extract all <|xxx> and <|xxx|> patterns
        let template = loadChatTemplate(modelDir: modelDir)
        if let template {
            // Match <|xxx>, <|xxx|>, and <|/xxx> patterns
            if let regex = try? NSRegularExpression(pattern: "<\\|[a-zA-Z_/][a-zA-Z0-9_/]*\\|?>") {
                let nsRange = NSRange(template.startIndex..., in: template)
                for match in regex.matches(in: template, range: nsRange) {
                    if let range = Range(match.range, in: template) {
                        tokens.insert(String(template[range]))
                    }
                }
            }
        }

        // Generate close-tag variants: <|turn|> → <|/turn|>, <|/turn>
        // Models like Qwen3.6 output <|/turn> as close tags for <|turn|> open tags
        var closeVariants = Set<String>()
        for token in tokens {
            guard token.hasPrefix("<|") && !token.contains("/") else { continue }
            let inner: String
            if token.hasSuffix("|>") {
                inner = String(token.dropFirst(2).dropLast(2))
            } else if token.hasSuffix(">") {
                inner = String(token.dropFirst(2).dropLast(1))
            } else { continue }
            closeVariants.insert("<|/\(inner)|>")
            closeVariants.insert("<|/\(inner)>")
        }
        tokens.formUnion(closeVariants)

        // Refine turn markers: if template uses <|turn>role format (Qwen3.6 style),
        // bare <|turn> / <|turn|> are role prefixes, NOT end-of-turn signals.
        // Replace them with <|turn>user to only stop at the start of a new user turn,
        // allowing the model to generate <|turn>model (response turn) freely.
        if let template, template.contains("<|turn>") {
            let turnPrefixes: Set<String> = ["<|turn>", "<|turn|>"]
            if tokens.intersection(turnPrefixes).isEmpty == false {
                tokens.subtract(turnPrefixes)
                tokens.subtract(["<|/turn>", "<|/turn|>"])
                tokens.insert("<|turn>user")
            }
        }

        // Remove EOS/BOS/PAD/UNK — those are structural, not output separators
        tokens.remove("")

        // Harmony format (GPT-OSS): structural tokens are part of output format.
        // Only <|return|> should be a stop signal. <|channel|>, <|message|>, <|end|>, etc.
        // are generated by the model as part of its multi-channel output format.
        if let template, template.contains("<|channel|>") {
            tokens.subtract([
                "<|end|>", "<|channel|>", "<|message|>",
                "<|call|>", "<|constrain|>",
                "<|/end|>", "<|/channel|>", "<|/message|>",
                "<|/call|>", "<|/constrain|>",
            ])
        }

        // Exclude semantic content tags — parsed by ThinkingParser, not stream delimiters.
        // These mark thinking/response boundaries and carry meaning for the application
        // layer. They must pass through unfiltered so the frontend can parse them.
        //
        // Control tokens (<|turn|>, <|im_end|>, etc.) are protocol-level separators:
        // they trigger generation stop (TurnStopProcessor) and must NOT appear in output.
        // Semantic tags are the opposite: they SHOULD appear in output.
        //
        // Keep in sync with ThinkingParser.swift supported tag patterns.
        tokens.subtract([
            "<think", "</think",             // DeepSeek-R1, Qwen-thinking
            "<thinking", "</thinking",       // alternative format (Claude, some OS models)
            "<|begin_of_thought|>", "<|end_of_thought|>",   // Qwen3.6+
            "<|begin_of_think|>", "<|end_of_think|>",       // variant
        ])

        return tokens.sorted()
    }

    /// Extract control tokens using the per-family processor for refinement.
    private static func extractControlTokensWithProcessor(
        processor: ChatTemplateProcessor,
        chatTemplate: String?,
        addedTokens: [[String: Any]]
    ) -> [String] {
        var rawTokens = Set<String>()
        var templateTokens = Set<String>()

        // From tokenizer.json added_tokens
        for entry in addedTokens {
            if entry["special"] as? Bool == true,
               let content = entry["content"] as? String,
               content.hasPrefix("<") {
                rawTokens.insert(content)
            }
            if entry["special"] as? Bool == false,
               let content = entry["content"] as? String,
               content.hasPrefix("<|") && (content.hasSuffix("|>") || content.hasSuffix(">")) {
                rawTokens.insert(content)
            }
        }

        // From chat_template
        if let template = chatTemplate {
            templateTokens = SharedControlTokenLogic.extractTemplateTokens(from: template)
        }

        return processor.refineControlTokens(
            rawTokens: rawTokens,
            templateTokens: templateTokens,
            chatTemplate: chatTemplate,
            addedTokens: addedTokens
        )
    }

    /// Load added_tokens from tokenizer.json.
    private static func loadAddedTokens(modelDir: URL) -> [[String: Any]] {
        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.json")
        guard let data = try? Data(contentsOf: tokenizerPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let added = json["added_tokens"] as? [[String: Any]] else {
            return []
        }
        return added
    }

    /// Detect if model uses thinking/reasoning tags by checking chat_template
    /// and tokenizer added_tokens (some models like Qwen3.6 use <|begin_of_thought|>
    /// in the tokenizer but not in the chat template).
    public static func detectThinkingModel(for modelId: String) -> Bool {
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        let template = loadChatTemplate(modelDir: modelDir) ?? ""
        if template.contains("<think") || template.contains("</think")
            || template.contains("<thinking") || template.contains("</thinking") {
            return true
        }
        // Fallback: check tokenizer added_tokens for thinking markers
        // (Qwen3.6 uses <|begin_of_thought|> which is not in the chat template)
        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.json")
        if let data = try? Data(contentsOf: tokenizerPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let added = json["added_tokens"] as? [[String: Any]] {
            for entry in added {
                if let content = entry["content"] as? String {
                    if content.contains("begin_of_thought") || content.contains("end_of_thought")
                        || content.contains("begin_of_think") || content.contains("end_of_think") {
                        return true
                    }
                }
            }
        }
        return false
    }

    /// Returns true when the chat template injects an opening thinking tag
    /// (`<think>\n` or similar) into the assistant generation prompt — meaning
    /// the model only generates the CLOSE tag (`</think>`).
    ///
    /// **Implicit (returns true)** — chat template literally contains
    /// `<think>\n` (or the `<thinking>` variant) inside the
    /// `add_generation_prompt` block. Model output stream looks like:
    /// `…thinking content…</think>…response…`. No opening tag in the model's
    /// output. Examples: real `mlx-community/Qwen3.6-*-4bit` quants (template
    /// emits `<think>\n` after `<|im_start|>assistant\n`); DeepSeek-R1.
    ///
    /// **Explicit (returns false)** — chat template injects no opening tag,
    /// OR the model's tokenizer carries the `<|begin_of_thought|>` /
    /// `<|end_of_thought|>` family-specific markers (in which case the model
    /// is expected to emit BOTH `<|begin_of_thought|>` and `<|end_of_thought|>`
    /// in its output stream — Qwen3.6 explicit-marker variants).
    ///
    /// **Why we don't classify by added_tokens shape alone:** `<think>` /
    /// `</think>` are routinely registered as added_tokens (so they decode
    /// atomically) on IMPLICIT models too — the chat template still injects
    /// the open tag as a prompt prefix. The presence of `<think>` as a token
    /// is a NECESSARY but NOT SUFFICIENT condition for explicit mode. The
    /// authoritative signal is the chat template, not the tokenizer.
    ///
    /// Pre-2026-05-05 implementation incorrectly treated `<think>` in
    /// `added_tokens` as proof of explicit mode, mis-classifying real-world
    /// `mlx-community/Qwen3.6-*-4bit` quants and routing all pre-close output
    /// to `responseContent` rather than `reasoning_content`.
    public static func isImplicitThinkingModel(for modelId: String) -> Bool {
        let modelDir = NovaMLXPaths.modelsDir.appendingPathComponent(modelId)
        let template = loadChatTemplate(modelDir: modelDir) ?? ""

        // Step 1: explicit-marker family-specific tokens override everything.
        // If the tokenizer registers `<|begin_of_thought|>` / `<|end_of_thought|>`
        // (Qwen3.6 explicit variant), the model emits both markers. NOT implicit.
        let tokenizerPath = modelDir.appendingPathComponent("tokenizer.json")
        if let data = try? Data(contentsOf: tokenizerPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let added = json["added_tokens"] as? [[String: Any]] {
            for entry in added {
                guard let content = entry["content"] as? String else { continue }
                if content.contains("begin_of_thought") || content.contains("end_of_thought")
                    || content.contains("begin_of_think") || content.contains("end_of_think") {
                    NovaMLXLog.info("[isImplicitThinkingModel] \(modelId): explicit family marker token \(content.prefix(40)) → false")
                    return false
                }
            }
        } else {
            NovaMLXLog.warning("[isImplicitThinkingModel] \(modelId): failed to load tokenizer.json from \(tokenizerPath.path)")
        }

        // Step 2: chat-template-driven detection. The template MUST contain
        // an opening `<think` / `<thinking` tag literal AND that literal must
        // be inside the `add_generation_prompt` block (i.e. emitted as part
        // of the prompt prefix, not part of message rendering).
        //
        // **Two byte-shapes to handle:**
        //  1. Real-world `tokenizer_config.json.chat_template` Jinja source —
        //     the `\n` after `<think>` is a Jinja literal escape sequence
        //     (preserved as 2 bytes: `\\` + `n`) until rendering time. So the
        //     template string contains the 11-byte sequence `'<think>\n'`
        //     where `\n` is BACKSLASH-N. Match: `"'<think>\\n'"`.
        //  2. Pre-rendered / fixture templates — `\n` is already a literal
        //     newline (1 byte `0x0A`). Match: `"'<think>"` followed by `'\n'`.
        //
        // Also accept double-quoted Jinja literals and the `<thinking>` family.
        // If `<think>` (with closing `>`) appears as a quoted Jinja literal in
        // the template AND no later `</think` close in the same string mode,
        // it's almost certainly an injection. We accept the simpler "quoted
        // open tag" check below as a sufficient proxy.
        let injectionLiterals: [String] = [
            "'<think>\\n'",     // Qwen3 / Qwen3.5 / Qwen3.6 mlx-community
            "\"<think>\\n\"",
            "'<thinking>\\n'",
            "\"<thinking>\\n\"",
            "'<think>'",        // some templates omit trailing \n
            "\"<think>\"",
            "'<thinking>'",
            "\"<thinking>\"",
        ]
        var hasInjectedOpen = injectionLiterals.contains(where: { template.contains($0) })
        // Fixture / pre-rendered shape: actual newline after `<think>`.
        if !hasInjectedOpen {
            // Check for `<think>` followed by an actual newline byte. Use
            // String.range(of:) for clarity over manual byte-walking. This
            // handles the test-fixture style (Swift triple-quoted literal
            // where `\n` is interpreted at parse time).
            if template.contains("<think>\n") || template.contains("<thinking>\n") {
                hasInjectedOpen = true
            }
        }

        if hasInjectedOpen {
            NovaMLXLog.info("[isImplicitThinkingModel] \(modelId): template injects opening think tag → true (implicit)")
            return true
        }

        // Step 3: fallback — template references thinking tags but doesn't
        // appear to inject. Treat as explicit.
        let templateMentionsThink = template.contains("<think") || template.contains("<thinking")
        if templateMentionsThink {
            NovaMLXLog.info("[isImplicitThinkingModel] \(modelId): template mentions think tag but no injection literal → false (explicit)")
            return false
        }

        // Step 4: no thinking format detected — not a thinking model.
        NovaMLXLog.info("[isImplicitThinkingModel] \(modelId): no thinking markers found → false")
        return false
    }

    private static func loadChatTemplate(modelDir: URL) -> String? {
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        if let data = try? Data(contentsOf: tcPath),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let template = json["chat_template"] as? String {
            return template
        }
        let jinjaPath = modelDir.appendingPathComponent("chat_template.jinja")
        return try? String(contentsOf: jinjaPath, encoding: .utf8)
    }
}

public struct Tokenizer: Sendable {
    public let encode: @Sendable (String) -> [Int]
    public let decode: @Sendable ([Int]) -> String
    public let eosToken: String?
    public let eosTokenId: Int?

    public init(
        encode: @Sendable @escaping (String) -> [Int],
        decode: @Sendable @escaping ([Int]) -> String,
        eosToken: String?,
        eosTokenId: Int?
    ) {
        self.encode = encode
        self.decode = decode
        self.eosToken = eosToken
        self.eosTokenId = eosTokenId
    }
}

public final class MLXEngine: InferenceEngineProtocol, @unchecked Sendable {
    public private(set) var isReady: Bool

    public let pool: EnginePool
    public let metricsStore: MetricsStore
    public let sessionManager: ChatSessionManager
    public let turboQuantService: TurboQuantService
    public let adapterService: AdapterService
    public let specDecoder: SpeculativeDecoder
    public let fusedBatchDecoder: FusedBatchDecoder
    public let budgetTracker: MemoryBudgetTracker
    public let memoryEnforcer: ProcessMemoryEnforcer?
    public lazy var batchScheduler = BatchScheduler(engine: self)
    public var settingsProvider: (@Sendable (String) -> ModelSettings?)?
    private var prefixCacheManagers: [String: PrefixCacheManager]
    /// Operational kill switch mirrored from `ServerConfig.prefixCacheEnabled`.
    /// Read inside `lock.withLock` from `getOrCreatePrefixCacheManager`. Default
    /// `true` matches the historical behavior; engine sync from configuration
    /// happens via `setPrefixCacheEnabled(_:)`.
    private var prefixCacheEnabled: Bool = true
    /// Per-model cache of `TokenMaskBuilder` instances. Built on first
    /// structured-output request; invalidated on model unload. Avoids the
    /// per-request cost of decoding every token in the vocabulary (250k+
    /// tokens for Qwen3.6 / Gemma-4).
    public let tokenMaskBuilderCache = TokenMaskBuilderCache()
    private var activeCount: Int
    private var activeTasks: [UUID: Task<Void, Never>]
    private let lock = NovaMLXLock()

    private var totalRequests: UInt64
    private var totalTokensGenerated: UInt64
    private var totalInferenceTime: TimeInterval

    private var deferredClearCounter: Int = 0
    private let deferredClearThreshold: Int = 2
    private let wiredPolicy = MLXLMCommon.WiredSumPolicy()

    public init(kvCacheCapacity: Int = 1024, maxMemoryMB: UInt64 = 2048, maxConcurrent: Int = 8, metricsStore: MetricsStore? = nil, maxProcessMemory: String = "auto") {
        // Safety net: physMem - 2GB for MLX allocator. Our ProcessMemoryEnforcer
        // runs at a lower soft limit and evicts proactively. This is only hit
        // if our enforcement layer fails.
        let physMem = ProcessInfo.processInfo.physicalMemory
        let safetyReserve: UInt64 = 2 * 1024 * 1024 * 1024
        let mlxLimitBytes = physMem > safetyReserve ? physMem - safetyReserve : physMem / 2
        MLX.Memory.memoryLimit = Int(mlxLimitBytes)
        MLX.Memory.cacheLimit = 512 * 1024 * 1024  // 512MB compute cache cap

        self.isReady = false
        self.pool = EnginePool(maxMemoryMB: maxMemoryMB)
        self.metricsStore = metricsStore ?? MetricsStore(baseDirectory: NovaMLXPaths.baseDir)
        self.sessionManager = ChatSessionManager()
        self.turboQuantService = TurboQuantService()
        self.adapterService = AdapterService()
        self.specDecoder = SpeculativeDecoder()
        self.fusedBatchDecoder = FusedBatchDecoder()
        self.budgetTracker = MemoryBudgetTracker(gpuLimitBytes: mlxLimitBytes)

        // ProcessMemoryEnforcer — configurable soft/hard limit with 1s polling
        let limit = ProcessMemoryLimit.parse(maxProcessMemory)
        self.memoryEnforcer = ProcessMemoryEnforcer(
            pool: pool,
            settingsProvider: nil, // set later via configureSettingsProvider
            limit: limit,
            physicalRAM: physMem
        )

        self.prefixCacheManagers = [:]
        self.activeCount = 0
        self.activeTasks = [:]
        self.totalRequests = 0
        self.totalTokensGenerated = 0
        self.totalInferenceTime = 0
        FusedSDPARegistration.register()
        NovaMLXLog.info("MLX Engine initialized (maxMemory: \(maxMemoryMB)MB, mlxLimit: \(mlxLimitBytes / 1024 / 1024)MB)")
    }

    static let tokenizerLoader = LocalTokenizerLoader()

    /// Ensure the model directory has a usable chat template, and that
    /// `tokenizer_config.json.chat_template` (canonical per HuggingFace) and any
    /// standalone `chat_template.jinja` are consistent.
    ///
    /// Background — swift-transformers (Hub.swift) prefers `chat_template.jinja`
    /// over `tokenizer_config.json.chat_template` when both exist; the .jinja
    /// content silently overwrites the config field. If the .jinja is stale or
    /// from a different family, the model is prompted in the wrong format and
    /// emits tokens that look like control-token leakage and turn-impersonation
    /// hallucination but are actually faithful continuation of the wrong format.
    ///
    /// Behavior here:
    ///   • If `ChatTemplateLibrary` has a maintained template for this model,
    ///     write it into `tokenizer_config.json.chat_template` AND remove any
    ///     stale `chat_template.jinja` so swift-transformers doesn't override it.
    ///   • If both `tokenizer_config.json.chat_template` and `chat_template.jinja`
    ///     exist but disagree, log a WARN and quarantine the .jinja file
    ///     (renamed `.jinja.disagreed-with-tokenizer_config`) — the canonical
    ///     `tokenizer_config.json` value wins.
    ///   • If neither source has a usable template, log a clear error. We
    ///     intentionally do NOT inject a generic fallback — guessing a chat
    ///     format for an arbitrary model corrupts inference more often than not.
    private static func ensureChatTemplate(modelDir: URL, modelId: String, family: ModelFamily, architecture: String? = nil) {
        let fm = FileManager.default
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        let jinjaPath = modelDir.appendingPathComponent("chat_template.jinja")
        let jsonPath = modelDir.appendingPathComponent("chat_template.json")
        let backupPath = modelDir.appendingPathComponent("chat_template.jinja.disabled-by-novamlx")

        // One-shot cleanup notice — if we have a pre-existing quarantine backup
        // and the current config is healthy (tokenizer_config.json has a
        // non-empty template, no live .jinja conflict), log a single info line
        // suggesting the user can delete the backup. We don't auto-delete:
        // the backup may be the user's only record of what the corrupt
        // template looked like.
        if fm.fileExists(atPath: backupPath.path) && !fm.fileExists(atPath: jinjaPath.path) {
            if let data = try? Data(contentsOf: tcPath),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let t = json["chat_template"] as? String, !t.isEmpty {
                NovaMLXLog.info("[ChatTemplate] \(modelId): pre-existing quarantine backup at \(backupPath.lastPathComponent); current tokenizer_config.json template (\(t.count) chars) is being used. The backup can be safely deleted if you no longer need it for inspection.")
            }
        }

        // 1. Read whatever templates currently exist on disk.
        let configTemplate = readConfigChatTemplate(at: tcPath)
        let jinjaTemplate: String? = fm.fileExists(atPath: jinjaPath.path)
            ? (try? String(contentsOf: jinjaPath, encoding: .utf8)).flatMap { $0.isEmpty ? nil : $0 }
            : nil

        // 2. Apply NovaMLX-maintained override if ChatTemplateLibrary has one.
        //    The override is the source of truth: write it to tokenizer_config and
        //    remove any .jinja file (which swift-transformers would otherwise prefer).
        let baselineForResolve = configTemplate ?? jinjaTemplate
        if let resolved = ChatTemplateLibrary.resolve(
            modelId: modelId,
            family: family,
            architecture: architecture,
            downloadedTemplate: baselineForResolve
        ) {
            if resolved != configTemplate {
                injectTemplate(resolved, into: modelDir, modelId: modelId)
            }
            quarantineJinjaIfPresent(jinjaPath: jinjaPath, reason: "NovaMLX override applied", modelId: modelId)
            return
        }

        // 3. No NovaMLX override — reconcile what's already on disk.
        switch (configTemplate, jinjaTemplate) {
        case let (.some(cfg), .some(jinja)) where cfg != jinja:
            // Disagreement. tokenizer_config.json is canonical per HF spec.
            // Quarantine the .jinja so swift-transformers stops overriding.
            NovaMLXLog.warning("[ChatTemplate] \(modelId): tokenizer_config.json.chat_template (\(cfg.count) chars) and chat_template.jinja (\(jinja.count) chars) DISAGREE — keeping config, quarantining .jinja (swift-transformers would otherwise prefer the .jinja file)")
            quarantineJinjaIfPresent(jinjaPath: jinjaPath, reason: "disagreed-with-tokenizer_config", modelId: modelId)

        case (.none, .some(let jinja)):
            // Promote .jinja content into tokenizer_config.json so it's the canonical source,
            // then remove the .jinja file so future updates don't drift.
            promoteJinjaToConfig(jinjaContent: jinja, tcPath: tcPath, modelId: modelId)
            quarantineJinjaIfPresent(jinjaPath: jinjaPath, reason: "promoted-to-tokenizer_config", modelId: modelId)

        case (.some, _):
            // tokenizer_config has a valid template; ensure no stale .jinja overrides it.
            if let jinja = jinjaTemplate, jinja == configTemplate {
                // Identical content — fine, no action.
                _ = jinja
            }

        case (.none, .none):
            // Last resort: chat_template.json (rare).
            if fm.fileExists(atPath: jsonPath.path) {
                NovaMLXLog.info("[ChatTemplate] \(modelId): using chat_template.json (no tokenizer_config.json template)")
                return
            }
            // No usable template anywhere. Do NOT inject a guessed fallback.
            NovaMLXLog.error("[ChatTemplate] \(modelId): no chat template found in tokenizer_config.json, chat_template.jinja, or chat_template.json — chat requests will fail. Re-download the model or supply a maintained template via ChatTemplateLibrary.")
        }
    }

    /// Read `chat_template` from `tokenizer_config.json` if present and non-empty.
    private static func readConfigChatTemplate(at tcPath: URL) -> String? {
        guard let data = try? Data(contentsOf: tcPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let str = json["chat_template"] as? String, !str.isEmpty { return str }
        // Some configs use an array form: [{"name": "default", "template": "..."}]
        if let arr = json["chat_template"] as? [[String: Any]] {
            for entry in arr {
                if entry["name"] as? String == "default", let t = entry["template"] as? String, !t.isEmpty {
                    return t
                }
            }
        }
        return nil
    }

    /// Inject a resolved template into tokenizer_config.json so the tokenizer picks it up.
    private static func injectTemplate(_ template: String, into modelDir: URL, modelId: String) {
        let tcPath = modelDir.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: tcPath),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            NovaMLXLog.warning("[ChatTemplate] \(modelId): no tokenizer_config.json to inject into")
            return
        }
        json["chat_template"] = template
        guard let newData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) else { return }
        do {
            try newData.write(to: tcPath, options: .atomic)
            NovaMLXLog.info("[ChatTemplate] \(modelId): wrote maintained template to tokenizer_config.json (\(template.count) chars)")
        } catch {
            NovaMLXLog.warning("[ChatTemplate] \(modelId): failed to write tokenizer_config.json: \(error)")
        }
    }

    /// Move a chat_template.jinja file out of the way so swift-transformers
    /// stops loading it. The file is renamed (not deleted) so the user can
    /// inspect or restore it manually if needed.
    private static func quarantineJinjaIfPresent(jinjaPath: URL, reason: String, modelId: String) {
        let fm = FileManager.default
        guard fm.fileExists(atPath: jinjaPath.path) else { return }
        let backup = jinjaPath.appendingPathExtension("disabled-by-novamlx")
        // If a previous quarantine file exists, leave it alone — never overwrite.
        guard !fm.fileExists(atPath: backup.path) else {
            // Quietly remove the live .jinja since we already have a backup; future
            // model updates would otherwise re-trigger the same cycle.
            try? fm.removeItem(at: jinjaPath)
            NovaMLXLog.info("[ChatTemplate] \(modelId): removed chat_template.jinja (existing backup at \(backup.lastPathComponent), reason=\(reason))")
            return
        }
        do {
            try fm.moveItem(at: jinjaPath, to: backup)
            NovaMLXLog.info("[ChatTemplate] \(modelId): quarantined chat_template.jinja → \(backup.lastPathComponent) (reason=\(reason))")
        } catch {
            NovaMLXLog.warning("[ChatTemplate] \(modelId): failed to quarantine chat_template.jinja: \(error)")
        }
    }

    /// Copy chat_template.jinja content into tokenizer_config.json's
    /// `chat_template` field so it becomes the canonical (HF-compliant) source.
    private static func promoteJinjaToConfig(jinjaContent: String, tcPath: URL, modelId: String) {
        guard let data = try? Data(contentsOf: tcPath),
              var json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            // No tokenizer_config.json — create one carrying just the template.
            let minimal: [String: Any] = ["chat_template": jinjaContent]
            if let newData = try? JSONSerialization.data(withJSONObject: minimal, options: [.prettyPrinted, .sortedKeys]) {
                try? newData.write(to: tcPath, options: .atomic)
                NovaMLXLog.info("[ChatTemplate] \(modelId): created tokenizer_config.json with template promoted from .jinja (\(jinjaContent.count) chars)")
            }
            return
        }
        json["chat_template"] = jinjaContent
        if let newData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]) {
            do {
                try newData.write(to: tcPath, options: .atomic)
                NovaMLXLog.info("[ChatTemplate] \(modelId): promoted chat_template.jinja into tokenizer_config.json (\(jinjaContent.count) chars)")
            } catch {
                NovaMLXLog.warning("[ChatTemplate] \(modelId): failed to promote .jinja into tokenizer_config.json: \(error)")
            }
        }
    }

    public func loadModel(from url: URL, config: ModelConfig) async throws -> ModelContainer {
        await CustomModelRegistration.ensureRegistered()
        let container = ModelContainer(identifier: config.identifier, config: config)
        NovaMLXLog.info("Loading model from: \(url.path)")

        // Ensure chat template exists — models without it will crash on inference.
        // Check tokenizer_config.json, chat_template.jinja, chat_template.json in order.
        let configJson = (try? Data(contentsOf: url.appendingPathComponent("config.json")))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        let arch = (configJson?["architectures"] as? [String])?.first
        Self.ensureChatTemplate(modelDir: url, modelId: config.identifier.id, family: config.identifier.family, architecture: arch)

        // --- Pre-load memory gate ---
        // Check ProcessMemoryEnforcer soft limit BEFORE loading weights.
        // If we can't fit the new model under the soft limit, evict LRU unpinned
        // models to make room. If that's not enough, fail fast with a clear error
        // instead of loading weights and immediately triggering enforcer eviction.
        if let enforcer = memoryEnforcer {
            let enforcerStatus = await enforcer.status
            if enforcerStatus.enabled && enforcerStatus.softLimitBytes > 0 {
                if let estimatedBytes = Self.estimateModelWeightSize(at: url) {
                    // Peak memory during eval(): model weights + Metal buffer materialization.
                    // Staged eval (for models >30GB) keeps peak to ~model_size + one batch.
                    // Without staged eval, peak can reach ~1.3x.
                    let safetyMargin: UInt64
                    if estimatedBytes > 30 * 1_073_741_824 {
                        // Staged eval: peak ≈ weight_size + 5GB batch overhead
                        safetyMargin = 5 * 1_073_741_824
                    } else if estimatedBytes > 20 * 1_073_741_824 {
                        safetyMargin = UInt64(Double(estimatedBytes) * 0.3)  // 30% for medium models
                    } else {
                        safetyMargin = UInt64(Double(estimatedBytes) * 0.2)  // 20% for small models
                    }
                    let neededBytes = estimatedBytes + safetyMargin

                    // Also check against Metal's recommended working set size.
                    // If peak would exceed it, fail fast with a clear message.
                    if let maxGPU = MLX.GPU.maxRecommendedWorkingSetBytes(), maxGPU > 0,
                       neededBytes > UInt64(maxGPU) {
                        let neededMB = neededBytes / 1_048_576
                        let maxMB = UInt64(maxGPU) / 1_048_576
                        NovaMLXLog.error("[MemoryGate] Model peak (\(neededMB)MB) exceeds Metal working set (\(maxMB)MB) for \(config.identifier.displayName)")
                        throw NovaMLXError.insufficientMemory(neededMB: neededMB, availableMB: maxMB, modelId: config.identifier.displayName)
                    }

                    let currentBytes = UInt64(MLX.Memory.activeMemory)
                    if currentBytes + neededBytes > enforcerStatus.softLimitBytes {
                        NovaMLXLog.info("[MemoryGate] Need \(neededBytes / 1_048_576)MB, have \(currentBytes / 1_048_576)MB free of \(enforcerStatus.softLimitBytes / 1_048_576)MB limit — attempting LRU eviction")
                        let ok = await ensureMemoryHeadroom(neededBytes: neededBytes, softLimitBytes: enforcerStatus.softLimitBytes)
                        if !ok {
                            let afterCurrent = UInt64(MLX.Memory.activeMemory)
                            let availableMB = enforcerStatus.softLimitBytes > afterCurrent ? (enforcerStatus.softLimitBytes - afterCurrent) / 1_048_576 : 0
                            let neededMB = neededBytes / 1_048_576
                            NovaMLXLog.error("[MemoryGate] Insufficient memory for \(config.identifier.displayName): need \(neededMB)MB, available \(availableMB)MB")
                            throw NovaMLXError.insufficientMemory(neededMB: neededMB, availableMB: availableMB, modelId: config.identifier.displayName)
                        }
                        NovaMLXLog.info("[MemoryGate] Headroom secured after LRU eviction")
                    }
                }
            }
        }
        // --- End pre-load memory gate ---

        // VLM models must be loaded via VLMModelFactory — they need vision tower
        // and VLM-specific prepare()/callAsFunction implementations
        let isVLM = config.modelType == .vlm
        let mlxContainer: MLXLMCommon.ModelContainer
        if isVLM {
            mlxContainer = try await VLMModelFactory.shared.loadContainer(
                from: url,
                using: Self.tokenizerLoader
            )
        } else {
            mlxContainer = try await LLMModelFactory.shared.loadContainer(
                from: url,
                using: Self.tokenizerLoader
            )
        }

        let mlxTokenizer = await mlxContainer.tokenizer
        let eosString = mlxTokenizer.eosToken
        let tokenizer = Tokenizer(
            encode: { text in mlxTokenizer.encode(text: text) },
            decode: { tokens in
                let result = mlxTokenizer.decode(tokenIds: tokens)
                // Fallback for special tokens (e.g. <think) that decode may skip:
                // if single-token decode returns empty, look up the token text directly
                if result.isEmpty && tokens.count == 1,
                   let tokenText = mlxTokenizer.convertIdToToken(tokens[0]) {
                    return tokenText
                }
                return result
            },
            eosToken: eosString,
            eosTokenId: mlxTokenizer.eosTokenId
        )

        container.setLoaded(mlxContainer: mlxContainer, tokenizer: tokenizer)

        let model = await mlxContainer.perform { (context: ModelContext) in
            SendableBox(context.model)
        }

        // Read model architecture params from config.json.
        // We need head_dim for KV memory calculation and max_position_embeddings
        // for context window enforcement. Every HuggingFace model has these.
        var headDim = 128
        let configFile = url.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configFile),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Multimodal models (Qwen3.5) store attention params in text_config
            let tc = (json["text_config"] as? [String: Any]) ?? json
            if let hd = tc["head_dim"] as? Int {
                headDim = hd
            } else if let hs = tc["hidden_size"] as? Int,
                      let nah = tc["num_attention_heads"] as? Int, nah > 0 {
                headDim = hs / nah
            }

            // Auto-detect context length from max_position_embeddings.
            // This is the standard HuggingFace field that every LLM model defines.
            // Without this, all models default to 4096 which is wrong for most modern
            // models (Phi-3.5=128K, Qwen=128K, Llama-3=128K, etc).
            if let maxPos = tc["max_position_embeddings"] as? Int, maxPos > 0 {
                container.config.contextLength = maxPos
                NovaMLXLog.info("Auto-detected context length: \(maxPos) tokens for \(config.identifier.displayName)")
            }

            // Detect hybrid linear attention models (e.g. Qwen3.5, bailing_hybrid).
            // These use MambaCache/ArraysCache for linear_attention layers + KVCacheSimple for full_attention layers.
            // FusedBatchScheduler only supports KVCacheSimple, so these must use ContinuousBatcher.
            if let layerTypes = tc["layer_types"] as? [String],
               layerTypes.contains("linear_attention") {
                container.config.hasLinearAttention = true
                NovaMLXLog.info("Detected hybrid linear attention model (\(layerTypes.filter { $0 == "linear_attention" }.count)/\(layerTypes.count) linear layers) — will use ContinuousBatcher")
            } else if let mt = tc["model_type"] as? String, mt == "bailing_hybrid" {
                container.config.hasLinearAttention = true
                NovaMLXLog.info("Detected bailing_hybrid model (MLA+GLA) — will use ContinuousBatcher")
            }
        }

        // Read generation_config.json — model author's recommended sampling params.
        // Priority: generation_config.json > per-family defaults > global ModelConfig defaults.
        var genConfig: GenerationConfig?
        let genConfigFile = url.appendingPathComponent("generation_config.json")
        if let data = try? Data(contentsOf: genConfigFile),
           let gen = try? JSONDecoder().decode(GenerationConfig.self, from: data) {
            genConfig = gen
            if let t = gen.temperature { container.config.temperature = t }
            if let p = gen.topP { container.config.topP = p }
            if let k = gen.topK { container.config.topK = k }
            if let r = gen.repetitionPenalty { container.config.repeatPenalty = r }
            if let f = gen.frequencyPenalty { container.config.frequencyPenalty = f }
            if let p = gen.presencePenalty { container.config.presencePenalty = p }
            if let m = gen.maxNewTokens { container.config.maxTokens = min(m, container.config.maxTokens) }
            NovaMLXLog.info("generation_config.json: temp=\(gen.temperature?.description ?? "-"), topP=\(gen.topP?.description ?? "-"), topK=\(gen.topK?.description ?? "-")")
        }

        // Apply per-family sampling defaults for fields NOT set by generation_config.
        // frequencyPenalty/repetitionPenalty always come from family defaults since
        // generation_config.json rarely includes them.
        let familyOpt = ModelFamilyRegistry.shared.optimization(for: config.identifier.id, family: config.identifier.family)
        if let fs = familyOpt.samplingDefaults {
            if genConfig?.temperature == nil, let t = fs.temperature { container.config.temperature = t }
            if genConfig?.topP == nil, let p = fs.topP { container.config.topP = p }
            if genConfig?.topK == nil, let k = fs.topK { container.config.topK = k }
            if let m = fs.minP { container.config.minP = m }
            if genConfig?.frequencyPenalty == nil, let f = fs.frequencyPenalty { container.config.frequencyPenalty = f }
            if genConfig?.repetitionPenalty == nil, let r = fs.repetitionPenalty { container.config.repeatPenalty = r }
            if let m = fs.maxTokens { container.config.maxTokens = min(m, container.config.maxTokens) }
            if fs.frequencyPenalty != nil || fs.topK != nil {
                NovaMLXLog.info("Family sampling defaults applied for \(config.identifier.family.rawValue): freqPenalty=\(fs.frequencyPenalty?.description ?? "-"), topK=\(fs.topK?.description ?? "-")")
            }
        }

        if let kvProvider = model.value as? KVCacheDimensionProvider, container.config.kvMemoryBytesPerToken == nil {
            let numLayers = kvProvider.kvHeads.count
            let kvHeadsSum = kvProvider.kvHeads.reduce(0, +)
            // KV cache per token = 2 (K+V) × sum(kvHeads per layer) × headDim × sizeof(Float16)
            let bytesPerToken = 2 * kvHeadsSum * headDim * MemoryLayout<Float16>.size
            container.config.kvMemoryBytesPerToken = bytesPerToken
            NovaMLXLog.info("Auto-calculated KV memory: \(bytesPerToken / 1024)KB/token (\(numLayers) layers, \(kvHeadsSum) total kvHeads, headDim=\(headDim))")
        }

        let weightsBytes = MLX.Memory.activeMemory
        if weightsBytes > 0 {
            let reservationTicket = wiredPolicy.ticket(size: Int(weightsBytes), kind: MLX.WiredMemoryTicketKind.reservation)
            let availableMem = Self.getAvailablePhysicalMemory()
            let neededMB = UInt64(weightsBytes) / 1_048_576
            // Only wire if there's enough free physical memory. Wired reservation
            // pins pages in RAM and will block indefinitely when free RAM is exhausted.
            // We also keep a 1GB headroom so the OS has breathing room.
            let headroom = UInt64(1024 * 1024 * 1024)
            if availableMem > UInt64(weightsBytes) + headroom {
                _ = await reservationTicket.start()
                container.setWiredReservation(reservationTicket)
                NovaMLXLog.info("Wired memory reservation: \(neededMB)MB for \(config.identifier.displayName)")
            } else {
                NovaMLXLog.warning("Skipping wired reservation for \(config.identifier.displayName): only \(availableMem / 1_048_576)MB free, need \(neededMB)MB + 1GB headroom. Model runs without wired memory guarantee.")
            }
        }

        // Chat-template sanity check — verify the rendering pipeline is healthy
        // before the model becomes serviceable. Failures here mean every chat
        // request will produce garbled output. We log warnings (not errors)
        // because some non-chat models (embeddings, base models) legitimately
        // have no chat template; the API layer will reject chat requests for
        // those models with a clear error message.
        await Self.runChatTemplateSanityCheck(
            mlxContainer: mlxContainer,
            modelId: config.identifier.id,
            modelType: config.modelType
        )

        // Optional upstream-fingerprint check (off by default; enable with
        // NOVAMLX_TEMPLATE_UPSTREAM_CHECK=1 env var). Compares the local
        // template (the one swift-transformers will actually render) against
        // the HuggingFace upstream copy. Result is cached for 24h.
        if config.modelType != .embedding {
            let templateForCheck: String = {
                let tcPath = url.appendingPathComponent("tokenizer_config.json")
                if let data = try? Data(contentsOf: tcPath),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let t = json["chat_template"] as? String, !t.isEmpty {
                    return t
                }
                let jinjaPath = url.appendingPathComponent("chat_template.jinja")
                return (try? String(contentsOf: jinjaPath, encoding: .utf8)) ?? ""
            }()
            if !templateForCheck.isEmpty {
                Task {
                    await ChatTemplateUpstreamCheck.shared.check(
                        modelId: config.identifier.id,
                        localTemplate: templateForCheck
                    )
                }
            }
        }

        // Auto-draft a test profile for this model if neither the bundled
        // registry nor the user has one. Writes to ~/.nova/profiles/_drafts/
        // for operator review. Skipped silently for embeddings (no chat).
        if config.modelType != .embedding {
            let registryHasOverride = ChatTemplateRegistry.shared.templateOverride(
                modelId: config.identifier.id,
                family: config.identifier.family,
                architecture: nil
            ) != nil
            ChatTemplateProfileDrafter.draftIfNeeded(
                modelId: config.identifier.id,
                modelDir: url,
                family: config.identifier.family,
                registryHasOverride: registryHasOverride
            )
        }

        pool.add(container)
        metricsStore.recordModelLoad()
        return container
    }

    /// Render a canonical test prompt through the loaded tokenizer and check the
    /// result for known failure modes:
    ///  • throws (no chat template, syntax error in Jinja)
    ///  • zero-length output (template silently produces empty string)
    ///  • unrendered Jinja (`{{`, `{%-` literals — template wasn't compiled)
    ///  • missing user content (the literal "ping_test_42" we injected isn't
    ///    represented in the rendered output → template loses user data)
    ///  • unbalanced common control tokens (`<|im_start|>` count != `<|im_end|>`,
    ///    `<start_of_turn>` != `<end_of_turn>`, etc.) — indicates a broken
    ///    template that opens turns it never closes.
    ///
    /// Each finding is logged with a stable `[ChatTemplateHealth]` prefix so
    /// operators can grep for it. This runs once per `loadModel`; the cost is
    /// negligible compared to weight loading.
    private static func runChatTemplateSanityCheck(
        mlxContainer: MLXLMCommon.ModelContainer,
        modelId: String,
        modelType: ModelType
    ) async {
        // Embedding-only models legitimately have no chat template. Skip.
        guard modelType != .embedding else { return }

        let tokenizer = await mlxContainer.tokenizer

        // Probe message designed to surface common bugs:
        // • Two roles to exercise turn separators.
        // • A unique sentinel string ("ping_test_42") that, after decode,
        //   should appear somewhere in the rendered output. If it doesn't,
        //   the template is dropping user content.
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": "ping_test_42"]
        ]

        let renderedTokens: [Int]
        do {
            renderedTokens = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: nil
            )
        } catch {
            NovaMLXLog.warning("[ChatTemplateHealth] \(modelId): apply_chat_template threw — chat requests will fail (\(error.localizedDescription))")
            return
        }

        guard !renderedTokens.isEmpty else {
            NovaMLXLog.warning("[ChatTemplateHealth] \(modelId): chat template rendered to ZERO tokens — template is empty or filters out all input")
            return
        }

        let rendered = tokenizer.decode(tokenIds: renderedTokens, skipSpecialTokens: false)

        var issues: [String] = []

        // Unrendered Jinja → template body leaked into output (compile/runtime failure)
        for marker in ["{{", "}}", "{%-", "%}", "{#", "#}"] {
            if rendered.contains(marker) {
                issues.append("contains unrendered Jinja `\(marker)`")
                break
            }
        }

        // Sentinel preservation — without this, the model never sees user input
        if !rendered.contains("ping_test_42") {
            issues.append("user content sentinel missing from rendered output (template drops user.content)")
        }

        // Unbalanced common control-token pairs — heuristic, only flag if both >0 and counts differ
        let pairs: [(String, String)] = [
            ("<|im_start|>", "<|im_end|>"),
            ("<start_of_turn>", "<end_of_turn>"),
            ("<|start|>", "<|end|>"),
        ]
        for (open, close) in pairs {
            let openCount = rendered.components(separatedBy: open).count - 1
            let closeCount = rendered.components(separatedBy: close).count - 1
            // For some templates the assistant turn may legitimately leave the
            // last `<|im_start|>` open (waiting for completion), so we tolerate
            // openCount = closeCount + 1.
            if openCount > 0 || closeCount > 0 {
                if openCount > closeCount + 1 || closeCount > openCount {
                    issues.append("unbalanced \(open)/\(close): \(openCount) vs \(closeCount)")
                }
            }
        }

        // Cross-format leak: detect markers from a DIFFERENT family than the
        // primary one. E.g. a Qwen ChatML template that ALSO contains <|turn>
        // or <start_of_turn> is almost certainly corrupted.
        let detected = ChatTemplateFormat.detectAll(from: rendered)
        if detected.count > 1 {
            let formats = detected.map { "\($0.format) (\($0.confidence))" }.joined(separator: ", ")
            issues.append("multi-format markers in rendered output: \(formats) — likely template corruption (one family format expected)")
        }

        if issues.isEmpty {
            NovaMLXLog.info("[ChatTemplateHealth] \(modelId): rendering pipeline OK (\(renderedTokens.count) tokens, format=\(detected.first?.format.label ?? "unknown"))")
        } else {
            NovaMLXLog.warning("[ChatTemplateHealth] \(modelId): \(issues.count) issue(s) detected — \(issues.joined(separator: "; "))")
        }
    }

    public func unloadModel(_ identifier: ModelIdentifier) {
        if let container = pool.get(identifier.id),
           let ticket = container.wiredReservationTicket {
            Task { await ticket.end() }
            NovaMLXLog.info("Wired memory reservation released for \(identifier.displayName)")
        }
        // Clear prefix cache for this model
        if let cacheManager = lock.withLock({ prefixCacheManagers.removeValue(forKey: identifier.id) }) {
            cacheManager.clear()
            NovaMLXLog.info("Prefix cache cleared for \(identifier.displayName)")
        }
        // Drop cached TokenMaskBuilder so its per-token vocabulary tables
        // (V × 3 arrays, 250k+ entries for Qwen3.6/Gemma-4) are released.
        let cacheRef = tokenMaskBuilderCache
        let modelIdForCache = identifier.id
        Task { await cacheRef.invalidate(modelId: modelIdForCache) }
        _ = pool.remove(identifier.id)
        metricsStore.recordModelUnload()
        MLX.Memory.clearCache()
        NovaMLXLog.info("GPU cache cleared after unloading \(identifier.displayName)")
    }

    public func getContainer(for modelId: String) -> ModelContainer? {
        pool.get(modelId)
    }

    public func getPrefixCacheStats(for modelId: String) -> PrefixCacheStats? {
        lock.withLock { prefixCacheManagers[modelId]?.getStats() }
    }

    public func clearPrefixCache(for modelId: String) {
        lock.withLock { prefixCacheManagers[modelId]?.clear() }
    }

    func getOrCreatePrefixCacheManager(modelId: String) -> PrefixCacheManager? {
        lock.withLock {
            // Operational kill switch (ServerConfig.prefixCacheEnabled). When
            // disabled, never instantiate or return a manager — callers should
            // treat `nil` as "no prefix cache for this request".
            guard prefixCacheEnabled else { return nil }
            if let existing = prefixCacheManagers[modelId] {
                return existing
            }
            let ssdDir = NovaMLXPaths.prefixCacheDir(for: modelId)
            let config = PrefixCacheConfig(
                blockSize: 64,
                maxBlocks: 4096,
                initialBlocks: 256,
                ssdCacheDir: ssdDir,
                ssdMaxSizeBytes: 5 * 1024 * 1024 * 1024  // 5 GB (down from 100 GB)
            )
            let manager = PrefixCacheManager(config: config, modelName: modelId)
            prefixCacheManagers[modelId] = manager
            return manager
        }
    }

    /// Toggle the prefix cache subsystem at runtime. When disabled, all
    /// existing managers are cleared and removed; subsequent requests will
    /// not load or store anything via the prefix cache. Mirrors
    /// `ServerConfig.prefixCacheEnabled`.
    public func setPrefixCacheEnabled(_ enabled: Bool) {
        lock.withLock {
            prefixCacheEnabled = enabled
            if !enabled {
                for (_, mgr) in prefixCacheManagers {
                    mgr.clear()
                }
                prefixCacheManagers.removeAll()
            }
        }
        NovaMLXLog.info("Prefix cache subsystem \(enabled ? "enabled" : "disabled")")
    }

    public var isPrefixCacheEnabled: Bool {
        lock.withLock { prefixCacheEnabled }
    }

    public func cleanupOrphanedCacheDirs(downloadedModelIds: Set<String>) {
        let fm = FileManager.default
        let cacheBase = NovaMLXPaths.prefixCacheBaseDir
        guard let contents = try? fm.contentsOfDirectory(at: cacheBase, includingPropertiesForKeys: nil) else { return }
        var cleaned = 0
        for dir in contents {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: dir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            let modelId = dir.lastPathComponent.replacingOccurrences(of: "_", with: "/")
            if !downloadedModelIds.contains(modelId) {
                try? fm.removeItem(at: dir)
                cleaned += 1
            }
        }
        if cleaned > 0 {
            NovaMLXLog.info("Cleaned \(cleaned) orphaned prefix cache directories")
        }
    }

    public func tokenizeMessages(_ messages: [ChatMessage], tokenizer: Tokenizer) -> [Int] {
        var allTokens: [Int] = []
        for msg in messages {
            if let content = msg.content {
                allTokens.append(contentsOf: tokenizer.encode(content))
            }
        }
        return allTokens
    }

    private func toSendable(_ value: Any) -> any Sendable {
        switch value {
        case let s as String: return s
        case let i as Int: return i
        case let d as Double: return d
        case let b as Bool: return b
        case let a as [Any]: return a.map { toSendable($0) } as any Sendable
        case let d as [String: Any]: return d.mapValues { toSendable($0) } as any Sendable
        default: return "\(value)"
        }
    }

    func buildUserInput(request: InferenceRequest, hasImages: Bool) async throws -> UserInput {
        let toolSpecs: [[String: any Sendable]]? = request.tools?.map { tool in
            tool.mapValues { toSendable($0) }
        }

        var additionalContext: [String: any Sendable]? = nil
        // Forward thinking-related kwargs to chat-template renderer.
        // Templates that don't reference these variables ignore them via
        // Jinja `is defined` semantics, so pass-through is safe across
        // families (Qwen / Gemma / Bailing / etc.).
        var ctx: [String: any Sendable] = [:]
        if let enableThinking = request.enableThinking {
            ctx["enable_thinking"] = enableThinking
        }
        if let preserveThinking = request.preserveThinking {
            ctx["preserve_thinking"] = preserveThinking
        }
        if !ctx.isEmpty {
            additionalContext = ctx
        }

        if hasImages {
            let chatMessages = request.messages.compactMap { msg -> Chat.Message? in
                let images: [UserInput.Image] = (msg.images ?? []).compactMap { urlString -> UserInput.Image? in
                    if urlString.hasPrefix("data:") {
                        let parts = urlString.split(separator: ",", maxSplits: 1)
                        guard parts.count == 2 else { return nil }
                        let base64Str = String(parts[1])
                        guard let data = Data(base64Encoded: base64Str) else { return nil }
                        guard let ciImage = CIImage(data: data) else { return nil }
                        return .ciImage(ciImage)
                    } else if urlString.hasPrefix("http") {
                        guard let url = URL(string: urlString) else { return nil }
                        return .url(url)
                    } else {
                        return .url(URL(fileURLWithPath: urlString))
                    }
                }
                return .user(msg.content ?? "", images: images)
            }
            return UserInput(prompt: .chat(chatMessages), tools: toolSpecs, additionalContext: additionalContext)
        } else {
            let messages: [Message] = request.messages.map { msg in
                var m: Message = ["role": msg.role.rawValue, "content": msg.content ?? ""]
                if let tcId = msg.toolCallId { m["tool_call_id"] = tcId }
                if let name = msg.name { m["name"] = name }
                if let tcs = msg.toolCalls {
                    m["tool_calls"] = tcs.map { tc in
                        ["function": ["name": tc.functionName, "arguments": tc.arguments]] as [String: any Sendable]
                    }
                }
                return m
            }
            return UserInput(prompt: .messages(messages), tools: toolSpecs, additionalContext: additionalContext)
        }
    }

    private func deferredClearCache() {
        lock.withLock {
            deferredClearCounter += 1
            if deferredClearCounter >= deferredClearThreshold {
                deferredClearCounter = 0
                MLX.Memory.clearCache()
            }
        }
    }

    private func createGenerationTicket(promptTokens: Int, maxTokens: Int, bytesPerToken: Int = 262_144) -> WiredMemoryTicket {
        let totalTokens = promptTokens + maxTokens
        let estimatedBytes = totalTokens * bytesPerToken
        return wiredPolicy.ticket(size: estimatedBytes, kind: MLX.WiredMemoryTicketKind.active)
    }

    /// Returns the approximate free physical memory (free + inactive + purgeable) in bytes.
    private static func getAvailablePhysicalMemory() -> UInt64 {
        let hostPort = mach_host_self()
        var pageSize: vm_size_t = 0
        host_page_size(hostPort, &pageSize)

        var hostInfo = vm_statistics64()
        var infoCount = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &hostInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(infoCount)) {
                host_statistics64(hostPort, HOST_VM_INFO64, $0, &infoCount)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return UInt64(hostInfo.free_count + hostInfo.inactive_count + hostInfo.purgeable_count) * UInt64(pageSize)
    }

    /// Estimate model weight size from config.json params or directory size.
    /// Used by the pre-load memory gate to decide whether we need LRU eviction.
    static func estimateModelWeightSize(at url: URL) -> UInt64? {
        let configFile = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configFile),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let numParams = json["num_params"] as? UInt64 {
            let dtype = json["torch_dtype"] as? String ?? "float16"
            let bytesPerParam: UInt64 = (dtype.contains("int8") || dtype.contains("q8")) ? 1 : 2
            return numParams * bytesPerParam
        }
        // Fallback: directory size (safetensors dominate)
        return FileManager.default.directorySize(at: url)
    }

    /// Evict LRU unpinned models until estimated new model fits under softLimitBytes.
    /// Returns true if enough headroom was freed, false otherwise.
    private func ensureMemoryHeadroom(neededBytes: UInt64, softLimitBytes: UInt64) async -> Bool {
        let maxIterations = 10
        for _ in 0..<maxIterations {
            let current = UInt64(MLX.Memory.activeMemory)
            if current + neededBytes <= softLimitBytes { return true }
            let freed = pool.evictLRU()
            if freed == nil { break }
            try? await Task.sleep(nanoseconds: 200_000_000)  // 200ms settle for Metal
        }
        return UInt64(MLX.Memory.activeMemory) + neededBytes <= softLimitBytes
    }

    public func preflightCheck(modelId: String, promptTokens: Int, maxTokens: Int) async throws {
        guard let maxGPU = GPU.maxRecommendedWorkingSetBytes(), maxGPU > 0 else { return }

        guard let container = getContainer(for: modelId) else { return }

        let adminOverride = container.kvMemoryOverride
        let autoCalculated = container.config.kvMemoryBytesPerToken
        let bytesPerToken = adminOverride ?? autoCalculated ?? 262144

        // Use budgetTracker's committed bytes (accurate) instead of MLX.Memory.activeMemory (unreliable)
        let budgetMetrics = await budgetTracker.metrics
        let currentCommitted = budgetMetrics.weightsMemoryBytes + budgetMetrics.committedKVBytes
        let estimatedKVBytes = UInt64(promptTokens + maxTokens) * UInt64(bytesPerToken)
        let projectedTotal = currentCommitted + estimatedKVBytes

        // Soft cap at 45% — early warning before hitting hard 50% MLX limit
        let safeLimit = UInt64(maxGPU) * 45 / 100

        guard projectedTotal <= safeLimit else {
            let neededMB = projectedTotal / 1024 / 1024
            let limitMB = safeLimit / 1024 / 1024
            NovaMLXLog.warning("Preflight REJECTED: model=\(modelId), estimated=\(neededMB)MB, limit=\(limitMB)MB, committed=\(currentCommitted / 1024 / 1024)MB, kvEstimate=\(estimatedKVBytes / 1024 / 1024)MB, bytesPerToken=\(bytesPerToken)")
            throw NovaMLXError.inferenceFailed(
                "Insufficient GPU memory: estimated \(neededMB)MB needed, limit \(limitMB)MB. Committed: \(currentCommitted / 1024 / 1024)MB"
            )
        }

        NovaMLXLog.info("Preflight OK: model=\(modelId), projected=\(projectedTotal / 1024 / 1024)MB, limit=\(safeLimit / 1024 / 1024)MB, committed=\(currentCommitted / 1024 / 1024)MB, bytesPerToken=\(bytesPerToken)")
    }

    /// Compute effective bytesPerToken for a model, factoring in TurboQuant compression.
    /// Returns raw FP16 bytesPerToken when TurboQuant is not configured.
    public func effectiveBytesPerToken(modelId: String) -> Int {
        guard let container = getContainer(for: modelId) else { return 262_144 }
        let raw = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144

        // Factor in TurboQuant compression
        if let turboConfig = turboQuantService.getConfig(forModel: modelId) {
            let compressionRatio = Double(16) / Double(turboConfig.bits)
            return max(1, Int(Double(raw) / compressionRatio))
        }

        // Also check per-model config kvBits
        if let kvBits = container.config.kvBits, kvBits < 16 {
            let compressionRatio = Double(16) / Double(kvBits)
            return max(1, Int(Double(raw) / compressionRatio))
        }

        return raw
    }

    /// Estimate total tokens for a request (prompt + max output).
    /// Prompt tokens are already in activeMemory, so we only budget for output tokens.
    public func estimateRequestTokens(modelId: String, request: InferenceRequest) -> Int {
        let maxTokens = request.maxTokens ?? getContainer(for: modelId)?.config.maxTokens ?? 4096
        return maxTokens
    }

    func buildGenerateParameters(
        request: InferenceRequest, config: ModelConfig, settings: ModelSettings? = nil
    ) -> GenerateParameters {
        let temperature = Float(request.temperature ?? config.temperature)
        let maxTokens = request.maxTokens ?? config.maxTokens
        let topP = Float(request.topP ?? config.topP)
        let topK = request.topK ?? config.topK
        let minP = request.minP ?? config.minP

        var params = GenerateParameters(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP
        )

        let repPenalty = request.repetitionPenalty ?? (config.repeatPenalty != 1.0 ? config.repeatPenalty : nil)
        if let repPenalty, repPenalty != 1.0 {
            params.repetitionPenalty = repPenalty
            params.repetitionContextSize = config.repeatLastN
        }

        let freq = request.frequencyPenalty ?? config.frequencyPenalty
        if freq != 0 {
            params.frequencyPenalty = freq
        }

        if let pres = request.presencePenalty, pres != 0 {
            params.presencePenalty = pres
        }

        if let settings {
            turboQuantService.applyToGenerateParameters(&params, modelId: config.identifier.id, settings: settings)
        } else if let fetchedSettings = settingsProvider?(config.identifier.id) {
            turboQuantService.applyToGenerateParameters(&params, modelId: config.identifier.id, settings: fetchedSettings)
        } else {
            let familyOpt = ModelFamilyRegistry.shared.optimization(
                for: config.identifier.id, family: config.identifier.family)
            if params.kvBits == nil, let defaultBits = familyOpt.defaultKVBits {
                params.kvBits = defaultBits
                params.kvGroupSize = familyOpt.safeGroupSize()
            }
            if params.kvBits == nil, let kvBits = config.kvBits {
                params.kvBits = kvBits
                params.kvGroupSize = config.kvGroupSize
            }
        }

        let familyOpt = ModelFamilyRegistry.shared.optimization(
            for: config.identifier.id, family: config.identifier.family)
        params.prefillStepSize = familyOpt.prefillStepSize

        if let seed = request.seed ?? config.seed {
            MLXRandom.seed(seed)
        }

        return params
    }

    // MARK: - Control Token Stop Detection

    private static let controlTokenLock = NSLock()
    private nonisolated(unsafe) static var _controlTokenCache: [String: [String]] = [:]

    /// Get cached control tokens for a model, extracting if needed
    static func controlTokensForModel(modelId: String) -> [String] {
        controlTokenLock.lock()
        if let cached = _controlTokenCache[modelId] {
            controlTokenLock.unlock()
            return cached
        }
        controlTokenLock.unlock()

        let tokens = ModelContainer.extractControlTokens(for: modelId)

        controlTokenLock.lock()
        _controlTokenCache[modelId] = tokens
        controlTokenLock.unlock()
        if !tokens.isEmpty {
            NovaMLXLog.info("[ControlTokens] Detected \(tokens.count) control tokens for \(modelId)")
        }
        return tokens
    }

    // MARK: - Defensive EOS injection (todo.markdown §2.9)
    //
    // If a model's `config.json` ships with a malformed or missing
    // `eos_token_id`, generation never halts and silently consumes the entire
    // `maxTokens` budget. The original Gemma-4 VLM `<turn|>` bug (todo.markdown
    // §5 row 1) was masked by `eos_token_id: [1, 106]` being correct in the
    // canonical config; a future quant author shipping `eos_token_id: []` (or
    // omitting 106) would re-open it.
    //
    // This helper unions the upstream-derived EOS set with hardcoded
    // family-specific canonical stop tokens. It logs a warning the first time
    // injection actually adds something for a given modelId, so we can detect
    // malformed quants in the wild via `~/.nova/novamlx.log`.
    //
    // Currently only Gemma is hardcoded — extend the switch as new families
    // surface canonical stop tokens that warrant defense.
    static let defensiveEosWarningLock = NSLock()
    nonisolated(unsafe) static var defensiveEosWarnedModels: Set<String> = []

    /// Returns `modelConfig.eosTokenIds` unioned with family-specific canonical
    /// stop tokens. Logs once-per-modelId when injection was needed.
    static func defensiveEosTokenIds(
        for modelConfig: ModelConfiguration,
        container: ModelContainer
    ) -> Set<Int> {
        var ids = modelConfig.eosTokenIds

        switch container.identifier.family {
        case .gemma:
            // Gemma 2/3/4 canonical: <bos>=1, <end_of_turn>=106. Both ship in
            // the official `eos_token_id: [1, 106]`. Empty / missing → silent
            // runaway VLM generation (token 106 never triggers stop).
            let canonical: Set<Int> = [1, 106]
            let missing = canonical.subtracting(ids)
            if !missing.isEmpty {
                warnDefensiveEosInjection(
                    modelId: container.identifier.id,
                    family: "gemma",
                    missing: missing
                )
                ids.formUnion(canonical)
            }
        default:
            // Other families: no defensive defaults today. Add cases here if a
            // family surfaces canonical stop tokens that aren't reliably
            // present in upstream `config.json`.
            break
        }

        return ids
    }

    private static func warnDefensiveEosInjection(
        modelId: String,
        family: String,
        missing: Set<Int>
    ) {
        defensiveEosWarningLock.lock()
        defer { defensiveEosWarningLock.unlock() }
        guard !defensiveEosWarnedModels.contains(modelId) else { return }
        defensiveEosWarnedModels.insert(modelId)
        let sorted = missing.sorted()
        NovaMLXLog.warning(
            "[DefensiveEOS] \(modelId): config.json eos_token_id missing canonical \(family) stop token(s) \(sorted) — auto-injecting. The model's config.json may be malformed; consider re-quantizing or reporting to the model author."
        )
    }

    /// Test-only hook: clear the once-per-model warning cache so tests can
    /// re-trigger the warning path on the same modelId.
    static func _resetDefensiveEosWarningCache() {
        defensiveEosWarningLock.lock()
        defer { defensiveEosWarningLock.unlock() }
        defensiveEosWarnedModels.removeAll()
    }

    private func buildTurnStopProcessor(
        modelId: String,
        container: ModelContainer?,
        tokenizer: Tokenizer,
        modelConfig: ModelConfiguration
    ) -> TurnStopProcessor? {
        let patterns = Self.controlTokensForModel(modelId: modelId)
        guard !patterns.isEmpty else { return nil }

        var eosIds = Set<Int>()
        if let eosId = tokenizer.eosTokenId { eosIds.insert(eosId) }
        // §2.9: defensive injection of family-canonical stop tokens.
        if let container = container {
            eosIds.formUnion(Self.defensiveEosTokenIds(for: modelConfig, container: container))
        } else {
            eosIds.formUnion(modelConfig.eosTokenIds)
        }

        let hallucinationPatterns = container?.chatTemplateProcessor?.hallucinationPatterns() ?? []

        return TurnStopProcessor(
            stopPatterns: patterns,
            eosTokenIds: eosIds,
            hallucinationPatterns: hallucinationPatterns,
            decode: tokenizer.decode
        )
    }

    /// Compose a base processor (repetition penalty), optional turn stop processor,
    /// and optional thinking-budget processor into a single LogitProcessor chain.
    ///
    /// `request` and `parameters` may be omitted to skip thinking-budget construction
    /// (used by call sites that don't have request context, such as warmup paths).
    /// All non-budget call sites historically passed only base + turnStop; the budget
    /// slot is opt-in via the request-aware overload below.
    private func composeWithTurnStop(
        baseProcessor: (any LogitProcessor)?,
        modelId: String,
        container: ModelContainer?,
        tokenizer: Tokenizer,
        modelConfig: ModelConfiguration,
        request: InferenceRequest? = nil,
        parameters: GenerateParameters? = nil
    ) -> (any LogitProcessor)? {
        let tsp = buildTurnStopProcessor(
            modelId: modelId,
            container: container,
            tokenizer: tokenizer,
            modelConfig: modelConfig
        )
        let budget: ThinkingBudgetProcessor?
        if let request = request, let parameters = parameters {
            budget = buildThinkingBudgetProcessor(
                request: request,
                container: container,
                tokenizer: tokenizer,
                parameters: parameters
            )
        } else {
            budget = nil
        }

        return ComposedLogitProcessor.compose(
            grammar: baseProcessor,
            turnStop: tsp,
            thinkingBudget: budget
        )
    }

    /// Build a `ThinkingBudgetProcessor` if the request + model state warrant it.
    ///
    /// **Activation rules:**
    ///  1. Opt-out: returns nil if `request.enableThinking == false` or
    ///     `request.thinkingBudget == 0` (explicit disable).
    ///  2. Explicit budget: returns processor with `request.thinkingBudget`
    ///     when > 0, regardless of temperature.
    ///  3. Smart default: when temperature == 0 AND the model is a thinking
    ///     model AND user did not specify a budget, applies an automatic
    ///     budget of `min(1024, max(256, maxTokens / 2))`. Greedy decoding on
    ///     reasoning models can lock the model in chain-of-thought on complex
    ///     prompts (Qwen team explicitly recommends `temperature ≥ 0.6` for
    ///     thinking mode). Verified reproducible against Python `mlx_lm` 0.31.3
    ///     on `mlx-community/Qwen3.6-27B-4bit` (T7 prompt: 2000 tokens, no
    ///     `</think>` emitted, all chain-of-thought).
    ///  4. Returns nil if no close-marker token can be resolved from the
    ///     tokenizer (model has no recognizable thinking format, or close
    ///     markers are multi-token like Harmony's `<|channel|>final<|message|>`
    ///     which require state-machine tracking — out of scope for the
    ///     single-token-mask processor).
    private func buildThinkingBudgetProcessor(
        request: InferenceRequest,
        container: ModelContainer?,
        tokenizer: Tokenizer,
        parameters: GenerateParameters
    ) -> ThinkingBudgetProcessor? {
        // Explicit opt-outs
        if request.enableThinking == false { return nil }
        if let b = request.thinkingBudget, b == 0 { return nil }

        let modelId = container?.identifier.id ?? request.model
        let temperature = parameters.temperature
        let maxTokens = parameters.maxTokens ?? container?.config.maxTokens ?? 4096

        // Determine effective budget
        let effective: Int? = {
            if let b = request.thinkingBudget, b > 0 { return b }
            // Smart default: temp == 0 + thinking model only.
            guard temperature == 0 else { return nil }
            guard ModelContainer.detectThinkingModel(for: modelId) else { return nil }
            return min(1024, max(256, maxTokens / 2))
        }()

        guard let budget = effective, budget > 0 else { return nil }

        let closeIds = ThinkingCloseMarkerResolver.resolveCloseTokenIds(
            for: tokenizer, modelId: modelId
        )
        if closeIds.isEmpty {
            // No single-token close marker — log once at info level so the
            // explicit-budget user knows the request silently degraded to
            // unbounded thinking.
            if request.thinkingBudget != nil {
                NovaMLXLog.info(
                    "[ThinkingBudget] \(modelId): explicit thinkingBudget=\(budget) requested but no single-token close marker resolvable from tokenizer (Harmony / multi-token formats not yet supported). Budget will not be enforced."
                )
            }
            return nil
        }

        if request.thinkingBudget == nil {
            // Smart-default applied — log so users on temp=0 + thinking can
            // see this in `~/.nova/novamlx.log` and understand what happened.
            // Logged at info level (one line per request) — avoids surprise
            // without spamming.
            NovaMLXLog.info(
                "[ThinkingBudget] \(modelId): temp=0 + thinking model detected — applying smart-default budget=\(budget) (closeIds=\(closeIds.sorted())). Set thinking_budget=0 to disable, or set thinking_budget=N for a custom budget."
            )
        }

        return ThinkingBudgetProcessor(
            budget: budget,
            closeTokenIds: closeIds,
            modelId: modelId
        )
    }

    static func trimControlTokens(_ text: String, patterns: [String]) -> String {
        guard !patterns.isEmpty else { return text }
        var result = text
        for pattern in patterns {
            if let range = result.range(of: pattern) {
                result = String(result[..<range.lowerBound])
            }
        }
        return result
    }

    /// Regex-based scrub: strip any <|xxx|>, <|xxx>, or <|xxx\n patterns from output.
    /// Catches multi-token control sequences (mask_start, tool_call, tool_code, etc.)
    /// that aren't registered as special tokens in the tokenizer.
    /// The loose pattern catches partial tokens that span decode boundaries.
    /// Excludes thinking markers (<|begin_of_thought|>, <|end_of_thought|>) which
    /// ThinkingParser needs to see for correct thinking/content partitioning.
    private static let thinkingMarkerWords: Set<String> = [
        "begin_of_thought", "end_of_thought",
        "begin_of_think", "end_of_think",
        "think", "/think",
        "thinking", "/thinking",
    ]

    static let controlTokenRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "(?:<\\|[a-zA-Z_/][a-zA-Z0-9_/]*(?:\\|>|>|\\n|\\s))|(?:<[a-zA-Z_/][a-zA-Z0-9_/]*\\|>)")
    }()

    /// Regex: matches <|channel|>word<|message|> — Harmony channel+type+message structure
    private static let channelMessageRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "<\\|channel\\|>\\w+<\\|message\\|>")
    }()

    /// Regex: matches <|start|>word<|message|> — Harmony start+role+message structure
    private static let startMessageRegex: NSRegularExpression = {
        try! NSRegularExpression(pattern: "<\\|start\\|>\\w+<\\|message\\|>")
    }()

    /// Regex: matches Gemma-4 channel-thought block <|channel>thought\n...<channel|>.
    /// The asymmetric markers (<|channel> vs <channel|>) wrap reasoning content;
    /// captured content is normalized into <think>...</think> so ThinkingParser
    /// extracts it as reasoning_content. dotMatchesLineSeparators allows multi-line
    /// reasoning spans to be captured in one match.
    static let gemmaChannelThoughtRegex: NSRegularExpression = {
        try! NSRegularExpression(
            pattern: "<\\|channel>thought\\n([\\s\\S]*?)<channel\\|>",
            options: [.dotMatchesLineSeparators]
        )
    }()

    /// Known Harmony-format channel type words and role names that can be left behind
    /// when streaming scrubs <|channel|> and <|message|> individually.
    public static let harmonyResidueWords: Set<String> = [
        "analysis", "final", "commentary",     // channel types
        "user", "assistant", "system", "developer",  // roles
    ]

    static func scrubControlTokens(_ text: String) -> String {
        // Pre-pass: Gemma-4 channel-thought normalization.
        // Pattern <|channel>thought\n...<channel|> wraps reasoning content. Convert
        // it to <think>...</think> so ThinkingParser can extract reasoning_content.
        // The pattern is highly specific (single-pipe asymmetric channel markers +
        // literal "thought\n" prefix + matching <channel|> close) and does not
        // collide with other families. Harmony uses <|channel|>type<|message|>
        // (double-pipe), Qwen/DeepSeek use <think>, etc. Safe no-op for non-Gemma-4.
        var result = text
        if result.contains("<|channel>thought") {
            let nsRange = NSRange(result.startIndex..., in: result)
            let matches = gemmaChannelThoughtRegex.matches(in: result, range: nsRange)
            for match in matches.reversed() where match.numberOfRanges >= 2 {
                if let full = Range(match.range, in: result),
                   let inner = Range(match.range(at: 1), in: result) {
                    let reasoning = String(result[inner])
                    result.replaceSubrange(full, with: "<think>" + reasoning + "</think>")
                }
            }
        }

        // Pre-pass: Harmony (gpt-oss) channel normalization.
        // Harmony emits assistant turns as
        //     <|channel|>analysis<|message|>...thinking...<|end|>
        //     <|start|>assistant<|channel|>final<|message|>...content...<|return|>
        // Without normalization the channel-message regex below strips both
        // boundaries entirely, leaving thinking and final content concatenated
        // with no separator — `ThinkingParser` then has no way to tell them
        // apart. Converting to explicit `<think>...</think>` tags lets the
        // parser extract reasoning_content correctly without any family-
        // specific knowledge in the parser itself.
        //
        // Strictly Harmony-only: the trigger string `<|channel|>analysis<|message|>`
        // is unique to gpt-oss / Harmony format. Other families (Qwen, Gemma,
        // DeepSeek, Bailing, Llama, etc.) never emit it.
        //
        // We only convert `<|channel|>final<|message|>` to `</think>` if a
        // preceding `<think>` exists in the same string — otherwise the model
        // skipped analysis and went straight to final, in which case we just
        // strip the boundary marker (no thinking content to close).
        if result.contains("<|channel|>analysis<|message|>") {
            result = result.replacingOccurrences(
                of: "<|channel|>analysis<|message|>",
                with: "<think>"
            )
        }
        if result.contains("<|channel|>final<|message|>") {
            if let openRange = result.range(of: "<think>"),
               let finalRange = result.range(of: "<|channel|>final<|message|>"),
               openRange.lowerBound < finalRange.lowerBound {
                result.replaceSubrange(finalRange, with: "</think>")
            } else {
                // No prior <think> — model skipped analysis. Just strip the
                // boundary; trailing final content stays as content.
                result = result.replacingOccurrences(
                    of: "<|channel|>final<|message|>",
                    with: ""
                )
            }
        }

        // First pass: remove remaining multi-token structures
        // (e.g. <|channel|>commentary<|message|>, <|start|>user<|message|>)
        // The analysis/final boundaries above are already converted; the regex
        // catches everything else.
        for regex in [channelMessageRegex, startMessageRegex] {
            let nsRange = NSRange(result.startIndex..., in: result)
            let matches = regex.matches(in: result, range: nsRange)
            for match in matches.reversed() {
                if let range = Range(match.range, in: result) {
                    result.replaceSubrange(range, with: "")
                }
            }
        }

        // Second pass: standard control token scrubbing
        let nsRange2 = NSRange(result.startIndex..., in: result)
        let matches = controlTokenRegex.matches(in: result, range: nsRange2)
        for match in matches.reversed() {
            if let range = Range(match.range, in: result) {
                let matched = String(result[range])
                let isThinkingMarker = thinkingMarkerWords.contains { matched.contains($0) }
                if !isThinkingMarker {
                    result.removeSubrange(range)
                }
            }
        }

        // Third pass: remove Harmony residue words that survive per-token streaming scrub.
        // In streaming, <|channel|> and <|message|> are scrubbed individually, leaving
        // channel type words ("analysis", "final") and role names ("user", "assistant")
        // as standalone text. Check if the result is EXACTLY a residue word or starts
        // with one followed by content.
        let trimmed = result.trimmingCharacters(in: .whitespaces)
        if harmonyResidueWords.contains(trimmed) {
            return ""
        }
        // Check if starts with a residue word (e.g. "finalYour name is Bob.")
        for word in harmonyResidueWords {
            if trimmed.hasPrefix(word) {
                let afterWord = trimmed.dropFirst(word.count)
                // Only strip if followed by uppercase (sentence start) or end of string.
                // Do NOT strip on newline — Qwen3 <|turn|>system\n leaves legitimate role text.
                if afterWord.isEmpty || afterWord.first?.isUppercase == true {
                    result = String(afterWord)
                }
            }
        }

        return result
    }

    /// Filter a streaming chunk for control tokens. Returns (cleanText, shouldStop).
    /// Handles both full matches and partial matches (where TurnStopProcessor
    /// forces EOS mid-token, leaving e.g. "<|turn" without the closing ">").
    static func filterControlInChunk(
        _ text: String,
        accumulated: inout String,
        yieldedCount: inout Int,
        patterns: [String]
    ) -> (String, Bool) {
        accumulated += text
        let totalLen = accumulated.count

        // 1. Full control token match — trim and stop
        for pattern in patterns {
            if let range = accumulated.range(of: pattern) {
                let safeEnd = accumulated.distance(from: accumulated.startIndex, to: range.lowerBound)
                let cleanText: String
                if safeEnd > yieldedCount {
                    cleanText = String(accumulated.dropFirst(yieldedCount).prefix(safeEnd - yieldedCount))
                } else {
                    cleanText = ""
                }
                yieldedCount = totalLen
                return (cleanText, true)
            }
        }

        // 2. Partial prefix check — if tail could be start of a control token, buffer it
        let maxLen = patterns.map(\.count).max() ?? 0
        let checkLen = min(totalLen, maxLen)
        var safeEnd = totalLen
        if checkLen > 0 {
            for i in 1...checkLen {
                let suffixStart = accumulated.index(accumulated.endIndex, offsetBy: -i)
                let suffix = accumulated[suffixStart...]
                for pattern in patterns {
                    if pattern.hasPrefix(suffix) {
                        safeEnd = min(safeEnd, totalLen - i)
                    }
                }
            }
        }

        // Yield safe portion (everything before the potential control token prefix)
        if safeEnd > yieldedCount {
            let cleanText = String(accumulated.dropFirst(yieldedCount).prefix(safeEnd - yieldedCount))
            yieldedCount = safeEnd
            return (cleanText, false)
        }
        return ("", false)
    }

    public func generate(_ request: InferenceRequest) async throws -> InferenceResult {
        // Dispatch to specialized paths based on request fields
        if let sessionId = request.sessionId {
            return try await generateWithSession(request, sessionId: sessionId)
        }
        if request.jsonSchemaDef != nil || request.responseFormat == .jsonObject ||
            request.regexPattern != nil || request.gbnfGrammar != nil {
            return try await generateWithProcessor(request)
        }

        guard let container = getContainer(for: request.model) else {
            throw NovaMLXError.modelNotFound(request.model)
        }
        guard let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }

        lock.withLock { activeCount += 1 }
        defer { lock.withLock { activeCount -= 1 } }

        let startTime = Date()

        let hasImages = request.messages.contains { msg in
            guard let imgs = msg.images, !imgs.isEmpty else { return false }
            return true
        }

        let userInput = try await buildUserInput(request: request, hasImages: hasImages)
        let input: LMInput
        do {
            input = try await mlxContainer.prepare(input: userInput)
        } catch {
            let desc = error.localizedDescription
            if desc.contains("chat template") {
                // VLM models can't use plain text fallback — their forward pass expects image embeddings
                if container.config.modelType == .vlm {
                    NovaMLXLog.error("[generate] \(request.model): VLM model has no chat template — cannot process request")
                    throw NovaMLXError.inferenceFailed("Model '\(request.model)' does not have a chat template and is not compatible with text-only requests.")
                }
                // LLM models: fall back to plain text encoding
                NovaMLXLog.warning("[generate] \(request.model): no chat template — falling back to plain text encoding")
                let plainText = request.messages.compactMap { $0.content }.joined(separator: "\n\n")
                let tokenizer = await mlxContainer.tokenizer
                let tokens = tokenizer.encode(text: plainText)
                input = LMInput(tokens: MLXArray(tokens))
            } else {
                throw error
            }
        }

        let promptTokenCount = input.text.tokens.size
        NovaMLXLog.info("Encoded \(promptTokenCount) tokens via chat template")
        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Starting generation for \(request.model)")

        let maxTokens = request.maxTokens ?? container.config.maxTokens

        // Context window check with accurate token count (after chat template)
        let genSettings = settingsProvider?(request.model)
        let genContextLength = genSettings?.maxContextWindow ?? container.config.contextLength
        if Int(promptTokenCount) + maxTokens > genContextLength {
            throw NovaMLXError.contextWindowExceeded(
                promptTokens: Int(promptTokenCount), maxTokens: maxTokens, contextLength: genContextLength
            )
        }

        try await preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        let parameters = buildGenerateParameters(request: request, config: container.config, settings: settingsProvider?(request.model))

        // Capture prompt tokens for prefix cache storage + loading
        // VLM streaming builds a fresh cache inside perform — skip fetch/store.
        let isVLM = container.config.modelType == .vlm
        let allPromptTokens: [Int]?
        let prefixResult: PrefixCacheManager.PrefixResult?
        if !isVLM,
           let _ = getOrCreatePrefixCacheManager(modelId: request.model), promptTokenCount > 0 {
            let tokens = input.text.tokens.asArray(Int32.self).map { Int($0) }
            allPromptTokens = tokens
            prefixResult = getOrCreatePrefixCacheManager(modelId: request.model)?.fetchPrefix(tokenIds: tokens)
        } else {
            allPromptTokens = nil
            prefixResult = nil
        }

        // Build processor (repetition penalty only)
        let baseProcessor = parameters.processor()
        let model = await mlxContainer.perform { context in SendableBox(context.model) }
        let mlxTokenizer = await mlxContainer.tokenizer
        let modelConfig = await mlxContainer.configuration

        let processor = composeWithTurnStop(
            baseProcessor: baseProcessor,
            modelId: request.model,
            container: container,
            tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil),
            modelConfig: modelConfig,
            request: request,
            parameters: parameters
        )

        let sampler = parameters.sampler()

        // Build effective input and cache — use prefix cache hit if valid
        var effectiveInput = input
        let kvCache: [KVCache]
        if let result = prefixResult, result.cachedTokenCount > 0, let restored = result.cache {
            let freshCaches = model.value.newCache(parameters: parameters)
            var cacheValid = restored.count == freshCaches.count
            // Validate: layer types must match, KV cache states must be 4D with 2 arrays
            if cacheValid {
                for i in 0..<restored.count {
                    let restoredType = String(describing: type(of: restored[i]))
                    let freshType = String(describing: type(of: freshCaches[i]))
                    guard restoredType == freshType else {
                        NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: L\(i) type mismatch (\(restoredType) vs \(freshType))")
                        cacheValid = false
                        break
                    }
                    let isKV = restored[i] is KVCacheSimple
                        || restored[i] is RotatingKVCache
                        || restored[i] is ChunkedKVCache
                        || restored[i] is QuantizedKVCache
                    if isKV {
                        // KV caches: must have exactly 2 state arrays (keys, values), both 4D
                        let state = restored[i].state
                        guard state.count == 2 else {
                            NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: L\(i) KV state count=\(state.count), expected 2")
                            cacheValid = false
                            break
                        }
                        for j in 0..<2 {
                            guard state[j].ndim == 4 else {
                                NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: L\(i).S\(j) ndim=\(state[j].ndim), expected 4")
                                cacheValid = false
                                break
                            }
                        }
                    }
                    // Fixed-size caches (MambaCache, etc.): type match is sufficient
                    if !cacheValid { break }
                }
            }

            if cacheValid {
                let cachedTokenCount = result.cachedTokenCount
                // RotatingKVCache (sliding window) is incompatible with prefix cache:
                // offset > maxSize breaks invariants, and shared KV across layers
                // means any mismatched layer corrupts the entire attention.
                // Fall back to full prefill if any RotatingKVCache present.
                let hasRotating = restored.contains { $0 is RotatingKVCache }
                if hasRotating {
                    NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: model has RotatingKVCache layers — skipping (incompatible with sliding window)")
                    kvCache = model.value.newCache(parameters: parameters)
                } else {
                    for layer in restored {
                        if let kv = layer as? BaseKVCache { kv.offset = cachedTokenCount }
                    }
                    NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache HIT: \(cachedTokenCount) cached, \(result.remainingTokenIds.count) remaining")
                    let remainingTokens = MLXArray(result.remainingTokenIds.map { Int32($0) })
                    effectiveInput = LMInput(text: .init(tokens: remainingTokens))
                    kvCache = restored
                }
            } else {
                NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache: validation failed — using fresh cache")
                kvCache = model.value.newCache(parameters: parameters)
            }
        } else {
            kvCache = model.value.newCache(parameters: parameters)
        }
        let kvCacheBox = MutableSendableBox<[KVCache]?>(kvCache)

        // Try creating iterator with restored cache; fall back to fresh on error
        let iterator: any TokenIteratorProtocol
        if let draftModelId = request.draftModel,
           let draftContainer = getContainer(for: draftModelId),
           let draftMlxContainer = draftContainer.mlxContainer {
            // Speculative decoding with a draft model
            let draftModel = await draftMlxContainer.perform { context in SendableBox(context.model) }
            var draftParams = parameters
            draftParams.kvBits = nil  // No quantized KV for draft
            let draftCache = draftModel.value.newCache(parameters: draftParams)
            let numDraft = request.numDraftTokens ?? 4

            do {
                let specIterator = try SpeculativeTokenIterator(
                    input: effectiveInput,
                    mainModel: model.value,
                    draftModel: draftModel.value,
                    mainCache: kvCache,
                    draftCache: draftCache,
                    parameters: parameters,
                    numDraftTokens: numDraft
                )
                NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Speculative decoding: draft=\(draftModelId), numDraft=\(numDraft)")
                iterator = specIterator
            } catch {
                NovaMLXLog.warning("[GENERATE:\(request.id.uuidString.prefix(8))] SpeculativeTokenIterator failed: \(error.localizedDescription) — falling back to normal TokenIterator")
                do {
                    iterator = try TokenIterator(
                        input: effectiveInput, model: model.value,
                        cache: kvCache,
                        processor: processor, sampler: sampler,
                        maxTokens: maxTokens
                    )
                } catch {
                    NovaMLXLog.error("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache iterator failed: \(error.localizedDescription) — falling back to full prefill")
                    let freshCache = model.value.newCache(parameters: parameters)
                    kvCacheBox.value = freshCache
                    iterator = try TokenIterator(
                        input: input, model: model.value,
                        cache: freshCache,
                        processor: processor, sampler: sampler,
                        maxTokens: maxTokens
                    )
                }
            }
        } else {
            // CRITICAL: TokenIterator.init runs model.prepare (full prefill + first token
            // sampling). For VLM models and any model using compiled MLX functions, this
            // MUST happen inside mlxContainer.perform to ensure correct lazy evaluation.
            // Running outside perform produces NaN/Inf logits (see FusedBatchScheduler
            // prefillSequence for the same pattern).
            if isVLM {
                // VLM models require ALL model forward passes inside mlxContainer.perform.
                // Run the entire generation loop (prefill + decode) inside perform,
                // collect all token IDs, then build the response outside.
                let tokenIds = effectiveInput.text.tokens.asArray(Int32.self)
                let reqTag = request.id.uuidString.prefix(8)

                // Build EOS token set — §2.9: defensive family-canonical injection.
                let stopTokenIds: Set<Int> = {
                    var ids = Self.defensiveEosTokenIds(for: modelConfig, container: container)
                    if let eosId = container.tokenizer?.eosTokenId { ids.insert(eosId) }
                    return ids
                }()
                let stopIdsBox = SetBox(stopTokenIds)
                let unknownTokenId = mlxTokenizer.unknownTokenId

                // Wrap non-Sendable values for capture in @Sendable closure
                let cacheBox = MutableSendableBox<[KVCache]?>(kvCache)
                let samplerBox = SamplerBox(sampler)
                // VLM path runs entirely inside `mlxContainer.perform`, so we must
                // thread the LogitProcessor chain (penalty + TurnStopProcessor +
                // hallucination patterns) through manually — TokenIterator is not
                // used here. Without this, multi-token hallucination patterns
                // (e.g. "\nuser\n" turn-impersonation) and repetition penalty are
                // silently bypassed for VLM models. See VLM streaming path for the
                // mirror fix.
                let processorBox = ProcessorBox(processor)

                let genResult = await mlxContainer.perform { context in
                    let modelObj = context.model
                    let caches = cacheBox.value!

                    // Prefill
                    let inputTokens = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)
                    let logits = modelObj(inputTokens, cache: caches)
                    eval(logits)
                    eval(caches)

                    // Initialize processor with the prompt — needed for
                    // repetition-penalty token history and TurnStopProcessor state.
                    processorBox.value?.prompt(inputTokens)

                    // Sample first token (after running through processor chain)
                    let rawFirst = logits[0..., -1, 0...]
                    let logitsToSample = processorBox.value?.process(logits: rawFirst) ?? rawFirst
                    let maxLogit = logitsToSample.asType(.float32).max().item(Float.self)
                    let stats = "max=\(String(format: "%.2f", maxLogit))"
                    var currentToken = samplerBox.value.sample(logits: logitsToSample)
                    eval(currentToken)
                    processorBox.value?.didSample(token: currentToken)

                    var generatedIds: [Int] = []
                    let tid = currentToken.item(Int.self)
                    if tid == unknownTokenId || stopIdsBox.value.contains(tid) {
                        return SendableBox(([Int](), stats, caches as [KVCache]))
                    }
                    generatedIds.append(tid)

                    // Decode loop — reshape token to (1,1) like TokenIterator.next() does
                    var count = 1
                    while count < maxTokens {
                        let stepInput = currentToken.reshaped(1, 1)
                        let stepLogits = modelObj(stepInput, cache: caches)
                        eval(stepLogits)
                        let rawStep = stepLogits[0..., -1, 0...]
                        let processedLogits = processorBox.value?.process(logits: rawStep) ?? rawStep
                        let sampled = samplerBox.value.sample(logits: processedLogits)
                        eval(sampled)
                        processorBox.value?.didSample(token: sampled)
                        let tokenId = sampled.item(Int.self)
                        if tokenId == unknownTokenId || stopIdsBox.value.contains(tokenId) {
                            break
                        }
                        generatedIds.append(tokenId)
                        currentToken = sampled
                        count += 1
                    }

                    return SendableBox((generatedIds, stats, caches as [KVCache]))
                }

                let (generatedIds, logitStats, finalCaches) = genResult.value
                kvCacheBox.value = finalCaches
                NovaMLXLog.info("[Prefill:\(reqTag)] VLM — \(tokenIds.count) prompt, \(generatedIds.count) generated, logits=[\(logitStats)]")

                // Decode generated tokens to text
                let decodedText = mlxTokenizer.decode(tokenIds: generatedIds)
                let cleanText = Self.scrubControlTokens(decodedText)

                let elapsed = Date().timeIntervalSince(startTime)
                let tps = generatedIds.count > 0 ? Double(generatedIds.count) / elapsed : 0

                lock.withLock {
                    totalRequests += 1
                    totalTokensGenerated += UInt64(generatedIds.count)
                    totalInferenceTime += elapsed
                }

                deferredClearCache()

                return InferenceResult(
                    id: request.id,
                    model: request.model,
                    text: cleanText,
                    tokensPerSecond: tps,
                    promptTokens: Int(promptTokenCount),
                    completionTokens: generatedIds.count,
                    finishReason: generatedIds.count >= maxTokens ? .length : .stop
                )
            } else {
                do {
                    iterator = try TokenIterator(
                        input: effectiveInput, model: model.value,
                        cache: kvCache,
                        processor: processor, sampler: sampler,
                        maxTokens: maxTokens
                    )
                } catch {
                    NovaMLXLog.error("[GENERATE:\(request.id.uuidString.prefix(8))] Prefix cache iterator failed: \(error.localizedDescription) — falling back to full prefill")
                    let freshCache = model.value.newCache(parameters: parameters)
                    kvCacheBox.value = freshCache
                    iterator = try TokenIterator(
                        input: input, model: model.value,
                        cache: freshCache,
                        processor: processor, sampler: sampler,
                        maxTokens: maxTokens
                    )
                }
            }
        }

        let bytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
        let ticket = createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: maxTokens, bytesPerToken: bytesPerToken)

        let (stream, _) = generateTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfig,
            tokenizer: mlxTokenizer,
            iterator: iterator,
            wiredMemoryTicket: ticket
        )

        var generatedText = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var genStopReason: String = "none"
        var collectedToolCalls: [ToolCallResult] = []

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                completionTokens += 1
            case .info(let info):
                completionTokens = info.generationTokenCount
                genStopReason = String(describing: info.stopReason)
                if case .length = info.stopReason { finishReason = .length }
                NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Info: tokens=\(info.generationTokenCount), stopReason=\(genStopReason), tps=\(String(format: "%.1f", info.tokensPerSecond))")
            case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                collectedToolCalls.append(tcResult)
                finishReason = .toolCalls
            }
        }

        NovaMLXLog.info("[GENERATE:\(request.id.uuidString.prefix(8))] Done: completionTokens=\(completionTokens), stopReason=\(genStopReason)")

        // Store KV cache in prefix cache for future requests with same prompt prefix
        if let tokenArray = allPromptTokens,
           let prefixCache = getOrCreatePrefixCacheManager(modelId: request.model),
           let kvCache = kvCacheBox.value {
            prefixCache.storeCache(tokenIds: tokenArray, cache: kvCache)
        }

        deferredClearCache()

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        lock.withLock {
            totalRequests += 1
            totalTokensGenerated += UInt64(completionTokens)
            totalInferenceTime += elapsed
        }

        metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

        NovaMLXLog.info("Generated \(completionTokens) tokens (\(String(format: "%.1f", tps)) tok/s)")
        NovaMLXLog.request(request.id.uuidString.prefix(8).description, "Completed: \(completionTokens) tokens, \(String(format: "%.1f", tps)) tok/s")

        let trimmedText = Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: Self.controlTokensForModel(modelId: request.model)))

        return InferenceResult(
            id: request.id, model: request.model, text: trimmedText,
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
            tokensPerSecond: tps, promptTokens: promptTokenCount, completionTokens: completionTokens,
            finishReason: finishReason
        )
    }

    public func stream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
        // Dispatch to specialized paths based on request fields
        if let sessionId = request.sessionId {
            return streamWithSession(request, sessionId: sessionId)
        }
        if request.jsonSchemaDef != nil || request.responseFormat == .jsonObject ||
            request.regexPattern != nil || request.gbnfGrammar != nil {
            return streamWithProcessor(request)
        }

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let container = self.getContainer(for: request.model) else {
                        throw NovaMLXError.modelNotFound(request.model)
                    }
                    guard let mlxContainer = container.mlxContainer else {
                        throw NovaMLXError.inferenceFailed("Model not loaded")
                    }

                    self.lock.withLock { self.activeCount += 1 }

                    let hasImages = request.messages.contains { msg in
                        guard let imgs = msg.images, !imgs.isEmpty else { return false }
                        return true
                    }

                    let userInput = try await self.buildUserInput(request: request, hasImages: hasImages)

                    let input = try await mlxContainer.prepare(input: userInput)

                    let parameters = self.buildGenerateParameters(request: request, config: container.config, settings: self.settingsProvider?(request.model))
                    let promptTokenCount = input.text.tokens.size
                    let maxTokens = request.maxTokens ?? container.config.maxTokens

                    // Context window check with accurate token count (after chat template)
                    let streamSettings = self.settingsProvider?(request.model)
                    let streamContextLength = streamSettings?.maxContextWindow ?? container.config.contextLength
                    if Int(promptTokenCount) + maxTokens > streamContextLength {
                        throw NovaMLXError.contextWindowExceeded(
                            promptTokens: Int(promptTokenCount), maxTokens: maxTokens, contextLength: streamContextLength
                        )
                    }

                    try await self.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

                    let isVLM = container.config.modelType == .vlm

                    // Capture prompt tokens for prefix cache storage + loading
                    // VLM streaming builds a fresh cache inside perform — skip fetch/store.
                    let allPromptTokens: [Int]?
                    let prefixResult: PrefixCacheManager.PrefixResult?
                    if !isVLM,
                       let _ = self.getOrCreatePrefixCacheManager(modelId: request.model),
                       promptTokenCount > 0 {
                        let tokens = input.text.tokens.asArray(Int32.self).map { Int($0) }
                        allPromptTokens = tokens
                        prefixResult = self.getOrCreatePrefixCacheManager(modelId: request.model)?.fetchPrefix(tokenIds: tokens)
                    } else {
                        allPromptTokens = nil
                        prefixResult = nil
                    }

                    // Build processor (repetition penalty only)
                    let baseProcessor = parameters.processor()
                    let model = await mlxContainer.perform { context in SendableBox(context.model) }
                    let mlxTokenizer = await mlxContainer.tokenizer
                    let modelConfig = await mlxContainer.configuration

                    let processor = self.composeWithTurnStop(
                        baseProcessor: baseProcessor,
                        modelId: request.model,
                        container: container,
                        tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil),
                        modelConfig: modelConfig,
                        request: request,
                        parameters: parameters
                    )

                    let sampler = parameters.sampler()

                    // VLM streaming: run entire generation inside perform.
                    // TokenIterator runs model forward passes outside perform, producing
                    // bad logits for VLM models (NaN/Inf or premature EOS). The non-streaming
                    // VLM path already runs inside perform — replicate that for streaming.
                    if isVLM {
                        let tokenIds = input.text.tokens.asArray(Int32.self).map { Int($0) }
                        let reqTag = request.id.uuidString.prefix(8).description

                        let stopTokenIds: Set<Int> = {
                            // §2.9: defensive family-canonical injection (mirror of non-stream VLM path).
                            var ids = MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container)
                            if let eosId = container.tokenizer?.eosTokenId { ids.insert(eosId) }
                            return ids
                        }()

                        let startTime = Date()
                        let kvCache = model.value.newCache(parameters: parameters)
                        let cacheBox = MutableSendableBox<[KVCache]?>(kvCache)
                        let samplerBox = SamplerBox(sampler)
                        let unknownTokenId = mlxTokenizer.unknownTokenId
                        // VLM streaming runs the entire generation inside
                        // `mlxContainer.perform`, so we manually thread the
                        // LogitProcessor chain (repetition penalty +
                        // TurnStopProcessor multi-token hallucination detection).
                        // Without this, the chain built above by
                        // `engine.composeWithTurnStop` is silently dropped on
                        // VLM requests. Mirrors the non-stream VLM fix.
                        let processorBox = ProcessorBox(processor)

                        // Yield role chunk immediately
                        continuation.yield(Token(id: 0, text: ""))

                        var completionTokens = 0
                        var finishReason: FinishReason = .stop

                        let genResult = await mlxContainer.perform { context in
                            let modelObj = context.model
                            let caches = cacheBox.value!

                            // Prefill
                            let inputTokens = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)
                            let logits = modelObj(inputTokens, cache: caches)
                            eval(logits)
                            eval(caches)

                            processorBox.value?.prompt(inputTokens)

                            // Sample first token
                            let rawFirst = logits[0..., -1, 0...]
                            let logitsToSample = processorBox.value?.process(logits: rawFirst) ?? rawFirst
                            var currentToken = samplerBox.value.sample(logits: logitsToSample)
                            eval(currentToken)
                            processorBox.value?.didSample(token: currentToken)

                            var generatedIds: [Int] = []
                            let tid = currentToken.item(Int.self)
                            if tid == unknownTokenId || stopTokenIds.contains(tid) {
                                return SendableBox((generatedIds, caches as [KVCache]))
                            }
                            generatedIds.append(tid)

                            // Decode loop
                            var count = 1
                            while count < maxTokens {
                                if Task.isCancelled { break }
                                let stepInput = currentToken.reshaped(1, 1)
                                let stepLogits = modelObj(stepInput, cache: caches)
                                eval(stepLogits)
                                let rawStep = stepLogits[0..., -1, 0...]
                                let processedLogits = processorBox.value?.process(logits: rawStep) ?? rawStep
                                let sampled = samplerBox.value.sample(logits: processedLogits)
                                eval(sampled)
                                processorBox.value?.didSample(token: sampled)
                                let tokenId = sampled.item(Int.self)
                                if tokenId == unknownTokenId || stopTokenIds.contains(tokenId) {
                                    break
                                }
                                generatedIds.append(tokenId)
                                currentToken = sampled
                                count += 1
                            }

                            return SendableBox((generatedIds, caches as [KVCache]))
                        }

                        let (generatedIds, finalCaches) = genResult.value
                        cacheBox.value = finalCaches
                        completionTokens = generatedIds.count

                        // Stream generated tokens as text chunks
                        let fullText = mlxTokenizer.decode(tokenIds: generatedIds)
                        let cleanText = Self.scrubControlTokens(fullText)
                        if !cleanText.isEmpty {
                            continuation.yield(Token(id: 0, text: cleanText))
                        }

                        if completionTokens >= maxTokens { finishReason = .length }

                        let elapsed = Date().timeIntervalSince(startTime)
                        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

                        NovaMLXLog.info("[STREAM:\(reqTag)] VLM: tokens=\(completionTokens), stopReason=\(finishReason), tps=\(String(format: "%.1f", tps))")

                        self.lock.withLock {
                            self.totalRequests += 1
                            self.totalTokensGenerated += UInt64(completionTokens)
                            self.totalInferenceTime += elapsed
                        }
                        self.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                        continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                        continuation.finish()
                        self.lock.withLock { self.activeCount -= 1 }
                        self.deferredClearCache()
                        return
                    }

                    // Build effective input and cache — use prefix cache hit if valid
                    var effectiveInput = input
                    let kvCache: [KVCache]
                    if let result = prefixResult, result.cachedTokenCount > 0, let restored = result.cache {
                        let freshCaches = model.value.newCache(parameters: parameters)
                        var cacheValid = restored.count == freshCaches.count
                        if cacheValid {
                            for i in 0..<restored.count {
                                let rType = String(describing: type(of: restored[i]))
                                let fType = String(describing: type(of: freshCaches[i]))
                                guard rType == fType else { cacheValid = false; break }
                                let isKV = restored[i] is KVCacheSimple
                                    || restored[i] is RotatingKVCache
                                    || restored[i] is ChunkedKVCache
                                    || restored[i] is QuantizedKVCache
                                if isKV {
                                    let state = restored[i].state
                                    guard state.count == 2, state.allSatisfy({ $0.ndim == 4 }) else {
                                        cacheValid = false; break
                                    }
                                }
                                // Fixed-size caches: type match is sufficient
                                if !cacheValid { break }
                            }
                        }
                        if cacheValid {
                            let cachedTokenCount = result.cachedTokenCount
                            let hasRotating = restored.contains { $0 is RotatingKVCache }
                            if hasRotating {
                                NovaMLXLog.info("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache: model has RotatingKVCache layers — skipping (incompatible with sliding window)")
                                kvCache = model.value.newCache(parameters: parameters)
                            } else {
                                for layer in restored {
                                    if let kv = layer as? BaseKVCache { kv.offset = cachedTokenCount }
                                }
                                NovaMLXLog.info("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache HIT: \(cachedTokenCount) cached, \(result.remainingTokenIds.count) remaining")
                                let remainingTokens = MLXArray(result.remainingTokenIds.map { Int32($0) })
                                effectiveInput = LMInput(text: .init(tokens: remainingTokens))
                                kvCache = restored
                            }
                        } else {
                            NovaMLXLog.warning("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache: validation failed — using fresh cache")
                            kvCache = model.value.newCache(parameters: parameters)
                        }
                    } else {
                        kvCache = model.value.newCache(parameters: parameters)
                    }
                    let kvCacheBox = MutableSendableBox<[KVCache]?>(kvCache)

                    // Try creating iterator with restored cache; fall back to fresh on error
                    let iterator: any TokenIteratorProtocol
                    if let draftModelId = request.draftModel,
                       let draftContainer = self.getContainer(for: draftModelId),
                       let draftMlxContainer = draftContainer.mlxContainer {
                        // Speculative decoding with a draft model
                        let draftModel = await draftMlxContainer.perform { context in SendableBox(context.model) }
                        var draftParams = parameters
                        draftParams.kvBits = nil
                        let draftCache = draftModel.value.newCache(parameters: draftParams)
                        let numDraft = request.numDraftTokens ?? 4

                        do {
                            let specIterator = try SpeculativeTokenIterator(
                                input: effectiveInput,
                                mainModel: model.value,
                                draftModel: draftModel.value,
                                mainCache: kvCache,
                                draftCache: draftCache,
                                parameters: parameters,
                                numDraftTokens: numDraft
                            )
                            NovaMLXLog.info("[STREAM:\(request.id.uuidString.prefix(8))] Speculative decoding: draft=\(draftModelId), numDraft=\(numDraft)")
                            iterator = specIterator
                        } catch {
                            NovaMLXLog.warning("[STREAM:\(request.id.uuidString.prefix(8))] SpeculativeTokenIterator failed: \(error.localizedDescription) — falling back to normal TokenIterator")
                            do {
                                iterator = try TokenIterator(
                                    input: effectiveInput, model: model.value,
                                    cache: kvCache,
                                    processor: processor, sampler: sampler,
                                    maxTokens: maxTokens
                                )
                            } catch {
                                NovaMLXLog.error("[STREAM:\(request.id.uuidString.prefix(8))] TokenIterator failed: \(error.localizedDescription) — falling back to full prefill")
                                let freshCache = model.value.newCache(parameters: parameters)
                                kvCacheBox.value = freshCache
                                iterator = try TokenIterator(
                                    input: input, model: model.value,
                                    cache: freshCache,
                                    processor: processor, sampler: sampler,
                                    maxTokens: maxTokens
                                )
                            }
                        }
                    } else {
                        do {
                            iterator = try TokenIterator(
                                input: effectiveInput, model: model.value,
                                cache: kvCache,
                                processor: processor, sampler: sampler,
                                maxTokens: maxTokens
                            )
                        } catch {
                            NovaMLXLog.error("[STREAM:\(request.id.uuidString.prefix(8))] Prefix cache iterator failed: \(error.localizedDescription) — falling back to full prefill")
                            let freshCache = model.value.newCache(parameters: parameters)
                            kvCacheBox.value = freshCache
                            iterator = try TokenIterator(
                                input: input, model: model.value,
                                cache: freshCache,
                                processor: processor, sampler: sampler,
                                maxTokens: maxTokens
                            )
                        }
                    }

                    let ticketBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262144
                    let ticket = self.createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: maxTokens, bytesPerToken: ticketBytesPerToken)
                    let (stream, _) = generateTask(
                        promptTokenCount: promptTokenCount,
                        modelConfiguration: modelConfig,
                        tokenizer: mlxTokenizer,
                        iterator: iterator,
                        wiredMemoryTicket: ticket
                    )

                    var finishReason: FinishReason = .stop
                    var completionTokens = 0
                    var infoStopReason: String = "none"
                    var streamAccumulated = ""
                    var streamYieldedCount = 0
                    let controlPatterns = Self.controlTokensForModel(modelId: request.model)
                    let startTime = Date()
                    let reqTag = request.id.uuidString.prefix(8).description

                    for await generation in stream {
                        if Task.isCancelled {
                            NovaMLXLog.error("[STREAM:\(reqTag)] Task cancelled during generation after \(completionTokens) tokens")
                            break
                        }
                        switch generation {
                        case .chunk(let text):
                            let (rawText, shouldStop) = Self.filterControlInChunk(text, accumulated: &streamAccumulated, yieldedCount: &streamYieldedCount, patterns: controlPatterns)
                            let cleanText = Self.scrubControlTokens(rawText)
                            if text.contains("<|turn") {
                                NovaMLXLog.warning("[STREAM:\(reqTag)] <|turn> in chunk: raw=\(rawText.count) clean=\(cleanText.count) stop=\(shouldStop) patterns=\(controlPatterns.prefix(3))")
                            }
                            if !cleanText.isEmpty {
                                completionTokens += 1
                                continuation.yield(Token(id: 0, text: cleanText))
                            }
                            if shouldStop { break }
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            infoStopReason = String(describing: info.stopReason)
                            if case .length = info.stopReason { finishReason = .length }
                            NovaMLXLog.info("[STREAM:\(reqTag)] Info: tokens=\(info.generationTokenCount), stopReason=\(infoStopReason), tps=\(String(format: "%.1f", info.tokensPerSecond))")
                        case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                continuation.yield(Token(id: 0, text: "", toolCall: tcResult))
                        }
                    }

                    NovaMLXLog.info("[STREAM:\(reqTag)] Loop ended: completionTokens=\(completionTokens), taskCancelled=\(Task.isCancelled), infoStopReason=\(infoStopReason)")

                    // Store KV cache in prefix cache for future requests
                    if let tokenArray = allPromptTokens,
                       let prefixCache = self.getOrCreatePrefixCacheManager(modelId: request.model),
                       let kvCache = kvCacheBox.value {
                        prefixCache.storeCache(tokenIds: tokenArray, cache: kvCache)
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    self.lock.withLock {
                        self.totalRequests += 1
                        self.totalTokensGenerated += UInt64(completionTokens)
                        self.totalInferenceTime += elapsed
                    }

                    self.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                    continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }

                self.lock.withLock { self.activeCount -= 1 }
                self.deferredClearCache()
            }

            self.lock.withLock { self.activeTasks[request.id] = task }

            continuation.onTermination = { _ in
                task.cancel()
                _ = self.lock.withLock { self.activeTasks.removeValue(forKey: request.id) }
            }
        }
    }

    public func abort(requestId: UUID) {
        lock.withLock {
            activeTasks[requestId]?.cancel()
            activeTasks.removeValue(forKey: requestId)
        }
        NovaMLXLog.info("Aborting request: \(requestId)")
    }

    public var currentActiveCount: Int { lock.withLock { activeCount } }
    public var loadedModelCount: Int { pool.loadedModelCount }
    public func listLoadedModels() -> [String] { pool.loadedModelIds }
    public var gpuActiveMemory: UInt64 { UInt64(MLX.Memory.activeMemory) }

    /// Wire settings provider into the memory enforcer and start 1s polling.
    /// Call once after config is loaded. The abortAllHandler cancels active
    /// inference tasks so KV cache is freed in single-model scenarios.
    public func startMemoryEnforcer() async {
        guard let enforcer = memoryEnforcer else { return }
        await enforcer.setAbortAllHandler { [weak self] in
            guard let self else { return }
            let tasks = self.lock.withLock { self.activeTasks.values }
            for task in tasks { task.cancel() }
            self.lock.withLock { self.activeTasks.removeAll() }
        }
        await enforcer.start()
    }

    /// Update settings provider for TTL checks in the enforcer.
    public func configureEnforcerSettings(_ provider: @escaping @Sendable (String) -> ModelSettings?) async {
        self.settingsProvider = provider
        await memoryEnforcer?.updateSettingsProvider(provider)
    }

    public func saveSession(_ sessionId: String) async throws {
        try await sessionManager.saveSession(sessionId)
    }

    public func hasCachedSession(_ sessionId: String) -> Bool {
        FileManager.default.fileExists(atPath: sessionManager.cacheURL(for: sessionId).path)
    }

    public func forkSession(from sourceId: String, into targetId: String, modelId: String) async throws {
        guard let container = getContainer(for: modelId),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(modelId)
        }
        try await sessionManager.fork(
            from: sourceId,
            into: targetId,
            mlxContainer: mlxContainer,
            kvBits: container.config.kvBits,
            kvGroupSize: container.config.kvGroupSize
        )
    }

    public func generateWithSession(
        _ request: InferenceRequest,
        sessionId: String
    ) async throws -> InferenceResult {
        guard let container = getContainer(for: request.model),
              let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.modelNotFound(request.model)
        }

        let lastUserMessage = request.messages.last(where: { $0.role == .user })?.content ?? ""
        let systemPrompt = request.messages.first(where: { $0.role == .system })?.content

        let estimatedPromptTokens = tokenizeMessages(request.messages, tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil)).count
        let maxTokens = request.maxTokens ?? container.config.maxTokens
        try await preflightCheck(modelId: request.model, promptTokens: estimatedPromptTokens, maxTokens: maxTokens)

        let sessionBox = sessionManager.getOrCreate(
            sessionId: sessionId,
            mlxContainer: mlxContainer,
            modelId: request.model,
            systemPrompt: systemPrompt,
            kvBits: container.config.kvBits,
            kvGroupSize: container.config.kvGroupSize
        )

        let startTime = Date()
        var generatedText = ""
        var promptTokens = 0
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var collectedToolCalls: [ToolCallResult] = []

        let stream = sessionBox.streamDetails(to: lastUserMessage)
        for try await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                collectedToolCalls.append(tcResult)
                finishReason = .toolCalls
            }
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        lock.withLock {
            totalRequests += 1
            totalTokensGenerated += UInt64(completionTokens)
            totalInferenceTime += elapsed
        }

        metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

        return InferenceResult(
            id: request.id, model: request.model, text: Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: Self.controlTokensForModel(modelId: request.model))),
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
            tokensPerSecond: tps, promptTokens: promptTokens,
            completionTokens: completionTokens, finishReason: finishReason
        )
    }

    public func streamWithSession(
        _ request: InferenceRequest,
        sessionId: String
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let engine = self

            let task = Task {
                do {
                    guard let container = engine.getContainer(for: request.model),
                          let mlxContainer = container.mlxContainer else {
                        throw NovaMLXError.modelNotFound(request.model)
                    }

                    let lastUserMessage = request.messages.last(where: { $0.role == .user })?.content ?? ""
                    let systemPrompt = request.messages.first(where: { $0.role == .system })?.content

                    let estimatedPromptTokens = engine.tokenizeMessages(request.messages, tokenizer: container.tokenizer ?? Tokenizer(encode: { _ in [] }, decode: { _ in "" }, eosToken: nil, eosTokenId: nil)).count
                    let maxTokens = request.maxTokens ?? container.config.maxTokens
                    try await engine.preflightCheck(modelId: request.model, promptTokens: estimatedPromptTokens, maxTokens: maxTokens)

                    let sessionBox = engine.sessionManager.getOrCreate(
                        sessionId: sessionId,
                        mlxContainer: mlxContainer,
                        modelId: request.model,
                        systemPrompt: systemPrompt,
                        kvBits: container.config.kvBits,
                        kvGroupSize: container.config.kvGroupSize
                    )

                    let startTime = Date()
                    var finishReason: FinishReason = .stop
                    var completionTokens = 0
                    var streamAccumulated = ""
                    var streamYieldedCount = 0
                    let controlPatterns = Self.controlTokensForModel(modelId: request.model)

                    let stream = sessionBox.streamDetails(to: lastUserMessage)
                    for try await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            let (rawText, shouldStop) = Self.filterControlInChunk(text, accumulated: &streamAccumulated, yieldedCount: &streamYieldedCount, patterns: controlPatterns)
                            let cleanText = Self.scrubControlTokens(rawText)
                            if !cleanText.isEmpty {
                                completionTokens += 1
                                continuation.yield(Token(id: 0, text: cleanText))
                            }
                            if shouldStop { break }
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                continuation.yield(Token(id: 0, text: "", toolCall: tcResult))
                        }
                    }

                    let elapsed = Date().timeIntervalSince(startTime)
                    engine.lock.withLock {
                        engine.totalRequests += 1
                        engine.totalTokensGenerated += UInt64(completionTokens)
                        engine.totalInferenceTime += elapsed
                    }

                    engine.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                    continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    public var metrics: EngineMetrics {
        lock.withLock {
            EngineMetrics(
                totalRequests: totalRequests,
                totalTokensGenerated: totalTokensGenerated,
                totalInferenceTime: totalInferenceTime,
                averageTokensPerSecond: totalInferenceTime > 0 ? Double(totalTokensGenerated) / totalInferenceTime : 0,
                activeRequests: activeCount
            )
        }
    }

    public func resetSessionMetrics() {
        lock.withLock {
            totalRequests = 0
            totalTokensGenerated = 0
            totalInferenceTime = 0
        }
    }

    public func generateWithProcessor(
        _ request: InferenceRequest
    ) async throws -> InferenceResult {
        guard let container = getContainer(for: request.model) else {
            throw NovaMLXError.modelNotFound(request.model)
        }
        guard let mlxContainer = container.mlxContainer else {
            throw NovaMLXError.inferenceFailed("Model not loaded")
        }
        guard let tokenizer = container.tokenizer else {
            throw NovaMLXError.inferenceFailed("Tokenizer not available for \(request.model)")
        }

        let sharedMaskBuilder = await tokenMaskBuilderCache.builder(
            for: request.model, tokenizer: tokenizer
        )
        let modelConfig = await mlxContainer.configuration
        // Build full EOS set: tokenizer primary + model config extras (e.g. Qwen3.6 has [248046, 248044])
        // §2.9: defensive family-canonical injection on top of the upstream set.
        var allEosIds = Set<Int>()
        if let eosId = tokenizer.eosTokenId { allEosIds.insert(eosId) }
        allEosIds = allEosIds.union(Self.defensiveEosTokenIds(for: modelConfig, container: container))

        let grammarProcessor: any LogitProcessor
        if let schema = request.jsonSchemaDef {
            grammarProcessor = SchemaGuidedProcessor(
                schema: schema, tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder, allEosTokenIds: allEosIds
            )
        } else if let pattern = request.regexPattern {
            grammarProcessor = try RegexLogitProcessor(
                pattern: pattern, tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder
            )
        } else if let grammar = request.gbnfGrammar {
            grammarProcessor = try GBNFLogitProcessor(
                grammar: grammar, tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder
            )
        } else {
            grammarProcessor = JSONLogitProcessor(
                tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder, allEosTokenIds: allEosIds
            )
        }

        let parameters = buildGenerateParameters(request: request, config: container.config, settings: settingsProvider?(request.model))
        let penaltyProcessor = parameters.processor()

        let turnStopProc = buildTurnStopProcessor(
            modelId: request.model,
            container: container,
            tokenizer: tokenizer,
            modelConfig: modelConfig
        )

        let processor: any LogitProcessor
        if let penalty = penaltyProcessor, let tsp = turnStopProc {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty, turnStopProcessor: tsp)
        } else if let penalty = penaltyProcessor {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty)
        } else if let tsp = turnStopProc {
            processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: nil, turnStopProcessor: tsp)
        } else {
            processor = grammarProcessor
        }

        var systemMessages = request.messages.filter { $0.role == .system }
        if request.responseFormat == .jsonObject {
            systemMessages.append(ChatMessage(role: .system, content: "You must respond with valid JSON only. Do not include any text, explanation, or markdown outside the JSON object."))
        }
        let allMessages = systemMessages + request.messages.filter { $0.role != .system }
        let mappedMessages: [Message] = allMessages.map { msg in
            // Mirror the rich mapping from `buildUserInput` so tool-call assistant
            // messages and tool-result messages preserve their `tool_calls` /
            // `tool_call_id` / `name` fields. Without this, multi-turn tool
            // conversations on the JSON-mode path lose all tool context after
            // the first round.
            var m: Message = ["role": msg.role.rawValue, "content": msg.content ?? ""]
            if let tcId = msg.toolCallId { m["tool_call_id"] = tcId }
            if let name = msg.name { m["name"] = name }
            if let tcs = msg.toolCalls {
                m["tool_calls"] = tcs.map { tc in
                    ["function": ["name": tc.functionName, "arguments": tc.arguments]] as [String: any Sendable]
                }
            }
            return m
        }
        // Tool definitions must be plumbed into the chat template the same way
        // `buildUserInput` does — otherwise swift-transformers receives
        // `tools=nil` and the Jinja `{%- if tools %}` branch is skipped, so the
        // model never sees tool schemas. Pre-fix: tool-using requests on the
        // JSON-mode path produced ~25-token prompts (system + user only) and
        // never invoked tools.
        let toolSpecs: [[String: any Sendable]]? = request.tools?.map { tool in
            tool.mapValues { toSendable($0) }
        }

        // Grammar/JSON mode is incompatible with thinking tokens — the JSON FSM
        // starts at .expectValue which doesn't allow < or > (thinking markers).
        // Force disable thinking so the model outputs JSON directly.
        var additionalContext: [String: any Sendable]? = nil
        var ctx: [String: any Sendable] = [:]
        ctx["enable_thinking"] = false
        if let preserveThinking = request.preserveThinking {
            ctx["preserve_thinking"] = preserveThinking
        }
        if !ctx.isEmpty { additionalContext = ctx }

        let userInput = UserInput(prompt: .messages(mappedMessages), tools: toolSpecs, additionalContext: additionalContext)
        let input = try await mlxContainer.prepare(input: userInput)

        let promptTokenCount = input.text.tokens.size
        let maxTokens = request.maxTokens ?? container.config.maxTokens
        try await preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

        // Use parameters.sampler() so temperature=0 returns ArgmaxSampler.
        let sampler = parameters.sampler()

        let model = await mlxContainer.perform { context in SendableBox(context.model) }
        let mlxTokenizer = await mlxContainer.tokenizer

        let iterator = try TokenIterator(
            input: input, model: model.value,
            processor: processor, sampler: sampler,
            maxTokens: request.maxTokens ?? container.config.maxTokens
        )

        let ticketBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
        let (stream, task) = generateTask(
            promptTokenCount: promptTokenCount,
            modelConfiguration: modelConfig,
            tokenizer: mlxTokenizer,
            iterator: iterator,
            wiredMemoryTicket: createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: request.maxTokens ?? container.config.maxTokens, bytesPerToken: ticketBytesPerToken)
        )

        let startTime = Date()
        var generatedText = ""
        var completionTokens = 0
        var finishReason: FinishReason = .stop
        var collectedToolCalls: [ToolCallResult] = []

        for await generation in stream {
            switch generation {
            case .chunk(let text):
                generatedText += text
                completionTokens += 1
            case .info(let info):
                completionTokens = info.generationTokenCount
                if case .length = info.stopReason { finishReason = .length }
            case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                collectedToolCalls.append(tcResult)
                finishReason = .toolCalls
            }
        }
        _ = task

        deferredClearCache()

        let elapsed = Date().timeIntervalSince(startTime)
        let tps = completionTokens > 0 ? Double(completionTokens) / elapsed : 0

        lock.withLock {
            totalRequests += 1
            totalTokensGenerated += UInt64(completionTokens)
            totalInferenceTime += elapsed
        }

        metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

        return InferenceResult(
            id: request.id, model: request.model, text: Self.scrubControlTokens(Self.trimControlTokens(generatedText, patterns: Self.controlTokensForModel(modelId: request.model))),
            toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls,
            tokensPerSecond: tps, promptTokens: promptTokenCount,
            completionTokens: completionTokens, finishReason: finishReason
        )
    }

    public func streamWithProcessor(
        _ request: InferenceRequest
    ) -> AsyncThrowingStream<Token, Error> {
        AsyncThrowingStream { continuation in
            let engine = self

            let task = Task {
                do {
                    guard let container = engine.getContainer(for: request.model),
                          let mlxContainer = container.mlxContainer else {
                        throw NovaMLXError.modelNotFound(request.model)
                    }
                    guard let tokenizer = container.tokenizer else {
                        throw NovaMLXError.inferenceFailed("Tokenizer not available for \(request.model)")
                    }

                    let sharedMaskBuilder = await engine.tokenMaskBuilderCache.builder(
                        for: request.model, tokenizer: tokenizer
                    )
                    let modelConfig = await mlxContainer.configuration
                    // Build full EOS set (see generateWithProcessor for rationale).
                    // §2.9: defensive family-canonical injection on top of the upstream set.
                    var allEosIds = Set<Int>()
                    if let eosId = tokenizer.eosTokenId { allEosIds.insert(eosId) }
                    allEosIds = allEosIds.union(MLXEngine.defensiveEosTokenIds(for: modelConfig, container: container))

                    let grammarProcessor: any LogitProcessor
                    if let schema = request.jsonSchemaDef {
                        grammarProcessor = SchemaGuidedProcessor(
                            schema: schema, tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder, allEosTokenIds: allEosIds
                        )
                    } else if let pattern = request.regexPattern {
                        grammarProcessor = try RegexLogitProcessor(
                            pattern: pattern, tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder
                        )
                    } else if let grammar = request.gbnfGrammar {
                        grammarProcessor = try GBNFLogitProcessor(
                            grammar: grammar, tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder
                        )
                    } else {
                        grammarProcessor = JSONLogitProcessor(
                            tokenizer: tokenizer, sharedBuilder: sharedMaskBuilder, allEosTokenIds: allEosIds
                        )
                    }

                    let parameters = engine.buildGenerateParameters(request: request, config: container.config, settings: engine.settingsProvider?(request.model))
                    let penaltyProc = parameters.processor()

                    let turnStopProc = engine.buildTurnStopProcessor(
                        modelId: request.model,
                        container: container,
                        tokenizer: tokenizer,
                        modelConfig: modelConfig
                    )

                    let processor: any LogitProcessor
                    if let penalty = penaltyProc, let tsp = turnStopProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty, turnStopProcessor: tsp)
                    } else if let penalty = penaltyProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: penalty)
                    } else if let tsp = turnStopProc {
                        processor = ComposedLogitProcessor(grammarProcessor: grammarProcessor, penaltyProcessor: nil, turnStopProcessor: tsp)
                    } else {
                        processor = grammarProcessor
                    }

                    var systemMessages = request.messages.filter { $0.role == .system }
                    if request.responseFormat == .jsonObject {
                        systemMessages.append(ChatMessage(role: .system, content: "You must respond with valid JSON only. Do not include any text, explanation, or markdown outside the JSON object."))
                    }
                    let allMessages = systemMessages + request.messages.filter { $0.role != .system }
                    let mappedMessages: [Message] = allMessages.map { msg in
                        // See generateWithProcessor for rationale — preserve
                        // tool_calls / tool_call_id / name across the JSON-mode
                        // streaming path.
                        var m: Message = ["role": msg.role.rawValue, "content": msg.content ?? ""]
                        if let tcId = msg.toolCallId { m["tool_call_id"] = tcId }
                        if let name = msg.name { m["name"] = name }
                        if let tcs = msg.toolCalls {
                            m["tool_calls"] = tcs.map { tc in
                                ["function": ["name": tc.functionName, "arguments": tc.arguments]] as [String: any Sendable]
                            }
                        }
                        return m
                    }
                    // Plumb tools into the chat template (see generateWithProcessor).
                    let toolSpecs: [[String: any Sendable]]? = request.tools?.map { tool in
                        tool.mapValues { engine.toSendable($0) }
                    }

                    // Grammar/JSON mode — force disable thinking (see generateWithProcessor)
                    var additionalContext: [String: any Sendable]? = nil
                    var ctx: [String: any Sendable] = [:]
                    ctx["enable_thinking"] = false
                    if let preserveThinking = request.preserveThinking {
                        ctx["preserve_thinking"] = preserveThinking
                    }
                    if !ctx.isEmpty { additionalContext = ctx }

                    let userInput = UserInput(prompt: .messages(mappedMessages), tools: toolSpecs, additionalContext: additionalContext)
                    let input = try await mlxContainer.prepare(input: userInput)
                    let promptTokenCount = input.text.tokens.size
                    let maxTokens = request.maxTokens ?? container.config.maxTokens
                    try await engine.preflightCheck(modelId: request.model, promptTokens: Int(promptTokenCount), maxTokens: maxTokens)

                    let sampler = parameters.sampler()

                    let model = await mlxContainer.perform { context in SendableBox(context.model) }
                    let mlxTokenizer = await mlxContainer.tokenizer

                    let iterator = try TokenIterator(
                        input: input, model: model.value,
                        processor: processor, sampler: sampler,
                        maxTokens: request.maxTokens ?? container.config.maxTokens
                    )

                    let sessionBytesPerToken = container.kvMemoryOverride ?? container.config.kvMemoryBytesPerToken ?? 262_144
                    let (stream, genTask) = generateTask(
                        promptTokenCount: promptTokenCount,
                        modelConfiguration: modelConfig,
                        tokenizer: mlxTokenizer,
                        iterator: iterator,
                        wiredMemoryTicket: engine.createGenerationTicket(promptTokens: Int(promptTokenCount), maxTokens: request.maxTokens ?? container.config.maxTokens, bytesPerToken: sessionBytesPerToken)
                    )

                    let startTime = Date()
                    var finishReason: FinishReason = .stop
                    var completionTokens = 0
                    var streamAccumulated = ""
                    var streamYieldedCount = 0
                    let controlPatterns = Self.controlTokensForModel(modelId: request.model)

                    for await generation in stream {
                        if Task.isCancelled { break }
                        switch generation {
                        case .chunk(let text):
                            let (rawText, shouldStop) = Self.filterControlInChunk(text, accumulated: &streamAccumulated, yieldedCount: &streamYieldedCount, patterns: controlPatterns)
                            let cleanText = Self.scrubControlTokens(rawText)
                            if !cleanText.isEmpty {
                                completionTokens += 1
                                continuation.yield(Token(id: 0, text: cleanText))
                            }
                            if shouldStop { break }
                        case .info(let info):
                            completionTokens = info.generationTokenCount
                            if case .length = info.stopReason { finishReason = .length }
                        case .toolCall(let tc):
                let argsData = try? JSONSerialization.data(withJSONObject: tc.function.arguments.mapValues { $0.anyValue }, options: [])
                let tcResult = ToolCallResult(
                    id: "call_\(UUID().uuidString.prefix(8).lowercased())",
                    functionName: tc.function.name,
                    arguments: argsData.map { String(decoding: $0, as: UTF8.self) } ?? "{}"
                )
                continuation.yield(Token(id: 0, text: "", toolCall: tcResult))
                        }
                    }
                    _ = genTask

                    let elapsed = Date().timeIntervalSince(startTime)
                    engine.lock.withLock {
                        engine.totalRequests += 1
                        engine.totalTokensGenerated += UInt64(completionTokens)
                        engine.totalInferenceTime += elapsed
                    }

                    engine.metricsStore.recordRequest(model: request.model, tokens: UInt64(completionTokens), inferenceTime: elapsed)

                    self.deferredClearCache()

                    continuation.yield(Token(id: 0, text: "", finishReason: finishReason))
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { _ in task.cancel() }
        }
    }
}

public struct EngineMetrics: Sendable {
    public let totalRequests: UInt64
    public let totalTokensGenerated: UInt64
    public let totalInferenceTime: TimeInterval
    public let averageTokensPerSecond: Double
    public let activeRequests: Int

    public init(totalRequests: UInt64 = 0, totalTokensGenerated: UInt64 = 0, totalInferenceTime: TimeInterval = 0, averageTokensPerSecond: Double = 0, activeRequests: Int = 0) {
        self.totalRequests = totalRequests
        self.totalTokensGenerated = totalTokensGenerated
        self.totalInferenceTime = totalInferenceTime
        self.averageTokensPerSecond = averageTokensPerSecond
        self.activeRequests = activeRequests
    }
}

// MARK: - VLM Token Iterators

/// Wrapper to capture non-Sendable LogitSampler in @Sendable closures
private final class SamplerBox: @unchecked Sendable {
    let value: any LogitSampler
    init(_ value: any LogitSampler) { self.value = value }
}

/// Wrapper to capture non-Sendable LogitProcessor in @Sendable closures
private final class ProcessorBox: @unchecked Sendable {
    var value: (any LogitProcessor)?
    init(_ value: (any LogitProcessor)?) { self.value = value }
}

/// Wrapper for Set<Int> capture in @Sendable closures
private final class SetBox: @unchecked Sendable {
    let value: Set<Int>
    init(_ value: Set<Int>) { self.value = value }
}
