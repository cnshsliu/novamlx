import GRDB

enum ConfigDBSchema {
    enum v1 {
        static func createAll(in db: Database) throws {
            try db.create(table: "config") { t in
                t.column("id", .integer).primaryKey()
                t.column("host", .text).notNull().defaults(to: "0.0.0.0")
                t.column("port", .integer).notNull().defaults(to: 6590)
                t.column("admin_port", .integer).notNull().defaults(to: 6591)
                t.column("tls_enabled", .boolean).notNull().defaults(to: false)
                t.column("tls_cert_path", .text)
                t.column("tls_key_path", .text)
                t.column("default_model", .text)
                t.column("models_dir", .text)
                t.column("hf_endpoint", .text).defaults(to: "https://huggingface.co")
                t.column("auth_url", .text)
                t.column("tknet_api_key", .text)
                t.column("cluster_config", .text).defaults(to: "{}")
                t.column("auto_load", .text).defaults(to: "{}")
                t.column("log_level", .text).defaults(to: "info")
            }

            try db.create(table: "api_keys") { t in
                t.column("id", .text).primaryKey()
                t.column("name", .text).notNull()
                t.column("key_hash", .text).notNull()
                t.column("raw_key", .text).notNull()
                t.column("key_prefix", .text).notNull()
                t.column("key_suffix", .text).notNull().defaults(to: "")
                t.column("created_at", .datetime).notNull()
                t.column("expires_at", .datetime)
                t.column("is_enabled", .boolean).notNull().defaults(to: true)
                t.column("rate_limit_per_second", .double)
                t.column("rate_limit_burst", .integer)
                t.column("allowed_models", .text) // JSON array
                t.column("allowed_endpoints", .text) // JSON array
                t.column("max_tokens_per_period", .integer)
                t.column("max_requests_per_period", .integer)
                t.column("usage_reset_period", .text).notNull().defaults(to: "daily")
                t.column("total_tokens_used", .integer).notNull().defaults(to: 0)
                t.column("total_requests", .integer).notNull().defaults(to: 0)
                t.column("last_used_at", .datetime)
                t.column("period_tokens", .integer).notNull().defaults(to: 0)
                t.column("period_requests", .integer).notNull().defaults(to: 0)
                t.column("period_reset_date", .text)
                t.column("per_model_tokens", .text).defaults(to: "{}") // JSON
            }
            try db.create(index: "idx_api_keys_hash", on: "api_keys", columns: ["key_hash"])

            try db.create(table: "model_settings") { t in
                t.column("model_id", .text).primaryKey()
                t.column("alias", .text)
                t.column("is_default", .boolean).notNull().defaults(to: false)
                t.column("is_pinned", .boolean).notNull().defaults(to: false)
                t.column("sampling_params", .text).defaults(to: "{}") // JSON
                t.column("ttl_seconds", .integer)
                t.column("context_window", .integer)
                t.column("draft_model", .text)
                t.column("updated_at", .datetime)
            }

            try db.create(table: "tokenhub_providers") { t in
                t.column("name", .text).primaryKey()
                t.column("endpoint", .text).notNull()
                t.column("api_key", .text)
                t.column("remote_model", .text)
                t.column("is_enabled", .boolean).notNull().defaults(to: true)
                t.column("is_managed", .boolean).notNull().defaults(to: false)
                t.column("load_balance_weight", .double).defaults(to: 1.0)
                t.column("total_requests", .integer).notNull().defaults(to: 0)
                t.column("total_tokens", .integer).notNull().defaults(to: 0)
                t.column("avg_latency_ms", .double)
                t.column("last_used_at", .datetime)
                t.column("extra_config", .text).defaults(to: "{}")
            }

            try db.create(table: "modelfiles") { t in
                t.column("name", .text).primaryKey()
                t.column("base_model", .text)
                t.column("system_prompt", .text)
                t.column("parameters", .text).defaults(to: "{}") // JSON
                t.column("tools", .text).defaults(to: "[]") // JSON array
                t.column("created_at", .datetime).notNull()
                t.column("updated_at", .datetime)
            }

            try db.create(table: "auth_session") { t in
                t.column("id", .integer).primaryKey()
                t.column("session_token", .text).defaults(to: "")
                t.column("auth_valid", .boolean).defaults(to: false)
                t.column("auth_plan", .text)
                t.column("auth_status", .text)
                t.column("auth_cancel_at_period_end", .boolean).defaults(to: false)
                t.column("auth_expires_at", .datetime)
                t.column("auth_cached_at", .datetime)
                t.column("user_email", .text)
            }

            try db.create(table: "cluster_policy") { t in
                t.column("id", .integer).primaryKey()
                t.column("policy_json", .text).defaults(to: "{}")
                t.column("updated_at", .datetime)
            }
        }
    }

    /// Adds columns required to fully represent `ServerConfig` (see
    /// Sources/NovaMLXCore/Types.swift). NOT NULL columns get defaults so
    /// existing v1 rows upgrade cleanly.
    static func v2AddServerFields(in db: Database) throws {
        try db.alter(table: "config") { t in
            t.add(column: "max_concurrent_requests", .integer).notNull().defaults(to: 16)
            t.add(column: "request_timeout", .double).notNull().defaults(to: 300.0)
            t.add(column: "context_scaling_target", .integer)
            t.add(column: "tls_key_password", .text)
            t.add(column: "max_request_size_mb", .double).notNull().defaults(to: 100.0)
            t.add(column: "max_process_memory", .text).notNull().defaults(to: "auto")
            t.add(column: "prefix_cache_enabled", .boolean).notNull().defaults(to: true)
        }
    }

    /// Brings `tokenhub_providers` to parity with the JSON-shape
    /// `TokenhubProvider` struct (Sources/NovaMLXCore/TokenhubTypes.swift).
    /// Adds 15 new columns; existing v1 columns stay (some become deprecated
    /// but remain for back-compat — Final Cleanup will drop them once C2-C4
    /// finish). NOT NULL columns backfill with sensible defaults so existing
    /// v1 rows upgrade cleanly.
    static func v2ExpandTokenhubColumns(in db: Database) throws {
        try db.alter(table: "tokenhub_providers") { t in
            t.add(column: "provider_id", .text)
            t.add(column: "include_in_load_balance", .boolean).notNull().defaults(to: true)
            t.add(column: "tags", .text)
            t.add(column: "is_local", .boolean).notNull().defaults(to: false)
            t.add(column: "is_free", .boolean).notNull().defaults(to: false)
            t.add(column: "supports_responses_api", .boolean).notNull().defaults(to: false)
            t.add(column: "supports_vision", .boolean).notNull().defaults(to: false)
            t.add(column: "vision_strategy", .text)
            t.add(column: "anthropic_endpoint", .text)
            t.add(column: "vision_companion_model", .text)
            t.add(column: "request_count", .integer).notNull().defaults(to: 0)
            t.add(column: "success_count", .integer).notNull().defaults(to: 0)
            t.add(column: "last_tested_at", .datetime)
            t.add(column: "last_status", .text)
            t.add(column: "context_window_override", .integer)
        }
    }

    /// Adds a `description` column to `modelfiles` so the SQLite record can
    /// represent the `Modelfile.description` field without overloading
    /// `parameters`.
    static func v3ModelfileAddDescription(in db: Database) throws {
        try db.alter(table: "modelfiles") { t in
            t.add(column: "description", .text)
        }
    }

    /// Append-only ledger for per-key / per-model / per-time usage analytics.
    static func v5APIKeyUsageEvents(in db: Database) throws {
        try db.create(table: "api_key_usage_events") { t in
            t.autoIncrementedPrimaryKey("id")
            t.column("key_id", .text) // nil = unattributed (open mode / no auth)
            t.column("recorded_at", .datetime).notNull()
            t.column("model", .text)
            t.column("endpoint", .text).notNull()
            t.column("prompt_tokens", .integer).notNull().defaults(to: 0)
            t.column("completion_tokens", .integer).notNull().defaults(to: 0)
            t.column("total_tokens", .integer).notNull()
        }
        try db.create(
            index: "idx_usage_key_time",
            on: "api_key_usage_events",
            columns: ["key_id", "recorded_at"]
        )
        try db.create(
            index: "idx_usage_time",
            on: "api_key_usage_events",
            columns: ["recorded_at"]
        )
        try db.create(
            index: "idx_usage_model",
            on: "api_key_usage_events",
            columns: ["model"]
        )
    }

    static func v6AllowUnlistedDownloads(in db: Database) throws {
        try db.alter(table: "config") { t in
            t.add(column: "allow_unlisted_downloads", .boolean).notNull().defaults(to: false)
        }
    }
}
