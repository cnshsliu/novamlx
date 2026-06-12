import GRDB

enum DataDBSchema {
    enum v1 {
        static func createAll(in db: Database) throws {
            try db.create(table: "model_registry") { t in
                t.column("model_id", .text).primaryKey()
                t.column("family", .text)
                t.column("model_type", .text) // llm, vlm, embed, audio, image
                t.column("source", .text) // huggingface, local, tokenhub
                t.column("local_path", .text)
                t.column("remote_url", .text)
                t.column("size_bytes", .integer)
                t.column("downloaded_at", .datetime)
                t.column("version", .text)
                t.column("architecture", .text)
            }

            try db.create(table: "loaded_models") { t in
                t.column("model_id", .text).primaryKey()
                t.column("loaded_at", .datetime).notNull()
            }

            try db.create(table: "metrics") { t in
                t.column("id", .integer).primaryKey()
                t.column("total_requests", .integer).notNull().defaults(to: 0)
                t.column("total_tokens", .integer).notNull().defaults(to: 0)
                t.column("total_inference_time_ms", .integer).notNull().defaults(to: 0)
                t.column("cache_hits", .integer).notNull().defaults(to: 0)
                t.column("cache_misses", .integer).notNull().defaults(to: 0)
                t.column("evictions", .integer).notNull().defaults(to: 0)
                t.column("per_model_stats", .text).defaults(to: "{}") // JSON
                t.column("per_model_cache", .text).defaults(to: "{}") // JSON
                t.column("updated_at", .datetime)
            }

            try db.create(table: "worker_deployments") { t in
                t.column("hostname", .text).primaryKey()
                t.column("phase", .text).notNull()
                t.column("username", .text)
                t.column("version", .text)
                t.column("started_at", .datetime)
                t.column("updated_at", .datetime)
                t.column("extra_json", .text).defaults(to: "{}")
            }

            try db.create(table: "chats") { t in
                t.column("id", .text).primaryKey()
                t.column("title", .text)
                t.column("model", .text).notNull()
                t.column("system_prompt", .text)
                t.column("created_at", .datetime).notNull()
                t.column("updated_at", .datetime).notNull()
            }

            try db.create(table: "chat_messages") { t in
                t.column("id", .text).primaryKey()
                t.column("chat_id", .text).notNull()
                    .references("chats", onDelete: .cascade)
                t.column("role", .text).notNull()
                t.column("content", .text)
                t.column("thinking_content", .text)
                t.column("created_at", .datetime).notNull()
                t.column("sort_order", .integer).notNull()
            }
            try db.create(index: "idx_chat_messages_chat_id", on: "chat_messages", columns: ["chat_id"])

            // FTS5 virtual table for full-text search
            try db.execute(sql: """
                CREATE VIRTUAL TABLE chat_messages_fts USING fts5(
                    content,
                    content=chat_messages,
                    content_rowid=rowid
                )
                """)
        }
    }
}
