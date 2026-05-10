"""Admin API example — model management, sessions, benchmarks."""

from novamlx import AdminClient

admin = AdminClient()  # auto-discovers api key from config

# List all models
models = admin.models.list()
for m in models.models:
    print(f"  {m.model_id}: loaded={m.loaded} downloaded={m.downloaded}")

# Load a model
admin.models.load("Qwen3-8B-MLX-4bit")

# Get model settings
settings = admin.models.get_settings("Qwen3-8B-MLX-4bit")
print(f"Max tokens: {settings.max_tokens}")
print(f"Thinking budget: {settings.thinking_budget}")

# Update settings
admin.models.update_settings("Qwen3-8B-MLX-4bit", {
    "max_tokens": 4096,
    "thinking_budget": 16384,
    "is_pinned": True,
})

# Device info
device = admin.device_info()
print(f"Chip: {device.chip}, Memory: {device.memory_gb}GB")

# Cache stats
cache = admin.cache.stats("Qwen3-8B-MLX-4bit")
print(f"Cache hits: {cache.hits}, misses: {cache.misses}")

# Sessions
sessions = admin.sessions.list()
for s in sessions.sessions:
    print(f"  Session {s.id}: model={s.model_id} tokens={s.token_count}")

# Run benchmark
admin.benchmark.start("Qwen3-8B-MLX-4bit", prompt_lengths=[512, 1024, 4096])
