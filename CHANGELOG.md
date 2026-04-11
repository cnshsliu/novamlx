# Changelog

All notable changes to NovaMLX will be documented in this file.

## [1.0.0] - 2025-04-09

### Added
- Pure Swift SPM project with 9 modular targets
- Paged in-memory KV Cache with LRU eviction
- SSD-backed KV Cache for persistent caching
- Continuous batching for concurrent request processing
- MLX engine with model loading and tokenization
- LLM and VLM inference pipelines
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Anthropic-compatible API (`/v1/messages`)
- SSE streaming support for both API formats
- Model manager with download, registry, and versioning
- Auto-registration of popular MLX community models
- Native macOS menu bar app with SwiftUI
- System monitoring (CPU, memory, GPU)
- Health check and stats endpoints
- Comprehensive test suite (Core, KVCache, Engine, Inference, ModelManager, API)
- MIT License

### Performance
- Zero Python dependencies
- Direct Metal acceleration via MLX
- Paged KV Cache with configurable memory limits
- SSD overflow for cold cache entries
- Continuous batching for maximum throughput
