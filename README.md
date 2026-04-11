# NovaMLX —— 纯 Swift，极致 MLX

> **The fastest, most modular pure-Swift LLM/VLM inference server for Apple Silicon. Zero Python dependencies.**

[![macOS 15+](https://img.shields.io/badge/macOS-15%2B-blue)](https://www.apple.com/macos)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Native-green)](https://developer.apple.com/metal/)
[![Swift 6.0](https://img.shields.io/badge/Swift-6.0-orange)](https://swift.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why NovaMLX?

| Feature | NovaMLX | oMLX | swift-mlx-server |
|---------|---------|------|-----------------|
| Pure Swift (no Python) | ✅ | ❌ | ✅ |
| Paged SSD KV Cache | ✅ | Basic | ❌ |
| Continuous Batching | ✅ | ❌ | ❌ |
| OpenAI API Compatible | ✅ | ✅ | ✅ |
| Anthropic API Compatible | ✅ | ❌ | ❌ |
| Native Menu Bar App | ✅ | ❌ | ❌ |
| VLM Support | ✅ | ✅ | ❌ |
| Auto Model Download | ✅ | ❌ | ✅ |
| LRU Multi-Model Mgmt | ✅ | ✅ | ❌ |
| Streaming SSE | ✅ | ✅ | ✅ |
| Token-by-token streaming | ✅ | ✅ | ✅ |

---

## Architecture

NovaMLX uses a **9-module architecture** designed for maximum modularity and testability:

```
┌─────────────────────────────────────────────────┐
│                   NovaMLXApp                     │
│              (Entry Point + Menu Bar)            │
├──────────┬──────────┬──────────┬────────────────┤
│ MenuBar  │   API    │Inference │  ModelManager   │
│ (SwiftUI)│(Hbird)   │(Batcher) │ (Download/Mgmt) │
├──────────┴────┬─────┴────┬─────┴────────────────┤
│               │ Engine   │                       │
│               │  (MLX)   │                       │
├───────────────┼──────────┼──────────────────────┤
│               │ KVCache  │                       │
│               │(Paged+SSD)│                      │
├───────────────┴──────────┴──────────────────────┤
│                   Core + Utils                   │
└─────────────────────────────────────────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| **NovaMLXCore** | Foundation types, protocols, configuration |
| **NovaMLXUtils** | Logging, system monitoring, extensions |
| **NovaMLXKVCache** | Paged in-memory + SSD KV cache (LRU eviction) |
| **NovaMLXEngine** | MLX model loading, tokenization, evaluation |
| **NovaMLXInference** | LLM/VLM pipelines, continuous batching |
| **NovaMLXModelManager** | Model registry, download, versioning |
| **NovaMLXAPI** | OpenAI + Anthropic compatible HTTP server |
| **NovaMLXMenuBar** | Native SwiftUI menu bar UI |
| **NovaMLXApp** | Application entry point |

---

## Quick Start

### Build

```bash
git clone https://github.com/your-org/novamlx.git
cd novamlx
swift build -c release
```

### Run

```bash
# Start the server with menu bar app
.build/release/NovaMLX

# Or use CLI mode
swift run NovaMLXApp
```

### API Usage

#### OpenAI Compatible

```bash
# List models
curl http://127.0.0.1:8080/v1/models

# Chat completion
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

#### Anthropic Compatible

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Configuration

Configuration is stored at `~/Library/Application Support/NovaMLX/`:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8080,
    "maxConcurrentRequests": 16
  },
  "defaultModel": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
}
```

---

## Testing

```bash
swift test
```

---

## Performance

NovaMLX leverages Apple's MLX framework with Metal acceleration:

- **Paged KV Cache**: LRU eviction keeps hot models in memory
- **SSD KV Cache**: Offload cold entries to SSD for 10x capacity
- **Continuous Batching**: Process multiple requests concurrently
- **Zero Python overhead**: Pure Swift with direct Metal access
- **Unified Memory**: Leverages Apple Silicon's shared memory architecture

---

## Requirements

- macOS 15.0+ (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Xcode 16.0+
- Swift 6.0+

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on top of:
- [MLX Swift](https://github.com/ml-explore/mlx-swift) by Apple
- [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples)
- [Hummingbird](https://github.com/hummingbird-project/hummingbird)
- Inspired by [oMLX](https://github.com/jundot/omlx), [SwiftLM](https://github.com/SharpAI/SwiftLM), and [swift-mlx-server](https://github.com/mzbac/swift-mlx-server)
