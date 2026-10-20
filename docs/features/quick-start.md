# Quick Start

**Pitch:** Install, set an API key or run open, then call `/v1/chat/completions` on port 6590.

功能目录里的最短路径：安装、可选地设置 API key、列出模型、发聊天、Anthropic、Tokenhub 或负载均衡请求。

## Catalog

```bash
# 通过 Homebrew 安装
brew install --head novamlx

# 或从源码构建
git clone https://github.com/novamlx/novamlx && cd novamlx
./build.sh
open dist/NovaMLX.app

# 配置 API Key（或不配置任何 key 走开放模式）
export NOVA_API_KEY=$(openssl rand -hex 24)

# 列出模型
curl http://localhost:6590/v1/models \
  -H "Authorization: Bearer $NOVA_API_KEY"

# 聊天
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role":"user","content":"Hello"}]
  }'

# Anthropic 格式
curl http://localhost:6590/v1/messages \
  -H "x-api-key: $NOVA_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"Hello"}]
  }'

# 代理云端模型（先在 Tokenhub UI 里配置 provider）
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tknet:deepseek-v4-flash",
    "messages": [{"role":"user","content":"Hello"}]
  }'

# 负载均衡器（先在 UI 里创建，再用 lb:<slug>）
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lb:my-pool",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

根 README 的安装与端口不同（tap `cnshsliu/novamlx`，示例端口 `8080`）。克隆地址在功能目录中写作 `https://github.com/novamlx/novamlx`。以你手头的远程与 `server.port` 为准。见 [配置](configuration.md)。

正文来自 `features.zh-CN.md`「快速开始」；标题与 pitch 为英文。
