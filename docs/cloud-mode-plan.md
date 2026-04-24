# NovaMLX Cloud Mode — 需求与设计文档

## 1. 概述

### 1.1 什么是 Cloud 模式

NovaMLX 当前仅在本地 Mac 上运行推理（Apple Silicon GPU）。Cloud 模式允许用户的 NovaMLX 实例**透明地将推理请求代理到远程推理引擎**，而客户端代码完全不变。

```
用户应用 (curl / OpenAI SDK / Agent 工具)
    |
    |  API 请求（不变）
    v
NovaMLX 本地服务 (127.0.0.1:6590)
    |
    |--- 模型名以 ":cloud" 结尾? --> 透传到远程数据中心
    |--- 否则 --> 本地 Apple Silicon 推理（现有逻辑）
    |
    v
远程推理引擎 (https://chat.baystoneai.com/v1)
    |
    v
SSE 流式响应原路返回给用户
```

### 1.2 核心原则

| 原则 | 说明 |
|------|------|
| **API 不变** | 客户端代码零修改，仍然连接 NovaMLX 本地端点 |
| **透明代理** | 用户感知不到推理发生在远程，SSE 流式逐 token 转发 |
| **`:cloud` 后缀路由** | 模型名以 `:cloud` 结尾 → 走云端；否则走本地 |
| **固定端点** | 远程端点硬编码为 `https://chat.baystoneai.com/v1`，无需用户配置 |
| **启动自动发现** | 每次启动从远程端点拉取可用模型列表 |
| **双协议透传** | 同时支持 OpenAI 和 Anthropic 格式的 API 透传 |

### 1.3 类似产品参考

- **Ollama Cloud**: 通过 `-cloud` 模型后缀自动路由到云端，本地 API 不变

---

## 2. 用户故事

### 2.1 场景一：数据中心的远程推理引擎

> 运维在数据中心部署了任意大模型推理引擎实例，该远程推理服务提供 OpenAI Compatible API endpoint。
> NovaMLX 启动时自动从 `https://chat.baystoneai.com/v1/models` 拉取远程可用模型列表。
> 本地 Models 页面同时显示本地模型和云端模型（云端模型名带 `:cloud` 后缀）。
> 用户选择云端模型时，本地请求自动透传到数据中心，响应流式返回。

### 2.2 场景二：混合推理

> 用户同时有本地小模型（如 Qwen3-8B）和云端大模型（如 Qwen3-235B:cloud）。
> 轻量任务走本地，重度任务选 `:cloud` 模型走云端。
> 两种模型统一在 `/v1/models` 列表中可见。

### 2.3 场景三：Agent 工具透明使用

> OpenClaw / OpenCode 等 Agent 工具连接本地 NovaMLX。
> Agent 选择 `:cloud` 模型时，请求自动走云端，Agent 无需任何配置变更。

---

## 3. 架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────┐
│              NovaMLX 本地服务                │
│                                             │
│  ┌─────────┐    ┌──────────────────────┐   │
│  │ API 层   │───>│    InferenceService   │   │
│  │(不变)    │    │  (路由决策中心)        │   │
│  └─────────┘    └──────┬───────────────┘   │
│                        │                    │
│              ┌─────────┼─────────┐          │
│              v         v         v          │
│        ┌─────────┐ ┌───────┐ ┌──────────┐  │
│        │ 本地引擎 │ │Worker │ │CloudBackend│ │
│        │(MLX)    │ │子进程  │ │ (新增)    │  │
│        └─────────┘ └───────┘ └──────┬───┘  │
│                                       │      │
│  ┌────────────────────────────┐      │      │
│  │ 启动时: GET /v1/models     │──────┘      │
│  │ → 拉取云端模型列表         │             │
│  │ → 合并到本地模型展示       │             │
│  └────────────────────────────┘             │
└─────────────────────────────────────────────┘
                         │
           HTTPS / SSE   │
                           v
         ┌──────────────────────────────────┐
         │ 固定端点 (硬编码)                  │
         │ https://chat.baystoneai.com/v1    │
         │                                    │
         │ 提供任意 OpenAI-compatible 推理     │
         │ - vLLM / TGI / Ollama / NovaMLX   │
         └──────────────────────────────────┘
```

### 3.2 路由决策

`InferenceService` 已有的路由逻辑：

```
当前：
  generate() -> workerMode ? worker.sendGenerate() : localEngine.generate()
  stream()   -> workerMode ? worker.sendStream()   : localEngine.stream()
```

新增 Cloud 分支：

```
目标：
  generate() -> model.hasSuffix(":cloud") ? cloudBackend.proxy(request)
             : workerMode ? worker.sendGenerate()
             : localEngine.generate()

  stream()   -> model.hasSuffix(":cloud") ? cloudBackend.proxyStream(request)
             : workerMode ? worker.sendStream()
             : localEngine.stream()
```

**判断逻辑**：模型名以 `:cloud` 结尾 → 剥离后缀，将原始模型名发送到远程端点。

### 3.3 Cloud 后端接口

```swift
// CloudBackend.swift (新增)

actor CloudBackend {
    // 固定端点，硬编码
    static let endpoint = URL(string: "https://chat.baystoneai.com/v1")!

    // 启动时拉取的云端模型缓存
    var cachedModels: [CloudModelInfo] = []

    // 启动时调用：拉取远程模型列表
    func fetchModels() async throws -> [CloudModelInfo]

    // 非流式推理 (OpenAI format)
    func proxy(_ request: InferenceRequest) async throws -> InferenceResult

    // 流式推理 (OpenAI format) — 返回与本地引擎完全相同的 AsyncThrowingStream<Token, Error>
    func proxyStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error>

    // 非流式推理 (Anthropic format)
    func proxyAnthropic(_ request: InferenceRequest) async throws -> InferenceResult

    // 流式推理 (Anthropic format)
    func proxyAnthropicStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error>

    // 健康检查
    func healthCheck() async -> Bool
}
```

### 3.4 数据流 — 流式推理

```
Client                          NovaMLX                      https://chat.baystoneai.com
  |                               |                              |
  | POST /v1/chat/completions     |                              |
  | model: "Qwen3-235B:cloud"     |                              |
  | (stream: true)                |                              |
  |------------------------------>|                              |
  |                               | 检测到 ":cloud" 后缀          |
  |                               | 剥离后缀 → "Qwen3-235B"      |
  |                               |                              |
  |                               | POST /v1/chat/completions    |
  |                               | model: "Qwen3-235B"          |
  |                               | (stream: true)               |
  |                               |----------------------------->|
  |                               |                              |
  |                               |      SSE: token chunk 1      |
  |                               |<-----------------------------|
  |     SSE: token chunk 1        |                              |
  |<------------------------------|                              |
  |                               |      ...                     |
  |                               |      SSE: [DONE]             |
  |                               |<-----------------------------|
  |     SSE: [DONE]               |                              |
  |<------------------------------|                              |
```

### 3.5 模型发现流程

```
NovaMLX 启动
    |
    v
GET https://chat.baystoneai.com/v1/models
    |
    v
解析响应:
  {
    "data": [
      { "id": "Qwen3-235B-4bit", ... },
      { "id": "DeepSeek-V3", ... }
    ]
  }
    |
    v
为每个远程模型添加 ":cloud" 后缀:
  - "Qwen3-235B-4bit:cloud"
  - "DeepSeek-V3:cloud"
    |
    v
合并到 appState.loadedModels / Models 页面展示
    |
    v
/v1/models API 也合并返回本地 + 云端模型
```

---

## 4. 配置设计

### 4.1 无需用户配置

远程端点 **硬编码** 在 `CloudBackend` 中：

```swift
extension CloudBackend {
    static let cloudBaseURL = URL(string: "https://chat.baystoneai.com/v1")!
}
```

用户不需要在 config.json 中配置 endpoint、模型映射等信息。模型列表从远程自动获取。

### 4.2 可选配置项（未来扩展）

`~/.nova/config.json` 中可选的 cloud 字段：

```json
{
  "cloud": {
    "apiKey": "sk-xxx",
    "enabled": true
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `enabled` | Bool | 否 | 默认 true。设为 false 可禁用 Cloud 模型发现和路由 |
| `apiKey` | String | 否 | 远程引擎认证 Key。如果远程不需要认证则留空 |

---

## 5. API 兼容性

### 5.1 NovaMLX → 远程引擎（透传）

NovaMLX 作为客户端向远程引擎发起请求。

#### OpenAI 格式透传

```
POST https://chat.baystoneai.com/v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen3-235B",          // 已剥离 ":cloud" 后缀
  "messages": [...],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": true
}
```

#### Anthropic 格式透传

```
POST https://chat.baystoneai.com/v1/messages
Content-Type: application/json
x-api-key: ...

{
  "model": "Qwen3-235B",          // 已剥离 ":cloud" 后缀
  "messages": [...],
  "max_tokens": 4096,
  "stream": true
}
```

**不检查远程能力**：远程引擎如果不支持 Anthropic 接口，请求会自然失败，错误信息透传给用户。用户能直接看到。

### 5.2 用户 → NovaMLX（不变）

用户/Agent 仍然使用 NovaMLX 的全部 API：
- `/v1/chat/completions` — OpenAI 格式
- `/v1/completions` — OpenAI completions 格式
- `/v1/messages` — Anthropic 格式

模型名带 `:cloud` 时自动走远程，否则走本地。

### 5.3 模型列表合并

`GET /v1/models` 返回本地 + 云端合并列表：

```json
{
  "data": [
    { "id": "Qwen3-8B-4bit", "owned_by": "local" },
    { "id": "Llama-3.1-70B-4bit", "owned_by": "local" },
    { "id": "Qwen3-235B:cloud", "owned_by": "baystoneai" },
    { "id": "DeepSeek-V3:cloud", "owned_by": "baystoneai" }
  ]
}
```

### 5.4 不适用的本地特性

| 场景 | Cloud 模式处理方式 |
|------|-------------------|
| Prefix Cache | 跳过，远程引擎自行管理 |
| TurboQuant | 不适用，远程引擎自行管理 |
| Worker 子进程 | Cloud 请求直接 HTTP 透传，不经过 Worker |
| Grammar/JSON Schema | 转发给远程引擎（如果支持） |
| Function Calling/Tools | 透传 |

---

## 6. 实现计划

### Phase 1: 核心透传（最小可用）

**目标**: 启动时拉取远程模型列表，`/v1/chat/completions` 流式/非流式推理透传到远程。

**新增文件**:

| 文件 | 说明 |
|------|------|
| `Sources/NovaMLXInference/CloudBackend.swift` | HTTP 客户端：模型发现、请求转发、SSE 解析 |

**修改文件**:

| 文件 | 修改内容 |
|------|----------|
| `Sources/NovaMLXInference/InferenceService.swift` | `generate()` / `stream()` 新增 `:cloud` 路由分支 |
| `Sources/NovaMLXAPI/APIServer.swift` | `isModelLoaded()` 兼容 cloud 模型；`/v1/models` 合并远程模型；Anthropic 端点也支持 cloud 路由 |
| `Sources/NovaMLXMenuBar/MenuBarAppState.swift` | 新增 `cloudModels` 状态，启动时拉取并缓存 |
| `Sources/NovaMLXMenuBar/ModelsPageView.swift` | Models 页面展示云端模型（带 `:cloud` 标识和云端 badge） |
| `Sources/NovaMLXMenuBar/ChatPageView.swift` | Chat 页面的模型 Picker 包含云端模型 |
| `Sources/NovaMLXApp/main.swift` | 启动时触发云端模型发现 |

**工作量**: 约 2-3 天

### Phase 2: GUI 完善 + 双协议

**目标**: GUI 完整展示云端模型状态，Anthropic API 也支持透传。

**修改文件**:

| 文件 | 修改内容 |
|------|----------|
| `Sources/NovaMLXMenuBar/StatusPageView.swift` | Status 页面显示 Cloud 连接状态、延迟 |
| `Sources/NovaMLXCore/LocalizationStrings.swift` | Cloud 相关 UI 字符串 (9 语言) |
| `Sources/NovaMLXInference/CloudBackend.swift` | 新增 Anthropic 格式透传 |

**工作量**: 约 1-2 天

### Phase 3: 高级功能

| 功能 | 说明 |
|------|------|
| 健康检查 & 状态指示 | 定期 ping 远程端点，Status 页面显示连接状态 |
| 延迟监控 | 显示 Cloud RTT |
| 用量统计 | 记录 Cloud 推理 token 消耗 |
| 定期刷新模型列表 | 每隔 N 分钟重新拉取远程模型 |

---

## 7. CloudBackend 核心实现思路

### 7.1 模型发现

```swift
static let cloudBaseURL = URL(string: "https://chat.baystoneai.com/v1")!

func fetchModels() async throws -> [CloudModelInfo] {
    let url = Self.cloudBaseURL.appendingPathComponent("models")
    let (data, _) = try await URLSession.shared.data(from: url)
    let response = try JSONDecoder().decode(OpenAIModelsResponse.self, from: data)
    return response.data.map { CloudModelInfo(id: $0.id) }
}

// CloudModelInfo → 本地显示名
// "Qwen3-235B" → "Qwen3-235B:cloud"
```

### 7.2 非流式请求 (OpenAI 格式)

```swift
func proxy(_ request: InferenceRequest) async throws -> InferenceResult {
    // 剥离 ":cloud" 后缀
    var req = request
    req.model = req.model.replacingOccurrences(of: ":cloud", with: "")

    var urlRequest = URLRequest(url: Self.cloudBaseURL.appendingPathComponent("chat/completions"))
    urlRequest.httpMethod = "POST"
    urlRequest.httpBody = try JSONEncoder().encode(mapToOpenAIFormat(req))

    let (data, _) = try await URLSession.shared.data(for: urlRequest)
    return try decodeOpenAIResponse(data)
}
```

### 7.3 流式请求 (OpenAI 格式)

```swift
func proxyStream(_ request: InferenceRequest) -> AsyncThrowingStream<Token, Error> {
    AsyncThrowingStream { continuation in
        Task {
            var req = request
            req.model = req.model.replacingOccurrences(of: ":cloud", with: "")

            var urlRequest = URLRequest(url: Self.cloudBaseURL.appendingPathComponent("chat/completions"))
            urlRequest.httpMethod = "POST"
            urlRequest.httpBody = try JSONEncoder().encode(mapToOpenAIStreamFormat(req))

            let (bytes, _) = try await URLSession.shared.bytes(for: urlRequest)

            for try await line in bytes.lines {
                guard line.hasPrefix("data: ") else { continue }
                let json = String(line.dropFirst(6))
                if json == "[DONE]" { break }
                if let token = try parseOpenAISSEChunk(json) {
                    continuation.yield(token)
                }
            }
            continuation.finish()
        }
    }
}
```

### 7.4 Anthropic 格式透传

与 OpenAI 格式类似，区别在于：
- 请求 URL: `/v1/messages`
- 请求体格式: Anthropic Messages API
- SSE 事件格式: `message_start`, `content_block_delta`, `message_delta`, `message_stop`
- 解析 Anthropic SSE 为 `Token` 对象

不检查远程是否支持——请求发出后如果远程返回错误，直接将错误透传给客户端。

---

## 8. GUI 展示设计

### 8.1 Models 页面

本地模型和云端模型在同一列表中显示：

```
┌─────────────────────────────────────────────────┐
│ Active Models                                    │
├─────────────────────────────────────────────────┤
│ ● Qwen3-8B-4bit               [Unload]          │  ← 本地模型（绿色圆点）
│ ☁ Qwen3-235B:cloud            [云端]            │  ← 云端模型（云朵图标）
│ ☁ DeepSeek-V3:cloud           [云端]            │  ← 云端模型（云朵图标）
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Inactive Models                                  │
├─────────────────────────────────────────────────┤
│ ○ Llama-3.1-70B-4bit          [Load]            │  ← 本地未加载模型
└─────────────────────────────────────────────────┘
```

云端模型特点：
- 显示 ☁ 图标（而非本地模型的 ● 绿点）
- 不需要 Load/Unload 操作
- 显示 "Cloud" badge
- 点击可以查看远程模型信息（从 `/v1/models` 响应中获取）

### 8.2 Chat 页面

模型 Picker 下拉框中，云端模型带 `☁` 前缀和 `:cloud` 后缀：

```
┌─────────────────────────────┐
│ Qwen3-8B-4bit               │  ← 本地
│ ☁ Qwen3-235B:cloud          │  ← 云端
│ ☁ DeepSeek-V3:cloud         │  ← 云端
└─────────────────────────────┘
```

### 8.3 Status 页面

新增 Cloud 连接状态：

```
Cloud Endpoint: https://chat.baystoneai.com/v1
Status: ● Connected
Models: 2 cloud models available
Latency: 45ms
```

---

## 9. 安全考虑

| 风险 | 对策 |
|------|------|
| 请求内容泄露 | 远程端点强制 HTTPS；传输全程加密 |
| API Key 管理 | Key 存储在本地 config.json（可选），不通过本地 API 暴露 |
| 端点硬编码安全 | 端点是用户自有的数据中心域名，非第三方服务 |

---

## 10. 测试策略

| 测试 | 方法 |
|------|------|
| 模型发现 | Mock `/v1/models` 响应，验证解析和 `:cloud` 后缀拼接 |
| OpenAI 透传 | 使用真实远程端点或 mock server 测试 |
| Anthropic 透传 | 测试 Anthropic 格式请求透传，验证失败场景的错误透传 |
| 流式正确性 | 比较 mock server 返回的 SSE 与 NovaMLX 转发的 SSE 一致性 |
| 超时处理 | 模拟远程响应超时，验证超时错误正确返回 |
| 网络不可用 | 断网时验证 cloud 模型列表获取失败不影响本地推理 |
