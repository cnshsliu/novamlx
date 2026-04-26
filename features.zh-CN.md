# NovaMLX — 功能参考

> **面向 Apple Silicon 的生产级纯 Swift LLM/VLM 推理服务器。**
> 兼容 OpenAI 与 Anthropic API。原生 macOS 菜单栏应用。基于 MLX 构建。

---

## 目录

- [推理引擎](#推理引擎)
- [Worker 子进程隔离](#worker-子进程隔离)
- [多模态支持](#多模态支持)
- [API 兼容性](#api-兼容性)
- [结构化输出](#结构化输出)
- [工具调用](#工具调用)
- [控制 Token 与思维链过滤](#控制-token-与思维链过滤)
- [KV 缓存与前缀共享](#kv-缓存与前缀共享)
- [TurboQuant KV 压缩](#turboquant-kv-压缩)
- [连续批处理](#连续批处理)
- [推测解码](#推测解码)
- [会话管理](#会话管理)
- [向量嵌入与重排序](#向量嵌入与重排序)

- [MCP — 模型上下文协议](#mcp--模型上下文协议)
- [Agent 集成](#agent-集成)
- [模型管理](#模型管理)
- [内存管理](#内存管理)
- [逐模型配置](#逐模型配置)
- [HuggingFace 集成](#huggingface-集成)
- [云端模型代理](#云端模型代理)
- [可观测性与基准测试](#可观测性与基准测试)
- [Web 界面](#web-界面)
- [macOS 菜单栏应用](#macos-菜单栏应用)
- [国际化](#国际化)
- [安全与中间件](#安全与中间件)
- [Homebrew 分发](#homebrew-分发)
- [配置参考](#配置参考)

---

## 推理引擎

NovaMLX 通过 MLX 在 Apple Silicon GPU 上直接运行 LLM 和 VLM 推理，零外部依赖（无需 Python 或远程服务）。

| 特性 | 详情 |
|------|------|
| **计算后端** | MLX（Apple Silicon GPU）、惰性求值、统一内存架构 |
| **模型格式** | SafeTensors（4-bit、8-bit、FP16 量化） |
| **50+ 模型架构** | Llama 3/3.1、Mistral/Mixtral、Qwen 2/2.5/3、Gemma 2/3、Phi 3.5/4、StarCoder2 等 |
| **采样策略** | Temperature、Top-P、Top-K、Min-P、频率/存在/重复惩罚、随机种子 |
| **流式输出** | 所有生成端点均支持 SSE 逐 Token 流式输出 |
| **最大生成长度** | 按请求或按模型可配置 |
| **融合批量解码** | 共享 KV 缓存的并发多序列解码 |
| **推测解码** | 基于 N-gram 模式和草稿模型的 Token 预测加速生成 |

---

## Worker 子进程隔离

推理 Worker 作为独立子进程运行，通过 stdin/stdout 的 JSON 消息与主 API 服务器通信。

| 特性 | 详情 |
|------|------|
| **崩溃隔离** | Worker 崩溃不影响 API 服务器——Supervisor 自动重启 |
| **内存隔离** | Worker 拥有独立的进程内存预算，与主应用分开 |
| **通信协议** | 基于 stdin/stdout 的双向 JSON 消息协议 |
| **健康上报** | 每 5 秒向父进程推送内存统计 |
| **代码签名** | Ad-hoc 签名满足 macOS 进程完整性要求 |
| **守护进程** | WorkerSupervisor 监控 Worker 健康，异常退出时自动重启，处理背压 |

---

## 多模态支持

完整的视觉语言模型（VLM）推理管线，支持图像理解。

| 特性 | 详情 |
|------|------|
| **图像输入** | Base64 Data URI、HTTP URL、本地文件路径 |
| **VLM 架构** | Qwen2-VL、Qwen2.5-VL、Qwen3-VL、LLaVA、Gemma3、Phi-3-Vision、Pixtral、Molmo、Idefics3、InternVL、PaliGemma、DeepSeek-VL2 等 |
| **视觉特征缓存** | 内存 LRU（20 条目）+ 可选 SSD 持久化，SHA-256 哈希，按模型隔离 |
| **API 适配** | 标准 OpenAI `image_url` 消息内容格式 |

### 图像识别示例

```bash
# 通过 base64 数据 URI 描述图像
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "这张图片里有什么？"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAA...=="}}
      ]
    }],
    "max_tokens": 100
  }'

# 通过 HTTP URL 描述图像
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片。"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }]
  }'
```

> 以上模型名称仅为示例，更多模型请通过软件内的 HuggingFace 浏览器或管理 API `/admin/models/download` 浏览和下载。

---

## API 兼容性

### OpenAI 兼容端点（端口 8080）

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/v1/models` | 列出可用模型 |
| `POST` | `/v1/chat/completions` | 对话补全（流式 + 非流式） |
| `POST` | `/v1/completions` | 文本补全（流式 + 非流式） |
| `POST` | `/v1/embeddings` | 文本向量嵌入 |
| `POST` | `/v1/rerank` | 文档重排序 |
| `POST` | `/v1/responses` | OpenAI Responses API |
| `GET` | `/v1/responses/{id}` | 获取已存储的响应 |
| `DELETE` | `/v1/responses/{id}` | 删除已存储的响应 |

### Anthropic 兼容端点

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/v1/messages` | Anthropic Messages API（流式 + 非流式） |
| `POST` | `/v1/messages/count_tokens` | Token 计数 |

### 工具端点

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（状态、GPU 内存、已加载模型、MCP） |
| `GET` | `/v1/stats` | 会话与历史推理指标 |
| `POST` | `/v1/mcp/execute` | 执行 MCP 工具调用 |

---

## 结构化输出

### JSON 模式（`response_format: json_object`）

基于字符级有限状态机，将输出约束为合法 JSON。内部跟踪 12 种状态（对象、数组、字符串、数字、字面量），确保括号匹配、转义正确。

### Schema 引导模式（`response_format: json_schema`）

完整的 JSON Schema 解析为类型树，支持：
- `object`（含必填字段追踪）
- `array`、`string`、`integer`、`number`、`boolean`、`null`
- `enum`（字符串值约束）
- `anyOf`、`oneOf`、`allOf` 组合
- Schema 状态与 JSON 状态同步追踪，实现字段级约束
- 基于 Token 级 logits 掩码的字符过滤

---

## 工具调用

自动检测并解析 7 种工具调用输出格式：

| 格式 | 模式 |
|------|------|
| **XML** | `<tool>{"name": "...", "arguments": {...}}</tool>` 或 `<function=name>...` |
| **括号** | `[TOOL_CALLS] [{"name": ...}]` |
| **标记** | `<\|tool_call\|>...<\|/tool_call\|>` |
| **命名空间 XML** | `<ns:tool_call><invoke name="...">...` |
| **GLM 格式** | `☐函数名☐参数名■参数值■` 分隔符对 |
| **Gemma 格式** | `call:functionName {key: value}` |
| **思维链回退** | 从 `<think...>` 块中提取工具调用 |

流式过滤器在输出流中实时屏蔽工具调用标记，确保终端用户看到纯净内容。

---

## 控制 Token 与思维链过滤

实时控制 Token 检测与过滤，保持生成输出干净并防止无限生成循环。

### TurnStopProcessor

检测生成文本中的轮次分隔符模式并强制发出 EOS Token。对使用语义标记但很少发出配置 EOS Token 的模型（如 Qwen3.6）至关重要。

| 特性 | 详情 |
|------|------|
| **模式匹配** | 可配置的停止模式（如 `<|turn|>`、`<|im_end|>`） |
| **状态机** | Active → StopDetected → Done 生命周期 |
| **EOS 强制** | 检测到模式时将非 EOS logits 全部设为 `-inf` |
| **累积追踪** | 追踪完整解码序列，而非单个 Token |

### 思维链标签架构

语义思维标签透传给用户，协议级控制 Token 被过滤：

| 行为 | 示例 |
|------|------|
| **透传** | `<think>推理过程</think>` — 用户可见思维内容 |
| **过滤** | `<|turn|>`、`<|im_end|>` — 流式输出前剥离 |
| **隐式开放** | `ThinkingParser` 处理无开放标签的 `</think>` 闭合 |
| **部分缓冲** | SSE 流中缓冲不完整的控制 Token 防止片段泄露 |

### 控制 Token 过滤管线

- 通用流式过滤器清理所有 Agent 输出格式中的控制 Token
- 多轮 Agent 对话保护——控制 Token 不会在轮次间泄露
- 支持正则 Token 匹配选项的自定义模式

---

## KV 缓存与前缀共享

借鉴 vLLM 的分块 KV 缓存架构，实现跨会话前缀复用。

| 特性 | 详情 |
|------|------|
| **块大小** | 64 个 Token（可配置） |
| **哈希算法** | SHA-1 链式哈希（父哈希 + Token + 模型名） |
| **块池** | 双向链表空闲队列，引用计数，写时复制分叉 |
| **SSD 持久化** | SafeTensors 格式，16 桶分片目录，GCD 异步写入队列 |
| **SSD 容量** | 默认 100 GB，LRU 驱逐 |
| **缓存类型** | KVCacheSimple、RotatingKVCache、QuantizedKVCache、ChunkedKVCache、MambaCache、ArraysCache、CacheList |
| **统计指标** | 命中/未命中/节省 Token/驱逐次数/共享块数/SSD 块数与大小 |

---

## TurboQuant KV 压缩

按模型可配置的 KV 缓存量化和内存压缩。

| 位宽 | 压缩比 | 适用场景 |
|------|--------|----------|
| 2-bit | **8.0×** | 极端内存压力 |
| 3-bit | **5.33×** | 高内存压力 |
| 4-bit | **4.0×** | 均衡模式（推荐默认值） |
| 6-bit | **2.67×** | 质量敏感场景 |
| 8-bit | **2.0×** | 最小质量损失 |

- **分组大小**：32、64（默认）、128
- **自动推荐**：基于模型体积/可用内存比和上下文长度自动选择最佳位宽
- **管理端点**：`GET /admin/api/turboquant` — 查看活跃配置
- **底层实现**：基于 mlx-swift-lm 的 `QuantizedKVCache`，使用优化 Metal 内核

---

## 连续批处理

支持优先级感知的请求批处理，实现高吞吐并发推理。

| 特性 | 详情 |
|------|------|
| **最大批量** | 8（可配置） |
| **优先级** | 低、普通、高 |
| **抢占调度** | 高优先级请求可抢占低优先级 |
| **指标** | 活跃请求数、队列深度、已入队/已完成/已抢占总数、峰值并发、平均等待时间 |
| **请求取消** | 支持单请求级别中断 |

---

## 推测解码

两种互补方式通过并行预测和验证多个 Token 来加速生成。

### N-gram 推测解码

内置于 `FusedBatchScheduler` 的基于模式的草稿 Token 生成：

| 特性 | 详情 |
|------|------|
| **方法** | 从最近上下文窗口进行 N-gram 模式匹配 |
| **草稿长度** | 可配置的每步候选 Token 数量 |
| **验证** | 目标模型并行验证所有草稿 Token |
| **开销** | 无需辅助模型——纯算法实现 |

### 草稿模型推测解码

通过 `SpeculativeTokenIterator` 使用专用小模型：

| 特性 | 详情 |
|------|------|
| **草稿模型** | 更小更快的模型生成候选 Token |
| **目标模型** | 完整模型在单次前向传播中验证所有候选 |
| **吞吐提升** | 对内存带宽瓶颈场景有显著加速 |
| **适用场景** | 单 Token 生成受内存限制的大模型 |

---

## 会话管理

持久化对话会话，跨轮次复用 KV 缓存。

| 特性 | 详情 |
|------|------|
| **最大会话数** | 64 个并发会话 |
| **会话 TTL** | 1800 秒自动回收 |
| **持久化** | KV 缓存保存/恢复为 SafeTensors 文件 |
| **会话分叉** | 深拷贝 KV 缓存到独立新会话 |
| **管理 API** | 列出、删除、保存、分叉会话 |

---

## 向量嵌入与重排序

### 嵌入服务
- **模型**：BERT、XLM-RoBERTa、ModernBERT、SigLIP、ColQwen2.5、NomicBERT、Jina v3
- **批处理**：自动填充与堆叠，注意力掩码
- **输出**：归一化池化向量
- **端点**：`POST /v1/embeddings`（OpenAI 兼容）

### 重排序服务
- **方法**：交叉编码器对打分（query `</s></s>` document）
- **归一化**：Min-Max 分数归一化
- **Top-N 过滤**：仅返回前 N 个结果
- **端点**：`POST /v1/rerank`（兼容 Cohere/Jina）

---

## MCP — 模型上下文协议

多服务器 MCP 客户端，支持工具发现与执行。

| 特性 | 详情 |
|------|------|
| **传输协议** | `stdio`（子进程）、`sse`（HTTP SSE）、`streamable-http`（HTTP POST） |
| **协议版本** | `2024-11-05` |
| **工具发现** | 连接时自动 `tools/list` |
| **命名空间** | 工具以 `{服务器名}__{工具名}` 命名，避免冲突 |
| **超时控制** | 按服务器可配置 |
| **管理 API** | 配置、启用/禁用、查看工具列表与服务器状态 |

---

## Agent 集成

内置外部 AI 编码 Agent 支持，通过自动生成配置将 NovaMLX 作为 LLM 后端。

| Agent | 描述 | 配置格式 |
|-------|------|----------|
| **OpenClaw** | 开源 AI Agent 框架，插件系统与工具调用 | JSON (`~/.openclaw/openclaw.json`) |
| **Hermes Agent** | 自主 AI Agent，多步推理与工具执行 | YAML (`~/.hermes/config.yaml`) |
| **OpenCode** | 基于终端的 AI 编程助手，支持代码编辑 | 环境变量 |

| 特性 | 详情 |
|------|------|
| **自动检测** | 扫描 `$PATH` 及常见安装位置中的 Agent 二进制 |
| **配置生成** | 一键生成包含端口、API Key、模型设置的配置 |
| **剪贴板复制** | 快速复制粘贴到 Agent 配置文件 |
| **安装链接** | 直达各 Agent 的 GitHub 仓库 |
| **GUI** | 菜单栏应用中独立的 Agents 标签页 |

---

## 模型管理

### 管理端点（端口 8081，Bearer 认证）

| 方法 | 端点 | 说明 |
|------|------|------|
| `GET` | `/admin/models` | 列出所有模型（下载/加载状态） |
| `POST` | `/admin/models/download` | 从 HuggingFace 下载模型 |
| `POST` | `/admin/models/load` | 加载模型到 GPU 内存 |
| `POST` | `/admin/models/unload` | 从 GPU 内存卸载 |
| `DELETE` | `/admin/models/{id}` | 删除模型文件 |
| `POST` | `/admin/models/discover` | 扫描文件系统发现新模型 |
| `GET` | `/admin/models/{id}/settings` | 获取逐模型配置 |
| `PUT` | `/admin/models/{id}/settings` | 更新逐模型配置 |
| `GET` | `/admin/api/memory` | 内存状态与压力监控 |

### 模型驱逐
- **LRU 驱逐**：超出内存预算时自动卸载非固定模型
- **TTL 自动卸载**：按模型可配置空闲超时
- **系统内存压力响应**：Critical → 卸载全部非固定模型；Warning → 仅保留 1 个
- **周期监控**（60s）：GPU 内存超过 80% 触发驱逐

---

## 内存管理

通过 `ProcessMemoryEnforcer` 实现系统级内存压力监控与自动模型驱逐。

### ProcessMemoryEnforcer

基于 Actor 的内存压力管理器，配置策略灵活：

| 特性 | 详情 |
|------|------|
| **轮询** | 1 秒间隔内存压力监控循环 |
| **软限制** | 超限时触发警告日志 |
| **硬限制** | 强制立即驱逐非固定模型 |
| **Auto 模式** | 为系统保留 `max(4GB, min(8GB, physMem/5))`，其余供 NovaMLX 使用 |
| **百分比模式** | 用户自定义物理内存百分比（如 `80%`） |
| **固定值模式** | 绝对字节限制（如 `24GB`、`4096MB`） |
| **OS 集成** | 响应 macOS 内存压力通知 |
| **GUI 配置** | 设置中提供模式选择器，按模式显示条件输入字段 |

### 内存预算追踪

每序列 KV 缓存预算与准入控制：

| 特性 | 详情 |
|------|------|
| **预留** | 每个序列在准入前预留 KV 缓存预算 |
| **释放** | 序列完成时释放预算 |
| **准入控制** | 预算不足时拒绝新请求 |
| **TurboQuant 加成** | KV 量化有效将准入容量翻倍 |

---

## 逐模型配置

每个推理参数均可按模型独立覆盖，持久化存储于 `model_settings.json`：

| 配置项 | 说明 |
|--------|------|
| `max_context_window` | 上下文长度覆盖 |
| `max_tokens` | 最大生成 Token 数 |
| `temperature` | 采样温度 |
| `top_p` / `top_k` / `min_p` | 采样策略 |
| `frequency_penalty` / `presence_penalty` / `repetition_penalty` | 重复控制 |
| `seed` | 确定性生成种子 |
| `ttl_seconds` | 空闲自动卸载超时 |
| `model_alias` | 模型 ID 别名 |
| `is_pinned` | 免驱逐保护 |
| `is_default` | 默认模型标记 |
| `kv_bits` / `kv_group_size` | TurboQuant 配置 |
| `thinking_budget` | 最大思维/推理 Token 数 |
| `display_name` / `description` | 人类可读的名称与描述 |

---

## HuggingFace 集成

直接从 HuggingFace 浏览、搜索和下载模型。

| 特性 | 详情 |
|------|------|
| **搜索** | 查询 HF Hub，支持仅 MLX 模型过滤，按趋势/下载量排序 |
| **模型详情** | 架构、标签、文件列表、许可证 |
| **下载** | 异步逐文件下载，带进度追踪和取消支持 |
| **仪表盘 UI** | 搜索框、结果列表、下载进度与取消按钮 |
| **自动注册** | 下载完成后自动发现并注册模型 |

---

## 云端模型代理

远程推理代理，使云端模型与本地模型并列显示。

| 特性 | 详情 |
|------|------|
| **协议** | OpenAI 兼容 API 透传 |
| **流式** | 完整 SSE 流式透传，逐 Token 中继 |
| **模型发现** | 云端模型与本地模型一起在 `/v1/models` 中列出 |
| **配置** | 按云端模型配置远程端点 URL 和 API Key |
| **适用场景** | 运行本地 GPU 放不下的大模型，或增加远端回退容量 |

---

## 可观测性与基准测试

### 性能基准测试
- **提示词长度**：可配置（默认：1024、4096）
- **指标**：TTFT（ms）、生成 tok/s、预填充 tok/s、峰值内存（GB）、端到端延迟（s）
- **端点**：`POST /admin/api/bench/start`、`GET /admin/api/bench/status`、`POST /admin/api/bench/cancel`

### 困惑度评估
- **指标**：逐文本困惑度、Token 计数、平均困惑度
- **端点**：`POST /admin/api/ppl/start`、`GET /admin/api/ppl/status`、`POST /admin/api/ppl/cancel`

### 推理指标
- 会话 + 持久化历史计数器
- 总请求数、生成 Token 数、平均 tok/s
- 逐模型请求计数、缓存命中率
- GPU 内存追踪

### 系统监控
- 设备信息：芯片、型号、内存、GPU/CPU 核心数、系统版本
- 更新检查：GitHub Releases API
- 指标清除：会话级和历史级

---

## Web 界面

### 聊天界面（`/chat`）
功能完备的 Web 聊天应用 — 无需管理员认证。
- 深色主题，紫色主题色
- SSE 流式响应，支持停止按钮
- 模型选择器
- 系统提示词配置弹窗
- 对话历史（localStorage，最多 50 条）
- Markdown 渲染，代码语法高亮与复制按钮
- 图片上传（拖拽、粘贴、文件选择器），支持 VLM
- 思维/推理过程气泡展示
- 图片灯箱弹窗
- API Key 输入与持久化存储
- 移动端响应式侧边栏

### 管理仪表盘（`/admin/dashboard`）
运维管理面板 — 需要管理员认证。
- 状态卡片（健康状态、GPU 内存）
- 设备信息表格
- MCP 服务器状态
- 基准测试运行器
- 困惑度运行器
- HuggingFace 模型浏览器（搜索、下载、进度）
- 下载任务管理与取消
- 指标管理（清除会话/历史）
- 每 5 秒自动刷新
- 响应式布局

---

## macOS 菜单栏应用

原生 SwiftUI 菜单栏组件，显示脑图标。

| 特性 | 详情 |
|------|------|
| **状态标签页** | 运行指示器、服务器地址、已加载模型数、GPU 内存、活跃请求、运行时间、tok/s |
| **模型标签页** | 已加载模型列表及状态指示器、已下载模型数、磁盘占用 |
| **Agents 标签页** | 外部 Agent 检测（OpenClaw、Hermes、OpenCode）、配置生成、安装链接 |
| **设置标签页** | 查看/编辑配置切换、内存限制配置、模型设置、服务器管理 |
| **轮询** | 2 秒统计刷新 |
| **窗口** | 280×200pt 浮动面板 |

---

## 国际化

完整的 i18n 系统，支持 9 种语言，覆盖所有 GUI 视图和 Web 界面。

| 语言 | 代码 |
|------|------|
| English | `en` |
| 简体中文 | `zh-Hans` |
| 繁體中文 (香港) | `zh-Hant-HK` |
| 繁體中文 (台灣) | `zh-Hant-TW` |
| 日本語 | `ja` |
| 한국어 | `ko` |
| Français | `fr` |
| Deutsch | `de` |
| Русский | `ru` |

| 特性 | 详情 |
|------|------|
| **调用模式** | `L10n.tr("key.path")` 编译期 Key 校验 |
| **覆盖范围** | 设置、Agents、聊天、菜单栏、仪表盘——所有面向用户的字符串 |
| **回退** | 缺失翻译回退到英文 |
| **Web UI** | 聊天和仪表盘同样完全国际化 |

---

## 安全与中间件

| 层级 | 实现 |
|------|------|
| **API 认证** | 端口 8080 的 Bearer Token 认证（`APIKeyAuthMiddleware`） |
| **管理认证** | 端口 8081 的 Bearer Token 或 `X-Admin-Key` 头（`AdminAuthMiddleware`） |
| **CORS** | `Access-Control-Allow-Origin: *`，完整方法和请求头支持 |
| **请求 ID** | `x-request-id` 头透传或自动生成 |
| **错误处理** | OpenAI 兼容的 JSON 错误体，正确的 HTTP 状态码 |
| **管理隔离** | 未配置 API Key 时管理 API 完全禁用 |
| **Worker 签名** | Worker 子进程 Ad-hoc 代码签名满足 macOS 进程完整性 |

---

## Homebrew 分发

通过 Homebrew 安装 NovaMLX，支持自动更新。

```bash
brew tap cnshsliu/nova
brew install --cask novamlx
```

| 特性 | 详情 |
|------|------|
| **Tap** | `cnshsliu/nova` Homebrew tap |
| **Cask** | `novamlx` — 安装 `.app` 包到 `/Applications` |
| **更新** | `brew upgrade --cask novamlx` 获取最新 GitHub Release |
| **版本检查** | 内置更新检查器查询 GitHub Releases API |

---

## 配置参考

### 服务器配置（`ServerConfig`）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `host` | `127.0.0.1` | 绑定地址 |
| `port` | `8080` | 推理 API 端口 |
| `adminPort` | `8081` | 管理 API 端口 |
| `apiKeys` | `[]` | API 密钥（空 = 禁用认证） |
| `maxConcurrentRequests` | `16` | 最大并发请求数 |
| `requestTimeout` | `300s` | 请求超时 |
| `contextScalingTarget` | `nil` | 报告 Token 数缩放目标（计费兼容） |
| `maxProcessMemory` | `auto` | 进程内存限制：`auto`、百分比（`80%`）或绝对值（`24GB`、`4096MB`） |
| `maxRequestSizeMB` | `100` | 最大请求体大小（MB） |

### 引擎配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kvCacheCapacity` | `1024` | KV 缓存块容量 |
| `maxMemoryMB` | `2048` | 模型最大 GPU 内存 |
| `maxConcurrent` | `8` | 最大并发推理任务 |

### 前缀缓存配置
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blockSize` | `64` | 每缓存块 Token 数 |
| `maxBlocks` | `4096` | 最大缓存块数 |
| `ssdMaxSizeBytes` | `100 GB` | 最大 SSD 缓存容量 |

---

## 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                   macOS 菜单栏应用                         │
│      状态 • 模型 • Agents • 设置 • 仪表盘                  │
├──────────────────────────────────────────────────────────┤
│                  NovaMLX API 服务器                        │
│  ┌─────────────────┐  ┌──────────────────────────────┐  │
│  │   推理 API       │  │         管理 API              │  │
│  │  (端口 8080)     │  │        (端口 8081)            │  │
│  │  • OpenAI        │  │  • 模型管理                    │  │
│  │  • Anthropic     │  │  • 逐模型配置                  │  │
│  │  • 向量嵌入      │  │  • 基准测试                    │  │
│  │  • 重排序        │  │  • HuggingFace 浏览器          │  │
│  │  • MCP 工具      │  │  • 内存监控                    │  │
│  │  • 云端代理      │  │  • 管理仪表盘                  │  │
│  └────────┬─────────┘  └──────────────┬───────────────┘  │
│           └──────────┬────────────────┘                   │
│                      ▼                                    │
│  ┌──────────────────────────────────────────────────┐    │
│  │            推理服务                                │    │
│  │  • 连续批处理器    • 会话管理器                     │    │
│  │  • 推测解码        • 引擎池                        │    │
│  │  • 模型配置        • MCP 客户端管理                 │    │
│  └────────────────────────┬─────────────────────────┘    │
│                           ▼                               │
│  ┌──────────────────────┬──────────────────────────┐     │
│  │   MLX 推理引擎       │  Worker 子进程             │     │
│  │  • LLM/VLM 生成      │  • stdin/stdout JSON IPC  │     │
│  │  • 结构化输出        │  • ProcessMemoryEnforcer   │     │
│  │  • TurboQuant        │  • WorkerSupervisor        │     │
│  │  • 前缀/SSD 缓存     │  • GPU 内存隔离             │     │
│  │  • 视觉特征缓存      │                            │     │
│  │  • TurnStopProcessor │                            │     │
│  │  • ControlTokenFilter│                            │     │
│  └────────────────────┬─┴──────────────────────────┘     │
│                       ▼                                   │
│  ┌──────────────────────────────────────────────────┐    │
│  │          MLX / Apple Silicon GPU                   │    │
│  │  • 惰性求值  • 统一内存  • Metal 加速              │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## 快速开始

```bash
# 编译
swift build -c release

# 启动（带 API Key）
./build/release/NovaMLX --api-key sk-your-key

# 对话补全
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "你好！"}],
    "stream": true
  }'
```
