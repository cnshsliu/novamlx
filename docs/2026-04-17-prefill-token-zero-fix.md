# Prefill First Token = 0 问题分析与修复方案

> 日期: 2026-04-17
> 状态: 进行中 (部分修复已提交, 待验证)

---

## 1. 问题描述

用户通过 Claude Code (Anthropic Code) 使用 NovaMLX 作为推理后端, 问了一个简单问题 "1+1=?"。Claude Code 会把完整对话上下文作为 prompt 发送给 NovaMLX, 导致 prompt token 数高达 **13317**。

服务端日志显示:

```
[Prefill:ED9C67C7] Manual prefill — 13317 tokens
[Prefill:ED9C67C7] First token: 0
MemoryBudget: released 16993MB for sequence ED9C67C7
```

`First token: 0` 触发了 `guard firstTokenId != 0` 保护, 抛出异常 "Prefill produced no first token"。这导致:

1. **Claude Code 客户端** 收到错误后每 60 秒重试一次, 最多重试 10 次, 共等待约 10 分钟
2. **NovaMLX 服务端** 每次重试都重新执行 prefill (13317 tokens), 消耗大量 GPU 算力和时间 (每次 prefill 约 20-30 秒)
3. **SSE keep-alive 连接** (DC127355) 持续发送心跳包, 长达 245+ 秒, 0 个 token 输出
4. 双方都在浪费时间, 用户完全不知道发生了什么

---

## 2. 根本原因分析

### 2.1 直接原因: Token ID 0 被错误地视为无效

`FusedBatchScheduler.swift` 中的 prefill 代码:

```swift
let firstTokenId = MLXSerializer.shared.perform {
    let prefillSampled = sampler.sample(logits: lastLogits[0..., -1, 0...])
    eval(prefillSampled)
    return prefillSampled.item(Int.self)
}

guard firstTokenId != 0 else {
    throw NovaMLXError.inferenceFailed("Prefill produced no first token")
}
```

**问题**: Token ID 0 对于大多数 tokenizer 是合法的:
- Phi-3.5 的 tokenizer 中, ID 0 是 `<unk>` (unknown token)
- 某些模型中 ID 0 是 `<pad>` 或其他特殊 token
- 它不是 "没有产出 token" 的信号

Sampler (categorical sampling with temperature) 在经过 13317 token 的 forward pass 后, 对最后一个位置的 logits 进行采样, 返回了 ID 0。这可能是因为:
- 13317 token 的序列导致 logits 数值不稳定 (梯度消失/爆炸), 使 token 0 的概率异常高
- 或者纯粹是采样概率的结果 — 如果 token 0 在 softmax 后概率最高, argMax 就会选它

**关键**: 我们没有检查 logits 是否真的有 NaN/Inf, 只是盲目地认为 ID 0 = 错误。

### 2.2 间接原因: 缺乏上下文窗口验证

13317 tokens 对 Phi-3.5-mini (context window = 128K) 来说完全没问题。但当前代码:
- `preflightCheck()` 只检查 GPU 内存是否足够, 不检查 prompt 是否超过模型 context window
- 没有在 API 层做 prompt token 数的快速估算和校验
- 即使 prompt 超过 context window, 也会尝试 prefill, 浪费 GPU 时间后才报错

### 2.3 间接原因: SSE 错误传播机制不完善

虽然 SSE stream handler 有 catch 块 (会发送 `event: error`), 但:

1. **非流式请求** (`stream=false`) 的错误通过 HTTP response 返回, 但 Claude Code 客户端似乎在收到错误后仍然重试 (这是客户端行为, 我们无法控制)
2. **Keep-alive 心跳** 在 inference 失败后仍然持续发送 — 因为 keep-alive Task 和 inference Task 是独立的, inference 失败后 keep-alive 不会自动停止
3. **没有超时机制** — 服务端不会主动终止过长的 prefill 或等待

### 2.4 连锁反应: 资源浪费循环

```
Client retry (60s timeout)
    → NovaMLX prefill (30s for 13K tokens)
        → Sampler returns token 0
            → Guard throws error
                → Budget released
                    → Client gets error, retries again
                        → Repeat ×10 = 10 minutes of wasted GPU time
```

---

## 3. 修复方案

### Fix 1: 移除 Token ID 0 的错误保护 ✅ 已完成

**文件**: `Sources/NovaMLXEngine/FusedBatchScheduler.swift`

**改动**:
- 移除 `guard firstTokenId != 0` — Token ID 0 是合法 token
- 改为检查 logits 是否包含 NaN/Inf (通过 sum() 检测)
- 只有 logits 真正损坏时才报错 (firstTokenId == -1)
- 添加 logit 统计信息 (sum, max, min, shape) 用于诊断

**理由**: 如果模型真的产出了 token 0 (`<unk>`), 说明它对这个 prompt 不确定, 但我们应该让推理继续, 而不是直接放弃。后续 decode 步骤可能会产出正常 token。

### Fix 2: 添加上下文窗口验证 ✅ 已完成

**文件**:
- `Sources/NovaMLXInference/InferenceService.swift` — 添加 generate() 和 stream() 的上下文窗口检查
- `Sources/NovaMLXEngine/MLXEngine.swift` — tokenizeMessages 改为 public

**改动**:
- 在 generate() 和 stream() 入口处, 用 tokenizer 计算 prompt token 数
- 如果 promptTokens + maxTokens > contextLength, 立即返回错误
- 非流式: throw NovaMLXError.inferenceFailed (返回 500 给客户端)
- 流式: 创建一个立即抛出错误的 AsyncThrowingStream

**效果**: 对于真正超长的 prompt, 客户端在几毫秒内收到明确错误, 不需要等待 prefill。

### Fix 3: 改进 API 层错误返回格式 (建议)

**问题**: 当前错误返回的是通用 HTTP 500 + NovaMLXError.inferenceFailed 消息。对于 Anthropic API 格式, 应该返回更符合 Claude API 的错误格式:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "Context window exceeded: prompt has 13317 tokens..."
  }
}
```

这样 Claude Code 客户端能更智能地处理错误 (可能减少 prompt 长度后重试, 而不是盲目重试)。

### Fix 4: 添加 Prefill 超时机制 (建议)

**问题**: 13317 token 的 prefill 在 Apple Silicon 上需要 20-30 秒。如果并发多个大 prompt 的 prefill, GPU 会被长时间占用。

**建议**:
- 为 prefill 添加超时 (如 120 秒)
- 超时后取消 prefill, 返回 408 Request Timeout
- 或者在 API 层根据 token 数估算 prefill 时间, 如果超过客户端 timeout, 直接拒绝

### Fix 5: Keep-alive 错误感知 (建议)

**问题**: `withSSEKeepAlive()` 中的 heartbeat Task 在 inference 失败后不会自动停止。

**建议**: 当 inference stream 结束 (正常或异常), heartbeat Task 应该立即停止:

```swift
// 当前: heartbeat 只在 Task.cancelled 时停止
while !Task.isCancelled { ... }

// 改进: 检测 inference stream 结束信号
// 用 onTermination 回调取消 heartbeat
```

---

## 4. 验证计划

### 4.1 基础验证

```bash
# 构建
swift build -c release

# 启动服务
.build/release/NovaMLX

# 发送大 prompt 测试 (模拟 Claude Code 场景)
python3 -c "
prompt = 'Hello ' * 3000  # ~13000 tokens
import json, urllib.request
req = urllib.request.Request(
    'http://127.0.0.1:6590/v1/messages',
    data=json.dumps({
        'model': 'mlx-community/Phi-3.5-mini-instruct-4bit',
        'max_tokens': 100,
        'messages': [{'role': 'user', 'content': prompt + 'What is 1+1?'}]
    }).encode(),
    headers={
        'Content-Type': 'application/json',
        'x-api-key': 'abcd1234',
        'anthropic-version': '2023-06-01'
    }
)
try:
    resp = urllib.request.urlopen(req, timeout=120)
    print(resp.read().decode())
except Exception as e:
    print(f'Error: {e}')
"
```

### 4.2 预期结果

- Token ID 0 不再被视为错误, 推理继续
- 即使第一个 token 是 `<unk>`, 后续 decode 产出的 token 应该正常
- 如果 prompt 真的超过 context window, 立即返回错误而不是挂起
- 日志中显示 logit 统计信息, 帮助诊断数值稳定性问题

---

## 5. 未解之谜

1. **为什么 13317 token 的 prompt 采样后第一个 token 是 ID 0?**
   - 可能是 logits 数值不稳定 (长序列导致 attention score 衰减)
   - 可能是采样温度/随机性的结果
   - 需要看 logit 统计信息才能确定 (Fix 1 中已添加诊断日志)

2. **Claude Code 为什么会发送 13317 token 的 prompt?**
   - Claude Code 会把完整对话历史 + 系统提示 + 工具定义作为上下文
   - 在编程场景中, 上下文很容易累积到 10K+ tokens
   - 这不是 bug, 是正常的 LLM 编程助手行为

3. **SSE DC127355 为什么持续 245+ 秒?**
   - 这可能是 Claude Code 维持的一个独立 SSE 连接
   - 与 prefill 错误可能不是同一个请求
   - 需要进一步调查客户端行为

---

## 6. 优先级

| Fix | 优先级 | 状态 |
|-----|--------|------|
| Fix 1: 移除 token 0 保护 | P0 - 紧急 | ✅ 已完成 |
| Fix 2: 上下文窗口验证 | P1 - 重要 | ✅ 已完成 |
| Fix 3: API 错误格式 | P2 - 建议 | ❌ 待做 |
| Fix 4: Prefill 超时 | P2 - 建议 | ❌ 待做 |
| Fix 5: Keep-alive 错误感知 | P2 - 建议 | ❌ 待做 |
