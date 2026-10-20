# Load Balancers

**Pitch:** Send `lb:<slug>` and let a named pool choose a local model or a Tokenhub provider.

请求使用 `lb:<slug>`。池成员可以是本地推理，也可以是 Tokenhub 上的远端 provider。

## Who it is for

要把流量分到多台本地模型或多个云端 provider，并自己指定策略的人。

## What it does

| 功能 | 详情 |
|------|------|
| **策略** | Tiered（默认）、round-robin、weighted、least-latency |
| **成员** | `.local`（推理服务）与 `.remote`（Tokenhub provider） |
| **LBProxy** | 每个请求一个 actor：选择成员、按序尝试、失败再试 |
| **管理 API** | 9 个端点：LB 与成员的 CRUD、试跑路由、统计 |
| **按成员统计** | 成功、失败、平均延迟；UI 中可见 |
| **UI** | `LoadBalancersPageView`：手风琴行、成员选择器、策略配置、每行 Play |
| **协议** | OpenAI、Anthropic、Responses。Anthropic 走翻译桥；Responses 走 `tknet:` 重写 |

菜单栏 Load Balancers 页与 Tokenhub 并列。Play 会跳到 Playground 并预选该 LB。

## How to try

先在 UI 或管理 API 创建 LB，再用 `lb:<slug>` 作为模型名。功能目录中的示例：

```bash
curl http://localhost:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lb:my-pool",
    "messages": [{"role":"user","content":"Hello"}]
  }'
```

管理端包括 LB 的 CRUD、成员管理与试跑路由。Anthropic 客户端打 `/v1/messages` 且模型名为 `lb:<slug>` 时，与 Tokenhub 透传共用路径，需要翻译时翻译桥自动生效。

## Limits / notes

- 策略名为目录所列四项。默认是 Tiered。
- 目录没有写重试次数或「首字节之后不再重试」的规则。
- 成员只有两类：本地推理与 Tokenhub provider。

## Related

- [Tokenhub](tokenhub.md)
- [Anthropic↔OpenAI 翻译桥](anthropic-openai-bridge.md)
- [菜单栏应用](menu-bar.md)
- 设计说明（仓库已有）：`docs/superpowers/specs/2026-06-14-tokenhub-loadbalance-design.md`

正文来自 `features.zh-CN.md`「负载均衡器」与「快速开始」；标题与 pitch 为英文。
