# 自适应验证长度（DFlash2 / DSpark）

## Title

DFlash2 与 DSpark：按草稿置信度决定目标模型验证多长

## One-line pitch

DFlash2 and DSpark verify a confidence-scheduled prefix instead of a fixed draft block, and greedy acceptance is unchanged.

DFlash2 与 DSpark 按置信度只验证一段草稿前缀，贪心接受规则与固定块长相同。

## Who it is for / when to care

已经在 Mac 上用 NovaMLX 跑 DFlash2 或 DSpark，并且固定块长在低置信度尾巴上仍在做完整目标模型验证的人。对照固定块长看接受长度和 tok/s 的人也可以用同一开关做 A/B。

People running DFlash2 or DSpark in NovaMLX who want the target model to score fewer low-confidence draft tokens, and who want a fixed-block switch for comparison.

## What shipped (user-visible behavior)

DFlash2 与 DSpark 默认改为自适应验证长度。草稿模型仍一次提出预算内的 token。调度器选择期望吞吐最高的前缀，只把这段前缀交给目标模型。

- DSpark 使用检查点自带的 confidence head（sigmoid）。头缺失或置信度全为 0 时，退回原来的固定长度。
- DFlash2 没有 confidence head。用草稿分数第一名与第二名的差距，经 sigmoid（默认尺度 2）得到 0–1 的置信度。
- 接受规则不变：目标模型对这段前缀做 argmax，留下最长匹配草稿，再附上第一个不一致位置的目标 token。温度必须为 0，否则这两条路径仍是普通单 token 解码。
- 固定块长仍可用。请求 `"adaptive_verify": "fixed"`，或进程环境变量 `NOVAMLX_ADAPTIVE_VERIFY=fixed`（`off` / `0` 同样关闭）。未设置时为 adaptive。
- 默认：最短前缀 1，最长为草稿预算；固定成本 2；快速 SDPA 窗口内每个草稿的边际成本 0.40，窗口外 0.55；快速窗口为 4 个草稿。DSpark 的固定模式仍停在这个窗口（最多 4 个草稿）。自适应模式只在置信度足够高时越过它。
- 加速器挂载不变：已知更快自动挂上，已知更慢不加载，不确定则加载时问一次并记住。自适应验证不会把投机解码改成永远开启。
- 每轮日志包含 `draft`（草稿长度）、`verified`（实际验证长度）、`accepted`（接受的草稿数）。

请求字段 `adaptive_verify_min` / `adaptive_verify_max` 覆盖最短与最长前缀。成本与窗口用环境变量：`NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST`、`NOVAMLX_ADAPTIVE_VERIFY_MARGINAL_COST`、`NOVAMLX_ADAPTIVE_VERIFY_SLOW_MARGINAL_COST`、`NOVAMLX_ADAPTIVE_VERIFY_FAST_DRAFTS`、`NOVAMLX_ADAPTIVE_VERIFY_MARGIN_SCALE`、`NOVAMLX_ADAPTIVE_VERIFY_TEMPERATURE`。

Default is adaptive. Fixed block remains available. The attach policy for companions is unchanged.

## How to try it

不加载模型，只看调度器选出的长度：

```bash
nova adaptive-verify --confidence 0.92,0.88,0.84,0.70,0.25,0.15,0.10
nova adaptive-verify --mode fixed --confidence 0.92,0.88,0.84,0.70,0.25
```

Mac 上对已加载的目标模型做 greedy A/B。DFlash2 / DSpark 伴侣按现有策略挂载，不要另开一套。两次请求除 `adaptive_verify` 外保持相同，`temperature` 为 0。

```bash
nova chat <model-id> --adaptive-verify adaptive
nova chat <model-id> --adaptive-verify fixed
```

或：

```bash
curl http://127.0.0.1:6590/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"<model-id>","temperature":0,"max_tokens":128,"adaptive_verify":"adaptive","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}]}'
```

把 `adaptive_verify` 换成 `fixed` 即为固定块长。日志里比较 `draft`、`verified`、`accepted`。进程级关闭：启动 NovaMLX 时设置 `NOVAMLX_ADAPTIVE_VERIFY=fixed`。

On a Mac, run the same greedy request twice, once with `adaptive_verify` set to `adaptive` and once to `fixed`. Compare `draft`, `verified`, and `accepted` in the log.

## Limits / out of scope

- 本机没有跑通 Mac GPU 吞吐对比。调度逻辑有单元测试；端到端 tok/s 要在 Apple Silicon 上用上面的 A/B 测。
- 不改变 n-gram 或普通小草稿模型的 `num_draft_tokens`。
- 不训练新的 confidence head，也不移植 Wide-EP、集群或 CUDA。
- 验证宽度越过快速 SDPA 窗口时，接近并列的 argmax 仍可能与逐 token 解码不同。这是既有的多 token 验证限制。
- 低并发时更短的验证可能更慢。此时用 `fixed`，或增大 `NOVAMLX_ADAPTIVE_VERIFY_FIXED_COST`。

No Mac GPU throughput numbers are claimed here. N-gram drafts, new confidence-head training, and cluster ports are out of scope.

## Suggested video beats

- 固定块长会把低置信度的草稿尾巴也送进目标模型。
- 自适应模式按置信度只验证一段前缀；接受规则与固定块相同。
- 同一条 temperature 0 的请求，切换 `adaptive` 与 `fixed`，看日志里的 `draft`、`verified`、`accepted`。
- 加速器该不该挂上，仍按原来的已知快慢策略，不因这个开关变成永远开启。

## Links

- [推测解码](../features/speculative-decoding.md)
- `nova adaptive-verify`
