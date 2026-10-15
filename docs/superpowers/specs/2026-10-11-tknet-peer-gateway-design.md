# Tknet Peer 网关 — 需求说明书

**日期：** 2026-10-11，2026-10-14 按已落地行为修订
**状态：** 四段已经落地。本文描述现在的网关。还没做的和当初明确不做的，都在 `TODO.md` 的「Tknet Peer — 2026-10-14 再审」，逐条实现或再决定。
**取代：** `docs/superpowers/specs/2026-09-24-tknet-peer-design.md` 里「Mac 上 Peer 嵌在 NovaMLX 里、用户只开一个应用」这一条。隧道协议、注册身份、密钥不出 Peer，仍然有效。

## 目标

Tknet Peer 是用户这边的本地网关。tknet 只跟这个网关打交道。网关后面谁在算，tknet 不用知道。

后面可以是这台电脑上的一个推理程序，可以是局域网里另外几台机器，也可以是一台超算。买来的请求从隧道进网关，网关再把它交给后面真正干活的服务，结果再原路送回 tknet。

引擎可以是 NovaMLX、Ollama、oMLX、vLLM，或任何能接 OpenAI 兼容 HTTP 的服务。系统先做 Mac，随后是 Windows 和 Linux。推理程序由用户自己安装、自己启动。

负载均衡放在 Peer 里。NovaMLX 里不再有负载均衡。NovaMLX 以后只做本机推理：在自己的端口上回答自己加载的模型。

用户就算不连接 tknet，只要想把多台推理服务合成一个入口，也用 Peer。Peer 在本机开一个 OpenAI 兼容端口。隧道是这个网关的另一面，不是另一套程序。

## 已经定下来的决定

1. **一个进程，两面。** 同一份 Peer 既在本机接用户的程序，也在连上 tknet 时接买来的请求。两边进同一个负载均衡。
2. **NovaMLX 只是一个成员。** Peer 用 HTTP 访问它，地址默认 `http://127.0.0.1:6590`。Peer 不链接 MLX，不读 NovaMLX 的模型管理器。
3. **成员名单跟着 Peer。** 今天远程成员存在 NovaMLX 的 Tokenhub 里（地址、密钥、上游模型名）。这份名单搬到 Peer。NovaMLX 不再代理到别的引擎，也不再代理到 tknet。
4. **沿用现有身份。** 不重新注册。这台机器已有的 peer 和连接密钥继续用。密钥仍然只留在本机。
5. **NovaMLX 自己的分布式切层不动。** 那是一个模型拆到多台 Mac 上算，不是网关在多个引擎之间挑一台。它留在 NovaMLX。
6. **完整支持 Anthropic Messages API。** 2026-10-15 定。成员类型保留。本机端口和隧道都按该协议转发：路径 `/v1/messages`，认证用 `x-api-key` 和 `anthropic-version`。2026-10-14「先从表单拿掉、转发一律拒绝」作废，以后不再执行。
7. **上游请求不预检密钥。** 2026-10-15 定。`PoolForwarder.makeUpstreamRequest` 保持现状。成员 key 为空时用 factory 的 key。两边都空就发出去，失败就失败。这是管理员配置，不在转发层拦。2026-10-14「缺密钥直接失败、不要空认证打出 401」作废。

## 现在的行为

成员保存地址、密钥引用、上游模型名、权重、是否启用、类型（OpenAI 兼容、本机 NovaMLX、Anthropic）。不保存密钥原文，也没有免费 / 付费标记。旧配置里的 `isFree` 读进来就丢掉。

池有名字、挑选方式、成员、重试次数、是否启用。一个对外模型名对应一个池。挑选方式四种：`round_robin`（默认）、`weighted`、`lowest_latency`、`random`。旧配置里的 `tiered` 读成 `round_robin`。不再把回环地址或「免费」排到前面：网关这台机器和真正推理的机器可以不是同一台。

失败时换下一台，次数是 `maxRetries`（默认 3）。成员返回 4xx 时把状态和正文交回调用方，不换下一台。连接失败、超时或 5xx 才换。池名不存在是 404。池停用或没有可转发的成员是 503。全部尝试都失败是 502。

本机端口默认 `127.0.0.1:6600`。提供 `GET /v1/models`，以及这些 POST：`/v1/chat/completions`、`/v1/messages`、`/v1/embeddings`、`/v1/images/generations`、`/v1/images/edits`、`/v1/images/background-removals`、`/v1/audio/speech`、`/v1/audio/transcriptions`、`/v1/decisions`。不做 `/v1/images/variations`。池子带接口字段；路径不在接口里时直接拒绝，不打到上游。图像生成与图像编辑是两种独立接口。Anthropic Messages 只走 `/v1/messages`，认证是 `x-api-key` 加 `anthropic-version`，客户端带来的 `anthropic-beta` 原样转给上游。`model` 是池的名字。转发时改成该成员的上游模型名，上游路径与进来的路径相同。图像编辑、去背景和转写可以是 multipart，只改 `model` 字段。回环地址或类型为 Local NovaMLX 的成员会带 `X-No-Autoload: 1`，避免本机 NovaMLX 为一次失败排队加载。隧道接受 `apiFormat: openai` 和 `apiFormat: anthropic`。没有路径的 OpenAI 帧只当作对话；没有路径的 Anthropic 帧当作 `/v1/messages`。详见 `docs/superpowers/specs/2026-10-14-peer-api-consistency.md`。

隧道和本机端口进同一套挑选。发给 tknet 的帧里没有成员地址。Mac 菜单栏应用有隧道、成员、池、账号（认领和收益）。命令行和这个应用共用 `~/.config/tknet-peer`，密钥在该目录下收紧权限的文件里，不在钥匙串。同一时间只跑一个 Peer 进程。

库和命令行在公开仓库 `https://github.com/cnshsliu/tknet-peer`，不链接 MLX。Mac 应用留在本仓库，按 `Package.swift` 里的版本号依赖那个包。

## 用户能看到什么

### 不连 tknet

Peer 监听 `127.0.0.1:6600`（可改）。用户的程序把 `base_url` 指到这里。

- 请求里的模型名是一个池的名字，例如 `coding`。Peer 在这个池里挑一台健康的成员，把模型名改写成那台机器要的名字，带上那台机器的密钥，转发过去，再把流式结果送回。
- 池里可以只有一台，也可以有多台。只有一台时，行为就是直通。
- 挑选方式是 `round_robin`（默认）、`weighted`、`lowest_latency`、`random`。不按本机或免费优先。
- 没有可转发的成员时返回 503。全部成员都失败时返回 502，并带上最后一次错误。

### 连上 tknet

还是这一个进程。启动后如果用户没有按过停止，就自动拨隧道；失败最多再试 3 次，然后停到下次启动或用户再按开始。只有停止会让下次启动不再自动拨。

tknet 看见的是这个 Peer 公布的能力：每个在卖的名字对应一个池。请求进隧道之后，走和本机端口同一套挑选和转发。tknet 不知道后面有几台机器。

### NovaMLX

NovaMLX 的端口只回答本机已加载的模型。请求里出现 `lb:` 或一个远程供应商名字时，它回答 404，并说明这类入口在 Peer。

侧边栏不再放 Load Balancers、Tokenhub、Tknet Peer 这三页。模型、对话、语音、图像这些本机能力留下。

## 组成

| 部分 | 现在做什么 |
|---|---|
| 成员表 | 地址、密钥引用、上游模型名、权重、启用、类型。不保存密钥原文，不标记免费 |
| 池 | 一个名字、四种挑选方式之一、一组成员、重试次数、启用。不检查各成员是不是同一个词表 |
| 转发 | 改写模型名，按进来的路径转发，流式回传，取消，超时。不跑模型 |
| 本机端口 | `GET /v1/models`，以及对话、嵌入、图像、语音的 POST。不提供 NovaMLX 的管理接口 |
| 隧道 | 注册、hello、心跳、请求、取消、能力更新。帧里没有成员地址 |
| 命令行 | `setup`、`serve`、`earnings`，成员和池的增删改。不安装推理引擎 |
| Mac 菜单栏 | 隧道、成员、池、账号（认领和收益）。不加载模型 |

密钥用 `FileSecretStore`，路径在 `~/.config/tknet-peer/secrets`。Mac 应用和命令行共用这一份。同一时间只跑一个 Peer 进程。

## 数据流

本机程序把请求交给 Peer。Peer 用模型名找到池，核对路径是否属于该池接口，再挑一个成员，把 body 里的 `model` 换成该成员的上游模型名，请求发到该成员上与进来相同的路径。tknet 隧道帧带路径时按该路径转发；没有路径的旧帧只走对话。成员返回的状态码、用量和首字时间记在这一跳上。挑下一台时不把上一家的半截输出拼进去。

成员指向 NovaMLX 时，Peer 加上现有的 `X-No-Autoload: 1`。模型没加载就尽快失败，让池去换一台，而不是在 NovaMLX 里排队等加载。

## 错误

- 池名不存在：404。
- 池被停用，或没有可转发的成员：503。
- 成员返回 4xx：把该状态和错误正文交回调用方，不把这次算成「再换一台能好」。
- 成员连接失败、超时或 5xx：记一次失败，换下一台，直到用完重试。
- 隧道断开：本机端口照常服务。自动重连只作用于隧道。

## 迁移

已有的池和 Tokenhub 供应商从 NovaMLX 的配置库读出来，写成 Peer 的配置。`local` 成员写成指向 `127.0.0.1:6590` 的 HTTP 成员，上游模型名就是原来的模型 id。`remote` 成员写成原来的 endpoint、密钥和 remote model。读完就停，不在 NovaMLX 里留一份还会被路由使用的副本。

现有 `tknet-peer` 配置里的单条 source 变成一个只含一台成员的池，名字用原来公布给 tknet 的模型名。这样已经在卖的名字不会变。

## 已经达到的验收

1. 不启动 NovaMLX，Peer 也能把一个池分到两个假的 HTTP 上游，并在第一台返回 500 时改走第二台。
2. 池里有一台指向正在跑的 NovaMLX 时，本机程序打 Peer 的 6600 端口能得到和直打 NovaMLX 相同的补全。
3. NovaMLX 收到 `lb:` 或 Tokenhub 供应商名时返回 404，侧边栏没有 Load Balancers、Tokenhub、Tknet Peer 这三页。
4. 隧道连上时，tknet 的一次请求进同一个池，成员地址不出现在发给 tknet 的帧里。
5. 用户按停止之后，下次启动不拨隧道，本机 6600 仍然可用。
6. `github.com/cnshsliu/tknet-peer` 单独能编出 `tknet-peer`，依赖图里没有 MLX。

还没做的能力、以及当初写明不做、现在要再决定的事项，见 `TODO.md`。

## 方案取舍

考虑过三条路。

1. **负载均衡留在 NovaMLX，Peer 只转发到 `lb:`。** 代码最短，但用户不装 NovaMLX 就没有池，也和「NovaMLX 只做本机推理」相反。
2. **负载均衡放进 Peer，Peer 是唯一入口。** 连不连 tknet 都用它。NovaMLX、Ollama、vLLM、超算都只是成员。这是选定的做法。
3. **再做一个独立的负载均衡进程，Peer 和 NovaMLX 都去调它。** 多一个必须启动的程序，也多一条故障。用户已经说负载均衡就在 Peer 里。

## 工作顺序

四段都已落地，记录在 `docs/superpowers/plans/2026-10-11-tknet-peer-gateway.md`：Peer 自己分发、NovaMLX 退出入口、Mac 菜单栏应用、库和命令行拆成不带 MLX 的包。接下来只从 `TODO.md` 里逐条做。
