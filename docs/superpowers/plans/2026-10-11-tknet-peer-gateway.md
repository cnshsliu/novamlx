# Tknet Peer 网关 — 工作计划

**日期：** 2026-10-11
**依据：** `docs/superpowers/specs/2026-10-11-tknet-peer-gateway-design.md`
**状态：** 待 Lucas 审阅。审阅通过之前不改代码。

这是分阶段的工作计划，不是逐步实现清单。每一段结束时都有可以单独运行的结果。某一段要开工时，再为那一段写逐步任务。

## 第一段：Peer 自己会分发

Peer 库里加上成员表、池和挑选。命令行可以增删成员和池。`serve` 在不拨隧道时也监听 `127.0.0.1:6600`。

验收：两个假的 HTTP 服务放进一个池，第一台返回 500 时，请求落到第二台。依赖图里没有 MLX。现有隧道测试仍然通过。

这一段先留在现在的仓库里，作为已经独立的 `NovaMLXTknetPeer` 和 `tknet-peer`。不搬仓库，避免和分发逻辑搅在一起。

## 第二段：NovaMLX 退出入口

把现有的池和 Tokenhub 供应商写成 Peer 配置。`local` 成员变成指向 `127.0.0.1:6590` 的 HTTP 成员。NovaMLX 去掉 `lb:` 路由、Tokenhub 代理，以及 Load Balancers、Tokenhub、Tknet Peer 三页。

验收：程序打 Peer 的 6600，后面是正在跑的 NovaMLX，补全和直打 NovaMLX 一致。再打 NovaMLX 的 `lb:` 得到 404。

NovaMLX 的分布式切层、本机模型加载和本机 API 不动。

## 第三段：Mac 应用壳

做一个只含 Peer 的菜单栏应用：成员、池、隧道开关、连接状态。启动后本机端口就开着。没有按过停止就自动拨隧道，失败最多再试 3 次。停止之后下次启动不拨，端口仍在。

沿用这台机器已有的 peer 身份和连接密钥，不重新注册。同一时间只允许一个 Peer 进程。

验收：不打开 NovaMLX 也能看成员和池。隧道连上后，tknet 的请求进同一个池，发给 tknet 的帧里没有成员地址。

## 第四段：离开这台 Mac

把 Peer 的库、命令行和测试放到单独的仓库，发布不带 MLX 的命令行。先做 Linux 和 Windows 的命令行。图形界面以后再说。

验收：在没有 NovaMLX 源码的机器上编出 `tknet-peer`，并把一个池分到一台局域网里的推理服务。

## 现在不做

- 不重新注册 peer。
- 不把推理代码拷进 Peer。
- 不在第一段就拆 Git 仓库。
- 不为 Windows 或 Linux 做界面。
- 不改 tknet 网站的调度和付款。
