# macOS Menu Bar App

**Pitch:** A native SwiftUI menu bar app: status icon, dropdown, and a window for models, chat, keys, and cluster.

原生 SwiftUI 菜单栏应用。状态图标、下拉菜单，以及弹出窗口里的各功能页。

## Who it is for

用菜单栏看吞吐、下载模型、试对话、管理 Key 与集群的人。

## What it does

### 页面

| 页面 | 内容 |
|------|------|
| **Status** | 实时 TPS 曲线（90 个采样、峰值、零值裁剪）、CPU / 内存 / GPU、设备信息、峰值 TPS |
| **Dashboard** | 已加载模型、活跃请求、内存、uptime、快速加载 |
| **Local Inference** | 已激活模型：盯住（Pin）、卸载、复制名称、Play 到 Playground。盯住打开时卸载按钮不出现，重启仍加载。已下载、尚未加载：类型分栏（全部 / LLM / VLM / Embed / Audio / Image）、按磁盘筛选、显示辅助模型（默认隐藏 DSpark / DFlash / MTP）。模型卡 |
| **Downloads** | 分类、推荐模型卡、分阶段进度、卡顿检测、镜像配置 |
| **Playground（Chat）** | LLM / ASR / TTS / Image 统一入口；按模型类型识别模式。Picker 区分本地直连与 Tokenhub HTTP。参数滑块、Disable-Thinking、粘性自动滚动、复制 cURL（OpenAI / Anthropic / Responses）。ASR 麦克风、TTS 扬声器、语音克隆 |
| **Tokenhub** | 20+ 提供商目录与 CRUD。每个 provider 的模型可复制、可 Play。端点测试 |
| **Load Balancers** | 手风琴行、本地与远端成员、策略与统计、每行 Play |
| **API Keys** | 整行手风琴、显示明文、复制、用量条、速率限制、白名单 |
| **Cluster** | Thunderbolt 与 ARP 扫描、Worker 健康、按节点的模型就绪 |
| **Settings** | HF endpoint 与镜像、内存模式、日志级别、集群开关、TurboQuant、tknet.ai 账号 |
| **Audio** | 独立的 ASR / TTS 页（与 Playground 分开） |
| **Agents** | 安装、启动、配置 OpenClaw、Hermes、OpenCode |

### 跨页面

| 功能 | 详情 |
|------|------|
| **Pick-to-Playground** | Active Models、Tokenhub 模型、Load Balancers 上的 `play.circle`：打开 Playground 并预选该模型 |
| **复制 cURL** | Playground 参数区：OpenAI / Anthropic / Responses。输出里是 `${NOVA_API_KEY}` 占位，密钥不进剪贴板 |
| **粘性滚动** | `MessageListBottomOffsetKey`，阈值 80 pt |
| **模型 picker** | `inferModeFromName` 与 `autoDetectMode` 在 LLM / ASR / TTS / Image 之间切换 |
| **HF Endpoint** | 切换镜像不必重启，经 `NovaMLXConfiguration.shared` 生效 |

界面文案覆盖 9 种语言。见 [国际化](internationalization.md)。

## How to try

启动 NovaMLX。菜单栏出现状态图标，下拉可看服务状态；弹出窗口进入上表各页。

从 Playground 复制 cURL 时，命令里的密钥位置是 `${NOVA_API_KEY}`，不是当前明文 key。

在 Downloads 或 Local Inference 里对模型点 Play，会进入 Playground 并选中该模型。Load Balancers 与 Tokenhub 模型行同样有 Play。

## Limits / notes

- 本页是菜单栏信息架构，不是每个控件的操作说明。
- 复制 cURL 故意不把密钥放进剪贴板。
- 根 README 写启动后服务在 `localhost:8080`。功能目录的默认公共端口是 `6590`。以本机配置为准，见 [配置](configuration.md)。

## Related

- [模型管理](model-management.md)
- [Tokenhub](tokenhub.md)
- [负载均衡](load-balancers.md)
- [API Key](api-keys.md)
- [Agent](agents.md)
- [国际化](internationalization.md)

正文来自 `features.zh-CN.md`「macOS 菜单栏应用」；标题与 pitch 为英文。
