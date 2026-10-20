# TCC Watcher

**Pitch:** A small helper that dismisses macOS privacy prompts blocking automation, after you grant it Accessibility once.

`TCCWatcher.app` 通过 System Events 关掉挡住自动化的 TCC 隐私窗。macOS 14 及以后需要你在辅助功能里手动加上这个 app。

## Catalog

| 功能 | 详情 |
|------|------|
| **目的** | 自动关闭阻断自动化的 macOS TCC 隐私弹窗 |
| **Bundle** | `TCCWatcher.app`（`com.novamlx.TCCWatcher`） |
| **安装** | `Scripts/install-tcc-watcher.sh` 配置 LaunchAgent |
| **机制** | 用 System Events 监视 TCC 窗。`inspectWindow` 内联在主 tell 块里（修复上下文传递问题） |
| **一次性** | 系统设置 → 隐私与安全性 → 辅助功能，用 `+` 加入 `TCCWatcher.app`。macOS 14+ 不会自动弹出授权 |

未挂载的外置模型目录会被直接跳过，以免触发 TCC，见 [模型管理](model-management.md)。那是另一条路径，不是这个 watcher。

正文来自 `features.zh-CN.md`「TCC 监控器」；标题与 pitch 为英文。
