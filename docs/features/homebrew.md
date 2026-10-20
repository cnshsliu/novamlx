# Homebrew Distribution

**Pitch:** Install from the Homebrew formula, or package a DMG; the worker must be codesigned or macOS kills it.

Homebrew formula、`./build.sh` 与 DMG。部署后需要对 bundle 做 ad-hoc codesign，否则系统会 SIGKILL worker。

## Catalog

| 渠道 | 详情 |
|------|------|
| **Formula** | `Formula/novamlx.rb` |
| **构建** | `./build.sh`：UUID 同步、codesign、MLX shader 编译、bundle 组装 |
| **DMG** | 脚本打磁盘映像 |
| **Codesign** | 部署后需要 `codesign --force --deep --sign -`，否则 macOS 会 SIGKILL worker |
| **Entitlements** | `audio-input`、按需的 `com.apple.security.device.camera`、用 PlistBuddy 写入的 `NSMicrophoneUsageDescription` |

功能目录里的安装示例是 `brew install --head novamlx`。根 README 写的是 `brew tap cnshsliu/novamlx` 然后 `brew install novamlx`。以你使用的 tap 说明为准。

麦克风用途字符串同时见 [音频](audio.md)。

正文来自 `features.zh-CN.md`「Homebrew 分发」与「快速开始」；标题与 pitch 为英文。
