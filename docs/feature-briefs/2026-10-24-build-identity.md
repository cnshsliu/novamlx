# 状态页标明当前是哪一份构建

## Title

状态页一行看清 Debug / Release、构建时间和 App 路径

## One-line pitch

The Status page footer names the running configuration, the build clock, and the `.app` path, so `/Applications` and a repo copy are not the same mystery process.

状态页底部一行写出 Debug 或 Release、构建到秒的时间，以及正在跑的 `.app` 路径。

## Who it is for / when to care

机器上同时有 `/Applications/NovaMLX.app` 和仓库 `dist/NovaMLX.app`，或者 debug 与 release 都装过，打开窗口后分不清当前进程是哪一份的人。

People who keep both an `/Applications` copy and a repo `dist` copy, or both a debug and a release build, and need to see which process the window belongs to.

## What shipped (user-visible behavior)

- 状态页最下面一行小字：`Debug · v1.7.0 · yyyy-MM-dd HH:mm:ss · /path/to/NovaMLX.app`。Release 构建把第一个词换成 `Release`。
- 时间是正在运行的可执行文件的修改时间，精确到秒。
- 路径是当前进程的 `.app`。`/Applications/NovaMLX.app` 和仓库里的 `dist/NovaMLX.app` 会显示成不同路径。
- 菜单栏标题是 `NovaMLX Debug` 或 `NovaMLX Release`。点开菜单，最上面两行是同一行摘要和路径。
- 设置页「关于」列出构建类型和所在目录。
- 左侧栏底部不再重复这段信息。
- `./build.sh` 在多个 SwiftPM 输出目录里选择最新的 `NovaMLX` 再同步到 `dist` 和已有的 `/Applications` 副本，避免装上旧目录里的二进制。

Debug versus Release is compiled into the binary, so copying the app does not change it. The clock is the executable’s modification time.

## How to try it

打开正在使用的 NovaMLX，进入 Status。看页面底部一行。菜单栏图标文字应带 Debug 或 Release。再点菜单，第一项是同一行摘要，第二项是 `.app` 路径。

Open NovaMLX, go to Status, and read the footer line. The menu-bar title includes Debug or Release, and the menu’s first two items repeat the summary and the path.

## Limits / out of scope

- 不显示 git 提交号，也不比较两份安装谁更新。
- 不阻止同时运行两份 App。
- 左侧栏不重复这段文字。
- 时间不是“编译开始”的时间戳文件，而是可执行文件的修改时间。

No git SHA, no side-by-side install comparison, and no second copy of the line in the sidebar.

## Suggested video beats

- 同时装了 `/Applications` 和仓库 `dist` 时，窗口本身看不出是哪一份。
- Status 页脚一行：Debug 或 Release、版本、几点几分几秒、`.app` 路径。
- 菜单栏标题直接写着 Debug 或 Release。

## Links

- [macOS Menu Bar App](../features/menu-bar.md)
- [CHANGELOG](../../CHANGELOG.md)
