# 模型行的「更多」菜单

## Title

Local Inference — More menu on downloaded, not-loaded rows

## One-line pitch

On each downloaded model that is not loaded, Load stays put and Convert to TIE, Move to another disk, and Delete sit in More, immediately to its right.

已下载、尚未加载的每一行仍是 Load 在原处；转换为 TIE、移动到另一块硬盘、删除收进右侧的「更多」。

## Who it is for / when to care

给在 Local Inference 里管理已下载模型的人。一行上同时排着转换、移动和删除时，行会变宽；这三项改从 Load 右边的 More 进去。加载、卸载、磁盘过滤、辅助模型开关的行为不变。

For people managing downloaded models on Local Inference. The row stays narrower: those three actions open from More, next to Load. Load, unload, the disk filter, and the companion checkbox behave as before.

## What shipped (user-visible behavior)

Local Inference 的「Downloaded, not loaded / 已下载，尚未加载」列表里，每一行在空闲时仍显示 **Load**（样式与点击后的加载过程与改前相同）。紧挨 Load 右侧是 **More**（简体「更多」，并带其余已有语言的译名）。菜单里按这个顺序有三项：

1. **Convert to TIE**（转换为 TIE）。只有该模型的 TIE 状态是可转换或不完整时可以点，提示文案与改前的行内按钮相同。正在加载或正在转换时这项不可用。转换中的进度文案仍出现在行上。
2. **Move to another disk**（移动到另一块硬盘）。点开后选择另一块已挂载、可写、且不是当前所在根目录的模型盘；取消则文件留在原地。选定后按设置里 Move Models 同一套库间移动执行：文件离开当前盘，登记改指向新位置。没有其他可写盘时会说明需要在设置里添加。模型已加载、或这一行正在加载 / 转换 / 移动时不能移动。移动过程中该行显示「正在移动…」。
3. **Delete**（删除）。仍弹出原来的删除确认（说明会永久删掉磁盘上的文件、不可撤销），菜单项是破坏性样式。确认后的卸载再删除与改前相同。

已加载列表上的 Unload、Play、Remove TIE 不进这个菜单。磁盘菜单和「显示辅助模型」只过滤列表，不随这次改动变化。未挂载磁盘那一节仍是重新下载和删除，没有 More。

On each idle downloaded-not-loaded row, **Load** stays the primary button. **More** is the next control on the right. Inside it, in order: **Convert to TIE** (enabled for convertible or incomplete TIE layouts, same hint as the old row button; disabled while loading or converting), **Move to another disk** (pick another mounted writable model root, then the existing vault move; cancel leaves the files; no other writable disk explains that one must be added in Settings; blocked while loaded, loading, converting, or already moving), and **Delete** (the same confirmation, destructive styling). A move in progress replaces the row actions with a moving label. Loaded rows, the disk filter, the companion checkbox, and the unmounted-disk section are unchanged.

## How to try it

打开菜单栏应用的 **Local Inference**。在「Downloaded, not loaded」里找一行已下载、未加载的模型。

- Load 仍在原处，右侧是 More。
- 打开 More：能转换为 TIE 的模型上，第一项可点，提示与改前相同；不能转换的模型上，这一项是停用的。
- 配置了至少两块可写模型盘时，选 Move to another disk，在列表里点另一块盘（名称旁有剩余空间，与重新下载选盘同一行文案）。取消后文件仍在原盘。选定后该行显示正在移动，完成后模型出现在目标盘上（磁盘过滤若选了原来的盘，这一行会从当前过滤结果里消失）。
- 只配置了一块可写盘时，Move to another disk 会说明没有其他可写盘。
- Delete 仍先确认，确认后模型从列表消失。
- 已加载的模型行仍是 Unload，没有这个 More。标题行的磁盘菜单和「显示辅助模型」仍只改列表。

Open Local Inference. On a downloaded, not-loaded row, Load is where it was and More is on its right. Convert to TIE follows the old enable rule. Move to another disk lists the other writable roots and relocates the files; Delete still confirms before removing them. A loaded row still offers Unload. The disk menu and companion checkbox only filter the lists.

## Limits / out of scope

- 不改变加载、卸载、推测解码开关、目录收录、Remove TIE、磁盘过滤、辅助模型过滤。
- 未挂载磁盘一节没有 More，也不能把未挂载的副本直接移走（盘不在）。
- 移动不会覆盖目标盘上已有的同名副本；空间不够时移动失败并显示原因，源文件保留。
- 这次没有新的模型格式，也没有视频成片。

Load, unload, companion and disk filters, Remove TIE, and the unmounted-disk section are out of scope. A move does not overwrite a copy that already exists on the destination, and it fails without deleting the source when space is short.

## Suggested video beats

- 打开 Local Inference，停在「已下载，尚未加载」的一行上，指出 Load 还在，右边是 More。
- 打开 More，按顺序点出三项：转换为 TIE、移动到另一块硬盘、删除。
- 能转换的模型上点转换，或说明不能转换时这一项是停用的。
- 有第二块盘时，选一块盘完成移动，再切磁盘过滤看到它换了盘。
- 删除仍要先确认。
- 收在已加载行：那里仍是卸载，没有把卸载塞进 More。

## Links

- PR: https://github.com/cnshsliu/novamlx/pull/30
- 模型列表与多目录：[docs/features/model-management.md](../features/model-management.md)
- TIE：[docs/features/tie.md](../features/tie.md)
- 行上的菜单：`Sources/NovaMLXMenuBar/ModelsPageView.swift`
- 可移动到哪些盘：`Sources/NovaMLXCore/ModelRootFilter.swift`
- 产品日志：根目录 `CHANGELOG.md` 的 `[Unreleased]`
