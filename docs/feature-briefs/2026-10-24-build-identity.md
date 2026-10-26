# The Status page names which build is running

## Title

One Status-page line for Debug / Release, build time, and the app path

## One-line pitch

The Status page footer names the running configuration, the build clock, and the `.app` path, so `/Applications` and a repo copy are not the same mystery process.

## Who it is for / when to care

People who have both `/Applications/NovaMLX.app` and the repo `dist/NovaMLX.app` on the machine, or who have installed both a debug and a release build, and cannot tell which process the open window belongs to.

## What shipped (user-visible behavior)

- A small line at the bottom of the Status page: `Debug · v1.7.0 · yyyy-MM-dd HH:mm:ss · /path/to/NovaMLX.app`. A Release build replaces the first word with `Release`.
- The time is the modification time of the executable that is running, to the second.
- The path is the current process's `.app`. `/Applications/NovaMLX.app` and the repo's `dist/NovaMLX.app` show up as different paths.
- The menu-bar title is `NovaMLX Debug` or `NovaMLX Release`. Open the menu and the first two rows are that same summary line and the path.
- Settings → About lists the build type and the directory.
- The bottom of the sidebar no longer repeats this text.
- `./build.sh` picks the newest `NovaMLX` among several SwiftPM output directories, then syncs it to `dist` and to an existing `/Applications` copy, so an older directory's binary is not what gets installed.

Debug versus Release is compiled into the binary, so copying the app does not change it. The clock is the executable’s modification time.

## How to try it

Open the NovaMLX you are using and go to Status. Read the line at the bottom of the page. The menu-bar icon text should include Debug or Release. Open the menu: the first item is the same summary line, and the second item is the `.app` path.

## Limits / out of scope

- No git commit id, and no comparison of which of two installs is newer.
- Nothing stops two copies of the app from running at once.
- The sidebar does not repeat this text.
- The time is not a "compile started" timestamp file. It is the executable's modification time. There is no git SHA, no side-by-side install comparison, and no second copy of the line in the sidebar.

## Suggested video beats

- With both `/Applications` and the repo `dist` installed, the window itself does not show which copy it is.
- One Status footer line: Debug or Release, version, the clock to the second, and the `.app` path.
- The menu-bar title says Debug or Release directly.

## Links

- [macOS Menu Bar App](../features/menu-bar.md)
- [CHANGELOG](../../CHANGELOG.md)
