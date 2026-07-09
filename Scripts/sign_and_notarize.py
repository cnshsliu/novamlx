#!/usr/bin/env python3
"""
NovaMLX macOS App Signing + Notarization + Auto-Install Automation

Mirrors VoiceVibeCode's sign_and_notarize.py, adapted for NovaMLX.

支持三种模式 + 权限控制：
- 默认：完整流程（build + sign + notarize + staple + install）
- --no-install：仅生成已签名并公证的 DMG
- --only-install：跳过签名公证，直接安装已有公证 DMG
- --no-reset-permissions：安装时不自动清理 TCC 权限（给有特殊需求的用户）
- --github-release：公证完成后自动上传 DMG 到 GitHub Release（cnshsliu/novamlx）
- --github-release --draft：创建为草稿 Release，不公开发布
- --github-release --release-notes "..."：自定义 Release Notes 内容

前置条件（仅首次需要）：
  xcrun notarytool store-credentials novamlx \
    --apple-id your-apple-id@example.com \
    --team-id YOURTEAMID \
    --password app-specific-password

  详见: https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution
"""

import subprocess
import sys
import re
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

BUNDLE_ID = "com.novamlx.app"
APP_NAME = "NovaMLX.app"
KEYCHAIN_PROFILE = "novamlx"
GITHUB_RELEASE_REPO = "cnshsliu/novamlx"

# VoiceVibeCode resets 3 services; NovaMLX only needs Microphone (ASR/TTS/voice
# clone) + Accessibility (TCC watcher automation). ListenEvent (input monitoring)
# isn't used.
TCC_SERVICES_TO_RESET = ["Microphone", "Accessibility"]

# Known-noisy messages from codesign/notarytool that are actually fine.
NORMAL_MESSAGE_NOTES = {
    "does not satisfy its designated requirement": "（此信息正常）",
    "notarization skipped": "（此信息正常，公证由本脚本使用 keychain profile 方式单独执行）",
    "a sealed resource is missing or invalid": "（此信息正常，常见于带用户手册的 DMG）",
}


def run_live(cmd, description, env=None, cwd=None, check=True):
    print(f"\n{'='*72}")
    print(f"▶️  {description}")
    print(f"   $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"   {datetime.now().strftime('%H:%M:%S')}")
    print('='*72)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env or os.environ,
        cwd=cwd or Path.cwd(),
        bufsize=1
    )

    if process.stdout is None:
        print("❌ 无法捕获子进程输出")
        sys.exit(1)
    for line in process.stdout:
        friendly_line = enhance_known_normal_messages(line)
        print(friendly_line, end='', flush=True)

    process.wait()

    if check and process.returncode != 0:
        print(f"\n❌ 该步骤失败（退出码 {process.returncode}）\n")
        sys.exit(process.returncode)

    print(f"\n✅ {description} — 完成\n")
    return process.returncode


def enhance_known_normal_messages(line: str) -> str:
    lower_line = line.lower()
    for keyword, note in NORMAL_MESSAGE_NOTES.items():
        if keyword in lower_line and note not in line:
            return line.rstrip("\n") + f"  {note}\n"
    return line


def detect_developer_id():
    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True, text=True
    )
    matches = re.findall(r'"(Developer ID Application: [^"]+)"', result.stdout)
    if not matches:
        print("❌ 未在钥匙串中找到 Developer ID Application 证书。")
        sys.exit(1)

    if len(matches) > 1:
        print("ℹ️  检测到多个 Developer ID 证书，默认使用第一个。")
    return matches[0]


def find_latest_dmg(dist_dir: Path) -> Path:
    # NovaMLX package.sh produces: NovaMLX-<version>-arm64.dmg
    dmgs = sorted(dist_dir.glob("NovaMLX-*-arm64.dmg"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dmgs:
        print(f"❌ 在 {dist_dir} 目录下未找到 NovaMLX-*-arm64.dmg 文件")
        sys.exit(1)
    return dmgs[0]


def reset_permissions_for_fresh_install():
    """安装到 /Applications 前清理旧权限（除非用户明确禁止）"""
    print("\n" + "="*72)
    print("🧹  正在清理旧的权限记录（因为 App 将安装到 /Applications）")
    print("   目的是避免之前从构建目录运行时授权的残留权限导致的问题。")
    print("="*72)

    for service in TCC_SERVICES_TO_RESET:
        subprocess.run(["tccutil", "reset", service, BUNDLE_ID], capture_output=True)
        print(f"   已重置 {service} 权限")

    print("\n   提示：下次启动应用时会重新弹出授权请求，请点击「允许」。")
    print("✅ 权限记录已清理完成\n")


def kill_running_app():
    print("\n" + "="*72)
    print("🛑  正在关闭当前正在运行的 NovaMLX 实例...")
    print("="*72)

    killed = False
    # Order matters: kill the menu bar host first, then worker subprocesses.
    for pattern in ["NovaMLX.app", "NovaMLXWorker", "NovaMLX"]:
        if subprocess.run(["pkill", "-f", pattern], capture_output=True).returncode == 0:
            killed = True
            print(f"   已关闭匹配进程：{pattern}")

    if not killed:
        print("   未发现正在运行的 NovaMLX 进程。")
    else:
        time.sleep(1.0)
    print("✅ 进程清理完成\n")


def install_dmg_to_applications(dmg_path: Path):
    print("\n" + "="*72)
    print("📥  正在将公证后的应用安装到 /Applications")
    print("="*72)

    mount_point = None
    try:
        attach = subprocess.run(
            ["hdiutil", "attach", "-readonly", str(dmg_path)],
            capture_output=True, text=True
        )
        if attach.returncode != 0:
            print("❌ 挂载 DMG 失败")
            sys.exit(1)

        for line in attach.stdout.splitlines()[::-1]:
            if "/Volumes/" in line:
                mount_point = line.strip().split("\t")[-1].strip()
                break

        if not mount_point or not Path(mount_point).exists():
            print("❌ 无法识别 DMG 挂载点")
            sys.exit(1)

        print(f"   DMG 已挂载至：{mount_point}")

        src_app = Path(mount_point) / APP_NAME
        if not src_app.exists():
            print(f"❌ DMG 内未找到 {APP_NAME}")
            sys.exit(1)

        dest = Path("/Applications") / APP_NAME
        if dest.exists():
            print(f"   检测到旧版本，正在删除：{dest}")
            run_live(["rm", "-rf", str(dest)], "删除旧版 NovaMLX.app")

        print(f"   正在复制新版本到 /Applications ...")
        run_live(["cp", "-R", str(src_app), "/Applications/"], "安装新版 NovaMLX.app")

    finally:
        if mount_point:
            run_live(["hdiutil", "detach", mount_point], "卸载 DMG", check=False)

    print("✅ 应用已成功安装到 /Applications\n")

    # Staple notarization ticket to the installed app.
    # The ticket stapled to the DMG is NOT transferred when copying the .app,
    # and macOS 26 requires the ticket on the app itself for TCC to work.
    installed_app = Path("/Applications") / APP_NAME
    print("\n" + "="*72)
    print("📋  正在为已安装的应用打上公证票据（staple）")
    print("="*72)
    run_live(
        ["xcrun", "stapler", "staple", str(installed_app)],
        "为 /Applications 中的应用打上公证票据"
    )
    print("✅ 公证票据已打上\n")


def launch_app():
    app_path = f"/Applications/{APP_NAME}"
    run_live(["open", app_path], "启动新安装的 NovaMLX")


def check_timestamp_service():
    """Verify Apple's code-signing timestamp service is reachable.

    codesign requires access to timestamp.apple.com to embed a trusted
    timestamp in the signature.  If the service is unreachable (e.g. blocked
    by a proxy/VPN), codesign will fail with "The timestamp service is not
    available." — wasting a long build.  This pre-flight check fails fast.
    """
    import socket
    import urllib.request
    import urllib.error

    host = "timestamp.apple.com"
    port = 80
    timeout = 8

    print(f"\n{'='*72}")
    print("🔍  预检：Apple 代码签名时间戳服务 (timestamp.apple.com)")
    print(f"{'='*72}")

    # Step 1: TCP connectivity test
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
    except (socket.timeout, socket.error, OSError) as e:
        print(f"   ❌ 无法建立 TCP 连接到 {host}:{port}")
        print(f"   原因: {e}")
        _print_timestamp_troubleshooting()
        sys.exit(1)

    # Step 2: HTTP request — Apple's TSA returns empty reply for plain GET,
    # which is fine; we just need to confirm it responds at the HTTP level.
    try:
        req = urllib.request.Request(f"http://{host}/ts01", method="GET")
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            print(f"   ✅ {host} 可达 (HTTP {resp.status})")
        except urllib.error.URLError as e:
            reason = str(e.reason) if hasattr(e, "reason") else str(e)
            if any(kw in reason.lower() for kw in ("empty reply", "connection reset", "reset by peer", "eof occurred")):
                print(f"   ✅ {host} 可达 (TSA 正常拒绝 GET 请求)")
            else:
                print(f"   ❌ {host} 返回异常: {reason}")
                _print_timestamp_troubleshooting()
                sys.exit(1)
        except Exception as e:
            print(f"   ❌ 请求 {host} 时出错: {e}")
            _print_timestamp_troubleshooting()
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ 请求 {host} 时出错: {e}")
        _print_timestamp_troubleshooting()
        sys.exit(1)

    print("✅ 时间戳服务检查通过\n")
    return True


def _print_timestamp_troubleshooting():
    print()
    print("   Apple 代码签名必须访问此服务，否则签名将失败。")
    print("   可能原因：")
    print("     • 代理/VPN 拦截或劫持了该域名")
    print("     • 网络连接异常")
    print("   建议：")
    print("     • 检查代理规则，将 timestamp.apple.com 设为直连")
    print("     • 或暂时关闭代理后重试")
    print(f"\n❌ 预检失败，终止构建。\n")


def get_version_from_dmg(dmg_path: Path) -> str:
    """Extract version string from DMG filename like NovaMLX-1.0.8-arm64.dmg.

    Exits the process on failure — callers assume a non-None return."""
    match = re.match(r"NovaMLX-(.+)-arm64\.dmg", dmg_path.name)
    if not match:
        print(f"⚠️  无法从 DMG 文件名提取版本号: {dmg_path.name}")
        print("   DMG 文件名应符合格式：NovaMLX-X.Y.Z-arm64.dmg")
        sys.exit(1)
    return match.group(1)


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def create_github_release(dmg_path: Path, draft: bool = False, release_notes: str | None = None):
    """Create a GitHub Release and upload the DMG asset to cnshsliu/novamlx."""
    version = get_version_from_dmg(dmg_path)
    tag = f"v{version}"
    title = f"v{version}"
    body = release_notes or f"""## NovaMLX v{version}

**System Requirements:** macOS 15.0+ (Sequoia), Apple Silicon
**Download:** `{dmg_path.name}` — signed + notarized

SHA-256: `{_sha256(dmg_path)}`
"""

    print(f"\n{'='*72}")
    print(f"🐙  GitHub Release: {GITHUB_RELEASE_REPO}")
    print(f"   Tag: {tag}")
    print(f"   Title: {title}")
    print(f"   Asset: {dmg_path.name}")
    print(f"   Draft: {draft}")
    print(f"{'='*72}")

    # Check if tag already exists — re-create if so.
    check_result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", GITHUB_RELEASE_REPO],
        capture_output=True, text=True
    )

    if check_result.returncode == 0:
        print(f"\n⚠️  Release {tag} 已存在于 {GITHUB_RELEASE_REPO}")
        print("   正在删除旧 Release 以便重新创建...")

        run_live(
            ["gh", "release", "delete", tag, "--repo", GITHUB_RELEASE_REPO, "--yes"],
            f"删除旧 Release {tag}",
            check=True
        )
        run_live(
            ["git", "tag", "-d", tag],
            f"删除本地 tag {tag}",
            check=False
        )
        run_live(
            ["git", "push", "origin", "--delete", tag],
            f"删除远程 tag {tag}",
            check=False
        )

    # Write release notes to a temp file to avoid shell quoting issues.
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, prefix='release_notes_') as f:
        f.write(body)
        notes_file = f.name

    try:
        cmd = [
            "gh", "release", "create", tag,
            str(dmg_path),
            "--repo", GITHUB_RELEASE_REPO,
            "--title", title,
            "--notes-file", notes_file,
        ]
        if draft:
            cmd.append("--draft")

        run_live(cmd, f"创建 GitHub Release {tag} 并上传 {dmg_path.name}")
    finally:
        try:
            os.unlink(notes_file)
        except OSError:
            pass

    print(f"\n{'='*72}")
    print(f"✅  GitHub Release 创建成功！")
    print(f"   仓库: {GITHUB_RELEASE_REPO}")
    print(f"   Tag: {tag}")
    print(f"   Asset: {dmg_path.name}")
    print(f"   {'（草稿）' if draft else '（已发布）'}")
    release_url = f"https://github.com/{GITHUB_RELEASE_REPO}/releases/tag/{tag}"
    print(f"   链接: {release_url}")
    print(f"{'='*72}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="NovaMLX macOS 发布自动化工具"
    )
    parser.add_argument("--no-install", action="store_true", help="仅生成已签名并公证的 DMG")
    parser.add_argument("--only-install", action="store_true", help="跳过签名和公证，直接安装已有公证 DMG")
    parser.add_argument("--no-reset-permissions", action="store_true",
                        help="安装到 /Applications 时不自动清理旧的 TCC 权限（保留用户已有授权）")
    parser.add_argument("--github-release", action="store_true",
                        help=f"公证完成后自动上传 DMG 到 GitHub Release（{GITHUB_RELEASE_REPO}）")
    parser.add_argument("--draft", action="store_true",
                        help="与 --github-release 配合使用，创建草稿 Release 而非直接发布")
    parser.add_argument("--release-notes", type=str, default=None,
                        help="自定义 GitHub Release Notes 内容（Markdown 格式）")
    parser.add_argument("--release-notes-file", type=str, default=None,
                        help="从文件读取 GitHub Release Notes 内容（Markdown 格式）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.only_install:
        print("NovaMLX macOS -- install-only mode\n")
        project_root = Path.cwd()
        dist_dir = project_root / "dist"
        dmg_path = find_latest_dmg(dist_dir)
        print(f"Using latest DMG: {dmg_path.name}\n")

        kill_running_app()
        if not args.no_reset_permissions:
            reset_permissions_for_fresh_install()
        install_dmg_to_applications(dmg_path)

        print("\n" + "="*72)
        print("Install complete. Permissions cleared.")
        print(f"App location: /Applications/{APP_NAME}")
        print("Please launch manually to simulate fresh user experience.")
        print("="*72 + "\n")
        return

    # Default / --no-install mode
    mode = "DMG only (no install)" if args.no_install else "Full (build + sign + notarize + clear perms + install)"
    print(f"NovaMLX macOS release automation  [{mode}]\n")

    kill_running_app()

    project_root = Path.cwd()
    dist_dir = project_root / "dist"

    # Pre-flight: ensure Apple timestamp service is reachable.
    check_timestamp_service()

    dev_id = detect_developer_id()
    print(f"Certificate: {dev_id}\n")

    # Scripts/package.sh reads DEVELOPER_ID from env and signs accordingly.
    package_script = project_root / "Scripts" / "package.sh"
    if not package_script.exists():
        print(f"❌ 未找到 {package_script}")
        sys.exit(1)

    env = os.environ.copy()
    env["DEVELOPER_ID"] = dev_id
    run_live(
        ["./Scripts/package.sh"],
        "Running Scripts/package.sh (build + sign + DMG)",
        env=env,
        cwd=project_root
    )

    dmg_path = find_latest_dmg(dist_dir)
    print(f"Latest DMG: {dmg_path.name}\n")

    run_live(
        ["xcrun", "notarytool", "submit", str(dmg_path),
         "--keychain-profile", KEYCHAIN_PROFILE, "--wait"],
        f"Notarizing {dmg_path.name}"
    )

    run_live(
        ["xcrun", "stapler", "staple", str(dmg_path)],
        "Stapling notarization ticket to DMG"
    )

    print("\n" + "="*72)
    print("Notarization complete!")
    print(f"DMG: {dmg_path}")
    print("="*72 + "\n")

    # GitHub Release upload (after notarization + staple, before install).
    if args.github_release:
        notes = args.release_notes
        if not notes and args.release_notes_file:
            notes_file_path = Path(args.release_notes_file)
            if notes_file_path.exists():
                notes = notes_file_path.read_text(encoding="utf-8")
            else:
                print(f"⚠️  Release Notes 文件不存在: {notes_file_path}")
        create_github_release(dmg_path, draft=args.draft, release_notes=notes)

    if args.no_install:
        msg = "Done (--no-install)."
        if args.github_release:
            msg += " GitHub Release uploaded."
        msg += " DMG ready for distribution."
        print(msg)
        print(f"Location: {dmg_path}\n")
        return

    kill_running_app()
    if not args.no_reset_permissions:
        reset_permissions_for_fresh_install()
    install_dmg_to_applications(dmg_path)

    print("\n" + "="*72)
    print("All done!")
    print("Permissions cleared. Please launch manually to simulate fresh user experience.")
    print(f"App location: /Applications/{APP_NAME}")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()
