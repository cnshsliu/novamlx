#!/usr/bin/env python3
"""Regenerate docs/features/supported-models.md from catalog/models.json.

The committed page is the readable catalog. Re-run this script after editing
catalog/models.json; do not hand-edit the generated tables.

    python3 Scripts/generate-supported-models-doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "catalog" / "models.json"
OUT_PATH = ROOT / "docs" / "features" / "supported-models.md"

# Display order. Unknown categories are appended after these.
CATEGORY_ORDER = (
    ("llm", "文本 LLM", "Text LLM"),
    ("vlm", "视觉语言", "Vision-language"),
    ("audio", "音频", "Audio"),
    ("image", "图像", "Image"),
    ("embedding", "嵌入", "Embedding"),
    ("decision", "决策", "Decision"),
)

EXACT_HEADER = (
    "| 名称 | Weight id | 大小 / quant | 状态 | 最低内存 | 下载 | Hugging Face |\n"
    "|------|-----------|--------------|------|----------|------|--------------|"
)

FAMILY_HEADER = (
    "| 名称 | Catalog id | 大小 / quant | 状态 | 最低内存 | 浏览 |\n"
    "|------|------------|--------------|------|----------|------|"
)


def md(text: object) -> str:
    value = "" if text is None else str(text)
    return value.replace("|", "\\|").replace("\n", " ")


def quant_in_size(size: str, quant: str) -> bool:
    folded_size = size.lower().replace("-", "").replace(" ", "")
    folded_quant = quant.lower().replace("-", "").replace(" ", "")
    return bool(folded_quant) and folded_quant in folded_size


def size_quant(entry: dict) -> str:
    size = entry.get("size")
    quant = entry.get("quant")
    size_text = size.strip() if isinstance(size, str) else ""
    quant_text = quant.strip() if isinstance(quant, str) else ""
    if size_text and quant_text and not quant_in_size(size_text, quant_text):
        return f"{size_text} / {quant_text}"
    if size_text:
        return size_text
    if quant_text:
        return quant_text
    return "—"


def min_ram(entry: dict) -> str:
    ram = entry.get("minRamGB")
    if isinstance(ram, int) and ram > 0:
        return f"{ram} GB"
    return "—"


def is_family(entry: dict) -> bool:
    weight_id = entry.get("id") or ""
    url = entry.get("url") or ""
    if weight_id.endswith("*"):
        return True
    return "/models?" in url or "search=" in url


def hf_link(weight_id: str) -> str:
    return f"[Hugging Face](https://huggingface.co/{weight_id})"


def search_stem(weight_id: str) -> str:
    raw = weight_id[:-1] if weight_id.endswith("*") else weight_id
    if "/" in raw:
        raw = raw.split("/", 1)[1]
    return raw.rstrip("-") or weight_id


def category_heading(category: str) -> str:
    for key, zh, en in CATEGORY_ORDER:
        if key == category:
            return f"{zh} / {en}"
    return category


def ordered_categories(entries: list[dict]) -> list[str]:
    present = {entry["category"] for entry in entries}
    ordered = [key for key, _, _ in CATEGORY_ORDER if key in present]
    extras = sorted(present - {key for key, _, _ in CATEGORY_ORDER})
    return ordered + extras


def exact_table(entries: list[dict]) -> str:
    lines = [EXACT_HEADER]
    for entry in entries:
        weight_id = entry["id"]
        lines.append(
            "| {name} | `{weight_id}` | {size} | {status} | {ram} | `nova download {weight_id}` | {link} |".format(
                name=md(entry.get("name") or weight_id),
                weight_id=md(weight_id),
                size=md(size_quant(entry)),
                status=md(entry.get("status") or "verified"),
                ram=md(min_ram(entry)),
                link=hf_link(weight_id),
            )
        )
    return "\n".join(lines)


def family_table(entries: list[dict]) -> str:
    lines = [FAMILY_HEADER]
    for entry in entries:
        weight_id = entry["id"]
        stem = search_stem(weight_id)
        lines.append(
            "| {name} | `{weight_id}` | {size} | {status} | {ram} | `nova search {stem}` |".format(
                name=md(entry.get("name") or weight_id),
                weight_id=md(weight_id),
                size=md(size_quant(entry)),
                status=md(entry.get("status") or "verified"),
                ram=md(min_ram(entry)),
                stem=md(stem),
            )
        )
    return "\n".join(lines)


def unsupported_table(entries: list[dict]) -> str:
    lines = [EXACT_HEADER]
    for entry in entries:
        weight_id = entry["id"]
        lines.append(
            "| {name} | `{weight_id}` | {size} | {status} | {ram} | 不下载 | {link} |".format(
                name=md(entry.get("name") or weight_id),
                weight_id=md(weight_id),
                size=md(size_quant(entry)),
                status=md(entry.get("status") or "unsupported"),
                ram=md(min_ram(entry)),
                link=hf_link(weight_id),
            )
        )
    return "\n".join(lines)


def render(catalog: dict) -> str:
    updated = (catalog.get("updatedAt") or "").split("T", 1)[0] or "unknown"
    models = catalog["models"]
    exact = [m for m in models if not is_family(m) and m.get("status") != "unsupported"]
    families = [m for m in models if is_family(m) and m.get("status") != "unsupported"]
    unsupported = [m for m in models if m.get("status") == "unsupported"]

    parts = [
        "<!-- Generated from catalog/models.json by Scripts/generate-supported-models-doc.py. Do not edit the tables by hand. -->",
        "",
        "# Supported models",
        "",
        "**Pitch:** Download an exact catalog weight id (`org/repo`), the same string as `nova download`.",
        "",
        "「支持」指「精确权重」一节中的 id。该 id 与 `nova download <id>` 相同。名字相近而 id 不同的仓库可能无法加载。",
        "",
        f"本页与应用内目录一致，数据日期 {updated}。`verified` 为已测条目；`preview` 在目录中，尚未标成 verified。`unsupported` 列在文末，NovaMLX 不会下载或加载。",
        "",
        "## 下载 / Download",
        "",
        "```bash",
        "nova download <id>",
        "nova search \"qwen\"",
        "```",
        "",
        "镜像在应用的下载页选择：Official（huggingface.co）、ModelScope，或自定义端点。ModelScope 与 Hugging Face 使用同一个 id。",
        "",
        "Hugging Face 上的可选链接由 id 拼出：`https://huggingface.co/<id>`。下载、取消与续传见 [Hugging Face](huggingface.md)。文件落盘见 [模型管理](model-management.md)。",
        "",
        "## 精确权重 / Exact weights",
        "",
        "下列 id 各对应一个仓库。",
        "",
    ]

    for category in ordered_categories(exact):
        rows = [m for m in exact if m["category"] == category]
        anchor = category
        parts.append(f'<a id="{anchor}"></a>')
        parts.append("")
        parts.append(f"### {category_heading(category)}")
        parts.append("")
        parts.append(exact_table(rows))
        parts.append("")

    parts.extend(
        [
            "## 家族浏览 / Family browse",
            "",
            "这些条目的 id 以 `*` 结尾，或对应 Hugging Face 搜索页，表示一族权重。用 `nova search` 列出具体 `org/repo`，再 `nova download` 那个 id。「精确权重」一节只收录具体 id。",
            "",
            family_table(families),
            "",
            "## 不加载 / Not loaded",
            "",
            "状态为 `unsupported` 的 id 留在目录里，便于辨认。NovaMLX 不下载、不加载。",
            "",
        ]
    )
    if unsupported:
        parts.append(unsupported_table(unsupported))
        parts.append("")
        for entry in unsupported:
            description = (entry.get("description") or "").strip()
            if description:
                parts.append(f"- `{entry['id']}` — {description}")
        parts.append("")
    else:
        parts.append("当前目录没有 `unsupported` 条目。")
        parts.append("")

    parts.extend(
        [
            f"表格由 [`catalog/models.json`](../../catalog/models.json) 生成（`updatedAt` {updated}）。重新生成：`python3 Scripts/generate-supported-models-doc.py`。",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> int:
    catalog = json.loads(CATALOG_PATH.read_text())
    text = render(catalog)
    forbidden = ("促销", "feature-briefs", "feature brief", "video-maker", "modelscope.cn")
    lowered = text.lower()
    for word in forbidden:
        if word.lower() in lowered:
            print(f"refusing to write: generated page contains {word!r}", file=sys.stderr)
            return 1
    OUT_PATH.write_text(text)
    exact = sum(1 for m in catalog["models"] if not is_family(m) and m.get("status") != "unsupported")
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({exact} exact ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
