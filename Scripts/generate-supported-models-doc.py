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
    ("llm", "Text LLM"),
    ("vlm", "Vision-language"),
    ("audio", "Audio"),
    ("image", "Image"),
    ("embedding", "Embedding"),
    ("decision", "Decision"),
)

EXACT_HEADER = (
    "| Name | Weight id | Size / quant | Status | Minimum memory | Download | Hugging Face |\n"
    "|------|-----------|--------------|--------|----------------|----------|--------------|"
)

FAMILY_HEADER = (
    "| Name | Catalog id | Size / quant | Status | Minimum memory | Browse |\n"
    "|------|------------|--------------|--------|----------------|--------|"
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
    for key, en in CATEGORY_ORDER:
        if key == category:
            return en
    return category


def ordered_categories(entries: list[dict]) -> list[str]:
    present = {entry["category"] for entry in entries}
    ordered = [key for key, _ in CATEGORY_ORDER if key in present]
    extras = sorted(present - {key for key, _ in CATEGORY_ORDER})
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
            "| {name} | `{weight_id}` | {size} | {status} | {ram} | will not download | {link} |".format(
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
        '"Supported" means an id in [Exact weights](#exact-weights). That id is the same string as `nova download <id>`. A repo with a similar name and a different id may fail to load.',
        "",
        f"This page matches the in-app catalog. Data date {updated}. `verified` entries have been tested; `preview` is in the catalog and is not marked verified. `unsupported` is listed at the end, and NovaMLX will not download or load it.",
        "",
        "## Download",
        "",
        "```bash",
        "nova download <id>",
        "nova search \"qwen\"",
        "```",
        "",
        "Pick a mirror on the app's Downloads page: Official (huggingface.co), ModelScope, or a custom endpoint. ModelScope and Hugging Face use the same id.",
        "",
        "Optional Hugging Face links are built from the id: `https://huggingface.co/<id>`. Download, cancel, and resume are covered in [Hugging Face](huggingface.md). Where files land is covered in [Model management](model-management.md).",
        "",
        '<a id="exact-weights"></a>',
        "",
        "## Exact weights",
        "",
        "Each id below is one repository.",
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
            "## Family browse",
            "",
            "These ids end in `*`, or point at a Hugging Face search page, and stand for a family of weights. Use `nova search` to list a concrete `org/repo`, then `nova download` that id. [Exact weights](#exact-weights) lists only concrete ids.",
            "",
            family_table(families),
            "",
            "## Not loaded",
            "",
            "Ids with status `unsupported` stay in the catalog so they can be recognized. NovaMLX does not download or load them.",
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
        parts.append("The catalog currently has no `unsupported` entries.")
        parts.append("")

    parts.extend(
        [
            f"Tables are generated from [`catalog/models.json`](../../catalog/models.json) (`updatedAt` {updated}). Regenerate with `python3 Scripts/generate-supported-models-doc.py`.",
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
