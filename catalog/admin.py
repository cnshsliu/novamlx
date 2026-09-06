#!/usr/bin/env python3
"""Local GUI for catalog/models.json. Bind 127.0.0.1 only.

    python3 catalog/admin.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / "models.json"
BUNDLE = ROOT.parent / "Sources" / "NovaMLXUtils" / "Resources" / "catalog" / "models.json"
HTML = ROOT / "admin.html"
HOST = "127.0.0.1"
PORT = int(os.environ.get("NOVAMLX_CATALOG_ADMIN_PORT", "8765"))

REQUIRED = ("id", "url", "name", "category", "family", "format")


def _is_family_id(ident: str) -> bool:
    """Trailing glob after owner/stem. Rejects org-wide `owner/*`."""
    if not ident.endswith("*") or ident.count("*") != 1:
        return False
    prefix = ident[:-1]
    parts = prefix.split("/")
    return len(parts) == 2 and bool(parts[0]) and bool(parts[1])


CATEGORIES = {"llm", "vlm", "embedding", "audio", "image"}
FAMILIES = {
    "llama", "mistral", "phi", "qwen", "gemma", "starcoder", "claude",
    "bailing", "deepseek", "gptOss", "whisper", "qwen3Asr", "qwen3Tts",
    "dotsTts", "stableDiffusion", "flux", "other",
}
FORMATS = {"mlx", "gguf"}
STATUSES = {"verified", "preview"}


def load_catalog() -> dict:
    with CATALOG.open(encoding="utf-8") as f:
        return json.load(f)


def validate(doc: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["Body must be a JSON object"]
    if doc.get("schemaVersion") != 1:
        errors.append("schemaVersion must be 1")
    models = doc.get("models")
    if not isinstance(models, list):
        errors.append("models must be an array")
        return errors
    seen: set[str] = set()
    for i, entry in enumerate(models):
        prefix = f"models[{i}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in REQUIRED:
            val = entry.get(key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{prefix}.{key} is required")
        ident = entry.get("id")
        if isinstance(ident, str):
            if ident in seen:
                errors.append(f"duplicate id: {ident}")
            seen.add(ident)
            if "*" in ident and not _is_family_id(ident):
                errors.append(
                    f"{prefix}.id family pattern must be owner/prefix* "
                    f"with a non-empty repo stem (e.g. mlx-community/Qwen3.8-*)"
                )
            # MTP heads are allowed in the catalog for download; the app still
            # refuses loading them as a chat model.
        if entry.get("category") not in CATEGORIES:
            errors.append(f"{prefix}.category must be one of {sorted(CATEGORIES)}")
        if entry.get("family") not in FAMILIES:
            errors.append(f"{prefix}.family must be one of {sorted(FAMILIES)}")
        if entry.get("format") not in FORMATS:
            errors.append(f"{prefix}.format must be mlx or gguf")
        status = entry.get("status", "verified")
        if status not in STATUSES:
            errors.append(f"{prefix}.status must be preview or verified")
        tags = entry.get("tags")
        if tags is not None and not (
            isinstance(tags, list) and all(isinstance(t, str) for t in tags)
        ):
            errors.append(f"{prefix}.tags must be an array of strings")
        caps = entry.get("capabilities")
        if caps is not None and not (
            isinstance(caps, list) and all(isinstance(t, str) for t in caps)
        ):
            errors.append(f"{prefix}.capabilities must be an array of strings")
        size_bytes = entry.get("sizeBytes")
        if size_bytes is not None and not (
            isinstance(size_bytes, int) and size_bytes >= 0
        ):
            errors.append(f"{prefix}.sizeBytes must be a non-negative integer")
        min_ram = entry.get("minRamGB")
        if min_ram is not None and not (isinstance(min_ram, int) and min_ram >= 0):
            errors.append(f"{prefix}.minRamGB must be a non-negative integer")
        added_at = entry.get("addedAt")
        if added_at is not None and not (
            isinstance(added_at, str) and len(added_at.strip()) >= 10
        ):
            errors.append(f"{prefix}.addedAt must be an ISO-8601 string")
    return errors


def write_catalog(doc: dict) -> None:
    doc = dict(doc)
    doc["schemaVersion"] = 1
    doc["updatedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    errors = validate(doc)
    if errors:
        raise ValueError("\n".join(errors))
    tmp = CATALOG.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(CATALOG)
    # Running NovaMLX reloads ~/.nova/cache/catalog/models.json on search.
    # Copy here so a Save takes effect without waiting for a GitHub push.
    try:
        cache = _nova_catalog_cache()
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CATALOG, cache)
    except OSError as exc:
        sys.stderr.write("catalog cache copy skipped: %s\n" % exc)


def _nova_catalog_cache() -> Path:
    cfg = Path.home() / ".config/novamlx/path"
    if "NOVA_DIR" in os.environ and os.environ["NOVA_DIR"].strip():
        base = Path(os.environ["NOVA_DIR"].strip())
    elif cfg.is_file():
        line = cfg.read_text(encoding="utf-8").splitlines()
        raw = line[0].strip() if line else ""
        base = Path(raw) if raw else Path.home() / ".nova"
    else:
        base = Path.home() / ".nova"
    return base / "cache" / "catalog" / "models.json"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(code, data, "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/admin.html"):
            self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/catalog":
            try:
                self._json(200, load_catalog())
            except Exception as exc:
                self._json(500, {"error": str(exc)})
            return
        if path == "/api/meta":
            self._json(
                200,
                {
                    "catalogPath": str(CATALOG),
                    "bundlePath": str(BUNDLE),
                    "categories": sorted(CATEGORIES),
                    "families": sorted(FAMILIES),
                    "formats": sorted(FORMATS),
                    "statuses": sorted(STATUSES),
                    "capabilities": [
                        "tools",
                        "vision",
                        "thinking",
                        "audio",
                        "imageGeneration",
                    ],
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_PUT(self) -> None:
        if urlparse(self.path).path != "/api/catalog":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            doc = json.loads(raw.decode("utf-8"))
            write_catalog(doc)
            self._json(200, load_catalog())
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/sync-bundle":
            self._json(404, {"error": "not found"})
            return
        try:
            BUNDLE.parent.mkdir(parents=True, exist_ok=True)
            BUNDLE.write_bytes(CATALOG.read_bytes())
            self._json(200, {"ok": True, "bundlePath": str(BUNDLE)})
        except Exception as exc:
            self._json(500, {"error": str(exc)})


def main() -> None:
    if not CATALOG.is_file():
        sys.exit(f"missing {CATALOG}")
    if not HTML.is_file():
        sys.exit(f"missing {HTML}")
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"Catalog admin: {url}")
    print(f"Editing:       {CATALOG}")
    threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
