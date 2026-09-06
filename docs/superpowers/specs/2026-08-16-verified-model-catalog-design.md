# Verified Model Catalog — Design Spec

**Date:** 2026-08-16
**Status:** Approved (spec review passed 2026-08-16)
**Owner:** lucasliu

## Context

NovaMLX is a local Apple Silicon inference app. It loads MLX (and a narrow GGUF path) weights from Hugging Face, ModelScope, and other hosts. Authors often tag repos `mlx` or ship GGUF that does not actually load in this engine.

Today the Downloads tab still **searches the public Hub**. A remote `suggested-models.json` (GitHub raw) is only an empty-state hint. The website `/models` page is a separate hardcoded marketing table with filters **All / Trending / Vision / Tool Calling**.

The result: users download models that were never run on NovaMLX, then file “it doesn’t load” as a product bug.

## Goals

1. **Catalog is the only default browse / search / download surface.** An entry is listed only after a human ran it on NovaMLX and inserted it into the file.
2. **Weights still come from the original URL** on that entry. NovaMLX does not rehost models.
3. **One public JSON file** on `novamlx.ai` is the source of truth for the app, the CLI, the admin API, and the marketing page.
4. **Hidden Advanced toggle** (off by default) unlocks arbitrary Hub URLs for operators and fine-tune owners.
5. Local-on-disk models are never blocked.

## Non-Goals

- Admin CMS or web UI for editing the catalog
- Auto-import from Hugging Face `mlx` tags or community submit forms
- Auto-insert from CI
- Rehosting or proxying weight files through novamlx.ai
- Gating load / unload / delete of models already on disk
- A `broken` status in the published file (remove the entry instead)
- Coupling the catalog to tknet / login / billing

## Decision

**Curated catalog + Settings escape hatch (Approach A).**

Default refuse any id not in the catalog. A Settings toggle `allowUnlistedDownloads` restores today’s Hub search and arbitrary download. The same rule applies to GUI, CLI, and admin API.

---

## 1. Policy and surfaces

### Gated unless Advanced is on

- Downloads tab browse and search
- `nova search` / `nova download`
- `GET /admin/api/hf/search`
- `POST /admin/api/hf/download`

Match is against catalog `id` (and pinned `revision` when present). A similarly named unlisted repo is not allowed.

### Never gated

- Models already on disk
- Local folder discovery (`~/.nova/models`, Hub cache)
- Load / unload / delete
- Cluster shard of an already-local model

### Advanced toggle

- Settings → Server, below Models Path
- Label: **Allow unverified downloads**
- Caption: *Search and download any Hugging Face URL. Unverified models may fail to load.*
- Off by default
- Persisted as `allowUnlistedDownloads` on `ServerConfig` / `~/.nova/config.json`
- Takes effect immediately; no restart

Refuse copy (GUI, CLI, admin JSON), Advanced off:

> `<id>` is not in the NovaMLX verified catalog. Turn on **Settings → Allow unverified downloads** if you want to try it anyway.

Admin / CLI status for that refuse: **HTTP 403**.

---

## 2. Catalog file

### Locations

| Role | Path |
|---|---|
| Authoring / source of truth | `~/dev/novamlx-website/static/catalog/models.json` |
| Public URL | `https://novamlx.ai/catalog/models.json` |
| App disk cache | `~/.nova/cache/catalog/models.json` |
| Bundled snapshot | copied into the app at release time |

SvelteKit already serves `static/` at the site root. No auth, no database.

Stop fetching `https://raw.githubusercontent.com/cnshsliu/novamlx/main/suggested-models.json`. Remove `suggested-models.json` from the NovaMLX repo as an authoring copy. One file only, on the website.

### Envelope

```json
{
  "schemaVersion": 1,
  "updatedAt": "2026-08-16T00:00:00Z",
  "models": []
}
```

- Unknown future fields are ignored.
- If `schemaVersion` is newer than the app understands, still decode known fields; do not crash.

### Entry fields

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Hub-style id. Search and `nova download` match this. |
| `url` | yes | Exact download origin (HF, ModelScope, elsewhere). |
| `category` | yes | `ModelType`: `llm`, `vlm`, `embedding`, `audio`, `image` |
| `name` | yes | Display name |
| `family` | yes | Existing `ModelFamily` |
| `format` | yes | `mlx` or `gguf` |
| `description` | no | Card text |
| `revision` | no | Git commit SHA. Omit / `null` = latest. Pin after a re-test. |
| `quant` | no | `4bit`, `8bit`, `fp16`, `mxfp4`, … |
| `size` / `sizeBytes` | no | UI + disk estimate |
| `minRamGB` | no | Soft hint, not a hard block |
| `tags` | no | Card chips |
| `capabilities` | no | `tools`, `vision`, `thinking`, `audio`, `imageGeneration` |
| `testedOn` | no | NovaMLX version that passed the manual test |
| `status` | no | `verified` (default) or `preview`. Only these two are listed. |

Identity: allowlist match is `id` (+ `revision` if set). Download uses `url`. Mirror setting only rewrites the host of that URL.

### Load order in the app

1. Launch: GET `https://novamlx.ai/catalog/models.json` (10s timeout).
2. 200 + valid JSON → replace memory and `~/.nova/cache/catalog/models.json`.
3. Network or parse failure → disk cache, then bundled snapshot.
4. All three empty → browse is empty; unlisted download still refused.

No hardcoded model array in Swift.

---

## 3. In-app behavior

Enforced in the host (ModelManager / admin download handler), not only in SwiftUI.

### Downloads tab, Advanced off

- Empty search: catalog cards, filtered by existing chips **All / LLM / VLM / Embed / Audio / Image**.
- Section title: **Verified models** (not “Suggested”).
- Typed search: local filter of the catalog (id, name, description, tags, family). No Hub / ModelScope request.
- Hide Hub-only toggles (`regex`, `mlx-community/`). Keep **Mirror**.
- `preview` entries show a Preview badge and remain downloadable.
- Unlisted id → refuse. No “Download Anyway.”
- Drop the filename heuristic MLX warning. The catalog (or the Advanced toggle) is the warning.

### Downloads tab, Advanced on

- Search bar is today’s Hub search (mirror, regex, `mlx-community/`).
- Catalog cards remain the empty-state list.
- Unlisted download allowed.
- Banner: *Unverified downloads are on. Models may not load.*

### CLI / admin

| Action | Advanced off | Advanced on |
|---|---|---|
| `nova search <q>` | Filter catalog | Hub search (today) |
| `nova download <approved-id>` | Catalog `url` + pinned `revision` | Same |
| `nova download <unknown-id>` | 403 + refuse copy | Hub download |
| `GET /admin/api/hf/search` | Catalog results, existing JSON shape | Hub |
| `POST /admin/api/hf/download` | 403 if id not in catalog | Hub |

---

## 4. Insert workflow and website page

### Add / update / remove

1. Download the candidate with Advanced on, or copy it into `~/.nova/models`.
2. Load it. Run the existing suite for that kind:
   - LLM / VLM: full API test profile (T1–T14 as applicable)
   - Audio / image / embedding: matching slice of `Scripts/test-all-models.sh`
3. It must load and produce a correct result on current NovaMLX. A Hub card is not enough.
4. Edit `~/dev/novamlx-website/static/catalog/models.json`. Prefer a `revision` SHA. Set `testedOn` to the NovaMLX version just used. `status` is `verified` after a clean pass, `preview` if it works but is not blessed.
5. Deploy the website. The app picks it up on the **next launch** (v1 has no in-session catalog refresh). The public `/models` page fetches the file on each visit.
6. **Remove** by deleting the object. Already-downloaded copies still load.
7. **Upstream change:** pinned `revision` stays frozen until bumped. Unpinned latest may drift — re-test before trusting.

### First publish

Seed from today’s `suggested-models.json` (add `url` + `format`). Treat the seed as **`preview`** unless the suite is re-run. Do not silently mark the old list `verified`.

### Optional later (not v1)

A helper that validates catalog JSON / prints a stub entry.

### `https://novamlx.ai/models`

- Fetch `/catalog/models.json` (same file the app uses).
- Stop using the hardcoded marketing table.
- Clickable subcategory chips: **All / LLM / VLM / Embed / Audio / Image** (replace All / Trending / Vision / Tool Calling).
- Optional secondary chip: Verified / Preview.
- Columns / cards: name, category, size, format, status, link to `url`.
- Drop “50+ families” / “And 40+ more” teaser copy. The page *is* the catalog.
- Fetch failure or empty file: short “catalog unavailable” state, not a stale hardcoded list.

### Release snapshot

When cutting a NovaMLX release, copy the live JSON into the app bundle so offline first-launch still has a list. The website file remains the live source of truth.

---

## Architecture

```
novamlx-website/static/catalog/models.json
        │
        │  deploy
        ▼
https://novamlx.ai/catalog/models.json
        │
        ├── GET (public, no auth) ──► novamlx.ai/models  (same filters as the app)
        │
        └── GET (app launch, 10s)
                │
                ▼
        ModelCatalog (NovaMLXModelManager)
                ├── memory
                ├── ~/.nova/cache/catalog/models.json
                └── bundled snapshot (release copy)
                │
                ▼
        allow(id) == catalog.contains(id)  OR  allowUnlistedDownloads
                │
                ├── DownloadsPageView  (browse / local search / refuse)
                ├── nova search / nova download
                └── /admin/api/hf/search|download
                │
                ▼  (if allowed)
        HuggingFaceService.startDownload(url from catalog, revision if set)
                │  mirrors rewrite host only
                ▼
        original host (HF / ModelScope / …)
```

Gate lives in one place (catalog membership + Advanced flag). UI does not implement a second policy.

---

## Config

New field on `ServerConfig` / persisted config:

```json
"allowUnlistedDownloads": false
```

Existing `huggingfaceEndpoint` is unchanged (mirror / transport only).

---

## Testing

- Decode a valid catalog; ignore unknown fields; survive a future `schemaVersion`.
- `allow(id)` true for listed ids, false for unknown when Advanced is off.
- `allow(id)` true for unknown when Advanced is on.
- Search with Advanced off never calls Hub.
- Download handler returns 403 for unknown id when Advanced is off; starts a task when on.
- Local discovery / load of an unlisted on-disk model still works with Advanced off.
- Catalog fetch failure falls back to disk cache, then bundle.
- Website `/models` chips are All / LLM / VLM / Embed / Audio / Image and filter by `category`.

No live Hub or live novamlx.ai required for unit tests — fixture JSON is enough.

---

## Migration

1. Author `static/catalog/models.json` from current `suggested-models.json` (`preview`, add `url` + `format`).
2. Deploy website so `https://novamlx.ai/catalog/models.json` and `/models` both work.
3. Point the app at that URL; add cache + bundled snapshot; add Settings toggle (default off).
4. Gate search / download in GUI, CLI, admin.
5. Delete repo-root `suggested-models.json` and the GitHub-raw fetch.

Existing downloaded models stay in the local registry and keep loading.

---

## Risks

| Risk | Mitigation |
|---|---|
| Catalog fetch fails (offline, site down) | Disk cache + release-bundled snapshot |
| Unpinned `url` drifts after author push | Prefer `revision`; re-test before bumping |
| Users cannot find a new release for days | `preview` entries; you control insert cadence |
| Advanced left on | Banner on Downloads tab; default remains off |
| Website and app drift | Single file; marketing page must not keep a second table |

---

## Open follow-ups (explicitly out of v1)

- Catalog validation / stub-entry CLI
- Periodic in-app refresh without relaunch
- Pinning revisions for the entire seed list
- GGUF entries beyond what the current Swift loader actually ran
