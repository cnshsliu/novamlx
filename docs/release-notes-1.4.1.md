## NovaMLX v1.4.1

**Requires** macOS 15+ (Sequoia), Apple Silicon. Signed and notarized.

### One MTP switch for native and companion heads

v1.4.0 added an **MTP** control for in-graph Lightning MTP (weights inside the same checkpoint). This release makes that **one switch** also cover **standalone companion MTP** (the extra draft pack Spec Boost auto-loads, e.g. Qwen3.5 / 27B + `*-MTP-4bit`).

#### GUI

**Models → Active models:** the **MTP** switch appears if the loaded model has a native MTP head **or** a companion MTP pack on disk.

- **On** (default) — native MTP decode, and auto-load / auto-inject a companion MTP pack when present (greedy / temp 0)
- **Off** — serial decode; no companion auto-load or auto-inject; a loaded companion MTP is unloaded

Hover the switch for a short explanation. DFlash2 and DSpark stay on the Boost bolt and are not gated by this switch.

#### Request API

Same fields as 1.4.0, now applying to **both** native and companion MTP:

```json
POST /v1/chat/completions
{
  "model": "your-model-id",
  "messages": [{"role": "user", "content": "Hello"}],
  "use_mtp": false
}
```

- `use_mtp: false` — no native MTP and no companion auto-inject for this request
- `use_mtp: true` — force MTP on even if the model switch is off
- omit `use_mtp` — follow the saved model setting (default on)

Persist:

```http
PUT /admin/models/{id}/settings
{"native_mtp": false}
```

`GET /admin/models` reports `nativeMtpAvailable` (native head **or** companion pack) and `nativeMtpEnabled`.

An explicit `draft_model` in the request still wins.

**SHA-256:** *(filled after notarization)*
