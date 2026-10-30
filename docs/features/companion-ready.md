# Admin / health readiness: `loaded` vs `companionReady`

**Pitch:** Stop lying that on-disk speculative companions are chat-loaded.

## Old vs new

| Field | Before | After |
|-------|--------|-------|
| `loaded` (admin `/admin/models`, load/unload responses) | `true` for resident models **and** on-disk DSpark / DFlash / MTP companions (`isModelLoaded` early-returned on draft-on-disk) | `true` only when weights/process are **resident/serving** (engine/worker container or host ASR/TTS/image/decision). Aligns with `/health.loadedModels`. |
| `companionReady` (new on each admin model row) | *(absent)* | `true` when an on-disk speculative draft can be used as `draft_model` / accelerator. **Not** chat-ready — must not be waited on for chat/OCR. |
| `/health.loadedModels` | Already resident-only | Unchanged (truthful). |
| `/health.companionReadyCount` (new) | *(absent)* | Count of downloaded models with `companionReady`. |
| `/v1/models` | Listed ghost companions because it filtered on `isModelLoaded` | Only residents (same predicate as `loaded`). |

## Why

Unload of a companion returned `loaded:false` then flipped back to `loaded:true` while `/health.loadedModels` stayed 0. Gate and UI treated that poison as “loaded chat models,” stormed unload, and raced Flint into `404 model_not_loaded`.

## Client wait rule

After `POST /admin/models/load` for a real chat/OCR model, wait until:

1. that id has admin **`loaded: true`**, and
2. `/health.loadedModels >= 1`.

Never treat **`companionReady` alone** as ready for chat/completions.

## Related

`Scripts/regression-perf-gate.py` waits on the truthful `loaded` + health signal; companion-id filtering remains only as a legacy guard.
