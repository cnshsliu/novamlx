# macOS Menu Bar App

**Pitch:** A native SwiftUI menu bar app: status icon, dropdown, and a window for models, chat, keys, and cluster.

A native SwiftUI menu bar app. A status icon, a dropdown menu, and the feature pages in the popover window.

## Who it is for

People who use the menu bar to watch throughput, download models, try a chat, and manage keys and the cluster.

## What it does

### Pages

| Page | Contents |
|------|----------|
| **Status** | Live TPS curve (90 samples, peak, zero-value clipping), CPU / memory / GPU, device info, peak TPS. One footer line: Debug or Release, version, build time (to the second), `.app` path |
| **Dashboard** | Loaded models, active requests, memory, uptime, quick load |
| **Local Inference** | Activated models: Pin, unload, copy name, Play into Playground. While Pin is on, the Unload button is hidden, and a restart still loads the model. Downloaded, not loaded: type tabs (All / LLM / VLM / Embed / Audio / Image), filter by disk, Show companions (DSpark / DFlash / MTP hidden by default). Model cards |
| **Downloads** | Categories, recommended model cards, staged progress, stall detection, mirror configuration |
| **Playground (Chat)** | One entry for LLM / ASR / TTS / Image; the mode follows the model type. The picker distinguishes a local direct connection from Tokenhub HTTP. Parameter sliders, Disable-Thinking, sticky autoscroll, copy cURL (OpenAI / Anthropic / Responses). ASR microphone, TTS speaker, voice clone |
| **Tokenhub** | A catalog of 20+ providers, with CRUD. Each provider's models can be copied and played. Endpoint test |
| **Load Balancers** | Accordion rows, local and remote members, strategy and stats, Play on each row |
| **API Keys** | Full-row accordion, show plaintext, copy, usage bar, rate limit, allowlist |
| **Cluster** | Thunderbolt and ARP scan, worker health, model readiness per node |
| **Settings** | HF endpoint and mirror, memory mode, log level, cluster switch, TurboQuant, tknet.ai account |
| **Audio** | A separate ASR / TTS page (apart from Playground) |
| **Agents** | Install, start, and configure OpenClaw, Hermes, and OpenCode |

### Across pages

| Feature | Details |
|---------|---------|
| **Pick-to-Playground** | `play.circle` on Active Models, Tokenhub models, and Load Balancers: opens Playground with that model preselected |
| **Copy cURL** | Playground parameter area: OpenAI / Anthropic / Responses. The output uses a `${NOVA_API_KEY}` placeholder; the secret does not go to the clipboard |
| **Sticky scroll** | `MessageListBottomOffsetKey`, threshold 80 pt |
| **Model picker** | `inferModeFromName` and `autoDetectMode` switch among LLM / ASR / TTS / Image |
| **HF Endpoint** | Switching mirrors does not require a restart. It takes effect through `NovaMLXConfiguration.shared` |

UI copy covers 9 languages. See [Internationalization](internationalization.md).

## How to try

Launch NovaMLX. A status icon appears in the menu bar. The dropdown shows service status. The popover opens the pages in the table above.

When you copy cURL from Playground, the key slot in the command is `${NOVA_API_KEY}`, not the current plaintext key.

Play on a model in Downloads or Local Inference opens Playground with that model selected. Load Balancer rows and Tokenhub model rows have Play as well.

## Limits / notes

- This page is the menu-bar information architecture, not a manual for every control.
- Copy cURL deliberately keeps the secret off the clipboard.
- The root README says the service listens on `localhost:8080` after launch. The feature catalog's default public port is `6590`. Follow the machine's own configuration. See [Configuration](configuration.md).

## Related

- [Model management](model-management.md)
- [Tokenhub](tokenhub.md)
- [Load balancers](load-balancers.md)
- [API Key](api-keys.md)
- [Agent](agents.md)
- [Internationalization](internationalization.md)
