# tknet-peer

Local gateway for a token factory. One process serves an OpenAI-compatible port and, when registered, a tknet tunnel. Behind the gateway are **pools**. A pool name is the model id clients send. Each pool picks a member. Members are HTTP endpoints (NovaMLX, Ollama, vLLM, or any OpenAI-compatible server). tknet only sees the pool id (`sourceId` on the wire). It does not see member addresses.

This directory is a Swift package. It does not need a NovaMLX checkout.

The Mac menu-bar app (`TknetPeer.app`) stays in [novamlx](https://github.com/cnshsliu/novamlx) and links this package. Windows and Linux ship the CLI. There is no Windows or Linux GUI in this tree.

## Build with no NovaMLX source

Requirements: Swift 6.0 or newer.

```bash
git clone https://github.com/cnshsliu/tknet-peer.git
cd tknet-peer
swift build -c release --product tknet-peer
swift test
```

Until that GitHub repository exists, use the copy inside novamlx:

```bash
# from a novamlx checkout — only the tknet-peer/ directory is compiled
swift build -c release --package-path tknet-peer --product tknet-peer
# or
./build.sh peer -c release
```

The binary is under `tknet-peer/.build/<triple>/release/tknet-peer`.

Confirm the dependency graph has no GPU stack and no Nova database library:

```bash
bash Scripts/check-mlx-free.sh
```

Package a release tarball:

```bash
bash Scripts/package-cli.sh
# dist/tknet-peer-<os>-<arch>.tar.gz
```

## Route a pool to a LAN inference service

Point a member at any machine that speaks `POST /v1/chat/completions`. The gateway listens on `127.0.0.1:6600` unless you pass `--listen-port`.

```bash
tknet-peer member add \
  --id lan-box \
  --endpoint http://192.168.1.20:8000/v1 \
  --upstream-model Qwen3-8B \
  --api-key sk-local
tknet-peer pool add --name coding --member lan-box
tknet-peer serve --no-tunnel
```

```bash
curl http://127.0.0.1:6600/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"coding","messages":[{"role":"user","content":"ping"}]}'
```

`model` is the **pool name** (`coding`). The member receives `model: Qwen3-8B`. One published name is one pool.

Automated check (fake server on this host's LAN address, or loopback if that address does not answer):

```bash
swift build -c release --product tknet-peer
bash Scripts/accept-lan-pool.sh
```

Copy the package to an empty directory and run build, tests, the dependency check, and the LAN check:

```bash
bash Scripts/verify-standalone.sh
```

## Publish this tree as its own repository

From the novamlx repository, with `gh` logged in:

```bash
bash tknet-peer/Scripts/publish-repo.sh git@github.com:cnshsliu/tknet-peer.git
```

That runs `git subtree split --prefix=tknet-peer`. If `github.com/cnshsliu/tknet-peer` does not exist, `gh repo create` makes a public repository with no README, then the script pushes the split history to `main`. NovaMLX depends on that repository with `from: "0.2.0"`. `./build.sh peer` still compiles the copy in this checkout.

## Layout

| Path | Role |
|---|---|
| `Sources/TknetPeer` | Library. Swift module `TknetPeer` (was `NovaMLXTknetPeer`). |
| `Sources/tknet-peer` | `tknet-peer` CLI. |
| `Tests/TknetPeerTests` | Swift Testing suite. |
| `Scripts/` | Package, acceptance, publish. |

Config stays in `~/.config/tknet-peer/` (`peer.json`, `peer.lock`). Only one process may hold the lock.

Secrets:

1. The CLI reads and writes `~/.config/tknet-peer/secrets/secrets.json` (mode `0600`) through `FileSecretStore`. `tknet-peer serve` always uses that file.
2. The Mac app (`TknetPeer.app` in novamlx) uses the login Keychain, service `ai.tknet.peer`, as its primary store.
3. On launch the app copies the file into the Keychain when the file changed (a CLI write while the app was quit). It does not delete the file.
4. Writes from the app go to the Keychain and the file, so the CLI still sees the connection key and member keys.

Build the Mac app from a novamlx checkout (needs macOS):

```bash
swift build -c release --product TknetPeerApp
swift test --package-path tknet-peer --filter SecretStoreMigrationTests
```

`tknet-peer migrate-from-novamlx` reads a copied `nova_config.db` via the `sqlite3` CLI. It does not link NovaMLX.
