# homebrew-novamlx

Homebrew tap for [NovaMLX](https://github.com/cnshsliu/novamlx) — Apple Silicon MLX inference server.

## Install

```bash
brew tap cnshsliu/novamlx
brew install novamlx
```

## Start as a service

```bash
brew services start novamlx
```

## Run manually

```bash
novamlx serve
```

## Uninstall

```bash
brew services stop novamlx
brew uninstall novamlx
brew untap cnshsliu/novamlx
```

## Requirements

- macOS 15.0 (Sequoia) or later
- Apple Silicon (M1/M2/M3/M4)
