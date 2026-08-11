# Setup

## Requirements

- Python 3.13+
- Docker running

## Installation

Clone the repo with submodules (benchmarks):

```bash
git clone --recurse-submodules git@github.com:r4wd3r/malo-benchmark.git
cd MaLO
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Run the setup script (installs deps + builds the Docker image):

```bash
./scripts/setup.sh
```

The setup script builds the Docker image from `docker/malo/Dockerfile`.

## Non-interactive Setup

```bash
./scripts/setup.sh --nix
./scripts/setup.sh --poetry
```

## Nix

```bash
nix develop
```

Or use:

```bash
./scripts/setup_nix.sh
```
