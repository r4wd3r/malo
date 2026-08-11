#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: scripts/setup.sh [--nix|--poetry]

Options:
  --nix     Use Nix to set up the environment (non-interactive).
  --poetry  Use Poetry to set up the environment (non-interactive).
  -h, --help  Show this help message.
EOF
}

install_choice=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nix)
      if [[ -n "$install_choice" && "$install_choice" != "nix" ]]; then
        echo "Error: --nix and --poetry are mutually exclusive."
        exit 1
      fi
      install_choice="nix"
      ;;
    --poetry)
      if [[ -n "$install_choice" && "$install_choice" != "poetry" ]]; then
        echo "Error: --nix and --poetry are mutually exclusive."
        exit 1
      fi
      install_choice="poetry"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ -z "$install_choice" ]]; then
  while [[ -z "$install_choice" ]]; do
    echo "Select install method:"
    echo "1) nix"
    echo "2) poetry"
    read -r -p "Enter choice [1-2]: " choice
    case "${choice,,}" in
      1|nix)
        install_choice="nix"
        ;;
      2|poetry)
        install_choice="poetry"
        ;;
      *)
        echo "Invalid choice. Please enter 1 or 2."
        ;;
    esac
  done
fi

if [[ "$install_choice" == "nix" ]]; then
  if ! command -v nix >/dev/null 2>&1; then
    echo "Nix was selected but not found. Install Nix, or rerun and choose Poetry."
    exit 1
  fi
  echo "Use: nix develop"
  exit 0
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Python is required but was not found. Install Python 3.13+ and try again."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found. Install Docker and try again."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Start Docker and try again."
  exit 1
fi

if ! command -v poetry >/dev/null 2>&1; then
  echo "Poetry not found; installing..."
  curl -sSL https://install.python-poetry.org | python -
  export PATH="$HOME/.local/bin:$PATH"
fi

poetry install

python scripts/build_docker.py
