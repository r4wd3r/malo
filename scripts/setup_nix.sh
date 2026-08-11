#!/usr/bin/env bash
set -euo pipefail

if ! command -v nix >/dev/null 2>&1; then
  echo "Nix is required but was not found. Install Nix and try again."
  exit 1
fi

nix develop
