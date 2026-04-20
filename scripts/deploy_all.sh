#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/scripts/deploy_pico.sh"
"$ROOT_DIR/scripts/deploy_jetson.sh"

echo "[deploy] Jetson and Pico deployment steps completed"

