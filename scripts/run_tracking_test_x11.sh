#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[tracking-x11] DISPLAY is not set. Reconnect with ssh -Y <user>@<jetson> and run this again." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

ZONES_PATH="${CAT_CANNON_ZONES:-configs/zones.yaml}"
if [[ "$ZONES_PATH" == "configs/zones.yaml" && ! -f "$ROOT_DIR/$ZONES_PATH" ]]; then
  ZONES_PATH="configs/zones.example.yaml"
fi

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m cat_cannon.app.tracking_test \
    --fixed-camera "${CAT_CANNON_FIXED_CAMERA:-/dev/fixed_cam}" \
    --turret-camera "${CAT_CANNON_TURRET_CAMERA:-/dev/turret_cam}" \
    --zones "$ZONES_PATH" \
    "$@"
