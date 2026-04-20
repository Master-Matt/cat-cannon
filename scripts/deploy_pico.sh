#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRMWARE_DIR="$ROOT_DIR/firmware/pico"
PORT="${PICO_PORT:-}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

usage() {
  cat <<'EOF'
Deploy Pico application files over an explicit serial connection.

Usage:
  ./scripts/deploy_pico.sh
  ./scripts/deploy_pico.sh --port /dev/ttyACM0

Notes:
  - If --port is omitted, the script tries to autodetect a single Pico-compatible serial device.
  - You can also set PICO_PORT=/dev/ttyACM0.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m cat_cannon.app.deploy_pico ${PORT:+--port "$PORT"}
