#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

CONFIG_PATH="${CAT_CANNON_CONFIG:-configs/app.yaml}"
if [[ "$CONFIG_PATH" == "configs/app.yaml" && ! -f "$ROOT_DIR/$CONFIG_PATH" ]]; then
  CONFIG_PATH="configs/app.example.yaml"
fi

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m cat_cannon.app.run --config "$CONFIG_PATH" "$@"
