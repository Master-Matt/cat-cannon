#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

# --- Singleton: only one instance at a time ---
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/cat-cannon-$UID}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/cat-cannon"
mkdir -p "$RUNTIME_DIR" "$STATE_DIR"
PIDFILE="$RUNTIME_DIR/app.pid"
if [[ -f "$PIDFILE" ]]; then
  OLD_PID=$(cat "$PIDFILE" 2>/dev/null)
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Cat Cannon already running (PID $OLD_PID)" >&2
    exit 0
  fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

LOG_FILE="$STATE_DIR/app.log"
exec "$PYTHON_BIN" -m cat_cannon.app.main "$@" >> "$LOG_FILE" 2>&1
