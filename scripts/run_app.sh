#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=scripts/_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

# --- Singleton: only one instance at a time ---
PIDFILE="$ROOT_DIR/.cat-cannon.pid"
if [[ -f "$PIDFILE" ]]; then
  OLD_PID=$(cat "$PIDFILE" 2>/dev/null)
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Cat Cannon already running (PID $OLD_PID)" >&2
    exit 0
  fi
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

LOG_FILE="$ROOT_DIR/cat-cannon.log"
exec "$PYTHON_BIN" -m cat_cannon.app.main "$@" >> "$LOG_FILE" 2>&1
