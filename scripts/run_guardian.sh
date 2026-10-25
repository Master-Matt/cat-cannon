#!/usr/bin/env bash
set -euo pipefail

# Cat Cannon guardian launcher.
# Runs the app under the self-healing guardian, which restarts it on a missed
# heartbeat / crash and escalates to a system reboot if restarts keep recurring.
# This is the Exec target for the autostart .desktop entry.

# shellcheck source=scripts/_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"

# --- Singleton: only one guardian at a time ---
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/cat-cannon-$UID}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/cat-cannon"
mkdir -p "$RUNTIME_DIR" "$STATE_DIR"
PIDFILE="$RUNTIME_DIR/guardian.pid"
if [[ -f "$PIDFILE" ]]; then
  OLD_PID=$(cat "$PIDFILE" 2>/dev/null)
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null \
     && tr '\0' ' ' < "/proc/$OLD_PID/cmdline" 2>/dev/null \
        | grep -q "cat_cannon.app.guardian"; then
    echo "Cat Cannon guardian already running (PID $OLD_PID)" >&2
    exit 0
  fi
  # Stale or reused PID — clear it and continue.
  rm -f "$PIDFILE"
fi
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

LOG_FILE="$STATE_DIR/guardian.log"
# Unattended/production launch: auto-arm on start so the turret is live after a
# boot or a guardian self-heal restart. (Manual scripts/run_app.sh stays unarmed
# for dev safety.) Override by setting CAT_CANNON_ARM=0.
ARM_ARG="--arm"
if [[ "${CAT_CANNON_ARM:-1}" == "0" ]]; then
  ARM_ARG=""
fi
exec "$PYTHON_BIN" -m cat_cannon.app.guardian \
  --config "${CAT_CANNON_CONFIG:-configs/app.yaml}" \
  -- "$PYTHON_BIN" -m cat_cannon.app.main $ARM_ARG "$@" >> "$LOG_FILE" 2>&1
