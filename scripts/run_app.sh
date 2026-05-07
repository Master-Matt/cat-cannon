#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

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

# Ensure DISPLAY is set for X11 — detect from the running GNOME session
if [[ -z "${DISPLAY:-}" ]]; then
  _gnome_pid=$(pgrep -u "$(id -u)" gnome-shell 2>/dev/null | head -1)
  if [[ -n "${_gnome_pid:-}" ]]; then
    export DISPLAY=$(tr '\0' '\n' < /proc/"$_gnome_pid"/environ 2>/dev/null | grep '^DISPLAY=' | cut -d= -f2)
    export XAUTHORITY=$(tr '\0' '\n' < /proc/"$_gnome_pid"/environ 2>/dev/null | grep '^XAUTHORITY=' | cut -d= -f2)
  fi
  export DISPLAY="${DISPLAY:-:1}"
fi

# Ensure XAUTHORITY is set (needed for X11 access from .desktop launchers)
if [[ -z "${XAUTHORITY:-}" ]]; then
  if [[ -f /run/user/$(id -u)/gdm/Xauthority ]]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  else
    export XAUTHORITY="$HOME/.Xauthority"
  fi
fi

# Use system Qt platform plugins instead of OpenCV's bundled ones
export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms

# Disable GNOME accessibility bridge (at-spi2-registryd spins CPU with Qt/OpenCV windows)
export NO_AT_BRIDGE=1
export QT_ACCESSIBILITY=0

# Prevent PyTorch/OpenBLAS from spawning threads/workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_FILE="$ROOT_DIR/cat-cannon.log"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m cat_cannon.app.main "$@" >> "$LOG_FILE" 2>&1
