#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Flash a Pico-family board in BOOTSEL mode with a UF2 image.

Usage:
  ./scripts/flash_pico_uf2.sh
  ./scripts/flash_pico_uf2.sh --uf2 /path/to/firmware.uf2
  ./scripts/flash_pico_uf2.sh --uf2 /path/to/firmware.uf2 --mount /media/$USER/RPI-RP2

Notes:
  - With no --uf2, this script uses the bundled Pico UF2 in firmware/pico/.
  - This script is only for the BOOTSEL / UF2 flashing step.
  - After the board reboots into MicroPython, run:
      ./scripts/deploy_pico.sh
    to copy main.py and pico_config.py.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLED_UF2="$ROOT_DIR/firmware/pico/RPI_PICO-20260406-v1.28.0.uf2"

UF2_PATH=""
MOUNT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uf2)
      UF2_PATH="${2:-}"
      shift 2
      ;;
    --mount)
      MOUNT_PATH="${2:-}"
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

if [[ -z "$UF2_PATH" ]]; then
  UF2_PATH="$BUNDLED_UF2"
fi

if [[ ! -f "$UF2_PATH" ]]; then
  echo "UF2 file not found: $UF2_PATH" >&2
  exit 1
fi

if [[ -z "$MOUNT_PATH" ]]; then
  for candidate in \
    "/media/$USER/RPI-RP2" \
    "/run/media/$USER/RPI-RP2" \
    "/Volumes/RPI-RP2" \
    "/mnt/RPI-RP2"
  do
    if [[ -d "$candidate" ]]; then
      MOUNT_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "$MOUNT_PATH" ]]; then
  echo "Could not find an RPI-RP2 mount automatically." >&2
  echo "Pass it explicitly with --mount /path/to/RPI-RP2" >&2
  exit 1
fi

if [[ ! -d "$MOUNT_PATH" ]]; then
  echo "Mount path does not exist: $MOUNT_PATH" >&2
  exit 1
fi

echo "[uf2] Copying $(basename "$UF2_PATH") to $MOUNT_PATH"
cp "$UF2_PATH" "$MOUNT_PATH/"
sync
echo "[uf2] Copy complete. Wait for the board to reboot, then run:"
echo "  ./scripts/deploy_pico.sh"
