#!/usr/bin/env bash
# Discover USB camera hardware IDs and install udev rules for stable /dev/fixed_cam
# and /dev/turret_cam symlinks.
set -euo pipefail

RULES_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/systemd/99-cat-cannon-cameras.rules"
RULES_DST="/etc/udev/rules.d/99-cat-cannon-cameras.rules"

echo "=== Connected V4L2 video devices ==="
for dev in /dev/video*; do
  [[ -e "$dev" ]] || continue
  index=$(udevadm info --query=property --name="$dev" 2>/dev/null | grep -oP 'ID_V4L_CAPABILITIES=.*' || true)
  # Only show capture devices (index 0)
  attr_index=$(cat "/sys/class/video4linux/$(basename "$dev")/index" 2>/dev/null || echo "")
  [[ "$attr_index" == "0" ]] || continue
  echo ""
  echo "--- $dev ---"
  udevadm info --query=all --name="$dev" 2>/dev/null | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL|ID_PATH|ATTR{index}' || true
done

echo ""
echo "=== Instructions ==="
echo "1. Edit $RULES_SRC"
echo "2. Replace FIXME_VENDOR/FIXME_PRODUCT with values from above"
echo "   (ID_VENDOR_ID → idVendor, ID_MODEL_ID → idProduct)"
echo "   OR use ID_PATH for port-based matching (uncomment examples in rules file)"
echo "3. Run: sudo cp $RULES_SRC $RULES_DST"
echo "4. Run: sudo udevadm control --reload-rules && sudo udevadm trigger"
echo "5. Verify: ls -la /dev/fixed_cam /dev/turret_cam"
echo ""

if [[ "${1:-}" == "--install" ]]; then
  echo "Installing rules..."
  sudo cp "$RULES_SRC" "$RULES_DST"
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  echo "Done. Checking symlinks..."
  ls -la /dev/fixed_cam /dev/turret_cam 2>/dev/null || echo "Symlinks not yet created — edit the rules file with correct IDs first."
fi
