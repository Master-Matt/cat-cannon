#!/usr/bin/env bash
set -euo pipefail

# Cat Cannon kiosk setup (idempotent). Run on the Jetson as the desktop user:
#
#   scripts/setup_kiosk.sh
#
# Configures the machine to come back up unattended and stay on the app:
#   * GDM auto-login so a reboot returns to the desktop without a password.
#   * Autostart the guardian (which runs + self-heals the app) on login.
#   * Suppress GNOME notifications / idle-dim / screen-blank / lock.
#   * Disable apport crash pop-ups.
#   * Passwordless `systemctl reboot` so the guardian can escalate.
#
# Steps that touch /etc use sudo and are each guarded so re-running is safe.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIOSK_USER="${SUDO_USER:-$USER}"
log() { printf '[kiosk] %s\n' "$*"; }

# --- 1. GDM auto-login --------------------------------------------------------
configure_autologin() {
  local conf=""
  for candidate in /etc/gdm3/custom.conf /etc/gdm/custom.conf; do
    [[ -f "$candidate" ]] && conf="$candidate" && break
  done
  if [[ -z "$conf" ]]; then
    log "GDM custom.conf not found; skipping auto-login (not a GDM system?)"
    return
  fi
  log "Enabling GDM auto-login for '$KIOSK_USER' in $conf"
  sudo cp -n "$conf" "${conf}.cat-cannon.bak" || true
  sudo python3 - "$conf" "$KIOSK_USER" <<'PY'
import configparser, sys
path, user = sys.argv[1], sys.argv[2]
cp = configparser.ConfigParser()
cp.optionxform = str
cp.read(path)
if not cp.has_section("daemon"):
    cp.add_section("daemon")
cp["daemon"]["AutomaticLoginEnable"] = "true"
cp["daemon"]["AutomaticLogin"] = user
with open(path, "w") as fh:
    cp.write(fh)
PY
}

# --- 2. Autostart the guardian on login --------------------------------------
configure_autostart() {
  local autostart_dir="$HOME/.config/autostart"
  mkdir -p "$autostart_dir"
  local dest="$autostart_dir/cat-cannon.desktop"
  log "Installing autostart entry -> $dest"
  cat > "$dest" <<EOF
[Desktop Entry]
Name=Cat Cannon
Comment=Cat Cannon turret control system (self-healing guardian)
Exec=$ROOT_DIR/scripts/run_guardian.sh
Path=$ROOT_DIR
Terminal=false
Type=Application
Icon=camera-video
Categories=Utility;
StartupNotify=true
X-GNOME-Autostart-enabled=true
EOF
}

# --- 3. Suppress GNOME popups / idle / lock ----------------------------------
configure_gnome() {
  if ! command -v gsettings >/dev/null 2>&1; then
    log "gsettings not available; skipping GNOME tweaks"
    return
  fi
  log "Disabling notifications, idle-dim, blanking, and screen lock"
  gsettings set org.gnome.desktop.notifications show-banners false || true
  gsettings set org.gnome.desktop.session idle-delay 0 || true
  gsettings set org.gnome.desktop.screensaver lock-enabled false || true
  gsettings set org.gnome.desktop.screensaver idle-activation-enabled false || true
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing' || true
  gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing' || true
  gsettings set org.gnome.settings-daemon.plugins.power idle-dim false || true
}

# --- 4. Disable apport crash pop-ups -----------------------------------------
configure_apport() {
  if [[ -f /etc/default/apport ]]; then
    log "Disabling apport crash pop-ups"
    sudo sed -i 's/^enabled=.*/enabled=0/' /etc/default/apport || true
    sudo systemctl stop apport.service 2>/dev/null || true
  fi
}

# --- 5. Passwordless reboot for the guardian ---------------------------------
configure_sudoers_reboot() {
  local dropin=/etc/sudoers.d/cat-cannon-reboot
  log "Granting passwordless 'systemctl reboot' to '$KIOSK_USER'"
  local reboot_bin systemctl_bin
  systemctl_bin="$(command -v systemctl || echo /usr/bin/systemctl)"
  reboot_bin="$(command -v reboot || echo /sbin/reboot)"
  printf '%s ALL=(root) NOPASSWD: %s reboot, %s\n' \
    "$KIOSK_USER" "$systemctl_bin" "$reboot_bin" | sudo tee "$dropin" >/dev/null
  sudo chmod 0440 "$dropin"
  sudo visudo -cf "$dropin" >/dev/null && log "sudoers drop-in validated"
}

main() {
  configure_autologin
  configure_autostart
  configure_gnome
  configure_apport
  configure_sudoers_reboot
  log "Done. Reboot to verify auto-login + guardian autostart."
}

main "$@"
