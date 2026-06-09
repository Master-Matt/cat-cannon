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

# --- 5. Force the display always-on (flaky-EDID panel) -----------------------
# The Jetson's nvidia/Tegra driver intermittently drops the panel a few seconds
# after boot because the panel's DDC/EDID is flaky (the kernel reports the
# output connected but with 0 bytes of EDID). When that happens X keeps running
# headless at the virtual size and the physical screen stays black. We pin the
# output on with a captured EDID so hot-plug detection and DDC are bypassed.
configure_display() {
  local edid_dest=/etc/X11/cat-cannon-edid.bin
  local xorg=/etc/X11/xorg.conf
  local committed="$ROOT_DIR/configs/cat-cannon-edid.bin"
  local edid_src=""

  # Prefer a live capture from the attached panel; fall back to the committed blob.
  if command -v xrandr >/dev/null 2>&1; then
    local tmp_edid="/tmp/cat-cannon-edid.$$.bin"
    if DISPLAY="${DISPLAY:-:0}" python3 - "$tmp_edid" <<'PY' 2>/dev/null
import subprocess, sys
out = subprocess.run(["xrandr", "--verbose"], capture_output=True, text=True).stdout
lines = out.splitlines()
hexs = []
i = 0
while i < len(lines):
    if "EDID:" in lines[i]:
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if len(s) == 32 and all(c in "0123456789abcdefABCDEF" for c in s):
                hexs.append(s); j += 1
            else:
                break
        if hexs:
            break
    i += 1
if not hexs:
    sys.exit(1)
data = bytes.fromhex("".join(hexs))
if data[:8] != bytes([0, 255, 255, 255, 255, 255, 255, 0]):
    sys.exit(1)
with open(sys.argv[1], "wb") as fh:
    fh.write(data)
PY
    then
      edid_src="$tmp_edid"
      log "Captured live panel EDID ($(stat -c%s "$tmp_edid") bytes)"
    fi
  fi
  if [[ -z "$edid_src" ]]; then
    if [[ -f "$committed" ]]; then
      edid_src="$committed"
      log "Using committed EDID blob $committed"
    else
      log "No live EDID and no committed blob; skipping display force"
      return
    fi
  fi

  sudo install -m 0644 "$edid_src" "$edid_dest"
  log "Installed EDID -> $edid_dest"
  [[ "$edid_src" == /tmp/* ]] && rm -f "$edid_src"

  if [[ ! -f "$xorg" ]]; then
    log "$xorg not found; skipping nvidia Device patch (not a Tegra system?)"
    return
  fi
  sudo cp -n "$xorg" "${xorg}.cat-cannon.bak" || true
  sudo python3 - "$xorg" "$edid_dest" <<'PY'
import re, sys
path, edid = sys.argv[1], sys.argv[2]
text = open(path).read()
opts = {
    "ConnectedMonitor": '"DP-1"',
    "CustomEDID": f'"DP-1:{edid}"',
    "ModeValidation": '"DP-1: NoMaxPClkCheck"',
}
m = re.search(r'(Section\s+"Device".*?EndSection)', text, re.S)
if not m:
    sys.exit('no Device section found')
block = m.group(1)
lines = block.splitlines()
# Drop any prior copies of the options we manage so re-running is idempotent.
managed = tuple(opts)
kept = [ln for ln in lines if not any(
    re.match(r'\s*Option\s+"%s"' % k, ln) for k in managed)]
inject = ['    Option      "%s" %s' % (k, v) for k, v in opts.items()]
end_idx = next(i for i, ln in enumerate(kept) if ln.strip() == "EndSection")
new_block = "\n".join(kept[:end_idx] + inject + kept[end_idx:])
open(path, "w").write(text[:m.start(1)] + new_block + text[m.end(1):])
PY
  log "Patched $xorg nvidia Device section (ConnectedMonitor + CustomEDID + ModeValidation)"
}

# --- 6. Passwordless reboot for the guardian ---------------------------------
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
  configure_display
  configure_sudoers_reboot
  log "Done. Reboot to verify auto-login + guardian autostart."
}

main "$@"
