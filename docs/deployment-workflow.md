# Deployment Workflow

This project is deployed as two coordinated artifacts:

- Jetson application package
- Pico firmware package

## Strategy

Use `USB serial` for runtime control and Pico deployment so the same cable can be used for both
control and updates.

## Pico Update Flow

1. Put the Pico into `BOOTSEL` mode if MicroPython is not already installed.
2. Flash the bundled UF2:

```bash
./scripts/flash_pico_uf2.sh
```

3. Plug the Pico into the Jetson or a development host if needed and run:

```bash
./scripts/deploy_pico.sh
```

4. The script copies `main.py` and `pico_config.py`, then resets the board.
5. Run a post-deploy handshake from the Jetson host before arming the system.

When the Pico is physically attached to the Jetson, run this step on the Jetson. The current
Jetson bring-up path uses `/dev/ttyACM0`; if serial permissions are still locked down, use `sudo`
for the deploy command or install a udev rule/dialout-group fix before live operation.

## Jetson Update Flow

1. Connect the host to the Jetson over USB OTG or LAN.
2. Run:

```bash
# Via USB OTG (fixed IP):
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --host 192.168.55.1 --user mdev

# Via LAN (find the Jetson IP on your network):
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --host <JETSON_LAN_IP> --user mdev
```

3. If you want the bundled systemd unit installed or restarted:

```bash
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh \
  --host <JETSON_IP> \
  --user mdev \
  --remote-dir /opt/cat-cannon \
  --install-service \
  --restart-service
```

Notes:

- the default remote directory is `/home/mdev/cat_cannon`
- `sshpass` is optional; if it is missing the script falls back to normal SSH password prompts
- the script syncs the repo with `rsync`, creates a remote `.venv`, and installs the selected extras
- `configs/app.yaml`, `configs/zones.yaml`, model engines, and `data/` are preserved on the Jetson
- if the Jetson has no working APT/DNS, add `--skip-system-packages` to avoid all package-manager steps

## Coordinated Update Flow

Recommended order:

1. Deploy Pico firmware
2. Verify `ping` and `status`
3. Deploy Jetson application
4. Restart Jetson supervisor
5. Run dry-run or replay validation before enabling live actuation

## Fixed Camera Bring-Up

Start with a dry run:

```bash
./scripts/run_fixed_camera.sh --camera 0 --show-window
```

Move to the live RP2040 controller only after the camera, detections, and zone geometry look right:

```bash
./scripts/run_fixed_camera.sh \
  --camera 0 \
  --port /dev/ttyACM0 \
  --live-controller \
  --show-window
```

Calibrate the counter polygons before arming the system:

```bash
./scripts/run_zone_calibrator.sh --camera 0 --output configs/zones.yaml --fullscreen
```

## Tracking Test over SSH X Forwarding

The tracking test UI is intended for bring-up with both Jetson cameras connected. It shows the
fixed camera with YOLO/zone/supervisor overlays, the turret camera for visual confirmation, and
teleop buttons for arm, safe stop, pan/tilt, and fire.

From the laptop:

```bash
# 192.168.55.1 = USB OTG; use LAN IP if connected via network
ssh -Y mdev@192.168.55.1
cd ~/cat_cannon
./scripts/run_tracking_test_x11.sh --live-controller --port /dev/ttyACM0
```

Useful defaults can be overridden without retyping the full command:

```bash
CAT_CANNON_FIXED_CAMERA=/dev/video0 CAT_CANNON_TURRET_CAMERA=/dev/video2 ./scripts/run_tracking_test_x11.sh
```

The tracking screen can switch to zone calibration with the `Zone Calibrator` button or `z`. The
zone calibration screen can switch back with the `Tracking Test` button or `t`.

Do not use `sudo` for the X11 UI unless you have copied the X authority cookie for root. Running
the UI with plain SSH X forwarding plus normal serial permissions avoids the common
`X11 connection rejected because of wrong authentication` failure.

## Dataset Capture and Replay Validation

For live data collection, run the main app or tracking screen with dataset capture enabled:

```bash
./scripts/run_app.sh \
  --collect-cat-dataset \
  --dataset-dir data/cat_training \
  --dataset-sample-hz 1.0
```

The recorder writes raw camera frames and YOLO labels for confident cat and person detections from
both cameras. Labels use class `0` for cats and class `1` for people. Keep this data local to the
Jetson until reviewed; generated datasets are intentionally excluded from deploy syncs.

For short operational review clips, enable turret event recording in `configs/app.yaml`:

```yaml
event_recording:
  enabled: true
  output_dir: data/event_videos
  post_shot_seconds: 15
  zone_confirm_seconds: 5
  zone_confirm_detections: 20
  zone_lost_seconds: 5
  discord_webhook_env: CAT_CANNON_DISCORD_WEBHOOK_URL
```

Set the Discord webhook URL in the Jetson environment rather than committing it:

```bash
export CAT_CANNON_DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'
```

The recorder starts tentatively on the first in-zone cat update, keeps the clip only after at least
`zone_confirm_detections` positive updates arrive within `zone_confirm_seconds`, and then waits for
`zone_lost_seconds` without an in-zone detection before stopping. Finished clips are posted with
mentions disabled and are also kept under `data/event_videos`. The event recorder reads config at
app startup, so restart the app after changing `configs/app.yaml`.

Synthetic smoke-test expectation:

- `load_event_recording_config("configs/app.yaml")` reports `enabled=True`
- `webhook_configured` is true, without printing the webhook URL
- a generated turret event writes an `.mp4` under `data/event_videos`
- Discord accepts the upload and receives a message like `zone=counter confirmed=20 shots=1`

After reviewing labels, validate the firing algorithm offline before enabling live actuation:

```bash
python scripts/replay_cat_algorithm_dataset.py \
  --dataset data/reviewed_cat_training \
  --config configs/app.yaml \
  --zones configs/zones.yaml \
  --aim-deadband-px 10
```

## Rollback Model

- Pico: keep the previous `main.py` and `pico_config.py` in source control and redeploy the older
  commit with `deploy_pico.sh`
- Jetson: reinstall from the previous git tag or commit and restart the service

## Operational Recommendation

Before each release:

- run replay tests on the Jetson
- verify the Pico handshake on a bench
- verify guided servo limits and saved center in the tracking screen
- keep live fire disabled until both devices report healthy status
