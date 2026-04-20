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

## Jetson Update Flow

1. Connect the host to the Jetson over OTG or LAN.
2. Run:

```bash
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --host 192.168.55.1 --user mdev
```

3. If you want the bundled systemd unit installed or restarted:

```bash
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh \
  --host 192.168.55.1 \
  --user mdev \
  --remote-dir /opt/cat-cannon \
  --install-service \
  --restart-service
```

Notes:

- the default remote directory is `/home/mdev/cat_cannon`
- `sshpass` is optional; if it is missing the script falls back to normal SSH password prompts
- the script syncs the repo with `rsync`, creates a remote `.venv`, and installs the selected extras
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
  --port /dev/ttyACM1 \
  --live-controller \
  --show-window
```

Calibrate the counter polygons before arming the system:

```bash
./scripts/run_zone_calibrator.sh --camera 0 --output configs/zones.yaml --fullscreen
```

## Rollback Model

- Pico: keep the previous `main.py` and `pico_config.py` in source control and redeploy the older
  commit with `deploy_pico.sh`
- Jetson: reinstall from the previous git tag or commit and restart the service

## Operational Recommendation

Before each release:

- run replay tests on the Jetson
- verify the Pico handshake on a bench
- keep live fire disabled until both devices report healthy status
