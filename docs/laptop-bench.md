# Laptop Bench Workflow

This workflow is for validating the full actuator control chain on a laptop before moving to the
Jetson runtime.

## What It Tests

- Pico firmware deployment over USB
- serial handshake and status polling
- watchdog heartbeats from host to Pico
- manual pan/tilt movement
- bounded fire command path
- one or two local USB webcams for visual confirmation

## Setup

1. Flash MicroPython UF2 to the Pico if needed.
2. Connect the Pico over USB.
3. If the board is in `BOOTSEL` mode, flash the bundled UF2:

```bash
./scripts/flash_pico_uf2.sh
```

4. After it reboots as a MicroPython device, deploy firmware files:

```bash
./scripts/deploy_pico.sh
```

If auto-detection does not work, pass the serial device explicitly:

```bash
./scripts/deploy_pico.sh --port /dev/ttyACM0
```

5. Create a Python environment and install bench dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,bench,vision]"
```

## Run

Single camera:

```bash
./scripts/bench.sh --camera 0
```

Single camera with YOLO11 detections overlaid on the primary feed:

```bash
./scripts/bench.sh --camera 0 --detect --yolo-model yolo11n.pt
```

Two cameras:

```bash
./scripts/bench.sh --camera 0 --secondary-camera 1
```

If multiple serial devices are present, pass `--port` explicitly.

## Zone Calibration

You can calibrate the counter polygons from the laptop before moving to the Jetson:

```bash
./scripts/run_zone_calibrator.sh --camera 0 --output configs/zones.yaml
```

For a touchscreen-sized full-screen layout:

```bash
./scripts/run_zone_calibrator.sh \
  --camera 0 \
  --output configs/zones.yaml \
  --window-width 1024 \
  --window-height 600 \
  --fullscreen
```

The calibrator works with either touch or a mouse:

- tap or click four corners to create one zone
- `Save Zones` writes the YAML file used by the fixed-camera runtime
- `Undo`, `Clear Pending`, and `Delete Last` help refine the layout without editing YAML by hand
- `Tracking Test` switches directly into the tracking/teleop screen

## Tracking Test UI

With both cameras connected, open the tracking screen:

```bash
./scripts/run_tracking_test.sh --fixed-camera 0 --turret-camera 2
```

On the Jetson with udev symlinks, use the device paths:

```bash
./scripts/run_tracking_test.sh --fixed-camera /dev/fixed_cam --turret-camera /dev/turret_cam
```

The fixed camera pane shows detections, zones, and tracking state. The turret camera pane is for
visual confirmation while using the on-screen teleop buttons or `e`, `x`, `w/a/s/d`, `space`, `p`,
and `q`.

## Quick Smoke Test

Before opening the webcam UI, you can validate the RP2040 path directly:

```bash
./scripts/smoke_test.sh --dry-fire
```

Live fire path:

```bash
./scripts/smoke_test.sh
```

## SSH Teleop

If the Pico is attached to the Jetson and you are SSH'd in from the laptop, use:

```bash
./scripts/run_teleop.sh --port /dev/ttyACM1
```

Controls:

- `e`: arm
- `x`: disarm and safe stop
- `w`: tilt up
- `a`: pan left
- `s`: tilt down
- `d`: pan right
- `space`: fire once
- `p`: print status
- `q`: quit

For relay and solenoid bench debugging, you can temporarily use a longer pulse:

```bash
./scripts/run_teleop.sh --port /dev/ttyACM1 --fire-ms 750 --arm-on-start
```

## Keyboard Controls

- `e`: enable actuation
- `d`: disable actuation
- `w`: tilt up
- `s`: tilt down
- `a`: pan left
- `f`: pan right
- `space`: fire once
- `p`: refresh status
- `q`: quit

## Safety

- Start with the solenoid mechanically disconnected.
- Confirm servo travel limits before enabling fire.
- Only test the firing output with a safe dummy load or disconnected driver stage first.
