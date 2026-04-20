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

## Quick Smoke Test

Before opening the webcam UI, you can validate the RP2040 path directly:

```bash
./scripts/smoke_test.sh --dry-fire
```

Live fire path:

```bash
./scripts/smoke_test.sh
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
