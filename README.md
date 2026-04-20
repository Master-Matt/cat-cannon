# Cat Cannon

Jetson-based cat deterrence system scaffold for a two-camera deployment:

- a fixed safety/perception camera that detects cats, people, and calibrated counter zones
- a pan/tilt turret camera that performs fine target centering
- an external actuation controller responsible for servo PWM and solenoid timing

This repository currently contains the application core:

- domain models for detections, zones, and targeting
- scene safety and counter-zone reasoning
- a supervisor state machine for arming, lockout, tracking, and cooldown
- adapter boundaries for DeepStream and actuation hardware
- tests for the decision logic

## Current Assumptions

- v1 target stack is JetPack 6.1 GA with DeepStream 7.1
- counter presence is modeled with calibrated polygons, not segmentation
- people in frame always disable actuation
- servo PWM and solenoid pulses are delegated to an external controller

## Project Layout

```text
src/cat_cannon/
  adapters/      Interface boundaries for perception and actuation
  app/           Supervisor orchestration entrypoints
  domain/        Pure logic for geometry, safety, state, and targeting
  services/      Higher-level decision services
configs/         Example runtime configuration
tests/           Unit tests for the decision core
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
pyright
```

For laptop webcam detection tests with YOLO11:

```bash
pip install -e ".[dev,bench,vision]"
```

## Laptop Bench Test

Once a Raspberry Pi Pico is flashed with the firmware in `firmware/pico/`, you can run a local
bench test from a laptop with one or two USB webcams:

```bash
./scripts/bench.sh --port /dev/ttyACM1 --camera 0
```

Optional second camera:

```bash
./scripts/bench.sh --port /dev/ttyACM1 --camera 0 --secondary-camera 1
```

If the Pico is the only matching serial device, you can omit `--port`.

Quick controller smoke test:

```bash
./scripts/smoke_test.sh --port /dev/ttyACM1 --dry-fire
```

YOLO11 detection overlay on the primary camera:

```bash
./scripts/bench.sh --port /dev/ttyACM1 --camera 0 --detect --yolo-model yolo11n.pt
```

If the Pico is in `BOOTSEL` mode, flash MicroPython first:

```bash
./scripts/flash_pico_uf2.sh
./scripts/deploy_pico.sh
```

`deploy_pico.sh` now uses plain USB serial plus `pyserial`; `mpremote` is not required.

Keyboard controls are shown in the bench window and documented in `docs/laptop-bench.md`.

Jetson OTG deploy example:

```bash
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --host 192.168.55.1 --user mdev
```

If the Jetson image is already provisioned and APT is offline or misconfigured:

```bash
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --skip-system-packages
```
