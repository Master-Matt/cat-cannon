# Cat Cannon

Jetson-based cat deterrence system scaffold for a two-camera deployment:

- a fixed safety/perception camera that detects cats, people, and calibrated counter zones
- a pan/tilt turret camera that performs fine target centering
- an external actuation controller responsible for servo PWM and solenoid timing

This repository currently contains the application core:

- domain models for detections, zones, and targeting
- scene safety and counter-zone reasoning
- a supervisor state machine for arming, lockout, tracking, and cooldown
- configurable Ultralytics YOLO/YOLOE perception adapters
- OpenCV operator screens for the eye view, zone calibration, and tracking/teleop
- dataset capture, fine-tuning dataset preparation, and algorithm replay tooling
- adapter boundaries for cameras and RP2040 actuation hardware
- tests for the decision logic and operator workflows

## Current Assumptions

- v1 target stack is JetPack 6.1 GA with DeepStream 7.1
- counter presence is modeled with calibrated polygons, not segmentation
- people in either camera frame always disable actuation
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

For offline dataset augmentation:

```bash
pip install -e ".[dev,bench,vision,training]"
```

## Laptop Bench Test

Once a Raspberry Pi Pico is flashed with the firmware in `firmware/pico/`, you can run a local
bench test from a laptop with one or two USB webcams:

```bash
./scripts/bench.sh --port /dev/ttyACM0 --camera 0
```

Optional second camera:

```bash
./scripts/bench.sh --port /dev/ttyACM0 --camera 0 --secondary-camera 1
```

If the Pico is the only matching serial device, you can omit `--port`.

Quick controller smoke test:

```bash
./scripts/smoke_test.sh --port /dev/ttyACM0 --dry-fire
```

Keyboard teleop over an SSH session to the Jetson:

```bash
./scripts/run_teleop.sh --port /dev/ttyACM0
```

YOLO11 detection overlay on the primary camera:

```bash
./scripts/bench.sh --port /dev/ttyACM0 --camera 0 --detect --yolo-model yolo11n.pt
```

If the Pico is in `BOOTSEL` mode, flash MicroPython first:

```bash
./scripts/flash_pico_uf2.sh
./scripts/deploy_pico.sh
```

`deploy_pico.sh` now uses plain USB serial plus `pyserial`; `mpremote` is not required.

Keyboard controls are shown in the bench window and documented in `docs/laptop-bench.md`.

Deploy to Jetson:

```bash
# Via USB OTG (fixed IP):
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --host 192.168.55.1 --user mdev

# Via LAN (find the Jetson's IP on your network):
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --host <JETSON_LAN_IP> --user mdev
```

If the Jetson image is already provisioned and APT is offline or misconfigured:

```bash
JETSON_PASSWORD=nvidia ./scripts/deploy_jetson.sh --skip-system-packages
```

## Fixed Camera Runtime

The main Jetson app starts on the eye screen and can switch between the eye, zone calibration,
and tracking test screens:

```bash
./scripts/run_app.sh --config configs/app.yaml --zones configs/zones.yaml
```

Dry-run fixed camera on Jetson or a Linux host:

```bash
./scripts/run_fixed_camera.sh --camera 0 --show-window
```

Live fixed camera with the RP2040 controller attached:

```bash
./scripts/run_fixed_camera.sh \
  --camera 0 \
  --port /dev/ttyACM0 \
  --live-controller \
  --show-window
```

By default the runtime starts disarmed. In the window:

- `a` arms the system
- `d` disarms the system
- `q` quits

## Tracking Test UI

Two-camera tracking test screen with live overlays and teleop controls:

```bash
./scripts/run_tracking_test.sh --fixed-camera /dev/fixed_cam --turret-camera /dev/turret_cam
```

For live RP2040 control, keep the UI disarmed until the camera view and zone overlay look right:

```bash
./scripts/run_tracking_test.sh \
  --fixed-camera /dev/fixed_cam \
  --turret-camera /dev/turret_cam \
  --port /dev/ttyACM0 \
  --live-controller
```

Controls are available as on-screen buttons and keyboard shortcuts:

- `z`: switch to zone calibration
- `e`: arm
- `x`: disarm and safe stop
- `h`: toggle human tracking for aiming tests; defaults off each time the screen opens
- `w/a/s/d`: tilt/pan
- `c`: save current servo center
- `l`: start guided limit setup
- `v`: save the current guided limit
- `0`: clear saved center and limits
- `space`: fire once when armed
- `p`: poll controller status
- `q`: quit

From an SSH session with X forwarding:

```bash
# 192.168.55.1 = USB OTG; use LAN IP if connected via network
ssh -Y mdev@192.168.55.1
cd ~/cat_cannon
./scripts/run_tracking_test_x11.sh --live-controller --port /dev/ttyACM0
```

If the camera symlinks are not installed yet, pass the Jetson device names explicitly:

```bash
CAT_CANNON_FIXED_CAMERA=/dev/video0 \
CAT_CANNON_TURRET_CAMERA=/dev/video2 \
./scripts/run_tracking_test_x11.sh --live-controller --port /dev/ttyACM0
```

## Zone Calibration

Touchscreen-friendly zone calibration on a `1024x600` display:

```bash
./scripts/run_zone_calibrator.sh --camera 0 --output configs/zones.yaml --fullscreen
```

How it works:

- tap arbitrary points around each counter zone
- tap near the first point to close the polygon
- tap `Save Zones` when the polygons look correct
- saved zones include normalized coordinates so they can be scaled across camera resolutions
- use the on-screen buttons or hotkeys to undo, clear pending points, or delete the last zone
- tap `Tracking Test` or press `t` to switch to the tracking test screen

You can run the same calibrator from the laptop bench setup:

```bash
./scripts/run_zone_calibrator.sh --camera 0 --output configs/zones.yaml
```

## Perception Configuration

`configs/app.example.yaml` contains the runtime perception knobs:

- `vision.yolo_detector: yolo` uses standard YOLO models such as `yolo11s.pt`
- `vision.yolo_detector: yoloe` uses promptable YOLOE with `vision.yoloe_prompts`
- `vision.yolo_imgsz` controls the inference image size without rebuilding the code
- `tracking.horizontal_deadband_px` and `tracking.vertical_deadband_px` control aim lock

The Jetson runtime preserves `configs/app.yaml` and `configs/zones.yaml` during deploy so local
calibration and model choices are not overwritten by example defaults.

## Training Data and Replay

The eye and tracking screens can collect raw, unannotated camera images plus YOLO-format labels
when confident cat or person detections occur:

```bash
./scripts/run_app.sh \
  --collect-cat-dataset \
  --dataset-dir data/cat_training \
  --dataset-sample-hz 1.0
```

Both fixed and turret cameras write to separate `images/<camera>` and `labels/<camera>`
subdirectories under the dataset root. Images are saved without overlays. Labels use class `0` for
cats and class `1` for people.

Turret event clips can also be recorded when the fixed camera sees a cat in a calibrated zone:

```yaml
event_recording:
  enabled: true
  output_dir: data/event_videos
  post_shot_seconds: 15
  zone_confirm_seconds: 5
  zone_confirm_detections: 20
  zone_lost_seconds: 5
  max_event_seconds: 180
  video_fps: 10
  max_width: 640
  discord_max_upload_mb: 8
  publish_requires_shot: true
  discord_webhook_env: CAT_CANNON_DISCORD_WEBHOOK_URL
```

With this enabled, the app starts a tentative turret recording when the fixed camera sees a cat in
a zone. The clip is kept only after at least `zone_confirm_detections` positive zone updates arrive
inside `zone_confirm_seconds`; once confirmed, recording stays open until the zone has been quiet
for `zone_lost_seconds`. If a shot fires or the turret still sees the target, recording stays open
until activity stops and `post_shot_seconds` have passed since the last shot. With
`publish_requires_shot`, no-shot diagnostic clips are kept locally but not posted to Discord. Use
`video_fps`, `max_width`, and `discord_max_upload_mb` to keep longer clips below the Discord webhook
upload limit. Set `CAT_CANNON_DISCORD_WEBHOOK_URL` on the Jetson to post finished clips to Discord
without committing the webhook secret.

Prepare a reviewed dataset for Ultralytics fine-tuning:

```bash
python scripts/prepare_cat_finetune_dataset.py \
  --source data/reviewed_cat_training \
  --output data/fine_tune/cat_v1 \
  --val-fraction 0.2 \
  --augment-train 2 \
  --augment-fixed-extra 2
```

Replay reviewed fixed/turret image-label pairs through the dry-run supervisor before live tests:

```bash
python scripts/replay_cat_algorithm_dataset.py \
  --dataset data/reviewed_cat_training \
  --config configs/app.yaml \
  --zones configs/zones.yaml \
  --aim-deadband-px 10
```
