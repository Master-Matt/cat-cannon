"""Main Cat Cannon application — starts at Eye screen with full navigation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

from cat_cannon.adapters.ultralytics_yolo import DEFAULT_YOLO_IMGSZ
from cat_cannon.config import DEFAULT_YOLOE_PROMPTS, YoloPrompt, load_vision_config

ScreenName = Literal["eye", "zone_calibration", "tracking_test"]


@dataclass
class AppConfig:
    fixed_camera: int | str = "/dev/fixed_cam"
    turret_camera: int | str | None = "/dev/turret_cam"
    turret_rotate_180: bool = True
    fixed_camera_width: int = 1280
    fixed_camera_height: int = 720
    turret_camera_width: int = 1280
    turret_camera_height: int = 720
    camera_fps: int = 30
    port: str | None = None
    baudrate: int = 115200
    fire_ms: int = 120
    config_path: str = "configs/app.yaml"
    zones_path: str = "configs/zones.yaml"
    yolo_model: str = "yolo11s.pt"
    yolo_device: str | None = None
    yolo_imgsz: int = DEFAULT_YOLO_IMGSZ
    yolo_detector: str = "yolo"
    yolo_prompts: tuple[YoloPrompt, ...] = DEFAULT_YOLOE_PROMPTS
    detect_interval: int = 5
    window_width: int = 1024
    window_height: int = 600
    fullscreen: bool = True
    live_controller: bool = True
    arm_on_start: bool = False


def parse_args(argv: list[str] | None = None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Cat Cannon — main application")
    parser.add_argument("--fixed-camera", default="/dev/fixed_cam")
    parser.add_argument("--turret-camera", default="/dev/turret_cam")
    parser.add_argument("--no-turret-rotate", action="store_true")
    parser.add_argument("--fixed-camera-width", type=int, default=1280)
    parser.add_argument("--fixed-camera-height", type=int, default=720)
    parser.add_argument("--turret-camera-width", type=int, default=1280)
    parser.add_argument("--turret-camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Override vision.yolo_imgsz from the app config",
    )
    parser.add_argument("--detect-interval", type=int, default=5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--no-fullscreen", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Don't connect to hardware")
    parser.add_argument("--arm", action="store_true")
    args = parser.parse_args(argv)
    vision_config = load_vision_config(args.config)
    yolo_imgsz = args.imgsz if args.imgsz is not None else vision_config.yolo_imgsz
    yolo_model = args.model if args.model else vision_config.selected_model_path
    return AppConfig(
        fixed_camera=args.fixed_camera,
        turret_camera=args.turret_camera,
        turret_rotate_180=not args.no_turret_rotate,
        fixed_camera_width=args.fixed_camera_width,
        fixed_camera_height=args.fixed_camera_height,
        turret_camera_width=args.turret_camera_width,
        turret_camera_height=args.turret_camera_height,
        camera_fps=args.camera_fps,
        port=args.port,
        baudrate=args.baudrate,
        config_path=args.config,
        zones_path=args.zones,
        yolo_model=yolo_model,
        yolo_device=args.device,
        yolo_imgsz=yolo_imgsz,
        yolo_detector=vision_config.yolo_detector,
        yolo_prompts=vision_config.yoloe_prompts,
        detect_interval=args.detect_interval,
        window_width=args.width,
        window_height=args.height,
        fullscreen=not args.no_fullscreen,
        live_controller=not args.dry_run,
        arm_on_start=args.arm,
    )


def run_app(config: AppConfig) -> None:
    current: ScreenName | None = "eye"

    while current is not None:
        if current == "eye":
            from cat_cannon.app.eye_screen import EyeConfig, run_eye_screen

            current = run_eye_screen(
                EyeConfig(
                    fixed_camera=config.fixed_camera,
                    turret_camera=config.turret_camera,
                    turret_rotate_180=config.turret_rotate_180,
                    fixed_camera_width=config.fixed_camera_width,
                    fixed_camera_height=config.fixed_camera_height,
                    turret_camera_width=config.turret_camera_width,
                    turret_camera_height=config.turret_camera_height,
                    camera_fps=config.camera_fps,
                    port=config.port,
                    baudrate=config.baudrate,
                    fire_ms=config.fire_ms,
                    config_path=config.config_path,
                    zones_path=config.zones_path,
                    yolo_model=config.yolo_model,
                    yolo_device=config.yolo_device,
                    yolo_imgsz=config.yolo_imgsz,
                    yolo_detector=config.yolo_detector,
                    yolo_prompts=config.yolo_prompts,
                    window_width=config.window_width,
                    window_height=config.window_height,
                    fullscreen=config.fullscreen,
                    live_controller=config.live_controller,
                    arm_on_start=config.arm_on_start,
                )
            )
            continue

        if current == "zone_calibration":
            from cat_cannon.app.calibrate_zones import CalibrationConfig, run_calibration_screen

            current = run_calibration_screen(
                CalibrationConfig(
                    camera=config.fixed_camera,
                    camera_width=config.fixed_camera_width,
                    camera_height=config.fixed_camera_height,
                    camera_fps=config.camera_fps,
                    output_path=config.zones_path,
                    zone_prefix="zone",
                    window_width=config.window_width,
                    window_height=config.window_height,
                    panel_width=min(260, config.window_width // 4),
                    fullscreen=config.fullscreen,
                    detect=True,
                    yolo_model=config.yolo_model,
                    yolo_device=config.yolo_device,
                    yolo_imgsz=config.yolo_imgsz,
                    yolo_detector=config.yolo_detector,
                    yolo_prompts=config.yolo_prompts,
                    config_path=config.config_path,
                )
            )
            continue

        if current == "tracking_test":
            from cat_cannon.app.tracking_test import TrackingTestConfig, run_tracking_test_screen

            current = run_tracking_test_screen(
                TrackingTestConfig(
                    fixed_camera=config.fixed_camera,
                    turret_camera=config.turret_camera,
                    turret_rotate_180=config.turret_rotate_180,
                    fixed_camera_width=config.fixed_camera_width,
                    fixed_camera_height=config.fixed_camera_height,
                    turret_camera_width=config.turret_camera_width,
                    turret_camera_height=config.turret_camera_height,
                    camera_fps=config.camera_fps,
                    port=config.port,
                    baudrate=config.baudrate,
                    fire_ms=config.fire_ms,
                    config_path=config.config_path,
                    zones_path=config.zones_path,
                    yolo_model=config.yolo_model,
                    yolo_device=config.yolo_device,
                    yolo_imgsz=config.yolo_imgsz,
                    yolo_detector=config.yolo_detector,
                    yolo_prompts=config.yolo_prompts,
                    detect_interval=config.detect_interval,
                    window_width=config.window_width,
                    window_height=config.window_height,
                    panel_width=280,
                    fullscreen=config.fullscreen,
                    live_controller=config.live_controller,
                )
            )
            continue

        break


def main(argv: list[str] | None = None) -> None:
    # Prevent PyTorch/ultralytics from spawning worker processes (fork bomb on Jetson)
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("fork", force=True)
    except (ImportError, RuntimeError):
        pass
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    run_app(parse_args(argv))


if __name__ == "__main__":
    main()
