"""Main Cat Cannon application — starts at Eye screen with full navigation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

ScreenName = Literal["eye", "zone_calibration", "tracking_test"]


@dataclass
class AppConfig:
    fixed_camera: int | str = "/dev/fixed_cam"
    turret_camera: int | str | None = "/dev/turret_cam"
    turret_rotate_180: bool = True
    port: str | None = None
    baudrate: int = 115200
    fire_ms: int = 120
    config_path: str = "configs/app.yaml"
    zones_path: str = "configs/zones.yaml"
    yolo_model: str = "yolo11s.pt"
    yolo_device: str | None = None
    yolo_imgsz: int = 640
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
    parser.add_argument("--port", default=None)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--zones", default="configs/zones.yaml")
    parser.add_argument("--model", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detect-interval", type=int, default=5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--no-fullscreen", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Don't connect to hardware")
    parser.add_argument("--arm", action="store_true")
    args = parser.parse_args(argv)
    return AppConfig(
        fixed_camera=args.fixed_camera,
        turret_camera=args.turret_camera,
        turret_rotate_180=not args.no_turret_rotate,
        port=args.port,
        baudrate=args.baudrate,
        config_path=args.config,
        zones_path=args.zones,
        yolo_model=args.model,
        yolo_device=args.device,
        yolo_imgsz=args.imgsz,
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
                    port=config.port,
                    baudrate=config.baudrate,
                    fire_ms=config.fire_ms,
                    config_path=config.config_path,
                    zones_path=config.zones_path,
                    yolo_model=config.yolo_model,
                    yolo_device=config.yolo_device,
                    yolo_imgsz=config.yolo_imgsz,
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
                    port=config.port,
                    baudrate=config.baudrate,
                    fire_ms=config.fire_ms,
                    config_path=config.config_path,
                    zones_path=config.zones_path,
                    yolo_model=config.yolo_model,
                    yolo_device=config.yolo_device,
                    yolo_imgsz=config.yolo_imgsz,
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
