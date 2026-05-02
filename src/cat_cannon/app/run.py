from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port
from cat_cannon.adapters.rp2040_serial import RP2040SerialController
from cat_cannon.adapters.ultralytics_yolo import (
    UltralyticsYoloDetector,
    YoloRuntimeConfig,
    build_detection_summary,
)
from cat_cannon.app.controller_session import ControllerSession
from cat_cannon.app.supervisor import SupervisorLoop, SupervisorStepResult
from cat_cannon.config import load_counter_zones, load_system_config


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "opencv-python is required for the fixed-camera runtime. Install with: "
            "pip install -e '.[bench,vision]'"
        ) from exc
    return cv2


@dataclass(frozen=True)
class RuntimeConfig:
    camera: int | str
    port: str | None
    baudrate: int
    config_path: str
    zones_path: str
    yolo_model: str
    yolo_device: str | None
    yolo_imgsz: int
    live_controller: bool
    arm_on_start: bool
    show_window: bool
    log_interval_s: float


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Cat Cannon fixed-camera runtime")
    parser.add_argument("--camera", default="/dev/fixed_cam", help="Fixed camera device path or index")
    parser.add_argument("--port", help="RP2040 serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--config", default="configs/app.example.yaml", help="System config path")
    parser.add_argument("--zones", default="configs/zones.example.yaml", help="Counter zones config path")
    parser.add_argument("--yolo-model", default="", help="YOLO model path (default: bundled yolo11s.pt)")
    parser.add_argument("--yolo-device", default=None, help="Optional inference device, e.g. cpu or 0")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--live-controller",
        action="store_true",
        help="Use the RP2040 controller instead of dry-run mode",
    )
    parser.add_argument("--arm-on-start", action="store_true", help="Start armed")
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show a live camera window with overlays and keyboard controls",
    )
    parser.add_argument(
        "--log-interval-s",
        type=float,
        default=2.0,
        help="How often to print status while headless",
    )
    args = parser.parse_args()
    camera_val = args.camera
    try:
        camera_val = int(camera_val)
    except (TypeError, ValueError):
        pass
    return RuntimeConfig(
        camera=camera_val,
        port=args.port,
        baudrate=args.baudrate,
        config_path=args.config,
        zones_path=args.zones,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
        live_controller=bool(args.live_controller),
        arm_on_start=bool(args.arm_on_start),
        show_window=bool(args.show_window),
        log_interval_s=float(args.log_interval_s),
    )


def _open_camera(cv2, device: int | str):
    from cat_cannon.adapters.camera import open_camera
    return open_camera(cv2, device)


def _resolve_port(port: str | None) -> str:
    try:
        return port or autodetect_port()
    except RP2040DiscoveryError as exc:
        raise SystemExit(str(exc)) from exc


def _draw_detections(cv2, frame, detections):
    colors = {
        "cat": (0, 165, 255),
        "person": (0, 0, 255),
    }
    for detection in detections:
        x1 = int(detection.bbox.x)
        y1 = int(detection.bbox.y)
        x2 = int(detection.bbox.x + detection.bbox.width)
        y2 = int(detection.bbox.y + detection.bbox.height)
        color = colors.get(detection.label, (255, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return frame


def _draw_zones(cv2, frame, zones):
    for zone in zones:
        points = [(int(point.x), int(point.y)) for point in zone.polygon]
        if len(points) < 2:
            continue
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            cv2.line(frame, start, end, (255, 200, 0), 2)
        anchor = points[0]
        cv2.putText(
            frame,
            zone.zone_id,
            (anchor[0], max(20, anchor[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
        )
    return frame


def _annotate_frame(cv2, frame, status_lines):
    height, width = frame.shape[:2]
    cv2.line(frame, (width // 2 - 20, height // 2), (width // 2 + 20, height // 2), (0, 255, 0), 1)
    cv2.line(frame, (width // 2, height // 2 - 20), (width // 2, height // 2 + 20), (0, 255, 0), 1)
    for index, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (10, 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return frame


def _status_lines(
    *,
    armed: bool,
    live_controller: bool,
    detection_summary: str,
    step_result: SupervisorStepResult,
) -> list[str]:
    zone = step_result.active_zone_id or "-"
    target = step_result.candidate_track_id or "-"
    return [
        f"mode={'live' if live_controller else 'dry-run'} armed={armed}",
        f"state={step_result.state.value} human={step_result.human_present} counter={step_result.counter_confirmed} zone={zone}",
        f"target={target} aim_locked={step_result.aim_locked} fire={step_result.fire_commanded}",
        detection_summary,
        "keys: a arm  d disarm  q quit",
    ]


def _print_status(
    *,
    armed: bool,
    live_controller: bool,
    detection_summary: str,
    step_result: SupervisorStepResult,
) -> None:
    zone = step_result.active_zone_id or "-"
    target = step_result.candidate_track_id or "-"
    print(
        "[runtime]",
        f"mode={'live' if live_controller else 'dry-run'}",
        f"armed={armed}",
        f"state={step_result.state.value}",
        f"human={step_result.human_present}",
        f"counter={step_result.counter_confirmed}",
        f"zone={zone}",
        f"target={target}",
        detection_summary,
    )


def main() -> None:
    config = parse_args()
    cv2 = _require_cv2()

    system_config = load_system_config(config.config_path)
    zones = load_counter_zones(config.zones_path)
    detector = UltralyticsYoloDetector.open(
        policy=system_config.detection_policy,
        runtime=YoloRuntimeConfig(
            model_path=config.yolo_model,
            device=config.yolo_device,
            imgsz=config.yolo_imgsz,
        ),
    )

    controller = NullTurretController()
    session: ControllerSession | None = None
    if config.live_controller:
        port = _resolve_port(config.port)
        controller = RP2040SerialController.open(
            port=port,
            baudrate=config.baudrate,
        )
        session = ControllerSession(controller=controller)

    supervisor = SupervisorLoop(config=system_config, zones=zones, controller=controller)
    camera = _open_camera(cv2, config.camera)

    armed = config.arm_on_start
    last_log = 0.0

    try:
        if session is not None:
            session.start()
            if armed:
                session.enable()
            else:
                session.disable()

        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit(f"Failed to read from fixed camera index {config.camera}")

            perception_frame = detector.detect(frame, source_id="fixed")
            detection_summary = build_detection_summary(
                detections=perception_frame.detections,
                policy=system_config.detection_policy,
            )
            step_result = supervisor.process_frame(
                detections=perception_frame.detections,
                frame_width=perception_frame.width,
                frame_height=perception_frame.height,
                armed=armed,
            )

            if config.show_window:
                frame = _draw_zones(cv2, frame, zones)
                frame = _draw_detections(cv2, frame, perception_frame.detections)
                frame = _annotate_frame(
                    cv2,
                    frame,
                    _status_lines(
                        armed=armed,
                        live_controller=config.live_controller,
                        detection_summary=detection_summary,
                        step_result=step_result,
                    ),
                )
                cv2.imshow("cat-cannon-fixed-camera", frame)

                key = cv2.waitKey(16) & 0xFF
                if key == 255:
                    continue
                if key == ord("q"):
                    break
                if key == ord("a"):
                    armed = True
                    if session is not None:
                        session.enable()
                elif key == ord("d"):
                    armed = False
                    if session is not None:
                        session.disable()
            else:
                now = time.monotonic()
                if now - last_log >= config.log_interval_s:
                    _print_status(
                        armed=armed,
                        live_controller=config.live_controller,
                        detection_summary=detection_summary,
                        step_result=step_result,
                    )
                    last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        if config.show_window:
            cv2.destroyAllWindows()
        if session is not None:
            session.stop()


if __name__ == "__main__":
    main()
