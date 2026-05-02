from __future__ import annotations

import argparse
from dataclasses import dataclass

from cat_cannon.config import load_system_config
from cat_cannon.domain.models import Detection
from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port
from cat_cannon.adapters.rp2040_serial import RP2040SerialController
from cat_cannon.adapters.ultralytics_yolo import (
    UltralyticsYoloDetector,
    YoloRuntimeConfig,
    build_detection_summary,
)
from cat_cannon.app.controller_session import ControllerSession


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "opencv-python is required for bench mode. Install with: pip install -e '.[bench]'"
        ) from exc
    return cv2


@dataclass
class BenchConfig:
    port: str | None
    camera: int
    secondary_camera: int | None
    baudrate: int
    step_deg: float
    fire_ms: int
    detect: bool
    config_path: str
    yolo_model: str
    yolo_device: str | None
    yolo_imgsz: int


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(description="Cat Cannon laptop bench harness")
    parser.add_argument("--port", help="RP2040 serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--camera", type=int, default=0, help="Primary webcam index")
    parser.add_argument("--secondary-camera", type=int, default=None, help="Optional second webcam index")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--step-deg", type=float, default=3.0, help="Pan/tilt delta per keypress")
    parser.add_argument("--fire-ms", type=int, default=120, help="Solenoid pulse duration")
    parser.add_argument("--detect", action="store_true", help="Enable YOLO11 detection overlay")
    parser.add_argument("--config", default="configs/app.example.yaml", help="System config path")
    parser.add_argument("--yolo-model", default="", help="YOLO model path (default: bundled yolo11s.pt)")
    parser.add_argument("--yolo-device", default=None, help="Optional inference device, e.g. cpu or 0")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()
    return BenchConfig(
        port=args.port,
        camera=args.camera,
        secondary_camera=args.secondary_camera,
        baudrate=args.baudrate,
        step_deg=args.step_deg,
        fire_ms=args.fire_ms,
        detect=args.detect,
        config_path=args.config,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
    )


def _open_camera(cv2, index: int):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise SystemExit(f"Failed to open camera index {index}")
    return camera


def _annotate_frame(cv2, frame, label: str, status_lines: list[str]):
    height, width = frame.shape[:2]
    cv2.line(frame, (width // 2 - 20, height // 2), (width // 2 + 20, height // 2), (0, 255, 0), 1)
    cv2.line(frame, (width // 2, height // 2 - 20), (width // 2, height // 2 + 20), (0, 255, 0), 1)
    cv2.putText(frame, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for index, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (10, 50 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
    return frame


def _draw_detections(cv2, frame, detections: list[Detection]):
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


def _combine_frames(cv2, primary, secondary):
    if secondary is None:
        return primary
    primary_height = primary.shape[0]
    secondary = cv2.resize(
        secondary,
        (int(secondary.shape[1] * (primary_height / secondary.shape[0])), primary_height),
    )
    return cv2.hconcat([primary, secondary])


def main() -> None:
    cv2 = _require_cv2()
    config = parse_args()
    try:
        port = config.port or autodetect_port()
    except RP2040DiscoveryError as exc:
        raise SystemExit(str(exc)) from exc

    controller = RP2040SerialController.open(
        port=port,
        baudrate=config.baudrate,
        fire_pulse_ms=config.fire_ms,
    )
    session = ControllerSession(controller=controller)
    system_config = load_system_config(config.config_path) if config.detect else None
    detector = None
    if config.detect and system_config is not None:
        detector = UltralyticsYoloDetector.open(
            policy=system_config.detection_policy,
            runtime=YoloRuntimeConfig(
                model_path=config.yolo_model,
                device=config.yolo_device,
                imgsz=config.yolo_imgsz,
            ),
        )

    primary = _open_camera(cv2, config.camera)
    secondary = _open_camera(cv2, config.secondary_camera) if config.secondary_camera is not None else None

    enabled = False
    try:
        session.start()
        status = controller.status()

        while True:
            ok_primary, primary_frame = primary.read()
            if not ok_primary:
                raise SystemExit("Failed to read from primary camera")
            detections: list[Detection] = []
            detection_summary = "detections=off"
            if detector is not None and system_config is not None:
                perception_frame = detector.detect(primary_frame, source_id="primary")
                detections = perception_frame.detections
                detection_summary = build_detection_summary(
                    detections=detections,
                    policy=system_config.detection_policy,
                )
                primary_frame = _draw_detections(cv2, primary_frame, detections)

            secondary_frame = None
            if secondary is not None:
                ok_secondary, secondary_frame = secondary.read()
                if not ok_secondary:
                    raise SystemExit("Failed to read from secondary camera")

            status_lines = [
                f"port={port}",
                f"enabled={status.payload.get('enabled', False)} pan={status.payload.get('pan_deg', '?')} tilt={status.payload.get('tilt_deg', '?')}",
                detection_summary,
                "keys: e enable  d disable  wasd move  space fire  p poll  q quit",
            ]
            primary_frame = _annotate_frame(cv2, primary_frame, "primary camera", status_lines)
            if secondary_frame is not None:
                secondary_frame = _annotate_frame(cv2, secondary_frame, "secondary camera", [])

            combined = _combine_frames(cv2, primary_frame, secondary_frame)
            cv2.imshow("cat-cannon-bench", combined)

            key = cv2.waitKey(16) & 0xFF
            if key == 255:
                continue

            if key == ord("q"):
                break
            if key == ord("e"):
                session.enable()
                enabled = True
            elif key == ord("d"):
                session.disable()
                enabled = False
            elif key == ord("w"):
                controller.apply_tracking_delta(0.0, -config.step_deg)
            elif key == ord("s"):
                controller.apply_tracking_delta(0.0, config.step_deg)
            elif key == ord("a"):
                controller.apply_tracking_delta(-config.step_deg, 0.0)
            elif key == ord("f"):
                controller.apply_tracking_delta(config.step_deg, 0.0)
            elif key == 32:  # space
                if enabled:
                    controller.fire()
            elif key == ord("p"):
                pass

            status = controller.status()
    finally:
        if secondary is not None:
            secondary.release()
        primary.release()
        cv2.destroyAllWindows()
        session.stop()


if __name__ == "__main__":
    main()
