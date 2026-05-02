from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cat_cannon.adapters.ultralytics_yolo import (
    UltralyticsYoloDetector,
    YoloRuntimeConfig,
    build_detection_summary,
)
from cat_cannon.app.zone_calibration import CalibrationLayout, ZoneCalibrationSession, map_display_to_frame
from cat_cannon.config import load_counter_zones, load_system_config, save_counter_zones
from cat_cannon.domain.models import CounterZone, Detection

ScreenName = Literal["zone_calibration", "tracking_test"]


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "opencv-python is required for zone calibration. Install with: "
            "pip install -e '.[bench]'"
        ) from exc
    return cv2


@dataclass(frozen=True)
class CalibrationConfig:
    camera: int | str
    output_path: str
    zone_prefix: str
    window_width: int
    window_height: int
    panel_width: int
    fullscreen: bool
    detect: bool = True
    detect_interval: int = 3
    yolo_model: str = "yolo11s.pt"
    yolo_device: str | None = None
    yolo_imgsz: int = 640
    config_path: str = "configs/app.example.yaml"


@dataclass(frozen=True)
class UiButton:
    key: str
    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def parse_args() -> CalibrationConfig:
    parser = argparse.ArgumentParser(description="Touchscreen counter-zone calibration tool")
    parser.add_argument("--camera", default="/dev/fixed_cam", help="Camera device path or index")
    parser.add_argument("--output", default="configs/zones.example.yaml", help="Output YAML path")
    parser.add_argument("--zone-prefix", default="zone", help="Prefix for generated zone ids")
    parser.add_argument("--window-width", type=int, default=1024, help="Calibration window width")
    parser.add_argument("--window-height", type=int, default=600, help="Calibration window height")
    parser.add_argument("--panel-width", type=int, default=224, help="Control panel width")
    parser.add_argument("--fullscreen", action="store_true", help="Open the calibration UI in fullscreen mode")
    parser.add_argument("--no-detect", action="store_true", help="Disable YOLO overlay for slow/debug sessions")
    parser.add_argument("--yolo-model", default="", help="YOLO model path (default: bundled yolo11s.pt)")
    parser.add_argument("--yolo-device", default=None, help="Inference device")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--config", default="configs/app.example.yaml", help="System config path")
    args = parser.parse_args()
    camera_arg = args.camera
    try:
        camera_arg = int(camera_arg)
    except ValueError:
        pass
    return CalibrationConfig(
        camera=camera_arg,
        output_path=args.output,
        zone_prefix=args.zone_prefix,
        window_width=args.window_width,
        window_height=args.window_height,
        panel_width=args.panel_width,
        fullscreen=bool(args.fullscreen),
        detect=not args.no_detect,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
        config_path=args.config,
    )


def _build_layout(
    *,
    frame_width: int,
    frame_height: int,
    window_width: int,
    window_height: int,
    panel_width: int,
) -> CalibrationLayout:
    preview_width = max(1, window_width - panel_width)
    preview_height = max(1, int(preview_width * (frame_height / frame_width)))
    if preview_height > window_height:
        preview_height = window_height
        preview_width = max(1, int(preview_height * (frame_width / frame_height)))
    preview_offset_x = max(0, (window_width - panel_width - preview_width) // 2)
    preview_offset_y = max(0, (window_height - preview_height) // 2)
    return CalibrationLayout(
        window_width=window_width,
        window_height=window_height,
        preview_width=preview_width,
        preview_height=preview_height,
        preview_offset_x=preview_offset_x,
        preview_offset_y=preview_offset_y,
        panel_x=window_width - panel_width,
    )


def preview_padding(layout: CalibrationLayout) -> tuple[int, int, int, int]:
    top = max(0, layout.preview_offset_y)
    bottom = max(0, layout.window_height - layout.preview_offset_y - layout.preview_height)
    left = max(0, layout.preview_offset_x)
    right = max(0, layout.panel_x - layout.preview_offset_x - layout.preview_width)
    return top, bottom, left, right


def _build_buttons(config: CalibrationConfig) -> list[UiButton]:
    panel_x = config.window_width - config.panel_width
    button_width = config.panel_width - 32
    button_height = 50
    top = 112
    spacing = 12
    labels = [
        ("tracking", "Tracking Test"),
        ("save", "Save Zones"),
        ("undo", "Undo"),
        ("clear", "Clear Pending"),
        ("delete", "Delete Last"),
        ("quit", "Quit"),
    ]
    buttons: list[UiButton] = []
    for index, (key, label) in enumerate(labels):
        y1 = top + index * (button_height + spacing)
        buttons.append(
            UiButton(
                key=key,
                label=label,
                x1=panel_x + 16,
                y1=y1,
                x2=panel_x + 16 + button_width,
                y2=y1 + button_height,
            )
        )
    return buttons


def _draw_button(cv2, canvas, button: UiButton) -> None:
    cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), (70, 70, 70), -1)
    cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), (220, 220, 220), 2)
    cv2.putText(
        canvas,
        button.label,
        (button.x1 + 12, button.y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
    )


def _draw_zones(cv2, canvas, zones, layout: CalibrationLayout, frame_width: int, frame_height: int) -> None:
    scale_x = layout.preview_width / frame_width
    scale_y = layout.preview_height / frame_height
    for zone in zones:
        points = [
            (
                int(layout.preview_offset_x + point.x * scale_x),
                int(layout.preview_offset_y + point.y * scale_y),
            )
            for point in zone.polygon
        ]
        if not points:
            continue
        for index, start in enumerate(points):
            end = points[(index + 1) % len(points)]
            cv2.line(canvas, start, end, (0, 200, 255), 2)
        cv2.putText(
            canvas,
            zone.zone_id,
            (points[0][0], max(24, points[0][1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 255),
            2,
        )


def _draw_pending_points(cv2, canvas, points, layout: CalibrationLayout, frame_width: int, frame_height: int) -> None:
    scale_x = layout.preview_width / frame_width
    scale_y = layout.preview_height / frame_height
    display_points = [
        (
            int(layout.preview_offset_x + point.x * scale_x),
            int(layout.preview_offset_y + point.y * scale_y),
        )
        for point in points
    ]
    for index, point in enumerate(display_points):
        cv2.circle(canvas, point, 8, (0, 255, 0), -1)
        cv2.putText(
            canvas,
            str(index + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    for index in range(len(display_points) - 1):
        cv2.line(canvas, display_points[index], display_points[index + 1], (0, 255, 0), 2)


def _draw_detections_on_preview(
    cv2,
    preview,
    detections: list[Detection],
    frame_width: int,
    frame_height: int,
    preview_width: int,
    preview_height: int,
) -> None:
    scale_x = preview_width / frame_width
    scale_y = preview_height / frame_height
    colors = {"cat": (0, 165, 255), "person": (0, 0, 255)}
    for detection in detections:
        x1 = int(detection.bbox.x * scale_x)
        y1 = int(detection.bbox.y * scale_y)
        x2 = int((detection.bbox.x + detection.bbox.width) * scale_x)
        y2 = int((detection.bbox.y + detection.bbox.height) * scale_y)
        color = colors.get(detection.label, (255, 255, 0))
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            preview,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def _render_ui(
    *,
    cv2,
    frame,
    layout: CalibrationLayout,
    session: ZoneCalibrationSession,
    buttons: list[UiButton],
    output_path: str,
    status_message: str,
    detections: list[Detection] | None = None,
    detection_summary: str = "",
):
    preview = cv2.resize(frame, (layout.preview_width, layout.preview_height))
    if detections:
        frame_height, frame_width = frame.shape[:2]
        _draw_detections_on_preview(
            cv2,
            preview,
            detections,
            frame_width,
            frame_height,
            layout.preview_width,
            layout.preview_height,
        )
    top, bottom, left, right = preview_padding(layout)
    canvas = cv2.copyMakeBorder(
        preview,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(18, 18, 18),
    )
    if canvas.shape[1] < layout.panel_x:
        canvas = cv2.copyMakeBorder(
            canvas,
            top=0,
            bottom=0,
            left=0,
            right=layout.panel_x - canvas.shape[1],
            borderType=cv2.BORDER_CONSTANT,
            value=(18, 18, 18),
        )
    canvas = cv2.copyMakeBorder(
        canvas,
        top=0,
        bottom=0,
        left=0,
        right=layout.window_width - canvas.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=(38, 38, 38),
    )

    cv2.rectangle(canvas, (layout.panel_x, 0), (layout.window_width, layout.window_height), (38, 38, 38), -1)
    cv2.putText(canvas, "Zone Calibrator", (layout.panel_x + 16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"Zones: {len(session.zones)}", (layout.panel_x + 16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(
        canvas,
        f"Pending taps: {len(session.pending_points)}/4",
        (layout.panel_x + 16, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (220, 220, 220),
        2,
    )
    cv2.putText(canvas, "Tap 4 corners per zone", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, status_message, (16, layout.window_height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(
        canvas,
        f"Output: {output_path}",
        (layout.panel_x + 16, layout.window_height - 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        canvas,
        "Hotkeys: t tracking  s save  u undo  x clear  r remove  q quit",
        (layout.panel_x + 16, layout.window_height - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
    )
    if detection_summary:
        cv2.putText(
            canvas,
            detection_summary,
            (layout.panel_x + 16, layout.window_height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 200, 255),
            1,
        )

    for button in buttons:
        _draw_button(cv2, canvas, button)

    frame_height, frame_width = frame.shape[:2]
    _draw_zones(cv2, canvas, session.zones, layout, frame_width, frame_height)
    _draw_pending_points(cv2, canvas, session.pending_points, layout, frame_width, frame_height)
    return canvas


def run_calibration_screen(config: CalibrationConfig) -> ScreenName | None:
    cv2 = _require_cv2()
    output_path = Path(config.output_path)
    session = ZoneCalibrationSession(zone_prefix=config.zone_prefix)
    if output_path.exists():
        try:
            session.zones.extend(load_counter_zones(output_path))
        except Exception:
            pass

    detector = None
    detection_policy = None
    if config.detect:
        system_config = load_system_config(config.config_path)
        detection_policy = system_config.detection_policy
        detector = UltralyticsYoloDetector.open(
            policy=detection_policy,
            runtime=YoloRuntimeConfig(
                model_path=config.yolo_model,
                device=config.yolo_device,
                imgsz=config.yolo_imgsz,
            ),
        )

    from cat_cannon.adapters.camera import open_camera
    camera = open_camera(cv2, config.camera)

    window_name = "cat-cannon-zone-calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.window_width, config.window_height)
    if config.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    buttons = _build_buttons(config)
    status_message = "Tap four corners to create a zone."
    latest_frame = None
    latest_layout = None
    should_exit = False
    next_screen: ScreenName | None = None

    def on_mouse(event, x, y, _flags, _userdata):
        nonlocal status_message, should_exit, next_screen
        if event != cv2.EVENT_LBUTTONDOWN or latest_frame is None or latest_layout is None:
            return

        for button in buttons:
            if button.contains(x, y):
                if button.key == "tracking":
                    next_screen = "tracking_test"
                    should_exit = True
                elif button.key == "save":
                    save_counter_zones(output_path, session.zones)
                    status_message = f"Saved {len(session.zones)} zones to {output_path}"
                elif button.key == "undo":
                    session.undo()
                    status_message = "Undid last point or zone."
                elif button.key == "clear":
                    session.clear_pending()
                    status_message = "Cleared pending points."
                elif button.key == "delete":
                    if session.zones:
                        session.zones.pop()
                        status_message = "Deleted last zone."
                    else:
                        status_message = "No zones to delete."
                elif button.key == "quit":
                    should_exit = True
                return

        if x >= latest_layout.panel_x:
            return
        if not (
            latest_layout.preview_offset_x <= x <= latest_layout.preview_offset_x + latest_layout.preview_width
            and latest_layout.preview_offset_y <= y <= latest_layout.preview_offset_y + latest_layout.preview_height
        ):
            return

        frame_height, frame_width = latest_frame.shape[:2]
        point = map_display_to_frame(
            display_x=x,
            display_y=y,
            frame_width=frame_width,
            frame_height=frame_height,
            layout=latest_layout,
        )
        zone = session.add_point(point)
        if zone is not None:
            status_message = f"Created {zone.zone_id}. Tap four corners for the next zone."
        else:
            status_message = f"Captured point {len(session.pending_points)}/4."

    cv2.setMouseCallback(window_name, on_mouse)

    try:
        frame_counter = 0
        detections: list[Detection] = []
        detection_summary = ""

        while True:
            ok, frame = camera.read()
            if not ok:
                raise SystemExit(f"Failed to read from camera {config.camera}")
            latest_frame = frame.copy()
            latest_layout = _build_layout(
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                window_width=config.window_width,
                window_height=config.window_height,
                panel_width=config.panel_width,
            )

            if detector is not None and detection_policy is not None:
                if frame_counter % config.detect_interval == 0:
                    perception = detector.detect(frame, source_id="fixed")
                    detections = perception.detections
                    detection_summary = build_detection_summary(
                        detections=detections, policy=detection_policy
                    )
            frame_counter += 1

            canvas = _render_ui(
                cv2=cv2,
                frame=frame,
                layout=latest_layout,
                session=session,
                buttons=buttons,
                output_path=str(output_path),
                status_message=status_message,
                detections=detections,
                detection_summary=detection_summary,
            )
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(16) & 0xFF
            if should_exit:
                break
            if key == 255:
                continue
            if key == ord("q"):
                break
            if key == ord("t"):
                next_screen = "tracking_test"
                break
            if key == ord("s"):
                save_counter_zones(output_path, session.zones)
                status_message = f"Saved {len(session.zones)} zones to {output_path}"
            elif key == ord("u"):
                session.undo()
                status_message = "Undid last point or zone."
            elif key == ord("x"):
                session.clear_pending()
                status_message = "Cleared pending points."
            elif key == ord("r"):
                if session.zones:
                    session.zones.pop()
                    status_message = "Deleted last zone."
                else:
                    status_message = "No zones to delete."
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return next_screen


def run_calibration_with_navigation(config: CalibrationConfig) -> None:
    current: ScreenName | None = "zone_calibration"
    while current is not None:
        if current == "zone_calibration":
            current = run_calibration_screen(config)
            continue

        from cat_cannon.app.tracking_test import TrackingTestConfig, run_tracking_test_screen

        current = run_tracking_test_screen(
            TrackingTestConfig(
                fixed_camera=config.camera,
                turret_camera="/dev/turret_cam",
                zones_path=config.output_path,
                window_width=config.window_width,
                window_height=config.window_height,
                panel_width=max(260, config.panel_width),
                fullscreen=config.fullscreen,
            )
        )


def main() -> None:
    run_calibration_with_navigation(parse_args())


if __name__ == "__main__":
    main()
