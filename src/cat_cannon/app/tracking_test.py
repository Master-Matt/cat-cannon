from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.adapters.interfaces import PerceptionFrame, TurretController
from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port
from cat_cannon.adapters.rp2040_serial import RP2040ProtocolError, RP2040SerialController
from cat_cannon.adapters.ultralytics_yolo import (
    UltralyticsYoloDetector,
    YoloRuntimeConfig,
    build_detection_summary,
)
from cat_cannon.app.controller_session import ControllerSession
from cat_cannon.app.supervisor import SupervisorLoop, SupervisorStepResult
from cat_cannon.config import load_counter_zones, load_system_config
from cat_cannon.domain.models import CounterZone, Detection
from cat_cannon.domain.safety import DetectionPolicy

ScreenName = Literal["zone_calibration", "tracking_test"]


class SessionLike(Protocol):
    def enable(self) -> None:
        ...

    def disable(self) -> None:
        ...


@dataclass(frozen=True)
class TrackingTestConfig:
    fixed_camera: int | str = "/dev/fixed_cam"
    turret_camera: int | str | None = "/dev/turret_cam"
    port: str | None = None
    baudrate: int = 115200
    fire_ms: int = 120
    step_deg: float = 3.0
    config_path: str = "configs/app.example.yaml"
    zones_path: str = "configs/zones.yaml"
    yolo_model: str = "yolo11s.pt"
    yolo_device: str | None = None
    yolo_imgsz: int = 640
    detect_interval: int = 3
    live_controller: bool = False
    arm_on_start: bool = False
    window_width: int = 1280
    window_height: int = 720
    panel_width: int = 280
    fullscreen: bool = False


@dataclass(frozen=True)
class TrackingTestState:
    armed: bool
    step_deg: float


@dataclass(frozen=True)
class TrackingControlResult:
    state: TrackingTestState
    message: str
    should_exit: bool = False
    next_screen: ScreenName | None = None


@dataclass
class TraceLog:
    lines: list[str] = field(default_factory=list)
    max_lines: int = 8

    def add(self, message: str, *, emit: bool = True) -> None:
        self.lines.append(message)
        if len(self.lines) > self.max_lines:
            self.lines = self.lines[-self.max_lines :]
        if emit:
            print(f"[tracking-ui] {message}", flush=True)


@dataclass(frozen=True)
class TrackingCameraDetections:
    fixed: PerceptionFrame
    fixed_summary: str
    turret: PerceptionFrame | None
    turret_summary: str


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PreviewPlacement:
    region: Rect
    preview: Rect


@dataclass(frozen=True)
class TrackingLayout:
    window_width: int
    window_height: int
    panel_x: int
    fixed: PreviewPlacement
    turret: PreviewPlacement | None


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


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "opencv-python is required for the tracking test UI. Install with: "
            "pip install -e '.[bench,vision]'"
        ) from exc
    return cv2


def resolve_zones_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.name == "zones.yaml":
        fallback = candidate.with_name("zones.example.yaml")
        if fallback.exists():
            return fallback
    return candidate


def _source_detection_summary(
    *,
    source_id: str,
    detections: list[Detection],
    policy: DetectionPolicy,
) -> str:
    return f"{source_id} {build_detection_summary(detections=detections, policy=policy)}"


def detect_tracking_cameras(
    *,
    detector,
    fixed_frame,
    turret_frame,
    policy: DetectionPolicy,
) -> TrackingCameraDetections:
    fixed = detector.detect(fixed_frame, source_id="fixed")
    turret = detector.detect(turret_frame, source_id="turret") if turret_frame is not None else None
    return TrackingCameraDetections(
        fixed=fixed,
        fixed_summary=_source_detection_summary(
            source_id="fixed",
            detections=fixed.detections,
            policy=policy,
        ),
        turret=turret,
        turret_summary=(
            _source_detection_summary(
                source_id="turret",
                detections=turret.detections,
                policy=policy,
            )
            if turret is not None
            else "turret unavailable"
        ),
    )


def _fit_preview(frame_width: int, frame_height: int, region: Rect) -> Rect:
    frame_width = max(1, frame_width)
    frame_height = max(1, frame_height)
    scale = min(region.width / frame_width, region.height / frame_height)
    width = max(1, int(frame_width * scale))
    height = max(1, int(frame_height * scale))
    return Rect(
        x=region.x + (region.width - width) // 2,
        y=region.y + (region.height - height) // 2,
        width=width,
        height=height,
    )


def build_tracking_layout(
    *,
    fixed_frame_width: int,
    fixed_frame_height: int,
    turret_frame_width: int | None,
    turret_frame_height: int | None,
    window_width: int,
    window_height: int,
    panel_width: int,
) -> TrackingLayout:
    panel_width = min(max(220, panel_width), max(220, window_width - 1))
    panel_x = max(1, window_width - panel_width)
    content_width = panel_x

    if turret_frame_width is None or turret_frame_height is None:
        fixed_region = Rect(0, 0, content_width, window_height)
        return TrackingLayout(
            window_width=window_width,
            window_height=window_height,
            panel_x=panel_x,
            fixed=PreviewPlacement(
                region=fixed_region,
                preview=_fit_preview(fixed_frame_width, fixed_frame_height, fixed_region),
            ),
            turret=None,
        )

    fixed_height = window_height // 2
    turret_height = window_height - fixed_height
    fixed_region = Rect(0, 0, content_width, fixed_height)
    turret_region = Rect(0, fixed_height, content_width, turret_height)
    return TrackingLayout(
        window_width=window_width,
        window_height=window_height,
        panel_x=panel_x,
        fixed=PreviewPlacement(
            region=fixed_region,
            preview=_fit_preview(fixed_frame_width, fixed_frame_height, fixed_region),
        ),
        turret=PreviewPlacement(
            region=turret_region,
            preview=_fit_preview(turret_frame_width, turret_frame_height, turret_region),
        ),
    )


def handle_tracking_control(
    control: str,
    *,
    state: TrackingTestState,
    session: SessionLike | None,
    controller: TurretController,
) -> TrackingControlResult:
    if control == "quit":
        return TrackingControlResult(state=state, message="quit requested", should_exit=True)

    if control == "zone_calibration":
        return TrackingControlResult(
            state=state,
            message="opening zone calibration",
            should_exit=True,
            next_screen="zone_calibration",
        )

    if control == "arm":
        if session is not None:
            session.enable()
        return TrackingControlResult(
            state=TrackingTestState(armed=True, step_deg=state.step_deg),
            message="armed",
        )

    if control in {"disarm", "safe_stop"}:
        if session is not None:
            session.disable()
        controller.safe_stop()
        return TrackingControlResult(
            state=TrackingTestState(armed=False, step_deg=state.step_deg),
            message="disarmed and safe-stopped",
        )

    if control == "status":
        status = _read_controller_status(controller)
        return TrackingControlResult(state=state, message=status)

    gated_controls = {"tilt_up", "tilt_down", "pan_left", "pan_right", "fire"}
    if control in gated_controls and not state.armed:
        return TrackingControlResult(
            state=state,
            message="controller is disarmed; press arm first",
        )

    if control == "tilt_up":
        controller.apply_tracking_delta(0.0, -state.step_deg)
        return TrackingControlResult(state=state, message=f"tilt up {state.step_deg:.1f} deg")
    if control == "tilt_down":
        controller.apply_tracking_delta(0.0, state.step_deg)
        return TrackingControlResult(state=state, message=f"tilt down {state.step_deg:.1f} deg")
    if control == "pan_left":
        controller.apply_tracking_delta(-state.step_deg, 0.0)
        return TrackingControlResult(state=state, message=f"pan left {state.step_deg:.1f} deg")
    if control == "pan_right":
        controller.apply_tracking_delta(state.step_deg, 0.0)
        return TrackingControlResult(state=state, message=f"pan right {state.step_deg:.1f} deg")
    if control == "fire":
        controller.fire()
        return TrackingControlResult(state=state, message="fired")

    return TrackingControlResult(state=state, message="")


def _read_controller_status(controller: TurretController) -> str:
    status_method = getattr(controller, "status", None)
    if status_method is None:
        return "status unavailable in dry-run mode"
    response = status_method()
    payload = getattr(response, "payload", {})
    return f"status {payload}"


def parse_args(argv: list[str] | None = None) -> TrackingTestConfig:
    parser = argparse.ArgumentParser(description="Cat Cannon tracking test UI with teleop controls")
    parser.add_argument(
        "--fixed-camera",
        default="/dev/fixed_cam",
        help="Fixed camera device path or index (e.g. /dev/fixed_cam or 0)",
    )
    parser.add_argument(
        "--turret-camera",
        default="/dev/turret_cam",
        help="Turret camera device path or index; use 'none' to disable",
    )
    parser.add_argument("--port", help="RP2040 serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--fire-ms", type=int, default=120)
    parser.add_argument("--step-deg", type=float, default=3.0)
    parser.add_argument("--config", default="configs/app.example.yaml", help="System config path")
    parser.add_argument("--zones", default="configs/zones.yaml", help="Counter zones config path")
    parser.add_argument("--yolo-model", default="", help="YOLO model path (default: bundled yolo11s.pt)")
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Optional inference device, e.g. cpu or 0",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--detect-interval", type=int, default=3, help="Run YOLO every N frames (higher = faster UI)")
    parser.add_argument(
        "--live-controller",
        action="store_true",
        help="Use the RP2040 instead of dry-run mode",
    )
    parser.add_argument("--arm-on-start", action="store_true", help="Start armed")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--panel-width", type=int, default=280)
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args(argv)

    def _parse_camera(value: str) -> int | str:
        try:
            return int(value)
        except ValueError:
            return value

    turret = args.turret_camera
    turret_parsed: int | str | None = None
    if turret is not None and turret.lower() not in ("none", "-1"):
        turret_parsed = _parse_camera(turret)

    return TrackingTestConfig(
        fixed_camera=_parse_camera(args.fixed_camera),
        turret_camera=turret_parsed,
        port=args.port,
        baudrate=args.baudrate,
        fire_ms=args.fire_ms,
        step_deg=args.step_deg,
        config_path=args.config,
        zones_path=args.zones,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_imgsz=args.yolo_imgsz,
        detect_interval=args.detect_interval,
        live_controller=bool(args.live_controller),
        arm_on_start=bool(args.arm_on_start),
        window_width=args.window_width,
        window_height=args.window_height,
        panel_width=args.panel_width,
        fullscreen=bool(args.fullscreen),
    )


def _open_camera(cv2, device: int | str):
    from cat_cannon.adapters.camera import open_camera
    return open_camera(cv2, device)


def _resolve_port(port: str | None) -> str:
    try:
        return port or autodetect_port()
    except RP2040DiscoveryError as exc:
        raise SystemExit(str(exc)) from exc


def _load_zones(path: str | Path) -> tuple[Path, list[CounterZone], str]:
    zones_path = resolve_zones_path(path)
    if not zones_path.exists():
        return zones_path, [], f"No zones file found at {path}; tracking will stay unconfirmed."
    return zones_path, load_counter_zones(zones_path), f"Loaded zones from {zones_path}"


def _build_buttons(config: TrackingTestConfig) -> list[UiButton]:
    panel_x = config.window_width - config.panel_width
    margin = 16
    gap = 10
    full_width = config.panel_width - margin * 2
    half_width = (full_width - gap) // 2
    row_height = 40
    y = 112

    def full(key: str, label: str, row: int) -> UiButton:
        y1 = y + row * (row_height + gap)
        return UiButton(
            key=key,
            label=label,
            x1=panel_x + margin,
            y1=y1,
            x2=panel_x + margin + full_width,
            y2=y1 + row_height,
        )

    def half(key: str, label: str, row: int, column: int) -> UiButton:
        y1 = y + row * (row_height + gap)
        x1 = panel_x + margin + column * (half_width + gap)
        return UiButton(
            key=key,
            label=label,
            x1=x1,
            y1=y1,
            x2=x1 + half_width,
            y2=y1 + row_height,
        )

    return [
        full("zone_calibration", "Zone Calibrator", 0),
        half("arm", "Arm", 1, 0),
        half("disarm", "Disarm", 1, 1),
        full("tilt_up", "Tilt Up", 2),
        half("pan_left", "Pan Left", 3, 0),
        half("pan_right", "Pan Right", 3, 1),
        full("tilt_down", "Tilt Down", 4),
        half("fire", "Fire", 5, 0),
        half("safe_stop", "Safe Stop", 5, 1),
        full("quit", "Quit", 6),
    ]


def _control_from_key(key: int) -> str | None:
    keymap = {
        ord("q"): "quit",
        ord("z"): "zone_calibration",
        ord("e"): "arm",
        ord("x"): "safe_stop",
        ord("w"): "tilt_up",
        ord("s"): "tilt_down",
        ord("a"): "pan_left",
        ord("d"): "pan_right",
        ord(" "): "fire",
        ord("p"): "status",
    }
    return keymap.get(key)


def _draw_button(cv2, canvas, button: UiButton, *, active: bool = False) -> None:
    fill = (40, 120, 80) if active else (70, 70, 70)
    cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), fill, -1)
    cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), (220, 220, 220), 2)
    cv2.putText(
        canvas,
        button.label,
        (button.x1 + 10, button.y1 + 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )


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


def _draw_zones(cv2, frame, zones: list[CounterZone]):
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


def _annotate_camera(cv2, frame, title: str, status_lines: list[str]):
    height, width = frame.shape[:2]
    cv2.line(frame, (width // 2 - 20, height // 2), (width // 2 + 20, height // 2), (0, 255, 0), 1)
    cv2.line(frame, (width // 2, height // 2 - 20), (width // 2, height // 2 + 20), (0, 255, 0), 1)
    cv2.putText(frame, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    for index, line in enumerate(status_lines):
        cv2.putText(
            frame,
            line,
            (10, 52 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
        )
    return frame


def _paste_frame(cv2, canvas, frame, placement: PreviewPlacement) -> None:
    resized = cv2.resize(frame, (placement.preview.width, placement.preview.height))
    y1 = placement.preview.y
    y2 = placement.preview.y + placement.preview.height
    x1 = placement.preview.x
    x2 = placement.preview.x + placement.preview.width
    canvas[y1:y2, x1:x2] = resized
    cv2.rectangle(
        canvas,
        (placement.region.x, placement.region.y),
        (
            placement.region.x + placement.region.width - 1,
            placement.region.y + placement.region.height - 1,
        ),
        (70, 70, 70),
        1,
    )


def _status_lines(
    *,
    state: TrackingTestState,
    live_controller: bool,
    detection_summary: str,
    step_result: SupervisorStepResult,
) -> list[str]:
    zone = step_result.active_zone_id or "-"
    target = step_result.candidate_track_id or "-"
    correction = "correction=-"
    if step_result.correction is not None:
        correction = (
            f"pan={step_result.correction.pan_delta:.2f} "
            f"tilt={step_result.correction.tilt_delta:.2f}"
        )
    return [
        f"mode={'live' if live_controller else 'dry-run'} armed={state.armed}",
        f"state={step_result.state.value} zone={zone} target={target}",
        f"locked={step_result.aim_locked} fire={step_result.fire_commanded}",
        correction,
        detection_summary,
    ]


def _draw_trace_log(cv2, canvas, *, layout: TrackingLayout, buttons: list[UiButton], trace_log: TraceLog) -> None:
    if not trace_log.lines:
        return
    x = layout.panel_x + 16
    start_y = buttons[-1].y2 + 28 if buttons else 112
    stop_y = layout.window_height - 94
    if start_y >= stop_y:
        return
    cv2.putText(
        canvas,
        "Trace",
        (x, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )
    max_lines = max(0, (stop_y - start_y - 12) // 18)
    for index, line in enumerate(trace_log.lines[-max_lines:]):
        cv2.putText(
            canvas,
            line[:42],
            (x, start_y + 22 + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (230, 230, 230),
            1,
        )


def _render_ui(
    *,
    cv2,
    fixed_frame,
    turret_frame,
    layout: TrackingLayout,
    zones: list[CounterZone],
    fixed_detections: list[Detection],
    turret_detections: list[Detection],
    buttons: list[UiButton],
    state: TrackingTestState,
    live_controller: bool,
    fixed_detection_summary: str,
    turret_detection_summary: str,
    step_result: SupervisorStepResult,
    status_message: str,
    zones_path: Path,
    trace_log: TraceLog,
):
    canvas = fixed_frame[0:1, 0:1].copy()
    canvas[:] = (24, 24, 24)
    canvas = cv2.resize(canvas, (layout.window_width, layout.window_height))

    fixed_display = fixed_frame.copy()
    fixed_display = _draw_zones(cv2, fixed_display, zones)
    fixed_display = _draw_detections(cv2, fixed_display, fixed_detections)
    fixed_display = _annotate_camera(
        cv2,
        fixed_display,
        "fixed tracking camera",
        _status_lines(
            state=state,
            live_controller=live_controller,
            detection_summary=fixed_detection_summary,
            step_result=step_result,
        ),
    )
    _paste_frame(cv2, canvas, fixed_display, layout.fixed)

    if turret_frame is not None and layout.turret is not None:
        turret_display = turret_frame.copy()
        turret_display = _draw_detections(cv2, turret_display, turret_detections)
        turret_display = _annotate_camera(
            cv2,
            turret_display,
            "turret camera",
            [
                turret_detection_summary,
                "teleop: e arm  x safe  wasd move  space fire",
                "z zone calibration  q quit",
            ],
        )
        _paste_frame(cv2, canvas, turret_display, layout.turret)

    cv2.rectangle(
        canvas,
        (layout.panel_x, 0),
        (layout.window_width, layout.window_height),
        (38, 38, 38),
        -1,
    )
    cv2.putText(
        canvas,
        "Tracking Test",
        (layout.panel_x + 16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        f"Armed: {'yes' if state.armed else 'no'}",
        (layout.panel_x + 16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (220, 220, 220),
        1,
    )
    cv2.putText(
        canvas,
        f"Step: {state.step_deg:.1f} deg",
        (layout.panel_x + 16, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (220, 220, 220),
        1,
    )
    for button in buttons:
        active = button.key == "arm" and state.armed
        _draw_button(cv2, canvas, button, active=active)
    _draw_trace_log(cv2, canvas, layout=layout, buttons=buttons, trace_log=trace_log)

    cv2.putText(
        canvas,
        status_message[:42],
        (layout.panel_x + 16, layout.window_height - 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (235, 235, 235),
        1,
    )
    cv2.putText(
        canvas,
        f"Zones: {zones_path}",
        (layout.panel_x + 16, layout.window_height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        canvas,
        "keys: z/e/x/wasd/space/p/q",
        (layout.panel_x + 16, layout.window_height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
    )
    return canvas


def run_tracking_test_screen(config: TrackingTestConfig) -> ScreenName | None:
    cv2 = _require_cv2()
    system_config = load_system_config(config.config_path)
    zones_path, zones, status_message = _load_zones(config.zones_path)
    detector = UltralyticsYoloDetector.open(
        policy=system_config.detection_policy,
        runtime=YoloRuntimeConfig(
            model_path=config.yolo_model,
            device=config.yolo_device,
            imgsz=config.yolo_imgsz,
        ),
    )

    controller: TurretController = NullTurretController()
    session: ControllerSession | None = None
    if config.live_controller:
        port = _resolve_port(config.port)
        controller = RP2040SerialController.open(
            port=port,
            baudrate=config.baudrate,
            fire_pulse_ms=config.fire_ms,
        )
        session = ControllerSession(controller=controller)

    supervisor = SupervisorLoop(config=system_config, zones=zones, controller=controller)
    fixed_camera = _open_camera(cv2, config.fixed_camera)
    turret_camera = (
        _open_camera(cv2, config.turret_camera) if config.turret_camera is not None else None
    )

    state = TrackingTestState(armed=config.arm_on_start, step_deg=config.step_deg)
    buttons = _build_buttons(config)
    trace_log = TraceLog()
    should_exit = False
    next_screen: ScreenName | None = None
    trace_log.add(f"starting fixed={config.fixed_camera} turret={config.turret_camera}")

    def apply_control(control: str, *, source: str) -> None:
        nonlocal state, status_message, should_exit, next_screen
        trace_log.add(f"{source} -> {control}")
        try:
            result = handle_tracking_control(
                control,
                state=state,
                session=session,
                controller=controller,
            )
        except RP2040ProtocolError as exc:
            status_message = f"controller error: {exc}"
            trace_log.add(status_message)
            return
        state = result.state
        if result.message:
            status_message = result.message
            trace_log.add(f"response: {result.message}")
        if config.live_controller and not result.should_exit and control != "status":
            try:
                trace_log.add(_read_controller_status(controller))
            except RP2040ProtocolError as exc:
                trace_log.add(f"status error: {exc}")
        should_exit = should_exit or result.should_exit
        next_screen = result.next_screen or next_screen

    def on_mouse(event: int, x: int, y: int, _flags: int, _userdata: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in buttons:
            if button.contains(x, y):
                apply_control(button.key, source=f"mouse {x},{y}")
                return

    window_name = "cat-cannon-tracking-test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.window_width, config.window_height)
    if config.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        if session is not None:
            session.start()
            trace_log.add("controller session started")
            if state.armed:
                session.enable()
                trace_log.add("controller armed on start")
            else:
                session.disable()
                trace_log.add("controller disarmed on start")

        frame_counter = 0
        camera_detections = None

        while True:
            ok_fixed, fixed_frame = fixed_camera.read()
            if not ok_fixed:
                raise SystemExit(f"Failed to read from fixed camera {config.fixed_camera}")

            turret_frame = None
            if turret_camera is not None:
                ok_turret, turret_frame = turret_camera.read()
                if not ok_turret:
                    raise SystemExit(
                        f"Failed to read from turret camera {config.turret_camera}"
                    )

            if camera_detections is None or frame_counter % config.detect_interval == 0:
                camera_detections = detect_tracking_cameras(
                    detector=detector,
                    fixed_frame=fixed_frame,
                    turret_frame=turret_frame,
                    policy=system_config.detection_policy,
                )
            frame_counter += 1
            perception_frame = camera_detections.fixed
            turret_perception = camera_detections.turret
            step_result = supervisor.process_frame(
                detections=perception_frame.detections,
                frame_width=perception_frame.width,
                frame_height=perception_frame.height,
                armed=state.armed,
                turret_detections=(
                    turret_perception.detections if turret_perception is not None else None
                ),
                turret_frame_width=(
                    turret_perception.width if turret_perception is not None else None
                ),
                turret_frame_height=(
                    turret_perception.height if turret_perception is not None else None
                ),
            )
            layout = build_tracking_layout(
                fixed_frame_width=fixed_frame.shape[1],
                fixed_frame_height=fixed_frame.shape[0],
                turret_frame_width=turret_frame.shape[1] if turret_frame is not None else None,
                turret_frame_height=turret_frame.shape[0] if turret_frame is not None else None,
                window_width=config.window_width,
                window_height=config.window_height,
                panel_width=config.panel_width,
            )
            canvas = _render_ui(
                cv2=cv2,
                fixed_frame=fixed_frame,
                turret_frame=turret_frame,
                layout=layout,
                zones=zones,
                fixed_detections=perception_frame.detections,
                turret_detections=(
                    camera_detections.turret.detections if camera_detections.turret is not None else []
                ),
                buttons=buttons,
                state=state,
                live_controller=config.live_controller,
                fixed_detection_summary=camera_detections.fixed_summary,
                turret_detection_summary=camera_detections.turret_summary,
                step_result=step_result,
                status_message=status_message,
                zones_path=zones_path,
                trace_log=trace_log,
            )
            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(16) & 0xFF
            if key != 255:
                control = _control_from_key(key)
                if control is not None:
                    label = chr(key) if 32 <= key <= 126 else str(key)
                    apply_control(control, source=f"key {label}")
            if should_exit:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if turret_camera is not None:
            turret_camera.release()
        fixed_camera.release()
        cv2.destroyAllWindows()
        if session is not None:
            session.stop()

    return next_screen


def run_tracking_test_with_navigation(config: TrackingTestConfig) -> None:
    current: ScreenName | None = "tracking_test"
    while current is not None:
        if current == "tracking_test":
            current = run_tracking_test_screen(config)
            continue

        from cat_cannon.app.calibrate_zones import CalibrationConfig, run_calibration_screen

        current = run_calibration_screen(
            CalibrationConfig(
                camera=config.fixed_camera,
                output_path=config.zones_path,
                zone_prefix="zone",
                window_width=config.window_width,
                window_height=config.window_height,
                panel_width=min(config.panel_width, 260),
                fullscreen=config.fullscreen,
                detect=True,
                yolo_model=config.yolo_model,
                yolo_device=config.yolo_device,
                yolo_imgsz=config.yolo_imgsz,
                config_path=config.config_path,
            )
        )


def main(argv: list[str] | None = None) -> None:
    run_tracking_test_with_navigation(parse_args(argv))


if __name__ == "__main__":
    main()
