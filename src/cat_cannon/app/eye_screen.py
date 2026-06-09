"""Main Eye screen — animated glowing eye that follows turret movement."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cat_cannon.adapters.ultralytics_yolo import DEFAULT_YOLO_IMGSZ
from cat_cannon.config import DEFAULT_YOLOE_PROMPTS, EventRecordingConfig, YoloPrompt
from cat_cannon.domain.models import SupervisorState

ScreenName = Literal["eye", "zone_calibration", "tracking_test"]


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "opencv-python is required for the eye screen. Install with: "
            "pip install opencv-python"
        ) from exc
    return cv2


@dataclass
class EyeConfig:
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
    collect_cat_dataset: bool = False
    dataset_dir: str = "data/cat_training"
    dataset_sample_hz: float = 1.0
    event_recording: EventRecordingConfig = field(default_factory=EventRecordingConfig)


@dataclass
class EyeState:
    armed: bool = False
    # Eye gaze position (normalized -1..1)
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    # Target gaze for smooth interpolation
    target_gaze_x: float = 0.0
    target_gaze_y: float = 0.0
    # Tracking state
    mode: Literal["idle", "alert", "tracking", "firing"] = "idle"
    # Persist detection timing for expiry
    last_detected: bool = False
    last_detection_time: float = 0.0
    # Random look timing
    next_look_time: float = field(default_factory=lambda: time.time() + random.uniform(3.0, 8.0))
    # Blink
    blink_until: float = 0.0


def build_eye_dataset_recorder(config: EyeConfig):
    if not config.collect_cat_dataset:
        return None
    from cat_cannon.app.yolo_dataset import CatDatasetRecorder

    return CatDatasetRecorder(
        root=config.dataset_dir,
        sample_interval_s=1.0 / max(0.001, config.dataset_sample_hz),
    )


def _lerp(current: float, target: float, speed: float) -> float:
    diff = target - current
    if abs(diff) < 0.005:
        return target
    return current + diff * speed


def _draw_eye(cv2, canvas: np.ndarray, state: EyeState, w: int, h: int) -> None:
    """Draw the animated glowing eye on the canvas."""
    cx, cy = w // 2, h // 2
    # Eye dimensions
    eye_radius_x = int(min(w, h) * 0.32)
    eye_radius_y = int(min(w, h) * 0.18)

    # Color based on mode
    if state.mode == "idle":
        glow_color = (0, 255, 80)  # Green
        iris_color = (0, 200, 60)
        pupil_color = (0, 80, 20)
    elif state.mode == "alert":
        glow_color = (0, 255, 255)  # Yellow
        iris_color = (0, 220, 220)
        pupil_color = (0, 100, 100)
    elif state.mode == "tracking":
        glow_color = (0, 200, 255)  # Orange-yellow
        iris_color = (0, 180, 240)
        pupil_color = (0, 80, 120)
    else:  # firing
        glow_color = (0, 0, 255)  # Red
        iris_color = (0, 0, 200)
        pupil_color = (0, 0, 80)

    # Blink check
    now = time.time()
    blink_factor = 1.0
    if now < state.blink_until:
        remaining = state.blink_until - now
        if remaining > 0.08:
            blink_factor = 0.05  # closed
        else:
            blink_factor = remaining / 0.08  # opening

    # Outer glow (multiple blurred ellipses)
    for i in range(3):
        alpha = 0.15 - i * 0.04
        size_mult = 1.0 + i * 0.12
        overlay = canvas.copy()
        cv2.ellipse(
            overlay,
            (cx, cy),
            (int(eye_radius_x * size_mult), int(eye_radius_y * size_mult * blink_factor)),
            0, 0, 360,
            glow_color,
            -1,
        )
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Eye white (dark sclera for sci-fi look)
    sclera_color = (15, 15, 15)
    cv2.ellipse(
        canvas,
        (cx, cy),
        (eye_radius_x, int(eye_radius_y * blink_factor)),
        0, 0, 360,
        sclera_color,
        -1,
    )
    # Sclera border glow
    cv2.ellipse(
        canvas,
        (cx, cy),
        (eye_radius_x, int(eye_radius_y * blink_factor)),
        0, 0, 360,
        glow_color,
        2,
    )

    if blink_factor < 0.2:
        return  # Eye closed, skip iris/pupil

    # Iris position based on gaze
    max_iris_offset_x = eye_radius_x * 0.45
    max_iris_offset_y = eye_radius_y * 0.35 * blink_factor
    iris_cx = int(cx + state.gaze_x * max_iris_offset_x)
    iris_cy = int(cy + state.gaze_y * max_iris_offset_y)
    iris_radius = int(min(eye_radius_x, eye_radius_y * blink_factor) * 0.55)

    # Iris ring
    cv2.circle(canvas, (iris_cx, iris_cy), iris_radius, iris_color, -1)

    # Iris detail rings
    for i in range(3):
        r = int(iris_radius * (0.9 - i * 0.15))
        shade = tuple(max(0, c - 30 * (i + 1)) for c in iris_color)
        cv2.circle(canvas, (iris_cx, iris_cy), r, shade, 1)

    # Pupil
    pupil_radius = int(iris_radius * 0.45)
    cv2.circle(canvas, (iris_cx, iris_cy), pupil_radius, pupil_color, -1)

    # Pupil highlight (small white dot)
    highlight_x = iris_cx - int(pupil_radius * 0.3)
    highlight_y = iris_cy - int(pupil_radius * 0.3)
    cv2.circle(canvas, (highlight_x, highlight_y), max(2, pupil_radius // 4), (200, 200, 200), -1)

    # Second smaller highlight
    h2_x = iris_cx + int(pupil_radius * 0.25)
    h2_y = iris_cy + int(pupil_radius * 0.2)
    cv2.circle(canvas, (h2_x, h2_y), max(1, pupil_radius // 6), (150, 150, 150), -1)


def _draw_circular_button(
    cv2,
    canvas: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    color: tuple,
    label: str,
    text_color: tuple = (255, 255, 255),
) -> None:
    """Draw a circular button with label."""
    cv2.circle(canvas, (center_x, center_y), radius, color, -1)
    cv2.circle(canvas, (center_x, center_y), radius, (200, 200, 200), 1)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    cv2.putText(canvas, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1)


def _point_in_circle(px: int, py: int, cx: int, cy: int, r: int) -> bool:
    return (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2


def run_eye_screen(config: EyeConfig) -> ScreenName | None:
    """Run the main eye screen. Returns next screen or None to exit."""
    import threading

    cv2 = _require_cv2()

    state = EyeState(armed=config.arm_on_start)
    next_screen: ScreenName | None = None

    # Button layout
    btn_radius = 28
    btn_margin = 45
    zones_btn_x = config.window_width - btn_margin
    zones_btn_y = config.window_height - btn_margin
    arm_btn_x = btn_margin
    arm_btn_y = config.window_height - btn_margin

    # Show window IMMEDIATELY before any heavy loading
    window_name = "Cat Cannon"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    if config.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, config.window_width, config.window_height)

    def on_mouse(event, x, y, flags, param):
        nonlocal state, next_screen
        if event == cv2.EVENT_LBUTTONDOWN:
            if _point_in_circle(x, y, zones_btn_x, zones_btn_y, btn_radius):
                next_screen = "zone_calibration"
            elif _point_in_circle(x, y, arm_btn_x, arm_btn_y, btn_radius):
                state.armed = not state.armed
                # Sync arm state to RP2040 via session
                _session = _bg_resources.get("session")
                if _session is not None:
                    try:
                        if state.armed:
                            _session.enable()
                        else:
                            _session.disable()
                    except Exception:
                        pass

    cv2.setMouseCallback(window_name, on_mouse)

    # --- Background resource loading ---
    # Load heavy resources in a thread so the eye animation stays responsive
    # and GNOME doesn't think the window is frozen.
    _bg_resources: dict = {}
    _bg_done = threading.Event()

    # Shared self-healing heartbeat state (populated lazily by the detection
    # loop once a turret camera + live controller exist; watched by a separate
    # health-monitor thread so a hung detection loop can still be detected).
    _hb: dict = {
        "cfg": None,
        "live": None,
        "watchdog": None,
        "active": False,
        "pending": None,
        "next_move_t": 0.0,
        "dir": 1,
    }
    _fault_event = threading.Event()
    _fault_reason: dict = {"reason": None}

    def _load_resources():
        from cat_cannon.adapters.camera import open_camera
        from cat_cannon.adapters.rp2040_serial import RP2040SerialController
        from cat_cannon.adapters.ultralytics_yolo import UltralyticsYoloDetector, YoloRuntimeConfig
        from cat_cannon.app.controller_session import ControllerSession
        from cat_cannon.app.event_video import build_event_video_recorder
        from cat_cannon.app.supervisor import SupervisorLoop
        from cat_cannon.app.tracking_test import _resolve_config_path
        from cat_cannon.config import load_counter_zones, load_system_config

        resolved_config_path = _resolve_config_path(config.config_path)
        system_config = load_system_config(resolved_config_path)
        _bg_resources["system_config"] = system_config
        try:
            from cat_cannon.config import load_heartbeat_config
            _bg_resources["heartbeat_config"] = load_heartbeat_config(resolved_config_path)
        except Exception:
            from cat_cannon.config import HeartbeatConfig
            _bg_resources["heartbeat_config"] = HeartbeatConfig()

        # Load zones
        try:
            from cat_cannon.app.tracking_test import resolve_zones_path
            zones_path = resolve_zones_path(config.zones_path)
            _bg_resources["zones"] = load_counter_zones(zones_path)
        except Exception:
            _bg_resources["zones"] = []

        # Open fixed camera (for zone detection)
        fixed_camera = None
        try:
            fixed_camera = open_camera(
                cv2,
                config.fixed_camera,
                width=config.fixed_camera_width,
                height=config.fixed_camera_height,
                fps=config.camera_fps,
                rotate_180=False,
            )
        except (Exception, SystemExit):
            pass
        _bg_resources["fixed_camera"] = fixed_camera

        # Open turret camera
        turret_camera = None
        if config.turret_camera:
            try:
                turret_camera = open_camera(
                    cv2,
                    config.turret_camera,
                    width=config.turret_camera_width,
                    height=config.turret_camera_height,
                    fps=config.camera_fps,
                    rotate_180=config.turret_rotate_180,
                )
            except (Exception, SystemExit):
                pass
        _bg_resources["turret_camera"] = turret_camera

        # Open controller
        controller = None
        session = None
        if config.live_controller:
            try:
                ctrl = RP2040SerialController.open(port=config.port, baudrate=config.baudrate)
                session = ControllerSession(
                    controller=ctrl,
                    servo_limits=system_config.servo_limits,
                )
                session.start()
                if config.arm_on_start:
                    session.enable()
                controller = ctrl
            except Exception:
                pass
        _bg_resources["controller"] = controller
        _bg_resources["session"] = session

        # Open detector (heaviest — TensorRT init)
        detector = None
        try:
            detector = UltralyticsYoloDetector.open(
                policy=system_config.detection_policy,
                runtime=YoloRuntimeConfig(
                    model_path=config.yolo_model,
                    device=config.yolo_device,
                    imgsz=config.yolo_imgsz,
                    detector=config.yolo_detector,
                    prompts=config.yolo_prompts,
                ),
            )
        except Exception:
            pass
        _bg_resources["detector"] = detector

        # Create supervisor (same code path as tracking test)
        zones = _bg_resources["zones"]
        if controller is not None:
            _bg_resources["supervisor"] = SupervisorLoop(
                config=system_config, zones=zones, controller=controller
            )
        else:
            from cat_cannon.adapters.controller import NullTurretController
            _bg_resources["supervisor"] = SupervisorLoop(
                config=system_config, zones=zones, controller=NullTurretController()
            )

        _bg_resources["event_recorder"] = (
            build_event_video_recorder(config.event_recording, fps=config.camera_fps)
            if turret_camera is not None
            else None
        )
        _bg_resources["dataset_recorder"] = build_eye_dataset_recorder(config)

        _bg_done.set()

    loader_thread = threading.Thread(target=_load_resources, daemon=True)
    loader_thread.start()

    # These get set once background loading completes
    system_config = None
    fixed_camera = None
    turret_camera = None
    controller = None

    # Shared state between detection thread and main loop (protected by lock)
    _detect_lock = threading.Lock()
    _latest_step_result = None
    _latest_gaze = None  # (gaze_x, gaze_y) or None
    _detect_running = True

    def _detection_loop():
        """Background thread: runs detection + supervisor at GPU speed."""
        nonlocal _latest_step_result, _latest_gaze, _detect_running
        frame_counter = 0
        last_fixed_perception = None
        fixed_detect_interval = max(1, int(config.detect_interval))

        while _detect_running:
            if not _bg_done.is_set():
                time.sleep(0.05)
                continue

            _detector = _bg_resources.get("detector")
            _supervisor = _bg_resources.get("supervisor")
            _fixed_cam = _bg_resources.get("fixed_camera")
            _turret_cam = _bg_resources.get("turret_camera")
            _sys_config = _bg_resources.get("system_config")
            _dataset_recorder = _bg_resources.get("dataset_recorder")

            if _detector is None or _supervisor is None:
                time.sleep(0.05)
                continue

            # --- Heartbeat: lazy init + liveness beat ---
            _hb_cfg = _bg_resources.get("heartbeat_config")
            if (
                _hb["cfg"] is None
                and _hb_cfg is not None
                and _hb_cfg.enabled
                and _turret_cam is not None
                and _bg_resources.get("controller") is not None
            ):
                from cat_cannon.app.heartbeat import Liveness, MotionConfig, MotionWatchdog
                _hb["cfg"] = _hb_cfg
                _hb["live"] = Liveness(_hb_cfg.liveness_timeout_s)
                _hb["watchdog"] = MotionWatchdog(
                    MotionConfig(
                        flow_min_magnitude_px=_hb_cfg.flow_min_magnitude_px,
                        direction_dot_min=_hb_cfg.direction_dot_min,
                        max_consecutive_failures=_hb_cfg.max_consecutive_motion_failures,
                        pan_flow_sign=_hb_cfg.pan_flow_sign,
                        tilt_flow_sign=_hb_cfg.tilt_flow_sign,
                    )
                )
                _hb["next_move_t"] = time.monotonic() + _hb_cfg.move_interval_s
                _hb["active"] = True
            if _hb["live"] is not None:
                _hb["live"].beat()

            # Fixed camera: detect at interval for zone confirmation
            fixed_detection_updated = False
            if _fixed_cam is not None and frame_counter % fixed_detect_interval == 0:
                ok_fixed, fixed_frame = _fixed_cam.read()
                if ok_fixed:
                    last_fixed_perception = _detector.detect(fixed_frame, source_id="fixed")
                    fixed_detection_updated = True
                    if _dataset_recorder is not None and _sys_config is not None:
                        try:
                            _dataset_recorder.maybe_record(
                                cv2=cv2,
                                source_id="fixed",
                                frame=fixed_frame,
                                detections=last_fixed_perception.detections,
                                policy=_sys_config.detection_policy,
                            )
                        except Exception:
                            pass

            fixed_detections = last_fixed_perception.detections if last_fixed_perception else []
            fixed_width = (
                last_fixed_perception.width
                if last_fixed_perception
                else config.fixed_camera_width
            )
            fixed_height = (
                last_fixed_perception.height
                if last_fixed_perception
                else config.fixed_camera_height
            )

            # Turret camera: detect every frame for responsive tracking
            turret_detections = None
            turret_width = None
            turret_height = None
            turret_perception = None
            turret_gray = None
            if _turret_cam is not None:
                ok_turret, turret_frame = _turret_cam.read()
                if ok_turret:
                    turret_perception = _detector.detect(turret_frame, source_id="turret")
                    turret_detections = turret_perception.detections
                    turret_width = turret_perception.width
                    turret_height = turret_perception.height
                    if _hb["active"]:
                        try:
                            from cat_cannon.adapters.optical_flow import to_gray
                            turret_gray = to_gray(cv2, turret_frame)
                        except Exception:
                            turret_gray = None
                    if _dataset_recorder is not None and _sys_config is not None:
                        try:
                            _dataset_recorder.maybe_record(
                                cv2=cv2,
                                source_id="turret",
                                frame=turret_frame,
                                detections=turret_perception.detections,
                                policy=_sys_config.detection_policy,
                            )
                        except Exception:
                            pass

            step_result = _supervisor.process_frame(
                detections=fixed_detections,
                frame_width=fixed_width,
                frame_height=fixed_height,
                armed=state.armed,
                turret_detections=turret_detections,
                turret_frame_width=turret_width,
                turret_frame_height=turret_height,
                fixed_detections_fresh=fixed_detection_updated,
                track_people=True,
            )

            _event_recorder = _bg_resources.get("event_recorder")
            if _event_recorder is not None and turret_perception is not None:
                try:
                    finalized = _event_recorder.update(
                        cv2=cv2,
                        turret_frame=turret_frame,
                        step_result=step_result,
                    )
                    if finalized is not None:
                        print(f"[event-video] saved {finalized.video_path}", flush=True)
                except Exception:
                    print("[event-video] recorder update failed", flush=True)

            # Compute gaze from turret detection
            gaze = None
            if turret_perception is not None and turret_detections and _sys_config is not None:
                policy = _sys_config.detection_policy
                cats = [
                    d for d in turret_detections
                    if d.label == policy.cat_class
                    and d.confidence >= policy.cat_confidence_threshold
                ]
                people = [
                    d for d in turret_detections
                    if d.label == policy.person_class
                    and d.confidence >= policy.person_confidence_threshold
                ]
                gaze_target = None
                if cats:
                    gaze_target = max(cats, key=lambda d: d.bbox.width * d.bbox.height)
                elif people:
                    gaze_target = max(people, key=lambda d: d.bbox.width * d.bbox.height)
                if gaze_target is not None:
                    bbox = gaze_target.bbox
                    target_cx = bbox.x + bbox.width / 2.0
                    target_cy = bbox.y + bbox.height / 2.0
                    gaze = (
                        -((target_cx / turret_perception.width - 0.5) * 2.0),
                        (target_cy / turret_perception.height - 0.5) * 2.0,
                    )

            with _detect_lock:
                _latest_step_result = step_result
                _latest_gaze = gaze

            # --- Heartbeat deliberate move + optical-flow confirmation ---
            if _hb["active"] and turret_gray is not None:
                _hb_now = time.monotonic()
                _pending = _hb["pending"]
                if _pending is not None:
                    if _hb_now >= _pending["deadline"]:
                        try:
                            from cat_cannon.adapters.optical_flow import mean_flow
                            _dx, _dy, _ = mean_flow(cv2, _pending["pre_gray"], turret_gray)
                            _hb["watchdog"].record(
                                pan_cmd=_pending["pan"],
                                tilt_cmd=_pending["tilt"],
                                dx=_dx,
                                dy=_dy,
                            )
                        except Exception:
                            pass
                        _hb["pending"] = None
                        _hb["next_move_t"] = _hb_now + _hb["cfg"].move_interval_s
                elif _hb_now >= _hb["next_move_t"] and state.mode == "idle":
                    _ctrl = _bg_resources.get("controller")
                    if _ctrl is not None:
                        _sign = _hb["dir"]
                        _pan = _hb["cfg"].move_pan_deg * _sign
                        _tilt = _hb["cfg"].move_tilt_deg * _sign
                        try:
                            _ctrl.apply_tracking_delta(_pan, _tilt)
                            _hb["pending"] = {
                                "pre_gray": turret_gray,
                                "pan": _pan,
                                "tilt": _tilt,
                                "deadline": _hb_now + _hb["cfg"].confirm_window_s,
                            }
                            _hb["dir"] = -_sign
                        except Exception:
                            _hb["next_move_t"] = _hb_now + _hb["cfg"].move_interval_s

            frame_counter += 1

    detect_thread = threading.Thread(target=_detection_loop, daemon=True)
    detect_thread.start()

    def _health_monitor():
        """Watch liveness + motion watchdog; on fault notify Discord once and
        signal the main loop to exit with the heartbeat sentinel code."""
        from cat_cannon.app.notify import post_discord_message

        while _detect_running and not _fault_event.is_set():
            time.sleep(0.5)
            if not _hb["active"]:
                continue
            _live = _hb["live"]
            _wd = _hb["watchdog"]
            reason = None
            if _live is not None and _live.is_stale():
                reason = "hang"
            elif _wd is not None and _wd.tripped:
                reason = "no_motion"
            if reason is not None:
                _fault_reason["reason"] = reason
                _cfg = _hb["cfg"]
                try:
                    post_discord_message(
                        _cfg.resolved_discord_webhook_url(),
                        f"\u26a0\ufe0f Cat Cannon missed heartbeat ({reason}) \u2014 restarting.",
                    )
                except Exception:
                    pass
                _fault_event.set()
                break

    health_thread = threading.Thread(target=_health_monitor, daemon=True)
    health_thread.start()

    try:
        while next_screen is None:
            now = time.time()

            # Self-healing heartbeat tripped -> leave the loop so cleanup runs
            # and the process exits with the sentinel code for the guardian.
            if _fault_event.is_set():
                break

            # --- Pick up background resources once ready ---
            if _bg_done.is_set() and system_config is None:
                system_config = _bg_resources.get("system_config")
                fixed_camera = _bg_resources.get("fixed_camera")
                turret_camera = _bg_resources.get("turret_camera")
                controller = _bg_resources.get("controller")

            # --- Read latest detection results (non-blocking) ---
            step_result = None
            with _detect_lock:
                step_result = _latest_step_result
                gaze = _latest_gaze

            if gaze is not None:
                state.target_gaze_x, state.target_gaze_y = gaze

            # --- Determine eye mode from supervisor state ---
            if step_result is not None:
                if step_result.state in (SupervisorState.FIRE, SupervisorState.COOLDOWN):
                    state.mode = "firing"
                    state.last_detected = True
                    state.last_detection_time = now
                elif step_result.state in (SupervisorState.TRACKING, SupervisorState.AIM_LOCK):
                    state.mode = "tracking"
                    state.last_detected = True
                    state.last_detection_time = now
                elif step_result.human_present or step_result.target_visible:
                    state.mode = "alert"
                    state.last_detected = True
                    state.last_detection_time = now
                else:
                    # No target — expire after 2s
                    if state.last_detected and (now - state.last_detection_time) > 2.0:
                        state.last_detected = False
                        state.mode = "idle"
                    elif not state.last_detected:
                        state.mode = "idle"
            else:
                state.mode = "idle"

            # --- Random looking when idle ---
            if state.mode == "idle":
                if now >= state.next_look_time:
                    state.target_gaze_x = random.uniform(-0.7, 0.7)
                    state.target_gaze_y = random.uniform(-0.4, 0.4)
                    state.next_look_time = now + random.uniform(30.0, 60.0)

                    # Move turret to follow the random look (single large move, no buzz).
                    # Skipped when the heartbeat owns idle motion (avoids double-moves).
                    if controller is not None and not _hb["active"]:
                        pan_delta = state.target_gaze_x * 15.0
                        tilt_delta = state.target_gaze_y * 10.0
                        try:
                            controller.apply_tracking_delta(pan_delta, tilt_delta)
                        except Exception:
                            pass

                    # Random blink sometimes
                    if random.random() < 0.3:
                        state.blink_until = now + 0.15

            # Also random blinks periodically
            if random.random() < 0.002:
                state.blink_until = now + 0.15

            # --- Smooth gaze interpolation ---
            speed = 0.12 if state.mode == "idle" else 0.25
            state.gaze_x = _lerp(state.gaze_x, state.target_gaze_x, speed)
            state.gaze_y = _lerp(state.gaze_y, state.target_gaze_y, speed)

            # --- Render ---
            canvas = np.zeros((config.window_height, config.window_width, 3), dtype=np.uint8)
            _draw_eye(cv2, canvas, state, config.window_width, config.window_height)

            # Draw zone button (bottom-right)
            _draw_circular_button(
                cv2, canvas, zones_btn_x, zones_btn_y, btn_radius,
                (80, 80, 80), "ZONE"
            )

            # Draw arm button (bottom-left)
            arm_color = (0, 0, 180) if state.armed else (60, 60, 60)
            arm_label = "ARMED" if state.armed else "SAFE"
            _draw_circular_button(
                cv2, canvas, arm_btn_x, arm_btn_y, btn_radius,
                arm_color, arm_label
            )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(16) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                next_screen = "zone_calibration"
            elif key == ord("a"):
                state.armed = not state.armed
                _session = _bg_resources.get("session")
                if _session is not None:
                    try:
                        if state.armed:
                            _session.enable()
                        else:
                            _session.disable()
                    except Exception:
                        pass

    except KeyboardInterrupt:
        pass
    finally:
        _detect_running = False
        detect_thread.join(timeout=2.0)
        try:
            health_thread.join(timeout=1.0)
        except Exception:
            pass
        _session = _bg_resources.get("session")
        if _session is not None:
            try:
                _session.stop()
            except Exception:
                pass
        elif controller is not None:
            try:
                controller.safe_stop()
                controller.close()
            except Exception:
                pass
        _event_recorder = _bg_resources.get("event_recorder")
        if _event_recorder is not None:
            try:
                finalized = _event_recorder.close()
                if finalized is not None:
                    print(f"[event-video] saved {finalized.video_path}", flush=True)
            except Exception:
                print("[event-video] recorder close failed", flush=True)
        if fixed_camera is not None:
            fixed_camera.release()
        if turret_camera is not None:
            turret_camera.release()
        cv2.destroyAllWindows()

    if _fault_event.is_set():
        import sys

        from cat_cannon.app.heartbeat import SENTINEL_EXIT_CODE

        reason = _fault_reason.get("reason") or "unknown"
        print(
            f"[heartbeat] fault ({reason}); exiting {SENTINEL_EXIT_CODE} for guardian",
            flush=True,
        )
        sys.exit(SENTINEL_EXIT_CODE)

    return next_screen
