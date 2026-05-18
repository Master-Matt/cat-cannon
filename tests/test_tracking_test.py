from pathlib import Path

import yaml

from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.adapters.interfaces import PerceptionFrame
from cat_cannon.adapters.rp2040_protocol import ControllerResponse
from cat_cannon.app.tracking_test import (
    TraceLog,
    TrackingTestConfig,
    TrackingTestState,
    _annotate_camera,
    _build_buttons,
    _control_from_key,
    _draw_camera_status_in_left_margin,
    _draw_button,
    _draw_panel_header,
    _next_limit_target,
    build_tracking_layout,
    detect_tracking_cameras,
    handle_tracking_control,
    resolve_zones_path,
)
from cat_cannon.domain.models import BoundingBox, Detection
from cat_cannon.domain.safety import DetectionPolicy


class FakeSession:
    def __init__(self) -> None:
        self.enabled: list[bool] = []

    def enable(self) -> None:
        self.enabled.append(True)

    def disable(self) -> None:
        self.enabled.append(False)


class FakeDetector:
    def __init__(self) -> None:
        self.source_ids: list[str] = []

    def detect(self, frame, source_id: str = "primary") -> PerceptionFrame:
        self.source_ids.append(source_id)
        return PerceptionFrame(
            source_id=source_id,
            width=640,
            height=480,
            detections=[
                Detection(
                    track_id=f"{source_id}-cat",
                    label="cat",
                    confidence=0.9,
                    bbox=BoundingBox(10, 20, 30, 40),
                )
            ],
        )


class FakeFrame:
    shape = (240, 320, 3)


class FakeCv2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self) -> None:
        self.put_text_calls: list[tuple[str, tuple[int, int]]] = []
        self.rectangle_calls: list[tuple[tuple[int, int], tuple[int, int], int]] = []

    def getTextSize(self, text: str, _font, scale: float, _thickness: int):
        return ((int(len(text) * scale * 8), int(scale * 20)), 4)

    def line(self, *_args) -> None:
        pass

    def rectangle(self, _frame, pt1, pt2, _color, thickness: int) -> None:
        self.rectangle_calls.append((pt1, pt2, thickness))

    def putText(self, _frame, text: str, origin, *_args) -> None:
        self.put_text_calls.append((text, origin))


class FakeLimitController(NullTurretController):
    def __init__(self, *, pan_deg: float, tilt_deg: float) -> None:
        super().__init__()
        self.pan_deg = pan_deg
        self.tilt_deg = tilt_deg
        self.limit_commands: list[dict[str, float]] = []
        self.relaxed = 0

    def status(self) -> ControllerResponse:
        return ControllerResponse(
            ok=True,
            sequence=1,
            status="status",
            payload={"pan_deg": self.pan_deg, "tilt_deg": self.tilt_deg},
        )

    def set_servo_limits(self, **limits: float) -> ControllerResponse:
        self.limit_commands.append(dict(limits))
        return ControllerResponse(ok=True, sequence=1, status="servo_limits_set", payload=limits)

    def relax(self) -> ControllerResponse:
        self.relaxed += 1
        return ControllerResponse(ok=True, sequence=1, status="relaxed", payload={})


def _policy() -> DetectionPolicy:
    return DetectionPolicy(
        cat_class="cat",
        person_class="person",
        cat_confidence_threshold=0.4,
        person_confidence_threshold=0.5,
        consecutive_counter_frames=2,
    )


def test_tracking_control_allows_manual_motion_while_disarmed() -> None:
    controller = NullTurretController()
    session = FakeSession()
    state = TrackingTestState(armed=False, step_deg=5.0)

    result = handle_tracking_control(
        "pan_left",
        state=state,
        session=session,
        controller=controller,
    )

    assert result.state.armed is False
    assert controller.pan_commands == [-5.0]
    assert controller.tilt_commands == [0.0]
    assert "pan left" in result.message


def test_tracking_control_blocks_fire_until_armed() -> None:
    controller = NullTurretController()
    state = TrackingTestState(armed=False, step_deg=5.0)

    result = handle_tracking_control(
        "fire",
        state=state,
        session=None,
        controller=controller,
    )

    assert result.state.armed is False
    assert controller.fired == 0
    assert "arm first" in result.message


def test_tracking_test_state_defaults_human_tracking_off() -> None:
    state = TrackingTestState(armed=True, step_deg=5.0)

    assert state.track_humans is False


def test_tracking_control_preserves_human_tracking_preference_when_arming() -> None:
    controller = NullTurretController()

    result = handle_tracking_control(
        "arm",
        state=TrackingTestState(armed=False, step_deg=5.0, track_humans=True),
        session=None,
        controller=controller,
    )

    assert result.state.armed is True
    assert result.state.track_humans is True


def test_tracking_control_arms_moves_fires_and_safe_stops() -> None:
    controller = NullTurretController()
    session = FakeSession()
    state = TrackingTestState(armed=False, step_deg=5.0)

    armed = handle_tracking_control("arm", state=state, session=session, controller=controller)
    moved = handle_tracking_control(
        "tilt_up",
        state=armed.state,
        session=session,
        controller=controller,
    )
    fired = handle_tracking_control(
        "fire",
        state=moved.state,
        session=session,
        controller=controller,
    )
    stopped = handle_tracking_control(
        "safe_stop",
        state=fired.state,
        session=session,
        controller=controller,
    )

    assert armed.state.armed is True
    assert session.enabled == [True, False]
    assert controller.tilt_commands == [-5.0]
    assert controller.fired == 1
    assert stopped.state.armed is False
    assert controller.stopped == 1


def test_tracking_control_cycles_limit_target() -> None:
    assert _next_limit_target("top") == "bottom"
    assert _next_limit_target("bottom") == "left"
    assert _next_limit_target("left") == "right"
    assert _next_limit_target("right") == "top"


def test_tracking_control_starts_guided_limit_flow_with_limits_disabled() -> None:
    controller = FakeLimitController(pan_deg=90.0, tilt_deg=90.0)

    result = handle_tracking_control(
        "set_limit_target",
        state=TrackingTestState(armed=False, step_deg=5.0, limit_target="right"),
        session=None,
        controller=controller,
    )

    assert result.state.limit_flow_active is True
    assert result.state.limit_target == "top"
    assert result.state.limit_samples == ()
    assert "move to top" in result.message
    assert controller.limit_commands == [
        {
            "pan_min_deg": 0.0,
            "pan_max_deg": 180.0,
            "tilt_min_deg": 30.0,
            "tilt_max_deg": 150.0,
        }
    ]


def test_tracking_control_requires_set_limits_before_saving_limit() -> None:
    controller = FakeLimitController(pan_deg=42.4, tilt_deg=88.8)

    result = handle_tracking_control(
        "save_limit",
        state=TrackingTestState(armed=False, step_deg=5.0),
        session=None,
        controller=controller,
        config_path="configs/app.example.yaml",
    )

    assert result.state.limit_flow_active is False
    assert "press Set Limits first" in result.message
    assert controller.limit_commands == []


def test_tracking_control_guides_all_limits_then_enforces_derived_range(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
  servo_center_pan_deg: 0
  servo_center_tilt_deg: 0
  servo_min_pan_deg: 0
  servo_max_pan_deg: 180
  servo_min_tilt_deg: 30
  servo_max_tilt_deg: 150
""",
        encoding="utf-8",
    )
    controller = FakeLimitController(pan_deg=90.0, tilt_deg=110.0)

    started = handle_tracking_control(
        "set_limit_target",
        state=TrackingTestState(armed=False, step_deg=5.0),
        session=None,
        controller=controller,
        config_path=str(config_path),
    )
    top = handle_tracking_control(
        "save_limit",
        state=started.state,
        session=None,
        controller=controller,
        config_path=str(config_path),
    )
    controller.tilt_deg = 70.0
    bottom = handle_tracking_control(
        "save_limit",
        state=top.state,
        session=None,
        controller=controller,
        config_path=str(config_path),
    )
    controller.pan_deg = 140.0
    left = handle_tracking_control(
        "save_limit",
        state=bottom.state,
        session=None,
        controller=controller,
        config_path=str(config_path),
    )
    controller.pan_deg = 40.0
    right = handle_tracking_control(
        "save_limit",
        state=left.state,
        session=None,
        controller=controller,
        config_path=str(config_path),
    )

    assert [top.state.limit_target, bottom.state.limit_target, left.state.limit_target] == [
        "bottom",
        "left",
        "right",
    ]
    assert right.state.limit_flow_active is False
    assert "limits saved" in right.message
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["tracking"]["servo_left_pan_deg"] == 140.0
    assert saved["tracking"]["servo_right_pan_deg"] == 40.0
    assert saved["tracking"]["servo_top_tilt_deg"] == 110.0
    assert saved["tracking"]["servo_bottom_tilt_deg"] == 70.0
    assert saved["tracking"]["servo_min_pan_deg"] == 40.0
    assert saved["tracking"]["servo_max_pan_deg"] == 140.0
    assert saved["tracking"]["servo_min_tilt_deg"] == 70.0
    assert saved["tracking"]["servo_max_tilt_deg"] == 110.0
    assert controller.limit_commands == [
        {
            "pan_min_deg": 0.0,
            "pan_max_deg": 180.0,
            "tilt_min_deg": 30.0,
            "tilt_max_deg": 150.0,
        },
        {
            "pan_min_deg": 40.0,
            "pan_max_deg": 140.0,
            "tilt_min_deg": 70.0,
            "tilt_max_deg": 110.0,
        },
    ]


def test_tracking_limit_buttons_are_set_save_and_clear() -> None:
    buttons = _build_buttons(TrackingTestConfig())
    button_keys = [button.key for button in buttons]

    assert "set_limit_target" in button_keys
    assert "save_limit" in button_keys
    assert "clear_limits" in button_keys
    assert "set_pan_min" not in button_keys
    assert "set_pan_max" not in button_keys
    assert "set_tilt_min" not in button_keys
    assert "set_tilt_max" not in button_keys
    assert _control_from_key(ord("l")) == "set_limit_target"
    assert _control_from_key(ord("v")) == "save_limit"
    assert _control_from_key(ord("0")) == "clear_limits"


def test_tracking_control_clears_servo_limits_and_updates_controller(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
system:
  arm_required: true
  cooldown_frames: 10
detection:
  cat_class: cat
  person_class: person
  cat_confidence_threshold: 0.45
  person_confidence_threshold: 0.55
  consecutive_counter_frames: 3
tracking:
  horizontal_deadband_px: 5
  vertical_deadband_px: 5
  horizontal_gain: 0.03
  vertical_gain: 0.03
  aim_offset_x_px: 0
  aim_offset_y_px: 20
  servo_center_pan_deg: 77
  servo_center_tilt_deg: 101
  servo_min_pan_deg: 88
  servo_max_pan_deg: 89
  servo_min_tilt_deg: 91
  servo_max_tilt_deg: 92
""",
        encoding="utf-8",
    )
    controller = FakeLimitController(pan_deg=88.5, tilt_deg=91.5)

    result = handle_tracking_control(
        "clear_limits",
        state=TrackingTestState(armed=False, step_deg=5.0),
        session=None,
        controller=controller,
        config_path=str(config_path),
    )

    assert result.message == "servo limits and center cleared"
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "servo_center_pan_deg" not in saved["tracking"]
    assert "servo_center_tilt_deg" not in saved["tracking"]
    assert "servo_min_pan_deg" not in saved["tracking"]
    assert "servo_max_pan_deg" not in saved["tracking"]
    assert "servo_min_tilt_deg" not in saved["tracking"]
    assert "servo_max_tilt_deg" not in saved["tracking"]
    assert controller.limit_commands == [
        {
            "pan_min_deg": 0.0,
            "pan_max_deg": 180.0,
            "tilt_min_deg": 30.0,
            "tilt_max_deg": 150.0,
        }
    ]


def test_tracking_camera_annotations_stay_top_left() -> None:
    cv2 = FakeCv2()

    _annotate_camera(cv2, FakeFrame(), "cam", ["ok", "state=tracking"])

    assert cv2.put_text_calls == []


def test_tracking_buttons_draw_labels_on_button_surfaces() -> None:
    cv2 = FakeCv2()
    button = _build_buttons(TrackingTestConfig())[0]

    _draw_button(cv2, object(), button)

    assert cv2.put_text_calls == [(button.label, (button.x1 + 10, button.y1 + 27))]


def test_tracking_panel_header_restores_screen_and_armed_status() -> None:
    cv2 = FakeCv2()
    layout = build_tracking_layout(
        fixed_frame_width=640,
        fixed_frame_height=480,
        turret_frame_width=640,
        turret_frame_height=480,
        window_width=1024,
        window_height=600,
        panel_width=280,
    )

    buttons = _build_buttons(TrackingTestConfig(window_width=1024, window_height=600, panel_width=280))

    _draw_panel_header(
        cv2,
        object(),
        layout=layout,
        state=TrackingTestState(
            armed=True,
            step_deg=3.0,
            track_humans=True,
            limit_flow_active=True,
            limit_target="left",
        ),
        status_message="Loaded zones",
    )
    rendered_text = [text for text, _origin in cv2.put_text_calls]
    origins = [origin for _text, origin in cv2.put_text_calls]

    assert "Tracking Test" in rendered_text
    assert "Armed: YES  Track: human" in rendered_text
    assert any(text.startswith("Limits: move to left") for text in rendered_text)
    assert not any(text.startswith("Status: Loaded zones") for text in rendered_text)
    assert not any(text.startswith("Zones:") for text in rendered_text)
    assert any(text.startswith("Keys:") for text in rendered_text)
    assert all(x >= layout.panel_x for x, _y in origins)
    assert not any(button.label in rendered_text for button in buttons)


def test_tracking_camera_status_uses_left_image_margin() -> None:
    cv2 = FakeCv2()
    layout = build_tracking_layout(
        fixed_frame_width=640,
        fixed_frame_height=480,
        turret_frame_width=640,
        turret_frame_height=480,
        window_width=1024,
        window_height=600,
        panel_width=280,
    )

    _draw_camera_status_in_left_margin(
        cv2,
        object(),
        placement=layout.fixed,
        title="fixed tracking camera",
        status_lines=["fixed cats=1 people=0", "state=tracking"],
    )

    rendered_text = [text for text, _origin in cv2.put_text_calls]
    origins = [origin for _text, origin in cv2.put_text_calls]

    assert "fixed tracking camera" in rendered_text
    assert "fixed cats=1 people=0" in rendered_text
    assert all(layout.fixed.region.x <= x < layout.fixed.preview.x for x, _y in origins)


def test_tracking_control_relaxes_servos_when_supported() -> None:
    controller = FakeLimitController(pan_deg=90, tilt_deg=90)

    result = handle_tracking_control(
        "relax",
        state=TrackingTestState(armed=False, step_deg=5.0),
        session=None,
        controller=controller,
    )

    assert result.message == "servos relaxed"
    assert controller.relaxed == 1


def test_tracking_control_can_switch_to_zone_calibration() -> None:
    result = handle_tracking_control(
        "zone_calibration",
        state=TrackingTestState(armed=False, step_deg=3.0),
        session=None,
        controller=NullTurretController(),
    )

    assert result.next_screen == "zone_calibration"
    assert result.should_exit is True


def test_tracking_detection_runs_yolo_for_fixed_and_tracking_cameras() -> None:
    detector = FakeDetector()

    result = detect_tracking_cameras(
        detector=detector,
        fixed_frame=object(),
        turret_frame=object(),
        policy=_policy(),
    )

    assert detector.source_ids == ["fixed", "turret"]
    assert result.fixed.source_id == "fixed"
    assert result.turret is not None
    assert result.turret.source_id == "turret"
    assert result.fixed_summary == "fixed cats=1 people=0"
    assert result.turret_summary == "turret cats=1 people=0"


def test_trace_log_keeps_recent_control_events() -> None:
    log = TraceLog(max_lines=2)

    log.add("key e -> arm", emit=False)
    log.add("response armed", emit=False)
    log.add("status enabled=True", emit=False)

    assert log.lines == ["response armed", "status enabled=True"]


def test_tracking_layout_splits_two_camera_previews_from_control_panel() -> None:
    layout = build_tracking_layout(
        fixed_frame_width=1280,
        fixed_frame_height=720,
        turret_frame_width=640,
        turret_frame_height=480,
        window_width=1024,
        window_height=600,
        panel_width=260,
    )

    assert layout.panel_x == 764
    assert layout.fixed.region.height == 300
    assert layout.turret is not None
    assert layout.turret.region.y == 300
    assert layout.fixed.preview.width <= layout.fixed.region.width
    assert layout.turret.preview.height <= layout.turret.region.height


def test_resolve_zones_path_falls_back_to_example_for_default_zones_yaml(tmp_path: Path) -> None:
    example = tmp_path / "configs" / "zones.example.yaml"
    example.parent.mkdir()
    example.write_text("zones: []\n", encoding="utf-8")

    resolved = resolve_zones_path(tmp_path / "configs" / "zones.yaml")

    assert resolved == example


def test_tracking_test_config_defaults_to_named_device_symlinks() -> None:
    config = TrackingTestConfig()

    assert config.fixed_camera == "/dev/fixed_cam"
    assert config.turret_camera == "/dev/turret_cam"
    assert config.fixed_camera_width == 1280
    assert config.fixed_camera_height == 720
    assert config.turret_camera_width == 1280
    assert config.turret_camera_height == 720
    assert config.zones_path == "configs/zones.yaml"
