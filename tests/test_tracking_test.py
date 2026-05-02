from pathlib import Path

from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.adapters.interfaces import PerceptionFrame
from cat_cannon.app.tracking_test import (
    TraceLog,
    TrackingTestConfig,
    TrackingTestState,
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


def _policy() -> DetectionPolicy:
    return DetectionPolicy(
        cat_class="cat",
        person_class="person",
        cat_confidence_threshold=0.4,
        person_confidence_threshold=0.5,
        consecutive_counter_frames=2,
    )


def test_tracking_control_blocks_motion_until_armed() -> None:
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
    assert controller.pan_commands == []
    assert "arm first" in result.message


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
    assert config.zones_path == "configs/zones.yaml"
