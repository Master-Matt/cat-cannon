from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.app.supervisor import SupervisorLoop
from cat_cannon.config import SystemConfig
from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, Point, SupervisorState
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration


def _supervisor() -> tuple[SupervisorLoop, NullTurretController]:
    controller = NullTurretController()
    config = SystemConfig(
        cooldown_frames=2,
        detection_policy=DetectionPolicy(
            cat_class="cat",
            person_class="person",
            cat_confidence_threshold=0.4,
            person_confidence_threshold=0.5,
            consecutive_counter_frames=2,
        ),
        tracking_calibration=TrackingCalibration(
            horizontal_deadband_px=20,
            vertical_deadband_px=20,
            horizontal_gain=0.05,
            vertical_gain=0.05,
            aim_offset_x_px=0,
            aim_offset_y_px=0,
        ),
    )
    zones = [
        CounterZone(
            zone_id="counter",
            polygon=(
                Point(0, 20),
                Point(200, 20),
                Point(200, 180),
                Point(0, 180),
            ),
        )
    ]
    return SupervisorLoop(config=config, zones=zones, controller=controller), controller


def _cat_detection() -> Detection:
    return Detection("cat-1", "cat", 0.9, BoundingBox(90, 75, 20, 50))


def _person_detection() -> Detection:
    return Detection("person-1", "person", 0.95, BoundingBox(20, 20, 60, 120))


def test_supervisor_loop_returns_tracking_state_and_zone_after_confirmation() -> None:
    supervisor, controller = _supervisor()

    first = supervisor.process_frame([_cat_detection()], frame_width=200, frame_height=200, armed=True)
    second = supervisor.process_frame([_cat_detection()], frame_width=200, frame_height=200, armed=True)

    assert first.state == SupervisorState.IDLE
    assert first.counter_confirmed is False
    assert second.state == SupervisorState.TRACKING
    assert second.counter_confirmed is True
    assert second.target_visible is True
    assert second.active_zone_id == "counter"
    assert len(controller.pan_commands) >= 1


def test_supervisor_loop_reports_human_lockout_and_safe_stop() -> None:
    supervisor, controller = _supervisor()

    result = supervisor.process_frame(
        [_cat_detection(), _person_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
    )

    assert result.state == SupervisorState.HUMAN_LOCKOUT
    assert result.human_present is True
    assert result.fire_commanded is False
    assert controller.stopped >= 1
