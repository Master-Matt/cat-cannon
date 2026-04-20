from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.app.replay import ReplayFrame, run_replay
from cat_cannon.app.supervisor import SupervisorLoop
from cat_cannon.config import SystemConfig
from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, Point
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
    return Detection("cat-1", "cat", 0.9, BoundingBox(80, 60, 20, 50))


def _person_detection() -> Detection:
    return Detection("person-1", "person", 0.95, BoundingBox(20, 20, 60, 120))


def test_replay_fires_after_confirmation_and_aim_lock() -> None:
    supervisor, controller = _supervisor()
    frames = [
        ReplayFrame([_cat_detection()], 200, 200),
        ReplayFrame([_cat_detection()], 200, 200),
        ReplayFrame([_cat_detection()], 200, 200),
        ReplayFrame([_cat_detection()], 200, 200),
    ]

    snapshots = run_replay(supervisor, frames)

    assert controller.fired == 1
    assert snapshots[-1].fire_count == 1


def test_replay_human_presence_forces_safe_stop_and_blocks_fire() -> None:
    supervisor, controller = _supervisor()
    frames = [
        ReplayFrame([_cat_detection()], 200, 200),
        ReplayFrame([_cat_detection(), _person_detection()], 200, 200),
        ReplayFrame([_cat_detection(), _person_detection()], 200, 200),
    ]

    snapshots = run_replay(supervisor, frames)

    assert controller.fired == 0
    assert controller.stopped >= 1
    assert snapshots[-1].stop_count >= 1
