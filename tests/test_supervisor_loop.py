from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.app.supervisor import SupervisorLoop
from cat_cannon.config import ServoLimits, SystemConfig
from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, Point, SupervisorState
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration


def _supervisor(
    *,
    servo_limits: ServoLimits | None = None,
) -> tuple[SupervisorLoop, NullTurretController]:
    controller = NullTurretController()
    config = SystemConfig(
        cooldown_frames=2,
        detection_policy=DetectionPolicy(
            cat_class="cat",
            person_class="person",
            cat_confidence_threshold=0.4,
            person_confidence_threshold=0.5,
            consecutive_counter_frames=2,
            confirmation_miss_tolerance_frames=5,
        ),
        tracking_calibration=TrackingCalibration(
            horizontal_deadband_px=20,
            vertical_deadband_px=20,
            horizontal_gain=0.05,
            vertical_gain=0.05,
            aim_offset_x_px=0,
            aim_offset_y_px=0,
        ),
        servo_limits=servo_limits or ServoLimits(),
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
    return Detection("cat-1", "cat", 0.9, BoundingBox(50, 40, 20, 50))


def _right_side_cat_detection() -> Detection:
    return Detection("cat-1", "cat", 0.9, BoundingBox(140, 40, 20, 50))


def _person_detection() -> Detection:
    return Detection("person-1", "person", 0.95, BoundingBox(20, 20, 60, 120))


def test_supervisor_loop_returns_tracking_state_and_zone_after_confirmation() -> None:
    supervisor, controller = _supervisor()

    first = supervisor.process_frame(
        [_cat_detection()], frame_width=200, frame_height=200, armed=True
    )
    second = supervisor.process_frame(
        [_cat_detection()], frame_width=200, frame_height=200, armed=True
    )

    assert first.state == SupervisorState.IDLE
    assert first.counter_confirmed is False
    assert second.state == SupervisorState.TRACKING
    assert second.counter_confirmed is True
    assert second.target_visible is True
    assert second.active_zone_id == "counter"
    assert len(controller.pan_commands) >= 1


def test_supervisor_loop_requires_human_hysteresis_before_lockout() -> None:
    supervisor, controller = _supervisor()

    result = None
    for index in range(4):
        result = supervisor.process_frame(
            [_cat_detection(), _person_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            now=index * 0.1,
        )
        assert result.human_present is False
        assert result.state != SupervisorState.HUMAN_LOCKOUT

    result = supervisor.process_frame(
        [_cat_detection(), _person_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        now=0.4,
    )
    assert result.state == SupervisorState.HUMAN_LOCKOUT
    assert result.human_present is True
    assert result.fire_commanded is False
    # Turret still tracks when armed (just won't fire) — no safe_stop


def test_supervisor_loop_releases_human_lockout_after_quiet_window() -> None:
    supervisor, _controller = _supervisor()

    for index in range(5):
        locked = supervisor.process_frame(
            [_cat_detection(), _person_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            now=index * 0.1,
        )
    assert locked.state == SupervisorState.HUMAN_LOCKOUT
    assert locked.human_present is True

    still_locked = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        now=2.0,
    )
    assert still_locked.state == SupervisorState.HUMAN_LOCKOUT
    assert still_locked.human_present is True

    released = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        now=3.0,
    )
    assert released.human_present is False
    assert released.state != SupervisorState.HUMAN_LOCKOUT


def test_supervisor_loop_does_not_apply_tracking_delta_when_disarmed() -> None:
    supervisor, controller = _supervisor()

    result = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=False,
    )

    assert result.state == SupervisorState.DISARMED
    assert controller.pan_commands == []
    assert controller.tilt_commands == []
    assert controller.stopped >= 1


def test_supervisor_does_not_fallback_track_while_disarmed_after_confirmation() -> None:
    supervisor, controller = _supervisor()

    supervisor.process_frame(
        [_cat_detection()], frame_width=200, frame_height=200, armed=False
    )
    result = supervisor.process_frame(
        [_cat_detection()], frame_width=200, frame_height=200, armed=False
    )

    assert result.counter_confirmed is True
    assert result.correction is None
    assert controller.pan_commands == []
    assert controller.tilt_commands == []
    assert controller.stopped >= 2


def test_supervisor_uses_turret_camera_for_targeting_when_available() -> None:
    supervisor, controller = _supervisor()

    # Cat at center of turret frame → should be aim-locked
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    # Confirm counter presence via fixed camera (2 frames needed)
    supervisor.process_frame([_cat_detection()], frame_width=200, frame_height=200, armed=True)
    result = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
    )

    assert result.counter_confirmed is True
    assert result.target_visible is True
    # Turret cat is centered → correction should be near zero / aim locked
    assert result.correction is not None
    assert result.aim_locked is True
    # No servo commands sent because target is already centered (below deadband)
    assert len(controller.pan_commands) == 0


def test_supervisor_leads_turret_horizontally_from_fixed_zone_when_turret_target_missing() -> None:
    supervisor, controller = _supervisor()

    supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
    )
    result = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
    )

    assert result.counter_confirmed is True
    assert result.aim_locked is False
    assert result.fire_commanded is False
    assert result.correction is not None
    assert result.correction.pan_delta > 0
    assert controller.pan_commands[-1] > 0
    assert controller.tilt_commands[-1] == 0.0


def test_supervisor_leads_turret_from_single_fixed_zone_hit_before_confirmation() -> None:
    supervisor, controller = _supervisor()

    result = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
    )

    assert result.counter_confirmed is False
    assert result.target_visible is False
    assert result.aim_locked is False
    assert result.fire_commanded is False
    assert result.correction is not None
    assert result.correction.pan_delta > 0
    assert controller.pan_commands[-1] > 0
    assert controller.tilt_commands[-1] == 0.0


def test_supervisor_logs_zone_activation_and_fire(capsys) -> None:
    supervisor, controller = _supervisor()
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    for _ in range(4):
        supervisor.process_frame(
            [_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    output = capsys.readouterr().out
    assert controller.fired == 1
    assert "[supervisor]" in output
    assert "zone_active zone=counter" in output
    assert "fire_commanded zone=counter" in output


def test_supervisor_does_not_fire_without_turret_cat_when_turret_camera_available() -> None:
    supervisor, controller = _supervisor()

    for _ in range(3):
        result = supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is True
    assert result.target_visible is True
    assert result.aim_locked is False
    assert result.turret_target_visible is False
    assert result.fire_permitted is False
    assert result.fire_commanded is False
    assert controller.fired == 0


def test_supervisor_does_not_fire_at_off_center_turret_cat() -> None:
    supervisor, controller = _supervisor()
    off_center_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))

    for _ in range(3):
        result = supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[off_center_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is True
    assert result.turret_target_visible is True
    assert result.turret_fire_aligned is False
    assert result.fire_permitted is False
    assert result.fire_commanded is False
    assert controller.fired == 0


def test_supervisor_does_not_fire_inside_wide_gate_until_aim_locked() -> None:
    supervisor, controller = _supervisor()
    near_center_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(125, 90, 20, 20))

    for _ in range(4):
        result = supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[near_center_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is True
    assert result.turret_target_visible is True
    assert result.turret_fire_aligned is True
    assert result.aim_locked is False
    assert result.fire_permitted is False
    assert result.fire_commanded is False
    assert controller.fired == 0


def test_supervisor_fires_when_fixed_cat_and_turret_cat_are_aligned() -> None:
    supervisor, controller = _supervisor()
    centered_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    for _ in range(4):
        result = supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is True
    assert result.turret_target_visible is True
    assert result.turret_fire_aligned is True
    assert result.fire_permitted is True
    assert controller.fired == 1


def test_supervisor_does_not_fire_when_pan_is_not_pointed_toward_fixed_cat() -> None:
    servo_limits = ServoLimits(
        pan_min_deg=40,
        pan_max_deg=140,
        tilt_min_deg=70,
        tilt_max_deg=110,
        pan_left_deg=140,
        pan_right_deg=40,
        tilt_top_deg=110,
        tilt_bottom_deg=70,
    )
    supervisor, controller = _supervisor(servo_limits=servo_limits)
    controller.last_status_payload = {"pan_deg": 140.0}
    centered_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    for _ in range(4):
        result = supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is True
    assert result.turret_target_visible is True
    assert result.turret_fire_aligned is True
    assert result.turret_direction_aligned is False
    assert result.fire_permitted is False
    assert controller.fired == 0


def test_supervisor_keeps_fixed_confirmation_for_four_missed_frames() -> None:
    supervisor, controller = _supervisor()
    centered_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    for _ in range(2):
        result = supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is True
    assert result.active_zone_id == "counter"

    missed_results = [
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )
        for _ in range(4)
    ]

    assert [result.counter_confirmed for result in missed_results] == [
        True,
        True,
        True,
        True,
    ]
    assert missed_results[-1].active_zone_id == "counter"
    assert controller.fired >= 1


def test_supervisor_drops_fixed_confirmation_on_fifth_missed_frame() -> None:
    supervisor, _controller = _supervisor()

    for _ in range(2):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    for _ in range(4):
        result = supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[],
            turret_frame_width=200,
            turret_frame_height=200,
        )
        assert result.counter_confirmed is True

    fifth_miss = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
    )

    assert fifth_miss.counter_confirmed is False
    assert fifth_miss.active_zone_id is None
    assert fifth_miss.fire_commanded is False


def test_supervisor_ignores_single_rogue_human_detection_for_fire_permission() -> None:
    supervisor, controller = _supervisor()
    centered_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    for _ in range(2):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    result = supervisor.process_frame(
        [_right_side_cat_detection(), _person_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[centered_turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.1,
    )

    assert result.human_present is False
    assert result.counter_confirmed is True
    assert result.state != SupervisorState.HUMAN_LOCKOUT
    assert result.fire_permitted is True

    fired = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[centered_turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.2,
    )

    assert fired.fire_commanded is True
    assert controller.fired == 1


def test_supervisor_requires_five_human_frames_before_lockout_blocks_fire() -> None:
    supervisor, controller = _supervisor()
    centered_turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(90, 90, 20, 20))

    for _ in range(2):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
        )

    results = [
        supervisor.process_frame(
            [_right_side_cat_detection(), _person_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )
        for index in range(5)
    ]

    assert [result.human_present for result in results] == [
        False,
        False,
        False,
        False,
        True,
    ]
    assert results[-1].state == SupervisorState.HUMAN_LOCKOUT
    assert results[-1].fire_commanded is False
    assert controller.fired >= 1


def test_supervisor_does_not_fire_when_only_bbox_edge_intersects_zone() -> None:
    supervisor, controller = _supervisor()
    edge_only_cat = Detection("cat-1", "cat", 0.9, BoundingBox(190, 40, 40, 50))

    for _ in range(5):
        result = supervisor.process_frame(
            [edge_only_cat],
            frame_width=240,
            frame_height=200,
            armed=True,
            turret_detections=[],
            turret_frame_width=240,
            turret_frame_height=200,
        )

    assert result.counter_confirmed is False
    assert result.target_visible is False
    assert result.fire_commanded is False
    assert controller.fired == 0


def test_supervisor_does_not_track_turret_people_by_default() -> None:
    supervisor, controller = _supervisor()

    turret_person = Detection("person-1", "person", 0.95, BoundingBox(20, 20, 20, 20))

    result = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[turret_person],
        turret_frame_width=200,
        turret_frame_height=200,
    )

    assert result.correction is None
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


def test_supervisor_tracks_turret_people_when_enabled_but_hysteresis_controls_lockout() -> None:
    supervisor, controller = _supervisor()

    turret_person = Detection("person-1", "person", 0.95, BoundingBox(20, 20, 20, 20))

    result = None
    for index in range(10):
        result = supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_person],
            turret_frame_width=200,
            turret_frame_height=200,
            track_people=True,
            now=index * 0.1,
        )

    assert result.human_present is True
    assert result.state == SupervisorState.HUMAN_LOCKOUT
    assert result.fire_commanded is False
    assert result.correction is not None
    assert controller.pan_commands
    assert controller.fired == 0
