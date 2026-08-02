from cat_cannon.adapters.controller import NullTurretController
from cat_cannon.app.supervisor import SupervisorLoop
from cat_cannon.config import ServoLimits, SystemConfig, TrackingTuning
from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, Point, SupervisorState
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration, TurretCorrection


def _supervisor(
    *,
    servo_limits: ServoLimits | None = None,
    tracking_tuning: TrackingTuning | None = None,
    acquisition_frame_threshold: int = 1,
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
        tracking_tuning=tracking_tuning
        or TrackingTuning(
            acquisition_frame_threshold=acquisition_frame_threshold,
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


def test_supervisor_does_not_confirm_zone_by_replaying_stale_fixed_detection() -> None:
    supervisor, controller = _supervisor()

    first = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        fixed_detections_fresh=True,
    )
    stale_replay = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        fixed_detections_fresh=False,
    )
    fresh_second_hit = supervisor.process_frame(
        [_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        fixed_detections_fresh=True,
    )

    assert first.counter_confirmed is False
    assert stale_replay.counter_confirmed is False
    assert stale_replay.active_zone_id is None
    assert fresh_second_hit.counter_confirmed is True
    assert fresh_second_hit.fixed_zone_fresh is True
    assert controller.fired == 0


def test_supervisor_uses_stale_fixed_hit_only_as_turret_lead_hint() -> None:
    supervisor, controller = _supervisor()

    supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        fixed_detections_fresh=True,
    )
    controller.pan_commands.clear()
    controller.tilt_commands.clear()

    stale_lead = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        fixed_detections_fresh=False,
    )

    assert stale_lead.counter_confirmed is False
    assert stale_lead.active_zone_id is None
    assert stale_lead.fire_commanded is False
    assert stale_lead.correction is not None
    assert stale_lead.correction.pan_delta > 0
    assert controller.pan_commands[-1] > 0
    assert controller.tilt_commands[-1] == 0.0
    assert controller.fired == 0


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


def test_supervisor_tracks_turret_only_cat_while_idle() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))

    results = [
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.2,
        )
        for index in range(5)
    ]

    assert all(result.correction is None for result in results[:4])
    assert results[-1].state == SupervisorState.IDLE
    assert results[-1].target_visible is False
    assert results[-1].turret_target_visible is True
    assert results[-1].correction is not None
    assert controller.pan_commands
    assert controller.tilt_commands


def test_supervisor_does_not_combine_spatially_distant_cat_boxes() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    left_cat = Detection("cat-left", "cat", 0.92, BoundingBox(5, 90, 20, 20))
    right_cat = Detection("cat-right", "cat", 0.92, BoundingBox(175, 90, 20, 20))

    results = [
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[detection],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )
        for index, detection in enumerate(
            (left_cat, right_cat, left_cat, right_cat, left_cat)
        )
    ]

    assert all(result.correction is None for result in results)
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


def test_supervisor_does_not_track_tiny_coherent_cat_box() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    tiny_false_cat = Detection(
        "false-cat",
        "cat",
        0.92,
        BoundingBox(120, 100, 4, 12),
    )

    results = [
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[tiny_false_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )
        for index in range(5)
    ]

    assert all(result.correction is None for result in results)
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


def test_tiny_person_still_tracks_and_engages_human_safety_lockout() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=1)
    tiny_person = Detection(
        "tiny-person",
        "person",
        0.95,
        BoundingBox(120, 100, 4, 12),
    )

    for index in range(5):
        result = supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[tiny_person],
            turret_frame_width=200,
            turret_frame_height=200,
            track_people=True,
            now=index * 0.1,
        )

    assert result.human_present is True
    assert result.correction is not None
    assert controller.pan_commands


def test_supervisor_drops_acquired_cat_after_spatial_jump() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    left_cat = Detection("cat-left", "cat", 0.92, BoundingBox(5, 90, 20, 20))
    right_cat = Detection("cat-right", "cat", 0.92, BoundingBox(175, 90, 20, 20))

    for index in range(5):
        acquired = supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[left_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )
    assert acquired.correction is not None

    controller.pan_commands.clear()
    controller.tilt_commands.clear()
    jumped = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[right_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.5,
    )

    assert jumped.correction is None
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


def test_supervisor_expires_partial_turret_acquisition_after_one_second() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))

    results = []
    for now in (0.0, 0.1, 0.2, 0.3, 1.31):
        results.append(
            supervisor.process_frame(
                [],
                frame_width=200,
                frame_height=200,
                armed=True,
                turret_detections=[turret_cat],
                turret_frame_width=200,
                turret_frame_height=200,
                now=now,
            )
        )

    assert all(result.correction is None for result in results)
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


def test_supervisor_keeps_acquired_target_through_a_brief_missed_frame() -> None:
    supervisor, _controller = _supervisor(acquisition_frame_threshold=5)
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))

    for index in range(5):
        acquired = supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )
    missed = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.5,
    )
    resumed = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.6,
    )

    assert acquired.correction is not None
    assert missed.correction is None
    assert resumed.correction is not None


def test_supervisor_counts_only_confidence_valid_detections_for_acquisition() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    valid_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))
    low_confidence_cat = Detection(
        "cat-1",
        "cat",
        0.39,
        BoundingBox(15, 90, 20, 20),
    )

    results = []
    for now, detection in (
        (0.0, valid_cat),
        (0.1, valid_cat),
        (0.2, valid_cat),
        (0.3, low_confidence_cat),
        (0.4, valid_cat),
    ):
        results.append(
            supervisor.process_frame(
                [],
                frame_width=200,
                frame_height=200,
                armed=True,
                turret_detections=[detection],
                turret_frame_width=200,
                turret_frame_height=200,
                now=now,
            )
        )

    assert all(result.correction is None for result in results)
    acquired = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[valid_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.5,
    )

    assert acquired.correction is not None
    assert controller.pan_commands


def test_supervisor_does_not_combine_target_classes_during_acquisition() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))
    turret_person = Detection("person-1", "person", 0.95, BoundingBox(20, 20, 20, 20))

    cat_results = [
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            track_people=True,
            now=index * 0.1,
        )
        for index in range(4)
    ]
    first_person = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[turret_person],
        turret_frame_width=200,
        turret_frame_height=200,
        track_people=True,
        now=0.4,
    )

    assert all(result.correction is None for result in cat_results)
    assert first_person.correction is None
    assert controller.pan_commands == []

    result = first_person
    for index in range(4):
        result = supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_person],
            turret_frame_width=200,
            turret_frame_height=200,
            track_people=True,
            now=0.5 + index * 0.1,
        )

    assert result.correction is not None
    assert controller.pan_commands


def test_supervisor_clears_tracking_filter_when_turret_target_disappears() -> None:
    supervisor, _controller = _supervisor()
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))

    supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
    )
    assert supervisor._filtered_pan != 0.0

    supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
    )

    assert supervisor._filtered_pan == 0.0
    assert supervisor._filtered_tilt == 0.0


def test_supervisor_tracks_current_turret_target_after_fixed_context_expires() -> None:
    supervisor, controller = _supervisor()
    turret_cat = Detection("cat-1", "cat", 0.92, BoundingBox(15, 90, 20, 20))

    for now in (0.0, 0.1):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            fixed_detections_fresh=True,
            now=now,
        )
    controller.pan_commands.clear()
    controller.tilt_commands.clear()

    stale_result = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        fixed_detections_fresh=False,
        now=3.0,
    )

    assert stale_result.counter_confirmed is True
    assert stale_result.correction is not None
    assert controller.pan_commands
    assert controller.tilt_commands


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


class _ReportedPanController(NullTurretController):
    pan_delta_sign = -1

    def __init__(self, pan_deg: float) -> None:
        super().__init__()
        self.pan_deg = pan_deg

    @property
    def last_status_payload(self) -> dict:
        return {
            "pan_deg": self.pan_deg,
            "pan_at_min": False,
            "pan_at_max": False,
            "tilt_at_min": False,
            "tilt_at_max": False,
        }


def test_fixed_camera_lead_stops_at_calibrated_absolute_pan_target() -> None:
    limits = ServoLimits(
        pan_min_deg=-99.0,
        pan_max_deg=117.0,
        pan_left_deg=117.0,
        pan_right_deg=-99.0,
    )
    supervisor, _ = _supervisor(servo_limits=limits)
    controller = _ReportedPanController(pan_deg=9.0)
    supervisor.controller = controller

    approaching = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.0,
    )

    assert approaching.correction is not None
    assert approaching.correction.pan_delta > 0.0
    assert controller.pan_commands

    controller.pan_deg = -45.0
    controller.pan_commands.clear()
    controller.tilt_commands.clear()
    arrived = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.1,
    )

    assert arrived.correction is not None
    assert arrived.correction.pan_delta == 0.0
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


def test_supervisor_requires_five_fixed_zone_hits_before_leading_turret() -> None:
    supervisor, controller = _supervisor(acquisition_frame_threshold=5)

    results = [
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.2,
        )
        for index in range(5)
    ]

    assert all(result.correction is None for result in results[:4])
    assert results[-1].counter_confirmed is True
    assert results[-1].target_visible is True
    assert results[-1].aim_locked is False
    assert results[-1].fire_commanded is False
    assert results[-1].correction is not None
    assert results[-1].correction.pan_delta > 0
    assert controller.pan_commands[-1] > 0
    assert controller.tilt_commands[-1] == 0.0


def test_supervisor_drops_old_fixed_camera_lead_after_half_a_second() -> None:
    supervisor, controller = _supervisor(
        tracking_tuning=TrackingTuning(
            acquisition_frame_threshold=1,
            fixed_lead_hold_seconds=0.5,
        )
    )

    supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        fixed_detections_fresh=True,
        now=0.0,
    )
    controller.pan_commands.clear()
    controller.tilt_commands.clear()

    stale = supervisor.process_frame(
        [_right_side_cat_detection()],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        fixed_detections_fresh=False,
        now=0.501,
    )

    assert stale.correction is None
    assert controller.pan_commands == []
    assert controller.tilt_commands == []


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


def test_supervisor_treats_fixed_pan_direction_as_diagnostic_after_turret_lock() -> None:
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
    assert result.fire_permitted is True
    assert controller.fired == 1


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


def test_supervisor_keeps_firing_while_turret_lock_survives_fixed_zone_loss() -> None:
    supervisor, controller = _supervisor()
    centered_turret_cat = Detection(
        "cat-1",
        "cat",
        0.92,
        BoundingBox(90, 90, 20, 20),
    )

    for index in range(4):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )
    fired_before_fixed_loss = controller.fired

    results = [
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=0.4 + index * 0.1,
        )
        for index in range(12)
    ]

    assert results[-1].counter_confirmed is True
    assert results[-1].target_visible is True
    assert results[-1].active_zone_id == "counter"
    assert results[-1].turret_target_visible is True
    assert results[-1].fire_permitted is True
    assert controller.fired >= fired_before_fixed_loss + 2


def test_supervisor_cancels_latched_engagement_for_human_lockout() -> None:
    supervisor, controller = _supervisor()
    centered_turret_cat = Detection(
        "cat-1",
        "cat",
        0.92,
        BoundingBox(90, 90, 20, 20),
    )

    for index in range(4):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )

    for index in range(5):
        result = supervisor.process_frame(
            [_person_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=0.4 + index * 0.1,
        )

    fired_at_lockout = controller.fired
    held_lockout = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[centered_turret_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.9,
    )

    assert result.state == SupervisorState.HUMAN_LOCKOUT
    assert result.fire_commanded is False
    assert held_lockout.state == SupervisorState.HUMAN_LOCKOUT
    assert held_lockout.fire_permitted is False
    assert controller.fired == fired_at_lockout


def test_supervisor_cancels_latched_engagement_after_turret_target_loss() -> None:
    supervisor, _controller = _supervisor()
    centered_turret_cat = Detection(
        "cat-1",
        "cat",
        0.92,
        BoundingBox(90, 90, 20, 20),
    )

    for index in range(4):
        supervisor.process_frame(
            [_right_side_cat_detection()],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[centered_turret_cat],
            turret_frame_width=200,
            turret_frame_height=200,
            now=index * 0.1,
        )

    for index in range(4):
        supervisor.process_frame(
            [],
            frame_width=200,
            frame_height=200,
            armed=True,
            turret_detections=[],
            turret_frame_width=200,
            turret_frame_height=200,
            now=0.4 + index * 0.1,
        )

    lost = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        now=1.31,
    )

    assert lost.counter_confirmed is False
    assert lost.target_visible is False
    assert lost.active_zone_id is None
    assert lost.fire_permitted is False
    assert lost.fire_commanded is False


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


def _tuning_supervisor(tuning: TrackingTuning) -> tuple[SupervisorLoop, NullTurretController]:
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
        servo_limits=ServoLimits(),
        tracking_tuning=tuning,
    )
    return SupervisorLoop(config=config, zones=[], controller=controller), controller


def test_apply_ema_tracking_clamps_tilt_symmetrically_with_pan() -> None:
    tuning = TrackingTuning(
        ema_alpha=1.0, gain=0.5, pan_clamp_deg=3.0, tilt_clamp_deg=3.0, deadband_deg=0.1
    )
    supervisor, controller = _tuning_supervisor(tuning)

    # Large correction in both axes; raw is clamped to +/-15, *gain=0.5 -> 7.5 desired,
    # which must be clamped down to the configured 3.0 on BOTH axes.
    for _ in range(5):
        supervisor._apply_ema_tracking(
            TurretCorrection(pan_delta=50.0, tilt_delta=50.0, aim_locked=False)
        )

    assert controller.pan_commands, "expected pan tracking commands"
    assert controller.tilt_commands, "expected tilt tracking commands"
    assert max(abs(v) for v in controller.pan_commands) <= 3.0 + 1e-9
    assert max(abs(v) for v in controller.tilt_commands) <= 3.0 + 1e-9
    # Tilt must not exceed pan once both clamps are equal (the bug was unclamped tilt).
    assert max(abs(v) for v in controller.tilt_commands) <= max(
        abs(v) for v in controller.pan_commands
    ) + 1e-9


def test_apply_ema_tracking_respects_independent_tilt_clamp() -> None:
    tuning = TrackingTuning(
        ema_alpha=1.0, gain=0.5, pan_clamp_deg=3.0, tilt_clamp_deg=1.0, deadband_deg=0.1
    )
    supervisor, controller = _tuning_supervisor(tuning)

    for _ in range(5):
        supervisor._apply_ema_tracking(
            TurretCorrection(pan_delta=50.0, tilt_delta=50.0, aim_locked=False)
        )

    assert max(abs(v) for v in controller.tilt_commands) <= 1.0 + 1e-9


class _AtLimitController(NullTurretController):
    """Controller stub that reports it is pinned at the tilt max limit."""

    tilt_delta_sign: int = 1
    pan_delta_sign: int = -1

    @property
    def last_status_payload(self) -> dict:
        return {"tilt_at_max": True, "tilt_at_min": False,
                "pan_at_max": False, "pan_at_min": False}


class _PinnedCornerController(NullTurretController):
    """Controller stub pinned at the physical bottom-left corner."""

    tilt_delta_sign: int = 1
    pan_delta_sign: int = -1

    def __init__(self) -> None:
        super().__init__()
        self.angle_commands: list[tuple[float, float]] = []
        self._status = {
            "tilt_at_max": True,
            "tilt_at_min": False,
            "pan_at_max": True,
            "pan_at_min": False,
        }

    @property
    def last_status_payload(self) -> dict:
        return dict(self._status)

    def set_angles(self, pan_deg: float, tilt_deg: float) -> None:
        self.angle_commands.append((pan_deg, tilt_deg))
        self._status.update(tilt_at_max=False, pan_at_max=False)


def _corner_supervisor(
    *,
    stuck_limit_seconds: float,
) -> tuple[SupervisorLoop, _PinnedCornerController, Detection]:
    controller = _PinnedCornerController()
    tuning = TrackingTuning(
        ema_alpha=1.0,
        gain=0.5,
        pan_clamp_deg=3.0,
        tilt_clamp_deg=3.0,
        deadband_deg=0.1,
        acquisition_frame_threshold=1,
        stuck_limit_seconds=stuck_limit_seconds,
    )
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
            servo_center_pan_deg=9.0,
            servo_center_tilt_deg=84.15,
        ),
        servo_limits=ServoLimits(
            pan_min_deg=-99.0,
            pan_max_deg=117.0,
            tilt_min_deg=63.15,
            tilt_max_deg=93.15,
        ),
        tracking_tuning=tuning,
    )
    supervisor = SupervisorLoop(config=config, zones=[], controller=controller)
    bottom_left_cat = Detection("cat-1", "cat", 0.92, BoundingBox(0, 160, 20, 20))
    return supervisor, controller, bottom_left_cat


def test_supervisor_recovers_target_stuck_at_corner_after_ten_seconds() -> None:
    supervisor, controller, bottom_left_cat = _corner_supervisor(
        stuck_limit_seconds=10.0
    )

    before_timeout = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[bottom_left_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=0.0,
    )
    still_waiting = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[bottom_left_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=9.999,
    )
    recovered = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[bottom_left_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=10.0,
    )

    assert before_timeout.corner_recovery_active is False
    assert still_waiting.corner_recovery_active is False
    assert recovered.corner_recovery_active is True
    assert controller.angle_commands == [(9.0, 84.15)]


def test_corner_recovery_quarantines_same_direction_until_target_clears() -> None:
    supervisor, controller, bottom_left_cat = _corner_supervisor(
        stuck_limit_seconds=0.0,
    )

    recovered = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[bottom_left_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=1.0,
    )
    controller.pan_commands.clear()
    controller.tilt_commands.clear()
    quarantined = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[bottom_left_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=1.1,
    )
    quarantined_pan_commands = list(controller.pan_commands)
    quarantined_tilt_commands = list(controller.tilt_commands)
    supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[],
        turret_frame_width=200,
        turret_frame_height=200,
        now=1.2,
    )
    resumed = supervisor.process_frame(
        [],
        frame_width=200,
        frame_height=200,
        armed=True,
        turret_detections=[bottom_left_cat],
        turret_frame_width=200,
        turret_frame_height=200,
        now=1.3,
    )

    assert recovered.corner_recovery_active is True
    assert quarantined.corner_recovery_active is True
    assert quarantined_pan_commands == []
    assert quarantined_tilt_commands == []
    assert controller.pan_commands
    assert controller.tilt_commands
    assert resumed.corner_recovery_active is False


def test_apply_ema_tracking_stops_pushing_into_tilt_limit() -> None:
    tuning = TrackingTuning(
        ema_alpha=1.0, gain=0.5, pan_clamp_deg=3.0, tilt_clamp_deg=3.0, deadband_deg=0.1
    )
    controller = _AtLimitController()
    config = SystemConfig(
        cooldown_frames=2,
        detection_policy=DetectionPolicy(
            cat_class="cat", person_class="person",
            cat_confidence_threshold=0.4, person_confidence_threshold=0.5,
            consecutive_counter_frames=2, confirmation_miss_tolerance_frames=5,
        ),
        tracking_calibration=TrackingCalibration(
            horizontal_deadband_px=20, vertical_deadband_px=20,
            horizontal_gain=0.05, vertical_gain=0.05,
            aim_offset_x_px=0, aim_offset_y_px=0,
        ),
        servo_limits=ServoLimits(),
        tracking_tuning=tuning,
    )
    supervisor = SupervisorLoop(config=config, zones=[], controller=controller)

    # Correction pushes tilt toward the max limit (positive). It must be suppressed.
    for _ in range(5):
        supervisor._apply_ema_tracking(
            TurretCorrection(pan_delta=0.0, tilt_delta=50.0, aim_locked=False)
        )

    assert controller.tilt_commands == [], "must not command past the tilt limit"
    assert supervisor._filtered_tilt == 0.0, "EMA accumulator must reset (anti-windup)"


def test_apply_ema_tracking_allows_moving_away_from_tilt_limit() -> None:
    tuning = TrackingTuning(
        ema_alpha=1.0, gain=0.5, pan_clamp_deg=3.0, tilt_clamp_deg=3.0, deadband_deg=0.1
    )
    controller = _AtLimitController()  # at MAX
    config = SystemConfig(
        cooldown_frames=2,
        detection_policy=DetectionPolicy(
            cat_class="cat", person_class="person",
            cat_confidence_threshold=0.4, person_confidence_threshold=0.5,
            consecutive_counter_frames=2, confirmation_miss_tolerance_frames=5,
        ),
        tracking_calibration=TrackingCalibration(
            horizontal_deadband_px=20, vertical_deadband_px=20,
            horizontal_gain=0.05, vertical_gain=0.05,
            aim_offset_x_px=0, aim_offset_y_px=0,
        ),
        servo_limits=ServoLimits(),
        tracking_tuning=tuning,
    )
    supervisor = SupervisorLoop(config=config, zones=[], controller=controller)

    # Negative tilt correction moves AWAY from the max limit — must be allowed.
    supervisor._apply_ema_tracking(
        TurretCorrection(pan_delta=0.0, tilt_delta=-50.0, aim_locked=False)
    )

    assert controller.tilt_commands, "moving away from the limit must be allowed"
    assert controller.tilt_commands[-1] < 0
