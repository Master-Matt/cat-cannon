from cat_cannon.app.idle_motion import IdleCentering, has_fresh_detection
from cat_cannon.config import ServoLimits
from cat_cannon.domain.targeting import TrackingCalibration


class FakeAbsoluteController:
    def __init__(self) -> None:
        self.velocities: list[tuple[float, float]] = []
        self.angles: list[tuple[float, float]] = []

    def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        self.velocities.append((pan_deg_s, tilt_deg_s))

    def set_angles(self, pan_deg: float, tilt_deg: float) -> None:
        self.angles.append((pan_deg, tilt_deg))


def _calibration(*, pan_deg: float = 0.0, tilt_deg: float = 0.0) -> TrackingCalibration:
    return TrackingCalibration(
        horizontal_deadband_px=0.0,
        vertical_deadband_px=0.0,
        horizontal_gain=0.0,
        vertical_gain=0.0,
        aim_offset_x_px=0.0,
        aim_offset_y_px=0.0,
        servo_center_pan_deg=pan_deg,
        servo_center_tilt_deg=tilt_deg,
    )


def test_armed_idle_returns_to_saved_center_after_ten_seconds_once() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = _calibration(pan_deg=6.0, tilt_deg=104.0)

    started = centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=ServoLimits(),
        now=100.0,
    )
    too_early = centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=ServoLimits(),
        now=109.999,
    )
    centered = centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=ServoLimits(),
        now=110.0,
    )
    repeated = centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=ServoLimits(),
        now=111.0,
    )

    assert started is False
    assert too_early is False
    assert centered is True
    assert repeated is False
    assert controller.velocities == [(0.0, 0.0)]
    assert controller.angles == [(6.0, 104.0)]


def test_idle_centering_runs_again_after_tracking() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = _calibration(pan_deg=-12.0, tilt_deg=90.0)
    limits = ServoLimits()

    centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=limits,
        now=0.0,
    )
    centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=limits,
        now=10.0,
    )
    centering.update(
        controller=controller,
        armed=True,
        detection_present=True,
        calibration=calibration,
        limits=limits,
        now=11.0,
    )
    centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=limits,
        now=12.0,
    )
    centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=limits,
        now=22.0,
    )

    assert controller.angles == [(-12.0, 90.0), (-12.0, 90.0)]


def test_detection_resets_idle_timer_before_centering() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = _calibration(pan_deg=8.0, tilt_deg=91.0)
    limits = ServoLimits()

    for detection_present, now in (
        (True, 10.0),
        (False, 19.0),
        (True, 19.5),
        (False, 20.0),
        (False, 29.999),
    ):
        moved = centering.update(
            controller=controller,
            armed=True,
            detection_present=detection_present,
            calibration=calibration,
            limits=limits,
            now=now,
        )
        assert moved is False

    moved = centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=limits,
        now=30.0,
    )

    assert moved is True
    assert controller.angles == [(8.0, 91.0)]


def test_idle_centering_uses_axis_midpoint_for_out_of_range_center() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = _calibration()

    centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=ServoLimits(tilt_min_deg=45.0, tilt_max_deg=135.0),
        now=0.0,
    )
    centering.update(
        controller=controller,
        armed=True,
        detection_present=False,
        calibration=calibration,
        limits=ServoLimits(tilt_min_deg=45.0, tilt_max_deg=135.0),
        now=10.0,
    )

    assert controller.angles == [(0.0, 90.0)]


def test_disarmed_idle_does_not_move_turret() -> None:
    controller = FakeAbsoluteController()

    moved = IdleCentering().update(
        controller=controller,
        armed=False,
        detection_present=False,
        calibration=_calibration(),
        limits=ServoLimits(),
        now=100.0,
    )

    assert moved is False
    assert controller.velocities == []
    assert controller.angles == []


def test_stale_fixed_detection_does_not_block_idle_timer() -> None:
    assert has_fresh_detection(
        fixed_detections=[object()],
        fixed_detection_updated=False,
        turret_detections=[],
    ) is False


def test_fresh_detection_from_either_camera_resets_idle_timer() -> None:
    detection = object()

    assert has_fresh_detection(
        fixed_detections=[detection],
        fixed_detection_updated=True,
        turret_detections=[],
    ) is True
    assert has_fresh_detection(
        fixed_detections=[],
        fixed_detection_updated=False,
        turret_detections=[detection],
    ) is True
