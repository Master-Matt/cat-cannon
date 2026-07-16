from cat_cannon.app.idle_motion import IdleCentering
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


def test_armed_idle_returns_to_saved_center_once() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = TrackingCalibration(
        servo_center_pan_deg=6.0,
        servo_center_tilt_deg=104.0,
    )

    first = centering.update(
        controller=controller,
        armed=True,
        idle=True,
        calibration=calibration,
        limits=ServoLimits(),
    )
    second = centering.update(
        controller=controller,
        armed=True,
        idle=True,
        calibration=calibration,
        limits=ServoLimits(),
    )

    assert first is True
    assert second is False
    assert controller.velocities == [(0.0, 0.0)]
    assert controller.angles == [(6.0, 104.0)]


def test_idle_centering_runs_again_after_tracking() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = TrackingCalibration(
        servo_center_pan_deg=-12.0,
        servo_center_tilt_deg=90.0,
    )
    limits = ServoLimits()

    centering.update(
        controller=controller,
        armed=True,
        idle=True,
        calibration=calibration,
        limits=limits,
    )
    centering.update(
        controller=controller,
        armed=True,
        idle=False,
        calibration=calibration,
        limits=limits,
    )
    centering.update(
        controller=controller,
        armed=True,
        idle=True,
        calibration=calibration,
        limits=limits,
    )

    assert controller.angles == [(-12.0, 90.0), (-12.0, 90.0)]


def test_idle_centering_uses_axis_midpoint_for_out_of_range_center() -> None:
    controller = FakeAbsoluteController()
    centering = IdleCentering()
    calibration = TrackingCalibration(
        servo_center_pan_deg=0.0,
        servo_center_tilt_deg=0.0,
    )

    centering.update(
        controller=controller,
        armed=True,
        idle=True,
        calibration=calibration,
        limits=ServoLimits(tilt_min_deg=45.0, tilt_max_deg=135.0),
    )

    assert controller.angles == [(0.0, 90.0)]


def test_disarmed_idle_does_not_move_turret() -> None:
    controller = FakeAbsoluteController()

    moved = IdleCentering().update(
        controller=controller,
        armed=False,
        idle=True,
        calibration=TrackingCalibration(),
        limits=ServoLimits(),
    )

    assert moved is False
    assert controller.velocities == []
    assert controller.angles == []
