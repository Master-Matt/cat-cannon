from __future__ import annotations

from dataclasses import dataclass

from cat_cannon.domain.models import BoundingBox


@dataclass(frozen=True)
class TrackingCalibration:
    horizontal_deadband_px: float
    vertical_deadband_px: float
    horizontal_gain: float
    vertical_gain: float
    aim_offset_x_px: float
    aim_offset_y_px: float


@dataclass(frozen=True)
class FrameSize:
    width: int
    height: int


@dataclass(frozen=True)
class TurretCorrection:
    pan_delta: float
    tilt_delta: float
    aim_locked: bool


def compute_turret_correction(
    bbox: BoundingBox,
    frame: FrameSize,
    calibration: TrackingCalibration,
) -> TurretCorrection:
    target = bbox.center
    error_x = target.x - (frame.width / 2.0 + calibration.aim_offset_x_px)
    error_y = target.y - (frame.height / 2.0 + calibration.aim_offset_y_px)

    aim_locked = (
        abs(error_x) <= calibration.horizontal_deadband_px
        and abs(error_y) <= calibration.vertical_deadband_px
    )

    if aim_locked:
        return TurretCorrection(pan_delta=0.0, tilt_delta=0.0, aim_locked=True)

    return TurretCorrection(
        pan_delta=error_x * calibration.horizontal_gain,
        tilt_delta=error_y * calibration.vertical_gain,
        aim_locked=False,
    )

