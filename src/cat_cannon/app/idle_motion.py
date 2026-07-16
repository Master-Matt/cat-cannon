from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cat_cannon.config import ServoLimits
from cat_cannon.domain.targeting import TrackingCalibration


class AbsoluteTurretController(Protocol):
    def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        ...

    def set_angles(self, pan_deg: float, tilt_deg: float) -> object:
        ...


def _configured_center_or_midpoint(center: float, minimum: float, maximum: float) -> float:
    lower = min(minimum, maximum)
    upper = max(minimum, maximum)
    if lower <= center <= upper:
        return float(center)
    return (lower + upper) / 2.0


@dataclass
class IdleCentering:
    """Return the turret to its configured center once per armed-idle period."""

    _center_commanded: bool = field(default=False, init=False)

    def update(
        self,
        *,
        controller: AbsoluteTurretController,
        armed: bool,
        idle: bool,
        calibration: TrackingCalibration,
        limits: ServoLimits,
    ) -> bool:
        if not armed or not idle:
            self._center_commanded = False
            return False
        if self._center_commanded:
            return False

        normalized_limits = limits.normalized()
        pan_center = _configured_center_or_midpoint(
            calibration.servo_center_pan_deg,
            normalized_limits.pan_min_deg,
            normalized_limits.pan_max_deg,
        )
        tilt_center = _configured_center_or_midpoint(
            calibration.servo_center_tilt_deg,
            normalized_limits.tilt_min_deg,
            normalized_limits.tilt_max_deg,
        )

        try:
            controller.set_velocity(0.0, 0.0)
            controller.set_angles(pan_center, tilt_center)
        except Exception:
            return False

        self._center_commanded = True
        return True
