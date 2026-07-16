from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from cat_cannon.config import ServoLimits
from cat_cannon.domain.targeting import TrackingCalibration

RETURN_TO_CENTER_DELAY_SECONDS = 30.0


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
    """Return the turret to center after a continuous armed-idle period."""

    return_delay_seconds: float = RETURN_TO_CENTER_DELAY_SECONDS
    _center_commanded: bool = field(default=False, init=False)
    _idle_started_at: float | None = field(default=None, init=False)

    def update(
        self,
        *,
        controller: AbsoluteTurretController,
        armed: bool,
        idle: bool,
        calibration: TrackingCalibration,
        limits: ServoLimits,
        now: float | None = None,
    ) -> bool:
        if not armed or not idle:
            self._center_commanded = False
            self._idle_started_at = None
            return False
        if self._center_commanded:
            return False

        now_s = time.monotonic() if now is None else float(now)
        if self._idle_started_at is None:
            self._idle_started_at = now_s
        idle_seconds = max(0.0, now_s - self._idle_started_at)
        if idle_seconds < max(0.0, self.return_delay_seconds):
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
