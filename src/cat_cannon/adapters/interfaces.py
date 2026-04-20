from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cat_cannon.domain.models import Detection


@dataclass(frozen=True)
class PerceptionFrame:
    source_id: str
    width: int
    height: int
    detections: list[Detection]


class PerceptionAdapter(Protocol):
    def read_frame(self) -> PerceptionFrame:
        """Return the latest frame metadata and detections."""


class TurretController(Protocol):
    def apply_tracking_delta(self, pan_delta: float, tilt_delta: float) -> None:
        """Apply bounded movement commands to the turret."""

    def fire(self) -> None:
        """Trigger one bounded actuation pulse."""

    def safe_stop(self) -> None:
        """Force the controller into a non-firing safe state."""

