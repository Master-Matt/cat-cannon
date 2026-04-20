from __future__ import annotations

from dataclasses import dataclass, field

from cat_cannon.adapters.interfaces import TurretController


@dataclass
class NullTurretController(TurretController):
    pan_commands: list[float] = field(default_factory=list)
    tilt_commands: list[float] = field(default_factory=list)
    fired: int = 0
    stopped: int = 0

    def apply_tracking_delta(self, pan_delta: float, tilt_delta: float) -> None:
        self.pan_commands.append(pan_delta)
        self.tilt_commands.append(tilt_delta)

    def fire(self) -> None:
        self.fired += 1

    def safe_stop(self) -> None:
        self.stopped += 1

