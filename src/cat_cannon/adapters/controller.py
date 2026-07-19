from __future__ import annotations

from dataclasses import dataclass, field

from cat_cannon.adapters.interfaces import TurretController


@dataclass
class NullTurretController(TurretController):
    pan_commands: list[float] = field(default_factory=list)
    tilt_commands: list[float] = field(default_factory=list)
    angle_commands: list[tuple[float, float]] = field(default_factory=list)
    fired: int = 0
    stopped: int = 0

    def apply_tracking_delta(self, pan_delta: float, tilt_delta: float) -> None:
        self.pan_commands.append(pan_delta)
        self.tilt_commands.append(tilt_delta)

    def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        self.pan_commands.append(pan_deg_s)
        self.tilt_commands.append(tilt_deg_s)

    def set_angles(self, pan_deg: float, tilt_deg: float) -> None:
        self.angle_commands.append((pan_deg, tilt_deg))

    def fire(self) -> None:
        self.fired += 1

    def safe_stop(self) -> None:
        self.stopped += 1
