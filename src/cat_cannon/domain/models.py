from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def bottom_center(self) -> Point:
        return Point(self.x + self.width / 2.0, self.y + self.height)


@dataclass(frozen=True)
class Detection:
    track_id: str
    label: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class CounterZone:
    zone_id: str
    polygon: tuple[Point, ...]


class SupervisorState(str, Enum):
    DISARMED = "disarmed"
    IDLE = "idle"
    HUMAN_LOCKOUT = "human_lockout"
    COUNTER_CONFIRMED = "counter_confirmed"
    TURRET_ACQUIRE = "turret_acquire"
    TRACKING = "tracking"
    AIM_LOCK = "aim_lock"
    FIRE = "fire"
    COOLDOWN = "cooldown"
    FAULT = "fault"

