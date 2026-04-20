from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cat_cannon.domain.models import CounterZone, Point
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration


@dataclass(frozen=True)
class SystemConfig:
    cooldown_frames: int
    detection_policy: DetectionPolicy
    tracking_calibration: TrackingCalibration


def load_system_config(path: str | Path) -> SystemConfig:
    raw = _load_yaml(path)
    detection = raw["detection"]
    tracking = raw["tracking"]
    system = raw["system"]

    return SystemConfig(
        cooldown_frames=int(system["cooldown_frames"]),
        detection_policy=DetectionPolicy(
            cat_class=str(detection["cat_class"]),
            person_class=str(detection["person_class"]),
            cat_confidence_threshold=float(detection["cat_confidence_threshold"]),
            person_confidence_threshold=float(detection["person_confidence_threshold"]),
            consecutive_counter_frames=int(detection["consecutive_counter_frames"]),
        ),
        tracking_calibration=TrackingCalibration(
            horizontal_deadband_px=float(tracking["horizontal_deadband_px"]),
            vertical_deadband_px=float(tracking["vertical_deadband_px"]),
            horizontal_gain=float(tracking["horizontal_gain"]),
            vertical_gain=float(tracking["vertical_gain"]),
            aim_offset_x_px=float(tracking["aim_offset_x_px"]),
            aim_offset_y_px=float(tracking["aim_offset_y_px"]),
        ),
    )


def load_counter_zones(path: str | Path) -> list[CounterZone]:
    raw = _load_yaml(path)
    zones: list[CounterZone] = []
    for zone in raw["zones"]:
        zones.append(
            CounterZone(
                zone_id=str(zone["id"]),
                polygon=tuple(Point(float(x), float(y)) for x, y in zone["points"]),
            )
        )
    return zones


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data

