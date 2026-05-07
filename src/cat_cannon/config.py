from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cat_cannon.domain.models import CounterZone, Point
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration


@dataclass(frozen=True)
class TrackingTuning:
    ema_alpha: float = 0.35
    gain: float = 0.5
    pan_clamp_deg: float = 3.0
    deadband_deg: float = 0.3
    frame_wait_ms: int = 10


@dataclass(frozen=True)
class SystemConfig:
    cooldown_frames: int
    detection_policy: DetectionPolicy
    tracking_calibration: TrackingCalibration
    tracking_tuning: TrackingTuning = TrackingTuning()


def load_system_config(path: str | Path) -> SystemConfig:
    raw = _load_yaml(path)
    detection = raw["detection"]
    tracking = raw["tracking"]
    system = raw["system"]
    tuning = raw.get("tracking_tuning", {})

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
            servo_center_pan_deg=float(tracking.get("servo_center_pan_deg", 0)),
            servo_center_tilt_deg=float(tracking.get("servo_center_tilt_deg", 0)),
        ),
        tracking_tuning=TrackingTuning(
            ema_alpha=float(tuning.get("ema_alpha", 0.5)),
            gain=float(tuning.get("gain", 0.7)),
            pan_clamp_deg=float(tuning.get("pan_clamp_deg", 4.0)),
            deadband_deg=float(tuning.get("deadband_deg", 0.3)),
            frame_wait_ms=int(tuning.get("frame_wait_ms", 10)),
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


def save_counter_zones(path: str | Path, zones: list[CounterZone]) -> None:
    payload = {
        "zones": [
            {
                "id": zone.zone_id,
                "points": [[point.x, point.y] for point in zone.polygon],
            }
            for zone in zones
        ]
    }
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def save_servo_center(path: str | Path, pan_deg: float, tilt_deg: float) -> None:
    """Update the servo center in the app config YAML without rewriting the whole file."""
    raw = _load_yaml(path)
    if "tracking" not in raw:
        raw["tracking"] = {}
    raw["tracking"]["servo_center_pan_deg"] = round(pan_deg, 2)
    raw["tracking"]["servo_center_tilt_deg"] = round(tilt_deg, 2)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data
