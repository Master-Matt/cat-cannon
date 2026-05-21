from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from cat_cannon.domain.models import CounterZone, Point
from cat_cannon.domain.safety import DetectionPolicy
from cat_cannon.domain.targeting import TrackingCalibration

DEFAULT_VISION_YOLO_IMGSZ = 640
DEFAULT_VISION_YOLOE_MODEL = "yoloe-11s-seg.pt"


@dataclass(frozen=True)
class YoloPrompt:
    label: str
    text: str


DEFAULT_YOLOE_PROMPTS: tuple[YoloPrompt, ...] = (
    YoloPrompt(label="person", text="person"),
    YoloPrompt(label="cat", text="cat"),
)


@dataclass(frozen=True)
class VisionConfig:
    yolo_imgsz: int = DEFAULT_VISION_YOLO_IMGSZ
    yolo_detector: str = "yolo"
    yolo_model: str = ""
    yoloe_model: str = DEFAULT_VISION_YOLOE_MODEL
    yoloe_prompts: tuple[YoloPrompt, ...] = DEFAULT_YOLOE_PROMPTS

    @property
    def selected_model_path(self) -> str:
        if self.yolo_detector == "yoloe":
            return self.yoloe_model
        return self.yolo_model

    @property
    def prompt_texts(self) -> tuple[str, ...]:
        return tuple(prompt.text for prompt in self.yoloe_prompts)

    @property
    def prompt_label_aliases(self) -> dict[str, str]:
        return {prompt.text: prompt.label for prompt in self.yoloe_prompts}


@dataclass(frozen=True)
class TrackingTuning:
    ema_alpha: float = 0.35
    gain: float = 0.5
    pan_clamp_deg: float = 3.0
    deadband_deg: float = 0.1
    frame_wait_ms: int = 10


@dataclass(frozen=True)
class HumanLockoutConfig:
    window_seconds: float = 2.0
    frame_threshold: int = 10


@dataclass(frozen=True)
class EventRecordingConfig:
    enabled: bool = False
    output_dir: str = "data/event_videos"
    post_shot_seconds: float = 15.0
    zone_confirm_seconds: float = 5.0
    zone_confirm_detections: int = 20
    zone_lost_seconds: float = 5.0
    max_event_seconds: float = 180.0
    discord_webhook_url: str = ""
    discord_webhook_env: str = "CAT_CANNON_DISCORD_WEBHOOK_URL"

    def resolved_discord_webhook_url(self) -> str:
        if self.discord_webhook_url.strip():
            return self.discord_webhook_url.strip()
        return os.environ.get(self.discord_webhook_env, "").strip()


@dataclass(frozen=True)
class ServoLimits:
    pan_min_deg: float = 0.0
    pan_max_deg: float = 180.0
    tilt_min_deg: float = 30.0
    tilt_max_deg: float = 150.0
    pan_left_deg: float | None = None
    pan_right_deg: float | None = None
    tilt_top_deg: float | None = None
    tilt_bottom_deg: float | None = None

    def normalized(self) -> ServoLimits:
        pan_min = min(self.pan_min_deg, self.pan_max_deg)
        pan_max = max(self.pan_min_deg, self.pan_max_deg)
        tilt_min = min(self.tilt_min_deg, self.tilt_max_deg)
        tilt_max = max(self.tilt_min_deg, self.tilt_max_deg)
        return ServoLimits(
            pan_min_deg=pan_min,
            pan_max_deg=pan_max,
            tilt_min_deg=tilt_min,
            tilt_max_deg=tilt_max,
            pan_left_deg=self.pan_left_deg,
            pan_right_deg=self.pan_right_deg,
            tilt_top_deg=self.tilt_top_deg,
            tilt_bottom_deg=self.tilt_bottom_deg,
        )

    @property
    def pan_delta_sign(self) -> int:
        if self.pan_left_deg is None or self.pan_right_deg is None:
            return -1
        return 1 if self.pan_right_deg > self.pan_left_deg else -1

    @property
    def tilt_delta_sign(self) -> int:
        if self.tilt_top_deg is None or self.tilt_bottom_deg is None:
            return 1
        return 1 if self.tilt_bottom_deg > self.tilt_top_deg else -1


@dataclass(frozen=True)
class SystemConfig:
    cooldown_frames: int
    detection_policy: DetectionPolicy
    tracking_calibration: TrackingCalibration
    fire_cooldown_seconds: float | None = None
    vision: VisionConfig = VisionConfig()
    tracking_tuning: TrackingTuning = TrackingTuning()
    human_lockout: HumanLockoutConfig = HumanLockoutConfig()
    servo_limits: ServoLimits = ServoLimits()


def load_system_config(path: str | Path) -> SystemConfig:
    raw = _load_yaml(path)
    detection = raw["detection"]
    tracking = raw["tracking"]
    system = raw["system"]
    tuning = raw.get("tracking_tuning", {})

    servo_limits = ServoLimits(
        pan_min_deg=float(tracking.get("servo_min_pan_deg", 0.0)),
        pan_max_deg=float(tracking.get("servo_max_pan_deg", 180.0)),
        tilt_min_deg=float(tracking.get("servo_min_tilt_deg", 30.0)),
        tilt_max_deg=float(tracking.get("servo_max_tilt_deg", 150.0)),
        pan_left_deg=_optional_float(tracking.get("servo_left_pan_deg")),
        pan_right_deg=_optional_float(tracking.get("servo_right_pan_deg")),
        tilt_top_deg=_optional_float(tracking.get("servo_top_tilt_deg")),
        tilt_bottom_deg=_optional_float(tracking.get("servo_bottom_tilt_deg")),
    ).normalized()

    return SystemConfig(
        cooldown_frames=int(system["cooldown_frames"]),
        fire_cooldown_seconds=_optional_float(system.get("fire_cooldown_seconds")),
        vision=_vision_config_from_raw(raw),
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
            ema_alpha=float(tuning.get("ema_alpha", 0.35)),
            gain=float(tuning.get("gain", 0.5)),
            pan_clamp_deg=float(tuning.get("pan_clamp_deg", 3.0)),
            deadband_deg=float(tuning.get("deadband_deg", 0.1)),
            frame_wait_ms=int(tuning.get("frame_wait_ms", 10)),
        ),
        human_lockout=_human_lockout_config_from_raw(raw),
        servo_limits=servo_limits,
    )


def load_vision_config(path: str | Path) -> VisionConfig:
    try:
        raw = _load_yaml(path)
    except FileNotFoundError:
        return VisionConfig()
    return _vision_config_from_raw(raw)


def load_event_recording_config(path: str | Path) -> EventRecordingConfig:
    try:
        raw = _load_yaml(path)
    except FileNotFoundError:
        return EventRecordingConfig()
    return _event_recording_config_from_raw(raw)


def load_counter_zones(path: str | Path) -> list[CounterZone]:
    raw = _load_yaml(path)
    root_frame = _frame_size(raw.get("frame"))
    zones: list[CounterZone] = []
    for zone in raw["zones"]:
        zone_frame = _frame_size(zone.get("frame"))
        if zone_frame == (None, None):
            zone_frame = root_frame
        reference_width, reference_height = zone_frame
        if "normalized_points" in zone:
            if reference_width is None or reference_height is None:
                raise ValueError("normalized counter zones require frame width and height")
            points = tuple(
                Point(float(x) * reference_width, float(y) * reference_height)
                for x, y in zone["normalized_points"]
            )
        else:
            points = tuple(Point(float(x), float(y)) for x, y in zone["points"])
        zones.append(
            CounterZone(
                zone_id=str(zone["id"]),
                polygon=points,
                reference_width=reference_width,
                reference_height=reference_height,
            )
        )
    return zones


def save_counter_zones(
    path: str | Path,
    zones: list[CounterZone],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> None:
    payload: dict[str, object] = {"zones": []}
    if frame_width is not None and frame_height is not None:
        payload["frame"] = {"width": int(frame_width), "height": int(frame_height)}

    serialized_zones: list[dict[str, object]] = []
    for zone in zones:
        zone_payload: dict[str, object] = {
            "id": zone.zone_id,
            "points": [[point.x, point.y] for point in zone.polygon],
        }
        reference_width = frame_width or zone.reference_width
        reference_height = frame_height or zone.reference_height
        if reference_width is not None and reference_height is not None:
            zone_payload["normalized_points"] = [
                [point.x / reference_width, point.y / reference_height]
                for point in zone.polygon
            ]
        serialized_zones.append(zone_payload)
    payload["zones"] = serialized_zones

    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def scale_counter_zones(
    zones: list[CounterZone],
    *,
    frame_width: int,
    frame_height: int,
) -> list[CounterZone]:
    scaled: list[CounterZone] = []
    for zone in zones:
        if not zone.reference_width or not zone.reference_height:
            scaled.append(CounterZone(zone_id=zone.zone_id, polygon=zone.polygon))
            continue
        scale_x = frame_width / zone.reference_width
        scale_y = frame_height / zone.reference_height
        scaled.append(
            CounterZone(
                zone_id=zone.zone_id,
                polygon=tuple(
                    Point(point.x * scale_x, point.y * scale_y)
                    for point in zone.polygon
                ),
            )
        )
    return scaled


def save_servo_center(path: str | Path, pan_deg: float, tilt_deg: float) -> None:
    raw = _load_yaml(path)
    tracking = raw.setdefault("tracking", {})
    tracking["servo_center_pan_deg"] = round(pan_deg, 2)
    tracking["servo_center_tilt_deg"] = round(tilt_deg, 2)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)


def save_servo_limit(path: str | Path, key: str, angle_deg: float) -> ServoLimits:
    valid_keys = {
        "servo_min_pan_deg",
        "servo_max_pan_deg",
        "servo_min_tilt_deg",
        "servo_max_tilt_deg",
    }
    if key not in valid_keys:
        raise ValueError(f"Unknown servo limit key: {key}")

    raw = _load_yaml(path)
    tracking = raw.setdefault("tracking", {})
    tracking[key] = round(float(angle_deg), 2)
    limits = _servo_limits_from_tracking(tracking)
    tracking["servo_min_pan_deg"] = limits.pan_min_deg
    tracking["servo_max_pan_deg"] = limits.pan_max_deg
    tracking["servo_min_tilt_deg"] = limits.tilt_min_deg
    tracking["servo_max_tilt_deg"] = limits.tilt_max_deg
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return limits


def save_servo_limit_calibration(
    path: str | Path,
    *,
    top_tilt_deg: float,
    bottom_tilt_deg: float,
    left_pan_deg: float,
    right_pan_deg: float,
) -> ServoLimits:
    raw = _load_yaml(path)
    tracking = raw.setdefault("tracking", {})
    tracking["servo_top_tilt_deg"] = round(float(top_tilt_deg), 2)
    tracking["servo_bottom_tilt_deg"] = round(float(bottom_tilt_deg), 2)
    tracking["servo_left_pan_deg"] = round(float(left_pan_deg), 2)
    tracking["servo_right_pan_deg"] = round(float(right_pan_deg), 2)

    limits = ServoLimits(
        pan_min_deg=min(left_pan_deg, right_pan_deg),
        pan_max_deg=max(left_pan_deg, right_pan_deg),
        tilt_min_deg=min(top_tilt_deg, bottom_tilt_deg),
        tilt_max_deg=max(top_tilt_deg, bottom_tilt_deg),
        pan_left_deg=float(left_pan_deg),
        pan_right_deg=float(right_pan_deg),
        tilt_top_deg=float(top_tilt_deg),
        tilt_bottom_deg=float(bottom_tilt_deg),
    ).normalized()
    tracking["servo_min_pan_deg"] = limits.pan_min_deg
    tracking["servo_max_pan_deg"] = limits.pan_max_deg
    tracking["servo_min_tilt_deg"] = limits.tilt_min_deg
    tracking["servo_max_tilt_deg"] = limits.tilt_max_deg
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return limits


def clear_servo_limits(path: str | Path) -> ServoLimits:
    raw = _load_yaml(path)
    tracking = raw.setdefault("tracking", {})
    _remove_servo_limit_keys(tracking)
    limits = ServoLimits()
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return limits


def clear_servo_calibration(path: str | Path) -> ServoLimits:
    raw = _load_yaml(path)
    tracking = raw.setdefault("tracking", {})
    _remove_servo_limit_keys(tracking)
    tracking.pop("servo_center_pan_deg", None)
    tracking.pop("servo_center_tilt_deg", None)
    limits = ServoLimits()
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw, handle, sort_keys=False)
    return limits


def _remove_servo_limit_keys(tracking: dict) -> None:
    for key in (
        "servo_min_pan_deg",
        "servo_max_pan_deg",
        "servo_min_tilt_deg",
        "servo_max_tilt_deg",
        "servo_left_pan_deg",
        "servo_right_pan_deg",
        "servo_top_tilt_deg",
        "servo_bottom_tilt_deg",
    ):
        tracking.pop(key, None)


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return data


def _vision_config_from_raw(raw: dict) -> VisionConfig:
    vision = raw.get("vision", {})
    if vision is None:
        vision = {}
    if not isinstance(vision, dict):
        raise ValueError("Expected mapping config at vision")
    return VisionConfig(
        yolo_imgsz=int(vision.get("yolo_imgsz", DEFAULT_VISION_YOLO_IMGSZ)),
        yolo_detector=_normalize_yolo_detector(vision.get("yolo_detector", "yolo")),
        yolo_model=str(vision.get("yolo_model", "") or ""),
        yoloe_model=str(
            vision.get("yoloe_model", DEFAULT_VISION_YOLOE_MODEL) or DEFAULT_VISION_YOLOE_MODEL
        ),
        yoloe_prompts=_parse_yoloe_prompts(vision.get("yoloe_prompts")),
    )


def _event_recording_config_from_raw(raw: dict) -> EventRecordingConfig:
    event = raw.get("event_recording", {})
    if event is None:
        event = {}
    if not isinstance(event, dict):
        raise ValueError("Expected mapping config at event_recording")
    return EventRecordingConfig(
        enabled=bool(event.get("enabled", False)),
        output_dir=str(event.get("output_dir", "data/event_videos") or "data/event_videos"),
        post_shot_seconds=float(event.get("post_shot_seconds", 15.0)),
        zone_confirm_seconds=float(event.get("zone_confirm_seconds", 5.0)),
        zone_confirm_detections=int(event.get("zone_confirm_detections", 20)),
        zone_lost_seconds=float(event.get("zone_lost_seconds", 5.0)),
        max_event_seconds=float(event.get("max_event_seconds", 180.0)),
        discord_webhook_url=str(event.get("discord_webhook_url", "") or ""),
        discord_webhook_env=str(
            event.get("discord_webhook_env", "CAT_CANNON_DISCORD_WEBHOOK_URL")
            or "CAT_CANNON_DISCORD_WEBHOOK_URL"
        ),
    )


def _human_lockout_config_from_raw(raw: dict) -> HumanLockoutConfig:
    lockout = raw.get("human_lockout", {})
    if lockout is None:
        lockout = {}
    if not isinstance(lockout, dict):
        raise ValueError("Expected mapping config at human_lockout")
    return HumanLockoutConfig(
        window_seconds=max(0.0, float(lockout.get("window_seconds", 2.0))),
        frame_threshold=max(1, int(lockout.get("frame_threshold", 10))),
    )


def _normalize_yolo_detector(value: object) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "standard": "yolo",
        "yolo": "yolo",
        "yolo11": "yolo",
        "yoloe": "yoloe",
        "yolo-e": "yoloe",
        "yolo11e": "yoloe",
        "yolo11-e": "yoloe",
    }
    if normalized not in aliases:
        raise ValueError("vision.yolo_detector must be one of: yolo, yoloe")
    return aliases[normalized]


def _parse_yoloe_prompts(raw_prompts: object) -> tuple[YoloPrompt, ...]:
    if raw_prompts is None:
        return DEFAULT_YOLOE_PROMPTS

    prompts: list[YoloPrompt] = []
    if isinstance(raw_prompts, dict):
        for label, values in raw_prompts.items():
            prompt_values = values if isinstance(values, list) else [values]
            for text in prompt_values:
                if text is not None and str(text).strip():
                    prompts.append(YoloPrompt(label=str(label), text=str(text)))
    elif isinstance(raw_prompts, list):
        for item in raw_prompts:
            if isinstance(item, dict):
                label = item.get("label")
                text = item.get("text", item.get("prompt"))
                if label is None or text is None:
                    raise ValueError("vision.yoloe_prompts list items require label and text")
                prompts.append(YoloPrompt(label=str(label), text=str(text)))
            elif item is not None and str(item).strip():
                text = str(item)
                prompts.append(YoloPrompt(label=text, text=text))
    else:
        raise ValueError("vision.yoloe_prompts must be a mapping or list")

    if not prompts:
        raise ValueError("vision.yoloe_prompts must contain at least one prompt")
    return tuple(prompts)


def _servo_limits_from_tracking(tracking: dict) -> ServoLimits:
    return ServoLimits(
        pan_min_deg=float(tracking.get("servo_min_pan_deg", 0.0)),
        pan_max_deg=float(tracking.get("servo_max_pan_deg", 180.0)),
        tilt_min_deg=float(tracking.get("servo_min_tilt_deg", 30.0)),
        tilt_max_deg=float(tracking.get("servo_max_tilt_deg", 150.0)),
        pan_left_deg=_optional_float(tracking.get("servo_left_pan_deg")),
        pan_right_deg=_optional_float(tracking.get("servo_right_pan_deg")),
        tilt_top_deg=_optional_float(tracking.get("servo_top_tilt_deg")),
        tilt_bottom_deg=_optional_float(tracking.get("servo_bottom_tilt_deg")),
    ).normalized()


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _frame_size(raw_frame: object) -> tuple[float | None, float | None]:
    if raw_frame is None:
        return None, None
    if not isinstance(raw_frame, dict):
        raise ValueError("counter zone frame must be a mapping")
    width = raw_frame.get("width")
    height = raw_frame.get("height")
    if width is None or height is None:
        return None, None
    return float(width), float(height)
