from __future__ import annotations

from dataclasses import dataclass

from cat_cannon.domain.geometry import detection_footpoint_in_zone
from cat_cannon.domain.models import CounterZone, Detection


@dataclass(frozen=True)
class DetectionPolicy:
    cat_class: str
    person_class: str
    cat_confidence_threshold: float
    person_confidence_threshold: float
    consecutive_counter_frames: int


@dataclass(frozen=True)
class SceneAssessment:
    human_present: bool
    candidate_cat: Detection | None
    cat_on_counter: bool
    active_zone_id: str | None


class CounterConfirmation:
    def __init__(self, required_frames: int) -> None:
        self._required_frames = required_frames
        self._current_track_id: str | None = None
        self._counter = 0

    def update(self, cat: Detection | None, is_on_counter: bool) -> bool:
        if cat is None or not is_on_counter:
            self._current_track_id = None
            self._counter = 0
            return False

        if cat.track_id != self._current_track_id:
            self._current_track_id = cat.track_id
            self._counter = 1
        else:
            self._counter += 1

        return self._counter >= self._required_frames


def assess_scene(
    detections: list[Detection],
    zones: list[CounterZone],
    policy: DetectionPolicy,
) -> SceneAssessment:
    human_present = any(
        detection.label == policy.person_class and detection.confidence >= policy.person_confidence_threshold
        for detection in detections
    )

    cats = [
        detection
        for detection in detections
        if detection.label == policy.cat_class and detection.confidence >= policy.cat_confidence_threshold
    ]
    cats.sort(key=lambda detection: detection.confidence, reverse=True)

    for cat in cats:
        for zone in zones:
            if detection_footpoint_in_zone(cat, zone):
                return SceneAssessment(
                    human_present=human_present,
                    candidate_cat=cat,
                    cat_on_counter=True,
                    active_zone_id=zone.zone_id,
                )

    return SceneAssessment(
        human_present=human_present,
        candidate_cat=cats[0] if cats else None,
        cat_on_counter=False,
        active_zone_id=None,
    )

