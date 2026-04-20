from cat_cannon.domain.models import BoundingBox, CounterZone, Detection, Point
from cat_cannon.domain.safety import CounterConfirmation, DetectionPolicy, assess_scene


def _policy() -> DetectionPolicy:
    return DetectionPolicy(
        cat_class="cat",
        person_class="person",
        cat_confidence_threshold=0.4,
        person_confidence_threshold=0.5,
        consecutive_counter_frames=3,
    )


def _zone() -> CounterZone:
    return CounterZone(
        zone_id="counter",
        polygon=(
            Point(0, 20),
            Point(100, 20),
            Point(100, 100),
            Point(0, 100),
        ),
    )


def test_assess_scene_flags_human_presence_even_when_cat_is_on_counter() -> None:
    detections = [
        Detection("cat-1", "cat", 0.8, BoundingBox(10, 10, 10, 20)),
        Detection("person-1", "person", 0.9, BoundingBox(50, 10, 20, 60)),
    ]

    assessment = assess_scene(detections, [_zone()], _policy())

    assert assessment.human_present is True
    assert assessment.cat_on_counter is True
    assert assessment.active_zone_id == "counter"


def test_counter_confirmation_requires_repeated_frames_for_same_track() -> None:
    confirmation = CounterConfirmation(required_frames=3)
    cat = Detection("cat-1", "cat", 0.9, BoundingBox(10, 10, 10, 20))

    assert confirmation.update(cat, True) is False
    assert confirmation.update(cat, True) is False
    assert confirmation.update(cat, True) is True

