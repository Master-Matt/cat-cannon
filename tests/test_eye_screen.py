from cat_cannon.app.eye_screen import (
    EyeState,
    FixedDetectionCadence,
    update_eye_gaze_target,
)
from cat_cannon.domain.models import BoundingBox, Detection
from cat_cannon.domain.safety import DetectionPolicy


def test_eye_gaze_recenters_when_current_detection_clears() -> None:
    state = EyeState(
        last_detected=True,
        target_gaze_x=0.7,
        target_gaze_y=-0.4,
    )

    update_eye_gaze_target(state, None)

    assert state.target_gaze_x == 0.0
    assert state.target_gaze_y == 0.0


def test_idle_random_gaze_is_preserved_without_a_prior_detection() -> None:
    state = EyeState(
        last_detected=False,
        target_gaze_x=0.5,
        target_gaze_y=0.2,
    )

    update_eye_gaze_target(state, None)

    assert state.target_gaze_x == 0.5
    assert state.target_gaze_y == 0.2


def test_fixed_detection_switches_to_burst_rate_after_valid_target() -> None:
    cadence = FixedDetectionCadence(
        normal_interval_frames=5,
        burst_seconds=1.0,
    )
    policy = DetectionPolicy(
        cat_class="cat",
        person_class="person",
        cat_confidence_threshold=0.45,
        person_confidence_threshold=0.55,
        consecutive_counter_frames=3,
    )
    cat = Detection("cat-1", "cat", 0.9, BoundingBox(10, 10, 20, 20))

    assert cadence.should_detect(frame_counter=0, now=0.0) is True
    cadence.observe(detections=[cat], policy=policy, now=0.0)

    assert all(
        cadence.should_detect(frame_counter=frame, now=frame * 0.2)
        for frame in range(1, 5)
    )
    assert cadence.should_detect(frame_counter=6, now=1.01) is False
