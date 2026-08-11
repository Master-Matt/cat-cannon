from threading import Event, Thread
from types import SimpleNamespace

from cat_cannon.app.eye_screen import (
    EyeState,
    FixedDetectionCadence,
    _set_x11_kiosk_window_properties,
    shutdown_eye_background_resources,
    update_eye_gaze_target,
    update_eye_mode,
)
from cat_cannon.domain.models import BoundingBox, Detection, SupervisorState
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


def test_x11_kiosk_window_is_borderless_and_bypasses_compositor() -> None:
    property_calls = []
    target = SimpleNamespace(
        change_property=lambda *args: property_calls.append(args),
    )

    _set_x11_kiosk_window_properties(
        target,
        motif_hints="motif",
        bypass_compositor="bypass",
        cardinal="cardinal",
    )

    assert property_calls == [
        ("motif", "motif", 32, [2, 0, 0, 0, 0]),
        ("bypass", "cardinal", 32, [1]),
    ]


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


def test_eye_shows_tracking_for_acquired_turret_camera_cat() -> None:
    state = EyeState()
    result = SimpleNamespace(
        state=SupervisorState.IDLE,
        human_present=False,
        target_visible=False,
        turret_target_visible=True,
    )

    update_eye_mode(state, result, now=10.0)

    assert state.mode == "tracking"
    assert state.last_detected is True
    assert state.last_detection_time == 10.0


def test_eye_shutdown_releases_resources_created_during_navigation() -> None:
    cancelled = Event()
    resources = {}
    camera = SimpleNamespace(release_calls=0)

    def release() -> None:
        camera.release_calls += 1

    camera.release = release

    def finish_loading_after_navigation() -> None:
        cancelled.wait()
        resources["turret_camera"] = camera

    loader = Thread(target=finish_loading_after_navigation)
    loader.start()

    shutdown_eye_background_resources(
        resources=resources,
        loader_cancelled=cancelled,
        workers=(loader,),
    )

    assert loader.is_alive() is False
    assert camera.release_calls == 1
