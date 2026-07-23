from cat_cannon.app.eye_screen import EyeState, update_eye_gaze_target


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
