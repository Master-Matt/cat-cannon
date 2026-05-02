from pathlib import Path


def test_x11_tracking_runner_checks_display_and_launches_tracking_ui() -> None:
    script = Path("scripts/run_tracking_test_x11.sh").read_text(encoding="utf-8")

    assert "${DISPLAY:-}" in script
    assert "ssh -Y" in script
    assert "-m cat_cannon.app.tracking_test" in script
    assert "--fixed-camera" in script
    assert "--turret-camera" in script
    assert 'CAT_CANNON_FIXED_CAMERA:-/dev/fixed_cam' in script
    assert 'CAT_CANNON_TURRET_CAMERA:-/dev/turret_cam' in script
