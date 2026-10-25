from pathlib import Path


def test_runtime_launchers_keep_logs_outside_deployment_directory() -> None:
    for script_name in ("run_app.sh", "run_guardian.sh"):
        script = Path("scripts", script_name).read_text(encoding="utf-8")

        assert "${XDG_STATE_HOME:-$HOME/.local/state}/cat-cannon" in script
        assert 'LOG_FILE="$STATE_DIR/' in script
        assert 'LOG_FILE="$ROOT_DIR/' not in script
