from __future__ import annotations

import pytest

from cat_cannon.app.camera_recovery import CameraRecoveryError, RecoveringCamera


class _ScriptedCamera:
    def __init__(
        self,
        reads: list[tuple[bool, object | None] | Exception],
    ) -> None:
        self._reads = list(reads)
        self.release_calls = 0

    def read(self) -> tuple[bool, object | None]:
        if not self._reads:
            return False, None
        result = self._reads.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def release(self) -> None:
        self.release_calls += 1


def test_recovering_camera_reopens_after_consecutive_failed_reads() -> None:
    stale = _ScriptedCamera([(False, None), (False, None), (False, None)])
    recovered = _ScriptedCamera([(True, "fresh-frame")])
    sleeps: list[float] = []
    keepalive_calls: list[bool] = []
    camera = RecoveringCamera(
        stale,
        reopen=lambda: recovered,
        name="turret",
        max_consecutive_read_failures=3,
        max_reopen_attempts=2,
        retry_delay_s=0.5,
        sleep=sleeps.append,
        keepalive=lambda: keepalive_calls.append(True),
    )

    assert camera.read() == (False, None)
    assert camera.read() == (False, None)
    assert camera.read() == (True, "fresh-frame")
    assert stale.release_calls == 1
    assert sleeps == [0.5]
    assert keepalive_calls == [True]


def test_recovering_camera_retries_while_device_reenumerates() -> None:
    stale = _ScriptedCamera([(False, None)])
    recovered = _ScriptedCamera([(True, "fresh-frame")])
    attempts = 0

    def reopen() -> _ScriptedCamera:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SystemExit("device path not ready")
        return recovered

    camera = RecoveringCamera(
        stale,
        reopen=reopen,
        name="turret",
        max_consecutive_read_failures=1,
        max_reopen_attempts=3,
        retry_delay_s=0.0,
        sleep=lambda _seconds: None,
    )

    assert camera.read() == (True, "fresh-frame")
    assert attempts == 2


def test_recovering_camera_treats_read_exception_as_failed_read() -> None:
    stale = _ScriptedCamera([OSError("device removed")])
    recovered = _ScriptedCamera([(True, "fresh-frame")])
    camera = RecoveringCamera(
        stale,
        reopen=lambda: recovered,
        name="turret",
        max_consecutive_read_failures=1,
        retry_delay_s=0.0,
        sleep=lambda _seconds: None,
    )

    assert camera.read() == (True, "fresh-frame")


def test_recovering_camera_escalates_after_reopen_budget_exhausted() -> None:
    stale = _ScriptedCamera([(False, None)])

    def reopen() -> _ScriptedCamera:
        raise SystemExit("camera absent")

    camera = RecoveringCamera(
        stale,
        reopen=reopen,
        name="turret",
        max_consecutive_read_failures=1,
        max_reopen_attempts=2,
        retry_delay_s=0.0,
        sleep=lambda _seconds: None,
    )

    with pytest.raises(CameraRecoveryError, match="turret camera unavailable"):
        camera.read()


def test_recovering_camera_delegates_release_to_active_capture() -> None:
    active = _ScriptedCamera([(True, "frame")])

    def reopen() -> _ScriptedCamera:
        return active

    camera = RecoveringCamera(active, reopen=reopen, name="turret")

    camera.release()

    assert active.release_calls == 1
