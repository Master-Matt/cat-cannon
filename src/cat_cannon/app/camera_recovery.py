"""Camera read recovery that survives transient USB re-enumeration."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from cat_cannon.app.heartbeat import CameraReadWatchdog


class CameraCapture(Protocol):
    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> Any: ...


class CameraRecoveryError(RuntimeError):
    """Raised when a camera cannot be reopened within its retry budget."""


class RecoveringCamera:
    """Wrap a capture and reopen it after repeated failed reads."""

    def __init__(
        self,
        camera: CameraCapture,
        *,
        reopen: Callable[[], CameraCapture],
        name: str,
        max_consecutive_read_failures: int = 5,
        max_reopen_attempts: int = 10,
        retry_delay_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        keepalive: Callable[[], None] | None = None,
    ) -> None:
        self._camera = camera
        self._reopen = reopen
        self._name = name
        self._read_watchdog = CameraReadWatchdog(max_consecutive_read_failures)
        self._max_reopen_attempts = max(1, int(max_reopen_attempts))
        self._retry_delay_s = max(0.0, float(retry_delay_s))
        self._sleep = sleep
        self._keepalive = keepalive

    def read(self) -> tuple[bool, Any]:
        try:
            succeeded, frame = self._camera.read()
        except Exception:
            succeeded, frame = False, None
        if not self._read_watchdog.record(succeeded=succeeded):
            return succeeded, frame
        return self._recover()

    def _recover(self) -> tuple[bool, Any]:
        self._release_quietly(self._camera)
        last_error: BaseException | None = None

        for attempt in range(1, self._max_reopen_attempts + 1):
            if self._keepalive is not None:
                self._keepalive()
            self._sleep(self._retry_delay_s)
            candidate: CameraCapture | None = None
            try:
                candidate = self._reopen()
                succeeded, frame = candidate.read()
            except (Exception, SystemExit) as exc:
                if candidate is not None:
                    self._release_quietly(candidate)
                last_error = exc
                continue

            if succeeded:
                self._camera = candidate
                self._read_watchdog.record(succeeded=True)
                print(
                    f"[camera] {self._name} recovered after {attempt} reopen attempt(s)",
                    flush=True,
                )
                return True, frame

            self._release_quietly(candidate)

        detail = ""
        if last_error is not None:
            detail = f": {type(last_error).__name__}: {last_error}"
        raise CameraRecoveryError(
            f"{self._name} camera unavailable after "
            f"{self._max_reopen_attempts} reopen attempts{detail}"
        )

    def release(self) -> Any:
        return self._camera.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._camera, name)

    @staticmethod
    def _release_quietly(camera: CameraCapture) -> None:
        try:
            camera.release()
        except Exception:
            pass
