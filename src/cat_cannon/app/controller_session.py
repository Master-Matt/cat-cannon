from __future__ import annotations

import threading
from dataclasses import dataclass, field

from cat_cannon.adapters.rp2040_serial import RP2040SerialController
from cat_cannon.config import ServoLimits


@dataclass
class ControllerSession:
    controller: RP2040SerialController
    servo_limits: ServoLimits | None = None
    heartbeat_interval_s: float = 0.5
    max_consecutive_failures: int = 3
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _fault_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _desired_enabled: bool | None = field(default=None, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _last_error: str | None = field(default=None, init=False)

    @property
    def faulted(self) -> bool:
        return self._fault_event.is_set()

    @property
    def fault_reason(self) -> str | None:
        with self._state_lock:
            return self._last_error

    def start(self) -> None:
        self.controller.handshake()
        if self.servo_limits is not None:
            self.controller.set_motion_directions(
                pan_delta_sign=self.servo_limits.pan_delta_sign,
                tilt_delta_sign=self.servo_limits.tilt_delta_sign,
            )
            self.controller.set_servo_limits(
                pan_min_deg=self.servo_limits.pan_min_deg,
                pan_max_deg=self.servo_limits.pan_max_deg,
                tilt_min_deg=self.servo_limits.tilt_min_deg,
                tilt_max_deg=self.servo_limits.tilt_max_deg,
            )
        self._stop_event.clear()
        self._fault_event.clear()
        with self._state_lock:
            self._consecutive_failures = 0
            self._last_error = None
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="rp2040-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def enable(self) -> None:
        with self._state_lock:
            self._desired_enabled = True
            self.controller.set_enabled(True)

    def disable(self) -> None:
        with self._state_lock:
            self._desired_enabled = False
            self.controller.set_enabled(False)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self.controller.safe_stop()
        except Exception:
            pass
        finally:
            self.controller.close()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            self._heartbeat_once()
            self._stop_event.wait(self.heartbeat_interval_s)

    def _heartbeat_once(self) -> bool:
        try:
            response = self.controller.heartbeat()
            payload = getattr(response, "payload", None)
            reported_enabled = (
                payload.get("enabled")
                if isinstance(payload, dict)
                else None
            )
            with self._state_lock:
                desired_enabled = self._desired_enabled
            if (
                isinstance(reported_enabled, bool)
                and desired_enabled is not None
                and reported_enabled != desired_enabled
            ):
                self.controller.set_enabled(desired_enabled)
                if desired_enabled:
                    print(
                        "[controller-session] "
                        "restored armed state after controller watchdog reset",
                        flush=True,
                    )
            with self._state_lock:
                self._consecutive_failures = 0
                if not self._fault_event.is_set():
                    self._last_error = None
            return True
        except Exception as exc:
            if self._stop_event.is_set():
                return False
            error = f"{type(exc).__name__}: {exc}"
            with self._state_lock:
                self._consecutive_failures += 1
                failure_count = self._consecutive_failures
                self._last_error = error
                if failure_count >= max(1, self.max_consecutive_failures):
                    self._fault_event.set()
            if failure_count in {1, max(1, self.max_consecutive_failures)}:
                print(
                    "[controller-session] heartbeat failed "
                    f"({failure_count}/{max(1, self.max_consecutive_failures)}): {error}",
                    flush=True,
                )
            return False
