"""Pure self-healing watchdog logic: liveness, motion confirmation, restarts.

Everything here is deliberately free of OpenCV / hardware / GUI imports so it can
be unit-tested with plain clocks and numbers. The integration glue lives in
``cat_cannon.app.eye_screen`` (in-app checks) and ``cat_cannon.app.guardian``
(out-of-process restart escalation).
"""

from __future__ import annotations

import enum
import json
import os
import time
from dataclasses import dataclass

# Exit code the running app uses to tell the guardian "I detected my own fault
# (hang / no motion) and exited on purpose — please restart me". Chosen from the
# 64-78 sysexits range to avoid clashing with ordinary crashes.
SENTINEL_EXIT_CODE = 70


class FaultReason(enum.Enum):
    HANG = "hang"
    NO_MOTION = "no_motion"


class RestartDecision(enum.Enum):
    RESTART = "restart"
    REBOOT = "reboot"


class Liveness:
    """Tracks the last heartbeat so a hung loop can be detected.

    The owning loop calls :meth:`beat` every iteration; a separate monitor calls
    :meth:`is_stale` to decide whether the loop has hung.
    """

    def __init__(self, timeout_s: float, *, now: float | None = None) -> None:
        self._timeout_s = float(timeout_s)
        self._last_beat = time.monotonic() if now is None else float(now)

    def beat(self, now: float | None = None) -> None:
        self._last_beat = time.monotonic() if now is None else float(now)

    @property
    def last_beat(self) -> float:
        return self._last_beat

    def age(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else float(now)
        return current - self._last_beat

    def is_stale(self, now: float | None = None) -> bool:
        return self.age(now) > self._timeout_s


@dataclass(frozen=True)
class MotionConfig:
    flow_min_magnitude_px: float = 0.6
    direction_dot_min: float = 0.15
    max_consecutive_failures: int = 3
    pan_flow_sign: int = 1
    tilt_flow_sign: int = 1


class MotionWatchdog:
    """Decides whether a commanded turret move was confirmed by optical flow.

    Pure logic: given the commanded ``(pan_delta, tilt_delta)`` and the observed
    mean flow ``(dx, dy)`` it checks both magnitude (something moved) and
    direction (it moved roughly the commanded way). Consecutive failures are
    counted; the watchdog ``tripped`` once the threshold is reached.
    """

    def __init__(self, config: MotionConfig) -> None:
        self._config = config
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def tripped(self) -> bool:
        return self._consecutive_failures >= self._config.max_consecutive_failures

    def reset(self) -> None:
        self._consecutive_failures = 0

    def is_confirmed(
        self,
        *,
        pan_cmd: float,
        tilt_cmd: float,
        dx: float,
        dy: float,
    ) -> bool:
        """Return True if the observed flow confirms the commanded move."""
        exp_x = self._config.pan_flow_sign * pan_cmd
        exp_y = self._config.tilt_flow_sign * tilt_cmd
        exp_mag = (exp_x * exp_x + exp_y * exp_y) ** 0.5
        if exp_mag <= 1e-9:
            # No meaningful command -> nothing to confirm; treat as confirmed so
            # we never trip on a no-op move.
            return True
        flow_mag = (dx * dx + dy * dy) ** 0.5
        if flow_mag < self._config.flow_min_magnitude_px:
            return False
        dot = (exp_x * dx + exp_y * dy) / (exp_mag * flow_mag)
        return dot >= self._config.direction_dot_min

    def record(
        self,
        *,
        pan_cmd: float,
        tilt_cmd: float,
        dx: float,
        dy: float,
    ) -> bool:
        """Evaluate a move, update the failure counter, return confirmed?"""
        confirmed = self.is_confirmed(pan_cmd=pan_cmd, tilt_cmd=tilt_cmd, dx=dx, dy=dy)
        if confirmed:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        return confirmed


@dataclass(frozen=True)
class RestartPolicyConfig:
    reboot_threshold: int = 3
    window_s: float = 600.0


class RestartPolicy:
    """Decides RESTART vs REBOOT from recent fault history (JSON-persistable).

    A fault is recorded for every app restart the guardian performs. If
    ``reboot_threshold`` faults occur within ``window_s`` the policy escalates to
    a reboot and clears the history (so we don't reboot-loop forever).
    """

    def __init__(
        self,
        config: RestartPolicyConfig,
        *,
        state_path: str | None = None,
    ) -> None:
        self._config = config
        self._state_path = os.path.expanduser(state_path) if state_path else None
        self._history: list[dict] = []
        self._load()

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def _load(self) -> None:
        if not self._state_path or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, encoding="utf-8") as handle:
                data = json.load(handle)
            history = data.get("history", [])
            if isinstance(history, list):
                self._history = [
                    {"ts": float(item["ts"]), "reason": str(item.get("reason", ""))}
                    for item in history
                    if isinstance(item, dict) and "ts" in item
                ]
        except (OSError, ValueError, KeyError, TypeError):
            self._history = []

    def _save(self) -> None:
        if not self._state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as handle:
                json.dump({"history": self._history}, handle)
        except OSError:
            pass

    def _prune(self, now: float) -> None:
        cutoff = now - self._config.window_s
        self._history = [item for item in self._history if item["ts"] >= cutoff]

    def record_fault(self, reason: str, *, now: float | None = None) -> RestartDecision:
        current = time.time() if now is None else float(now)
        self._prune(current)
        self._history.append({"ts": current, "reason": reason})
        if len(self._history) >= self._config.reboot_threshold:
            self._history = []
            self._save()
            return RestartDecision.REBOOT
        self._save()
        return RestartDecision.RESTART

    def clear(self) -> None:
        self._history = []
        self._save()
