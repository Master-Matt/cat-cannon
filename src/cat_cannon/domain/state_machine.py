from __future__ import annotations

from dataclasses import dataclass

from cat_cannon.domain.models import SupervisorState


@dataclass(frozen=True)
class SupervisorInputs:
    armed: bool
    human_present: bool
    counter_confirmed: bool
    target_visible: bool
    aim_locked: bool
    faulted: bool = False


@dataclass(frozen=True)
class TransitionResult:
    state: SupervisorState
    fire_commanded: bool


class SupervisorStateMachine:
    def __init__(self, cooldown_frames: int, cooldown_seconds: float | None = None) -> None:
        self._cooldown_frames = cooldown_frames
        self._cooldown_seconds = None if cooldown_seconds is None else max(0.0, cooldown_seconds)
        self._cooldown_remaining = 0
        self._cooldown_until: float | None = None
        self.state = SupervisorState.DISARMED

    def advance(self, inputs: SupervisorInputs, *, now: float | None = None) -> TransitionResult:
        now_s = 0.0 if now is None else float(now)
        if inputs.faulted:
            self.state = SupervisorState.FAULT
            return TransitionResult(self.state, fire_commanded=False)

        if not inputs.armed:
            self.state = SupervisorState.DISARMED
            self._cooldown_remaining = 0
            self._cooldown_until = None
            return TransitionResult(self.state, fire_commanded=False)

        if inputs.human_present:
            self.state = SupervisorState.HUMAN_LOCKOUT
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.COOLDOWN:
            if self._cooldown_seconds is None:
                self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
                cooldown_done = self._cooldown_remaining == 0
            else:
                cooldown_done = self._cooldown_until is None or now_s >= self._cooldown_until
            if cooldown_done:
                self._cooldown_until = None
                # Re-fire immediately if target is still centered
                if inputs.counter_confirmed and inputs.aim_locked:
                    self.state = SupervisorState.AIM_LOCK
                else:
                    self.state = SupervisorState.IDLE
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.DISARMED:
            self.state = SupervisorState.IDLE

        if not inputs.counter_confirmed:
            self.state = SupervisorState.IDLE
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.IDLE:
            self.state = SupervisorState.COUNTER_CONFIRMED

        if self.state == SupervisorState.COUNTER_CONFIRMED:
            self.state = SupervisorState.TURRET_ACQUIRE

        if self.state == SupervisorState.TURRET_ACQUIRE:
            self.state = (
                SupervisorState.TRACKING
                if inputs.target_visible
                else SupervisorState.TURRET_ACQUIRE
            )
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.TRACKING:
            if not inputs.target_visible:
                self.state = SupervisorState.TURRET_ACQUIRE
            elif inputs.aim_locked:
                self.state = SupervisorState.AIM_LOCK
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.AIM_LOCK:
            if not inputs.target_visible:
                self.state = SupervisorState.TURRET_ACQUIRE
                return TransitionResult(self.state, fire_commanded=False)
            if not inputs.aim_locked:
                self.state = SupervisorState.TRACKING
                return TransitionResult(self.state, fire_commanded=False)
            self.state = SupervisorState.FIRE
            self._cooldown_remaining = self._cooldown_frames
            if self._cooldown_seconds is not None:
                self._cooldown_until = now_s + self._cooldown_seconds
            return TransitionResult(self.state, fire_commanded=True)

        if self.state == SupervisorState.FIRE:
            self.state = SupervisorState.COOLDOWN
            return TransitionResult(self.state, fire_commanded=False)

        return TransitionResult(self.state, fire_commanded=False)
