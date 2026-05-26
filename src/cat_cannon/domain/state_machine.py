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
    fire_permitted: bool | None = None
    faulted: bool = False


@dataclass(frozen=True)
class TransitionResult:
    state: SupervisorState
    fire_commanded: bool


class SupervisorStateMachine:
    def __init__(
        self,
        cooldown_frames: int,
        cooldown_seconds: float | None = None,
        burst_count: int = 1,
        burst_interval_seconds: float | None = None,
    ) -> None:
        self._cooldown_frames = cooldown_frames
        self._cooldown_seconds = None if cooldown_seconds is None else max(0.0, cooldown_seconds)
        self._burst_count = max(1, int(burst_count))
        self._burst_interval_seconds = (
            None
            if burst_interval_seconds is None
            else max(0.0, float(burst_interval_seconds))
        )
        self._burst_remaining = 0
        self._cooldown_remaining = 0
        self._cooldown_until: float | None = None
        self.state = SupervisorState.DISARMED

    def advance(self, inputs: SupervisorInputs, *, now: float | None = None) -> TransitionResult:
        now_s = 0.0 if now is None else float(now)
        if inputs.faulted:
            self.state = SupervisorState.FAULT
            self._reset_burst()
            return TransitionResult(self.state, fire_commanded=False)

        if not inputs.armed:
            self.state = SupervisorState.DISARMED
            self._cooldown_remaining = 0
            self._cooldown_until = None
            self._reset_burst()
            return TransitionResult(self.state, fire_commanded=False)

        if inputs.human_present:
            self.state = SupervisorState.HUMAN_LOCKOUT
            self._reset_burst()
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.HUMAN_LOCKOUT:
            self.state = SupervisorState.IDLE

        if self.state == SupervisorState.COOLDOWN:
            cooldown_done = self._advance_cooldown(now_s)
            if cooldown_done:
                self._cooldown_until = None
                if self._burst_remaining > 0:
                    if (
                        inputs.counter_confirmed
                        and inputs.target_visible
                        and self._fire_ready(inputs)
                    ):
                        self._start_fire(now_s)
                        return TransitionResult(self.state, fire_commanded=True)
                    self._reset_burst()
                if inputs.counter_confirmed and self._fire_ready(inputs):
                    self.state = SupervisorState.AIM_LOCK
                elif inputs.counter_confirmed and inputs.target_visible:
                    self.state = SupervisorState.TRACKING
                elif inputs.counter_confirmed:
                    self.state = SupervisorState.TURRET_ACQUIRE
                else:
                    self.state = SupervisorState.IDLE
            return TransitionResult(self.state, fire_commanded=False)

        if self.state == SupervisorState.DISARMED:
            self.state = SupervisorState.IDLE

        if not inputs.counter_confirmed:
            self.state = SupervisorState.IDLE
            self._reset_burst()
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
                self._reset_burst()
                return TransitionResult(self.state, fire_commanded=False)
            if not self._fire_ready(inputs):
                self.state = SupervisorState.TRACKING
                self._reset_burst()
                return TransitionResult(self.state, fire_commanded=False)
            self._start_fire(now_s)
            return TransitionResult(self.state, fire_commanded=True)

        if self.state == SupervisorState.FIRE:
            self.state = SupervisorState.COOLDOWN
            return TransitionResult(self.state, fire_commanded=False)

        return TransitionResult(self.state, fire_commanded=False)

    def _advance_cooldown(self, now_s: float) -> bool:
        if self._cooldown_seconds is None:
            self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
            return self._cooldown_remaining == 0
        return self._cooldown_until is None or now_s >= self._cooldown_until

    def _start_cooldown(self, now_s: float, *, burst: bool) -> None:
        self._cooldown_remaining = self._cooldown_frames
        interval = self._cooldown_seconds
        if burst and self._burst_interval_seconds is not None:
            interval = self._burst_interval_seconds
        if interval is not None:
            self._cooldown_until = now_s + interval

    def _reset_burst(self) -> None:
        self._burst_remaining = 0

    def _start_fire(self, now_s: float) -> None:
        self.state = SupervisorState.FIRE
        if self._burst_remaining <= 0:
            self._burst_remaining = self._burst_count - 1
        else:
            self._burst_remaining -= 1
        self._start_cooldown(now_s, burst=self._burst_remaining > 0)

    def _fire_ready(self, inputs: SupervisorInputs) -> bool:
        if inputs.fire_permitted is None:
            return inputs.aim_locked
        return inputs.aim_locked and inputs.fire_permitted
