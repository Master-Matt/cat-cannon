from __future__ import annotations

from dataclasses import dataclass

from cat_cannon.adapters.interfaces import TurretController
from cat_cannon.config import SystemConfig
from cat_cannon.domain.models import CounterZone, Detection, SupervisorState
from cat_cannon.domain.safety import CounterConfirmation, DetectionPolicy, assess_scene
from cat_cannon.domain.state_machine import SupervisorInputs, SupervisorStateMachine
from cat_cannon.domain.targeting import FrameSize, TurretCorrection, compute_turret_correction


@dataclass(frozen=True)
class SupervisorStepResult:
    state: SupervisorState
    fire_commanded: bool
    human_present: bool
    counter_confirmed: bool
    target_visible: bool
    aim_locked: bool
    active_zone_id: str | None
    candidate_track_id: str | None
    correction: TurretCorrection | None


@dataclass
class SupervisorLoop:
    config: SystemConfig
    zones: list[CounterZone]
    controller: TurretController

    def __post_init__(self) -> None:
        self._confirmation = CounterConfirmation(
            required_frames=self.config.detection_policy.consecutive_counter_frames
        )
        self._machine = SupervisorStateMachine(cooldown_frames=self.config.cooldown_frames)
        # EMA-filtered tracking state (same algorithm as eye_screen)
        self._filtered_pan = 0.0
        self._filtered_tilt = 0.0

    def _find_turret_cat(self, turret_detections: list[Detection], policy: DetectionPolicy) -> Detection | None:
        cats = [
            d for d in turret_detections
            if d.label == policy.cat_class
            and d.confidence >= policy.cat_confidence_threshold
        ]
        if not cats:
            return None
        cats.sort(key=lambda d: d.confidence, reverse=True)
        return cats[0]

    def _find_turret_target(self, turret_detections: list[Detection], policy: DetectionPolicy) -> Detection | None:
        """Find best target on turret cam: prefer cats, fall back to people."""
        cat = self._find_turret_cat(turret_detections, policy)
        if cat is not None:
            return cat
        people = [
            d for d in turret_detections
            if d.label == policy.person_class
            and d.confidence >= policy.person_confidence_threshold
        ]
        if not people:
            return None
        people.sort(key=lambda d: d.confidence, reverse=True)
        return people[0]

    def _apply_ema_tracking(self, correction: TurretCorrection) -> None:
        """EMA-filtered position tracking — uses tracking_tuning config."""
        tuning = self.config.tracking_tuning
        # Clamp raw correction
        pan_delta = max(-15.0, min(15.0, correction.pan_delta))
        tilt_delta = max(-15.0, min(15.0, correction.tilt_delta))
        # EMA filter
        self._filtered_pan = tuning.ema_alpha * pan_delta + (1.0 - tuning.ema_alpha) * self._filtered_pan
        self._filtered_tilt = tuning.ema_alpha * tilt_delta + (1.0 - tuning.ema_alpha) * self._filtered_tilt
        # Apply gain, clamp pan to prevent overshoot
        cmd_pan = max(-tuning.pan_clamp_deg, min(tuning.pan_clamp_deg, self._filtered_pan * tuning.gain))
        cmd_tilt = self._filtered_tilt * tuning.gain
        if abs(cmd_pan) > tuning.deadband_deg or abs(cmd_tilt) > tuning.deadband_deg:
            self.controller.apply_tracking_delta(cmd_pan, cmd_tilt)

    def process_frame(
        self,
        detections,
        frame_width: int,
        frame_height: int,
        armed: bool,
        turret_detections: list | None = None,
        turret_frame_width: int | None = None,
        turret_frame_height: int | None = None,
        detection_policy_override: DetectionPolicy | None = None,
    ) -> SupervisorStepResult:
        policy = detection_policy_override or self.config.detection_policy
        # Fixed camera: zone intersection + counter confirmation + human presence
        assessment = assess_scene(detections=detections, zones=self.zones, policy=policy)
        counter_confirmed = self._confirmation.update(
            assessment.candidate_cat,
            assessment.cat_on_counter and not assessment.human_present,
        )

        aim_locked = False
        correction: TurretCorrection | None = None
        target_visible = assessment.candidate_cat is not None and counter_confirmed

        # States where we should keep tracking even if fixed camera loses confirmation
        tracking_states = {
            SupervisorState.TRACKING,
            SupervisorState.AIM_LOCK,
            SupervisorState.FIRE,
            SupervisorState.COOLDOWN,
        }
        should_track = target_visible or self._machine.state in tracking_states

        # Turret camera: track any visible target (cat or person) when armed
        turret_target = None
        if armed and turret_detections is not None and turret_frame_width and turret_frame_height:
            turret_target = self._find_turret_target(turret_detections, policy)
            if turret_target is not None:
                correction = compute_turret_correction(
                    bbox=turret_target.bbox,
                    frame=FrameSize(width=turret_frame_width, height=turret_frame_height),
                    calibration=self.config.tracking_calibration,
                )
                aim_locked = correction.aim_locked
                self._apply_ema_tracking(correction)
        elif should_track and assessment.candidate_cat is not None:
            # Fallback: no turret camera, use fixed camera for targeting
            correction = compute_turret_correction(
                bbox=assessment.candidate_cat.bbox,
                frame=FrameSize(width=frame_width, height=frame_height),
                calibration=self.config.tracking_calibration,
            )
            aim_locked = correction.aim_locked
            if not assessment.human_present:
                self._apply_ema_tracking(correction)
        else:
            # No target — reset EMA state
            self._filtered_pan = 0.0
            self._filtered_tilt = 0.0

        result = self._machine.advance(
            SupervisorInputs(
                armed=armed,
                human_present=assessment.human_present,
                counter_confirmed=counter_confirmed,
                target_visible=target_visible,
                aim_locked=aim_locked,
            )
        )

        # Only safe_stop when disarmed; only fire when no human present
        if not armed:
            self.controller.safe_stop()
        elif result.fire_commanded and not assessment.human_present:
            self.controller.fire()

        return SupervisorStepResult(
            state=result.state,
            fire_commanded=result.fire_commanded,
            human_present=assessment.human_present,
            counter_confirmed=counter_confirmed,
            target_visible=target_visible,
            aim_locked=aim_locked,
            active_zone_id=assessment.active_zone_id,
            candidate_track_id=assessment.candidate_cat.track_id if assessment.candidate_cat else None,
            correction=correction,
        )
