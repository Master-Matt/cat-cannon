from __future__ import annotations

from dataclasses import dataclass

from cat_cannon.adapters.interfaces import TurretController
from cat_cannon.config import SystemConfig
from cat_cannon.domain.models import CounterZone, Detection, SupervisorState
from cat_cannon.domain.safety import CounterConfirmation, assess_scene
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

    def _find_turret_cat(self, turret_detections: list[Detection]) -> Detection | None:
        cats = [
            d for d in turret_detections
            if d.label == self.config.detection_policy.cat_class
            and d.confidence >= self.config.detection_policy.cat_confidence_threshold
        ]
        if not cats:
            return None
        cats.sort(key=lambda d: d.confidence, reverse=True)
        return cats[0]

    def process_frame(
        self,
        detections,
        frame_width: int,
        frame_height: int,
        armed: bool,
        turret_detections: list | None = None,
        turret_frame_width: int | None = None,
        turret_frame_height: int | None = None,
    ) -> SupervisorStepResult:
        # Fixed camera: zone intersection + counter confirmation + human presence
        assessment = assess_scene(detections=detections, zones=self.zones, policy=self.config.detection_policy)
        counter_confirmed = self._confirmation.update(
            assessment.candidate_cat,
            assessment.cat_on_counter and not assessment.human_present,
        )

        aim_locked = False
        correction: TurretCorrection | None = None
        target_visible = assessment.candidate_cat is not None and counter_confirmed

        # Turret camera: targeting correction when cat confirmed on counter
        if target_visible and turret_detections is not None and turret_frame_width and turret_frame_height:
            turret_cat = self._find_turret_cat(turret_detections)
            if turret_cat is not None:
                correction = compute_turret_correction(
                    bbox=turret_cat.bbox,
                    frame=FrameSize(width=turret_frame_width, height=turret_frame_height),
                    calibration=self.config.tracking_calibration,
                )
                aim_locked = correction.aim_locked
                if armed and not assessment.human_present:
                    self.controller.apply_tracking_delta(correction.pan_delta, correction.tilt_delta)
        elif target_visible and assessment.candidate_cat is not None:
            # Fallback: no turret camera, use fixed camera for targeting
            correction = compute_turret_correction(
                bbox=assessment.candidate_cat.bbox,
                frame=FrameSize(width=frame_width, height=frame_height),
                calibration=self.config.tracking_calibration,
            )
            aim_locked = correction.aim_locked
            if armed and not assessment.human_present:
                self.controller.apply_tracking_delta(correction.pan_delta, correction.tilt_delta)

        result = self._machine.advance(
            SupervisorInputs(
                armed=armed,
                human_present=assessment.human_present,
                counter_confirmed=counter_confirmed,
                target_visible=target_visible,
                aim_locked=aim_locked,
            )
        )

        if assessment.human_present or not armed:
            self.controller.safe_stop()
        elif result.fire_commanded:
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
