from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from cat_cannon.adapters.interfaces import TurretController
from cat_cannon.config import HumanLockoutConfig, SystemConfig, scale_counter_zones
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
    turret_target_visible: bool = False
    turret_fire_aligned: bool = False
    turret_direction_aligned: bool = True
    fire_permitted: bool = False
    fixed_zone_fresh: bool = True


class HumanLockoutHysteresis:
    def __init__(self, config: HumanLockoutConfig) -> None:
        self.window_seconds = max(0.0, float(config.window_seconds))
        self.frame_threshold = max(1, int(config.frame_threshold))
        self._human_detection_times: deque[float] = deque()
        self._locked = False

    def update(self, *, human_detected: bool, now: float) -> bool:
        now_s = float(now)
        if human_detected:
            self._human_detection_times.append(now_s)
        self._prune(now=now_s)

        if len(self._human_detection_times) >= self.frame_threshold:
            self._locked = True
        elif self._locked:
            self._locked = False
        return self._locked

    def _prune(self, *, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._human_detection_times and self._human_detection_times[0] < cutoff:
            self._human_detection_times.popleft()


@dataclass
class SupervisorLoop:
    config: SystemConfig
    zones: list[CounterZone]
    controller: TurretController

    def __post_init__(self) -> None:
        self._confirmation = CounterConfirmation(
            required_frames=self.config.detection_policy.consecutive_counter_frames,
            max_missed_frames=(
                self.config.detection_policy.confirmation_miss_tolerance_frames
            ),
        )
        self._machine = SupervisorStateMachine(
            cooldown_frames=self.config.cooldown_frames,
            cooldown_seconds=self.config.fire_cooldown_seconds,
            burst_count=self.config.fire_burst_count,
            burst_interval_seconds=self.config.fire_burst_interval_seconds,
        )
        self._human_lockout = HumanLockoutHysteresis(self.config.human_lockout)
        # EMA-filtered tracking state (same algorithm as eye_screen)
        self._filtered_pan = 0.0
        self._filtered_tilt = 0.0
        self._logged_active_zone_id: str | None = None
        self._confirmed_zone_id: str | None = None
        self._confirmed_track_id: str | None = None
        self._last_fixed_lead_cat: Detection | None = None
        self._last_fixed_lead_frame_width = 0
        self._last_fixed_lead_frame_height = 0
        self._last_fixed_lead_at: float | None = None

    def _find_turret_cat(
        self,
        turret_detections: list[Detection],
        policy: DetectionPolicy,
        *,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> Detection | None:
        cats = [
            d for d in turret_detections
            if d.label == policy.cat_class
            and d.confidence >= policy.cat_confidence_threshold
        ]
        if not cats:
            return None
        if frame_width is not None and frame_height is not None:
            frame_center_x = frame_width / 2.0
            frame_center_y = frame_height / 2.0
            cats.sort(
                key=lambda detection: (
                    abs(detection.bbox.center.x - frame_center_x)
                    + abs(detection.bbox.center.y - frame_center_y),
                    -detection.confidence,
                )
            )
        else:
            cats.sort(key=lambda d: d.confidence, reverse=True)
        return cats[0]

    def _find_turret_person(
        self,
        turret_detections: list[Detection],
        policy: DetectionPolicy,
    ) -> Detection | None:
        people = [
            d for d in turret_detections
            if d.label == policy.person_class
            and d.confidence >= policy.person_confidence_threshold
        ]
        if not people:
            return None
        people.sort(key=lambda d: d.confidence, reverse=True)
        return people[0]

    def _find_turret_target(
        self,
        turret_detections: list[Detection],
        policy: DetectionPolicy,
        *,
        track_people: bool = False,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> Detection | None:
        cat = self._find_turret_cat(
            turret_detections,
            policy,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if cat is not None:
            return cat
        if track_people:
            return self._find_turret_person(turret_detections, policy)
        return None

    def _turret_cat_in_fire_gate(
        self,
        *,
        cat: Detection | None,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if cat is None:
            return False
        tolerance_px = self.config.tracking_tuning.fire_aim_tolerance_px
        target = cat.bbox.center
        aim_x = frame_width / 2.0 + self.config.tracking_calibration.aim_offset_x_px
        aim_y = frame_height / 2.0 + self.config.tracking_calibration.aim_offset_y_px
        return (
            abs(target.x - aim_x) <= tolerance_px
            and abs(target.y - aim_y) <= tolerance_px
        )

    def _turret_pan_matches_fixed_target_direction(
        self,
        *,
        cat: Detection | None,
        frame_width: int,
    ) -> bool:
        if cat is None or frame_width <= 0:
            return True
        limits = self.config.servo_limits
        if limits.pan_left_deg is None or limits.pan_right_deg is None:
            return True
        current_pan = self._last_reported_pan_deg()
        if current_pan is None:
            return True
        target_ratio = max(0.0, min(1.0, cat.bbox.center.x / frame_width))
        expected_pan = limits.pan_left_deg + (
            (limits.pan_right_deg - limits.pan_left_deg) * target_ratio
        )
        return (
            abs(current_pan - expected_pan)
            <= self.config.tracking_tuning.fire_pan_tolerance_deg
        )

    def _last_reported_pan_deg(self) -> float | None:
        payload = getattr(self.controller, "last_status_payload", None)
        if not isinstance(payload, dict):
            return None
        pan_deg = payload.get("pan_deg")
        if pan_deg is None:
            return None
        try:
            return float(pan_deg)
        except (TypeError, ValueError):
            return None

    def _apply_ema_tracking(self, correction: TurretCorrection) -> None:
        """EMA-filtered position tracking — uses tracking_tuning config."""
        tuning = self.config.tracking_tuning
        # Clamp raw correction
        pan_delta = max(-15.0, min(15.0, correction.pan_delta))
        tilt_delta = max(-15.0, min(15.0, correction.tilt_delta))
        # EMA filter
        self._filtered_pan = (
            tuning.ema_alpha * pan_delta
            + (1.0 - tuning.ema_alpha) * self._filtered_pan
        )
        self._filtered_tilt = (
            tuning.ema_alpha * tilt_delta
            + (1.0 - tuning.ema_alpha) * self._filtered_tilt
        )
        # Apply gain, clamp pan and tilt symmetrically to prevent overshoot
        cmd_pan = max(
            -tuning.pan_clamp_deg,
            min(tuning.pan_clamp_deg, self._filtered_pan * tuning.gain),
        )
        cmd_tilt = max(
            -tuning.tilt_clamp_deg,
            min(tuning.tilt_clamp_deg, self._filtered_tilt * tuning.gain),
        )
        # Stop commanding past a servo limit: if the controller reports it is
        # already at a tilt/pan limit, zero any further into-limit command and
        # reset the EMA accumulator (anti-windup) so it doesn't oscillate at the
        # edge. Direction in firmware space is delta_sign * cmd.
        status = getattr(self.controller, "last_status_payload", {}) or {}
        tilt_sign = getattr(self.controller, "tilt_delta_sign", 1)
        pan_sign = getattr(self.controller, "pan_delta_sign", 1)
        tilt_dir = tilt_sign * cmd_tilt
        if (tilt_dir > 0 and status.get("tilt_at_max")) or (
            tilt_dir < 0 and status.get("tilt_at_min")
        ):
            cmd_tilt = 0.0
            self._filtered_tilt = 0.0
        pan_dir = pan_sign * cmd_pan
        if (pan_dir > 0 and status.get("pan_at_max")) or (
            pan_dir < 0 and status.get("pan_at_min")
        ):
            cmd_pan = 0.0
            self._filtered_pan = 0.0
        if abs(cmd_pan) > tuning.deadband_deg or abs(cmd_tilt) > tuning.deadband_deg:
            self.controller.apply_tracking_delta(cmd_pan, cmd_tilt)

    def _fixed_camera_horizontal_lead(
        self,
        *,
        cat: Detection,
        frame_width: int,
        frame_height: int,
    ) -> TurretCorrection:
        correction = compute_turret_correction(
            bbox=cat.bbox,
            frame=FrameSize(width=frame_width, height=frame_height),
            calibration=self.config.tracking_calibration,
        )
        return TurretCorrection(
            pan_delta=correction.pan_delta,
            tilt_delta=0.0,
            aim_locked=False,
        )

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
        fixed_detections_fresh: bool = True,
        track_people: bool = False,
        now: float | None = None,
    ) -> SupervisorStepResult:
        now_s = time.monotonic() if now is None else float(now)
        policy = detection_policy_override or self.config.detection_policy
        # Fixed camera: zone intersection + counter confirmation + human presence
        fixed_frame_zones = scale_counter_zones(
            self.zones,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        assessment = assess_scene(detections=detections, zones=fixed_frame_zones, policy=policy)
        turret_human_present = False
        if turret_detections is not None:
            turret_human_present = self._find_turret_person(turret_detections, policy) is not None
        fixed_human_present = assessment.human_present if fixed_detections_fresh else False
        raw_human_present = fixed_human_present or turret_human_present
        human_present = self._human_lockout.update(
            human_detected=raw_human_present,
            now=now_s,
        )
        human_blocks_fire = human_present
        if fixed_detections_fresh:
            if (
                assessment.cat_on_counter
                and assessment.candidate_cat is not None
                and not human_present
            ):
                self._last_fixed_lead_cat = assessment.candidate_cat
                self._last_fixed_lead_frame_width = frame_width
                self._last_fixed_lead_frame_height = frame_height
                self._last_fixed_lead_at = now_s
            elif human_present or not assessment.cat_on_counter:
                self._last_fixed_lead_cat = None
        if fixed_detections_fresh:
            counter_confirmed = self._confirmation.update(
                assessment.candidate_cat,
                assessment.cat_on_counter and not human_blocks_fire,
            )
        else:
            counter_confirmed = self._confirmation.confirmed
        if counter_confirmed:
            if fixed_detections_fresh and assessment.active_zone_id is not None:
                self._confirmed_zone_id = assessment.active_zone_id
            if fixed_detections_fresh and assessment.candidate_cat is not None:
                self._confirmed_track_id = assessment.candidate_cat.track_id
        elif fixed_detections_fresh:
            self._confirmed_zone_id = None
            self._confirmed_track_id = None

        aim_locked = False
        correction: TurretCorrection | None = None
        turret_target_visible = False
        turret_fire_aligned = False
        turret_direction_aligned = True
        target_visible = counter_confirmed
        active_zone_id = self._confirmed_zone_id if counter_confirmed else None
        candidate_track_id = self._confirmed_track_id if counter_confirmed else None

        # States where we should keep tracking even if fixed camera loses confirmation
        tracking_states = {
            SupervisorState.TRACKING,
            SupervisorState.AIM_LOCK,
            SupervisorState.FIRE,
            SupervisorState.COOLDOWN,
        }
        fixed_target_recent = (
            self._last_fixed_lead_at is not None
            and now_s - self._last_fixed_lead_at <= 2.0
        )
        should_track = (
            target_visible or self._machine.state in tracking_states
        ) and fixed_target_recent
        fixed_lead_cat = (
            assessment.candidate_cat
            if fixed_detections_fresh and assessment.cat_on_counter
            else None
        )
        fixed_lead_frame_width = frame_width
        fixed_lead_frame_height = frame_height
        if (
            fixed_lead_cat is None
            and self._last_fixed_lead_cat is not None
            and self._last_fixed_lead_at is not None
        ):
            if now_s - self._last_fixed_lead_at <= 2.0:
                fixed_lead_cat = self._last_fixed_lead_cat
                fixed_lead_frame_width = self._last_fixed_lead_frame_width
                fixed_lead_frame_height = self._last_fixed_lead_frame_height
        should_lead_from_fixed = fixed_lead_cat is not None and not human_present

        # Turret camera: track the active target class only when armed.
        turret_target = None
        turret_camera_available = (
            turret_detections is not None
            and turret_frame_width is not None
            and turret_frame_height is not None
        )
        if armed and turret_camera_available:
            turret_target = self._find_turret_target(
                turret_detections,
                policy,
                track_people=track_people,
                frame_width=turret_frame_width,
                frame_height=turret_frame_height,
            )
            if turret_target is not None:
                correction = compute_turret_correction(
                    bbox=turret_target.bbox,
                    frame=FrameSize(width=turret_frame_width, height=turret_frame_height),
                    calibration=self.config.tracking_calibration,
                )
                aim_locked = correction.aim_locked
                if turret_target.label == policy.cat_class:
                    turret_target_visible = True
                    turret_fire_aligned = self._turret_cat_in_fire_gate(
                        cat=turret_target,
                        frame_width=turret_frame_width,
                        frame_height=turret_frame_height,
                    )
                    turret_direction_aligned = (
                        self._turret_pan_matches_fixed_target_direction(
                            cat=assessment.candidate_cat,
                            frame_width=frame_width,
                        )
                    )
                self._apply_ema_tracking(correction)
            elif should_lead_from_fixed:
                correction = self._fixed_camera_horizontal_lead(
                    cat=fixed_lead_cat,
                    frame_width=fixed_lead_frame_width,
                    frame_height=fixed_lead_frame_height,
                )
                self._apply_ema_tracking(correction)
        elif armed and should_track and assessment.candidate_cat is not None:
            # Fallback: no turret camera, use fixed camera for targeting
            correction = compute_turret_correction(
                bbox=assessment.candidate_cat.bbox,
                frame=FrameSize(width=frame_width, height=frame_height),
                calibration=self.config.tracking_calibration,
            )
            aim_locked = correction.aim_locked
            if not human_present:
                self._apply_ema_tracking(correction)
        else:
            # No target — reset EMA state
            self._filtered_pan = 0.0
            self._filtered_tilt = 0.0

        base_fire_permitted = counter_confirmed and not human_blocks_fire
        if (
            turret_camera_available
            and self.config.tracking_tuning.fire_requires_turret_target
        ):
            fire_permitted = (
                base_fire_permitted
                and turret_target_visible
                and aim_locked
            )
        else:
            fire_permitted = base_fire_permitted and aim_locked

        result = self._machine.advance(
            SupervisorInputs(
                armed=armed,
                human_present=human_present,
                counter_confirmed=counter_confirmed,
                target_visible=target_visible,
                aim_locked=aim_locked,
                fire_permitted=fire_permitted,
            ),
            now=now_s,
        )
        self._log_activation_and_fire(
            assessment_active_zone_id=active_zone_id,
            result=result,
            human_present=human_present,
            counter_confirmed=counter_confirmed,
            target_visible=target_visible,
            aim_locked=aim_locked,
            candidate_track_id=candidate_track_id,
            correction=correction,
            turret_target_visible=turret_target_visible,
            turret_fire_aligned=turret_fire_aligned,
            turret_direction_aligned=turret_direction_aligned,
            fire_permitted=fire_permitted,
            fixed_zone_fresh=(
                fixed_detections_fresh
                and assessment.active_zone_id is not None
                and counter_confirmed
            ),
        )

        # Only safe_stop when disarmed; only fire when no human present
        if not armed:
            self.controller.safe_stop()
        elif result.fire_commanded and not human_present:
            self.controller.fire()

        return SupervisorStepResult(
            state=result.state,
            fire_commanded=result.fire_commanded,
            human_present=human_present,
            counter_confirmed=counter_confirmed,
            target_visible=target_visible,
            aim_locked=aim_locked,
            active_zone_id=active_zone_id,
            candidate_track_id=candidate_track_id,
            correction=correction,
            turret_target_visible=turret_target_visible,
            turret_fire_aligned=turret_fire_aligned,
            turret_direction_aligned=turret_direction_aligned,
            fire_permitted=fire_permitted,
            fixed_zone_fresh=(
                fixed_detections_fresh
                and assessment.active_zone_id is not None
                and counter_confirmed
            ),
        )

    def _log_activation_and_fire(
        self,
        *,
        assessment_active_zone_id: str | None,
        result,
        human_present: bool,
        counter_confirmed: bool,
        target_visible: bool,
        aim_locked: bool,
        candidate_track_id: str | None,
        correction: TurretCorrection | None,
        turret_target_visible: bool,
        turret_fire_aligned: bool,
        turret_direction_aligned: bool,
        fire_permitted: bool,
        fixed_zone_fresh: bool,
    ) -> None:
        active_zone_id = assessment_active_zone_id if counter_confirmed else None
        if active_zone_id != self._logged_active_zone_id:
            if active_zone_id is None and self._logged_active_zone_id is not None:
                print(
                    "[supervisor] "
                    f"ts={_utc_log_timestamp()} zone_clear "
                    f"previous_zone={self._logged_active_zone_id} "
                    f"state={result.state.value} human={human_present}",
                    flush=True,
                )
            elif active_zone_id is not None:
                print(
                    "[supervisor] "
                    f"ts={_utc_log_timestamp()} zone_active "
                    f"zone={active_zone_id} track={candidate_track_id or '-'} "
                    f"state={result.state.value} target_visible={target_visible} "
                    f"aim_locked={aim_locked} human={human_present} "
                    f"turret_target={turret_target_visible} "
                    f"turret_aligned={turret_fire_aligned} "
                    f"turret_direction={turret_direction_aligned} "
                    f"fire_permitted={fire_permitted} "
                    f"fixed_fresh={fixed_zone_fresh}",
                    flush=True,
                )
            self._logged_active_zone_id = active_zone_id

        if result.fire_commanded:
            correction_text = "-"
            if correction is not None:
                correction_text = (
                    f"pan={correction.pan_delta:.2f},tilt={correction.tilt_delta:.2f}"
                )
            print(
                "[supervisor] "
                f"ts={_utc_log_timestamp()} fire_commanded "
                f"zone={assessment_active_zone_id or '-'} "
                f"track={candidate_track_id or '-'} "
                f"aim_locked={aim_locked} human={human_present} "
                f"turret_target={turret_target_visible} "
                f"turret_aligned={turret_fire_aligned} "
                f"turret_direction={turret_direction_aligned} "
                f"fire_permitted={fire_permitted} "
                f"fixed_fresh={fixed_zone_fresh} "
                f"correction={correction_text}",
                flush=True,
            )


def _utc_log_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
