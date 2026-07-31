from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from cat_cannon.adapters.interfaces import TurretController
from cat_cannon.app.idle_motion import configured_center_angles
from cat_cannon.config import HumanLockoutConfig, SystemConfig, scale_counter_zones
from cat_cannon.domain.models import CounterZone, Detection, SupervisorState
from cat_cannon.domain.safety import CounterConfirmation, DetectionPolicy, assess_scene
from cat_cannon.domain.state_machine import SupervisorInputs, SupervisorStateMachine
from cat_cannon.domain.targeting import FrameSize, TurretCorrection, compute_turret_correction

ACQUISITION_MAX_CENTER_JUMP_RATIO = 0.15


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
    corner_recovery_active: bool = False


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


class TargetAcquisitionHysteresis:
    """Require repeated, spatially coherent observations before tracking."""

    def __init__(self, *, window_seconds: float, frame_threshold: int) -> None:
        self.window_seconds = max(0.0, float(window_seconds))
        self.frame_threshold = max(1, int(frame_threshold))
        self._observations: dict[
            str,
            deque[tuple[float, tuple[float, float]]],
        ] = {}
        self._acquired_keys: set[str] = set()

    def update(
        self,
        *,
        target_key: str | None,
        target_center: tuple[float, float] | None,
        now: float,
    ) -> bool:
        now_s = float(now)
        self._prune(now=now_s)
        if target_key is None or target_center is None:
            return False

        observations = self._observations.setdefault(target_key, deque())
        if observations:
            previous_center = observations[-1][1]
            center_jump = (
                (target_center[0] - previous_center[0]) ** 2
                + (target_center[1] - previous_center[1]) ** 2
            ) ** 0.5
            if center_jump > ACQUISITION_MAX_CENTER_JUMP_RATIO:
                observations.clear()
                self._acquired_keys.discard(target_key)

        observations.append((now_s, target_center))
        if len(observations) >= self.frame_threshold:
            self._acquired_keys.add(target_key)
        return target_key in self._acquired_keys

    def reset(self) -> None:
        self._observations.clear()
        self._acquired_keys.clear()

    def is_acquired(self, *, target_key: str | None, now: float) -> bool:
        self._prune(now=float(now))
        return target_key is not None and target_key in self._acquired_keys

    def _prune(self, *, now: float) -> None:
        cutoff = now - self.window_seconds
        expired_keys: list[str] = []
        for target_key, observations in self._observations.items():
            while observations and observations[0][0] < cutoff:
                observations.popleft()
            if not observations:
                expired_keys.append(target_key)

        for target_key in expired_keys:
            del self._observations[target_key]
            self._acquired_keys.discard(target_key)


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
        acquisition_window = self.config.tracking_tuning.acquisition_window_seconds
        acquisition_threshold = self.config.tracking_tuning.acquisition_frame_threshold
        self._fixed_target_acquisition = TargetAcquisitionHysteresis(
            window_seconds=acquisition_window,
            frame_threshold=acquisition_threshold,
        )
        self._turret_target_acquisition = TargetAcquisitionHysteresis(
            window_seconds=acquisition_window,
            frame_threshold=acquisition_threshold,
        )
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
        self._stuck_limit_started_at: float | None = None
        self._stuck_limit_directions: tuple[int, int] = (0, 0)
        self._corner_recovery_active = False
        self._corner_recovery_directions: tuple[int, int] = (0, 0)
        self._corner_recovery_failure_logged = False

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

    @staticmethod
    def _normalized_detection_center(
        detection: Detection | None,
        *,
        frame_width: int | None,
        frame_height: int | None,
    ) -> tuple[float, float] | None:
        if (
            detection is None
            or frame_width is None
            or frame_height is None
            or frame_width <= 0
            or frame_height <= 0
        ):
            return None
        center = detection.bbox.center
        return (
            max(0.0, min(1.0, center.x / frame_width)),
            max(0.0, min(1.0, center.y / frame_height)),
        )

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
        expected_pan = self._fixed_camera_expected_pan(
            cat=cat,
            frame_width=frame_width,
        )
        if expected_pan is None:
            return True
        current_pan = self._last_reported_pan_deg()
        if current_pan is None:
            return True
        return (
            abs(current_pan - expected_pan)
            <= self.config.tracking_tuning.fire_pan_tolerance_deg
        )

    def _fixed_camera_expected_pan(
        self,
        *,
        cat: Detection | None,
        frame_width: int,
    ) -> float | None:
        if cat is None or frame_width <= 0:
            return None
        limits = self.config.servo_limits
        if limits.pan_left_deg is None or limits.pan_right_deg is None:
            return None
        target_ratio = max(0.0, min(1.0, cat.bbox.center.x / frame_width))
        return limits.pan_left_deg + (
            (limits.pan_right_deg - limits.pan_left_deg) * target_ratio
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

    @staticmethod
    def _direction(value: float) -> int:
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 0

    def _outward_limit_directions(
        self,
        correction: TurretCorrection,
    ) -> tuple[int, int]:
        """Return camera-space directions still pushing into physical limits."""
        status = getattr(self.controller, "last_status_payload", {}) or {}
        pan_sign = getattr(self.controller, "pan_delta_sign", 1)
        tilt_sign = getattr(self.controller, "tilt_delta_sign", 1)
        pan_firmware_dir = self._direction(pan_sign * correction.pan_delta)
        tilt_firmware_dir = self._direction(tilt_sign * correction.tilt_delta)

        pan_outward = (pan_firmware_dir > 0 and status.get("pan_at_max")) or (
            pan_firmware_dir < 0 and status.get("pan_at_min")
        )
        tilt_outward = (tilt_firmware_dir > 0 and status.get("tilt_at_max")) or (
            tilt_firmware_dir < 0 and status.get("tilt_at_min")
        )
        return (
            self._direction(correction.pan_delta) if pan_outward else 0,
            self._direction(correction.tilt_delta) if tilt_outward else 0,
        )

    def _reset_corner_recovery(self) -> None:
        self._stuck_limit_started_at = None
        self._stuck_limit_directions = (0, 0)
        self._corner_recovery_active = False
        self._corner_recovery_directions = (0, 0)
        self._corner_recovery_failure_logged = False

    def _recovery_still_blocks(self, correction: TurretCorrection) -> bool:
        if not self._corner_recovery_active:
            return False
        if correction.aim_locked:
            self._reset_corner_recovery()
            return False

        current_directions = (
            self._direction(correction.pan_delta),
            self._direction(correction.tilt_delta),
        )
        for blocked, current in zip(
            self._corner_recovery_directions,
            current_directions,
            strict=True,
        ):
            if blocked and current == -blocked:
                self._reset_corner_recovery()
                return False
        return True

    def _recover_stuck_corner(
        self,
        correction: TurretCorrection,
        *,
        now: float,
    ) -> bool:
        """Center and quarantine a target that cannot converge at a hard limit."""
        if self._recovery_still_blocks(correction):
            self._filtered_pan = 0.0
            self._filtered_tilt = 0.0
            return True

        directions = self._outward_limit_directions(correction)
        if directions == (0, 0) or correction.aim_locked:
            self._stuck_limit_started_at = None
            self._stuck_limit_directions = (0, 0)
            return False
        if directions != self._stuck_limit_directions:
            self._stuck_limit_started_at = now
            self._stuck_limit_directions = directions

        if self._stuck_limit_started_at is None:
            self._stuck_limit_started_at = now
        stuck_seconds = max(0.0, self.config.tracking_tuning.stuck_limit_seconds)
        if now - self._stuck_limit_started_at < stuck_seconds:
            return False

        pan_center, tilt_center = configured_center_angles(
            self.config.tracking_calibration,
            self.config.servo_limits,
        )
        try:
            self.controller.set_velocity(0.0, 0.0)
            self.controller.set_angles(pan_center, tilt_center)
        except Exception as exc:
            if not self._corner_recovery_failure_logged:
                print(f"[corner-recovery] center command failed: {exc!r}", flush=True)
                self._corner_recovery_failure_logged = True
            self._stuck_limit_started_at = now
            return False

        self._filtered_pan = 0.0
        self._filtered_tilt = 0.0
        self._corner_recovery_active = True
        self._corner_recovery_directions = directions
        self._stuck_limit_started_at = None
        self._stuck_limit_directions = (0, 0)
        self._corner_recovery_failure_logged = False
        print(
            "[corner-recovery] centered untrackable target "
            f"pan={pan_center:.2f} tilt={tilt_center:.2f}",
            flush=True,
        )
        return True

    def _fixed_camera_horizontal_lead(
        self,
        *,
        cat: Detection,
        frame_width: int,
        frame_height: int,
    ) -> TurretCorrection:
        expected_pan = self._fixed_camera_expected_pan(
            cat=cat,
            frame_width=frame_width,
        )
        current_pan = self._last_reported_pan_deg()
        if expected_pan is not None and current_pan is not None:
            pan_sign = getattr(
                self.controller,
                "pan_delta_sign",
                self.config.servo_limits.pan_delta_sign,
            )
            camera_pan_delta = (expected_pan - current_pan) * (
                1 if pan_sign >= 0 else -1
            )
            if (
                abs(camera_pan_delta)
                <= self.config.tracking_tuning.deadband_deg
            ):
                camera_pan_delta = 0.0
            return TurretCorrection(
                pan_delta=camera_pan_delta,
                tilt_delta=0.0,
                aim_locked=False,
            )

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
        if not armed:
            self._reset_corner_recovery()
            self._fixed_target_acquisition.reset()
            self._turret_target_acquisition.reset()
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
        fixed_acquisition_candidate = (
            assessment.candidate_cat
            if (
                armed
                and fixed_detections_fresh
                and assessment.cat_on_counter
            )
            else None
        )
        self._fixed_target_acquisition.update(
            target_key=(
                fixed_acquisition_candidate.label
                if fixed_acquisition_candidate is not None
                else None
            ),
            target_center=self._normalized_detection_center(
                fixed_acquisition_candidate,
                frame_width=frame_width,
                frame_height=frame_height,
            ),
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
            and now_s - self._last_fixed_lead_at
            <= self.config.tracking_tuning.fixed_lead_hold_seconds
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
            if (
                now_s - self._last_fixed_lead_at
                <= self.config.tracking_tuning.fixed_lead_hold_seconds
            ):
                fixed_lead_cat = self._last_fixed_lead_cat
                fixed_lead_frame_width = self._last_fixed_lead_frame_width
                fixed_lead_frame_height = self._last_fixed_lead_frame_height
        fixed_target_acquired = self._fixed_target_acquisition.is_acquired(
            target_key=(
                fixed_lead_cat.label
                if fixed_lead_cat is not None
                else None
            ),
            now=now_s,
        )
        # Turret camera: track the active target class only when armed.
        turret_target_candidate = None
        turret_camera_available = (
            turret_detections is not None
            and turret_frame_width is not None
            and turret_frame_height is not None
        )
        if (
            armed
            and turret_detections is not None
            and turret_frame_width is not None
            and turret_frame_height is not None
        ):
            turret_target_candidate = self._find_turret_target(
                turret_detections,
                policy,
                track_people=track_people,
                frame_width=turret_frame_width,
                frame_height=turret_frame_height,
            )
        turret_target_acquired = self._turret_target_acquisition.update(
            target_key=(
                turret_target_candidate.label
                if turret_target_candidate is not None
                else None
            ),
            target_center=self._normalized_detection_center(
                turret_target_candidate,
                frame_width=turret_frame_width,
                frame_height=turret_frame_height,
            ),
            now=now_s,
        )
        turret_target = (
            turret_target_candidate
            if turret_target_acquired
            else None
        )
        if (
            armed
            and turret_detections is not None
            and turret_frame_width is not None
            and turret_frame_height is not None
        ):
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
                if not self._recover_stuck_corner(correction, now=now_s):
                    self._apply_ema_tracking(correction)
            elif (
                fixed_lead_cat is not None
                and fixed_target_acquired
                and not human_present
            ):
                self._reset_corner_recovery()
                correction = self._fixed_camera_horizontal_lead(
                    cat=fixed_lead_cat,
                    frame_width=fixed_lead_frame_width,
                    frame_height=fixed_lead_frame_height,
                )
                if (
                    self._direction(correction.pan_delta)
                    != self._direction(self._filtered_pan)
                ):
                    self._filtered_pan = 0.0
                self._apply_ema_tracking(correction)
            else:
                self._reset_corner_recovery()
                self._filtered_pan = 0.0
                self._filtered_tilt = 0.0
        elif (
            armed
            and should_track
            and fixed_target_acquired
            and assessment.candidate_cat is not None
        ):
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
            corner_recovery_active=self._corner_recovery_active,
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
        correction_text = "-"
        if correction is not None:
            correction_text = (
                f"pan={correction.pan_delta:.2f},"
                f"tilt={correction.tilt_delta:.2f}"
            )
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
                    f"fixed_fresh={fixed_zone_fresh} "
                    f"correction={correction_text}",
                    flush=True,
                )
            self._logged_active_zone_id = active_zone_id

        if result.fire_commanded:
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
