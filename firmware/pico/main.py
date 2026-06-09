import json
import sys
import time
import uselect
from machine import PWM, Pin

import pico_config as cfg


class Servo:
    def __init__(self, pin_id, min_deg, max_deg, home_deg):
        self._pwm = PWM(Pin(pin_id))
        self._pwm.freq(cfg.SERVO_FREQUENCY_HZ)
        self._physical_min_deg = min_deg
        self._physical_max_deg = max_deg
        self._min_deg = min_deg
        self._max_deg = max_deg
        self.angle_deg = home_deg
        self._attached = False
        self.write(home_deg, force=True)

    def write(self, angle_deg, force=False):
        clamped = min(self._max_deg, max(self._min_deg, angle_deg))
        if not force and abs(clamped - self.angle_deg) <= cfg.SERVO_EPS_DEG:
            return False
        self.angle_deg = clamped
        duty = self._angle_to_duty_u16(clamped)
        self._pwm.duty_u16(duty)
        self._attached = True
        return True

    def set_limits(self, min_deg, max_deg):
        self._min_deg = min(min_deg, max_deg)
        self._max_deg = max(min_deg, max_deg)
        return self.write(self.angle_deg)

    def delta(self, amount_deg):
        return self.write(self.angle_deg + amount_deg)

    def blocked_toward(self, amount_deg):
        if amount_deg < 0:
            return self.angle_deg <= self._min_deg + cfg.SERVO_EPS_DEG
        if amount_deg > 0:
            return self.angle_deg >= self._max_deg - cfg.SERVO_EPS_DEG
        return False

    def detach(self):
        """Stop PWM signal to eliminate servo buzz."""
        self._pwm.duty_u16(0)
        self._attached = False

    @property
    def attached(self):
        return self._attached

    @property
    def min_deg(self):
        return self._min_deg

    @property
    def max_deg(self):
        return self._max_deg

    def _angle_to_duty_u16(self, angle_deg):
        span_deg = self._physical_max_deg - self._physical_min_deg or 1.0
        fraction = (angle_deg - self._physical_min_deg) / span_deg
        pulse_us = cfg.SERVO_MIN_US + fraction * (cfg.SERVO_MAX_US - cfg.SERVO_MIN_US)
        period_us = 1000000.0 / cfg.SERVO_FREQUENCY_HZ
        duty_fraction = pulse_us / period_us
        return int(max(0, min(65535, duty_fraction * 65535)))


class Controller:
    def __init__(self):
        self.pan = Servo(cfg.PAN_SERVO_PIN, cfg.PAN_MIN_DEG, cfg.PAN_MAX_DEG, cfg.PAN_HOME_DEG)
        self.tilt = Servo(
            cfg.TILT_SERVO_PIN, cfg.TILT_MIN_DEG, cfg.TILT_MAX_DEG, cfg.TILT_HOME_DEG
        )
        inactive_level = self._solenoid_level(False)
        try:
            self.solenoid = Pin(cfg.SOLENOID_PIN, Pin.OUT, value=inactive_level)
        except TypeError:
            self.solenoid = Pin(cfg.SOLENOID_PIN, Pin.OUT)
            self.solenoid.value(inactive_level)
        self.led = Pin(cfg.STATUS_LED_PIN, Pin.OUT)
        self.enabled = False
        self.last_contact_ms = time.ticks_ms()
        self.last_move_ms = time.ticks_ms()
        self.fire_until_ms = None
        # Velocity mode: degrees per second for continuous smooth motion
        self.pan_vel = 0.0
        self.tilt_vel = 0.0
        self._last_tick_ms = time.ticks_ms()

    def _solenoid_level(self, active):
        if cfg.SOLENOID_ACTIVE_LOW:
            return 0 if active else 1
        return 1 if active else 0

    def _set_solenoid(self, active):
        self.solenoid.value(self._solenoid_level(active))

    def _solenoid_active(self):
        raw_level = bool(self.solenoid.value())
        return not raw_level if cfg.SOLENOID_ACTIVE_LOW else raw_level

    def _bounded(self, value, min_value, max_value):
        return min(max_value, max(min_value, float(value)))

    def _set_servo_limits(self, payload):
        pan_min = self._bounded(
            payload.get("pan_min_deg", self.pan.min_deg),
            cfg.PAN_MIN_DEG,
            cfg.PAN_MAX_DEG,
        )
        pan_max = self._bounded(
            payload.get("pan_max_deg", self.pan.max_deg),
            cfg.PAN_MIN_DEG,
            cfg.PAN_MAX_DEG,
        )
        tilt_min = self._bounded(
            payload.get("tilt_min_deg", self.tilt.min_deg),
            cfg.TILT_MIN_DEG,
            cfg.TILT_MAX_DEG,
        )
        tilt_max = self._bounded(
            payload.get("tilt_max_deg", self.tilt.max_deg),
            cfg.TILT_MIN_DEG,
            cfg.TILT_MAX_DEG,
        )
        self.pan.set_limits(pan_min, pan_max)
        self.tilt.set_limits(tilt_min, tilt_max)
        self.pan_vel = 0.0
        self.tilt_vel = 0.0

    def tick(self):
        now = time.ticks_ms()
        dt_ms = time.ticks_diff(now, self._last_tick_ms)
        self._last_tick_ms = now

        if self.fire_until_ms is not None and time.ticks_diff(self.fire_until_ms, now) <= 0:
            self._set_solenoid(False)
            self.fire_until_ms = None

        if time.ticks_diff(now, self.last_contact_ms) > cfg.WATCHDOG_TIMEOUT_MS:
            self.enabled = False
            self._set_solenoid(False)
            self.fire_until_ms = None
            self.led.value(0)
            self.pan_vel = 0.0
            self.tilt_vel = 0.0

        # Apply velocity: continuous servo interpolation at tick rate (~50Hz)
        if abs(self.pan_vel) > 0.1 or abs(self.tilt_vel) > 0.1:
            dt_s = dt_ms / 1000.0
            pan_delta = self.pan_vel * dt_s
            tilt_delta = self.tilt_vel * dt_s
            if self.pan.blocked_toward(pan_delta):
                self.pan_vel = 0.0
                pan_moved = False
            else:
                pan_moved = self.pan.delta(pan_delta)
            if self.tilt.blocked_toward(tilt_delta):
                self.tilt_vel = 0.0
                tilt_moved = False
            else:
                tilt_moved = self.tilt.delta(tilt_delta)
            if pan_moved or tilt_moved:
                self.last_move_ms = now

        # Auto-relax servos after idle period to stop buzzing. The tilt axis is
        # gravity-loaded, so keep it energized (configurable) to preserve holding
        # torque and avoid sticky re-engagement; pan may relax freely.
        relax_tilt = getattr(cfg, "SERVO_IDLE_RELAX_TILT", True)
        idle_ms = time.ticks_diff(now, self.last_move_ms)
        if idle_ms > cfg.SERVO_IDLE_RELAX_MS:
            if self.pan.attached:
                self.pan.detach()
            if relax_tilt and self.tilt.attached:
                self.tilt.detach()

    def handle(self, message):
        self.last_contact_ms = time.ticks_ms()
        command = message.get("command")
        payload = message.get("payload", {})
        seq = message.get("seq")

        try:
            if command == "ping":
                return self._ok(seq, "pong", self._status_payload())
            if command == "heartbeat":
                return self._ok(seq, "heartbeat", self._status_payload())
            if command == "status":
                return self._ok(seq, "status", self._status_payload())
            if command == "set_enabled":
                self.enabled = bool(payload.get("enabled", False))
                self.led.value(1 if self.enabled else 0)
                if not self.enabled:
                    self._set_solenoid(False)
                    self.fire_until_ms = None
                return self._ok(seq, "enabled", self._status_payload())
            if command == "set_angles":
                pan_moved = self.pan.write(float(payload["pan_deg"]))
                tilt_moved = self.tilt.write(float(payload["tilt_deg"]))
                if pan_moved or tilt_moved:
                    self.last_move_ms = time.ticks_ms()
                return self._ok(seq, "angles_set", self._status_payload())
            if command == "set_servo_limits":
                self._set_servo_limits(payload)
                return self._ok(seq, "servo_limits_set", self._status_payload())
            if command == "apply_delta":
                pan_moved = self.pan.delta(float(payload.get("pan_delta_deg", 0.0)))
                tilt_moved = self.tilt.delta(float(payload.get("tilt_delta_deg", 0.0)))
                if pan_moved or tilt_moved:
                    self.last_move_ms = time.ticks_ms()
                return self._ok(seq, "delta_applied", self._status_payload())
            if command == "set_velocity":
                self.pan_vel = float(payload.get("pan_deg_s", 0.0))
                self.tilt_vel = float(payload.get("tilt_deg_s", 0.0))
                self.last_move_ms = time.ticks_ms()
                return self._ok(seq, "velocity_set", self._status_payload())
            if command == "relax":
                self.pan_vel = 0.0
                self.tilt_vel = 0.0
                self.pan.detach()
                self.tilt.detach()
                return self._ok(seq, "relaxed", self._status_payload())
            if command == "safe_stop":
                self.pan_vel = 0.0
                self.tilt_vel = 0.0
                self._set_solenoid(False)
                self.fire_until_ms = None
                return self._ok(seq, "safe_stop", self._status_payload())
            if command == "set_fire_output":
                active = bool(payload.get("active", False))
                if active and not self.enabled:
                    return self._error(seq, "disabled", self._status_payload())
                self.fire_until_ms = None
                self._set_solenoid(active)
                return self._ok(seq, "fire_output_set", self._status_payload())
            if command == "fire":
                if not self.enabled:
                    return self._error(seq, "disabled", self._status_payload())
                duration_ms = min(
                    cfg.MAX_FIRE_DURATION_MS,
                    max(1, int(payload.get("duration_ms", cfg.DEFAULT_FIRE_DURATION_MS))),
                )
                self._set_solenoid(True)
                self.fire_until_ms = time.ticks_add(time.ticks_ms(), duration_ms)
                return self._ok(seq, "firing", self._status_payload())
            return self._error(seq, "unknown_command", {"command": command})
        except Exception as exc:
            return self._error(seq, "exception", {"error": str(exc), "command": command})

    def _status_payload(self):
        return {
            "enabled": self.enabled,
            "pan_deg": self.pan.angle_deg,
            "tilt_deg": self.tilt.angle_deg,
            "pan_min_deg": self.pan.min_deg,
            "pan_max_deg": self.pan.max_deg,
            "tilt_min_deg": self.tilt.min_deg,
            "tilt_max_deg": self.tilt.max_deg,
            "pan_at_min": self.pan.angle_deg <= self.pan.min_deg + cfg.SERVO_EPS_DEG,
            "pan_at_max": self.pan.angle_deg >= self.pan.max_deg - cfg.SERVO_EPS_DEG,
            "tilt_at_min": self.tilt.angle_deg <= self.tilt.min_deg + cfg.SERVO_EPS_DEG,
            "tilt_at_max": self.tilt.angle_deg >= self.tilt.max_deg - cfg.SERVO_EPS_DEG,
            "pan_attached": self.pan.attached,
            "tilt_attached": self.tilt.attached,
            "solenoid_active": self._solenoid_active(),
        }

    @staticmethod
    def _ok(seq, status, payload):
        return {"ok": True, "seq": seq, "status": status, "payload": payload}

    @staticmethod
    def _error(seq, status, payload):
        return {"ok": False, "seq": seq, "status": status, "payload": payload}


def main():
    controller = Controller()
    poller = uselect.poll()
    poller.register(sys.stdin, uselect.POLLIN)

    while True:
        controller.tick()
        events = poller.poll(20)
        if not events:
            continue

        # Drain all buffered lines, only process the latest one.
        # This prevents command queue buildup when the host sends
        # faster than we process (e.g. 30Hz tracking commands).
        line = None
        while True:
            l = sys.stdin.readline()
            if not l:
                break
            line = l
            # Check if more data is ready without blocking
            more = poller.poll(0)
            if not more:
                break

        if not line:
            continue

        try:
            message = json.loads(line)
        except Exception as exc:
            response = {"ok": False, "seq": None, "status": "invalid_json", "payload": {"error": str(exc)}}
        else:
            response = controller.handle(message)

        sys.stdout.write(json.dumps(response) + "\n")


if __name__ == "__main__":
    main()
