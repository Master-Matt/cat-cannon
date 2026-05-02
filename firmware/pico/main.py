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
        self._min_deg = min_deg
        self._max_deg = max_deg
        self.angle_deg = home_deg
        self.write(home_deg)

    def write(self, angle_deg):
        clamped = min(self._max_deg, max(self._min_deg, angle_deg))
        self.angle_deg = clamped
        duty = self._angle_to_duty_u16(clamped)
        self._pwm.duty_u16(duty)

    def delta(self, amount_deg):
        self.write(self.angle_deg + amount_deg)

    def _angle_to_duty_u16(self, angle_deg):
        span_deg = self._max_deg - self._min_deg or 1.0
        fraction = (angle_deg - self._min_deg) / span_deg
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
        self.solenoid = Pin(cfg.SOLENOID_PIN, Pin.OUT)
        self.solenoid.value(0)
        self.led = Pin(cfg.STATUS_LED_PIN, Pin.OUT)
        self.enabled = False
        self.last_contact_ms = time.ticks_ms()
        self.fire_until_ms = None

    def tick(self):
        now = time.ticks_ms()
        if self.fire_until_ms is not None and time.ticks_diff(self.fire_until_ms, now) <= 0:
            self.solenoid.value(0)
            self.fire_until_ms = None

        if time.ticks_diff(now, self.last_contact_ms) > cfg.WATCHDOG_TIMEOUT_MS:
            self.enabled = False
            self.solenoid.value(0)
            self.fire_until_ms = None
            self.led.value(0)

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
                    self.solenoid.value(0)
                    self.fire_until_ms = None
                return self._ok(seq, "enabled", self._status_payload())
            if command == "set_angles":
                self.pan.write(float(payload["pan_deg"]))
                self.tilt.write(float(payload["tilt_deg"]))
                return self._ok(seq, "angles_set", self._status_payload())
            if command == "apply_delta":
                self.pan.delta(float(payload.get("pan_delta_deg", 0.0)))
                self.tilt.delta(float(payload.get("tilt_delta_deg", 0.0)))
                return self._ok(seq, "delta_applied", self._status_payload())
            if command == "safe_stop":
                self.solenoid.value(0)
                self.fire_until_ms = None
                return self._ok(seq, "safe_stop", self._status_payload())
            if command == "set_fire_output":
                active = bool(payload.get("active", False))
                if active and not self.enabled:
                    return self._error(seq, "disabled", self._status_payload())
                self.fire_until_ms = None
                self.solenoid.value(1 if active else 0)
                return self._ok(seq, "fire_output_set", self._status_payload())
            if command == "fire":
                if not self.enabled:
                    return self._error(seq, "disabled", self._status_payload())
                duration_ms = min(
                    cfg.MAX_FIRE_DURATION_MS,
                    max(1, int(payload.get("duration_ms", cfg.DEFAULT_FIRE_DURATION_MS))),
                )
                self.solenoid.value(1)
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
            "solenoid_active": bool(self.solenoid.value()),
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

        line = sys.stdin.readline()
        if not line:
            continue

        try:
            message = json.loads(line)
        except Exception as exc:
            response = {"ok": False, "seq": None, "status": "invalid_json", "payload": {"error": str(exc)}}
        else:
            response = controller.handle(message)

        sys.stdout.write(json.dumps(response) + "\n")


main()
