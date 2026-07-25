from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path


class FakePin:
    OUT = 1

    def __init__(self, pin_id, _mode=None, value=None) -> None:
        self.pin_id = pin_id
        self._value = 0 if value is None else value

    def value(self, value=None):
        if value is None:
            return self._value
        self._value = value
        return self._value


class FakePWM:
    def __init__(self, pin: FakePin) -> None:
        self.pin = pin
        self.frequency = None
        self.duties: list[int] = []

    def freq(self, value: int) -> None:
        self.frequency = value

    def duty_u16(self, value: int) -> None:
        self.duties.append(value)


class FakePoll:
    def register(self, *_args) -> None:
        pass

    def poll(self, *_args):
        return []


def _load_pico_main(monkeypatch):
    firmware_dir = Path(__file__).resolve().parents[1] / "firmware" / "pico"
    monkeypatch.syspath_prepend(str(firmware_dir))
    monkeypatch.setitem(sys.modules, "machine", types.SimpleNamespace(PWM=FakePWM, Pin=FakePin))
    monkeypatch.setitem(
        sys.modules,
        "uselect",
        types.SimpleNamespace(POLLIN=1, poll=lambda: FakePoll()),
    )
    monkeypatch.setattr(time, "ticks_ms", lambda: 0, raising=False)
    monkeypatch.setattr(time, "ticks_diff", lambda first, second: first - second, raising=False)
    monkeypatch.setattr(time, "ticks_add", lambda value, delta: value + delta, raising=False)
    sys.modules.pop("pico_config", None)
    module_name = "_cat_cannon_pico_main_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, firmware_dir / "main.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_soft_servo_limits_do_not_remap_pwm_calibration(monkeypatch) -> None:
    pico_main = _load_pico_main(monkeypatch)
    servo = pico_main.Servo(pin_id=0, min_deg=0.0, max_deg=180.0, home_deg=90.0)
    duty_at_81_before_limits = servo._angle_to_duty_u16(81.0)

    servo.set_limits(80.0, 100.0)

    assert servo._angle_to_duty_u16(81.0) == duty_at_81_before_limits


def test_soft_servo_limits_only_clamp_target_angle(monkeypatch) -> None:
    pico_main = _load_pico_main(monkeypatch)
    servo = pico_main.Servo(pin_id=0, min_deg=0.0, max_deg=180.0, home_deg=90.0)
    expected_duty_at_89 = servo._angle_to_duty_u16(89.0)

    servo.set_limits(88.0, 89.0)

    assert servo.angle_deg == 89.0
    assert servo._pwm.duties[-1] == expected_duty_at_89


def test_pan_servo_accepts_negative_angle_commands(monkeypatch) -> None:
    pico_main = _load_pico_main(monkeypatch)
    controller = pico_main.Controller()

    controller.handle(
        {
            "seq": 1,
            "command": "set_angles",
            "payload": {"pan_deg": 0.0, "tilt_deg": 90.0},
        }
    )
    response = controller.handle(
        {
            "seq": 2,
            "command": "apply_delta",
            "payload": {"pan_delta_deg": -3.0, "tilt_delta_deg": 0.0},
        }
    )

    assert controller.pan.min_deg == -180.0
    assert controller.pan.angle_deg == -3.0
    assert response["payload"]["pan_deg"] == -3.0


def test_detached_servo_reattaches_when_same_angle_is_commanded(monkeypatch) -> None:
    pico_main = _load_pico_main(monkeypatch)
    servo = pico_main.Servo(pin_id=0, min_deg=0.0, max_deg=180.0, home_deg=90.0)
    expected_duty = servo._angle_to_duty_u16(90.0)

    servo.detach()
    writes_after_detach = len(servo._pwm.duties)
    moved = servo.write(90.0)

    assert moved is True
    assert servo.attached is True
    assert len(servo._pwm.duties) == writes_after_detach + 1
    assert servo._pwm.duties[-1] == expected_duty
