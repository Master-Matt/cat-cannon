import threading

from cat_cannon.app.controller_session import ControllerSession
from cat_cannon.config import ServoLimits


class FakeController:
    def __init__(self) -> None:
        self.handshakes = 0
        self.heartbeats = 0
        self.safe_stops = 0
        self.closed = 0
        self.enabled = []
        self.servo_limits = []
        self.motion_directions = []
        self.heartbeat_payload = {"enabled": True}
        self.heartbeat_error: Exception | None = None
        self.enabled_again = threading.Event()

    def handshake(self):
        self.handshakes += 1

    def heartbeat(self):
        self.heartbeats += 1
        if self.heartbeat_error is not None:
            raise self.heartbeat_error
        return type("Response", (), {"payload": dict(self.heartbeat_payload)})()

    def set_enabled(self, enabled: bool):
        self.enabled.append(enabled)
        if self.enabled.count(True) >= 2:
            self.enabled_again.set()

    def set_servo_limits(self, **limits):
        self.servo_limits.append(limits)

    def set_motion_directions(self, **directions):
        self.motion_directions.append(directions)

    def safe_stop(self):
        self.safe_stops += 1

    def close(self):
        self.closed += 1


def test_controller_session_manages_handshake_enable_disable_and_stop() -> None:
    controller = FakeController()
    session = ControllerSession(controller=controller, heartbeat_interval_s=0.01)

    session.start()
    session.enable()
    session.disable()
    session.stop()

    assert controller.handshakes == 1
    assert controller.heartbeats >= 1
    assert controller.enabled[:2] == [True, False]
    assert controller.enabled[-1] is False
    assert controller.safe_stops == 1
    assert controller.closed == 1


def test_controller_session_applies_servo_limits_after_handshake() -> None:
    controller = FakeController()
    session = ControllerSession(
        controller=controller,
        servo_limits=ServoLimits(
            pan_min_deg=20,
            pan_max_deg=150,
            tilt_min_deg=40,
            tilt_max_deg=120,
        ),
        heartbeat_interval_s=60,
    )

    session.start()
    session.stop()

    assert controller.handshakes == 1
    assert controller.servo_limits == [
        {
            "pan_min_deg": 20,
            "pan_max_deg": 150,
            "tilt_min_deg": 40,
            "tilt_max_deg": 120,
        }
    ]


def test_controller_session_applies_camera_pov_motion_directions() -> None:
    controller = FakeController()
    session = ControllerSession(
        controller=controller,
        servo_limits=ServoLimits(
            pan_min_deg=20,
            pan_max_deg=150,
            tilt_min_deg=40,
            tilt_max_deg=120,
            pan_left_deg=150,
            pan_right_deg=20,
            tilt_top_deg=120,
            tilt_bottom_deg=40,
        ),
        heartbeat_interval_s=60,
    )

    session.start()
    session.stop()

    assert controller.motion_directions == [
        {
            "pan_delta_sign": -1,
            "tilt_delta_sign": -1,
        }
    ]


def test_controller_session_restores_armed_state_after_watchdog_disable() -> None:
    controller = FakeController()
    controller.heartbeat_payload = {"enabled": False}
    session = ControllerSession(controller=controller, heartbeat_interval_s=0.01)

    session.start()
    session.enable()
    recovered = controller.enabled_again.wait(timeout=0.5)
    session.stop()

    assert recovered is True
    assert controller.enabled[:2] == [True, True]


def test_controller_session_faults_after_consecutive_heartbeat_failures() -> None:
    controller = FakeController()
    controller.heartbeat_error = OSError("serial device disconnected")
    session = ControllerSession(controller=controller, max_consecutive_failures=2)

    assert session._heartbeat_once() is False
    assert session.faulted is False
    assert session._heartbeat_once() is False

    assert session.faulted is True
    assert session.fault_reason == "OSError: serial device disconnected"


def test_controller_session_resets_transient_heartbeat_failure_count() -> None:
    controller = FakeController()
    session = ControllerSession(controller=controller, max_consecutive_failures=2)
    controller.heartbeat_error = OSError("transient serial timeout")

    assert session._heartbeat_once() is False
    controller.heartbeat_error = None
    assert session._heartbeat_once() is True
    controller.heartbeat_error = OSError("another transient serial timeout")
    assert session._heartbeat_once() is False

    assert session.faulted is False
