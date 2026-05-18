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

    def handshake(self):
        self.handshakes += 1

    def heartbeat(self):
        self.heartbeats += 1

    def set_enabled(self, enabled: bool):
        self.enabled.append(enabled)

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
    assert controller.enabled == [True, False]
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
