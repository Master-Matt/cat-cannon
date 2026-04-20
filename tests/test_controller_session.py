from cat_cannon.app.controller_session import ControllerSession


class FakeController:
    def __init__(self) -> None:
        self.handshakes = 0
        self.heartbeats = 0
        self.safe_stops = 0
        self.closed = 0
        self.enabled = []

    def handshake(self):
        self.handshakes += 1

    def heartbeat(self):
        self.heartbeats += 1

    def set_enabled(self, enabled: bool):
        self.enabled.append(enabled)

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
