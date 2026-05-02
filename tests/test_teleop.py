from cat_cannon.app.teleop import TeleopState, handle_key


class FakeController:
    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.fired = 0
        self.fire_output_states: list[bool] = []
        self.safe_stops = 0
        self.status_calls = 0

    def apply_tracking_delta(self, pan_delta: float, tilt_delta: float) -> None:
        self.moves.append((pan_delta, tilt_delta))

    def fire(self) -> None:
        self.fired += 1

    def set_fire_output(self, active: bool) -> None:
        self.fire_output_states.append(active)

    def safe_stop(self) -> None:
        self.safe_stops += 1

    def status(self):
        self.status_calls += 1
        return type("Status", (), {"payload": {"enabled": True, "pan_deg": 90.0, "tilt_deg": 90.0}})()


class FakeSession:
    def __init__(self) -> None:
        self.enabled: list[bool] = []

    def enable(self) -> None:
        self.enabled.append(True)

    def disable(self) -> None:
        self.enabled.append(False)


def test_handle_key_moves_with_wasd_when_armed() -> None:
    controller = FakeController()
    session = FakeSession()
    state = TeleopState(armed=True, step_deg=4.0)

    state, message, should_exit = handle_key("w", state=state, session=session, controller=controller)
    assert state.armed is True
    assert controller.moves[-1] == (0.0, -4.0)
    assert should_exit is False
    assert "tilt up" in message

    state, message, _ = handle_key("a", state=state, session=session, controller=controller)
    assert controller.moves[-1] == (-4.0, 0.0)
    assert "pan left" in message


def test_handle_key_ignores_motion_until_armed() -> None:
    controller = FakeController()
    session = FakeSession()
    state = TeleopState(armed=False, step_deg=3.0)

    next_state, message, should_exit = handle_key("d", state=state, session=session, controller=controller)

    assert next_state.armed is False
    assert controller.moves == []
    assert should_exit is False
    assert "arm first" in message


def test_handle_key_arms_disarms_and_safe_stops() -> None:
    controller = FakeController()
    session = FakeSession()
    state = TeleopState(armed=False, step_deg=3.0)

    state, message, _ = handle_key("e", state=state, session=session, controller=controller)
    assert state.armed is True
    assert session.enabled == [True]
    assert "armed" in message

    state, message, _ = handle_key("x", state=state, session=session, controller=controller)
    assert state.armed is False
    assert session.enabled == [True, False]
    assert controller.safe_stops == 1
    assert "disarmed" in message


def test_handle_key_fires_only_when_armed() -> None:
    controller = FakeController()
    session = FakeSession()
    state = TeleopState(armed=False, step_deg=3.0)

    _, message, _ = handle_key(" ", state=state, session=session, controller=controller)
    assert controller.fired == 0
    assert "arm first" in message

    state = TeleopState(armed=True, step_deg=3.0)
    _, message, _ = handle_key(" ", state=state, session=session, controller=controller)
    assert controller.fired == 1
    assert "fired" in message


def test_handle_key_supports_fire_output_toggle() -> None:
    controller = FakeController()
    session = FakeSession()
    state = TeleopState(armed=False, step_deg=3.0)

    _, message, _ = handle_key("r", state=state, session=session, controller=controller)
    assert controller.fire_output_states == []
    assert "arm first" in message

    state = TeleopState(armed=True, step_deg=3.0)
    _, message, _ = handle_key("r", state=state, session=session, controller=controller)
    assert controller.fire_output_states == [True]
    assert "fire output on" in message

    _, message, _ = handle_key("f", state=state, session=session, controller=controller)
    assert controller.fire_output_states == [True, False]
    assert "fire output off" in message


def test_handle_key_supports_status_and_quit() -> None:
    controller = FakeController()
    session = FakeSession()
    state = TeleopState(armed=False, step_deg=3.0)

    _, message, should_exit = handle_key("p", state=state, session=session, controller=controller)
    assert controller.status_calls == 1
    assert "status" in message
    assert should_exit is False

    _, message, should_exit = handle_key("q", state=state, session=session, controller=controller)
    assert should_exit is True
    assert "quit" in message
