from cat_cannon.adapters.rp2040_serial import RP2040ProtocolError, RP2040SerialController


class FakeSerial:
    def __init__(self, responses):
        self.responses = list(responses)
        self.writes = []
        self.closed = False
        self.input_resets = 0

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def readline(self) -> bytes:
        if not self.responses:
            return b""
        return self.responses.pop(0)

    def reset_input_buffer(self) -> None:
        self.input_resets += 1

    def close(self) -> None:
        self.closed = True


def test_controller_sends_expected_fire_command() -> None:
    transport = FakeSerial([b'{"ok":true,"seq":1,"status":"firing","payload":{}}\n'])
    controller = RP2040SerialController(transport=transport, fire_pulse_ms=140)

    controller.fire()

    assert transport.writes == [b'{"seq":1,"command":"fire","payload":{"duration_ms":140}}\n']


def test_controller_sends_expected_set_fire_output_command() -> None:
    transport = FakeSerial([b'{"ok":true,"seq":1,"status":"fire_output_set","payload":{"solenoid_active":true}}\n'])
    controller = RP2040SerialController(transport=transport)

    controller.set_fire_output(True)

    assert transport.writes == [b'{"seq":1,"command":"set_fire_output","payload":{"active":true}}\n']


def test_controller_sends_expected_servo_limits_command() -> None:
    transport = FakeSerial([b'{"ok":true,"seq":1,"status":"servo_limits_set","payload":{}}\n'])
    controller = RP2040SerialController(transport=transport)

    controller.set_servo_limits(
        pan_min_deg=12.5,
        pan_max_deg=160.0,
        tilt_min_deg=35.0,
        tilt_max_deg=125.5,
    )

    assert transport.writes == [
        (
            b'{"seq":1,"command":"set_servo_limits",'
            b'"payload":{"pan_min_deg":12.5,"pan_max_deg":160.0,'
            b'"tilt_min_deg":35.0,"tilt_max_deg":125.5}}\n'
        )
    ]


def test_controller_uses_detected_motion_directions_for_camera_pov_deltas() -> None:
    transport = FakeSerial(
        [
            b'{"ok":true,"seq":1,"status":"delta_applied","payload":{}}\n',
            b'{"ok":true,"seq":2,"status":"velocity_set","payload":{}}\n',
        ]
    )
    controller = RP2040SerialController(transport=transport)
    controller.set_motion_directions(pan_delta_sign=1, tilt_delta_sign=-1)

    controller.apply_tracking_delta(2.5, 1.5)
    controller.set_velocity(-4.0, 3.0)

    assert transport.writes == [
        b'{"seq":1,"command":"apply_delta","payload":{"pan_delta_deg":2.5,"tilt_delta_deg":-1.5}}\n',
        b'{"seq":2,"command":"set_velocity","payload":{"pan_deg_s":-4.0,"tilt_deg_s":-3.0}}\n',
    ]


def test_controller_raises_on_failed_response() -> None:
    transport = FakeSerial([b'{"ok":false,"seq":1,"status":"disabled","payload":{}}\n'])
    controller = RP2040SerialController(transport=transport)

    try:
        controller.fire()
    except RP2040ProtocolError as exc:
        assert "disabled" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RP2040ProtocolError")


def test_controller_skips_non_json_serial_noise_before_valid_response() -> None:
    transport = FakeSerial(
        [
            b'MicroPython v1.28.0 on 2026-04-06; Raspberry Pi Pico with RP2040\r\n',
            b'Type "help()" for more information.\r\n',
            b'{"ok":true,"seq":1,"status":"pong","payload":{"enabled":false}}\n',
        ]
    )
    controller = RP2040SerialController(transport=transport)

    response = controller.handshake()

    assert response.status == "pong"
    assert response.payload == {"enabled": False}


def test_controller_raises_when_only_noise_is_received() -> None:
    transport = FakeSerial([b"Traceback (most recent call last):\r\n"] * 7)
    controller = RP2040SerialController(transport=transport)

    try:
        controller.handshake()
    except RP2040ProtocolError as exc:
        assert "No valid RP2040 JSON response" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RP2040ProtocolError")
