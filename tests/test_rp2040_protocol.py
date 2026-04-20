from cat_cannon.adapters.rp2040_protocol import ControllerResponse, build_request


def test_request_serializes_to_json_line() -> None:
    request = build_request(7, "apply_delta", pan_delta_deg=1.25, tilt_delta_deg=-0.5)

    assert request.to_wire() == (
        b'{"seq":7,"command":"apply_delta","payload":{"pan_delta_deg":1.25,"tilt_delta_deg":-0.5}}\n'
    )


def test_response_parses_from_json_line() -> None:
    response = ControllerResponse.from_wire(
        b'{"ok":true,"seq":3,"status":"pong","payload":{"enabled":false}}\n'
    )

    assert response.ok is True
    assert response.sequence == 3
    assert response.status == "pong"
    assert response.payload == {"enabled": False}

