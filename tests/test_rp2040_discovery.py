from types import SimpleNamespace

from cat_cannon.adapters.rp2040_discovery import RP2040DiscoveryError, autodetect_port, list_candidate_ports


def test_list_candidate_ports_filters_to_rp2040_like_devices(monkeypatch) -> None:
    fake_ports = [
        SimpleNamespace(device="/dev/ttyACM0", description="Pico", vid=0x2E8A, pid=0x0005),
        SimpleNamespace(device="/dev/ttyUSB0", description="USB-Serial Adapter", vid=0x1A86, pid=0x7523),
    ]

    from cat_cannon.adapters import rp2040_discovery

    monkeypatch.setattr(rp2040_discovery, "list_ports", SimpleNamespace(comports=lambda: fake_ports))

    candidates = list_candidate_ports()

    assert [candidate.device for candidate in candidates] == ["/dev/ttyACM0"]


def test_autodetect_port_returns_single_candidate(monkeypatch) -> None:
    from cat_cannon.adapters import rp2040_discovery

    monkeypatch.setattr(
        rp2040_discovery,
        "list_candidate_ports",
        lambda: [SimpleNamespace(device="/dev/ttyACM0")],
    )

    assert autodetect_port() == "/dev/ttyACM0"


def test_autodetect_port_raises_for_multiple_candidates(monkeypatch) -> None:
    from cat_cannon.adapters import rp2040_discovery

    monkeypatch.setattr(
        rp2040_discovery,
        "list_candidate_ports",
        lambda: [SimpleNamespace(device="/dev/ttyACM0"), SimpleNamespace(device="/dev/ttyACM1")],
    )

    try:
        autodetect_port()
    except RP2040DiscoveryError as exc:
        assert "Multiple RP2040-compatible serial devices found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected RP2040DiscoveryError")
