from __future__ import annotations

from dataclasses import dataclass

try:
    from serial.tools import list_ports
except ImportError:  # pragma: no cover - environment dependent
    list_ports = None


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str
    vid: int | None
    pid: int | None


class RP2040DiscoveryError(RuntimeError):
    """Raised when a Pico-compatible port cannot be resolved unambiguously."""


def list_candidate_ports() -> list[SerialPortInfo]:
    if list_ports is None:  # pragma: no cover
        raise RuntimeError("pyserial is required for serial-port discovery.")

    candidates: list[SerialPortInfo] = []
    for port in list_ports.comports():
        info = SerialPortInfo(
            device=str(port.device),
            description=str(port.description),
            vid=getattr(port, "vid", None),
            pid=getattr(port, "pid", None),
        )
        if _looks_like_rp2040(info):
            candidates.append(info)
    return candidates


def autodetect_port() -> str:
    candidates = list_candidate_ports()
    if not candidates:
        raise RP2040DiscoveryError("No RP2040-compatible serial device was found.")
    if len(candidates) > 1:
        joined = ", ".join(candidate.device for candidate in candidates)
        raise RP2040DiscoveryError(
            f"Multiple RP2040-compatible serial devices found: {joined}. Pass --port explicitly."
        )
    return candidates[0].device


def _looks_like_rp2040(port: SerialPortInfo) -> bool:
    description = port.description.lower()
    if "pico" in description or "rp2040" in description or "usb serial device" in description:
        return True
    # Raspberry Pi Pico / RP2040 USB VID/PID seen in common MicroPython and USB-CDC setups.
    if port.vid == 0x2E8A:
        return True
    return False
