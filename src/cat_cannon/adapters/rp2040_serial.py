from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from cat_cannon.adapters.interfaces import TurretController
from cat_cannon.adapters.rp2040_discovery import autodetect_port
from cat_cannon.adapters.rp2040_protocol import ControllerResponse, build_request

try:
    import serial
except ImportError:  # pragma: no cover - exercised only in runtime environments without pyserial
    serial = None


class SerialLike(Protocol):
    def write(self, data: bytes) -> int:
        ...

    def readline(self) -> bytes:
        ...

    def reset_input_buffer(self) -> None:
        ...

    def close(self) -> None:
        ...


class RP2040ProtocolError(RuntimeError):
    """Raised when the RP2040 returns an invalid or failed response."""


@dataclass
class RP2040SerialController(TurretController):
    transport: SerialLike
    fire_pulse_ms: int = 120
    pan_delta_sign: int = -1
    tilt_delta_sign: int = 1
    _sequence: int = field(default=0, init=False)
    _io_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _max_noise_lines: int = field(default=8, init=False, repr=False)

    @classmethod
    def open(
        cls,
        port: str | None = None,
        baudrate: int = 115200,
        timeout: float = 0.5,
        fire_pulse_ms: int = 120,
    ) -> "RP2040SerialController":
        if serial is None:  # pragma: no cover
            raise RuntimeError("pyserial is required to open the RP2040 serial controller.")
        resolved_port = port or autodetect_port()
        transport = serial.Serial(port=resolved_port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.2)
        if hasattr(transport, "reset_input_buffer"):
            transport.reset_input_buffer()
        return cls(transport=transport, fire_pulse_ms=fire_pulse_ms)

    def handshake(self) -> ControllerResponse:
        return self._send("ping")

    def set_enabled(self, enabled: bool) -> ControllerResponse:
        return self._send("set_enabled", enabled=enabled)

    def heartbeat(self) -> ControllerResponse:
        return self._send("heartbeat")

    def set_angles(self, pan_deg: float, tilt_deg: float) -> ControllerResponse:
        return self._send("set_angles", pan_deg=pan_deg, tilt_deg=tilt_deg)

    def apply_tracking_delta(self, pan_delta: float, tilt_delta: float) -> None:
        self._send(
            "apply_delta",
            pan_delta_deg=self.pan_delta_sign * pan_delta,
            tilt_delta_deg=self.tilt_delta_sign * tilt_delta,
        )

    def set_velocity(self, pan_deg_s: float, tilt_deg_s: float) -> None:
        """Set continuous servo velocity in degrees/sec. Pico interpolates at 50Hz."""
        self._send(
            "set_velocity",
            pan_deg_s=self.pan_delta_sign * pan_deg_s,
            tilt_deg_s=self.tilt_delta_sign * tilt_deg_s,
        )

    def set_motion_directions(self, *, pan_delta_sign: int, tilt_delta_sign: int) -> None:
        """Map camera-POV deltas to physical servo-angle deltas."""
        self.pan_delta_sign = 1 if pan_delta_sign >= 0 else -1
        self.tilt_delta_sign = 1 if tilt_delta_sign >= 0 else -1

    def set_servo_limits(
        self,
        *,
        pan_min_deg: float,
        pan_max_deg: float,
        tilt_min_deg: float,
        tilt_max_deg: float,
    ) -> ControllerResponse:
        return self._send(
            "set_servo_limits",
            pan_min_deg=pan_min_deg,
            pan_max_deg=pan_max_deg,
            tilt_min_deg=tilt_min_deg,
            tilt_max_deg=tilt_max_deg,
        )

    def fire(self) -> None:
        self._send("fire", duration_ms=self.fire_pulse_ms)

    def set_fire_output(self, active: bool) -> ControllerResponse:
        return self._send("set_fire_output", active=active)

    def safe_stop(self) -> None:
        self._send("safe_stop")

    def relax(self) -> ControllerResponse:
        """Detach servos to stop buzzing. They re-engage on next movement."""
        return self._send("relax")

    def status(self) -> ControllerResponse:
        return self._send("status")

    def close(self) -> None:
        self.transport.close()

    def _send(self, command: str, **payload: object) -> ControllerResponse:
        with self._io_lock:
            self._sequence += 1
            self.transport.reset_input_buffer()
            request = build_request(self._sequence, command, **payload)
            self.transport.write(request.to_wire())
            response = self._read_response(command=command, sequence=self._sequence)
        if not response.ok:
            raise RP2040ProtocolError(
                f"RP2040 command '{command}' failed with status '{response.status}': {response.payload}"
            )
        if response.sequence != self._sequence:
            raise RP2040ProtocolError(
                f"RP2040 response sequence mismatch: expected {self._sequence}, got {response.sequence}"
            )
        return response

    def _read_response(self, command: str, sequence: int) -> ControllerResponse:
        skipped_lines: list[str] = []
        for _ in range(self._max_noise_lines):
            raw_line = self.transport.readline()
            if not raw_line:
                continue
            try:
                return ControllerResponse.from_wire(raw_line)
            except (KeyError, TypeError, ValueError):
                decoded = raw_line.decode("utf-8", errors="replace").strip()
                if decoded:
                    skipped_lines.append(decoded)

        detail = ""
        if skipped_lines:
            detail = f" Skipped lines: {skipped_lines!r}"
        raise RP2040ProtocolError(
            f"No valid RP2040 JSON response for '{command}' sequence {sequence}.{detail}"
        )
