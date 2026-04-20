from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

try:
    import serial
except ImportError:  # pragma: no cover - exercised only in runtime environments without pyserial
    serial = None


DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT = 1.0
DEFAULT_CHUNK_SIZE = 192
DEFAULT_RAW_REPL_ATTEMPTS = 3
DEFAULT_RAW_REPL_RETRY_DELAY_S = 0.2


class MicroPythonDeployError(RuntimeError):
    """Raised when firmware deployment to the Pico fails."""


def _read_until(transport, marker: bytes) -> bytes:
    if not hasattr(transport, "read_until"):
        raise MicroPythonDeployError("Serial transport must support read_until().")
    chunk = transport.read_until(marker)
    if not chunk.endswith(marker):
        raise MicroPythonDeployError(
            f"Timed out waiting for {marker!r}; received {chunk!r} instead."
        )
    return chunk


def _enter_raw_repl(
    transport,
    *,
    attempts: int = DEFAULT_RAW_REPL_ATTEMPTS,
    retry_delay_s: float = DEFAULT_RAW_REPL_RETRY_DELAY_S,
) -> None:
    last_banner = b""
    for attempt in range(attempts):
        transport.write(b"\r\x03\x03\x01")
        banner = _read_until(transport, b">")
        if b"raw REPL" in banner:
            return
        last_banner = banner
        if attempt < attempts - 1:
            time.sleep(retry_delay_s)

    raise MicroPythonDeployError(f"Unexpected raw REPL banner: {last_banner!r}")


def _exec_raw(transport, command: str) -> str:
    transport.write(command.encode("utf-8"))
    transport.write(b"\x04")

    acknowledgement = transport.read(2)
    if acknowledgement != b"OK":
        raise MicroPythonDeployError(
            f"Raw REPL did not acknowledge command; received {acknowledgement!r}"
        )

    stdout = _read_until(transport, b"\x04")[:-1]
    stderr = _read_until(transport, b"\x04")[:-1]
    _read_until(transport, b">")

    if stderr:
        raise MicroPythonDeployError(
            "Pico rejected deployment command:\n" + stderr.decode("utf-8", errors="replace")
        )

    return stdout.decode("utf-8", errors="replace")


def _write_file_contents(transport, remote_path: str, content: str, chunk_size: int) -> None:
    _exec_raw(transport, f"f = open({remote_path!r}, 'w')")
    try:
        for offset in range(0, len(content), chunk_size):
            chunk = content[offset : offset + chunk_size]
            _exec_raw(transport, f"f.write({chunk!r})")
    finally:
        _exec_raw(transport, "f.close()")


def write_text_file(
    transport,
    remote_path: str,
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    _enter_raw_repl(transport)
    _write_file_contents(transport, remote_path=remote_path, content=content, chunk_size=chunk_size)


def _hard_reset(transport) -> None:
    transport.write(b"import machine\nmachine.reset()\n")
    transport.write(b"\x04")
    if hasattr(transport, "flush"):
        transport.flush()


def deploy_files(
    port: str,
    files: Iterable[tuple[str, Path]],
    *,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout: float = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    transport_factory: Callable[..., object] | None = None,
) -> None:
    if transport_factory is None:
        if serial is None:
            raise MicroPythonDeployError("pyserial is required for Pico deployment.")
        transport_factory = serial.Serial

    transport = transport_factory(port=port, baudrate=baudrate, timeout=timeout)
    try:
        _enter_raw_repl(transport)
        for remote_name, local_path in files:
            content = Path(local_path).read_text(encoding="utf-8")
            _write_file_contents(
                transport,
                remote_path=remote_name,
                content=content,
                chunk_size=chunk_size,
            )
        _hard_reset(transport)
    finally:
        transport.close()
