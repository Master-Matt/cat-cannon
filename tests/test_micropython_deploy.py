from pathlib import Path

import pytest

from cat_cannon.adapters.micropython_deploy import (
    MicroPythonDeployError,
    deploy_files,
    write_text_file,
)


class FakeTransport:
    def __init__(self, read_chunks=None, until_chunks=None) -> None:
        self.read_chunks = list(read_chunks or [])
        self.until_chunks = list(until_chunks or [])
        self.writes = []
        self.flush_count = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def read(self, size: int) -> bytes:
        if not self.read_chunks:
            raise AssertionError(f"Unexpected read({size})")
        chunk = self.read_chunks.pop(0)
        if len(chunk) != size:
            raise AssertionError(f"Expected {size} bytes, got {len(chunk)}")
        return chunk

    def read_until(self, marker: bytes) -> bytes:
        if not self.until_chunks:
            raise AssertionError(f"Unexpected read_until({marker!r})")
        return self.until_chunks.pop(0)

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def _exec_responses(command_count: int) -> tuple[list[bytes], list[bytes]]:
    read_chunks = [b"OK"] * command_count
    until_chunks = [b"raw REPL; CTRL-B to exit\r\n>"]
    for _ in range(command_count):
        until_chunks.extend([b"\x04", b"\x04", b">"])
    return read_chunks, until_chunks


def test_write_text_file_enters_raw_repl_and_chunks_writes() -> None:
    read_chunks, until_chunks = _exec_responses(command_count=4)
    transport = FakeTransport(read_chunks=read_chunks, until_chunks=until_chunks)

    write_text_file(
        transport=transport,
        remote_path="main.py",
        content="abcdefgh",
        chunk_size=4,
    )

    assert transport.writes == [
        b"\r\x03\x03\x01",
        b"f = open('main.py', 'w')",
        b"\x04",
        b"f.write('abcd')",
        b"\x04",
        b"f.write('efgh')",
        b"\x04",
        b"f.close()",
        b"\x04",
    ]


def test_write_text_file_retries_after_normal_repl_banner() -> None:
    transport = FakeTransport(
        read_chunks=[b"OK", b"OK", b"OK"],
        until_chunks=[
            b'MicroPython v1.28.0 on 2026-04-06; Raspberry Pi Pico with RP2040\r\nType "help()" for more information.\r\n>',
            b"raw REPL; CTRL-B to exit\r\n>",
            b"\x04",
            b"\x04",
            b">",
            b"\x04",
            b"\x04",
            b">",
            b"\x04",
            b"\x04",
            b">",
        ],
    )

    write_text_file(
        transport=transport,
        remote_path="main.py",
        content="abc",
        chunk_size=8,
    )

    assert transport.writes[:2] == [b"\r\x03\x03\x01", b"\r\x03\x03\x01"]


def test_write_text_file_raises_after_exhausting_raw_repl_retries() -> None:
    transport = FakeTransport(
        until_chunks=[
            b"MicroPython ready\r\n>",
            b"still not raw\r\n>",
            b"plain repl again\r\n>",
        ]
    )

    with pytest.raises(MicroPythonDeployError, match="Unexpected raw REPL banner"):
        write_text_file(transport=transport, remote_path="main.py", content="abc")


def test_deploy_files_uploads_all_files_and_resets_board(tmp_path: Path) -> None:
    main_file = tmp_path / "main.py"
    config_file = tmp_path / "pico_config.py"
    main_file.write_text("print('hello')\n", encoding="utf-8")
    config_file.write_text("ENABLED = True\n", encoding="utf-8")

    read_chunks, until_chunks = _exec_responses(command_count=6)
    transport = FakeTransport(read_chunks=read_chunks, until_chunks=until_chunks)

    def fake_transport_factory(port: str, baudrate: int, timeout: float):
        assert port == "/dev/ttyACM0"
        assert baudrate == 115200
        assert timeout == 1.0
        return transport

    deploy_files(
        port="/dev/ttyACM0",
        files=[
            ("main.py", main_file),
            ("pico_config.py", config_file),
        ],
        transport_factory=fake_transport_factory,
    )

    assert transport.writes[:13] == [
        b"\r\x03\x03\x01",
        b"f = open('main.py', 'w')",
        b"\x04",
        b"f.write(\"print('hello')\\n\")",
        b"\x04",
        b"f.close()",
        b"\x04",
        b"f = open('pico_config.py', 'w')",
        b"\x04",
        b"f.write('ENABLED = True\\n')",
        b"\x04",
        b"f.close()",
        b"\x04",
    ]
    assert transport.writes[-2:] == [b"import machine\nmachine.reset()\n", b"\x04"]
    assert transport.flush_count == 1
    assert transport.closed
