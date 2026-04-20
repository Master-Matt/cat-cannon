from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ControllerRequest:
    command: str
    payload: dict[str, Any]
    sequence: int

    def to_wire(self) -> bytes:
        message = {
            "seq": self.sequence,
            "command": self.command,
            "payload": self.payload,
        }
        return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ControllerResponse:
    ok: bool
    sequence: int | None
    status: str
    payload: dict[str, Any]

    @classmethod
    def from_wire(cls, raw_line: bytes) -> "ControllerResponse":
        decoded = json.loads(raw_line.decode("utf-8"))
        return cls(
            ok=bool(decoded["ok"]),
            sequence=decoded.get("seq"),
            status=str(decoded.get("status", "")),
            payload=dict(decoded.get("payload", {})),
        )


def build_request(sequence: int, command: str, **payload: Any) -> ControllerRequest:
    return ControllerRequest(command=command, payload=payload, sequence=sequence)

