from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

PerceptionT = TypeVar("PerceptionT")


@dataclass
class PerceptionCache(Generic[PerceptionT]):
    """Keep a perception result only while its source frame is recent."""

    max_age_seconds: float
    _perception: PerceptionT | None = field(default=None, init=False)
    _updated_at: float | None = field(default=None, init=False)

    def update(self, perception: PerceptionT, *, now: float | None = None) -> None:
        self._perception = perception
        self._updated_at = time.monotonic() if now is None else float(now)

    def clear(self) -> None:
        self._perception = None
        self._updated_at = None

    def current(self, *, now: float | None = None) -> PerceptionT | None:
        if self._perception is None or self._updated_at is None:
            return None
        now_s = time.monotonic() if now is None else float(now)
        if now_s - self._updated_at > max(0.0, self.max_age_seconds):
            self.clear()
            return None
        return self._perception
