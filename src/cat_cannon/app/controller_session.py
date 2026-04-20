from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from cat_cannon.adapters.rp2040_serial import RP2040SerialController


@dataclass
class ControllerSession:
    controller: RP2040SerialController
    heartbeat_interval_s: float = 0.5
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self.controller.handshake()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="rp2040-heartbeat", daemon=True)
        self._thread.start()

    def enable(self) -> None:
        self.controller.set_enabled(True)

    def disable(self) -> None:
        self.controller.set_enabled(False)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            self.controller.safe_stop()
        finally:
            self.controller.close()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            self.controller.heartbeat()
            self._stop_event.wait(self.heartbeat_interval_s)

