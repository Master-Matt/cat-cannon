from __future__ import annotations

from dataclasses import dataclass

from cat_cannon.app.supervisor import SupervisorLoop
from cat_cannon.domain.models import Detection


@dataclass(frozen=True)
class ReplayFrame:
    detections: list[Detection]
    frame_width: int
    frame_height: int
    armed: bool = True


@dataclass(frozen=True)
class ReplaySnapshot:
    frame_index: int
    fire_count: int
    stop_count: int


def run_replay(supervisor: SupervisorLoop, frames: list[ReplayFrame]) -> list[ReplaySnapshot]:
    snapshots: list[ReplaySnapshot] = []
    for index, frame in enumerate(frames):
        supervisor.process_frame(
            detections=frame.detections,
            frame_width=frame.frame_width,
            frame_height=frame.frame_height,
            armed=frame.armed,
        )
        snapshots.append(
            ReplaySnapshot(
                frame_index=index,
                fire_count=getattr(supervisor.controller, "fired", 0),
                stop_count=getattr(supervisor.controller, "stopped", 0),
            )
        )
    return snapshots

