from __future__ import annotations

from dataclasses import dataclass, field

from cat_cannon.domain.models import CounterZone, Point


@dataclass(frozen=True)
class CalibrationLayout:
    window_width: int
    window_height: int
    preview_width: int
    preview_height: int
    preview_offset_x: int
    preview_offset_y: int
    panel_x: int


def map_display_to_frame(
    *,
    display_x: int,
    display_y: int,
    frame_width: int,
    frame_height: int,
    layout: CalibrationLayout,
) -> Point:
    local_x = min(max(display_x - layout.preview_offset_x, 0), layout.preview_width)
    local_y = min(max(display_y - layout.preview_offset_y, 0), layout.preview_height)
    scale_x = frame_width / layout.preview_width
    scale_y = frame_height / layout.preview_height
    return Point(x=local_x * scale_x, y=local_y * scale_y)


@dataclass
class ZoneCalibrationSession:
    zone_prefix: str = "zone"
    zones: list[CounterZone] = field(default_factory=list)
    pending_points: list[Point] = field(default_factory=list)

    def add_point(self, point: Point) -> CounterZone | None:
        self.pending_points.append(point)
        if len(self.pending_points) < 4:
            return None
        zone = CounterZone(
            zone_id=f"{self.zone_prefix}-{len(self.zones) + 1}",
            polygon=tuple(self.pending_points[:4]),
        )
        self.zones.append(zone)
        self.pending_points.clear()
        return zone

    def undo(self) -> None:
        if self.pending_points:
            self.pending_points.pop()
            return
        if self.zones:
            self.zones.pop()

    def clear_pending(self) -> None:
        self.pending_points.clear()
