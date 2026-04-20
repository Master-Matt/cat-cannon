from __future__ import annotations

from cat_cannon.domain.models import CounterZone, Detection, Point


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    if len(polygon) < 3:
        return False

    inside = False
    previous = polygon[-1]
    for current in polygon:
        intersects = (
            (current.y > point.y) != (previous.y > point.y)
            and point.x
            < (previous.x - current.x) * (point.y - current.y) / (previous.y - current.y + 1e-9)
            + current.x
        )
        if intersects:
            inside = not inside
        previous = current
    return inside


def detection_footpoint_in_zone(detection: Detection, zone: CounterZone) -> bool:
    return point_in_polygon(detection.bbox.bottom_center, zone.polygon)

