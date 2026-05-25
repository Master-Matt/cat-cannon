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


def point_in_or_on_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    if len(polygon) < 3:
        return False
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        if _point_on_segment(point, previous, current):
            return True
    return point_in_polygon(point, polygon)


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (point.y - start.y) * (end.x - start.x) - (
        point.x - start.x
    ) * (end.y - start.y)
    if abs(cross) > 1e-9:
        return False
    min_x, max_x = sorted((start.x, end.x))
    min_y, max_y = sorted((start.y, end.y))
    return min_x - 1e-9 <= point.x <= max_x + 1e-9 and (
        min_y - 1e-9 <= point.y <= max_y + 1e-9
    )


def _segments_intersect(
    a1: Point, a2: Point, b1: Point, b2: Point
) -> bool:
    """Check if line segment a1-a2 intersects segment b1-b2."""
    def cross(o: Point, a: Point, b: Point) -> float:
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases (treat as non-intersecting for simplicity)
    return False


def bbox_intersects_zone(detection: Detection, zone: CounterZone) -> bool:
    """Check if any part of the detection bbox overlaps with the zone polygon."""
    bbox = detection.bbox
    polygon = zone.polygon

    # Check if any bbox corner is inside the polygon
    corners = (
        Point(bbox.x, bbox.y),
        Point(bbox.x + bbox.width, bbox.y),
        Point(bbox.x + bbox.width, bbox.y + bbox.height),
        Point(bbox.x, bbox.y + bbox.height),
    )
    for corner in corners:
        if point_in_polygon(corner, polygon):
            return True

    # Check if any polygon vertex is inside the bbox
    for vertex in polygon:
        if (bbox.x <= vertex.x <= bbox.x + bbox.width and
                bbox.y <= vertex.y <= bbox.y + bbox.height):
            return True

    # Check if any polygon edge intersects any bbox edge
    bbox_edges = (
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    )
    for i in range(len(polygon)):
        poly_a = polygon[i]
        poly_b = polygon[(i + 1) % len(polygon)]
        for edge_a, edge_b in bbox_edges:
            if _segments_intersect(poly_a, poly_b, edge_a, edge_b):
                return True

    return False


def detection_footpoint_in_zone(detection: Detection, zone: CounterZone) -> bool:
    return point_in_polygon(detection.bbox.bottom_center, zone.polygon)


def detection_center_in_zone(detection: Detection, zone: CounterZone) -> bool:
    return point_in_or_on_polygon(detection.bbox.center, zone.polygon)
