from cat_cannon.domain.geometry import point_in_polygon
from cat_cannon.domain.models import Point


def test_point_in_polygon_returns_true_for_interior_point() -> None:
    polygon = (
        Point(0, 0),
        Point(10, 0),
        Point(10, 10),
        Point(0, 10),
    )

    assert point_in_polygon(Point(5, 5), polygon) is True


def test_point_in_polygon_returns_false_for_exterior_point() -> None:
    polygon = (
        Point(0, 0),
        Point(10, 0),
        Point(10, 10),
        Point(0, 10),
    )

    assert point_in_polygon(Point(12, 5), polygon) is False

